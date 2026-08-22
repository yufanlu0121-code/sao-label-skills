"""Extract structured records from a corpus via the Anthropic Message Batches API.

TEMPLATE: adapt build_schema() and TOPLEVEL_KEYS to your output schema. Everything
else -- homogeneity enforcement, truncation and contamination checks, spread-sampled
pilots, resumability, multi-account keys -- is domain-independent.

Narratives are read from `narratives.parquet` (built by `prepare_full.py`) and one
JSON extraction is written per filing to `raw_full/{sao_id}.json`. No intermediate
text files are created: the universe is ~8.6k filings and this directory syncs to
Dropbox.

The measurement instrument is `prompt.md`, read byte-for-byte from disk and sent as
the system prompt; the narrative is the sole user message. `prompt.md` must not be
edited mid-run -- every filing has to be scored by the same instrument, and
`assert_homogeneous` aborts a submit whose treatment differs from earlier batches.

Three subcommands, run in order:

    python3 extract_api.py submit --limit 20     # pilot: 20 filings
    python3 extract_api.py status                # poll until "ended"
    python3 extract_api.py fetch                 # write raw_full/*.json

Then re-run `submit` with no --limit for the rest. `submit` only ever queues filings
that lack a valid `raw_full/{sao_id}.json`, so an interrupted or partially failed run
is resumed by simply running the three commands again.

Credentials: one API key per line in `~/.sao_keys.txt` (override with SAO_KEY_FILE),
or a single key in ANTHROPIC_API_KEY. With several keys, `--key N` picks which account
submits a batch, so a run can be split across accounts when one is short on quota. A
batch belongs to the account that created it, so the index used at submit time is
recorded in the state file and reused automatically by `status` and `fetch` --
fetching with the wrong key returns a 404.

Requires `pip install anthropic`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import anthropic
import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

BASE_DIR = Path(__file__).resolve().parent

# Narratives are read from one parquet, not a directory of text files: the universe
# is ~8.6k filings and this directory syncs to Dropbox. Rebound by main().
NARRATIVES_PATH = BASE_DIR / "narratives.parquet"

# Extraction output lands outside Dropbox by default. 8,475 small JSON files written
# over hours would churn the sync client, and a half-synced output directory is a
# genuine corruption risk for a run this long. Override with --out-dir.
DEFAULT_OUT_ROOT = Path(
    os.environ.get("SAO_OUT_ROOT", Path.home() / "sao_extract_output")
)
OUT_DIR = DEFAULT_OUT_ROOT / "raw_full"
PROMPT_PATH = BASE_DIR / "prompt.md"
STATE_PATH = BASE_DIR / "batch_state.json"
FAILED_PATH = BASE_DIR / "failed.json"
# Searched in order. `~/.sao_keys.txt` comes first and is the intended home: this
# project lives in a Dropbox folder, and a plaintext API key inside it would be
# synced to the cloud. Nothing here writes a key file.
KEY_CANDIDATES = [
    Path(os.environ["SAO_KEY_FILE"]) if os.environ.get("SAO_KEY_FILE") else None,
    Path.home() / ".sao_keys.txt",
    BASE_DIR / "keys.txt",
]

MODEL = "claude-opus-5"
MAX_TOKENS = 32_000

# With thinking enabled, reasoning tokens are drawn from the same max_tokens budget as
# the answer, so a ceiling set for the expected answer length can be consumed by
# reasoning and truncate the response. Truncation is not random: it strikes the
# longest, most statement-dense filings, which is exactly the wrong missingness. The
# ceiling is therefore set well above the projected worst case, and a response that
# still comes back at `max_tokens` is rejected rather than written.
TRUNCATION_HEADROOM = 0.90

# A clean response is bare JSON. Reasoning markers or fenced code leaking into the
# answer mean the model wrote prose where structured output was required; such a
# response is quarantined rather than parsed, because a leaked marker inside a string
# field would still parse as valid JSON and pass silently.
# Deliberately narrow. A bare "<" cannot be a marker: SAO narratives routinely write
# "less than" as "<", e.g. "reserves < 1% of surplus", and flagging those would
# quarantine valid extractions. The structural bare-JSON check below is what catches
# prose leaking in; these markers catch reasoning or fences surviving inside it.
CONTAMINATION_MARKERS = ("<thinking", "</thinking", "```")
DEFAULT_EFFORT = "high"

# The API ceiling is 100,000 requests / 256 MB per batch. A smaller cap keeps any
# single failure from taking the whole run with it and makes progress visible.
MAX_PER_BATCH = 500

TOPLEVEL_KEYS = {
    "statements",
    "rmad_conclusion",
    "rmad_verbatim",
    "materiality_amount",
    "materiality_pct",
    "materiality_denominator",
    "reserve_stance",
    "reserve_stance_verbatim",
    "carried_vs_estimate",
    "section_truncated",
}

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout
)
logger = logging.getLogger(__name__)


def thinking_config(no_thinking: bool) -> dict[str, str]:
    """Build the thinking parameter.

    Disabling thinking roughly triples reproducibility on this task (see HANDOFF §3):
    adaptive thinking takes a different reasoning path on each call, and that is the
    dominant source of run-to-run variance in which statements get extracted.

    Args:
        no_thinking: Whether to disable thinking.

    Returns:
        The `thinking` request parameter.
    """
    return {"type": "disabled"} if no_thinking else {"type": "adaptive"}


def run_signature(
    system: str, effort: str, structured: bool, no_thinking: bool = False
) -> dict[str, Any]:
    """Describe the exact treatment applied to every filing in a batch.

    Homogeneity is the point: an estimate built from filings scored under different
    prompts, models, or thinking settings is not one measurement but several. This
    signature is stored with each batch so that claim can be checked rather than
    assumed, and `assert_homogeneous` refuses a run that would mix treatments.

    Args:
        system: The full system prompt text.
        effort: The effort level.
        structured: Whether the JSON Schema constraint is applied.
        no_thinking: Whether thinking is disabled.

    Returns:
        A new dict of the treatment parameters, including a hash of the prompt.
    """
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "thinking": thinking_config(no_thinking)["type"],
        "effort": effort,
        "structured": structured,
        "prompt_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest()[:16],
    }


def observed_models(state: dict[str, Any]) -> dict[str, int]:
    """Tally the models that actually served the requests fetched so far.

    Args:
        state: The batch-tracking state.

    Returns:
        A new dict mapping model id to the number of filings it served.
    """
    tally: dict[str, int] = {}
    for batch in state["batches"]:
        for model, count in (batch.get("response_models") or {}).items():
            tally[model] = tally.get(model, 0) + count
    return tally


def assert_homogeneous(state: dict[str, Any], signature: dict[str, Any]) -> None:
    """Refuse to submit a batch whose treatment differs from earlier batches.

    Checks both what was requested and what was actually served. The requested model
    is not a guarantee: a served response carries its own `model` field, and a run
    whose filings were answered by two different models is two measurements, not one.

    Args:
        state: The batch-tracking state.
        signature: The treatment about to be applied.

    Raises:
        SystemExit: If an earlier batch used different parameters, or if the
            responses already fetched were served by a different model.
    """
    for batch in state["batches"]:
        earlier = batch.get("signature")
        if earlier and earlier != signature:
            differing = {
                k: (earlier.get(k), signature.get(k))
                for k in set(earlier) | set(signature)
                if earlier.get(k) != signature.get(k)
            }
            raise SystemExit(
                f"Batch {batch['id']} was submitted with different parameters: "
                f"{differing}. Extracting one sample under two treatments makes the "
                "results incomparable. Either match the earlier settings, or start a "
                "clean run with a different --out-dir."
            )

    served = observed_models(state)
    unexpected = {m: n for m, n in served.items() if m != signature["model"]}
    if unexpected:
        raise SystemExit(
            f"Responses already fetched for this run were served by {unexpected}, "
            f"not {signature['model']}. Those filings were measured by a different "
            "model and cannot be pooled with new ones. Investigate before continuing; "
            "if the served model is acceptable, restart under a fresh --out-dir."
        )
    if len(served) > 1:
        raise SystemExit(
            f"This run has already been served by more than one model: {served}. "
            "Do not extend it; restart under a fresh --out-dir."
        )


def load_keys() -> list[str]:
    """Read the available API keys, preferring the key file over the environment.

    The key file holds one key per line; blank lines and `#` comments are ignored.
    Keys are never logged -- only their index and a masked tail.

    Returns:
        A new list of API key strings, in file order.

    Raises:
        SystemExit: If no key can be found.
    """
    for candidate in KEY_CANDIDATES:
        if candidate is None or not candidate.exists():
            continue
        found = re.findall(
            r"sk-ant-[A-Za-z0-9_-]{20,}", candidate.read_text(encoding="utf-8")
        )
        if found:
            logger.info("Using %d key(s) from %s", len(found), candidate)
            return found

    env_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if env_key:
        return [env_key]

    raise SystemExit(
        "No API key found. Put one key per line in ~/.sao_keys.txt "
        "(or set SAO_KEY_FILE / ANTHROPIC_API_KEY)."
    )


def client_for(index: int) -> anthropic.Anthropic:
    """Build a client bound to one of the configured accounts.

    Args:
        index: Zero-based position of the key in the key file.

    Returns:
        An `Anthropic` client using that key.

    Raises:
        SystemExit: If `index` is out of range.
    """
    keys = load_keys()
    if not 0 <= index < len(keys):
        raise SystemExit(
            f"--key {index} is out of range: {len(keys)} key(s) configured."
        )
    return anthropic.Anthropic(api_key=keys[index])


def nullable_enum(*values: str) -> dict[str, Any]:
    """Build a schema for an enum field that may also be null.

    The validator rejects a two-member `type` list combined with `enum`
    ("Enum value 'adverse' does not match declared type '['string', 'null']'"),
    so nullability is expressed as an `anyOf` branch instead. The prompt requires
    a real null -- not an empty string or a sentinel -- for anything unstated.

    Args:
        *values: The permitted string values.

    Returns:
        A JSON Schema fragment accepting any of `values`, or null.
    """
    return {
        "anyOf": [
            {"type": "string", "enum": list(values)},
            {"type": "null"},
        ]
    }


def build_schema() -> dict[str, Any]:
    """Build the JSON Schema that constrains the model's output.

    Mirrors `prompt.md` exactly.

    Returns:
        A JSON Schema object suitable for `output_config.format`.
    """
    statement = {
        "type": "object",
        "properties": {
            "verbatim": {"type": "string"},
            "topic": {"type": "string"},
            "orientation": {
                "type": "string",
                "enum": ["retrospective", "prospective", "both"],
            },
            "object": {
                "type": "string",
                "enum": [
                    "exposure",
                    "estimate_method",
                    "data_quality",
                    "external_party",
                ],
            },
            "moment": {"type": "string", "enum": ["level", "uncertainty", "both"]},
            "direction": nullable_enum("adverse", "favorable", "unstated"),
            "persistence": {
                "type": "string",
                "enum": ["one_off", "ongoing", "unstated"],
            },
            "quantified": {"type": "boolean"},
            "quantified_text": {"type": ["string", "null"]},
            "hedge": {
                "type": "string",
                "enum": ["unconditional", "conditional", "negated"],
            },
        },
        "required": [
            "verbatim",
            "topic",
            "orientation",
            "object",
            "moment",
            "direction",
            "persistence",
            "quantified",
            "quantified_text",
            "hedge",
        ],
        "additionalProperties": False,
    }

    return {
        "type": "object",
        "properties": {
            "statements": {"type": "array", "items": statement},
            "rmad_conclusion": {
                "type": "string",
                "enum": ["yes", "no", "conditional", "not_stated"],
            },
            "rmad_verbatim": {"type": ["string", "null"]},
            "materiality_amount": {"type": ["number", "null"]},
            "materiality_pct": {"type": ["number", "null"]},
            "materiality_denominator": nullable_enum(
                "surplus", "reserves", "net_income", "other"
            ),
            "reserve_stance": {
                "type": "string",
                "enum": [
                    "comfortable",
                    "qualified_comfort",
                    "concerned",
                    "not_stated",
                ],
            },
            "reserve_stance_verbatim": {"type": ["string", "null"]},
            "carried_vs_estimate": nullable_enum(
                "below_range",
                "low_end",
                "below_central",
                "central",
                "above_central",
                "high_end",
                "above_range",
                "within_range_unspecified",
            ),
            "section_truncated": {"type": "boolean"},
        },
        "required": sorted(TOPLEVEL_KEYS),
        "additionalProperties": False,
    }


def is_done(stem: str) -> bool:
    """Report whether a filing already has a parseable extraction on disk.

    Existence alone is not proof of completion: a run cut off mid-write leaves a
    truncated file that would otherwise be treated as done forever.

    Args:
        stem: The filing key, `{naic_code}_{filing_year}`.

    Returns:
        True if `raw_api/{stem}.json` exists and parses as JSON.
    """
    path = OUT_DIR / f"{stem}.json"
    if not path.exists():
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    return True


def inspect_output(raw_text: str) -> str | None:
    """Check a response body for anything other than bare JSON.

    Args:
        raw_text: The model's text block, exactly as returned.

    Returns:
        A short reason string if the output is contaminated, else None.
    """
    stripped = raw_text.strip()
    if not stripped.startswith("{"):
        return "preamble_before_json"
    if not stripped.endswith("}"):
        return "trailing_after_json"

    lowered = stripped.lower()
    for marker in CONTAMINATION_MARKERS:
        if marker in lowered:
            return f"marker:{marker}"
    return None


def load_narratives() -> pd.DataFrame:
    """Read the extraction universe.

    Returns:
        A new DataFrame indexed by `sao_id`, sorted, with a `text` column.

    Raises:
        SystemExit: If the narratives parquet has not been built yet.
    """
    if not NARRATIVES_PATH.exists():
        raise SystemExit(
            f"{NARRATIVES_PATH.name} not found. Build it first:\n"
            '  python3 prepare_full.py "/path/to/sao_texts_full_v3.parquet"'
        )
    frame = pd.read_parquet(NARRATIVES_PATH)
    frame = frame.rename(columns={"section_relevant_comments": "text"})
    return frame.set_index("sao_id").sort_index()


def spread_selection(narratives: pd.DataFrame, todo: list[str], n: int) -> list[str]:
    """Pick n pending filings spanning the length distribution.

    `--limit` alone takes the first n by `sao_id`, which orders by NAIC code and so
    draws an arbitrary, length-unrepresentative slice. A pilot exists to surface
    length-dependent failures -- truncation above all -- so it has to reach both
    ends of the distribution. The longest pending filing is always included, since
    that is where the token ceiling is tested.

    Args:
        narratives: The corpus, indexed by `sao_id`.
        todo: Pending `sao_id` values.
        n: How many to select.

    Returns:
        A new list of `sao_id` values, sorted by length.
    """
    if n >= len(todo):
        return todo
    ordered = narratives.loc[todo].sort_values("n_words").index.tolist()
    step = len(ordered) / n
    picked = {ordered[int(i * step)] for i in range(n)}
    picked.add(ordered[-1])  # the longest: the truncation stress case
    return narratives.loc[list(picked)].sort_values("n_words").index.tolist()


def pending(narratives: pd.DataFrame) -> list[str]:
    """List the filings with no valid extraction output yet.

    Args:
        narratives: The extraction universe, indexed by `sao_id`.

    Returns:
        A new list of `sao_id` values, sorted.
    """
    return [key for key in narratives.index if not is_done(key)]


def load_state() -> dict[str, Any]:
    """Read the batch-tracking state file.

    Returns:
        A new dict with a `batches` list; empty if no state file exists yet.
    """
    if not STATE_PATH.exists():
        return {"batches": []}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state: dict[str, Any]) -> None:
    """Write the batch-tracking state file.

    Args:
        state: The state dict to persist.
    """
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def build_request(
    key: str,
    text: str,
    system: str,
    effort: str,
    structured: bool,
    no_thinking: bool = False,
) -> Request:
    """Build one batch request for a single narrative.

    Args:
        key: The filing's `sao_id`, used as the batch `custom_id`.
        text: The Relevant Comments narrative.
        system: The contents of `prompt.md`.
        effort: One of low | medium | high | xhigh | max.
        structured: Whether to constrain output with a JSON Schema.
        no_thinking: Whether to disable thinking.

    Returns:
        A `Request` whose `custom_id` is the filing key.
    """
    params: dict[str, Any] = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": system,
        "thinking": thinking_config(no_thinking),
        "output_config": {"effort": effort},
        "messages": [{"role": "user", "content": text}],
    }
    if structured:
        params["output_config"]["format"] = {
            "type": "json_schema",
            "schema": build_schema(),
        }

    return Request(
        custom_id=key,
        params=MessageCreateParamsNonStreaming(**params),
    )


def cmd_submit(args: argparse.Namespace) -> None:
    """Queue every pending narrative into one or more batches.

    Args:
        args: Parsed command-line arguments.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    system = PROMPT_PATH.read_text(encoding="utf-8")
    narratives = load_narratives()

    todo = pending(narratives)
    if args.limit:
        todo = (
            spread_selection(narratives, todo, args.limit)
            if args.spread
            else todo[: args.limit]
        )
    if not todo:
        logger.info("Nothing pending. All narratives have a valid extraction.")
        return

    state = load_state()
    # A filing already sitting in an unfetched batch must not be queued twice, or
    # it would be paid for twice and the second result would overwrite the first.
    in_flight = {
        cid
        for batch in state["batches"]
        if not batch.get("fetched")
        for cid in batch["custom_ids"]
    }
    todo = [key for key in todo if key not in in_flight]
    if not todo:
        logger.info(
            "All pending narratives are already in an unfetched batch. "
            "Run `status`, then `fetch`."
        )
        return

    signature = run_signature(
        system, args.effort, not args.no_structured, args.no_thinking
    )
    assert_homogeneous(state, signature)
    logger.info("Treatment: %s", signature)

    client = client_for(args.key)
    for start in range(0, len(todo), MAX_PER_BATCH):
        chunk = todo[start : start + MAX_PER_BATCH]
        requests = [
            build_request(
                key,
                narratives.at[key, "text"],
                system,
                args.effort,
                not args.no_structured,
                args.no_thinking,
            )
            for key in chunk
        ]
        batch = client.messages.batches.create(requests=requests)
        state["batches"].append(
            {
                "id": batch.id,
                "n": len(chunk),
                "key_index": args.key,
                "signature": signature,
                "custom_ids": list(chunk),
                "fetched": False,
            }
        )
        save_state(state)
        logger.info(
            "Submitted batch %s with %d requests on key %d.",
            batch.id,
            len(chunk),
            args.key,
        )

    logger.info(
        "Done. Poll with `python3 extract_api.py status`, then `fetch` when ended."
    )


def cmd_status(args: argparse.Namespace) -> None:
    """Report the processing status of every unfetched batch.

    Args:
        args: Parsed command-line arguments.
    """
    state = load_state()
    open_batches = [b for b in state["batches"] if not b.get("fetched")]
    if not open_batches:
        logger.info("No unfetched batches.")
    for entry in open_batches:
        client = client_for(entry.get("key_index", 0))
        batch = client.messages.batches.retrieve(entry["id"])
        counts = batch.request_counts
        logger.info(
            "%s | key %d | %s | processing=%d succeeded=%d errored=%d expired=%d",
            entry["id"],
            entry.get("key_index", 0),
            batch.processing_status,
            counts.processing,
            counts.succeeded,
            counts.errored,
            counts.expired,
        )

    narratives = load_narratives()
    remaining = len(pending(narratives))
    logger.info(
        "On disk: %s/%s extracted.",
        f"{len(narratives) - remaining:,}",
        f"{len(narratives):,}",
    )
    served = observed_models(state)
    if served:
        logger.info("Served by: %s", served)


def cmd_fetch(args: argparse.Namespace) -> None:
    """Download results from every ended batch and write them to `raw_api/`.

    Results arrive in arbitrary order and are keyed by `custom_id`, never by
    position. A response that hit the token ceiling is rejected rather than
    written, so the filing stays pending and is re-queued by the next `submit`.

    Args:
        args: Parsed command-line arguments.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    failures: list[dict[str, str]] = []

    for entry in state["batches"]:
        if entry.get("fetched"):
            continue

        # A batch belongs to the account that created it; retrieving it with a
        # different key returns 404, so reuse the key recorded at submit time.
        client = client_for(entry.get("key_index", 0))
        batch = client.messages.batches.retrieve(entry["id"])
        if batch.processing_status != "ended":
            logger.info(
                "Batch %s is %s — skipping.", entry["id"], batch.processing_status
            )
            continue

        written = 0
        truncated = 0
        stop_reasons: dict[str, int] = {}
        near_limit: list[tuple[str, int]] = []
        response_models: dict[str, int] = {}
        usage_in = 0
        usage_out = 0
        for result in client.messages.batches.results(entry["id"]):
            stem = result.custom_id
            kind = result.result.type

            if kind != "succeeded":
                reason = kind
                if kind == "errored":
                    reason = f"errored:{result.result.error.type}"
                failures.append({"stem": stem, "reason": reason})
                continue

            message = result.result.message
            # The served model is recorded per filing, not assumed from the request:
            # assert_homogeneous uses it to catch a run split across two models.
            response_models[message.model] = response_models.get(message.model, 0) + 1
            stop_reasons[message.stop_reason] = (
                stop_reasons.get(message.stop_reason, 0) + 1
            )
            usage_in += message.usage.input_tokens
            usage_out += message.usage.output_tokens

            if message.stop_reason == "max_tokens":
                failures.append({"stem": stem, "reason": "max_tokens"})
                truncated += 1
                continue
            if message.usage.output_tokens >= MAX_TOKENS * TRUNCATION_HEADROOM:
                near_limit.append((stem, message.usage.output_tokens))
            if message.stop_reason == "refusal":
                failures.append({"stem": stem, "reason": "refusal"})
                continue

            text = next((b.text for b in message.content if b.type == "text"), "")
            polluted = inspect_output(text)
            if polluted:
                failures.append({"stem": stem, "reason": f"contaminated:{polluted}"})
                continue

            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                failures.append({"stem": stem, "reason": "json_decode"})
                continue

            missing = TOPLEVEL_KEYS - set(data)
            if missing:
                failures.append(
                    {"stem": stem, "reason": f"missing_keys:{sorted(missing)}"}
                )
                continue
            if not isinstance(data.get("statements"), list):
                failures.append({"stem": stem, "reason": "statements_not_list"})
                continue

            (OUT_DIR / f"{stem}.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            written += 1

        entry["fetched"] = True
        entry["response_models"] = response_models
        entry["usage"] = {"input_tokens": usage_in, "output_tokens": usage_out}
        save_state(state)
        entry["stop_reasons"] = stop_reasons
        entry["near_limit"] = len(near_limit)
        logger.info(
            "Batch %s: wrote %d files | served by %s | %s in / %s out tokens",
            entry["id"],
            written,
            response_models or "n/a",
            f"{usage_in:,}",
            f"{usage_out:,}",
        )
        # Reported every fetch, not only on failure: a rising near-limit count is the
        # early warning that the corpus tail is longer than the pilot suggested.
        logger.info(
            "Batch %s: stop_reason %s | %d response(s) at >=%.0f%% of the %s ceiling",
            entry["id"],
            stop_reasons or "n/a",
            len(near_limit),
            TRUNCATION_HEADROOM * 100,
            f"{MAX_TOKENS:,}",
        )
        if truncated:
            logger.error(
                "%d response(s) in batch %s hit max_tokens (%d) and were rejected. "
                "Truncation targets the longest filings, so this is not random "
                "missingness. Raise MAX_TOKENS and re-run `submit` before continuing.",
                truncated,
                entry["id"],
                MAX_TOKENS,
            )
        if near_limit:
            worst = max(n for _, n in near_limit)
            logger.warning(
                "%d response(s) used >=%.0f%% of the %d token ceiling (worst %d). "
                "Headroom is thin; consider raising MAX_TOKENS.",
                len(near_limit),
                TRUNCATION_HEADROOM * 100,
                MAX_TOKENS,
                worst,
            )
        if len(response_models) > 1:
            logger.warning(
                "Batch %s was served by more than one model: %s. These filings are "
                "not a single measurement.",
                entry["id"],
                response_models,
            )

    if failures:
        FAILED_PATH.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        logger.warning(
            "%d requests did not produce a usable extraction — see %s. "
            "They remain pending; re-run `submit` to retry them.",
            len(failures),
            FAILED_PATH.name,
        )

    narratives = load_narratives()
    remaining = len(pending(narratives))
    logger.info(
        "On disk: %s/%s extracted.",
        f"{len(narratives) - remaining:,}",
        f"{len(narratives):,}",
    )


def main() -> None:
    """Parse arguments and dispatch to the requested subcommand."""
    global NARRATIVES_PATH, OUT_DIR, STATE_PATH, FAILED_PATH, MAX_TOKENS

    parser = argparse.ArgumentParser(description=__doc__)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--narratives",
        default="narratives.parquet",
        help="parquet holding the extraction universe (default narratives.parquet)",
    )
    common.add_argument(
        "--out-dir",
        default="raw_full",
        help="output directory name, created under SAO_OUT_ROOT "
        f"(default {DEFAULT_OUT_ROOT})",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    submit = sub.add_parser(
        "submit", parents=[common], help="queue pending narratives into batches"
    )
    submit.add_argument(
        "--limit", type=int, default=0, help="queue at most N filings (0 = all)"
    )
    submit.add_argument(
        "--spread",
        action="store_true",
        help="with --limit, pick filings spanning the length distribution and "
        "always include the longest, instead of the first N by sao_id",
    )
    submit.add_argument(
        "--effort",
        default=DEFAULT_EFFORT,
        choices=["low", "medium", "high", "xhigh", "max"],
        help=f"thinking effort (default {DEFAULT_EFFORT})",
    )
    submit.add_argument(
        "--no-structured",
        action="store_true",
        help="disable the JSON Schema constraint and parse free text instead",
    )
    submit.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help=f"response token ceiling, shared by reasoning and answer "
        f"(default {MAX_TOKENS})",
    )
    submit.add_argument(
        "--no-thinking",
        action="store_true",
        help="disable thinking; ~3x more reproducible and cheaper (see HANDOFF §3)",
    )
    submit.add_argument(
        "--key",
        type=int,
        default=0,
        help="which account to bill, as a 0-based index into keys.txt",
    )
    submit.set_defaults(func=cmd_submit)

    status = sub.add_parser(
        "status", parents=[common], help="poll unfetched batches"
    )
    status.set_defaults(func=cmd_status)

    fetch = sub.add_parser(
        "fetch", parents=[common], help="download ended batches into the output dir"
    )
    fetch.set_defaults(func=cmd_fetch)

    args = parser.parse_args()

    # The state file is per input set: mixing the pilot and the full universe in one
    # batch_state.json would let a pilot batch mark full-run filings as in flight.
    if getattr(args, "max_tokens", None):
        MAX_TOKENS = args.max_tokens
    NARRATIVES_PATH = BASE_DIR / args.narratives
    OUT_DIR = DEFAULT_OUT_ROOT / args.out_dir
    # State lives beside the output it describes, so moving or deleting a run takes
    # its bookkeeping with it and cannot leave a stale state file behind.
    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH = DEFAULT_OUT_ROOT / f"batch_state_{args.out_dir}.json"
    FAILED_PATH = DEFAULT_OUT_ROOT / f"failed_{args.out_dir}.json"

    args.func(args)


if __name__ == "__main__":
    main()
