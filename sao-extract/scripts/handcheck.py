"""Check extracted quotations against their source text, and diagnose failures.

TEMPLATE: adapt `DEFAULT_IDS`, `STATEMENT_COLUMNS`, and `DOCUMENT_FIELDS` to your
schema. The substring machinery -- furniture stripping, normalization ladder, splice
diagnosis -- is domain-independent.

When a schema asks the model to quote the source, "did it actually quote?" is a
checkable property, and checking it naively produces false alarms. This script
renders each document beside its extraction and locates every quoted span in the
source at four strictnesses:

    exact      the string as returned
    ws         whitespace collapsed on both sides (PDF text wraps mid-sentence)
    punct      ws, plus curly quotes/dashes folded and wrapped hyphens rejoined
    +hdr       punct, with running page headers and footers removed -- AUTHORITATIVE

Only `+hdr` is informative. The three weaker levels are reported to show how much of
an apparent failure is PDF noise: on real filings `exact` matched 0 of 56 spans that
were all, in fact, verbatim quotations.

A failure at `+hdr` is diagnosed rather than merely counted. A spliced quote has a
signature -- its longest matching prefix and longest matching suffix are each real
runs of source text and together account for the whole span -- which distinguishes
"joined two non-adjacent passages" from "altered one word mid-quote" (prefix and
suffix overlap) and from paraphrase (no matching tail).

See `references/pitfalls.md` #10 for why the furniture strip is not optional.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
NARRATIVES_PATH = BASE_DIR / "narratives.parquet"
DEFAULT_OUT_ROOT = Path(
    os.environ.get("SAO_OUT_ROOT", Path.home() / "sao_extract_output")
)
DEFAULT_IDS = ["11598_2024_v1", "42722_2020_v1", "27014_2019_v1"]
# The corpus builder normalizes the narrative into `text`; the SAO project's own
# parquet keeps the source column name. Detected rather than configured, so the bare
# command works against either.
TEXT_COLUMNS = ("text", "section_relevant_comments")

STATEMENT_COLUMNS = [
    "verbatim",
    "topic",
    "orientation",
    "object",
    "moment",
    "direction",
    "hedge",
]
DOCUMENT_FIELDS = [
    "rmad_conclusion",
    "rmad_verbatim",
    "materiality_amount",
    "materiality_pct",
    "materiality_denominator",
    "reserve_stance",
    "reserve_stance_verbatim",
    "carried_vs_estimate",
    "section_truncated",
]
PUNCT_MAP = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", " ": " ",
    "­": "", "ʼ": "'", "′": "'",
}

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout
)
logger = logging.getLogger(__name__)


FURNITURE_PATTERNS = (
    re.compile(r"^Page\s+\d+(\s+of\s+\d+)?$", re.I),
    re.compile(r"^SAO\s+\d{1,2}/\d{1,2}/\d{2,4}$", re.I),
)
FURNITURE_MAX_CHARS = 80
FURNITURE_MIN_REPEATS = 3


def strip_furniture(text: str) -> str:
    """Remove running page headers and footers from a PDF-derived narrative.

    The source PDFs inject a running header or footer into the text layer, often
    mid-sentence. `prompt.md` explicitly permits the model to read straight through
    it ("Page headers and footers injected mid-sentence by the source PDF may be
    skipped over"), so a quote that spans one is a correct contiguous quotation, not
    a splice. Leaving the furniture in makes the substring check report false
    splices at exactly those seams.

    Furniture is identified from the document itself rather than from a hard-coded
    company name: a line that is short and repeats several times through the text is
    running furniture, whatever it says. Two real filings show why the name cannot be
    hard-coded -- one repeats `Applied Medico-Legal Solutions, RRG` 14 times, another
    repeats an auditor's footer, `A member firm of Ernst & Young Global Limited`.
    Long lines are never removed, so a repeated sentence of substance survives.

    Args:
        text: The raw narrative.

    Returns:
        A new string with furniture lines dropped.
    """
    lines = text.split("\n")
    stripped = [line.strip() for line in lines]
    repeats = Counter(line for line in stripped if line)
    keep = []
    for raw, line in zip(lines, stripped):
        if any(pattern.match(line) for pattern in FURNITURE_PATTERNS):
            continue
        if (
            line
            and len(line) < FURNITURE_MAX_CHARS
            and repeats[line] >= FURNITURE_MIN_REPEATS
        ):
            continue
        keep.append(raw)
    return "\n".join(keep)


def normalize_ws(text: str) -> str:
    """Collapse all whitespace runs to single spaces.

    Args:
        text: Any string.

    Returns:
        A new string with whitespace collapsed and ends stripped.
    """
    return " ".join(text.split())


def normalize_punct(text: str) -> str:
    r"""Collapse whitespace, fold typographic punctuation, repair hyphenation.

    Source narratives come from PDF text layers, which wrap mid-sentence, break
    hyphenated compounds across lines, and use curly quotes inconsistently. None of
    that is evidence the model failed to quote contiguously, so it is normalized away
    before the substring test is called a failure. `-\s+` collapses to `-` rather
    than closing the word up, because the wrapped compounds here are genuinely
    hyphenated -- `non- collectability` is the source's rendering of
    `non-collectability`, not of `noncollectability`.

    Args:
        text: Any string.

    Returns:
        A new normalized string.
    """
    folded = text
    for src, dst in PUNCT_MAP.items():
        folded = folded.replace(src, dst)
    folded = re.sub(r"-\s+", "-", folded)
    return normalize_ws(folded)


def match_levels(verbatim: str, narrative: str) -> tuple[bool, bool, bool, bool]:
    """Test whether a verbatim occurs in the narrative at four strictnesses.

    The last level -- punctuation-normalized with running headers and footers
    removed -- is the authoritative one. The three weaker levels are kept only to
    show how much of an apparent failure is PDF noise rather than a real defect.

    Args:
        verbatim: The quoted span returned by the model.
        narrative: The source narrative text.

    Returns:
        A tuple of (exact, ws-normalized, punct-normalized, furniture-stripped).
    """
    return (
        verbatim in narrative,
        normalize_ws(verbatim) in normalize_ws(narrative),
        normalize_punct(verbatim) in normalize_punct(narrative),
        normalize_punct(verbatim) in normalize_punct(strip_furniture(narrative)),
    )


def diagnose(verbatim: str, narrative: str) -> str:
    """Explain why a verbatim could not be located in the source.

    A spliced quote has a distinctive signature: its longest matching prefix and its
    longest matching suffix are each a genuine run of source text, and together they
    account for the whole span. The model took a head from one passage and a tail
    from another and joined them at a seam -- the failure mode `prompt.md` bans
    outright. An overlapping prefix and suffix instead means a word was altered
    inside an otherwise contiguous quote, which is a transcription slip, not a
    splice. No suffix match at all suggests paraphrase.

    Args:
        verbatim: The quoted span, already punctuation-normalized.
        narrative: The source narrative, already punctuation-normalized.

    Returns:
        A short human-readable diagnosis, or an em dash when the span matches.
    """
    if verbatim in narrative:
        return "—"
    words = verbatim.split()
    total = len(words)
    prefix = 0
    for k in range(1, total + 1):
        if " ".join(words[:k]) in narrative:
            prefix = k
        else:
            break
    suffix = 0
    for k in range(1, total + 1):
        if " ".join(words[-k:]) in narrative:
            suffix = k
        else:
            break
    if prefix + suffix > total:
        return f"internal edit (~word {prefix} of {total})"
    if prefix and suffix and prefix + suffix >= total - 1:
        return f"**SPLICE** at word {prefix} of {total}"
    if not suffix:
        return f"no tail match (paraphrase?), {prefix}/{total} head"
    return f"gap of {total - prefix - suffix} words (head {prefix}, tail {suffix})"


def cell(value: Any, limit: int | None = None) -> str:
    """Render a value safely inside a markdown table cell.

    Args:
        value: Any JSON-decoded value.
        limit: Optional character budget, after which the text is elided.

    Returns:
        A new single-line string with pipes escaped.
    """
    if value is None:
        return "_null_"
    text = normalize_ws(str(value)).replace("|", "\\|")
    if limit is not None and len(text) > limit:
        text = text[:limit] + "…"
    return text


def narrative_of(row: pd.Series) -> str:
    """Return the narrative text from whichever column holds it.

    Args:
        row: One row of the corpus.

    Returns:
        The narrative text.

    Raises:
        SystemExit: If no known text column is present.
    """
    for column in TEXT_COLUMNS:
        if column in row.index:
            return row[column]
    raise SystemExit(f"No text column found; expected one of {TEXT_COLUMNS}.")


def render_filing(sao_id: str, row: pd.Series, extraction: dict[str, Any]) -> str:
    """Render one filing's hand-check section.

    Args:
        sao_id: The filing key.
        row: Its row from `narratives.parquet`, carrying metadata and text.
        extraction: The decoded extraction JSON.

    Returns:
        A new markdown string for this filing.
    """
    statements = extraction.get("statements", [])
    narrative = narrative_of(row)
    out = [
        f"## {sao_id}",
        "",
        f"**Company:** {row['company_name']}  ",
        f"**NAIC:** {row['naic_code']}  ",
        f"**Filing year:** {row['filing_year']}  ",
        f"**Narrative words (parquet `n_words`):** {row['n_words']}  ",
        f"**Statements extracted:** {len(statements)}",
        "",
        "",
        "### 1. Narrative text (verbatim from narratives.parquet)",
        "",
        "```text",
        narrative,
        "```",
        "",
        "### 2. Extracted statements",
        "",
        "| # | " + " | ".join(STATEMENT_COLUMNS) + " |",
        "|---" * (len(STATEMENT_COLUMNS) + 1) + "|",
    ]
    for i, statement in enumerate(statements, 1):
        cells = [cell(statement.get(col)) for col in STATEMENT_COLUMNS]
        out.append(f"| {i} | " + " | ".join(cells) + " |")
    if not statements:
        out.append("| — | _no risk statements returned_ |" + " |" * 6)

    out += ["", "### 3. Document-level fields", "", "| field | value |", "|---|---|"]
    for field in DOCUMENT_FIELDS:
        out.append(f"| `{field}` | {cell(extraction.get(field))} |")

    out += [
        "",
        "### 4. Verbatim substring check",
        "",
        "| # | exact | ws | punct | +hdr | diagnosis | verbatim (first 90 chars) |",
        "|---|---|---|---|---|---|---|",
    ]
    tally = [0, 0, 0, 0]
    splices = 0
    normalized_source = normalize_punct(strip_furniture(narrative))
    for i, statement in enumerate(statements, 1):
        verbatim = statement.get("verbatim", "")
        exact, ws, punct, hdr = match_levels(verbatim, narrative)
        tally = [tally[0] + exact, tally[1] + ws, tally[2] + punct, tally[3] + hdr]
        flags = ["YES" if hit else "NO" for hit in (exact, ws, punct, hdr)]
        verdict = diagnose(normalize_punct(verbatim), normalized_source)
        splices += verdict.startswith("**SPLICE")
        out.append(
            f"| {i} | {flags[0]} | {flags[1]} | {flags[2]} | {flags[3]} | {verdict} "
            f"| {cell(verbatim, 90)} |"
        )
    total = len(statements)
    out += [
        "",
        f"**{tally[0]} of {total} matched exactly.**  ",
        f"**{tally[1]} of {total} matched after whitespace normalization.**  ",
        f"**{tally[2]} of {total} matched after punctuation normalization.**  ",
        f"**{tally[3]} of {total} matched once running headers/footers are removed "
        "— this is the level that counts.**  ",
        f"**{splices} of {total} are spliced** — non-adjacent passages joined, which",
        "`prompt.md` forbids.",
        "",
        "---",
        "",
    ]
    return "\n".join(out), total, tally[3], splices


def build(ids: list[str], out_dir: Path, prompt_hash: str) -> str:
    """Assemble the whole hand-check document.

    Args:
        ids: The `sao_id` values to render.
        out_dir: Directory of extraction JSON files.
        prompt_hash: Short hash of the instrument that produced them.

    Returns:
        A new markdown string.
    """
    narratives = pd.read_parquet(NARRATIVES_PATH).set_index("sao_id")
    header = [
        f"# Hand-check: {len(ids)} pilot extractions",
        "",
        f"Instrument: `prompt.md` @ `{prompt_hash}`  ",
        "Source narratives: `narratives.parquet`  ",
        f"Source extractions: `{out_dir}`",
        "",
        "Section 4 reports the substring check at three strictnesses; only a failure",
        "at `punct` means the span was not quoted contiguously. See the module",
        "docstring in `handcheck.py`.",
        "",
        "---",
        "",
    ]
    body = []
    n_all = n_found = n_splice = 0
    for sao_id in ids:
        path = out_dir / f"{sao_id}.json"
        if not path.exists():
            logger.warning("%s missing from %s, skipped.", sao_id, out_dir)
            continue
        extraction = json.loads(path.read_text())
        section, total, found, splices = render_filing(
            sao_id, narratives.loc[sao_id], extraction
        )
        body.append(section)
        n_all += total
        n_found += found
        n_splice += splices
    pct = 100 * n_found / n_all if n_all else 0.0
    summary = [
        "## Summary",
        "",
        "| | count | of total |",
        "|---|---|---|",
        f"| Statements | {n_all} | |",
        f"| Located in source (headers/footers stripped) | {n_found} | {pct:.1f}% |",
        f"| **Spliced** (non-adjacent passages joined) | **{n_splice}** | "
        f"{100 * n_splice / n_all if n_all else 0:.1f}% |",
        "",
        "Splicing is banned by `prompt.md`. A spliced `verbatim` is not a quotation,",
        "so any downstream check that relies on locating it in the source will fail,",
        "and the statement's boundaries are the model's rather than the actuary's.",
        "",
        "---",
        "",
    ]
    return "\n".join(header) + "\n".join(summary) + "".join(body)


def instrument_hash(out_dir: Path) -> str:
    """Identify the instrument that produced the extractions in `out_dir`.

    Hashing the current `prompt.md` would mislabel an archived run: the prompt on
    disk is the one in force now, not the one that scored those filings. The batch
    state file beside the output records the signature that was actually sent, so it
    is the authority whenever it exists.

    Args:
        out_dir: Directory of extraction JSON files.

    Returns:
        The prompt hash, suffixed to flag a fallback to the current prompt.md.
    """
    state = out_dir.parent / f"batch_state_{out_dir.name}.json"
    if state.exists():
        batches = json.loads(state.read_text()).get("batches", [])
        hashes = {b.get("signature", {}).get("prompt_sha256") for b in batches}
        hashes.discard(None)
        if len(hashes) == 1:
            return hashes.pop()
        if hashes:
            return " + ".join(sorted(hashes)) + " (MIXED -- not one instrument)"
    digest = hashlib.sha256((BASE_DIR / "prompt.md").read_bytes()).hexdigest()[:16]
    return f"{digest} (from current prompt.md; no batch state found)"


def main() -> None:
    """Parse arguments and write the hand-check markdown."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="raw_full", help="extraction directory")
    parser.add_argument("--ids", nargs="*", default=DEFAULT_IDS, help="sao_id values")
    parser.add_argument("--out", default="handcheck.md", help="output markdown file")
    args = parser.parse_args()

    out_dir = DEFAULT_OUT_ROOT / args.dir
    if not out_dir.exists():
        raise SystemExit(f"{out_dir} does not exist.")

    prompt_hash = instrument_hash(out_dir)
    document = build(args.ids, out_dir, prompt_hash)
    target = DEFAULT_OUT_ROOT / args.out
    target.write_text(document.replace(str(out_dir), f"~/{out_dir.relative_to(Path.home())}"))
    logger.info("Wrote %s (%d bytes) for instrument %s.", target, len(document), prompt_hash)


if __name__ == "__main__":
    main()
