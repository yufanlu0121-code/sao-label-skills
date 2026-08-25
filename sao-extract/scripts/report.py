"""Summarise the extraction outputs: progress, field distributions, share moments.

Reproduces the progress report used throughout the interactive run. Reads whatever
extractions exist, so it is useful mid-run as well as at the end.

    python3 report.py                  # read raw_api/
    python3 report.py --dir raw        # read the interactive-run outputs instead
    python3 report.py --compare        # agreement between raw/ and raw_api/

`_test` files are diagnostic copies and are excluded everywhere.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
NARRATIVES_PATH = BASE_DIR / "narratives.parquet"

# Extractions live outside Dropbox (see HANDOFF ss2), so --dir is resolved against
# the same output root extract_api.py writes to, not against this directory. An
# absolute --dir still overrides, and a relative one is taken from the output root.
DEFAULT_OUT_ROOT = Path(
    os.environ.get("SAO_OUT_ROOT", Path.home() / "sao_extract_output")
)

STATEMENT_FIELDS = [
    "orientation",
    "object",
    "moment",
    "direction",
    "persistence",
    "quantified",
    "hedge",
]
DOCUMENT_FIELDS = [
    "rmad_conclusion",
    "reserve_stance",
    "carried_vs_estimate",
    "materiality_denominator",
    "section_truncated",
]
PERCENTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout
)
logger = logging.getLogger(__name__)


def load_extractions(directory: Path) -> dict[str, dict[str, Any]]:
    """Read every parseable extraction in a directory, keyed by filing stem.

    A file that fails to parse is skipped rather than raising: a run cut off
    mid-write leaves truncated JSON behind, and the report should still run.

    Args:
        directory: The directory holding `{stem}.json` extractions.

    Returns:
        A new dict mapping filing stem to the parsed record.
    """
    records: dict[str, dict[str, Any]] = {}
    bad = 0
    for path in sorted(directory.glob("*.json")):
        if path.stem.endswith("_test"):
            continue
        try:
            records[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            bad += 1
    if bad:
        logger.warning("%d file(s) in %s/ did not parse.", bad, directory.name)
    return records


def filing_frame(records: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Build the filing-level table of counts and orientation/moment shares.

    Shares are NaN, not zero, for a filing with no risk statements: the actuary
    wrote nothing to take a share of, and averaging a zero in would understate
    every share. There are a handful of such filings, so the choice matters.

    Args:
        records: Parsed extractions keyed by filing stem.

    Returns:
        A new DataFrame indexed by filing stem.
    """
    rows = []
    for stem, record in records.items():
        statements = record.get("statements") or []
        n = len(statements)
        orientation = [s.get("orientation") for s in statements]
        moment = [s.get("moment") for s in statements]
        obj = [s.get("object") for s in statements]
        persistence = [s.get("persistence") for s in statements]
        hedge = [s.get("hedge") for s in statements]
        quantified = [bool(s.get("quantified")) for s in statements]

        def share(values: list[Any], *wanted: Any) -> float:
            """Fraction of statements whose value is in `wanted`; NaN if none."""
            if n == 0:
                return float("nan")
            return sum(v in wanted for v in values) / n

        rows.append(
            {
                "stem": stem,
                "n_statements": n,
                "prospective_share": share(orientation, "prospective", "both"),
                "retrospective_share": share(orientation, "retrospective", "both"),
                "uncertainty_share": share(moment, "uncertainty", "both"),
                "level_share": share(moment, "level", "both"),
                "method_shock": share(obj, "estimate_method"),
                "ongoing_share": share(persistence, "ongoing"),
                "unconditional_share": share(hedge, "unconditional"),
                "quantified_share": share(quantified, True),
                "rmad_conclusion": record.get("rmad_conclusion"),
                "reserve_stance": record.get("reserve_stance"),
                "section_truncated": record.get("section_truncated"),
            }
        )
    return pd.DataFrame(rows).set_index("stem").sort_index()


def report_progress(records: dict[str, dict[str, Any]], directory: Path) -> None:
    """Log how many of the sampled filings have been extracted.

    Args:
        records: Parsed extractions keyed by filing stem.
        directory: The directory they were read from.
    """
    total = (
        len(pd.read_parquet(NARRATIVES_PATH, columns=["sao_id"]))
        if NARRATIVES_PATH.exists()
        else len(records)
    )
    done = len(records)
    pct = done / total * 100 if total else 0.0
    logger.info("=" * 62)
    logger.info("PROGRESS (%s/)", directory.name)
    logger.info("=" * 62)
    logger.info("Extracted: %s/%s (%.0f%%)", f"{done:,}", f"{total:,}", pct)


def report_counts(frame: pd.DataFrame) -> None:
    """Log statement-count summaries.

    Args:
        frame: The filing-level table.
    """
    counts = frame["n_statements"]
    logger.info("Risk statements: %s total", f"{int(counts.sum()):,}")
    logger.info(
        "Statements per filing: mean %.1f | range %d-%d | %d filing(s) with zero",
        counts.mean(),
        int(counts.min()),
        int(counts.max()),
        int((counts == 0).sum()),
    )


def report_distributions(records: dict[str, dict[str, Any]]) -> None:
    """Log the value distribution of every categorical field.

    Args:
        records: Parsed extractions keyed by filing stem.
    """
    logger.info("")
    logger.info("--- STATEMENT-LEVEL FIELDS ---")
    for field in STATEMENT_FIELDS:
        values = [s.get(field) for r in records.values() for s in r.get("statements") or []]
        summary = " / ".join(f"{k} {v}" for k, v in Counter(values).most_common())
        logger.info("%-14s %s", field, summary)

    logger.info("")
    logger.info("--- DOCUMENT-LEVEL FIELDS ---")
    for field in DOCUMENT_FIELDS:
        values = [r.get(field) for r in records.values()]
        summary = " / ".join(f"{k} {v}" for k, v in Counter(values).most_common())
        logger.info("%-24s %s", field, summary)


def report_conformance(records: dict[str, dict[str, Any]]) -> None:
    """Check the instrument's conditional rules, which the JSON Schema cannot express.

    A JSON Schema constrains each field independently. `prompt.md` also carries rules
    that relate one field to another -- `direction` is populated only when `moment` is
    `level` or `both`, and is null otherwise -- and a violation of those is structurally
    invisible to the schema, parses cleanly, and passes every other check in the
    pipeline.

    This check inherited the one rule worth keeping from the retired `validate.py`. Its
    other check, a 20-80 word band on `verbatim`, is deliberately not reproduced:
    `prompt.md` states that band as an aim and explicitly overrides it ("if a single
    sentence exceeds 80 words, quote it whole"), so testing it as a rule flags roughly
    an eighth of the corpus as violations that are not violations.

    Args:
        records: Extraction records keyed by filing id.
    """
    violations: list[tuple[str, int, str]] = []
    for key, record in records.items():
        for index, statement in enumerate(record.get("statements", [])):
            moment = statement.get("moment")
            direction = statement.get("direction")
            if moment == "uncertainty" and direction is not None:
                violations.append((key, index, f"moment=uncertainty, direction={direction}"))
            elif moment in {"level", "both"} and direction is None:
                violations.append((key, index, f"moment={moment}, direction=null"))

    logger.info("")
    logger.info("--- INSTRUMENT CONFORMANCE (cross-field rules) ---")
    if not violations:
        logger.info("direction/moment rule: no violations")
        return
    logger.warning(
        "direction/moment rule: %d violation(s) — the schema cannot catch these, "
        "and they are extraction defects, not parse errors.",
        len(violations),
    )
    for key, index, detail in violations[:10]:
        logger.warning("  %s#%d  %s", key, index, detail)
    if len(violations) > 10:
        logger.warning("  ... and %d more", len(violations) - 10)


def report_shares(frame: pd.DataFrame) -> None:
    """Log the filing-level moments used to judge regression viability.

    Args:
        frame: The filing-level table.
    """
    share_cols = [
        "prospective_share",
        "uncertainty_share",
        "retrospective_share",
        "method_shock",
        "ongoing_share",
        "unconditional_share",
        "quantified_share",
    ]
    stats = frame[share_cols].describe(percentiles=PERCENTILES).T
    stats = stats[["mean", "std", "10%", "25%", "50%", "75%", "90%"]].round(3)

    logger.info("")
    logger.info("--- FILING-LEVEL SHARES (regression viability) ---")
    logger.info("\n%s", stats.to_string())
    logger.info(
        "Filings with >=1 statement: %d | with zero (shares are NaN): %d",
        int((frame["n_statements"] > 0).sum()),
        int((frame["n_statements"] == 0).sum()),
    )


def report_coverage(records: dict[str, dict[str, Any]]) -> None:
    """Log extraction coverage by filing year.

    The full run is a population, not a stratified sample, so there are no strata to
    report; year is the dimension that matters for panel balance.

    Args:
        records: Parsed extractions keyed by sao_id.
    """
    if not NARRATIVES_PATH.exists():
        return
    index = pd.read_parquet(NARRATIVES_PATH, columns=["sao_id", "filing_year"])
    index = index.assign(done=index["sao_id"].isin(records))

    coverage = index.groupby("filing_year").agg(
        done=("done", "sum"), total=("done", "count")
    )
    coverage["pct"] = (coverage["done"] / coverage["total"] * 100).round(0).astype(int)
    logger.info("")
    logger.info("--- COVERAGE BY FILING YEAR ---")
    logger.info("\n%s", coverage.to_string())


def report_topics(records: dict[str, dict[str, Any]], top: int) -> None:
    """Log the most frequent free-text topic labels.

    Args:
        records: Parsed extractions keyed by filing stem.
        top: How many labels to show.
    """
    topics = [
        (s.get("topic") or "").strip()
        for r in records.values()
        for s in r.get("statements") or []
    ]
    counts = Counter(topics)
    logger.info("")
    logger.info("--- TOPICS: %d unique, top %d ---", len(counts), top)
    for label, n in counts.most_common(top):
        logger.info("%5d  %s", n, label)


def report_comparison(left: Path, right: Path) -> None:
    """Log agreement between two independent extraction passes.

    Only filings present in both directories are compared. Agreement here is a
    reliability diagnostic, not a correctness check: neither pass is ground truth.

    Args:
        left: Directory of the first pass.
        right: Directory of the second pass.
    """
    a = filing_frame(load_extractions(left))
    b = filing_frame(load_extractions(right))
    shared = a.index.intersection(b.index)

    logger.info("")
    logger.info("=" * 62)
    logger.info("TWO-PASS AGREEMENT: %s/ vs %s/", left.name, right.name)
    logger.info("=" * 62)
    if shared.empty:
        logger.info("No filings appear in both directories yet.")
        return
    logger.info("Filings in both: %d", len(shared))

    for field in ("rmad_conclusion", "reserve_stance", "section_truncated"):
        agree = (a.loc[shared, field] == b.loc[shared, field]).mean()
        logger.info("%-20s exact agreement: %.1f%%", field, agree * 100)

    diff = (a.loc[shared, "n_statements"] - b.loc[shared, "n_statements"]).abs()
    corr = a.loc[shared, "n_statements"].corr(b.loc[shared, "n_statements"])
    logger.info(
        "n_statements         mean |diff| %.2f | exact %.1f%% | corr %.3f",
        diff.mean(),
        (diff == 0).mean() * 100,
        corr,
    )


def main() -> None:
    """Parse arguments and emit the requested report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir", default="raw_full", help="directory of extractions (default raw_full)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="also compare raw/ against raw_api/",
    )
    parser.add_argument("--top", type=int, default=20, help="topics to list")
    args = parser.parse_args()

    directory = DEFAULT_OUT_ROOT / args.dir
    if not directory.exists():
        raise SystemExit(f"{directory} does not exist.")

    records = load_extractions(directory)
    if not records:
        raise SystemExit(f"No parseable extractions in {directory}.")

    frame = filing_frame(records)
    report_progress(records, directory)
    report_counts(frame)
    report_distributions(records)
    report_shares(frame)
    report_conformance(records)
    report_coverage(records)
    report_topics(records, args.top)

    if args.compare:
        report_comparison(DEFAULT_OUT_ROOT / "raw", DEFAULT_OUT_ROOT / "raw_api")


if __name__ == "__main__":
    main()
