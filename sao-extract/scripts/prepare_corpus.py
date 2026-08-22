"""Build the extraction corpus as a single parquet file.

TEMPLATE: adapt TEXT_COL, KEY_COL, COLUMNS and DEFAULT_YEARS to your source table.
The control flow -- filter, dedupe, single-file output, cost estimate -- is the part
worth keeping.

Selects every filing in the source with a non-empty Relevant Comments section,
restricts to filing years 2019-2025, drops repeated scrapes, and writes the result
to `narratives.parquet` -- one file, not one file per filing.

    python3 prepare_full.py "/path/to/sao_texts_full_v3.parquet"

`extract_api.py` reads that parquet directly, so no intermediate text files are
produced. Rows are keyed by `sao_id`: 119 eligible rows across the source have a
null `naic_code`, so `{naic_code}_{filing_year}` cannot address them, whereas
`sao_id` is unique throughout.

Deduplication keeps the lowest `dup_suffix` (v1) per `naic_code + filing_year`. The
source carries repeated scrapes of the same filing that are near-always identical;
extracting all of them would pay twice for the same text. `--no-dedup` keeps every
scrape, which is only useful for studying scrape-to-scrape variation.

The source parquet is opened read-only and never written back.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
NARRATIVES_PATH = BASE_DIR / "narratives.parquet"

TEXT_COL = "section_relevant_comments"
KEY_COL = "sao_id"
COLUMNS = [KEY_COL, "naic_code", "filing_year", "company_name", "dup_suffix", TEXT_COL]

DEFAULT_YEARS = (2019, 2025)

# Fitted on 40 filings spanning the length range, R^2 = 0.981. Used only to print a
# cost estimate; nothing downstream depends on it.
TOKENS_PER_WORD = 2.2602
TOKENS_INTERCEPT = 1304
MEAN_OUTPUT_TOKENS = 3056

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout
)
logger = logging.getLogger(__name__)


def resolve_source(raw: str | None) -> Path:
    """Resolve the source parquet path from the command line or environment.

    Args:
        raw: The path given on the command line, if any.

    Returns:
        The path to the source parquet.

    Raises:
        SystemExit: If no path was supplied.
        FileNotFoundError: If the supplied path does not exist.
    """
    given = raw or os.environ.get("SAO_PARQUET", "")
    if not given:
        raise SystemExit(
            "No source parquet given. Pass it as the first argument or set "
            'SAO_PARQUET, e.g.\n  python3 prepare_full.py "/path/to/'
            'sao_texts_full_v3.parquet"'
        )
    source = Path(given).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Source parquet not found at {str(source)!r}.")
    return source


def load_eligible(path: Path, years: tuple[int, int]) -> pd.DataFrame:
    """Read the source and keep rows with a usable narrative in range.

    Args:
        path: Path to the source parquet.
        years: Inclusive (min, max) filing-year filter.

    Returns:
        A new DataFrame of eligible rows with an `n_words` column.
    """
    frame = pd.read_parquet(path, columns=COLUMNS)
    logger.info("Read %s rows from %s", f"{len(frame):,}", path.name)

    frame = frame.copy()
    frame["filing_year"] = frame["filing_year"].astype("Int64")

    text = frame[TEXT_COL]
    eligible = frame.loc[text.notna() & (text.astype(str).str.strip() != "")].copy()
    logger.info("Non-empty narrative: %s", f"{len(eligible):,}")

    lo, hi = years
    eligible = eligible.loc[eligible["filing_year"].between(lo, hi)].copy()
    logger.info("Filing years %d-%d: %s", lo, hi, f"{len(eligible):,}")

    eligible["n_words"] = eligible[TEXT_COL].astype(str).str.split().str.len()
    return eligible


def dedupe(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the lowest `dup_suffix` version of each filing.

    `dup_suffix` is sorted numerically (v2 before v10, unlike a string sort). Rows
    with a null `naic_code` cannot be grouped by filing, so all of them are kept.

    Args:
        frame: Eligible rows, possibly with repeated scrapes.

    Returns:
        A new DataFrame with one row per naic_code + filing_year.
    """
    version = frame["dup_suffix"].astype(str).str.extract(r"(\d+)", expand=False)
    ordered = frame.assign(_v=pd.to_numeric(version, errors="coerce").fillna(0))
    ordered = ordered.sort_values(["naic_code", "filing_year", "_v"], kind="mergesort")

    keyed = ordered.loc[ordered["naic_code"].notna()]
    unkeyed = ordered.loc[ordered["naic_code"].isna()]
    if len(unkeyed):
        logger.warning(
            "%d row(s) have a null naic_code and cannot be deduplicated by filing; "
            "all are kept.",
            len(unkeyed),
        )

    deduped = keyed.drop_duplicates(subset=["naic_code", "filing_year"], keep="first")
    result = pd.concat([deduped, unkeyed]).drop(columns="_v")
    logger.info(
        "After dedupe: %s (dropped %s repeated scrapes)",
        f"{len(result):,}",
        f"{len(frame) - len(result):,}",
    )
    return result.reset_index(drop=True)


def write_narratives(frame: pd.DataFrame) -> None:
    """Write the universe to a single parquet keyed by `sao_id`.

    Args:
        frame: The eligible rows to write.

    Raises:
        ValueError: If `sao_id` is not unique, which would make rows unaddressable.
    """
    if frame[KEY_COL].duplicated().any():
        raise ValueError(f"{KEY_COL} is not unique; rows would not be addressable.")

    out = frame[
        [KEY_COL, "naic_code", "filing_year", "company_name", "dup_suffix", "n_words",
         TEXT_COL]
    ].sort_values(KEY_COL, ignore_index=True)
    out.to_parquet(NARRATIVES_PATH, index=False)

    size_mb = NARRATIVES_PATH.stat().st_size / 1e6
    logger.info(
        "Wrote %s (%s rows, %.1f MB)",
        NARRATIVES_PATH.name,
        f"{len(out):,}",
        size_mb,
    )


def report_cost(frame: pd.DataFrame) -> None:
    """Log an estimated Batches API cost for extracting this set.

    Args:
        frame: The eligible rows that were written.
    """
    input_tokens = (TOKENS_INTERCEPT + TOKENS_PER_WORD * frame["n_words"]).sum()
    output_tokens = MEAN_OUTPUT_TOKENS * len(frame)
    batch = input_tokens / 1e6 * 2.50 + output_tokens / 1e6 * 12.50

    logger.info(
        "Estimated %.1fM input + %.1fM output tokens: about $%s on the Batches API "
        "with Opus 5, about $%s synchronous.",
        input_tokens / 1e6,
        output_tokens / 1e6,
        f"{batch:,.0f}",
        f"{batch * 2:,.0f}",
    )
    logger.info(
        "Filings per year:\n%s", frame["filing_year"].value_counts().sort_index()
    )


def main() -> None:
    """Build the extraction universe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", help="path to sao_texts_full_v3.parquet")
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="keep every scrape rather than the lowest dup_suffix per filing",
    )
    parser.add_argument(
        "--years",
        nargs=2,
        type=int,
        default=list(DEFAULT_YEARS),
        metavar=("MIN", "MAX"),
        help=f"inclusive filing-year range (default {DEFAULT_YEARS[0]} "
        f"{DEFAULT_YEARS[1]})",
    )
    args = parser.parse_args()

    source = resolve_source(args.source)
    eligible = load_eligible(source, (args.years[0], args.years[1]))
    if not args.no_dedup:
        eligible = dedupe(eligible)

    write_narratives(eligible)
    report_cost(eligible)
    logger.info("Next: python3 extract_api.py submit --limit 20")


if __name__ == "__main__":
    main()
