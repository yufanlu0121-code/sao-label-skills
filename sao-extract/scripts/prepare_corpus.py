"""Build the extraction corpus as a single parquet file.

TEMPLATE: adapt the column names, the eligibility filter, and the identifier-recovery
rule to your source. The structure -- one parquet rather than one file per document,
a stable key, deduplication with an explicit fallback, and recovery of identifiers
that a metadata join failed to supply -- is domain-independent.
"""

from __future__ import annotations

import argparse
import re
import logging
import os
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
NARRATIVES_PATH = BASE_DIR / "narratives.parquet"

TEXT_COL = "section_relevant_comments"
KEY_COL = "sao_id"
COLUMNS = [
    KEY_COL,
    "naic_code",
    "filing_year",
    "company_name",
    "dup_suffix",
    "pdf_basename",
    TEXT_COL,
]
# `{Company}_PNC-AS_{YYYY-MM-DD}...` -- the source filename carries the identifiers
# that the CIQ manifest join failed to supply.
BASENAME_RE = re.compile(r"^(.*?)_PNC-AS_(\d{4})-(\d{2})-(\d{2})")

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


def load_eligible(
    path: Path, years: tuple[int, int], recover: bool = True
) -> pd.DataFrame:
    """Read the source and keep rows with a usable narrative in range.

    Args:
        path: Path to the source parquet.
        years: Inclusive (min, max) filing-year filter.
        recover: Whether to recover missing identifiers from `pdf_basename` first.

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

    if recover:
        eligible = recover_metadata(eligible)
    else:
        eligible["_recovered"] = False

    lo, hi = years
    eligible = eligible.loc[eligible["filing_year"].between(lo, hi)].copy()
    logger.info("Filing years %d-%d: %s", lo, hi, f"{len(eligible):,}")

    eligible["n_words"] = eligible[TEXT_COL].astype(str).str.split().str.len()
    return eligible


def normalize_name(value: object) -> str:
    """Reduce a company name to a comparable form.

    Args:
        value: A company name, or anything else.

    Returns:
        A new lowercase string with punctuation and common suffixes removed.
    """
    text = re.sub(r"[^a-z0-9 ]", " ", str(value).lower())
    text = re.sub(r"\b(inc|llc|corp|corporation|co|company|the|a|an)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def recover_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill `filing_year`, `company_name` and `naic_code` from the source filename.

    114 eligible rows carry no `filing_year`, `naic_code` or `company_name`: their
    metadata comes from the CIQ archive manifest, and they were ingested without one
    (72 of them as loose `standalone` PDFs from a single 2021 event). The year filter
    then drops them silently, which is a data-quality exclusion masquerading as a
    design one -- and not a random one: 74 of the 114 are 2021 filings.

    `pdf_basename` carries both identifiers verbatim, and the NAIC code is recovered
    by matching the parsed name against names that appear elsewhere in this source
    with a code. Rows that cannot be matched keep their nulls and are handled by
    `dedupe` as before.

    Recovery is marked in `_recovered` so downstream work can report these rows
    separately, and so `dedupe` can prefer an original row over a recovered one when
    both describe the same filing.

    Args:
        frame: Eligible rows, before the year filter.

    Returns:
        A new DataFrame with recovered identifiers filled in.
    """
    out = frame.copy()
    out["_recovered"] = False
    missing = out["filing_year"].isna()
    if not missing.any():
        return out

    parsed = out.loc[missing, "pdf_basename"].map(
        lambda b: BASENAME_RE.match(str(b))
    )
    names = parsed.map(lambda m: m.group(1) if m else None)
    years = parsed.map(lambda m: int(m.group(2)) if m else pd.NA)

    lookup: dict[str, object] = {}
    coded = out.loc[out["naic_code"].notna()]
    for name, code in zip(coded["company_name"], coded["naic_code"]):
        if isinstance(name, str) and name.strip():
            lookup.setdefault(normalize_name(name), code)
    codes = names.map(lambda n: lookup.get(normalize_name(n)) if n else None)

    out.loc[missing, "company_name"] = names
    out.loc[missing, "filing_year"] = pd.array(years, dtype="Int64")
    out.loc[missing, "naic_code"] = codes
    out.loc[missing, "_recovered"] = True
    logger.info(
        "Recovered metadata for %d row(s) from pdf_basename: %d with a year, "
        "%d also matched to a naic_code.",
        int(missing.sum()),
        int(years.notna().sum()),
        int(codes.notna().sum()),
    )
    return out


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
    # `_recovered` precedes `_v` so that when an original row and a row recovered
    # from its filename describe the same filing, the original wins. Reversing this
    # would drop a filing that has already been extracted and silently replace it
    # with a different scrape of the same document.
    ordered = ordered.sort_values(
        ["naic_code", "filing_year", "_recovered", "_v"], kind="mergesort"
    )

    keyed = ordered.loc[ordered["naic_code"].notna()]
    unkeyed = ordered.loc[ordered["naic_code"].isna()]

    deduped = keyed.drop_duplicates(subset=["naic_code", "filing_year"], keep="first")

    # A null naic_code defeats the primary key, and keeping every such row meant
    # repeated scrapes of one filing survived as separate observations: United Fire
    # Group 2023 entered the corpus four times, four correlated rows in the panel and
    # four times the extraction cost for one document. Company name plus filing year
    # identifies the filing well enough to collapse them. Rows that lack a usable
    # name too are kept, since nothing can group them.
    named = unkeyed.loc[unkeyed["company_name"].notna()]
    nameless = unkeyed.loc[unkeyed["company_name"].isna()]
    fallback = named.drop_duplicates(subset=["company_name", "filing_year"], keep="first")
    if len(named) > len(fallback):
        logger.warning(
            "%d row(s) have a null naic_code; deduplicated on company_name + "
            "filing_year instead, dropping %d repeated scrape(s).",
            len(named),
            len(named) - len(fallback),
        )
    if len(nameless):
        logger.warning(
            "%d row(s) have neither a naic_code nor a company_name and cannot be "
            "deduplicated at all; all are kept.",
            len(nameless),
        )

    result = pd.concat([deduped, fallback, nameless]).drop(columns="_v")
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
         "_recovered", TEXT_COL]
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
        "--no-recover",
        action="store_true",
        help="skip recovering identifiers from pdf_basename for rows missing them",
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
    eligible = load_eligible(
        source, (args.years[0], args.years[1]), recover=not args.no_recover
    )
    if not args.no_dedup:
        eligible = dedupe(eligible)

    write_narratives(eligible)
    report_cost(eligible)
    logger.info("Next: python3 extract_api.py submit --limit 20")


if __name__ == "__main__":
    main()
