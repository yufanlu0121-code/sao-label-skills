"""Assemble extraction records into item-level and document-level tables.

TEMPLATE: adapt the categorical lists and the aggregate definitions to your schema.
The structure -- addressing records by a stable key rather than a parsed filename,
joining the corpus for metadata, reconciling the two tables against each other, and
refusing to assemble an incomplete run -- is domain-independent.

Documents with no extracted items get NaN for every share variable, not 0. A document
containing no items is a different object from one containing items that all happen to
be of the other category, and collapsing the two silently changes the regression
sample. `n_items` is 0 and `has_statements` is False for those rows; how to handle
them is left to the analysis.

Two guards worth keeping. The loader refuses to run while any document in the corpus
lacks an extraction, because assembling early drops those rows with nothing to show
for it. And the two tables are reconciled -- item rows must equal the summed per-
document counts -- which catches a key collision that would otherwise surface as a
quietly wrong regression.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
NARRATIVES_PATH = BASE_DIR / "narratives.parquet"
# Extractions and the tables built from them live outside the synced project
# directory, alongside the raw records they derive from.
DEFAULT_OUT_ROOT = Path(
    os.environ.get("SAO_OUT_ROOT", Path.home() / "sao_extract_output")
)
RAW_DIRNAME = "raw_full"
KEY = "sao_id"

SPARSITY_THRESHOLD = 0.05

STATEMENT_CATEGORICALS = [
    "orientation",
    "object",
    "moment",
    "direction",
    "persistence",
    "hedge",
    "quantified",
]
DOCUMENT_CATEGORICALS = [
    "rmad_conclusion",
    "reserve_stance",
    "carried_vs_estimate",
    "materiality_denominator",
    "section_truncated",
]
DOCUMENT_PASSTHROUGH = DOCUMENT_CATEGORICALS + [
    "rmad_verbatim",
    "reserve_stance_verbatim",
    "materiality_amount",
    "materiality_pct",
]

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout
)
logger = logging.getLogger(__name__)


def load_records(raw_dir: Path) -> list[dict[str, Any]]:
    """Load every extraction record, keyed by `sao_id`.

    Records are addressed by `sao_id`, never by a `{naic_code}_{filing_year}` pair
    parsed out of the filename. Some filings have a null `naic_code` and fall back to
    a company-name key, so filename parsing raised on every record; and a parsed pair
    cannot be joined back to the corpus for the metadata the tables need. The corpus
    parquet is the authority for `naic_code`, `filing_year`, `company_name` and
    `_recovered`.

    Args:
        raw_dir: Directory of extraction JSON files.

    Returns:
        A list of records carrying their corpus metadata.

    Raises:
        FileNotFoundError: If the corpus or the extraction directory is missing.
        RuntimeError: If any filing in the corpus has no extraction.
    """
    if not NARRATIVES_PATH.exists():
        raise FileNotFoundError(
            f"{NARRATIVES_PATH.name} not found — run prepare_full.py first."
        )
    corpus = pd.read_parquet(NARRATIVES_PATH).set_index(KEY)
    if "_recovered" not in corpus.columns:
        corpus["_recovered"] = False

    records = []
    missing = []
    for sao_id in corpus.index:
        path = raw_dir / f"{sao_id}.json"
        if not path.exists():
            missing.append(sao_id)
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        row = corpus.loc[sao_id]
        record[KEY] = sao_id
        record["naic_code"] = row["naic_code"]
        record["filing_year"] = int(row["filing_year"])
        record["company_name"] = row["company_name"]
        record["n_words"] = int(row["n_words"])
        record["_recovered"] = bool(row["_recovered"])
        records.append(record)

    if missing:
        raise RuntimeError(
            f"{len(missing)} filing(s) in the corpus have no extraction, e.g. "
            f"{missing[:3]}. Assembling now would silently drop them — run "
            "`extract_api.py submit` until the corpus is complete."
        )
    logger.info("Loaded %s records from %s/", f"{len(records):,}", raw_dir.name)
    return records


def build_statements(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Explode records into a statement-level frame.

    Args:
        records: Raw extraction records.

    Returns:
        A new DataFrame with one row per risk statement.
    """
    rows = []
    for record in records:
        statements = record.get("statements")
        if not isinstance(statements, list):
            continue
        for i, stmt in enumerate(statements):
            if not isinstance(stmt, dict):
                continue
            rows.append(
                {
                    KEY: record[KEY],
                    "stmt_id": f"{record[KEY]}#{i}",
                    "naic_code": record["naic_code"],
                    "filing_year": record["filing_year"],
                    "_recovered": record["_recovered"],
                    "statement_index": i,
                    **{k: stmt.get(k) for k in (
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
                    )},
                }
            )

    frame = pd.DataFrame(rows)
    logger.info("Statement-level rows: %d", len(frame))
    return frame


def build_filing_level(
    records: list[dict[str, Any]], statements: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate statements to filing level and attach document-level fields.

    Args:
        records: Raw extraction records.
        statements: The statement-level frame.

    Returns:
        A new DataFrame with one row per filing.
    """
    doc = pd.DataFrame(
        [
            {
                KEY: r[KEY],
                "naic_code": r["naic_code"],
                "filing_year": r["filing_year"],
                "company_name": r["company_name"],
                "n_words": r["n_words"],
                "_recovered": r["_recovered"],
                **{f: r.get(f) for f in DOCUMENT_PASSTHROUGH},
                "_model": r.get("_model"),
                "_cost_usd": r.get("_cost_usd"),
                "_prompt_sha256": r.get("_prompt_sha256"),
            }
            for r in records
        ]
    )

    if statements.empty:
        agg = pd.DataFrame(columns=[KEY])
    else:
        grouped = statements.groupby(KEY)
        agg = pd.DataFrame(
            {
                "n_statements": grouped.size(),
                "prospective_share": grouped["orientation"].apply(
                    lambda s: s.isin(["prospective", "both"]).mean()
                ),
                "retrospective_share": grouped["orientation"].apply(
                    lambda s: s.isin(["retrospective", "both"]).mean()
                ),
                "uncertainty_share": grouped["moment"].apply(
                    lambda s: s.isin(["uncertainty", "both"]).mean()
                ),
                # Two different constructs that were both called `method_shock`:
                # report.py and HANDOFF s7.1 define it as the *share* of statements
                # about the actuary's own method, while this file previously computed
                # a 0/1 "any such statement" indicator. Both are meaningful -- how
                # much of the narrative is about method risk, versus whether method
                # risk was raised at all -- so they are kept under separate names
                # rather than one silently winning. Nothing was ever assembled under
                # the old name: this script has never run to completion.
                "method_shock": grouped["object"].apply(
                    lambda s: (s == "estimate_method").mean()
                ),
                "method_shock_any": grouped["object"].apply(
                    lambda s: int((s == "estimate_method").any())
                ),
                "ongoing_share": grouped["persistence"].apply(
                    lambda s: (s == "ongoing").mean()
                ),
                "quantified_share": grouped["quantified"].apply(
                    lambda s: s.fillna(False).astype(bool).mean()
                ),
                "unconditional_share": grouped["hedge"].apply(
                    lambda s: (s == "unconditional").mean()
                ),
            }
        ).reset_index()

    filing = doc.merge(agg, on=KEY, how="left")

    # Zero-statement filings: count and flag are known, shares are undefined.
    filing["n_statements"] = filing["n_statements"].fillna(0).astype(int)
    filing["has_statements"] = filing["n_statements"] > 0
    for column in (
        "prospective_share",
        "retrospective_share",
        "uncertainty_share",
        "method_shock",
        "method_shock_any",
        "ongoing_share",
        "quantified_share",
        "unconditional_share",
    ):
        filing[column] = np.where(filing["has_statements"], filing[column], np.nan)

    logger.info(
        "Filing-level rows: %d (%d with no statements)",
        len(filing),
        int((~filing["has_statements"]).sum()),
    )
    return filing


def report_distributions(statements: pd.DataFrame, filing: pd.DataFrame) -> None:
    """Log the distribution of every categorical field.

    Args:
        statements: The statement-level frame.
        filing: The filing-level frame.
    """
    logger.info("=== statement-level categoricals ===")
    for column in STATEMENT_CATEGORICALS:
        if column not in statements:
            continue
        counts = statements[column].value_counts(dropna=False)
        shares = (counts / len(statements) * 100).round(1)
        logger.info(
            "%s\n%s", column, pd.DataFrame({"n": counts, "pct": shares}).to_string()
        )

    logger.info("=== document-level categoricals ===")
    for column in DOCUMENT_CATEGORICALS:
        if column not in filing:
            continue
        counts = filing[column].value_counts(dropna=False)
        shares = (counts / len(filing) * 100).round(1)
        logger.info(
            "%s\n%s", column, pd.DataFrame({"n": counts, "pct": shares}).to_string()
        )


def report_crosstab(statements: pd.DataFrame) -> None:
    """Log the orientation x object x moment crosstab and flag sparse margins.

    The full three-way table has 3 x 4 x 3 = 36 cells, so the mean cell holds
    about 2.8% of statements and a blanket "<5% of statements" rule would flag
    nearly every cell regardless of the data. The three-way table is therefore
    reported as raw counts, and the 5% sparsity flag is applied to the two-way
    margins, where it carries information.

    Args:
        statements: The statement-level frame.
    """
    needed = {"orientation", "object", "moment"}
    if statements.empty or not needed.issubset(statements.columns):
        logger.warning("Cannot build crosstab — statement frame is empty or missing columns.")
        return

    total = len(statements)

    logger.info("=== orientation x object x moment (counts, %d statements) ===", total)
    three_way = statements.groupby(
        ["orientation", "object", "moment"], dropna=False
    ).size().rename("n").reset_index()
    three_way["pct"] = (three_way["n"] / total * 100).round(1)
    logger.info("\n%s", three_way.sort_values("n", ascending=False).to_string(index=False))

    empty = 36 - len(three_way)
    if empty > 0:
        logger.info("%d of 36 possible three-way cells are empty.", empty)

    logger.info("=== two-way margins (sparsity flagged below %.0f%%) ===", SPARSITY_THRESHOLD * 100)
    for a, b in (("orientation", "object"), ("orientation", "moment"), ("object", "moment")):
        table = pd.crosstab(statements[a], statements[b], dropna=False)
        logger.info("%s x %s\n%s", a, b, table.to_string())
        shares = table / total
        sparse = [
            (str(r), str(c), int(table.loc[r, c]), round(shares.loc[r, c] * 100, 1))
            for r in table.index
            for c in table.columns
            if shares.loc[r, c] < SPARSITY_THRESHOLD
        ]
        if sparse:
            logger.warning(
                "Sparse cells in %s x %s (n, pct): %s",
                a,
                b,
                ", ".join(f"{r}/{c}: {n} ({p}%)" for r, c, n, p in sparse),
            )


def main() -> None:
    """Assemble both tables and write them beside the extraction records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir", default=RAW_DIRNAME, help=f"extraction directory (default {RAW_DIRNAME})"
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_ROOT),
        help="where the two parquet tables are written",
    )
    args = parser.parse_args()

    raw_dir = DEFAULT_OUT_ROOT / args.dir
    if not raw_dir.exists():
        raise SystemExit(f"{raw_dir} does not exist — run `extract_api.py fetch` first.")
    records = load_records(raw_dir)
    statements = build_statements(records)
    filing = build_filing_level(records, statements)

    if len(filing) != len(records):
        raise RuntimeError(f"{len(filing)} filing rows from {len(records)} records — key collision.")
    if not statements.empty and len(statements) != int(filing["n_statements"].sum()):
        raise RuntimeError(
            f"{len(statements)} statement rows but n_statements sums to "
            f"{int(filing['n_statements'].sum())} — counts do not reconcile."
        )

    if statements[KEY].isna().any() or statements["stmt_id"].duplicated().any():
        raise RuntimeError("stmt_id is not unique — statement rows are unaddressable.")
    if filing[KEY].duplicated().any():
        raise RuntimeError(f"{KEY} is not unique in the filing table.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    statements_path = out_dir / "statements.parquet"
    filing_path = out_dir / "filing_level.parquet"
    statements.to_parquet(statements_path, index=False)
    filing.to_parquet(filing_path, index=False)
    logger.info("Wrote %s and %s", statements_path, filing_path)

    report_distributions(statements, filing)
    report_crosstab(statements)


if __name__ == "__main__":
    main()
