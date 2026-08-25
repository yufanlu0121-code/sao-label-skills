"""Export a hand-annotation workbook for the validation sample.

Draws filings at random from the completed corpus and writes one markdown section
per filing: the narrative, then the model's labels beside empty columns for the
annotator. Nothing here scores, judges, or pre-fills a human column -- the point of
the exercise is to measure where a careful human reading differs from the model, and
a machine-suggested answer in a human column would destroy that measurement.

The fields left for coding are `orientation`, `object`, and `moment` at statement
level and `rmad_conclusion` at document level. Boundary rules live in `codebook.md`;
this file deliberately does not restate them, so the codebook stays the single
authority that can be amended and dated.

⚠️ This workbook shows the model's labels next to the blank columns. That is
convenient, but it is not blind annotation: seeing a label first anchors the
annotator and inflates apparent agreement. `--blind` writes the narrative and empty
columns only, with the model's labels in a separate key file to be consulted after
coding. Prefer `--blind` when the resulting rate is going into a bias correction.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
NARRATIVES_PATH = BASE_DIR / "narratives.parquet"
DEFAULT_OUT_ROOT = Path(
    os.environ.get("SAO_OUT_ROOT", Path.home() / "sao_extract_output")
)
TEXT_COLUMNS = ("text", "section_relevant_comments")
RANDOM_STATE = 666

STATEMENT_FIELDS = ["orientation", "object", "moment"]
DOCUMENT_FIELDS = ["rmad_conclusion"]

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout
)
logger = logging.getLogger(__name__)


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


def cell(value: Any) -> str:
    """Render a value safely inside a markdown table cell.

    Args:
        value: Any JSON-decoded value.

    Returns:
        A new single-line string with pipes escaped.
    """
    if value is None:
        return "_null_"
    return " ".join(str(value).split()).replace("|", "\\|")


def render(
    index: int,
    sao_id: str,
    row: pd.Series,
    extraction: dict[str, Any],
    blind: bool,
) -> str:
    """Render one filing's annotation section.

    Args:
        index: Position in the sample, used as a stable section number.
        sao_id: The filing key.
        row: Its corpus row.
        extraction: The decoded extraction JSON.
        blind: Whether to omit the model's labels.

    Returns:
        A new markdown string.
    """
    statements = extraction.get("statements", [])
    out = [
        f"## {index}. `{sao_id}`",
        "",
        f"**Company:** {row['company_name']}  ",
        f"**NAIC:** {row['naic_code']}  |  **Year:** {row['filing_year']}  |  "
        f"**Words:** {row['n_words']}  |  **Statements:** {len(statements)}",
        "",
        "### Narrative",
        "",
        "```text",
        narrative_of(row),
        "```",
        "",
        "### Statements",
        "",
    ]
    if blind:
        header = ["#", "verbatim"] + [f"your {f}" for f in STATEMENT_FIELDS]
    else:
        header = ["#", "verbatim"]
        for field in STATEMENT_FIELDS:
            header += [f"model {field}", f"your {field}"]
    out += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    for number, statement in enumerate(statements, 1):
        cells = [str(number), cell(statement.get("verbatim"))]
        for field in STATEMENT_FIELDS:
            if not blind:
                cells.append(cell(statement.get(field)))
            cells.append(" ")
        out.append("| " + " | ".join(cells) + " |")
    if not statements:
        out.append(
            "| — | _no risk statements returned_ |" + " |" * (len(header) - 2)
        )

    out += ["", "### Document", ""]
    if blind:
        out += ["| field | your value |", "|---|---|"]
        for field in DOCUMENT_FIELDS:
            out.append(f"| `{field}` |  |")
    else:
        out += ["| field | model | your value |", "|---|---|---|"]
        for field in DOCUMENT_FIELDS:
            out.append(f"| `{field}` | {cell(extraction.get(field))} |  |")
    out += ["", "---", ""]
    return "\n".join(out)


def build(n: int, out_dir: Path, blind: bool) -> tuple[str, pd.DataFrame]:
    """Draw the sample and render the workbook.

    Args:
        n: How many filings to draw.
        out_dir: Directory of extraction JSON files.
        blind: Whether to omit the model's labels.

    Returns:
        A tuple of the markdown document and the drawn sample's index frame.
    """
    corpus = pd.read_parquet(NARRATIVES_PATH).set_index("sao_id")
    extracted = [k for k in corpus.index if (out_dir / f"{k}.json").exists()]
    if len(extracted) < len(corpus):
        logger.warning(
            "%d of %d filings have no extraction and cannot be drawn.",
            len(corpus) - len(extracted),
            len(corpus),
        )
    frame = corpus.loc[extracted]
    drawn = frame.sample(n=min(n, len(frame)), random_state=RANDOM_STATE)

    header = [
        f"# Validation sample — {len(drawn)} filings",
        "",
        f"Drawn at random from the {len(frame)} extracted filings, "
        f"`random_state={RANDOM_STATE}`.  ",
        "Boundary rules: `codebook.md`. Annotate to that file, and amend it there "
        "rather than here if a case is not covered.",
        "",
        "Code `orientation`, `object`, `moment` per statement and `rmad_conclusion` "
        "per filing in the blank columns.",
        "",
        "**This file is the record.** Type your codes into the blank columns and "
        "save it in place — it lives in the project directory so it syncs and is "
        "backed up. Nothing regenerates it: `validation_sample.py` refuses to "
        "overwrite an existing workbook without `--force`.",
        "",
    ]
    if not blind:
        header += [
            "⚠️ The model's labels are shown beside the blank columns. This is not "
            "blind annotation — reading a label first anchors the coder and inflates "
            "agreement. Use `--blind` if this sample's disagreement rate is going "
            "into a bias correction.",
            "",
        ]
    header += ["---", ""]

    body = []
    for position, sao_id in enumerate(drawn.index, 1):
        extraction = json.loads((out_dir / f"{sao_id}.json").read_text())
        body.append(render(position, sao_id, drawn.loc[sao_id], extraction, blind))
    return "\n".join(header) + "".join(body), drawn.reset_index()


def main() -> None:
    """Parse arguments and write the workbook and its index."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100, help="filings to draw")
    parser.add_argument("--dir", default="raw_full", help="extraction directory")
    parser.add_argument(
        "--out",
        default="validation_coding.md",
        help="output file, relative to this directory unless absolute",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="also write the draw index as CSV beside the workbook",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing workbook (destroys any coding already entered)",
    )
    parser.add_argument(
        "--blind",
        action="store_true",
        help="omit the model's labels so annotation is not anchored",
    )
    args = parser.parse_args()

    out_dir = DEFAULT_OUT_ROOT / args.dir
    if not out_dir.exists():
        raise SystemExit(f"{out_dir} does not exist.")

    # The workbook is the one artefact here that a human types into, so it lives
    # beside the code in the synced project directory rather than in the extraction
    # output root. The rule that keeps output out of Dropbox exists because 8,532
    # machine-written JSON files churn the sync client; a single hand-edited file is
    # the case that rule is protecting, not the case it is guarding against.
    target = Path(args.out)
    if not target.is_absolute():
        target = BASE_DIR / target
    if target.exists() and not args.force:
        raise SystemExit(
            f"{target} already exists. Coding entered there would be overwritten — "
            "pass --force only if you are sure, or choose another --out."
        )

    document, drawn = build(args.n, out_dir, args.blind)
    target.write_text(document, encoding="utf-8")
    written = [target.name]
    if args.index:
        index_path = target.with_suffix(".csv")
        drawn[
            ["sao_id", "naic_code", "filing_year", "company_name", "n_words"]
        ].to_csv(index_path, index=False)
        written.append(index_path.name)
    logger.info(
        "Wrote %s (%d filings, %d bytes) in %s.",
        " and ".join(written),
        len(drawn),
        len(document),
        target.parent,
    )


if __name__ == "__main__":
    main()
