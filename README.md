# sao-label-skills

Private skills for Claude Code.

Install by copying a skill directory into `~/.claude/skills/`:

```bash
cp -R sao-extract ~/.claude/skills/
```

## sao-extract

Reproducible, homogeneous LLM extraction over a whole corpus via the Anthropic
Batches API — turning many documents into one structured record each, for
statistical analysis.

Covers the parts that are easy to get wrong and do not raise errors: truncation
that selectively eats the longest documents, output contamination that survives
JSON parsing, treatment drift across batches, prompt revisions silently voided by a
stale schema, quotation checks that report false splices on PDF page furniture, and
the measurement reliability (ICC) and validity of variables derived this way.

Built for the SAO project — extracting risk statements from US P&C Statement of
Actuarial Opinion filings — and every number quoted in the reference files is a real
measurement from that run, including the mistakes.

### Where that run actually stands

| | |
|---|---|
| Extraction | **Complete.** 8,532 filings, 95,142 statements, 21 batches, $395.37 |
| Homogeneity | One treatment signature throughout; all 8,533 responses served by one model |
| Failures | None outstanding. 1 truncation in 8,470, fixed by re-submitting unchanged |
| Quotation check | 56/56 spans located in source on the hand-checked filings, 0 splices |
| Validation sample | **Drawn, not yet coded.** 1,000 statements across 25 strata + 150 documents |
| Misclassification rates | **Not yet measured.** The workbook exists; the hand-coding has not been done |

The corpus was 8,475 for most of the run and is 8,532 now: deduplication was fixed
(−3, one filing had entered four times) and 60 filings whose metadata a failed join had
lost were recovered from their source filenames (+60). Both are documented as pitfalls
rather than tidied away.

Nothing here reports an accuracy figure for the extraction, because none has been
measured yet. `references/validation.md` describes how it will be.

### Contents

| Path | What it is |
|---|---|
| `SKILL.md` | The six-step pipeline and which decisions belong to the researcher |
| `references/pitfalls.md` | Nine silent failure modes, with the exact errors and fixes |
| `references/reproducibility.md` | Measuring run-to-run variance; what ICC means for inference |
| `references/validation.md` | Human-annotated validation sample; codebook discipline; correcting for measured misclassification |
| `assets/codebook_template.md` | Starting point for the annotator's codebook |
| `scripts/` | Working submit/status/fetch pipeline, corpus builder, quotation check, validation-sample workbook, reporting |
