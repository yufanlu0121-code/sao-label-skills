# sao-label-skills

Private skills for Claude Code.

Install by copying a skill directory into `~/.claude/skills/`:

```bash
cp -R sao-extract ~/.claude/skills/
```

---

## sao-extract

Reproducible, homogeneous LLM extraction over a whole corpus via the Anthropic Batches
API — turning many documents into one structured record each, for statistical analysis.

### The problem it addresses

Running a prompt over N documents is easy. Producing a **dataset you can defend in a
regression** is not, and the gap is almost entirely made of failures that do not raise
errors.

An extraction that becomes research data has to satisfy three things at once:

- **Homogeneity** — every document scored under the identical instrument, model and
  configuration, and provably so, not just intended. A run split across two prompt
  versions or two model versions is two measurements presented as one.
- **Resumability** — at N in the thousands, interruption is certain. Resuming must never
  double-bill, silently skip a document, or treat a half-written file as finished.
- **Honesty about its own error** — the extracted fields carry measurement error that
  attenuates coefficients. That error has to be measured and reported, not assumed away.

The failure modes that break these are quiet. A truncated response looks like a short
one. A prompt revision that the JSON Schema constrains away completes cleanly and
changes nothing. A metadata join that lost some rows removes them at the next filter,
with no exclusion recorded and no reason to suspect the slice removed was biased. None
of these throw.

### When to use it

Whenever a corpus has to become a dataset: extracting structured fields, codes or
annotations from many documents — filings, transcripts, reports, articles, abstracts,
case notes. Also mid-job, when a run hits truncation, JSON parse failures, cost
surprises, drift between batches, or the question of how reproducible the resulting
variables actually are.

Especially where the extracted fields become regression variables, since it covers
measurement reliability, validity against human annotation, and the correction that
follows from them.

### The pipeline

Eight steps, in order. Step 4's numbers decide whether step 5 is worth running at all.

1. Build the corpus as **one file**, not one file per document
2. Freeze the instrument (the prompt) and hash it into a treatment signature
3. Constrain output with a JSON Schema
4. Pilot: measure truncation, contamination and reproducibility
5. Submit via the Batches API, resumably
6. Check the output against the source
7. Validate against human annotation, and correct for the error you measure
8. Assemble, reporting reliability alongside the estimates

`SKILL.md` works through each, and marks which decisions belong to the researcher rather
than the implementer — year ranges, deduplication keys, whether thinking stays on,
whether one pass is enough. Those are research-design choices; the skill surfaces them
with numbers attached instead of quietly picking the cheaper option.

### The failure-mode catalogue

`references/pitfalls.md` documents fifteen, each with the mechanism, the symptom and the
fix:

| | |
|---|---|
| 1 | Truncation eats the longest documents |
| 1a | Truncation is not a function of document length |
| 1b | The near-limit counter must include truncated responses |
| 2 | Output contamination survives JSON parsing |
| 3 | Nullable enums are rejected in JSON Schema |
| 4 | "Done" cannot mean "the file exists" |
| 5 | Batch results come back in arbitrary order |
| 6 | A batch belongs to the account that created it |
| 7 | The served model can differ from the requested model |
| 8 | Composite keys break on null components |
| 9 | Cost fits extrapolate badly |
| 10 | Quotation checks report false splices on PDF furniture |
| 11 | A revised prompt silently constrained by a stale schema |
| 12 | Document ids that are legal locally but illegal in the API |
| 13 | A failure file that never clears |
| 14 | A metadata join that fails silently removes a biased slice |
| 15 | The context you show the annotator can encode the model's answer |

Several are counter-intuitive enough to be worth naming here. Truncation is **not** a
length problem, so probing your longest documents does not bound the risk — with
adaptive thinking the reasoning tokens vary independently of the input, and the remedy
for a truncated document is to re-submit it unchanged rather than raise the ceiling.
Quotation checks that verify a quoted span appears in the source will report false
splices on any PDF corpus unless running page furniture is stripped first. And a blind
annotation workbook is not blind merely because the labels were withheld: anything whose
*shape* varies with the model's output leaks it.

### Measurement quality

Two distinct questions, routinely conflated, each with its own reference file:

- **Reproducibility** (`references/reproducibility.md`) — does the model agree with
  *itself* across passes? Measured by repeated runs; reported as ICC, which determines
  how much coefficients on the extracted variables are attenuated.
- **Validity** (`references/validation.md`) — does it agree with a *careful human
  reading*? Measured by a hand-annotated validation sample under a written codebook,
  and carried into the second stage as a misclassification correction.

A model can be perfectly reproducible and consistently wrong. Reliability statistics say
nothing about validity, and repeated passes cannot detect a systematic error.

The validation design has two rules that are easy to break and fatal when broken. The
annotator's codebook and the extraction prompt must stay **separate documents** —
copying boundary rules from one into the other drives the measured disagreement toward
zero by construction. And when the sample is stratified on the model's own predicted
labels, which is the efficient design, the **inclusion probabilities must reach the
analysis stage**: an unweighted analysis of a disproportionately allocated sample is
wrong, will not error, and nothing downstream will flag it.

### Contents

| Path | What it is |
|---|---|
| `SKILL.md` | The eight-step pipeline, and which decisions belong to the researcher |
| `references/pitfalls.md` | Fifteen silent failure modes, with the exact errors and fixes |
| `references/reproducibility.md` | Measuring run-to-run variance; what ICC means for inference |
| `references/validation.md` | Validation sample design, codebook discipline, stratification and weights |
| `assets/codebook_template.md` | Starting point for the annotator's codebook |
| `scripts/prepare_corpus.py` | Builds the single-file corpus; deduplication with a fallback key; identifier recovery |
| `scripts/extract_api.py` | submit / status / fetch, with homogeneity enforcement and the pre-run checks |
| `scripts/handcheck.py` | Locates quoted spans in the source and diagnoses the failures |
| `scripts/validation_sample.py` | Simple random annotation sample as markdown, `--blind` to withhold labels |
| `scripts/build_validation_workbook.py` | Stratified two-level draw, recorded weights, protected xlsx |
| `scripts/report.py` | Progress, field distributions, document-level shares |

The scripts are working code rather than sketches, written against one project and
marked `TEMPLATE` where they need adapting — the schema, the column names, the
eligibility filter. The control flow is the part to keep.

### Provenance and status

Derived from a real research extraction over a full corpus of regulatory filings, run
end to end: pilot, full run, quotation check, metadata recovery, and the validation
instrument. The figures quoted inside the reference files are real measurements from
that run, including the ones that record mistakes — a falsified assumption is left in
place with the correction beside it, because the reasoning that produced it is the part
worth transferring.

What this repository does **not** contain is an accuracy figure for LLM extraction. The
validation instrument is here and the sample is drawn; the hand-coding that would
produce a misclassification rate has not been done. No number is quoted for it, and
none should be inferred.
