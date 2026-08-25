---
name: sao-extract
description: Run a reproducible, homogeneous LLM extraction over a whole corpus of documents via the Anthropic Batches API, producing one structured record per document for statistical analysis. Built for the SAO project — extracting risk statements from US P&C Statement of Actuarial Opinion filings — but applies to any corpus-to-dataset job. Use this whenever the user wants to extract structured fields, codes, or annotations from many documents (filings, transcripts, reports, articles, abstracts, case notes) into a dataset, including phrases like "extract X from all these documents", "code these texts", "turn this corpus into a dataset", "run my prompt over N files", or "content analysis at scale". Also use it when someone is midway through such a job and hits truncation, JSON parse failures, cost surprises, drift between batches, or asks how reproducible their LLM-derived variables are. Especially important for research where the extracted fields become regression variables, since it covers measurement reliability (ICC), validity against hand annotation, treatment homogeneity, and the silent failure modes that bias results. Also use it when revising a prompt mid-project, verifying that quoted spans really appear in the source, or building a hand-annotated validation sample to correct estimates for misclassification.
---

# SAO Extract — reproducible LLM corpus extraction

Turning a corpus into a dataset with an LLM is not "run a prompt N times". The
output becomes measurement — often regression variables — so the run has to be
**homogeneous** (every document scored by the same instrument), **resumable**
(interruptions are certain at N in the thousands), and **honest about its own
reliability**.

This skill covers the pipeline and, more importantly, the failure modes that
silently corrupt results rather than raising errors.

## The shape of the job

1. Build the corpus as **one file**, not one file per document
2. Freeze the instrument (the prompt) and hash it
3. Constrain output with a JSON Schema
4. Pilot: measure truncation, contamination, and reproducibility
5. Submit via the Batches API, resumably
6. Check the output against the source
7. Validate against human annotation, and correct for the error you measure
8. Assemble, reporting reliability alongside the estimates

Work through these in order. Step 4's numbers determine whether step 5 is worth
running at all.

## 1. One corpus file, never one file per document

Read documents from a single parquet (or database) keyed by a stable id.

Materialising one `.txt` per document is the reflex to resist. At N in the
thousands it churns sync clients (Dropbox, iCloud, OneDrive), takes minutes to
delete, and buys nothing — the text has to be loaded into memory to build each API
request either way.

Key rows by an id that always exists. A composite key built from metadata
(`{entity}_{year}`) breaks the moment some rows have a null component, and the
breakage is silent: those documents just never get processed.

The same nulls break deduplication, which is worse because it is invisible: rows the
key cannot group are usually kept, so repeated scrapes of one document survive as
separate observations — four correlated rows in the panel and four times the cost for
one document. Give deduplication an explicit fallback key and state what happens to
rows that match neither.

And check what the metadata join dropped before it reached you. Documents that lost
their metadata are removed by whatever filter comes next, with no exclusion recorded,
and the slice removed is rarely random — `references/pitfalls.md` #14 has a case where
65% of them came from one year, which then looked like the thinnest year in the
panel. Reconcile the exclusion counts; if they do not sum, that is the tell.

## 2. Freeze the instrument

The prompt is a measurement instrument. Editing it mid-run makes documents scored
before and after incomparable, and nothing in the API will tell you this happened.

Keep it in its own file, read it byte-for-byte at submit time, and hash it into a
treatment signature stored with every batch:

```python
{"model": "...", "max_tokens": N, "thinking": "adaptive",
 "effort": "high", "structured": True, "prompt_sha256": "..."}
```

Before submitting, compare the signature against batches already submitted and
**abort on any difference**. This is the difference between "we intended one
treatment" and "we can show one treatment".

Also record **the model that actually served each response** (`message.model`),
not just the one requested. A run silently split across model versions is two
measurements presented as one. See `references/pitfalls.md`.

**When the instrument legitimately changes.** Researchers do revise prompts, and the
revision is a new instrument rather than an edit. Four things follow, and all four
are easy to skip: snapshot the prompt *before* editing (a prompt edited in place is
unrecoverable, and it is the audit trail for everything already extracted under it);
diff the revision against the JSON Schema, since a new field or enum value the schema
does not list is silently constrained away and the revision measures nothing; archive
records extracted under the old instrument rather than merging them; and re-run the
pilot, because all four pre-run checks belong to the instrument, not the corpus.
`references/pitfalls.md` #11 has the detail.

## 3. Constrain the output

Use `output_config.format` with a JSON Schema rather than asking for JSON in
prose. This eliminates parse failures structurally instead of handling them.

Nullable enums need `anyOf`, not a type list — `{"type": ["string","null"],
"enum": [...]}` is rejected. See `references/pitfalls.md` for the exact error and
fix.

## 4. Pilot before committing

Run a small pilot and check four things. Each maps to a failure mode that
otherwise contaminates the full run quietly.

**Pilot on a spread, not the first N.** Taking the first N documents by id samples
whatever the id happens to sort by, which is rarely length. A pilot exists to surface
length-dependent failures, so it has to reach both ends of the distribution and must
include the longest document.

**Truncation.** With thinking enabled, reasoning tokens draw from the *same*
`max_tokens` budget as the answer. A ceiling sized for the expected answer gets
eaten by reasoning and the response truncates. This is not random loss — it hits
the longest, most content-dense documents, producing missingness correlated with
whatever makes documents long.

Deliberately pilot the **longest** documents in the corpus, not a random sample,
and confirm `stop_reason == "end_turn"`. Set the ceiling several times above the
observed worst case.

**Contamination.** Structured output should return bare JSON. Check the response
body before parsing: it must start `{` and end `}`, and must not contain reasoning
tags or code fences. A leaked marker *inside a string field* still parses as valid
JSON and passes silently.

Keep the check narrow. Domain text routinely contains `<` (as "less than"), so a
bare `<` cannot be a marker — flagging it quarantines valid records.

**Reproducibility.** Run the same documents twice under identical settings. This
is the single most informative thing you can do, and it is usually skipped.
`references/reproducibility.md` covers what to measure and how to interpret it.

**Cost.** Fit output tokens on document length across the *full* length range.
A fit estimated on short documents extrapolates badly to the tail.

⚠️ **A length-ordered probe does not bound truncation risk.** With adaptive thinking,
output length is driven by the reasoning path, not the input: on a full run the only
truncated document sat at the 98th percentile of length while the longest completed,
and re-submitting it unchanged finished in 3,295 tokens against 32,000+ before. Probe
the longest documents to learn what long documents cost, then treat truncation as a
lottery to be retried, not a length threshold to be engineered around
(`references/pitfalls.md` #1a).

## 5. Submit resumably

Use the Batches API — half price, and asynchronous so long generations don't hit
HTTP timeouts. (A large `max_tokens` is refused on non-streaming synchronous
requests for that reason; batches are exempt.)

Three properties matter:

- **"Done" means the output file parses**, not that it exists. An interrupted
  write leaves truncated JSON; existence checks mark it done forever and the
  document silently drops out.
- **Results are keyed by `custom_id`, never by position.** Batch results return in
  arbitrary order.
- **Failures stay pending.** A truncated, contaminated, or errored response is
  quarantined and not written, so the next submit retries it. Never hand-patch. Make
  the failure record describe what is outstanding *now* — one written only when a
  fetch produces failures will keep a stale entry alive after a successful retry, and
  anything keyed on its existence then fires forever (`references/pitfalls.md` #13).
- **The transport id is not necessarily your document id.** `custom_id` must match
  `^[a-zA-Z0-9_-]{1,64}$`, and one offending key rejects the whole batch with a 400,
  leaving a chunked run half-submitted. Substitute a deterministic digest and keep a
  reverse mapping (#12).

Write output **outside any synced folder**. Thousands of small files appearing over
hours will fight the sync client, and a half-synced output directory is real
corruption risk.

`scripts/extract_api.py` is a working implementation of all of this — submit /
status / fetch, with the homogeneity check, both pre-run checks, and multi-account
key handling. Adapt the schema and column names; keep the control flow.

## 6. Check the output against the source

If the schema asks the model to quote the source, whether it actually quoted is
mechanically checkable — one of the few quality properties that needs no judgment.
Locate every quoted span in its source document.

Write this check carefully or it will lie to you. PDF text layers inject running
headers and footers mid-sentence, and a quote that reads straight through one fails
a naive substring test with the exact signature of a spliced quote. On a real
hand-check that produced 7 false splices in 56 statements and a confident, wrong
conclusion about a prompt revision; the true count was 0. Strip running furniture
first, rejoin wrapped hyphens as `-\s+` → `-`, and report a normalization ladder
rather than one number — `exact` matched 0 of 56 genuine verbatim quotations on
those filings. `references/pitfalls.md` #10 has the full trap;
`scripts/handcheck.py` implements the check.

Diagnose failures rather than counting them. A spliced quote has a signature: its
longest matching prefix and longest matching suffix are each real runs of source
text and together account for the whole span. That separates it from a single
altered word mid-quote (prefix and suffix overlap) and from paraphrase (no matching
tail).

## 7. Validate against human annotation

Reproducibility is not validity. A model can be perfectly reproducible and
consistently wrong, and repeated passes cannot detect it — only a human reading can.

Draw a validation sample, annotate it by hand under a written codebook, and compare
field by field. The disagreement rate is the misclassification rate the downstream
estimates get corrected for.

**Stratify on the model's own labels, and carry the weights.** A simple random sample
spends nearly all its rows on the majority class. Stratifying on predicted class with
a floor per cell makes rare categories estimable — and is valid only if the inclusion
probabilities reach the analysis stage, because an unweighted analysis of a
disproportionately allocated sample is wrong and will not error. Compute `pi` at the
finest level you actually allocated at, and check that `sum(weight)` reconstructs the
population exactly. `scripts/build_validation_workbook.py` implements this;
`references/validation.md` has the four failure modes.

The one rule that makes or breaks this: **the annotation codebook and the extraction
prompt must stay separate documents.** Copying the annotator's boundary rules into
the prompt drives the measured disagreement toward zero by construction. The
temptation is strongest exactly when annotation reveals a boundary the prompt handles
badly.

**Withholding the labels is not the same as being blind.** Anything whose shape varies
with the model's output leaks it — most easily the context you provide to help the
coder. Build every row's context the same way, without consulting the model's output,
and audit by crosstabbing each derived column against the model's label: any column
that predicts it is a channel. `references/pitfalls.md` #15 has a case where a
helpfully centred context window encoded the answer for a quarter of the sample while
no label appeared anywhere in the workbook.

Expect uneven error rates across fields, and report them rather than smoothing them.
A field with a genuinely contestable category boundary will disagree more, and that
surfaces honestly as a wider bias-corrected interval. `references/validation.md`
covers the design, the codebook, and why a low error rate still has to be carried
into the second stage.

## 8. Assemble, and report reliability with the estimates

Explode the records into an item-level table and a document-level table, and
**reconcile them against each other** — item rows must equal the summed per-document
counts. That check catches a key collision, which otherwise surfaces as a quietly wrong
regression rather than an error.

Address records by a stable key, not by parsing the filename. A `{entity}_{year}` stem
breaks on the documents whose entity id is null — exactly the ones a fallback key was
introduced for — and cannot be joined back to the corpus for the metadata the tables
need anyway.

Refuse to assemble while any document still lacks an extraction. Assembling early drops
those rows silently, and the resulting table looks complete.

`scripts/assemble.py` implements this. Two things it will not do for you: decide whether
a document with no items gets NaN or 0 for its share variables (it uses NaN plus an
explicit indicator, because 0 asserts a measurement that was never made), and decide
between two constructs that are easy to conflate — the *share* of items of some kind,
and whether the document contains *any*. Both are meaningful and they are not the same
variable; this script emits both under distinct names rather than letting one silently
win.

If the extracted fields become regression variables, their measurement error
attenuates coefficients. Compute the ICC from repeated passes and state it. This
is the design's answer to "how reliable is LLM extraction", and reviewers
increasingly ask.

Document-level fields (one value per document) are typically far more stable than
item-level extraction (lists of spans). Report them separately rather than letting
one number stand for both.

## Working with the user

The decisions this skill surfaces — year ranges, deduplication, whether to keep
thinking on, whether one pass is enough — are **research-design decisions, not
implementation details**. Surface them with numbers attached and let the user
decide. Do not quietly pick the cheaper option.

Two that recur:

- **Deduplication.** Corpora often carry repeated scrapes of the same document.
  Deduplicating is usually right, but it is the user's call, and the key you
  deduplicate on determines what "one observation" means.
- **Thinking on or off.** Disabling thinking substantially improves reproducibility
  and cost. Users may still keep it on deliberately — it is the capability that
  distinguishes this from classical NLP. Present the tradeoff; respect the answer.

## Reference files

- `references/pitfalls.md` — the failure modes in detail, with the exact errors and
  fixes. Read this before writing any extraction code.
- `references/reproducibility.md` — how to measure run-to-run variance and what the
  numbers mean for downstream inference.
- `references/validation.md` — the human-annotated validation sample: codebook
  discipline, blind annotation, and correcting estimates for measured
  misclassification.
- `scripts/extract_api.py` — reference implementation of the submit/status/fetch
  pipeline.
- `scripts/prepare_corpus.py` — builds the single-file corpus from a source table.
- `scripts/handcheck.py` — renders documents beside their extractions and locates
  every quoted span in the source, diagnosing any that fail.
- `scripts/report.py` — progress, field distributions, and document-level shares.
- `scripts/assemble.py` — builds the item-level and document-level tables, reconciled
  against each other.
- `scripts/validation_sample.py` — draws a simple random annotation sample as
  markdown, with `--blind` to withhold the model's labels.
- `scripts/build_validation_workbook.py` — the stratified version: two draws, recorded
  inclusion probabilities and weights, and a protected xlsx with dropdowns.
