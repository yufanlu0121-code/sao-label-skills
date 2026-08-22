---
name: sao-extract
description: Run a reproducible, homogeneous LLM extraction over a whole corpus of documents via the Anthropic Batches API, producing one structured record per document for statistical analysis. Built for the SAO project — extracting risk statements from US P&C Statement of Actuarial Opinion filings — but applies to any corpus-to-dataset job. Use this whenever the user wants to extract structured fields, codes, or annotations from many documents (filings, transcripts, reports, articles, abstracts, case notes) into a dataset, including phrases like "extract X from all these documents", "code these texts", "turn this corpus into a dataset", "run my prompt over N files", or "content analysis at scale". Also use it when someone is midway through such a job and hits truncation, JSON parse failures, cost surprises, drift between batches, or asks how reproducible their LLM-derived variables are. Especially important for research where the extracted fields become regression variables, since it covers measurement reliability (ICC), treatment homogeneity, and the silent failure modes that bias results.
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
6. Assemble, reporting reliability alongside the estimates

Work through these in order. Steps 4's numbers determine whether step 5 is worth
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

## 3. Constrain the output

Use `output_config.format` with a JSON Schema rather than asking for JSON in
prose. This eliminates parse failures structurally instead of handling them.

Nullable enums need `anyOf`, not a type list — `{"type": ["string","null"],
"enum": [...]}` is rejected. See `references/pitfalls.md` for the exact error and
fix.

## 4. Pilot before committing

Run a small pilot and check four things. Each maps to a failure mode that
otherwise contaminates the full run quietly.

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
  quarantined and not written, so the next submit retries it. Never hand-patch.

Write output **outside any synced folder**. Thousands of small files appearing over
hours will fight the sync client, and a half-synced output directory is real
corruption risk.

`scripts/extract_api.py` is a working implementation of all of this — submit /
status / fetch, with the homogeneity check, both pre-run checks, and multi-account
key handling. Adapt the schema and column names; keep the control flow.

## 6. Report reliability with the estimates

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
- `scripts/extract_api.py` — reference implementation of the submit/status/fetch
  pipeline.
- `scripts/prepare_corpus.py` — builds the single-file corpus from a source table.
