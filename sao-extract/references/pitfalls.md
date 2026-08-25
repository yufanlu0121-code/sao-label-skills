# Failure modes

Every item here was hit in a real 8,475-document extraction. They share a property:
none raise an error. They corrupt the dataset quietly.

## 1. Truncation eats the longest documents

With thinking enabled, reasoning tokens come out of the **same `max_tokens` budget**
as the answer. A ceiling sized for the expected answer length gets consumed by
reasoning and the response truncates.

The damage is not random. Long documents produce more reasoning *and* more output,
so truncation concentrates in the longest, most content-dense documents — exactly
the ones carrying the most information. The result is missingness correlated with
document length, which is usually correlated with whatever you are studying.

**Measured example.** Corpus of 8,475 filings, median 1,145 words, max 5,808. A
ceiling of 16,000 looked generous — mean output was ~2,350 tokens. Probing the 12
longest documents:

| | |
|---|---|
| Output tokens | 4,888 – **13,802** |
| Worst case vs 16,000 ceiling | **86%** |
| Items extracted | 23–37 (corpus mean 10.8) |

Twelve documents out of 8,475 already reached 86% of the ceiling. The real tail is
longer than any small probe reaches. Raised to 32,000, worst case fell to 43%.

**What to do.** Pilot the *longest* documents specifically — sort by length and take
the top ~12, not a random sample. Confirm `stop_reason == "end_turn"`. Then set the
ceiling several times above the observed worst case; output tokens are only billed
when generated, so headroom is free.

**Enforce it permanently.** Reject any response with `stop_reason == "max_tokens"`
rather than writing it, so the document stays pending. Warn when a completed
response uses ≥90% of the ceiling — that is the early signal that headroom is
thinning as you move through the corpus.

**A large ceiling forces batches or streaming.** The SDK refuses a non-streaming
synchronous request that could exceed ten minutes:

```
Streaming is required for operations that may take longer than 10 minutes
```

The Batches API is asynchronous and exempt. Ad-hoc synchronous scripts at a large
ceiling must use `client.messages.stream(...)`.

### 1a. Truncation is not a function of document length

A length-ordered probe does **not** bound the worst case. On a full 8,472-document
run the single truncated response came from a document at the **98th percentile of
length**, while the longest document in the corpus completed comfortably. With
adaptive thinking the reasoning tokens come from the same budget and vary
independently of the input, so a mid-length document can consume the whole ceiling.

The same document, re-submitted **unchanged**, completed on the first retry using
**3,295 output tokens** against the 32,000+ it had consumed before. That is the
remedy: re-submit at the same ceiling, because the reasoning path is redrawn on every
call. Raising `max_tokens` changes the treatment signature and splits the run into
two measurements — reserve it for repeated truncation, never for one document.

### 1b. The near-limit counter must include truncated responses

If the headroom counter is tallied *after* the `max_tokens` rejection, it excludes
exactly the responses it exists to detect. A full run reported `0 response(s) at
>=90% of the ceiling` on every one of 18 batches while a document was being truncated
inside one of them. Tally before the rejection, and print the truncation count on the
same line.

## 2. Output contamination survives JSON parsing

Structured output should return bare JSON, but reasoning markers or code fences can
leak into the answer. The dangerous case is a marker **inside a string field** — it
parses as perfectly valid JSON and passes every downstream check.

Inspect the raw response body before parsing:

```python
def inspect_output(raw_text: str) -> str | None:
    stripped = raw_text.strip()
    if not stripped.startswith("{"):
        return "preamble_before_json"
    if not stripped.endswith("}"):
        return "trailing_after_json"
    lowered = stripped.lower()
    for marker in ("<thinking", "</thinking", "```"):
        if marker in lowered:
            return f"marker:{marker}"
    return None
```

**Keep the marker list narrow.** A bare `<` is not a marker. Domain text writes
"less than" as `<` constantly — `reserves < 1% of surplus` is ordinary prose, and
flagging it quarantines valid records. The structural bare-JSON check is what
catches leaked prose; the markers only catch reasoning or fences surviving inside
otherwise-valid JSON.

Verify contamination separately for each thinking configuration. A clean result
with thinking disabled says nothing about adaptive thinking.

## 3. Nullable enums are rejected in JSON Schema

The intuitive form fails:

```python
{"type": ["string", "null"], "enum": ["adverse", "favorable", None]}
```

```
400 output_config.format.schema: Invalid schema:
Enum value 'adverse' does not match declared type '['string', 'null']'
```

Use `anyOf`:

```python
def nullable_enum(*values: str) -> dict:
    return {"anyOf": [{"type": "string", "enum": list(values)}, {"type": "null"}]}
```

Nullable *non-enum* fields are fine as `{"type": ["string", "null"]}`.

## 4. "Done" cannot mean "the file exists"

An interrupted write leaves truncated JSON on disk. An existence check marks that
document complete forever and it silently drops out of the dataset.

```python
def is_done(key: str) -> bool:
    path = OUT_DIR / f"{key}.json"
    if not path.exists():
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    return True
```

This was found only because a later batch happened to re-examine the file. Assume
it will happen; N in the thousands means interruptions are certain.

## 5. Batch results come back in arbitrary order

Key results by `custom_id`. Never by position. This is easy to get right and
catastrophic to get wrong — every record silently attached to the wrong document.

## 6. A batch belongs to the account that created it

With multiple API keys, retrieving a batch with a different key returns 404. Record
which key submitted each batch and reuse it for status and fetch.

## 7. The served model can differ from the requested model

`message.model` reports what actually answered. Tally it per batch and abort the
next submit if anything differs from what was requested, or if one run has been
served by two models. A run split across model versions is two measurements
presented as one, and nothing else surfaces it.

## 8. Composite keys break on null components

`{entity_id}_{year}` fails silently for rows with a null `entity_id` — those
documents are unaddressable and never processed. Use a key guaranteed unique and
non-null across the source table, and carry the metadata in an index instead.

## 9. Cost fits extrapolate badly

Fitting output tokens on document length using only short documents underestimates
the tail. Fit across the full observed range, including the longest documents from
the truncation probe.

Input tokens are worth measuring rather than assuming: dense domain text (names,
figures, legal terminology) tokenises far worse than English prose. One corpus
measured **2.26 tokens per word** against a typical ~1.3.

## 10. Quotation checks report false splices on PDF furniture

If the schema asks the model to quote the source, verifying that the quote is
really in the source is one of the few fully mechanical checks available. Written
naively it produces alarming, entirely false results.

Source PDFs inject a running header or footer into the text layer, frequently
**mid-sentence**:

```
...and has not exhibited large numbers of late reported claims, and because this

Applied Medico-Legal Solutions, RRG
SAO 12/31/2024

Page 3
business is a relatively small portion of the Company's total experience...
```

A model that reads straight through this is behaving correctly — and if the prompt
says so explicitly ("page headers and footers injected mid-sentence may be skipped
over"), it is doing exactly what it was told. But a substring test against the raw
text fails, and the failure has the exact signature of a spliced quote: the longest
matching prefix and the longest matching suffix are each genuine runs of source
text and together account for the whole span.

On a real hand-check this produced **7 false splices in 56 statements (12.5%)** and
a confident, wrong conclusion that a prompt revision had backfired. After stripping
furniture the true count was **0 of 56**.

Two fixes, both required:

- **Strip running furniture before matching.** Identify it from the document, not
  from a hard-coded name: a short line (< 80 chars) repeating three or more times
  is running furniture whatever it says. Long lines are never removed, so a repeated
  sentence of substance survives. Hard-coding fails immediately — one filing repeats
  the insurer's name with `Page N` markers, the next repeats an auditor's footer
  (`A member firm of Ernst & Young Global Limited`) with no page numbers at all.
- **Rejoin wrapped hyphens as `-\s+` → `-`, not by closing the word up.** PDF
  wrapping splits hyphenated compounds: the source reads `non- collectability`
  where the quote reads `non-collectability`. Collapsing to `noncollectability`
  fails to match and looks like a mid-quote edit.

Report the normalization ladder (`exact` / `ws` / `punct` / `+hdr`) rather than one
number. On real filings `exact` matched **0 of 56** spans that were all genuine
verbatim quotations, so an exact-match rate says nothing about extraction quality
and everything about the PDF.

**Never act on a splice count from a checker that has not stripped page furniture.**
`scripts/handcheck.py` implements this.

## 11. A revised prompt silently constrained by a stale schema

When the instrument legitimately changes mid-project, the JSON Schema is a
*separate* structural contract and does not change with it. A revision that adds a
field, renames one, or introduces an enum value the schema does not list is
**silently voided**: `output_config.format` constrains the model to the old shape,
the run completes cleanly, and the revision has no effect on the data.

Diff the revised prompt against the schema before spending anything. Fields, enum
members, and nullability all have to line up. A revision that only sharpens judgment
rules — how to adjudicate an existing enum, how to choose quote boundaries — needs no
schema change, and confirming that is a two-minute check that prevents paying for a
run that measures nothing new.

Then treat the revision as what it is: a new instrument.

- **Snapshot the prompt before editing it.** A prompt edited in place is
  unrecoverable, and the superseded text is the audit trail for anything already
  extracted under it. On the SAO run the original prompt survived only in the sync
  provider's version history.
- **Archive, never merge.** Records extracted under the old instrument are a
  different measurement. Move them out of the output directory together with their
  batch state, with a README saying why they cannot be recombined.
- **Re-run the pilot.** All four pre-run checks belong to the instrument, not to the
  corpus, and statement counts drive output length: the SAO revision raised the
  longest document from 22 extracted statements to 38.
- **Label archived reports by the instrument that produced them**, read from that
  run's batch state — not by hashing the prompt file on disk, which now holds a
  different instrument.

## 12. Document ids that are legal locally but illegal in the API

`custom_id` on the Batches API must match `^[a-zA-Z0-9_-]{1,64}$`. A key built from
metadata will eventually violate it — a fallback to the company name yields
`United Fire Group, Inc._2023_v1`, with spaces, a comma and a period.

The failure is a 400 that rejects **the entire batch**, naming one offending request
by index:

```
requests.449.custom_id: String should match pattern '^[a-zA-Z0-9_-]{1,64}$'
```

Nothing in that batch is queued. On a run split into chunks, the chunks before it
succeed and the rest do not, so the run stops half-submitted.

Substitute a deterministic digest for offending keys, `cid_<sha256[:40]>`, and store
the reverse mapping with the batch so results are written under the real id. Keep
valid keys unchanged so existing batches are unaffected, and make the substitution
deterministic rather than positional so a resubmission produces the same id.

Note the coupling that makes this more than cosmetic: if `custom_id` doubles as the
output filename, supporting these keys means threading an id mapping through submit,
fetch, the in-flight check, and the done check. Decide early whether the transport id
and the storage id are the same thing.

## 13. A failure file that never clears

If failures are written only when a fetch produces them, a record left by an earlier
fetch survives a successful retry. The file then describes a state that no longer
exists, and any check keyed on its existence — including a stop condition — fires
forever on a stale entry.

Rewrite it against reality on every fetch: merge new failures with what is recorded,
filter to the ids that still have no valid output, and delete the file when nothing
is outstanding. Log how many records were cleared, so a retry that worked is visible.

## 14. A metadata join that fails silently removes a biased slice

Documents whose metadata comes from a join can lose it, and the filters that follow
then drop them without anyone deciding to. On a real corpus 114 eligible documents
carried no year, no entity code and no name, so a filing-year filter removed them
before any exclusion was recorded — a data-quality exclusion wearing the costume of a
design one.

It was not random. The cause was provenance: those documents were ingested without
the manifest the metadata comes from, and **65% of them belonged to a single year**,
which was consequently the thinnest year in the panel.

Three lessons:

- **Reconcile the exclusion arithmetic.** The documented drop counts did not sum to
  the observed total, and the gap was exactly the unattributable rows. A sample table
  whose rows do not add up is hiding something.
- **Look for identifiers outside the join.** The source filename carried both the
  company name and the opinion date for all 114, and a second, independent recovery
  from the document body agreed with it on every one.
- **Recovered rows are mostly duplicates, so count before running.** Of 114, only 43
  were new estimable observations: 44 duplicated documents already in the corpus
  under proper metadata, and 21 could not be matched to an entity code at all. Decide
  on the number of *new* rows, not the number of orphans.

When recovering, mark the recovered rows and make deduplication prefer the original,
or a recovered scrape can displace a document that has already been processed.

## 15. The context you show the annotator can encode the model's answer

A blind workbook withholds the model's labels. That is not sufficient. Anything whose
*shape* varies with the model's output is a channel, and the obvious place it hides is
the context you helpfully provide.

A concrete case. The workbook shows ~600 characters of narrative around the sentence
carrying a document-level conclusion, so the coder reads the author's own words. The
natural implementation centres that window on the sentence the model extracted. But
when the model returns nothing for that field, there is no sentence to centre on, and
those documents get a visibly different kind of window. On a 150-document draw the
correlation was near-total:

| context built from | model said `not_stated` | model said anything else |
|---|---|---|
| the extracted sentence | 0 | 113 |
| fallback | 37 | 0 |

A coder who notices that "this one doesn't show me a clear sentence" is being told the
model's answer. Nothing in the workbook contained a label.

**Build the context the same way for every row, without consulting the model's
output.** Locating the passage by keyword instead of by extracted sentence covered 136
of 150 documents identically across all classes. The remaining 14 fall back because the
document never mentions the subject — a property of the source text, which is what
should drive that code anyway, and not something the model told the coder.

Audit for this by crosstabbing every derived column against the model's label. Any
column that predicts it is a leak: context provenance, row height, ordering, how many
rows a document contributes, whether a field is blank. `_recovered`-style provenance
flags are usually safe; anything derived from the model's own output usually is not.

Also check what the sampling design itself reveals. Stratifying on predicted class
means *which* rows were drawn from a document is weakly informative — a filing
contributing only its one rare-category row is a hint. That one is inherent to the
design rather than fixable, and belongs in the paper's limitations.

## 16. A JSON Schema cannot express your instrument's conditional rules

`output_config.format` constrains each field independently. Prompts routinely carry
rules that relate one field to another — *this field is null unless that field takes
one of these values* — and the schema has no way to express them. A violation parses
cleanly, satisfies every structural check, and lands in the dataset.

On a real corpus one such rule ("populate `direction` only when `moment` is `level` or
`both`") was violated by **16 statements in 95,142**. Nothing in the pipeline was
looking: not the schema, not the contamination check, not the reconciliation between
tables. 0.017% is small, but the variable built from that field treats those rows as
directional when the instrument says they are not, and the error is systematic rather
than random — it concentrates in the rows where the model had a directional reading it
could not express through the field it was given.

Enumerate the conditional rules in your prompt and check them explicitly after fetch.
They are cheap to write and they are the only thing looking.

**Check the rules, not the aims.** Instruments usually mix the two, and testing an aim
as a rule buries the real violations in noise. The same prompt asked for quotes of
20–80 words *and* said to quote a longer sentence whole — so a word-band check flags
14.6% of the corpus, none of it a violation. Only rules the instrument states
unconditionally belong in a conformance check.
