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
