# Codebook — human annotation of the validation sample

**Version 1, YYYY-MM-DD.** Governs the manual annotation of the N-document
validation sample. Amendments are logged at the foot of this file.

## What this is, and what it is not

This codebook is the instrument for the **human** annotator. The extraction prompt
is the instrument for the **model**. They are deliberately separate documents and
must stay that way.

The purpose of the validation sample is to measure the rate at which the model's
labels differ from a careful human reading. Copying these boundary rules into the
prompt would drive that measured rate toward zero by construction and destroy the
quantity being estimated. The disagreement is the measurement, not a defect to be
engineered away in advance.

Misclassification rates will not be uniform across fields. A field with a genuinely
contestable category boundary will show a higher rate, and that shows up honestly as
a wider bias-corrected confidence interval. Driving the error rate to zero is not the
goal; measuring it and correcting for it is.

## Procedure

- **Annotate blind.** Label each document before looking at the model's extraction
  for it. Reading the model's labels first anchors the annotator and inflates
  apparent agreement.
- **Draw the sample independently of the model's output** — at random from the full
  corpus with a fixed seed. Do not draw from the pilot documents, and do not select
  on anything the model produced.
- **Freeze this file before annotation starts.** If a case genuinely is not covered,
  add a rule, date it in the amendment log, and re-check the documents already
  annotated under the old wording. Silently changing the rule mid-pass reintroduces
  exactly the inconsistency this document exists to prevent.

## Codebook — boundary rules

One rule per recurring boundary, written the first time it comes up. State the
disposition first, then the reasoning, then the reserved case.

Prefer rules that make a choice a function of the source text over rules that
forbid one. Worked example, from the SAO project:

> **Granularity**: a risk assertion and its dismissal are separate statements when
> the narrative separates them. When the actuary writes both in one sentence, keep
> them as one statement labelled `negated`.

That does not eliminate the two treatments — it makes choosing between them
reproducible rather than arbitrary.

## Amendment log

| Date | Rule | Change |
|---|---|---|
| YYYY-MM-DD | — | Version 1. |
