# Validation: measuring the extraction's error rate

Extracted fields that become regression variables carry measurement error. Two
different quantities are involved and they are routinely conflated:

- **Reproducibility** — does the model agree with *itself* across passes? Measured
  by repeated runs; see `reproducibility.md`.
- **Validity** — does the model agree with a *careful human reading*? Measured by a
  hand-annotated validation sample, which is what this file covers.

A model can be perfectly reproducible and consistently wrong. Reliability statistics
say nothing about validity.

## The design

Draw a validation sample — 100 documents is a reasonable default — annotate it by
hand, and compare field by field against the model's labels. The disagreement rate
is the misclassification rate the downstream estimates are corrected for.

Three properties matter and each is easy to break:

**The annotation codebook and the extraction prompt must be separate documents.**
The codebook governs the human annotator; the prompt governs the model. Copying the
codebook's boundary rules into the prompt drives the measured disagreement toward
zero *by construction* and destroys the quantity being estimated. This is tempting
precisely when annotation surfaces a boundary the prompt handles badly — resist it,
or accept that the validation no longer measures anything.

**Annotate blind, and draw independently.** Label each document before seeing the
model's output for it; reading the model's labels first anchors the annotator and
inflates apparent agreement. Draw at random from the full corpus with a fixed seed,
not from the pilot, and never select on anything the model produced.

**Freeze the codebook before annotation starts.** If a case genuinely is not
covered, add a rule, date it in an amendment log, and re-check the documents already
annotated under the old wording. Silently changing the rule mid-pass reintroduces
exactly the inconsistency the codebook exists to prevent.

## Writing the codebook

Write down every boundary judgment the first time it comes up. The point is not to
anticipate every case — it is that the annotator decides each recurring boundary
*once*, and the hundredth document is labelled the same way as the first.

Rules that resolve an inconsistency are better than rules that forbid one. From the
SAO codebook:

> **Granularity**: a risk assertion and its dismissal are separate statements when
> the narrative separates them. When the actuary writes both in one sentence, keep
> them as one statement labelled `negated`.

That does not eliminate the two treatments; it makes the choice between them a
function of the source text, so it is reproducible rather than arbitrary.

## A high error rate on a field is a result

Some fields will disagree more than others, usually where a category boundary is
genuinely contestable. In the SAO run the `object` field carries a boundary between
"a statement about the input data" and "a statement about the estimate" — regulatory
ratio tests fall on the estimate side — and that field will show a higher
misclassification rate than the rest.

That is a finding, reported as a wider bias-corrected confidence interval. It is not
a defect to be engineered away *before* it is measured. Suppressing it by teaching
the model the annotator's rules removes the evidence, not the error.

## Stratifying on the model's own labels

A simple random sample spends almost all its rows on the majority class. If the point
is to estimate error rates *conditional on predicted class*, stratify on the model's
own labels and allocate a floor to each cell. That is a valid design — and valid only
if the inclusion probabilities reach the analysis stage.

**Record `pi` and `weight = 1/pi` for every sampled row.** An unweighted analysis of a
disproportionately allocated sample is simply wrong, it will not error, and nothing
downstream will flag it. Write them into the draw index that the coding is joined back
to, not only into the workbook.

Four things that are easy to get wrong, from a real 1,000-row draw over 95,142
statements:

- **Compute `pi` at the finest level you actually allocated at.** Balancing filing
  years inside each stratum makes `pi` vary within the stratum; treating it as constant
  per stratum biases every corrected estimate. Here `pi` is a (stratum × year) quantity.
- **Check the weights reconstruct the population.** `sum(weight)` over the drawn rows
  must equal the population size — 95,142 and 8,532 exactly, in that run. Any drift
  means `pi` was computed at the wrong level.
- **Do not let rare cells fall out of the estimand.** 12 of 36 cells held under 0.5% of
  statements. Allocating them nothing gives 1,156 statements a zero inclusion
  probability and silently restricts what the sample can speak for; allocating them 5
  rows each tripled the weight spread (46× → 256×, design effect 2.8 → 3.5) to estimate
  nothing. Pooling all 12 into one `RARE` stratum with a full allocation kept every
  statement in scope at an unchanged weight spread.
- **Expect a floor-driven design to be exactly its floor.** With a floor of 40 on 24
  qualifying cells, "proportional with a floor" is just the floor: proportionality only
  binds above n = 1,020 and the cap only above n = 1,358. Compute where your thresholds
  actually bite before describing the design as proportional.

Report the realised stratum counts and the weight range before coding starts. A design
that cannot be sanity-checked on one page is one whose weights will be quietly dropped
later.

## Why correction, not accuracy, is the target

The standing caution comes from Battaglia et al. on two-step estimators that use
machine-generated variables as data: in their remote-work application the classifier
was highly accurate, and most two-step estimates still fell outside the
bias-corrected confidence intervals.

The lesson is that a low error rate does not license ignoring it. Measure the rate,
carry it into the second stage, and report intervals that reflect it. An unmeasured
error rate cannot be corrected for at any level of accuracy.
