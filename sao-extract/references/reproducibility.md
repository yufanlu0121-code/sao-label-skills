# Measuring reproducibility

LLM extraction is stochastic. If the extracted fields become variables in an
analysis, that stochasticity is measurement error and it attenuates every
coefficient estimated from them. It is measurable, cheap to measure, and almost
always skipped.

## The experiment

Run the same ~20 documents **twice under identical settings**. Nothing differs
between passes, so any disagreement is pure run-to-run variance.

This is the comparison that matters, and it is easy to skip past into more
interesting-looking comparisons (model A vs B, effort level X vs Y). Without the
same-settings baseline, those comparisons are uninterpretable — you cannot tell a
real effect from noise.

**A worked example of exactly that mistake.** An extraction pilot compared four
effort levels and found overlaps of 0.64–0.67 between them, which looked like a
clear effort effect. Adding the same-settings repeat showed:

| Comparison | Overlap |
|---|---|
| **high vs high, identical settings** | **0.69** |
| high vs medium | 0.64 |
| high vs xhigh | 0.67 |
| high vs max | 0.67 |

Repeating the *same* configuration reproduced only 69% of items — indistinguishable
from comparing against a different configuration. Essentially the entire apparent
effect was noise. Run the baseline first.

## What to measure

**Item-level: Jaccard overlap.** For list-valued output (extracted spans, quotes,
codes), normalise each item to a comparable string and compute
`|A ∩ B| / |A ∪ B|` per document, then average.

**Document-level: exact agreement.** For single-valued fields, the fraction of
documents where repeated passes agree.

**Aggregates: ICC.** This is the number that matters for inference. For a count or
score `y` with repeated passes per document:

```python
within_sd  = N.std(axis=1, ddof=1).mean()   # noise
between_sd = N.mean(axis=1).std(ddof=1)     # signal
icc = between_sd**2 / (between_sd**2 + within_sd**2)
```

`icc` is the reliability of a single pass. With `k` averaged passes:

```python
reliability_k = between_sd**2 / (between_sd**2 + within_sd**2 / k)
```

Under classical measurement error, a regression coefficient on a variable with
reliability `r` is attenuated toward zero by approximately `r`. So ICC 0.90 means
roughly 10% attenuation; ICC 0.98 means ~2%.

## Interpreting the results

Expect a wide spread across output types. From a real extraction:

| Level | Reproducibility |
|---|---|
| Individual extracted spans | Jaccard 0.69 |
| Filing-level counts | ICC 0.904 |
| Composition shares | corr 0.77–0.91 |
| Single-valued document fields | 20/20 exact |

**Item churn overstates the problem.** A 0.69 span overlap sounds alarming, but the
aggregates built from those spans reached ICC 0.90 — items at the margin come and
go while the totals stay stable. Report the level you actually analyse. Letting one
number stand for both understates the reliable fields and overstates the noisy ones.

## Reducing variance

Ranked by effect, measured on the same corpus:

**Disable thinking.** Adaptive thinking takes a different reasoning path every call
and is typically the dominant variance source. Measured effect: overlap 0.69 → 0.81,
ICC 0.904 → 0.984, at equal recall (10.8 vs 10.6 items per document) and ~15% lower
cost. Verify recall is genuinely preserved before adopting it — consistency is
worthless if it comes from extracting less.

This is a research-design decision, not an optimisation. Thinking is a real
capability and users may keep it deliberately. Present the numbers; respect the
answer; report the attenuation either way.

**Average multiple passes.** Reliability rises as `k` grows, with diminishing
returns: 0.904 → 0.950 (k=2) → 0.966 (k=3). Cost scales linearly, so this is the
expensive lever. Note that a single change of configuration beat tripling passes in
the case above.

**Aggregate with a k-of-n rule.** Across 3 passes, 51% of items appeared in all
three, 21% in two, and 28% in only one. Keeping items that appear in ≥2 passes
discards the unstable tail and yields a defensible "consensus" extraction.

**Lower effort.** Often more reproducible than higher effort — the relationship is
not monotone. Do not assume more reasoning means more stable output. Above a
mid-range effort, output length can stop tracking document length entirely (R²
falling from ~0.74 to ~0.05), which is a sign reasoning has decoupled from the task.

## Reporting

Put the ICC in the methods section alongside the estimates, and state which level
it applies to. It is the design's answer to "how reliable is LLM extraction", and
a number like 0.90 is a strong answer — far stronger than not having measured.
