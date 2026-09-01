# Frozen RMAD-basis supplementary instrument v1.0

This additive instrument classifies the reserve basis of each filing's
narrative RMAD conclusion as `net`, `gross`, `both`, or `unspecified`. It does
not modify the frozen primary extraction prompt.

## Frozen treatment

- Model: `claude-opus-5`
- Thinking: adaptive
- Effort: high
- Maximum output: 32,000 tokens
- Structured output: JSON Schema
- Random state: 666
- Universe: 8,532 filings
- Prompt SHA-256: `34f5c8c116cc920ecf5233584d075bf3a0c48bb4c48efed59b888ed4ded910e4`
- Canonical schema SHA-256: `c076fd08aeae48f3470ada9953f169b83dd0385d6c4d85ad267b68a0be436517`
- Ordered input-id SHA-256: `2e09400772c9a0673af931a9c5cba3434fbb8235de866d7518f3c1cf8444c2fb`

The full run used the Anthropic Messages API throughout: 8,529 responses came
through Message Batches, while three repeatedly rejected Batch-worker requests
were recovered through streaming with the identical frozen treatment. All
8,532 responses were served by `claude-opus-5` and ended with `end_turn`.

## Pilot gates

The 300-filing challenge pilot included every baseline-complete narrative-only
warning, every mechanically detected dual-basis Exhibit B layout, and
fixed-seed random fillers.

- Repeat agreement: 99.0%; Cohen's kappa 0.9838.
- Gross retention ratio: 39.877 versus 56.188 for other warning bases;
  one-sided Welch p=0.001801 in both replicates.
- Dual Exhibit B layout among `both`: odds ratio 10.255; one-sided Fisher
  p=0.03225 in both replicates.
- All pilot gates passed. Gross labels agreed exactly in both runs (99/99).

## Full-corpus checks

- Distribution: 5,779 `unspecified`, 2,049 `net`, 430 `gross`, and 274 `both`.
- Evidence was returned for all 2,753 basis-specific labels.
- Gross-basis warnings had a retention ratio 22.575 percentage points below
  other explicit warning bases; one-sided Welch p=2.25e-08.
- The full-corpus Exhibit B association remained positive (odds ratio 5.513)
  but did not meet the 5% threshold (one-sided Fisher p=0.06349). This is
  reported as directional, not a passed full-corpus significance check.
- The evidence normalization ladder source-located 7,470/8,532 quotations;
  fuzzy similarity was at least 75 for 8,526/8,532. The six lower scores were
  retained as quote-quality diagnostics rather than hand-corrected labels.

## Downstream robustness result

The net-basis encoding sets an original narrative warning to zero only when
`rmad_basis=gross`; every other original value is preserved. It reclassified
125 warnings corpus-wide and 87 observations in the 4,933-row baseline
regression sample.

| Encoding | Cell B N | Cell B mean ACDR | Difference vs A | Welch p |
| --- | ---: | ---: | ---: | ---: |
| Original | 116 | 6.355 | 2.637 | 0.0132 |
| Net basis | 36 | 12.025 | 8.305 | 0.0063 |

The four-cell contrast therefore strengthens after removing gross-only
warnings. The interaction estimate does not: the expanded interaction changes
from 6.437 (p=0.0082) to 4.620 (p=0.2572), with much lower precision because
cell B falls from 116 to 36. Both encodings must be reported; the four-cell
result and the conditional interaction answer different questions.

No independent human annotation was created. The deterministic gross-evidence
audit is prompt-derived QA and must not be represented as blind human
validation.
