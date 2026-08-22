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
JSON parsing, treatment drift across batches, and the measurement reliability
(ICC) of variables derived this way.

Built for the SAO project — extracting risk statements from 8,475 US P&C
Statement of Actuarial Opinion filings — and the numbers quoted in the reference
files are real measurements from that run.

### Contents

| Path | What it is |
|---|---|
| `SKILL.md` | The six-step pipeline and which decisions belong to the researcher |
| `references/pitfalls.md` | Nine silent failure modes, with the exact errors and fixes |
| `references/reproducibility.md` | Measuring run-to-run variance; what ICC means for inference |
| `scripts/` | Working submit/status/fetch pipeline, corpus builder, reporting |
