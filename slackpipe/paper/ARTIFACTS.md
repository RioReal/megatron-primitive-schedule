# SlackPipe CAL Paper Artifacts

This directory contains the source of a historical CAL evaluation excerpt.
It is not a report of current Megatron runtime performance. Generated figures,
tables, and PDFs are not distributed in the monorepo.

## Required Generated Inputs

Run the manifest-backed evaluation and analysis described in
[`../EVALUATION.md`](../EVALUATION.md) and
[`../docs/evaluation_analysis.md`](../docs/evaluation_analysis.md), then supply:

- `paper/figures/fig_cal_normalized_makespan.pdf`
  - main analysis `fig_cal_normalized_makespan.pdf`
- `paper/tables/table_cal_summary.tex`
  - main analysis `table_cal_summary.tex`
- `paper/tables/table_cal_oracle_summary.tex`
  - oracle analysis `table_cal_summary.tex`

## Build

From `slackpipe/`, with those inputs and a LaTeX installation available:

```bash
make -C paper
```

This invokes `pdflatex` twice and writes `paper/main.pdf`.
