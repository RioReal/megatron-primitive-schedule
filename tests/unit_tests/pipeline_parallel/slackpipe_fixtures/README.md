# Schedule Regression Fixtures

These small fixtures exercise real PP=2 runtime execution without running a
solver or depending on ignored experiment directories during ordinary tests.
The production parser validates the plans before execution.

- `m1_delayed_match.plan.json` preserves the original delayed-matching-receive
  regression. It requires RMA; do not use this order to claim general P2P progress.
- `solver_order_regressions.json` preserves four distinct solver-produced worker
  orders tested at monorepo checkpoint `5078391de5b76ad510c1faf24eab09549b560aad`.
  Each contains 64 native structured operations (B=8, N=4, W=2, L=16).

The latter is a curated runtime regression fixture, not a benchmark result or
calibrated cost profile. Only schema, dimensions, split, placement, and exact
native operation lists are retained. Unused timing/cost/provenance output and
machine-specific paths are deliberately absent. The source schedules were:

| Case | Cost model used to generate the original order | Split |
| --- | --- | --- |
| M0_equal_joint | Equal abstract costs | [1,4,7,4] |
| M2_shared_slopes_joint | Shared measured layer slopes | [1,3,7,5] |
| M3_full_joint | Measured slopes and stage-role biases | [4,4,4,4] |
| measured | Measured-cost SlackPipe predecessor schedule | [1,3,7,5] |

Tests do not claim to reproduce optimization or calibration from these snapshots.
The separate end-to-end validation generates a fresh profile and solver plan.
All four orders undergo full parameter/loss/gradient/post-SGD equivalence using
RMA, with exact trace-order checks. Missing fixture files are errors, not skips.
