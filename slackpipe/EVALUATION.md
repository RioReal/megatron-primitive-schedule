# SlackPipe CAL Evaluation Protocol

This document freezes the evaluation protocol for the SlackPipe CAL paper
experiments. Production results must be generated from manifest-backed runs and
analyzed by the checked-in analyzer, not by ad hoc shell loops.

## Research Questions

1. Does joint partition-and-schedule optimization reduce makespan relative to a
   deterministic uniform baseline?
2. How much does joint optimization improve over partition-only,
   schedule-only, sequential partition-then-schedule, and alternating
   partition/schedule baselines?
3. Do equal-memory comparisons remain feasible when retained activation memory
   is capped at the uniform-baseline level?
4. How sensitive are the results to controlled communication delay?
5. On tiny oracle profiles, do canonical methods match proven optima when those
   optima are actually proven?

## Canonical Methods

The original main profile uses six canonical CAL methods:

- `uniform-breadth-first`: fixed uniform partition and fixed breadth-first
  worker-local order.
- `partition-only-fixed-order`: optimizes the stage layer partition while
  fixing the breadth-first worker-local order.
- `schedule-only-uniform`: fixes the uniform partition and optimizes
  worker-local order with unrestricted per-worker `NoOverlap`.
- `sequential-partition-then-schedule`: optimizes the partition under the fixed
  order, then fixes that partition and optimizes the schedule.
- `alternating-partition-schedule`: alternates fixed-order partition refinement
  and fixed-split schedule refinement under one global row deadline.
- `joint-unrestricted-no-overlap`: jointly optimizes partition and schedule
  with unrestricted per-worker `NoOverlap`.

The opt-in `main_big_5min_1f1b` profile adds
`uniform-interleaved-1f1b`, a deterministic uniform-partition baseline with
stage-local 1F1B streams merged under the cyclic stage-to-worker mapping. It
does not rename or replace `uniform-breadth-first`.

Oracle profile rows are separate `partition-only-fixed-order` rows using the
`enumerate` backend on tiny instances. They are not hidden objectives inside
baseline runs. No wrapper, no-slack, no-hint, or no-upper-bound ablation method
is part of the canonical main matrix unless it has a checked-in method contract
and manifest rows.

## Terminology Corrections

- The original fixed baseline is breadth-first wavefront order. Use
  `uniform-interleaved-1f1b` only for manifests that include that method.
- Direct joint uses unrestricted `NoOverlap`, not a predecessor-window
  restriction.
- Communication is a constant inter-worker delay in integer ticks, not an
  alpha-beta payload model.
- Activation units are controlled evaluation units, not physical bytes unless
  `activation_bytes_per_unit` is explicitly set.

## Main Matrix

The production main profile uses:

- `W in {2, 4, 8}`
- `stage_multiplier in {1, 2, 4}`
- `N = W * stage_multiplier`
- `B/N in {0.5, 1, 2}` with `B = int(N * ratio)`
- `L = 8N`
- cyclic logical-stage-to-worker mapping, `stage % W`
- forward layer cost `1`
- backward layer cost `2`
- communication profiles:
  - `none = 0` ticks
  - `moderate = 2` ticks
  - `heavy = 8` ticks
- activation modes:
  - uncapped diagnostic
  - equal-memory with materialized uniform-baseline caps
- deterministic baseline seed: one row
- solver-backed seeds: `0,1,2`
- row time limit: 300 seconds for production main rows
- solver threads: 1

The main row count is:

```text
81 configurations * 2 activation modes * (1 deterministic + 5 solver-backed * 3 seeds) = 2592
```

The opt-in `main_big_5min_1f1b` profile uses the same matrix, adds the 1F1B
deterministic baseline, defaults to a 300-second row budget and 8 solver
threads, includes both uncapped and equal-memory rows, and declares:

```text
81 configurations * 2 activation modes * (2 deterministic + 5 solver-backed * 3 seeds) = 2754
```

The opt-in `main_big_5min_1f1b_uncapped` profile uses the same matrix, methods,
300-second row budget, 8 solver threads, and `cpsat` fixed-order partition
backend, but includes uncapped rows only:

```text
81 configurations * (2 deterministic + 5 solver-backed * 3 seeds) = 1377
```

The analyzer can additionally emit
`fig_slackpipe_utilization_vs_batch.pdf` for uncapped 1F1B big-profile results.
This figure reports simulated compute utilization, not hardware GPU
utilization:

```text
100 * B * L * (forward_cost_per_layer + backward_cost_per_layer) / (W * makespan)
```

With the controlled costs, this is `100 * 3 * B * L / (W * makespan)`.
Communication and pipeline bubbles reduce utilization only through the makespan
denominator. The default paper-style utilization figure filters to `N/W = 2`,
uses `B/W` as the x-axis, and shows moderate and heavy communication panels.

## Ablation And Oracle Profiles

Ablation profiles may include only truthful, implemented, contract-hashed
method switches. Do not add label-only variants.

Oracle profile rows are separate tiny rows. An oracle result contributes an
optimum only when its solver status proves optimality. Do not report a hidden
`JointOptimum` solve.

## Fairness Policy

- Every row fixes `B`, `N`, `W`, `L`, mapping, cost ratios, communication,
  activation model, activation cap fields, seed, thread count, and time limit in
  the manifest.
- Every solver-backed row receives one global deadline.
- `budget_policy_version = 1` identifies the fair time-budget rules.
- Deterministic methods are not duplicated by seed; solver-backed methods are.
- Fallback use is serialized and counted by analysis.
- Feasible rows must pass independent validation.
- Result rows must match the manifest hash and run id.
- Result rows must match the expected git commit policy and method contract
  hash.
- The runner refuses to append into a result root with a different manifest hash
  unless `--force-new-manifest` is explicitly supplied.
- The runner warns and records resource metadata when `jobs * solver_threads`
  exceeds available cores.

## Activation-Memory Policy

The default activation model is `linear_in_stage_layers` with
`activation_units_per_layer = 1`.

Equal-memory caps are derived once per configuration from the deterministic
uniform-breadth-first baseline. The manifest materializes the cap vector and
derivation hash before any method row is run.

Equal-memory aggregation eligibility requires:

- `activation_cap_satisfied = true`
- deterministic baseline rows use `deterministic_postconstruction_check`
- solver-backed rows use solver-side cap enforcement
- solver-backed rows record `activation_cap_enforced_in_solver = true`
- solver-backed rows record `activation_cap_constraints_added = true`

`posthoc_only` rows are diagnostics only and must not enter equal-memory
makespan comparisons.

## Analysis Policy

Use `scripts/analyze_cal_results.py`.

- Preflight checks must pass before strict aggregation.
- Protocol guards reject mixed manifest hashes, git commits, method contract
  hashes, budget policy versions, evaluation method versions, validation
  versions, activation analysis versions, and activation cap formulation
  versions.
- Normalized makespan is `method_makespan / uniform_breadth_first_makespan` for
  the same workload key.
- Seed aggregation is median within workload/method.
- Paper-level aggregation is geometric mean across workload-level seed medians.
- The uniform baseline is a reference line, not a primary method bar.
- Oracle gaps are reported only against proven oracle optima.
- Convergence figures require serialized incumbent trace data; final results
  are not converted into synthetic curves.
- Missing, incompatible, unavailable, validation-failed, fallback, and
  cap-ineligible rows are reported in preflight outputs.

## Reproduction Commands

Smoke commands are safe for rehearsal:

```bash
cmake -S . -B /tmp/slackpipe_cal/build/no-or \
  -DSLACKPIPE_ENABLE_ORTOOLS=OFF \
  -DSLACKPIPE_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/slackpipe_cal/build/no-or \
  --target slackpipe_core_tests slackpipe_cli slackpipe_benchmark -j2
ctest --test-dir /tmp/slackpipe_cal/build/no-or --output-on-failure

BUILD_DIR=/tmp/slackpipe_cal/build/ortools \
ORTOOLS_PREFIX=/opt/or-tools \
BUILD_JOBS=2 \
scripts/validate_ortools_evaluation.sh

python3 scripts/generate_cal_manifest.py \
  --profile smoke \
  --output /tmp/slackpipe_cal/manifests/cal_smoke.jsonl \
  --binary /tmp/slackpipe_cal/build/ortools/slackpipe_cli \
  --time-limit-seconds 5 \
  --solver-threads 1 \
  --seed-list 0

python3 scripts/run_cal_manifest.py \
  --manifest /tmp/slackpipe_cal/manifests/cal_smoke.jsonl \
  --output-root /tmp/slackpipe_cal/results \
  --binary /tmp/slackpipe_cal/build/ortools/slackpipe_cli \
  --jobs 1 \
  --dry-run

python3 scripts/run_cal_manifest.py \
  --manifest /tmp/slackpipe_cal/manifests/cal_smoke.jsonl \
  --output-root /tmp/slackpipe_cal/results \
  --binary /tmp/slackpipe_cal/build/ortools/slackpipe_cli \
  --jobs 1 \
  --fail-fast

python3 scripts/analyze_cal_results.py \
  --manifest /tmp/slackpipe_cal/manifests/cal_smoke.jsonl \
  --results-root /tmp/slackpipe_cal/results \
  --output-dir /tmp/slackpipe_cal/analysis \
  --mode both \
  --include-groups smoke \
  --strict \
  --format pdf,png,csv,json,tex
```

Production commands are prepared but must not be run by default:

```bash
python3 scripts/generate_cal_manifest.py \
  --profile main \
  --output manifests/cal_main.jsonl \
  --binary build/ortools/slackpipe_cli \
  --time-limit-seconds 300 \
  --solver-threads 1 \
  --seed-list 0,1,2

python3 scripts/run_cal_manifest.py \
  --manifest manifests/cal_main.jsonl \
  --output-root results/cal_main \
  --binary build/ortools/slackpipe_cli \
  --jobs 1

python3 scripts/analyze_cal_results.py \
  --manifest manifests/cal_main.jsonl \
  --results-root results/cal_main \
  --output-dir results/cal_main_analysis \
  --mode both \
  --include-groups main \
  --strict \
  --format pdf,png,csv,json,tex
```

## Output Structure

Recommended layout:

```text
<root>/
  manifests/
    cal_smoke.jsonl
    cal_smoke.jsonl.metadata.json
  results/
    .slackpipe_cal_manifest.json
    smoke/<configuration>/<activation_mode>/<method>/<repetition>/result.json
  logs/
    configure_no_or.log
    build_no_or.log
    ctest_no_or.log
    validate_ortools_evaluation.log
    smoke_dry_run_full.json
    smoke_run.log
    analyze_smoke.log
  validation/
    <method>.validation.txt
    representative_summary.json
  analysis/
    preflight_cal_rows.csv
    fig_cal_normalized_makespan.pdf
    table_cal_summary.tex
    cal_eval_summary.md
```

Figures and paper tables should be copied from `analysis/` only after strict
preflight passes.

## Known Limitations

- This is simulator evaluation, not physical cluster measurement.
- Layer costs are homogeneous and controlled.
- Communication is constant inter-worker delay; no network contention is
  modeled.
- No recomputation is modeled.
- Activation memory excludes parameters, gradients, optimizer state,
  workspaces, communication buffers, allocator fragmentation, and temporary
  tensors.
- The CAL page budget means not every ablation belongs in the main paper.
