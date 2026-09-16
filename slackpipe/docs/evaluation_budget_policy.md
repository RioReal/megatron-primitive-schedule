# Evaluation Budget Policy

SlackPipe budget policy version `1` means one global wall-clock deadline per
reported method row. All internal work required to produce that row counts
against the same deadline: reference split construction, incumbent generation,
heuristic search, model construction, CP-SAT execution, fallback selection, and
result validation. Activation uniform-baseline derivation performed inside a
row is also recorded and counted in the row's reported total runtime.

The authoritative total runtime is measured independently with a monotonic
clock. Phase timings are diagnostic and must not be summed to reconstruct the
total.

## Phase Caps

Direct joint CP-SAT uses a 10% incumbent cap:

- incumbent limit = min(10% of requested total, current remaining time)
- unused incumbent time remains available to joint CP-SAT
- model construction begins only if the global deadline has not expired
- activation retained-interval and cumulative-resource construction is charged
  to model-build time
- CP-SAT receives only the remaining global time after model construction

Canonical SlackPipe uses the same 10% preparation cap for its reference phase.
The prepared reference incumbent is passed to the nested joint solver, so the
nested solver does not spend another independent 10% incumbent budget. The
nested solver receives only the exact remaining parent budget.

The alias `schedule-only-partition-only` is preserved for compatibility, but
canonical outputs use `sequential-partition-then-schedule`. Its online
fixed-order partition phase is capped at 50% of the total row budget, and the
fixed-split schedule solver receives all remaining time.

`alternating-partition-schedule` uses one parent deadline. At each phase start,
the phase cap is the remaining global time divided by the remaining planned
partition and schedule phases under `--alternating-max-rounds`. Unused time
carries forward. No phase starts after the global deadline expires.

## Fallback Status

If CP-SAT was launched and returns `UNKNOWN`, the raw solver status remains
`UNKNOWN`. If the global deadline expires before CP-SAT is launched, the raw
solver status is `NOT_RUN`. In both cases a row is reported as `FEASIBLE` only
when a validated incumbent or deterministic fixed-split fallback exists.

No CP-SAT model is counted as solved unless `SolveCpModel` was actually called.
If a preparatory CP-SAT model is used inside a reported method, it contributes
to `cp_sat_models_solved` separately from the joint model.

## Result Validation

Result validation runs after the solver/fallback decision and before a feasible
canonical result is reported. Its runtime is recorded as
`validation_runtime_seconds`, appended to `phase_budget.phases[]` as
`result_validation`, and included in `total_runtime_seconds`.

Validation failure does not change the raw solver status, but it does invalidate
the reported canonical outcome: `reported_status = "INVALID_RESULT"`,
`feasible = false`, and `makespan = null`.

If activation-cap enforcement is requested, a fallback incumbent is reportable
as feasible only when independent activation analysis satisfies the shared cap.
Cap-violating fallbacks are not used for objective bounds, hints, solver
fallbacks, sequential handoff, or alternating accepted state. The cap is never
enlarged to save an incumbent.

If OR-Tools or the required cumulative-demand capability is unavailable, a
solver-backed capped row reports `UNAVAILABLE` within the same row budget rather
than running uncapped optimization.

## Oracle Separation

Ordinary benchmark rows do not launch hidden `JointOptimum` solves. Oracle or
proof runs must be emitted as their own method rows with their own run id, time
limit, seed, thread count, status, bounds, and timings. Legacy gap columns may
remain present, but downstream analysis should populate oracle comparisons by
joining compatible result records.

## Runtime Tolerance

Solver limits are never set above the remaining global budget. Total wall-clock
runtime can exceed the requested budget only by teardown, serialization, and OS
scheduling overhead after the last optimization phase has stopped. OR-Tools
accepts positive floating-point second limits; SlackPipe passes the exact
positive remaining value and does not round it upward.

## Validation Build

The current developer build may have OR-Tools disabled. To run OR-Tools-enabled
validation without launching the production sweep, run:

```bash
ORTOOLS_PREFIX=/opt/or-tools scripts/validate_ortools_evaluation.sh
```

The script configures `build/ortools` with `SLACKPIPE_ENABLE_ORTOOLS=ON`, fails
clearly if `ortoolsConfig.cmake` is absent, verifies `slackpipe_cli build-info`,
runs the `slackpipe_core_ortools_smoke_tests` CTest target, and executes tiny
canonical uncapped and equal-memory CLI smoke runs for the canonical methods.
Bundled OR-Tools distributions can require explicit `absl_DIR` and
`Protobuf_DIR`; the script passes `${ORTOOLS_PREFIX}/lib/cmake/absl` and
`${ORTOOLS_PREFIX}/lib/cmake/protobuf` when present.

Manual equivalent:

```bash
cmake -S . -B build/ortools -DSLACKPIPE_ENABLE_ORTOOLS=ON -DSLACKPIPE_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/path/to/ortools -Dortools_DIR=/path/to/ortools/lib/cmake/ortools -Dabsl_DIR=/path/to/ortools/lib/cmake/absl -DProtobuf_DIR=/path/to/ortools/lib/cmake/protobuf
cmake --build build/ortools --target slackpipe_core_tests slackpipe_cli slackpipe_benchmark -j
ctest --test-dir build/ortools --output-on-failure -R '^slackpipe_core_ortools_smoke_tests$'
```
