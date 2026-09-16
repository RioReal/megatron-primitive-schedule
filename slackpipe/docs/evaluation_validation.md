# Evaluation Validation

SlackPipe evaluation outputs are checked by an independent result validator
before they are reported as feasible canonical results.

The validator lives in `include/slackpipe/result_validator.h` and
`src/result_validator.cc`. It does not call `EvaluateSchedule`, CP-SAT model
builders, solver helpers, or a solver. It reconstructs the schedule from the
serialized result artifacts:

- instance constants and cyclic `stage % worker` mapping
- selected logical-stage layer partition
- worker-local operation order
- serialized derived worker predecessors
- serialized operation intervals when present
- reported makespan and feasibility status

Method registry fields such as `evaluation_method_version`,
`method_contract_hash`, and `alternating_trace` are provenance. The validator
does not trust them for schedule correctness; it checks the serialized partition,
order, predecessor edges, timings, and claimed objective directly.

## Semantics Checked

For a feasible returned solution, validation checks:

- partition length, non-emptiness/min-layer constraints, and total layer count
- every operation appears exactly once in one worker-local order
- operation names and serialized ids agree on microbatch, operation position,
  phase, stage, and worker
- data dependencies along each microbatch's extended forward/backward chain
- FIFO dependencies between microbatches at every operation position
- worker-local predecessor edges derived from adjacent operations in each worker
  order, excluding data predecessors and same-position FIFO edges
- acyclicity of the combined data, FIFO, and worker-predecessor graph
- independent earliest-start reconstruction for the returned order
- serialized operation intervals, duration, worker overlap, and makespan

Communication is validated only for the implemented
`constant_inter_worker_delay` model. Same-worker data edges have zero delay;
inter-worker data edges use `communication_ticks`. Alpha-beta metadata is
rejected until an alpha-beta model is actually implemented.

## Outcome Policy

If validation fails for a feasible result, the canonical outcome is rewritten:

- `feasible = false`
- `optimal = false`
- `reported_status = "INVALID_RESULT"`
- `makespan = null`
- `result_validation_passed = false`
- `result_validation_error` carries the stable error code and message

The raw solver status is preserved in `solver_status_raw`.

`UNKNOWN` or `NOT_RUN` raw solver status can still pass validation when the
result reports a valid incumbent or deterministic fallback as feasible. A
non-feasible result may pass validation only if it does not report a makespan.

Activation-memory analysis runs after schedule validation for validated
schedules. It reconstructs retained activation lifetimes from serialized
operation timings and does not use solver variables or logs. If
activation-cap enforcement is requested and the independent analyzer finds a
cap violation, the same invalid-result policy is applied. If the cap is present
only for diagnostics, schedule feasibility is preserved and
`activation_cap_satisfied = false` marks the row as memory-infeasible.

Solver-side cap metadata is treated as a claim that still must be checked. A
solver-enforced row succeeds only when schedule validation passes, activation
analysis passes, every worker peak is within the serialized cap vector, and the
solver result says cap constraints were actually added. A proven lack of
cap-feasible schedules is reported as `INFEASIBLE`; a returned solver-enforced
schedule that violates the independent analyzer is `INVALID_RESULT` and records
the solver/analyzer disagreement.

## Offline Command

One canonical JSON result can be checked without running OR-Tools:

```bash
slackpipe_cli validate-result --input result.json
```

The command prints the structured validation JSON. It exits with code `0` when
validation passes and nonzero when validation fails or the result file cannot be
read/parsed.

Activation metrics can be recomputed from the same serialized result:

```bash
slackpipe_cli validate-result --input result.json --activation-summary
```

The summary path accepts the same activation model and cap flags as normal CLI
runs and still requires schedule validation to pass before memory analysis.
