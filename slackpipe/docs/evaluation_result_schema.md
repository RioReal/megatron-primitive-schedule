# Evaluation Result Schema

SlackPipe evaluation outputs carry a canonical JSON object named
`canonical_result`. The top-level `schema_version`, `canonical_method`, and
`actual_solver_path` fields mirror the same values for quick filtering. The
top-level `budget_policy_version` mirrors the canonical value.

Schema version `1` records what was requested, what actually ran, which
variables were fixed or optimized, active restrictions, fallback behavior,
solution choices, and provenance. Existing legacy fields are retained for
compatibility. Additive optional fields do not require a schema-version bump;
budget-policy compatibility is tracked separately by `budget_policy_version`.

`budget_policy_version = 1` means one global method deadline with all internal
optimization charged to the reported row. `evaluation_method_version = 1`
identifies the canonical method contracts described in
[`evaluation_methods.md`](evaluation_methods.md). See
[`evaluation_budget_policy.md`](evaluation_budget_policy.md).

Feasible canonical results are validated by the independent checker described in
[`evaluation_validation.md`](evaluation_validation.md). Validation metadata is
serialized inside `canonical_result.result_validation`; the legacy
`result_validation_passed` and `result_validation_error` fields mirror its
top-level pass/fail status for simple filters.

Canonical results also include independent retained-activation memory fields.
See [`evaluation_activation_memory.md`](evaluation_activation_memory.md) for the
controlled models, half-open lifetime semantics, cap modes, and limitations.

When rows are executed through `scripts/run_cal_manifest.py`, the runner adds
top-level manifest/provenance fields such as `manifest_id`, `manifest_hash`,
`manifest_run_id`, `runner_command`, logs, timestamps, exit code, hostname, and
`runner_outcome`. These wrapper fields do not modify the nested
`canonical_result`; analysis preflight should validate both the wrapper and the
canonical object. See [`evaluation_manifests.md`](evaluation_manifests.md).

## Canonical Methods

| Canonical method | Fixed variables | Optimized variables |
|---|---|---|
| `uniform-breadth-first` | uniform partition, breadth-first worker order | none |
| `uniform-interleaved-1f1b` | uniform partition, interleaved 1F1B worker order | none |
| `partition-only-fixed-order` | breadth-first worker order | stage layer partition |
| `schedule-only-uniform` | uniform partition | worker order through unrestricted CP-SAT `NoOverlap` |
| `sequential-partition-then-schedule` | partition phase fixes breadth-first order; schedule phase fixes phase-1 partition | stage layer partition, then worker order |
| `alternating-partition-schedule` | current order in partition phases; current partition in schedule phases | stage layer partition and worker order across alternating phases |
| `joint-unrestricted-no-overlap` | problem size, costs, dependencies, cyclic mapping | stage layer partition and worker order through unrestricted CP-SAT `NoOverlap` |
| `canonical-slackpipe-fixed-split` | SlackPipe reference partition | worker order through unrestricted CP-SAT `NoOverlap` |
| `canonical-slackpipe-stage-local` | stage-local neighborhood around the reference partition | partition inside the stage-local budget and worker order |
| `canonical-slackpipe-global` | only base problem constants | partition and worker order; its incumbent/budget policy is SlackPipe-specific |
| `canonical-slackpipe-worker-aggregate-fixed` | worker aggregate layer totals from the reference partition | redistribution among stages on each worker and worker order |
| `canonical-slackpipe-worker-aggregate-local` | worker aggregate layer-load neighborhood | partition inside the worker-aggregate budget and worker order |

Legacy CLI and benchmark aliases remain accepted, but outputs use the canonical
method names above. A method is not labeled predecessor-restricted unless the
optimization model actually enforces such a restriction.

## Decision Fields

`partition_decision` is one of:

- `fixed_uniform`
- `fixed_supplied`
- `optimized_global`
- `optimized_global_then_fixed`
- `alternating_fixed_order_partition`
- `optimized_stage_local`
- `optimized_worker_aggregate_fixed`
- `optimized_worker_aggregate_local`
- `not_applicable`

`schedule_decision` is one of:

- `fixed_breadth_first`
- `fixed_canonical_interleaved_1f1b`
- `alternating_fixed_split_no_overlap`
- `optimized_no_overlap`
- `not_applicable`

The current C++ implementation serializes fixed BFS schedules as
`fixed_breadth_first`. The CP-SAT joint and schedule-only paths serialize
`optimized_no_overlap`.

## Communication Model

The current implementation records:

- `communication_model = "constant_inter_worker_delay"`
- `communication_ticks = <active scalar delay>`
- `communication_alpha = null`
- `communication_beta = null`
- `communication_payload = null`

This schema intentionally does not claim alpha-beta behavior until that model
exists in the solver.

## Predecessor Restrictions

Production joint CP-SAT uses unrestricted `NoOverlap` per worker. The
`PredNCandidates` helper exists, but it is not an active optimization-model
restriction.

When a predecessor-candidate option is requested, outputs record:

- `predecessor_candidate_restriction_requested = true`
- `predecessor_candidate_restriction_active = false`
- `predecessor_candidate_rule = "unrestricted_no_overlap"`

The standalone CLI also emits a visible warning for this requested-but-inactive
case.

## Fallback Semantics

`solver_status_raw` records the raw CP-SAT status. `reported_status` records the
status returned to legacy callers. If CP-SAT reports `UNKNOWN` and SlackPipe
returns an incumbent:

- `solver_status_raw = "UNKNOWN"`
- `reported_status = "FEASIBLE"`
- `fallback_used = true`
- `returned_solution_source = "incumbent"`
- `optimal = false`
- `best_objective_bound` remains the solver bound when available

`fallback_reason` explains the fallback path when available.

Fixed-order partition rows additionally record:

- `fixed_order_partition_backend_requested`
- `fixed_order_partition_backend_effective`
- `estimated_partition_count`
- `enumeration_safety_threshold`
- `cp_sat_launched`
- `cp_sat_models_solved`

The CP-SAT backend returns a validated incumbent only for `UNKNOWN` or
pre-launch deadline expiry, and records that as a fallback. Explicit `cpsat`
does not silently enumerate when OR-Tools is unavailable.

If CP-SAT is not launched because the global deadline expired:

- `solver_status_raw = "NOT_RUN"`
- `reported_status = "FEASIBLE"` only when a validated fallback exists
- `fallback_used = true` only for that validated fallback
- `fallback_reason = "global_deadline_expired_before_cp_sat"` or
  `"global_deadline_expired_before_joint_cp_sat"`
- `cp_sat_models_solved` does not count the skipped CP-SAT model

## Budget Fields

Version `1` adds:

- `budget_policy_version`
- `evaluation_method_version`
- `fixed_schedule_rule`
- `uniform_partition_rule`
- `method_contract_hash`
- `time_to_first_feasible_seconds`
- `time_to_best_solution_seconds`
- `validation_runtime_seconds`
- `phase_budget.reference_phase_limit_seconds`
- `phase_budget.schedule_solver_effective_limit_seconds`
- `phase_budget.phases[]`

Each phase entry may include requested/effective limits, remaining time before
and after the phase, runtime, `expired_before_start`, and status. Unavailable
phase values are `null`; they are not serialized as fabricated zeroes.

For canonical SlackPipe, `reference_runtime_seconds` is the preparation phase
that constructs the reference split and schedule. It is disjoint from the
nested joint model-build and solver runtimes.

Validation is appended as a `result_validation` phase and is included in
`total_runtime_seconds`.

Activation uniform-baseline derivation, when performed inside a row, is
serialized as an `activation_uniform_baseline_analysis` phase and is included in
`total_runtime_seconds`.

Alternating and sequential-style rows may also record
`alternating_max_rounds`, `alternating_completed_rounds`,
`alternating_convergence_reason`, `alternating_trace`, and
`intermediate_partition_only_makespan`.

## Validation Fields

`result_validation` records the independent validation result:

- `validation_version`
- `passed`
- `error_code`, `error_category`, and `message`
- offending operations, worker, and edge type when available
- expected/actual ticks or counts when available
- reconstructed, serialized, and reported makespans
- operation and edge inventory counts
- cycle witness and warnings
- `validation_runtime_seconds`

If validation fails for a feasible returned solution, the canonical outcome is
rewritten to `reported_status = "INVALID_RESULT"`, `feasible = false`, and
`makespan = null`. `solver_status_raw` is preserved for provenance.

If activation-cap enforcement is requested and independent analysis finds a
violation, the canonical outcome is rewritten the same way with
`result_validation_error` beginning `activation_cap_violation`. If a cap is
configured only for diagnostics, cap violations are reported in
`activation_cap_satisfied` and do not invalidate schedule feasibility.

## Activation Cap Fields

Activation analysis is always independent of solver variables, but solver-backed
methods may additionally constrain CP-SAT with retained-activation cumulative
resources. Canonical results record:

- `activation_cap_enforcement_requested`
- `activation_cap_enforcement_mode`: `none`, `posthoc_only`, `solver`,
  `unsupported`, or `deterministic_postconstruction_check`
- `activation_cap_solver_supported`
- `activation_cap_solver_support_level`
- `activation_cap_enforced_in_solver`
- `activation_cap_constraints_added`
- `activation_retained_interval_count`
- `activation_cumulative_constraint_count`
- `activation_variable_demand_count`
- `activation_fixed_demand_count`
- `activation_workers_with_constraints`
- `activation_constraint_build_runtime_seconds`
- `incumbent_rejected_for_activation_cap`
- `activation_model_validation_agreement`
- `activation_model_disagreement_details`
- `activation_cap_unsupported_reason`
- `activation_cap_formulation_version`

For alternating and sequential rows, each trace entry also records whether the
phase requested cap enforcement, whether the phase was supported, whether
constraints were added, whether the candidate satisfied the cap, and whether a
candidate was rejected for cap violation.

## Derived Worker Predecessors

`derived_worker_predecessors` is derived deterministically from
`worker_local_operation_order`. For each worker, adjacent operations become a
worker-predecessor edge when they have different operation positions and the
edge is not already the operation's data predecessor. Each edge stores numeric
operation ids and stable operation names:

```json
{"operation_id": 7, "operation": "B0(b1)", "predecessor_id": 4, "predecessor": "F0(b1)", "worker": 0}
```

## Null Values

Unavailable values are serialized as JSON `null`. The literal string
`"unknown"` is not used for git provenance. `git_dirty_scope` records whether
the dirty-tree value came from build-time CMake provenance or from a runtime
environment override.

Benchmark CSV output flattens the most important canonical fields. Nested
solution structures, including the derived predecessor map, are JSON-only.
