# Evaluation Methods

This document is the developer contract for the canonical SlackPipe evaluation
baselines. Use `slackpipe_cli describe-method METHOD` for the same contract as
structured JSON.

## Fixed Schedule Rules

The original deterministic fixed worker-local order is
`uniform-breadth-first`.

`BreadthFirstOrders(instance)` places every operation on its cyclic worker
`stage % W`, then sorts each worker by:

```text
(microbatch + operation_position, -operation_position, microbatch)
```

For `B=3, N=2, W=1`, the single worker order begins:

```text
F0(b0), F1(b0), F0(b1), B1(b0), F1(b1), F0(b2), B0(b0), B1(b1), F1(b2), ...
```

Exact non-interleaved execution would run all forward work before backward
work. The breadth-first implementation is a wavefront over operation positions,
so its canonical name remains `uniform-breadth-first`.

Legacy alias `eval-bfs` is accepted for this baseline. Outputs use
`uniform-breadth-first`.

`uniform-interleaved-1f1b` is a separate deterministic fixed-order baseline for
the same uniform partition. For each logical stage, it constructs a stage-local
1F1B stream with warmup, steady-state forward/backward alternation, and drain.
When `N > W`, the cyclic stage-to-worker mapping gives a worker multiple logical
stage streams; the worker-local order is the deterministic topological merge of
those streams. For `B=3, N=2, W=1`, the single worker order begins:

```text
F0(b0), F1(b0), B1(b0), F0(b1), B0(b0), F1(b1), B1(b1), ...
```

Aliases `eval-1f1b`, `uniform-1f1b`, and `interleaved-1f1b` are accepted.

## Uniform Partition Rule

The uniform partition is deterministic and as-even-as-possible:

1. Give every logical stage `min_layers`.
2. Divide the remaining layers evenly across stages.
3. Assign any remainder to the lowest stage indices first.

For example, `L=10, N=4, min_layers=1` produces `[3, 3, 2, 2]`.

## Canonical Methods

| Method | Fixed variables | Optimized variables | Solver |
|---|---|---|---|
| `uniform-breadth-first` | uniform partition, breadth-first worker order | none | deterministic |
| `uniform-interleaved-1f1b` | uniform partition, interleaved 1F1B worker order | none | deterministic |
| `partition-only-fixed-order` | breadth-first worker order | stage layer partition | fixed-order partition CP-SAT, or explicit enumeration backend |
| `schedule-only-uniform` | uniform partition | worker-local order via unrestricted `NoOverlap` | OR-Tools |
| `sequential-partition-then-schedule` | partition phase fixes breadth-first order; schedule phase fixes the partition found in phase 1 | partition, then order | OR-Tools for schedule phase |
| `alternating-partition-schedule` | each partition phase fixes current order; each schedule phase fixes current partition | partition and order, alternating | OR-Tools for schedule phases |
| `joint-unrestricted-no-overlap` | problem constants only | partition and worker-local order jointly | OR-Tools |

All methods share `B`, `N`, `W`, `L`, cyclic mapping, operation costs,
data/FIFO dependencies, communication ticks, seed/thread policy for solver-backed
phases, canonical result schema, independent validation, and independent
activation-memory reporting. Equal `B`, `N`, `W`, and `L` does not by itself
imply equal activation memory; worker-local order can change retained activation
lifetimes.

The fixed-order partition entry point accepts any complete valid worker-local
order and preserves it exactly. The production backend is `cpsat`: it optimizes
only the stage layer partition while adding explicit adjacent worker-order
constraints for the supplied order. Data dependencies carry the configured
communication delay, FIFO dependencies are zero-delay per operation position,
and no unconstrained worker `NoOverlap` is used in this fixed-order model.

`--fixed-order-partition-backend` accepts:

- `cpsat`: requires an OR-Tools build and never falls back to enumeration.
- `enumerate`: exact split enumeration, intended for tiny or oracle checks.
- `auto`: uses CP-SAT when available; without OR-Tools it enumerates only when
  the split count is below `--enumeration-threshold`, otherwise reports
  `UNAVAILABLE`.

Outputs record requested/effective backend, partition-count estimate,
enumeration safety threshold, whether CP-SAT launched, CP-SAT model count, raw
solver status, fallback reason/source, and selected partition/order.

Activation caps remain independently checked after every validated schedule.
For solver-backed equal-memory comparisons, post-hoc checking is diagnostic
only; the row must request solver enforcement and must record both
`activation_cap_enforced_in_solver = true` and
`activation_cap_constraints_added = true`. OR-Tools-disabled builds report
capped solver-backed rows as `UNAVAILABLE`. Builds with cumulative constraints
but without variable cumulative demands can enforce fixed demands only; optimized
`linear_in_stage_layers` partitions require variable-demand support.

## Phase Policies

`sequential-partition-then-schedule` uses one global deadline. The fixed-order
partition phase requests the `cpsat` backend by default and is capped at 50% of
the requested row budget. Unused time carries forward, and the schedule phase
receives only the remaining parent deadline.

`alternating-partition-schedule` starts from the `uniform-breadth-first`
incumbent. It repeats partition and schedule phases up to
`--alternating-max-rounds`, default `4`. At each phase start:

```text
phase_limit = remaining_global_time / remaining_planned_phases
```

The method stops after the first complete round with no strict makespan
improvement, after the maximum rounds, or when the global deadline expires. A
failed, invalid, or cap-violating phase never replaces the incumbent. Ties are
deterministic.

## Joint Semantics

`joint-unrestricted-no-overlap` uses the existing direct CP-SAT joint model. It
does not enforce predecessor-candidate windows. The existing 10% incumbent
budget policy is unchanged, and the remaining time is passed to the direct joint
model.

## Aliases

Accepted legacy aliases:

| Alias | Canonical output |
|---|---|
| `eval-bfs` | `uniform-breadth-first` |
| `eval-1f1b`, `uniform-1f1b`, `interleaved-1f1b` | `uniform-interleaved-1f1b` |
| `optimize-bfs` | `partition-only-fixed-order` |
| `partition-only` | `partition-only-fixed-order` |
| `schedule-only` | `schedule-only-uniform` |
| `schedule-only-uniform` | `schedule-only-uniform` |
| `schedule-only-partition-only` | `sequential-partition-then-schedule` |
| `joint`, `joint-cpsat`, `direct-joint`, `optimize-joint` | `joint-unrestricted-no-overlap` |

## Contract Version

`evaluation_method_version = 1` identifies these method contracts. Each
canonical result also records `method_contract_hash`, `fixed_schedule_rule`, and
`uniform_partition_rule`.

Use `slackpipe_cli build-info` to verify build provenance before production
runs. It reports git commit/dirty state, build type, OR-Tools compiled status,
OR-Tools version when available, and schema/evaluation/budget/validation
versions.

CAL evaluation sweeps should be launched through the manifest workflow in
[`evaluation_manifests.md`](evaluation_manifests.md), not by ad hoc shell loops.
The manifest records one row per method run, expected method contract hash, seed
policy, fixed-order backend, and whether the row is uncapped diagnostic or
equal-memory.

Manifest-backed result directories should then be analyzed with
[`evaluation_analysis.md`](evaluation_analysis.md). That analyzer enforces the
method contracts before aggregation, reports fallback and infeasible-cap rows in
preflight outputs, and keeps the canonical method labels in the fixed order used
by the CAL figures.

## Figure 1 Status

The deterministic baselines are computed through the implementation and
independent validator; values are not hardcoded. Do not report
`uniform-breadth-first` as interleaved 1F1B. Use
`uniform-interleaved-1f1b` only for runs whose manifest includes that canonical
method.
