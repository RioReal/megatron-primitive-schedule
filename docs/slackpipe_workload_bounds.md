# Workload Bounds and Gap Reporting

## Model Constraints

The current `OptimizeJointSplitAndScheduleCpSat` path in
`slackpipe/src/joint_solver_cpsat.cc` creates one integer duration variable for
each of the `2*B*N` complete F/B operations. Uniform and affine durations depend
on layer-count variables; v2 range durations use integer prefix arrays selected
by variable cuts through `AddElement`. SlackPipe's joint phase calls this same
builder. Fixed-order/BFS reference optimizers are separate paths.

Previously, the explicit worker-load bounds were inside `!HasCostProfile()`.
They now use those actual duration variables for every cost model:

```
makespan >= sum(duration[id] for id assigned to worker w)  for every w
W * makespan >= sum(duration[id] for all id)
```

No additional B factor is applied. Dependencies, FIFO, non-overlap, activation
constraints, partition restrictions and incumbent policies are unchanged. These
inequalities follow from nonnegative starts, serialized worker intervals and
every operation's completion preceding the final backward completion; they do
not exclude any feasible schedule. `workload_constraint_count` on joint results
records W+1 after these constraints are built, or zero if that path was not built.

## Independent Calculation

`ComputeWorkloadLowerBound(instance)` in `workload_bound.cc` does not inspect an
incumbent, schedule, placement load or solver response. It uses integer ticks
from the existing cost loader and computes `ceil(total_work_lower_bound / W)`
using integer quotient and remainder (no floating rounding).

* Uniform: `B * L * (forward_ticks + backward_ticks)` is invariant.
* v2 range: sum the final forward and backward prefix entries, then sum each
  stage's forward/backward role biases, and multiply by B. Contiguous ranges
  cover every layer once, so prefix differences telescope. N=1 uses the existing
  first-role convention, not first plus last. This total is partition-invariant.
* Affine: let `c_s = forward_slope_s + backward_slope_s`, `a_s` be the combined
  stage bias, and `m = min_layers`. A safe minimum per microbatch is
  `sum(m*c_s + a_s) + (L-N*m)*min(c_s)`. It is attainable over unrestricted legal
  contiguous partitions. Extra partition restrictions can only increase the
  minimum. The total is marked invariant only if all combined slopes agree or
  there are no movable layers. No incumbent work is used as a joint lower bound.
* Invalid/unsupported instances or tick overflow yield null bounds and an explicit
  unavailable reason, rather than a guessed value.

Rounding is the current C++ loader's semantics: v1 milliseconds are multiplied
by 1000; v2 microsecond prefix entries/biases are rounded individually using
`std::round`; strictly positive quantities rounded to zero are promoted to one
tick. Exactly zero stays zero. The evaluator and joint builder do **not** clamp
the final prefix difference plus bias to one. Rounded adjacent prefix entries
can therefore produce zero-duration operations. The bound follows this behavior
exactly; it does not sum rounded layer deltas or invent a post-difference clamp.
An exporter/runtime with different rounding/clamping semantics must be checked
separately before claiming equivalence with this model.

The legacy `AnalyticalGlobalLowerBound` now uses measured workload and a complete
microbatch chain lower bound for profiles instead of the unrelated abstract F/B
ratio. This is reporting/certification only, not an incumbent or search policy
change. Raw solver status and `proven_optimal` are never upgraded by reporting.

## Compatible Output

Existing `best_bound_ticks`, `best_bound`, canonical `best_objective_bound`,
`optimality_gap` and canonical `relative_optimality_gap` keep their original
raw-bound meanings. New `lower_bound_report` objects appear in joint/SlackPipe
JSON and canonical results:

| Field | Meaning |
| --- | --- |
| `raw_solver_bound_ticks` | Original solver bound, or null when unavailable in the canonical outcome |
| `total_work_lower_bound_ticks` | Invariant total work or a valid all-partitions minimum |
| `total_work_partition_invariant` | Whether this is the actual total for every allowed partition |
| `workload_bound_reason` | Calculation used, or explicit unavailability |
| `independent_workload_lower_bound_ticks` | Integer ceiling of total-work lower bound divided by W |
| `effective_lower_bound_ticks` | Maximum of usable raw bound and independent bound |
| `effective_relative_optimality_gap` | `(feasible_makespan_ticks - effective_lower_bound_ticks) / feasible_makespan_ticks` |
| `gap_formula` | Explicit denominator/formula, gap expressed as a fraction |
| `inconsistent_with_incumbent` | Flags a bound exceeding the supplied feasible objective; gap is null |

Zero/default raw bounds on timeout do not suppress an available independent
bound. No feasible returned solution, rejected results, or nonpositive makespan
give a null effective gap. A fallback keeps the underlying solver status and
existing fallback policy. The report never changes FEASIBLE to OPTIMAL.

The artifact audit additionally separates **conditional** phase bounds from
global joint bounds. `raw_bound_globally_valid=false` retains the raw bound but
excludes it from the global effective bound for fixed-partition/order,
alternating/sequential, or restricted/balance-pruned optimization. Those rows
use the independent all-partitions workload bound for their effective global
gap. Legacy raw-gap fields remain conditional and must not certify the joint
optimum. JSON emits exactly one top-level `lower_bound_report`.

## Validation Performed

All builds/execution were in Docker. Before modifying the builder, the existing
OR-enabled binary solved three B=2/N=2/W=2/L=4 unrestricted instances to proven
optimality. The updated binary returns the same optima:

| Cost model | Before optimum | After optimum | Independent workload bound |
| --- | ---: | ---: | ---: |
| Uniform F=1/B=2 | 18 | 18 | 12 |
| v1 affine profile | 42 | 42 | 30 |
| v2 range profile | 54 | 54 | 37 |

Inputs and before/after outputs are preserved under the ignored
`slackpipe/build/workload_bound_validation/` directory. No original experiment
results or profiles were edited. Regression tests cover all partitions of tiny
instances, variable slopes, loader rounding, role biases, zero durations, ceil
division, overflow, raw-field/status preservation, missing solutions, fallback,
post-validation rejection, profile constraint construction and known optima.

Actual deadline tests returned a 63-tick fallback with raw solver bound 0,
independent/effective bound 37 and effective gap 26/63, retaining FEASIBLE/NOT_RUN.
With no incumbent, the result remained NO_VALID_SOLUTION/NOT_RUN with null gap.
The solver-generated measured-cost PP=2 plan also passed Megatron numerical
equivalence: initial parameters, loss, gradients and updated parameters all had
maximum absolute difference zero.

No controlled before/after solver-speed study was performed. Correctness runs
and their wall times do not establish a speedup.

## Nemotron Data Limitation

This checkout supports `slackpipe.cost_profile.v2`, but the original 8B profile,
818.391-ms result and its source/build provenance were not found in the local
experiment/profile/plan directories. The local Nemotron fixture is not a
substitute. The original experiment's builder version therefore cannot be
identified or assumed identical to the current one.

Using only the supplied aggregates, arithmetic gives 3048.440/4 = 762.110 ms,
predicted utilization 3048.440/(4*818.391) = 93.12297%, and the stated gap formula
gives 50.95608% using 401.371 ms versus 6.87703% using 762.110 ms. These are
conditional arithmetic checks, **not an original-profile replay** or an
independent proof that its total work is invariant across repartitioning. No
uniform-cost approximation or hard-coded experiment constants enter the code.
