#!/usr/bin/env python3

"""Profiling-guided layer partition optimizer for the primitive pipeline schedule."""

import argparse
import math
from dataclasses import dataclass
from itertools import product
from typing import List, Optional, Sequence, Tuple, Union


Number = Union[int, float]


@dataclass(frozen=True)
class PrimitivePartitionSolution:
    t_split: List[int]
    makespan: float
    forward_weights: List[float]
    backward_weights: List[float]
    objective: Tuple[float, float, float]
    search_method: str


def _as_float_list(name: str, values: Optional[Sequence[Number]], length: int) -> List[float]:
    if values is None:
        return [0.0 for _ in range(length)]
    if len(values) != length:
        raise ValueError(f"{name} length must equal N")
    output = [float(value) for value in values]
    if any(value < 0 for value in output):
        raise ValueError(f"{name} entries must be non-negative")
    return output


def _as_int_bounds(
    name: str, value: Optional[Union[int, Sequence[int]]], length: int, default: Optional[int]
) -> List[Optional[int]]:
    if value is None:
        return [default for _ in range(length)]
    if isinstance(value, int):
        return [value for _ in range(length)]
    if len(value) != length:
        raise ValueError(f"{name} length must equal N")
    return [None if item is None else int(item) for item in value]


def _validate_optimizer_inputs(
    B: int,
    N: int,
    J: int,
    total_layers: int,
    a_fwd: Number,
    a_bwd: Number,
    bias_fwd: Sequence[Number],
    bias_bwd: Sequence[Number],
    min_layers_per_stage: Union[int, Sequence[int]],
    max_layers_per_stage: Optional[Union[int, Sequence[int]]],
):
    if B <= 0:
        raise ValueError("B must be positive")
    if N <= 0:
        raise ValueError("N must be positive")
    if J <= 0:
        raise ValueError("J must be positive")
    if N % J != 0:
        raise ValueError("N must be divisible by J")
    if total_layers <= 0:
        raise ValueError("total_layers must be positive")
    a_fwd = float(a_fwd)
    a_bwd = float(a_bwd)
    if a_fwd < 0 or a_bwd < 0:
        raise ValueError("a_fwd and a_bwd must be non-negative")

    fwd = [a_fwd for _ in range(N)]
    bwd = [a_bwd for _ in range(N)]
    fixed_forward = _as_float_list("bias_fwd", bias_fwd, N)
    fixed_backward = _as_float_list("bias_bwd", bias_bwd, N)

    mins = _as_int_bounds("min_layers_per_stage", min_layers_per_stage, N, None)
    if any(value is None or value < 0 for value in mins):
        raise ValueError("min_layers_per_stage entries must be non-negative integers")
    maxes = _as_int_bounds("max_layers_per_stage", max_layers_per_stage, N, None)
    if any(value is not None and value < 0 for value in maxes):
        raise ValueError("max_layers_per_stage entries must be non-negative integers")
    for stage, (lower, upper) in enumerate(zip(mins, maxes)):
        if upper is not None and lower > upper:
            raise ValueError(f"min layer bound exceeds max layer bound for stage {stage}")
    if sum(mins) > total_layers:
        raise ValueError("sum(min_layers_per_stage) exceeds total_layers")
    if any(upper is not None for upper in maxes):
        max_total = sum(upper if upper is not None else total_layers for upper in maxes)
        if max_total < total_layers:
            raise ValueError("sum(max_layers_per_stage) is smaller than total_layers")

    return fwd, bwd, fixed_forward, fixed_backward, mins, maxes


def primitive_weights_for_partition(
    t_split: Sequence[int],
    a_fwd: Number,
    a_bwd: Number,
    bias_fwd: Sequence[Number],
    bias_bwd: Sequence[Number],
) -> Tuple[List[float], List[float]]:
    """Return per-stage operation weights for the shared-slope + stage-bias model."""

    N = len(t_split)
    fwd_bias = _as_float_list("bias_fwd", bias_fwd, N)
    bwd_bias = _as_float_list("bias_bwd", bias_bwd, N)
    a_fwd = float(a_fwd)
    a_bwd = float(a_bwd)
    forward = [fwd_bias[stage] + a_fwd * int(t_split[stage]) for stage in range(N)]
    backward = [bwd_bias[stage] + a_bwd * int(t_split[stage]) for stage in range(N)]
    return forward, backward


def _get_prev_b_dependency(
    microbatch_id: int,
    op_index: int,
    B: int,
    N: int,
    J: int,
) -> Tuple[int, int]:
    if op_index == N - 1:
        if microbatch_id == 0:
            return B - 1, op_index - J
        return microbatch_id - 1, N

    if op_index == N:
        return microbatch_id, op_index - 1

    if microbatch_id > 0:
        return microbatch_id - 1, op_index

    dep_microbatch_id = B - 1
    if op_index < N - 1 or op_index >= N + J:
        dep_op_index = op_index - J
    else:
        dep_op_index = 2 * N - op_index - 1
    return dep_microbatch_id, dep_op_index


def primitive_profile_guided_makespan(
    B: int,
    N: int,
    J: int,
    t_split: Sequence[int],
    a_fwd: Number,
    a_bwd: Number,
    bias_fwd: Sequence[Number],
    bias_bwd: Sequence[Number],
) -> float:
    """Evaluate primitive schedule makespan for a layer partition and profiled costs."""

    if len(t_split) != N:
        raise ValueError("t_split length must equal N")
    if any(int(value) <= 0 for value in t_split):
        raise ValueError("t_split entries must be positive integers")

    forward_weights, backward_weights = primitive_weights_for_partition(
        t_split, a_fwd, a_bwd, bias_fwd, bias_bwd
    )
    # Operations are n in [0, 2N). Backward op n maps to stage 2N - n - 1.
    weights = list(forward_weights) + list(reversed(backward_weights))
    num_ops = 2 * N
    table = [[math.inf for _ in range(num_ops)] for _ in range(B)]

    for microbatch_id in range(B):
        table[microbatch_id][0] = (microbatch_id + 1) * weights[0]

    for op_index in range(1, J):
        table[0][op_index] = sum(weights[: op_index + 1])

    while True:
        progress = False
        unresolved = 0
        for microbatch_id in range(B):
            for op_index in range(num_ops):
                if table[microbatch_id][op_index] != math.inf:
                    continue
                unresolved += 1
                if not (0 <= op_index - 1 < num_ops):
                    continue
                if table[microbatch_id][op_index - 1] == math.inf:
                    continue
                dep_microbatch_id, dep_op_index = _get_prev_b_dependency(
                    microbatch_id, op_index, B, N, J
                )
                if not (0 <= dep_microbatch_id < B and 0 <= dep_op_index < num_ops):
                    raise RuntimeError(
                        "primitive dependency is out of bounds "
                        f"({dep_microbatch_id=}, {dep_op_index=})"
                    )
                if table[dep_microbatch_id][dep_op_index] == math.inf:
                    continue
                table[microbatch_id][op_index] = (
                    max(table[dep_microbatch_id][dep_op_index], table[microbatch_id][op_index - 1])
                    + weights[op_index]
                )
                progress = True

        if unresolved == 0:
            return max(max(row) for row in table)
        if not progress:
            raise RuntimeError("primitive schedule dependency resolution made no progress")


def _objective(
    B: int,
    N: int,
    J: int,
    t_split: Sequence[int],
    fwd_per_layer: Sequence[Number],
    bwd_per_layer: Sequence[Number],
    fixed_fwd: Optional[Sequence[Number]],
    fixed_bwd: Optional[Sequence[Number]],
) -> Tuple[float, float, float]:
    makespan = primitive_profile_guided_makespan(
        B,
        N,
        J,
        t_split,
        fwd_per_layer[0],
        bwd_per_layer[0],
        fixed_fwd,
        fixed_bwd,
    )
    forward, backward = primitive_weights_for_partition(
        t_split, fwd_per_layer[0], bwd_per_layer[0], fixed_fwd, fixed_bwd
    )
    stage_totals = [fwd + bwd for fwd, bwd in zip(forward, backward)]
    return makespan, max(stage_totals), max(stage_totals) - min(stage_totals)


def _composition_count(
    total_layers: int, mins: Sequence[int], maxes: Sequence[Optional[int]]
) -> int:
    remaining = total_layers - sum(mins)
    counts = [0 for _ in range(remaining + 1)]
    counts[0] = 1
    for lower, upper in zip(mins, maxes):
        extra_limit = remaining if upper is None else upper - lower
        next_counts = [0 for _ in range(remaining + 1)]
        for used, count in enumerate(counts):
            if count == 0:
                continue
            for extra in range(extra_limit + 1):
                if used + extra <= remaining:
                    next_counts[used + extra] += count
        counts = next_counts
    return counts[remaining]


def _enumerate_partitions(total_layers, mins, maxes):
    N = len(mins)
    current = [0 for _ in range(N)]

    def visit(stage, remaining):
        if stage == N:
            if remaining == 0:
                yield list(current)
            return

        min_rest = sum(mins[stage + 1 :])
        max_rest = sum(
            maxes[idx] if maxes[idx] is not None else remaining
            for idx in range(stage + 1, N)
        )
        lower = max(mins[stage], remaining - max_rest)
        upper_bound = maxes[stage] if maxes[stage] is not None else remaining
        upper = min(upper_bound, remaining - min_rest)
        for layers in range(lower, upper + 1):
            current[stage] = layers
            yield from visit(stage + 1, remaining - layers)

    yield from visit(0, total_layers)


def _solve_with_exact_enumeration(
    B,
    N,
    J,
    total_layers,
    fwd,
    bwd,
    fixed_forward,
    fixed_backward,
    mins,
    maxes,
):
    best_split = None
    best_score = None
    for candidate in _enumerate_partitions(total_layers, mins, maxes):
        score = _objective(B, N, J, candidate, fwd, bwd, fixed_forward, fixed_backward)
        if best_score is None or score < best_score:
            best_split = list(candidate)
            best_score = score
    if best_split is None or best_score is None:
        raise RuntimeError("exact enumeration found no valid primitive partition")
    return best_split, best_score, "exact_enumeration"


def _greedy_seed(total_layers, fwd_per_layer, bwd_per_layer, mins, maxes):
    costs = [float(fwd) + float(bwd) for fwd, bwd in zip(fwd_per_layer, bwd_per_layer)]
    t_split = list(mins)
    remaining = total_layers - sum(t_split)
    for _ in range(remaining):
        candidates = [
            stage
            for stage in range(len(t_split))
            if maxes[stage] is None or t_split[stage] < maxes[stage]
        ]
        if not candidates:
            raise ValueError("no valid stage can receive remaining layers")
        stage = min(candidates, key=lambda idx: ((t_split[idx] + 1) * costs[idx], idx))
        t_split[stage] += 1
    return t_split


def _scaled_int(value, scale):
    return int(round(float(value) * scale))


def _max_stage_weight_bound(total_layers, fwd, bwd, fixed_forward, fixed_backward, scale):
    max_weight = 0
    for stage in range(len(fwd)):
        max_weight = max(
            max_weight,
            _scaled_int(fixed_forward[stage], scale)
            + _scaled_int(fwd[stage], scale) * total_layers,
            _scaled_int(fixed_backward[stage], scale)
            + _scaled_int(bwd[stage], scale) * total_layers,
        )
    return max(1, max_weight)


def _solve_with_cp_sat(
    B,
    N,
    J,
    total_layers,
    fwd,
    bwd,
    fixed_forward,
    fixed_backward,
    mins,
    maxes,
):
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        return None

    scale = 1000
    model = cp_model.CpModel()
    layers = []
    for stage in range(N):
        upper = maxes[stage] if maxes[stage] is not None else total_layers
        layers.append(model.NewIntVar(mins[stage], upper, f"layers_{stage}"))
    model.Add(sum(layers) == total_layers)

    fwd_weights = []
    bwd_weights = []
    max_weight = _max_stage_weight_bound(
        total_layers, fwd, bwd, fixed_forward, fixed_backward, scale
    )
    for stage in range(N):
        fwd_weight = model.NewIntVar(0, max_weight, f"fwd_weight_{stage}")
        bwd_weight = model.NewIntVar(0, max_weight, f"bwd_weight_{stage}")
        model.Add(
            fwd_weight
            == _scaled_int(fixed_forward[stage], scale)
            + _scaled_int(fwd[stage], scale) * layers[stage]
        )
        model.Add(
            bwd_weight
            == _scaled_int(fixed_backward[stage], scale)
            + _scaled_int(bwd[stage], scale) * layers[stage]
        )
        fwd_weights.append(fwd_weight)
        bwd_weights.append(bwd_weight)

    weights = list(fwd_weights) + list(reversed(bwd_weights))
    num_ops = 2 * N
    horizon = max_weight * max(1, B * num_ops + J + 1)
    table = [
        [model.NewIntVar(0, horizon, f"T_{microbatch}_{op}") for op in range(num_ops)]
        for microbatch in range(B)
    ]

    for microbatch_id in range(B):
        model.Add(table[microbatch_id][0] == (microbatch_id + 1) * weights[0])
    for op_index in range(1, J):
        model.Add(table[0][op_index] == sum(weights[: op_index + 1]))

    for microbatch_id in range(B):
        for op_index in range(num_ops):
            if op_index == 0 or (microbatch_id == 0 and 1 <= op_index < J):
                continue
            dep_microbatch_id, dep_op_index = _get_prev_b_dependency(
                microbatch_id, op_index, B, N, J
            )
            previous = table[microbatch_id][op_index - 1]
            dependency = table[dep_microbatch_id][dep_op_index]
            max_dependency = model.NewIntVar(0, horizon, f"maxdep_{microbatch_id}_{op_index}")
            model.AddMaxEquality(max_dependency, [previous, dependency])
            model.Add(table[microbatch_id][op_index] == max_dependency + weights[op_index])

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(
        makespan, [table[microbatch_id][num_ops - 1] for microbatch_id in range(B)]
    )
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8
    solver.parameters.max_time_in_seconds = 60.0
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("OR-Tools CP-SAT could not find a primitive partition")

    t_split = [solver.Value(var) for var in layers]
    score = _objective(B, N, J, t_split, fwd, bwd, fixed_forward, fixed_backward)
    search_method = "cp_sat" if status == cp_model.OPTIMAL else "cp_sat_feasible"
    return t_split, score, search_method


def _local_search(
    seed,
    B,
    N,
    J,
    fwd_per_layer,
    bwd_per_layer,
    fixed_fwd,
    fixed_bwd,
    mins,
    maxes,
):
    best = list(seed)
    best_score = _objective(B, N, J, best, fwd_per_layer, bwd_per_layer, fixed_fwd, fixed_bwd)

    while True:
        improved = False
        candidate_best = best
        candidate_score = best_score
        for src, dst in product(range(N), repeat=2):
            if src == dst:
                continue
            if best[src] <= mins[src]:
                continue
            if maxes[dst] is not None and best[dst] >= maxes[dst]:
                continue
            candidate = list(best)
            candidate[src] -= 1
            candidate[dst] += 1
            score = _objective(
                B, N, J, candidate, fwd_per_layer, bwd_per_layer, fixed_fwd, fixed_bwd
            )
            if score < candidate_score:
                candidate_best = candidate
                candidate_score = score
                improved = True
        if not improved:
            return best, best_score
        best = candidate_best
        best_score = candidate_score


def solve_primitive_profile_guided_partition(
    B,
    N,
    J,
    total_layers,
    a_fwd,
    a_bwd,
    bias_fwd,
    bias_bwd,
    min_layers_per_stage=1,
    max_layers_per_stage=None,
):
    """Find an integer layer partition using profiled primitive operation costs.

    The dependency recurrence is the primitive/notebook recurrence. Only the
    weight model changes from ``t`` and ``2 * t`` to profiled forward/backward
    per-layer costs plus optional fixed per-stage overheads.
    """

    (
        fwd,
        bwd,
        fixed_forward,
        fixed_backward,
        mins,
        maxes,
    ) = _validate_optimizer_inputs(
        B,
        N,
        J,
        total_layers,
        a_fwd,
        a_bwd,
        bias_fwd,
        bias_bwd,
        min_layers_per_stage,
        max_layers_per_stage,
    )

    search_space = _composition_count(total_layers, mins, maxes)
    exact_enumeration_limit = 250_000
    if search_space <= exact_enumeration_limit:
        best_split, best_score, search_method = _solve_with_exact_enumeration(
            B,
            N,
            J,
            total_layers,
            fwd,
            bwd,
            fixed_forward,
            fixed_backward,
            mins,
            maxes,
        )
    else:
        cp_sat_solution = _solve_with_cp_sat(
            B,
            N,
            J,
            total_layers,
            fwd,
            bwd,
            fixed_forward,
            fixed_backward,
            mins,
            maxes,
        )
        if cp_sat_solution is not None:
            best_split, best_score, search_method = cp_sat_solution
        else:
            search_method = "greedy_local"
            seed = _greedy_seed(total_layers, fwd, bwd, mins, maxes)
            best_split, best_score = _local_search(
                seed, B, N, J, fwd, bwd, fixed_forward, fixed_backward, mins, maxes
            )

    forward_weights, backward_weights = primitive_weights_for_partition(
        best_split, fwd[0], bwd[0], fixed_forward, fixed_backward
    )
    return PrimitivePartitionSolution(
        t_split=best_split,
        makespan=best_score[0],
        forward_weights=forward_weights,
        backward_weights=backward_weights,
        objective=best_score,
        search_method=search_method,
    )


def t_split_to_megatron_vpp_string(t_split, pp_size, vpp_size):
    if pp_size <= 0 or vpp_size <= 0:
        raise ValueError("pp_size and vpp_size must be positive")
    if len(t_split) != pp_size * vpp_size:
        raise ValueError("t_split length must equal pp_size * vpp_size")
    rows = []
    for pp_rank in range(pp_size):
        row = []
        for vp_rank in range(vpp_size):
            logical_stage = vp_rank * pp_size + pp_rank
            row.append(str(int(t_split[logical_stage])))
        rows.append(",".join(row))
    return ";".join(rows)


def _parse_csv_floats(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--B", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--J", type=int, required=True)
    parser.add_argument("--total-layers", type=int, required=True)
    parser.add_argument("--a-fwd", type=float, required=True)
    parser.add_argument("--a-bwd", type=float, required=True)
    parser.add_argument("--bias-fwd", required=True)
    parser.add_argument("--bias-bwd", required=True)
    parser.add_argument("--min-layers-per-stage", type=int, default=1)
    parser.add_argument("--max-layers-per-stage", type=int, default=None)
    parser.add_argument("--pp-size", type=int, default=None)
    parser.add_argument("--vpp-size", type=int, default=None)
    args = parser.parse_args()

    solution = solve_primitive_profile_guided_partition(
        B=args.B,
        N=args.N,
        J=args.J,
        total_layers=args.total_layers,
        a_fwd=args.a_fwd,
        a_bwd=args.a_bwd,
        bias_fwd=_parse_csv_floats(args.bias_fwd),
        bias_bwd=_parse_csv_floats(args.bias_bwd),
        min_layers_per_stage=args.min_layers_per_stage,
        max_layers_per_stage=args.max_layers_per_stage,
    )

    print(f"search_method = {solution.search_method}")
    print(f"t_split = {solution.t_split}")
    print(f"makespan = {solution.makespan:.6f}")
    print(f"forward_weights = {[round(value, 6) for value in solution.forward_weights]}")
    print(f"backward_weights = {[round(value, 6) for value in solution.backward_weights]}")
    if args.pp_size is not None and args.vpp_size is not None:
        print(
            "--virtual-pipeline-layer-partition "
            f"\"{t_split_to_megatron_vpp_string(solution.t_split, args.pp_size, args.vpp_size)}\""
        )


if __name__ == "__main__":
    main()
