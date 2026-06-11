#!/usr/bin/env python3

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

COMPUTE_PHASES = {"forward_compute", "backward_compute"}


def parse_partition(value, pp_size, vpp_size, num_layers):
    rows = []
    for row_text in value.split(";"):
        row_text = row_text.strip()
        if not row_text:
            raise ValueError("partition contains an empty row")
        row = []
        for item in row_text.split(","):
            item = item.strip()
            if not item:
                raise ValueError("partition contains an empty value")
            try:
                layer_count = int(item)
            except ValueError as exc:
                raise ValueError(f"partition value {item!r} is not an integer") from exc
            if layer_count <= 0:
                raise ValueError("partition values must be positive integers")
            row.append(layer_count)
        rows.append(row)

    if len(rows) != pp_size:
        raise ValueError(f"partition has {len(rows)} rows, expected pp-size {pp_size}")
    if any(len(row) != vpp_size for row in rows):
        raise ValueError(f"all partition rows must have vpp-size {vpp_size} columns")
    total = sum(sum(row) for row in rows)
    if total != num_layers:
        raise ValueError(f"partition sums to {total}, expected num-layers {num_layers}")
    return rows


def matrix_to_t_split(partition, pp_size, vpp_size):
    return [partition[pp_rank][vp_rank] for vp_rank in range(vpp_size) for pp_rank in range(pp_size)]


def t_split_to_matrix(t_split, pp_size, vpp_size):
    return [
        [t_split[vp_rank * pp_size + pp_rank] for vp_rank in range(vpp_size)]
        for pp_rank in range(pp_size)
    ]


def format_partition(t_split, pp_size, vpp_size):
    matrix = t_split_to_matrix(t_split, pp_size, vpp_size)
    return ";".join(",".join(str(value) for value in row) for row in matrix)


def load_compute_events(trace_dir, iterations):
    events = []
    for iteration in iterations:
        pattern = f"rank_*_iter_{iteration}.jsonl"
        paths = sorted(Path(trace_dir).glob(pattern))
        if not paths:
            raise RuntimeError(f"No trace files matching {pattern!r} found in {trace_dir}")

        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    event = json.loads(line)
                    if event.get("phase") not in COMPUTE_PHASES:
                        continue
                    missing = {"phase", "t_start", "t_end", "logical_stage"} - set(event)
                    if missing:
                        raise RuntimeError(
                            f"{path}:{line_number} is missing required fields: {sorted(missing)}"
                        )
                    logical_stage = event["logical_stage"]
                    if logical_stage is None:
                        raise RuntimeError(f"{path}:{line_number} has logical_stage=None")
                    event["logical_stage"] = int(logical_stage)
                    event.setdefault("iteration", iteration)
                    event["duration_ms"] = (float(event["t_end"]) - float(event["t_start"])) * 1000.0
                    events.append(event)

    if not events:
        raise RuntimeError(f"No compute events found in {trace_dir}")
    return events


def infer_num_microbatches(events):
    microbatches = {
        int(event["microbatch"])
        for event in events
        if isinstance(event.get("microbatch"), int)
        or (isinstance(event.get("microbatch"), float) and event["microbatch"].is_integer())
    }
    if not microbatches:
        raise RuntimeError("Cannot infer num_microbatches from trace events.")
    return max(microbatches) + 1


def profile_by_stage(events, t_split, num_stages, iterations):
    totals = defaultdict(lambda: {"forward": 0.0, "backward": 0.0})
    for event in events:
        stage = event["logical_stage"]
        if not (0 <= stage < num_stages):
            raise RuntimeError(f"logical_stage {stage} is outside expected range [0, {num_stages})")
        key = (event["iteration"], stage)
        if event["phase"] == "forward_compute":
            totals[key]["forward"] += event["duration_ms"]
        elif event["phase"] == "backward_compute":
            totals[key]["backward"] += event["duration_ms"]

    rows = []
    iterations = list(iterations)
    for stage in range(num_stages):
        forward_values = []
        backward_values = []
        total_values = []
        for iteration in iterations:
            values = totals.get((iteration, stage))
            if values is None:
                raise RuntimeError(
                    f"Missing compute events for logical stage {stage} in iteration {iteration}."
                )
            forward_values.append(values["forward"])
            backward_values.append(values["backward"])
            total_values.append(values["forward"] + values["backward"])

        if not forward_values or not backward_values:
            raise RuntimeError(
                f"Missing forward/backward compute events for logical stage {stage}."
            )

        layers = t_split[stage]
        forward_ms = statistics.median(forward_values)
        backward_ms = statistics.median(backward_values)
        total_ms = statistics.median(total_values)
        rows.append(
            {
                "stage": stage,
                "layers": layers,
                "forward_ms": forward_ms,
                "backward_ms": backward_ms,
                "total_ms": total_ms,
                "forward_ms_per_layer": forward_ms / layers,
                "backward_ms_per_layer": backward_ms / layers,
                "total_ms_per_layer": total_ms / layers,
            }
        )
    return rows


def completion_table_makespan(num_microbatches, num_stages, pp_size, forward_costs, backward_costs):
    num_ops = 2 * num_stages
    weights = list(forward_costs) + list(reversed(backward_costs))
    table = [[math.inf for _ in range(num_ops)] for _ in range(num_microbatches)]

    for microbatch_id in range(num_microbatches):
        table[microbatch_id][0] = (microbatch_id + 1) * weights[0]

    for op_index in range(1, pp_size):
        table[0][op_index] = sum(weights[: op_index + 1])

    while True:
        progress = False
        unresolved = 0
        for microbatch_id in range(num_microbatches):
            for op_index in range(num_ops):
                if table[microbatch_id][op_index] != math.inf:
                    continue
                unresolved += 1
                if table[microbatch_id][op_index - 1] == math.inf:
                    continue
                dep_microbatch_id, dep_op_index = get_prev_b_dependency(
                    microbatch_id, op_index, num_microbatches, num_stages, pp_size
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
            raise RuntimeError("primitive schedule makespan simulator made no progress")


def get_prev_b_dependency(microbatch_id, op_index, num_microbatches, num_stages, pp_size):
    if op_index == num_stages - 1:
        if microbatch_id == 0:
            return num_microbatches - 1, op_index - pp_size
        return microbatch_id - 1, num_stages

    if op_index == num_stages:
        return microbatch_id, op_index - 1

    if microbatch_id > 0:
        return microbatch_id - 1, op_index

    dep_microbatch_id = num_microbatches - 1
    if op_index < num_stages - 1 or op_index >= num_stages + pp_size:
        dep_op_index = op_index - pp_size
    else:
        dep_op_index = 2 * num_stages - op_index - 1
    return dep_microbatch_id, dep_op_index


def stage_costs(t_split, forward_per_layer, backward_per_layer):
    forward = [layers * cost for layers, cost in zip(t_split, forward_per_layer)]
    backward = [layers * cost for layers, cost in zip(t_split, backward_per_layer)]
    total = [fwd + bwd for fwd, bwd in zip(forward, backward)]
    return forward, backward, total


def objective(t_split, forward_per_layer, backward_per_layer, num_microbatches, pp_size):
    forward, backward, total = stage_costs(t_split, forward_per_layer, backward_per_layer)
    makespan = completion_table_makespan(
        num_microbatches, len(t_split), pp_size, forward, backward
    )
    return (makespan, max(total), max(total) - min(total))


def greedy_balanced_split(num_layers, forward_per_layer, backward_per_layer):
    costs = [fwd + bwd for fwd, bwd in zip(forward_per_layer, backward_per_layer)]
    t_split = [1 for _ in costs]
    for _ in range(num_layers - len(costs)):
        best_stage = min(
            range(len(costs)),
            key=lambda stage: ((t_split[stage] + 1) * costs[stage], stage),
        )
        t_split[best_stage] += 1
    return t_split


def improve_split(initial, forward_per_layer, backward_per_layer, num_microbatches, pp_size):
    best = list(initial)
    best_score = objective(best, forward_per_layer, backward_per_layer, num_microbatches, pp_size)

    while True:
        improved = False
        candidate_best = best
        candidate_score = best_score
        for src in range(len(best)):
            if best[src] <= 1:
                continue
            for dst in range(len(best)):
                if src == dst:
                    continue
                candidate = list(best)
                candidate[src] -= 1
                candidate[dst] += 1
                score = objective(
                    candidate, forward_per_layer, backward_per_layer, num_microbatches, pp_size
                )
                if score < candidate_score:
                    candidate_best = candidate
                    candidate_score = score
                    improved = True
        if not improved:
            return best, best_score
        best = candidate_best
        best_score = candidate_score


def suggest_split(current, num_layers, forward_per_layer, backward_per_layer, num_microbatches, pp_size):
    seeds = [list(current), greedy_balanced_split(num_layers, forward_per_layer, backward_per_layer)]
    best_split = None
    best_score = None
    for seed in seeds:
        split, score = improve_split(
            seed, forward_per_layer, backward_per_layer, num_microbatches, pp_size
        )
        if best_score is None or score < best_score:
            best_split = split
            best_score = score
    return best_split, best_score


def print_profile(rows, pp_size):
    print("\nprofiling_by_logical_stage")
    print("--------------------------")
    print(
        f"{'stage':>5} {'pp':>3} {'vp':>3} {'layers':>6} "
        f"{'fwd_ms':>10} {'bwd_ms':>10} {'total_ms':>10} "
        f"{'fwd/layer':>11} {'bwd/layer':>11} {'total/layer':>12}"
    )
    for row in rows:
        stage = row["stage"]
        print(
            f"{stage:>5} {stage % pp_size:>3} {stage // pp_size:>3} {row['layers']:>6} "
            f"{row['forward_ms']:>10.3f} {row['backward_ms']:>10.3f} {row['total_ms']:>10.3f} "
            f"{row['forward_ms_per_layer']:>11.3f} "
            f"{row['backward_ms_per_layer']:>11.3f} "
            f"{row['total_ms_per_layer']:>12.3f}"
        )


def print_stage_costs(title, t_split, forward_per_layer, backward_per_layer, pp_size):
    forward, backward, total = stage_costs(t_split, forward_per_layer, backward_per_layer)
    print(f"\n{title}")
    print("-" * len(title))
    print(f"{'stage':>5} {'pp':>3} {'vp':>3} {'layers':>6} {'fwd_ms':>10} {'bwd_ms':>10} {'total_ms':>10}")
    for stage, layers in enumerate(t_split):
        print(
            f"{stage:>5} {stage % pp_size:>3} {stage // pp_size:>3} {layers:>6} "
            f"{forward[stage]:>10.3f} {backward[stage]:>10.3f} {total[stage]:>10.3f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Profile Megatron pipeline traces and suggest a virtual layer partition."
    )
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--iteration-start", type=int, required=True)
    parser.add_argument("--iteration-end", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--pp-size", type=int, required=True)
    parser.add_argument("--vpp-size", type=int, required=True)
    parser.add_argument("--virtual-pipeline-layer-partition", required=True)
    args = parser.parse_args()

    if args.iteration_start < 0 or args.iteration_end < 0:
        raise RuntimeError("iteration range must be non-negative")
    if args.iteration_start > args.iteration_end:
        raise RuntimeError("--iteration-start must be <= --iteration-end")
    if args.pp_size <= 0 or args.vpp_size <= 0:
        raise RuntimeError("--pp-size and --vpp-size must be positive")

    num_stages = args.pp_size * args.vpp_size
    if args.num_layers < num_stages:
        raise RuntimeError("num-layers must be at least pp-size * vpp-size")

    partition = parse_partition(
        args.virtual_pipeline_layer_partition, args.pp_size, args.vpp_size, args.num_layers
    )
    current_t_split = matrix_to_t_split(partition, args.pp_size, args.vpp_size)
    iterations = range(args.iteration_start, args.iteration_end + 1)
    events = load_compute_events(args.trace_dir, iterations)
    num_microbatches = infer_num_microbatches(events)
    profile_rows = profile_by_stage(events, current_t_split, num_stages, iterations)
    forward_per_layer = [row["forward_ms_per_layer"] for row in profile_rows]
    backward_per_layer = [row["backward_ms_per_layer"] for row in profile_rows]

    suggested_t_split, suggested_score = suggest_split(
        current_t_split,
        args.num_layers,
        forward_per_layer,
        backward_per_layer,
        num_microbatches,
        args.pp_size,
    )
    current_score = objective(
        current_t_split, forward_per_layer, backward_per_layer, num_microbatches, args.pp_size
    )

    print_profile(profile_rows, args.pp_size)
    print_stage_costs(
        "predicted_stage_cost_before",
        current_t_split,
        forward_per_layer,
        backward_per_layer,
        args.pp_size,
    )
    print_stage_costs(
        "predicted_stage_cost_after",
        suggested_t_split,
        forward_per_layer,
        backward_per_layer,
        args.pp_size,
    )

    print("\nsuggestion")
    print("----------")
    print(f"num_microbatches_inferred = {num_microbatches}")
    print(f"current_t_split           = {current_t_split}")
    print(f"suggested_t_split         = {suggested_t_split}")
    print(
        "suggested --virtual-pipeline-layer-partition "
        f"\"{format_partition(suggested_t_split, args.pp_size, args.vpp_size)}\""
    )
    print(f"predicted_makespan_before_ms = {current_score[0]:.3f}")
    print(f"predicted_makespan_after_ms  = {suggested_score[0]:.3f}")
    print(f"predicted_max_stage_before_ms = {current_score[1]:.3f}")
    print(f"predicted_max_stage_after_ms  = {suggested_score[1]:.3f}")


if __name__ == "__main__":
    main()
