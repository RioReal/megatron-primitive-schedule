#!/usr/bin/env python3

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

try:
    from tools.primitive_partition_optimizer import (
        primitive_profile_guided_makespan,
        primitive_weights_for_partition,
        solve_primitive_profile_guided_partition,
        t_split_to_megatron_vpp_string,
    )
except ModuleNotFoundError:
    from primitive_partition_optimizer import (
        primitive_profile_guided_makespan,
        primitive_weights_for_partition,
        solve_primitive_profile_guided_partition,
        t_split_to_megatron_vpp_string,
    )

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
            layer_count = int(item)
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
    return [
        partition[pp_rank][vp_rank] for vp_rank in range(vpp_size) for pp_rank in range(pp_size)
    ]


def format_partition(t_split, pp_size, vpp_size):
    return t_split_to_megatron_vpp_string(t_split, pp_size, vpp_size)


def validate_partition_string_shape(value, pp_size, vpp_size):
    rows = value.split(";")
    if len(rows) != pp_size:
        raise RuntimeError("suggested partition string has the wrong number of PP rows")
    if any(len(row.split(",")) != vpp_size for row in rows):
        raise RuntimeError("suggested partition string has the wrong number of VPP columns")


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
                    if event["logical_stage"] is None:
                        raise RuntimeError(f"{path}:{line_number} has logical_stage=None")
                    event["logical_stage"] = int(event["logical_stage"])
                    event.setdefault("iteration", iteration)
                    event["duration_ms"] = (
                        float(event["t_end"]) - float(event["t_start"])
                    ) * 1000.0
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
                # Diagnostic only: observed/layer = shared slope + bias/layer + noise.
                # These ratios are not learned as stage-specific model slopes.
                "observed_fwd_ms_per_layer": forward_ms / layers,
                "observed_bwd_ms_per_layer": backward_ms / layers,
                "observed_total_ms_per_layer": total_ms / layers,
            }
        )
    return rows


def percentile(values, percentile_value):
    if not values:
        raise ValueError("cannot compute percentile of an empty list")
    if not (0.0 <= percentile_value <= 100.0):
        raise ValueError("percentile must be in [0, 100]")
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile_value / 100.0
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def fit_shared_slope_stage_bias(layers, observed_costs, estimator="min", percentile_value=20.0):
    """Fit observed_cost[s] ~= a * layers[s] + bias[s].

    The observed per-layer ratio is diagnostic only:
    observed_cost[s] / layers[s] = shared_slope + bias[s] / layers[s] + noise.
    """

    ratios = [cost / layer_count for layer_count, cost in zip(layers, observed_costs)]
    if estimator == "min":
        slope = min(ratios)
    elif estimator == "median":
        slope = statistics.median(ratios)
    elif estimator == "percentile":
        slope = percentile(ratios, percentile_value)
    else:
        raise ValueError(f"unsupported shared slope estimator {estimator!r}")
    bias = [
        max(0.0, cost - slope * layer_count)
        for layer_count, cost in zip(layers, observed_costs)
    ]
    return slope, bias


def estimate_single_run_shared_slope_and_bias(
    layers,
    costs,
    estimator="min",
    percentile_value=20.0,
):
    return fit_shared_slope_stage_bias(
        layers,
        costs,
        estimator=estimator,
        percentile_value=percentile_value,
    )


def build_cost_model(profile_rows, estimator, percentile_value):
    layers = [row["layers"] for row in profile_rows]
    forward_costs = [row["forward_ms"] for row in profile_rows]
    backward_costs = [row["backward_ms"] for row in profile_rows]
    a_fwd, bias_fwd = fit_shared_slope_stage_bias(
        layers,
        forward_costs,
        estimator=estimator,
        percentile_value=percentile_value,
    )
    a_bwd, bias_bwd = fit_shared_slope_stage_bias(
        layers,
        backward_costs,
        estimator=estimator,
        percentile_value=percentile_value,
    )
    return {
        "a_fwd": a_fwd,
        "a_bwd": a_bwd,
        "bias_fwd": bias_fwd,
        "bias_bwd": bias_bwd,
    }


def stage_costs(t_split, cost_model):
    forward, backward = primitive_weights_for_partition(
        t_split,
        cost_model["a_fwd"],
        cost_model["a_bwd"],
        cost_model["bias_fwd"],
        cost_model["bias_bwd"],
    )
    total = [fwd + bwd for fwd, bwd in zip(forward, backward)]
    return forward, backward, total


def makespan_for_split(t_split, cost_model, num_microbatches, pp_size):
    return primitive_profile_guided_makespan(
        B=num_microbatches,
        N=len(t_split),
        J=pp_size,
        t_split=t_split,
        a_fwd=cost_model["a_fwd"],
        a_bwd=cost_model["a_bwd"],
        bias_fwd=cost_model["bias_fwd"],
        bias_bwd=cost_model["bias_bwd"],
    )


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
            f"{row['forward_ms']:>10.3f} {row['backward_ms']:>10.3f} "
            f"{row['total_ms']:>10.3f} {row['observed_fwd_ms_per_layer']:>11.3f} "
            f"{row['observed_bwd_ms_per_layer']:>11.3f} "
            f"{row['observed_total_ms_per_layer']:>12.3f}"
        )


def print_cost_model(cost_model, estimator):
    print("\nfitted_cost_model")
    print("-----------------")
    print(f"shared_slope_estimator = {estimator}")
    print(f"a_fwd = {round(cost_model['a_fwd'], 6)}")
    print(f"a_bwd = {round(cost_model['a_bwd'], 6)}")
    print(f"bias_fwd = {[round(value, 6) for value in cost_model['bias_fwd']]}")
    print(f"bias_bwd = {[round(value, 6) for value in cost_model['bias_bwd']]}")


def print_stage_costs(title, t_split, cost_model, pp_size):
    forward, backward, total = stage_costs(t_split, cost_model)
    print(f"\n{title}")
    print("-" * len(title))
    print(
        f"{'stage':>5} {'pp':>3} {'vp':>3} {'layers':>6} {'fwd_ms':>10} "
        f"{'bwd_ms':>10} {'total_ms':>10} {'bias_fwd':>10} {'bias_bwd':>10}"
    )
    for stage, layers in enumerate(t_split):
        print(
            f"{stage:>5} {stage % pp_size:>3} {stage // pp_size:>3} {layers:>6} "
            f"{forward[stage]:>10.3f} {backward[stage]:>10.3f} {total[stage]:>10.3f} "
            f"{cost_model['bias_fwd'][stage]:>10.3f} {cost_model['bias_bwd'][stage]:>10.3f}"
        )


def validate_solution(solution, num_layers, pp_size, vpp_size):
    if len(solution.t_split) != pp_size * vpp_size:
        raise RuntimeError("suggested t_split has the wrong length")
    if sum(solution.t_split) != num_layers:
        raise RuntimeError("suggested t_split does not sum to num_layers")
    if any(layers < 1 for layers in solution.t_split):
        raise RuntimeError("suggested t_split contains a stage with zero layers")
    validate_partition_string_shape(
        format_partition(solution.t_split, pp_size, vpp_size), pp_size, vpp_size
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
    parser.add_argument(
        "--shared-slope-estimator",
        choices=["min", "median", "percentile"],
        default="min",
    )
    parser.add_argument("--shared-slope-percentile", type=float, default=20.0)
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
        args.virtual_pipeline_layer_partition,
        args.pp_size,
        args.vpp_size,
        args.num_layers,
    )
    current_t_split = matrix_to_t_split(partition, args.pp_size, args.vpp_size)
    iterations = range(args.iteration_start, args.iteration_end + 1)
    events = load_compute_events(args.trace_dir, iterations)
    num_microbatches = infer_num_microbatches(events)
    profile_rows = profile_by_stage(events, current_t_split, num_stages, iterations)
    cost_model = build_cost_model(
        profile_rows,
        estimator=args.shared_slope_estimator,
        percentile_value=args.shared_slope_percentile,
    )
    solution = solve_primitive_profile_guided_partition(
        B=num_microbatches,
        N=num_stages,
        J=args.pp_size,
        total_layers=args.num_layers,
        a_fwd=cost_model["a_fwd"],
        a_bwd=cost_model["a_bwd"],
        bias_fwd=cost_model["bias_fwd"],
        bias_bwd=cost_model["bias_bwd"],
    )
    validate_solution(solution, args.num_layers, args.pp_size, args.vpp_size)

    current_makespan = makespan_for_split(
        current_t_split,
        cost_model,
        num_microbatches,
        args.pp_size,
    )
    optimized_makespan = solution.makespan
    _before_forward, _before_backward, before_total = stage_costs(current_t_split, cost_model)
    _after_forward, _after_backward, after_total = stage_costs(solution.t_split, cost_model)

    print_profile(profile_rows, args.pp_size)
    print_cost_model(cost_model, args.shared_slope_estimator)
    print_stage_costs("predicted_stage_cost_before", current_t_split, cost_model, args.pp_size)
    print_stage_costs("predicted_stage_cost_after", solution.t_split, cost_model, args.pp_size)

    print("\nsuggestion")
    print("----------")
    print(f"num_microbatches_inferred = {num_microbatches}")
    print(f"search_method             = {solution.search_method}")
    print(f"shared_slope_estimator    = {args.shared_slope_estimator}")
    print(f"a_fwd                     = {cost_model['a_fwd']:.6f}")
    print(f"a_bwd                     = {cost_model['a_bwd']:.6f}")
    print(f"bias_fwd                  = {[round(value, 6) for value in cost_model['bias_fwd']]}")
    print(f"bias_bwd                  = {[round(value, 6) for value in cost_model['bias_bwd']]}")
    print(f"current_t_split           = {current_t_split}")
    print(f"suggested_t_split         = {solution.t_split}")
    print(
        "suggested --virtual-pipeline-layer-partition "
        f"\"{format_partition(solution.t_split, args.pp_size, args.vpp_size)}\""
    )
    print(f"predicted_makespan_before_ms = {current_makespan:.3f}")
    print(f"predicted_makespan_after_ms  = {optimized_makespan:.3f}")
    print(f"predicted_max_stage_before_ms = {max(before_total):.3f}")
    print(f"predicted_max_stage_after_ms  = {max(after_total):.3f}")


if __name__ == "__main__":
    main()
