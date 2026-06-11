#!/usr/bin/env python3

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

COMPUTE_PHASES = {"forward_compute", "backward_compute"}


def load_compute_events(trace_dir: str, iterations):
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
                    missing = {"phase", "t_start", "t_end"} - set(event)
                    if missing:
                        raise RuntimeError(
                            f"{path}:{line_number} is missing required fields: {sorted(missing)}"
                        )
                    event.setdefault("iteration", iteration)
                    event["duration"] = float(event["t_end"]) - float(event["t_start"])
                    events.append(event)

    if not events:
        raise RuntimeError(f"No compute events found in {trace_dir}")
    return events


def add_duration(accumulator, key, event):
    phase = event["phase"]
    if phase == "forward_compute":
        accumulator[key]["forward"] += event["duration"]
    elif phase == "backward_compute":
        accumulator[key]["backward"] += event["duration"]


def normalize_group_key(key):
    if key is None:
        return "unknown"
    if isinstance(key, tuple):
        return tuple(normalize_group_key(part) for part in key)
    return key


def format_key(key):
    if isinstance(key, tuple):
        return ", ".join(str(part) for part in key)
    return str(key)


def summarize(events, key_func):
    accumulator = defaultdict(lambda: {"forward": 0.0, "backward": 0.0})
    for event in events:
        add_duration(accumulator, normalize_group_key(key_func(event)), event)

    rows = []
    for key, totals in sorted(accumulator.items(), key=lambda item: format_key(item[0])):
        forward = totals["forward"]
        backward = totals["backward"]
        ratio = backward / forward if forward else None
        rows.append(
            {
                "key": key,
                "forward_s": forward,
                "backward_s": backward,
                "ratio": ratio,
            }
        )
    return rows


def format_ratio(ratio):
    return "n/a" if ratio is None else f"{ratio:.4f}"


def print_table(title, rows):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"{'key':<24} {'forward_ms':>14} {'backward_ms':>14} {'bwd/fwd':>10}")
    for row in rows:
        print(
            f"{format_key(row['key']):<24} "
            f"{row['forward_s'] * 1000.0:>14.3f} "
            f"{row['backward_s'] * 1000.0:>14.3f} "
            f"{format_ratio(row['ratio']):>10}"
        )


def ratio_values(rows):
    return [row["ratio"] for row in rows if row["ratio"] is not None]


def ratio_stats(rows):
    values = ratio_values(rows)
    if not values:
        return {
            "key": "ratios",
            "count": 0,
            "mean_ratio": None,
            "median_ratio": None,
            "std_ratio": None,
            "cv_ratio": None,
        }
    mean_ratio = statistics.mean(values)
    std_ratio = statistics.pstdev(values)
    return {
        "key": "ratios",
        "count": len(values),
        "mean_ratio": mean_ratio,
        "median_ratio": statistics.median(values),
        "std_ratio": std_ratio,
        "cv_ratio": None if mean_ratio == 0 else std_ratio / mean_ratio,
    }


def median_ratio_by_group(events, group_func):
    per_iteration_group = summarize(
        events, lambda event: (event.get("iteration", "unknown"), group_func(event))
    )
    grouped = defaultdict(list)
    for row in per_iteration_group:
        if row["ratio"] is None:
            continue
        _iteration, group = row["key"]
        grouped[group].append(row["ratio"])
    return [
        {"key": key, "count": len(values), "median_ratio": statistics.median(values)}
        for key, values in sorted(grouped.items(), key=lambda item: format_key(item[0]))
    ]


def print_ratio_stats(row):
    print("\nratio_stats")
    print("-----------")
    print(f"{'count':>8} {'mean':>10} {'median':>10} {'std':>10} {'cv':>10}")
    print(
        f"{row['count']:>8} "
        f"{format_ratio(row['mean_ratio']):>10} "
        f"{format_ratio(row['median_ratio']):>10} "
        f"{format_ratio(row['std_ratio']):>10} "
        f"{format_ratio(row['cv_ratio']):>10}"
    )


def print_median_table(title, rows):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"{'key':<24} {'count':>8} {'median_bwd/fwd':>16}")
    for row in rows:
        print(
            f"{format_key(row['key']):<24} "
            f"{row['count']:>8} "
            f"{format_ratio(row['median_ratio']):>16}"
        )


def write_csv(path, summaries):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "summary",
                "key",
                "forward_s",
                "backward_s",
                "ratio",
                "count",
                "mean_ratio",
                "median_ratio",
                "std_ratio",
                "cv_ratio",
            ],
        )
        writer.writeheader()
        for summary_name, rows in summaries:
            for row in rows:
                output = {"summary": summary_name, "key": format_key(row["key"])}
                for field in writer.fieldnames:
                    if field in {"summary", "key"}:
                        continue
                    value = row.get(field, "")
                    output[field] = "" if value is None else value
                writer.writerow(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--iteration-start", type=int, default=None)
    parser.add_argument("--iteration-end", type=int, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    args = parser.parse_args()

    if args.iteration is not None:
        iterations = [args.iteration]
    elif args.iteration_start is not None and args.iteration_end is not None:
        if args.iteration_start > args.iteration_end:
            raise RuntimeError("--iteration-start must be <= --iteration-end")
        iterations = list(range(args.iteration_start, args.iteration_end + 1))
    else:
        raise RuntimeError("Provide --iteration or --iteration-start/--iteration-end.")

    events = load_compute_events(args.trace_dir, iterations)

    if len(iterations) == 1:
        summaries = [
            ("global", summarize(events, lambda _event: "all")),
            ("per_rank", summarize(events, lambda event: event.get("rank", "unknown"))),
            (
                "per_logical_stage",
                summarize(events, lambda event: event.get("logical_stage", "unknown")),
            ),
            (
                "per_pp_vp",
                summarize(
                    events,
                    lambda event: (
                        event.get("pp_rank", "unknown"),
                        event.get("vp_rank", "unknown"),
                    ),
                ),
            ),
            (
                "per_microbatch",
                summarize(events, lambda event: event.get("microbatch", "unknown")),
            ),
        ]

        for title, rows in summaries:
            print_table(title, rows)
    else:
        per_iteration = summarize(events, lambda event: event.get("iteration", "unknown"))
        stats = ratio_stats(per_iteration)
        per_rank_median = median_ratio_by_group(events, lambda event: event.get("rank", "unknown"))
        per_stage_median = median_ratio_by_group(
            events, lambda event: event.get("logical_stage", "unknown")
        )
        summaries = [
            ("per_iteration_global", per_iteration),
            ("ratio_stats", [stats]),
            ("per_rank_median", per_rank_median),
            ("per_logical_stage_median", per_stage_median),
        ]

        print_table("per_iteration_global", per_iteration)
        print_ratio_stats(stats)
        print_median_table("per_rank_median", per_rank_median)
        print_median_table("per_logical_stage_median", per_stage_median)

    if args.output_csv is not None:
        write_csv(args.output_csv, summaries)
        print(f"\nSaved CSV summaries to {args.output_csv}")


if __name__ == "__main__":
    main()
