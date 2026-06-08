#!/usr/bin/env python3

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

COMPUTE_PHASES = {"forward_compute", "backward_compute"}


def load_compute_events(trace_dir: str, iteration: int):
    pattern = f"rank_*_iter_{iteration}.jsonl"
    paths = sorted(Path(trace_dir).glob(pattern))
    if not paths:
        raise RuntimeError(f"No trace files matching {pattern!r} found in {trace_dir}")

    events = []
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
                event["duration"] = float(event["t_end"]) - float(event["t_start"])
                events.append(event)

    if not events:
        raise RuntimeError(f"No compute events found for iteration {iteration} in {trace_dir}")
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


def write_csv(path, summaries):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["summary", "key", "forward_s", "backward_s", "ratio"],
        )
        writer.writeheader()
        for summary_name, rows in summaries:
            for row in rows:
                writer.writerow(
                    {
                        "summary": summary_name,
                        "key": format_key(row["key"]),
                        "forward_s": row["forward_s"],
                        "backward_s": row["backward_s"],
                        "ratio": "" if row["ratio"] is None else row["ratio"],
                    }
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output-csv", type=str, default=None)
    args = parser.parse_args()

    events = load_compute_events(args.trace_dir, args.iteration)

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
                lambda event: (event.get("pp_rank", "unknown"), event.get("vp_rank", "unknown")),
            ),
        ),
        (
            "per_microbatch",
            summarize(events, lambda event: event.get("microbatch", "unknown")),
        ),
    ]

    for title, rows in summaries:
        print_table(title, rows)

    if args.output_csv is not None:
        write_csv(args.output_csv, summaries)
        print(f"\nSaved CSV summaries to {args.output_csv}")


if __name__ == "__main__":
    main()
