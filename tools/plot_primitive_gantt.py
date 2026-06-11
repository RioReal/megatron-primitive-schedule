#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_trace(trace_dir: str, iterations) -> pd.DataFrame:
    rows = []
    for iteration in iterations:
        pattern = f"rank_*_iter_{iteration}.jsonl"
        paths = sorted(Path(trace_dir).glob(pattern))
        if not paths:
            raise RuntimeError(f"No trace files matching {pattern!r} found in {trace_dir}")

        for path in paths:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        event = json.loads(line)
                        event.setdefault("iteration", iteration)
                        rows.append(event)

    if not rows:
        raise RuntimeError(f"No trace events found in {trace_dir}")

    df = pd.DataFrame(rows)

    required = {"rank", "phase", "t_start", "t_end"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Trace is missing required fields: {missing}")

    # Normalize time so the plot starts at zero.
    t0 = df["t_start"].min()
    df["start_ms"] = (df["t_start"] - t0) * 1000.0
    df["duration_ms"] = (df["t_end"] - df["t_start"]) * 1000.0

    return df


def make_label(row) -> str:
    op = row.get("op_type", "?")
    mb = row.get("microbatch", "?")
    stage = row.get("logical_stage", "?")
    vp = row.get("vp_rank", "?")
    phase = row.get("phase", "")
    short_phase = {
        "forward_recv": "F recv",
        "forward_compute": "F",
        "forward_send": "F send",
        "backward_recv": "B recv",
        "backward_compute": "B",
        "backward_send": "B send",
    }.get(phase, phase)

    return f"{short_phase} mb{mb} s{stage} v{vp}"


def plot_gantt(df: pd.DataFrame, output: str, title: str):
    # Sort ranks top-to-bottom.
    ranks = sorted(df["rank"].unique())
    y_positions = {rank: i for i, rank in enumerate(ranks)}

    phase_order = [
        "forward_recv",
        "forward_compute",
        "forward_send",
        "backward_recv",
        "backward_compute",
        "backward_send",
    ]

    # Let matplotlib choose colors from its default cycle.
    phase_to_color_index = {phase: i for i, phase in enumerate(phase_order)}
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig_height = max(3.0, 0.75 * len(ranks) + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    for _, row in df.sort_values(["rank", "start_ms"]).iterrows():
        rank = row["rank"]
        y = y_positions[rank]
        phase = row["phase"]
        color = color_cycle[phase_to_color_index.get(phase, 0) % len(color_cycle)]

        ax.barh(
            y=y,
            width=row["duration_ms"],
            left=row["start_ms"],
            height=0.55,
            color=color,
            edgecolor="black",
            linewidth=0.4,
        )

        # Label only reasonably wide boxes to avoid unreadable clutter.
        if row["duration_ms"] >= 2.0:
            ax.text(
                row["start_ms"] + row["duration_ms"] / 2,
                y,
                make_label(row),
                ha="center",
                va="center",
                fontsize=7,
                clip_on=True,
            )

    ax.set_yticks([y_positions[r] for r in ranks])
    ax.set_yticklabels([f"rank {r}" for r in ranks])
    ax.invert_yaxis()

    ax.set_xlabel("Time since first traced event (ms)")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    # Legend.
    handles = []
    labels = []
    for phase in phase_order:
        if phase in set(df["phase"]):
            color = color_cycle[phase_to_color_index[phase] % len(color_cycle)]
            handles.append(plt.Rectangle((0, 0), 1, 1, color=color))
            labels.append(phase)

    ax.legend(handles, labels, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    print(f"Saved Gantt plot to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--iteration-start", type=int, default=None)
    parser.add_argument("--iteration-end", type=int, default=None)
    parser.add_argument("--output", default="primitive_gantt.png")
    parser.add_argument("--title", default="Pipeline runtime schedule")
    parser.add_argument("--microbatch", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--phase", type=str, default=None)
    parser.add_argument("--compute-only", action="store_true")
    args = parser.parse_args()

    if args.iteration is not None:
        iterations = [args.iteration]
    elif args.iteration_start is not None and args.iteration_end is not None:
        if args.iteration_start > args.iteration_end:
            raise RuntimeError("--iteration-start must be <= --iteration-end")
        iterations = range(args.iteration_start, args.iteration_end + 1)
    else:
        raise RuntimeError("Provide --iteration or --iteration-start/--iteration-end.")

    df = load_trace(args.trace_dir, iterations)

    if args.compute_only:
        df = df[df["phase"].isin({"forward_compute", "backward_compute"})]
        if df.empty:
            raise RuntimeError("No compute events remain after --compute-only filtering.")

    if args.microbatch is not None and "microbatch" in df.columns:
        df = df[df["microbatch"] == args.microbatch]

    if args.rank is not None:
        df = df[df["rank"] == args.rank]

    if args.phase is not None:
        df = df[df["phase"] == args.phase]

    if df.empty:
        raise RuntimeError("No events remain after filtering.")

    plot_gantt(df, args.output, args.title)


if __name__ == "__main__":
    main()
