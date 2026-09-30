#!/usr/bin/env python3
"""Plot evaluated CLI CSV intervals, without simulation or timestamp rescaling.

Example:
    python slackpipe/scripts/plot_schedule_comparison.py \
        --left-prefix build/run/slackpipe --right-prefix build/run/octopipe \
        --output-dir build/run
Each prefix must have the existing CLI .csv and .plan.json artifacts.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def load_schedule(prefix: Path) -> dict:
    """Read native operation intervals and check coverage, placement and timing."""
    csv_path, plan_path = Path(f"{prefix}.csv"), Path(f"{prefix}.plan.json")
    plan = json.loads(plan_path.read_text())
    with csv_path.open(newline="") as handle:
        operations = [
            {k: v if k in ("name", "kind") else int(v) for k, v in row.items()}
            for row in csv.DictReader(handle)
        ]
    b, n, w = (plan[k] for k in ("num_microbatches", "num_stages", "num_workers"))
    keys = [(op["kind"], op["microbatch"], op["stage"]) for op in operations]
    expected = {(kind, m, s) for kind in ("F", "B") for m in range(b) for s in range(n)}
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Missing, duplicate, or non-F/B operations")
    makespan = max(op["end"] for op in operations)
    if makespan != plan["predicted_makespan"]:
        raise ValueError("CSV and plan makespans disagree")
    for op in operations:
        if op["worker"] != plan["stage_to_worker"][op["stage"]]:
            raise ValueError("CSV and plan placement disagree")
        if op["start"] < 0 or op["duration"] <= 0 or op["end"] - op["start"] != op["duration"]:
            raise ValueError("Expected positive, consistent compute intervals")
    lanes = [
        sorted((op for op in operations if op["worker"] == r), key=lambda x: x["start"])
        for r in range(w)
    ]
    leading, trailing, residual, busy = [], [], [], []
    for rank, lane in enumerate(lanes):
        actual_order = [(op["kind"], op["microbatch"], op["stage"]) for op in lane]
        plan_order = [
            (op["kind"], op["microbatch"], op["stage"]) for op in plan["operations"][rank]
        ]
        if actual_order != plan_order:
            raise ValueError("CSV worker order differs from exported plan")
        gaps = [right["start"] - left["end"] for left, right in zip(lane, lane[1:])]
        if any(gap < 0 for gap in gaps):
            raise ValueError("Overlapping worker compute intervals")
        leading.append(lane[0]["start"] if lane else makespan)
        trailing.append(makespan - lane[-1]["end"] if lane else 0)
        residual.append(sum(gaps))
        busy.append(sum(op["duration"] for op in lane))
    boundary = [a + b for a, b in zip(leading, trailing)]
    bubble = [a + b for a, b in zip(boundary, residual)]
    if any(work + gap != makespan for work, gap in zip(busy, bubble)):
        raise ValueError("Compute plus bubbles must cover each worker timeline")
    metrics = dict(
        makespan=makespan,
        num_stages=n,
        num_workers=w,
        unique_operations=len(keys),
        leading=leading,
        trailing=trailing,
        boundary=boundary,
        residual=residual,
        bubble=bubble,
        boundary_sum=sum(boundary),
        residual_sum=sum(residual),
        delta_b=max(bubble) - min(bubble),
        busy=busy,
        utilization=[value / makespan for value in busy],
        layer_split=plan["layer_split"],
        stage_to_worker=plan["stage_to_worker"],
        source_sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (csv_path, plan_path)
        },
    )
    return dict(plan=plan, lanes=lanes, metrics=metrics)


def render(schedules: list[dict], labels: list[str], output: Path, xmax: int) -> None:
    """Draw exact intervals on a shared tick scale; blank intervals are bubbles."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = [
        "#83b8dd",
        "#efb374",
        "#9ccba0",
        "#e8949a",
        "#b4a4d4",
        "#c2b08e",
        "#dfacd0",
        "#a6c8c5",
        "#5f92ba",
        "#dcc66b",
        "#72a77b",
        "#b37891",
    ]
    fig, axes = plt.subplots(1, len(schedules), figsize=(12 * len(schedules), 5.4), squeeze=False)
    for ax, schedule, label in zip(axes[0], schedules, labels):
        m = schedule["metrics"]
        for rank, lane in enumerate(schedule["lanes"]):
            for op in lane:
                ax.barh(
                    rank,
                    op["duration"],
                    left=op["start"],
                    height=0.68,
                    color=colors[op["stage"] % len(colors)],
                    edgecolor="#333333",
                    linewidth=0.4,
                    hatch="//" if op["kind"] == "B" else None,
                    zorder=3,
                )
                if op["duration"] >= xmax * 0.012:
                    ax.text(
                        (op["start"] + op["end"]) / 2,
                        rank,
                        f'{op["kind"]}{op["microbatch"]}',
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        zorder=4,
                    )
        ax.set_title(
            f'{label} | makespan {m["makespan"]} ticks\n'
            f'split {m["layer_split"]} | placement {m["stage_to_worker"]}',
            fontsize=10,
            pad=15,
        )
        ax.set_yticks(
            range(m["num_workers"]), [f"W{r}\n{u:.1%} busy" for r, u in enumerate(m["utilization"])]
        )
        ax.set_ylim(m["num_workers"] - 0.4, -0.6)
        ax.set_xlim(0, xmax * 1.025)
        ax.set_xlabel("Simulated time (ticks)")
        ax.axvline(m["makespan"], color="#555555", linestyle="--", linewidth=1)
        ax.grid(axis="x", color="#dddddd", linewidth=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    stages = max(s["metrics"]["num_stages"] for s in schedules)
    legend = [Patch(facecolor=colors[s % len(colors)], label=f"Stage {s}") for s in range(stages)]
    legend += [
        Patch(facecolor="white", edgecolor="#333333", label="F: forward"),
        Patch(facecolor="white", edgecolor="#333333", hatch="//", label="B: complete backward"),
    ]
    fig.legend(
        handles=legend, loc="lower center", ncol=min(10, len(legend)), frameon=False, fontsize=8
    )
    fig.suptitle(
        "Common evaluated compute intervals | labels: F/B + microbatch | blank gaps: model bubbles",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    """Create two individual figures, a shared-scale pair, and metric summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-prefix", required=True, type=Path)
    parser.add_argument("--right-prefix", required=True, type=Path)
    parser.add_argument("--left-label", default="SlackPipe")
    parser.add_argument("--right-label", default="OctoPipe Algorithm 1 (placement enabled)")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--allow-stage-refinement",
        action="store_true",
        help="Compare the same solver with N and N+W stages",
    )
    args = parser.parse_args()
    schedules = [load_schedule(prefix) for prefix in (args.left_prefix, args.right_prefix)]
    if args.allow_stage_refinement:
        base, refined = (s["plan"] for s in schedules)
        if refined["num_stages"] != base["num_stages"] + base["num_workers"]:
            raise ValueError("Expected N+W stages for SlackPipe-refined")
    for key in ("num_microbatches", "num_workers", "num_layers", "cost_model") + (
        () if args.allow_stage_refinement else ("num_stages",)
    ):
        if schedules[0]["plan"].get(key) != schedules[1]["plan"].get(key):
            raise ValueError(f"Incomparable inputs: {key}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = [args.left_label, args.right_label]
    xmax = max(s["metrics"]["makespan"] for s in schedules)
    names = ("slackpipe", "slackpipe_refined" if args.allow_stage_refinement else "octopipe")
    for s, label, filename in zip(schedules, labels, (f"{name}_gantt.png" for name in names)):
        render([s], [label], args.output_dir / filename, xmax)
    render(schedules, labels, args.output_dir / "comparison_gantt.png", xmax)
    summary = {name: s["metrics"] for name, s in zip(names, schedules)}
    (args.output_dir / "comparison_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
