"""Publication-style GPU logical-envelope figure from rank-local compact JSON."""

import argparse
import json
from pathlib import Path

from megatron.core.pipeline_parallel.slackpipe.figure_trace import validate_compact_trace


def load_panel(directory: Path, iteration: int) -> dict:
    traces = [json.loads(path.read_text()) for path in sorted(directory.glob("rank*_trace.json"))]
    if not traces:
        raise ValueError(f"No compact traces in {directory}")
    traces.sort(key=lambda trace: trace["rank"])
    if [t["rank"] for t in traces] != list(range(traces[0]["config"]["pp"])):
        raise ValueError("Missing or duplicate ranks")
    if len({t["config"]["host"] for t in traces}) != 1:
        raise ValueError("Cross-host clock alignment is unsupported")

    def ns(trace, record, key):
        return trace["base_time_ns"] + round(record[key] * 1000)

    steps = []
    for trace in traces:
        validate_compact_trace(trace)
        selected = [step for step in trace["steps"] if step["iteration"] == iteration]
        if len(selected) != 1:
            raise ValueError(f"Missing selected iteration {iteration}")
        steps.append(selected[0])
    start = min(ns(trace, step, "start_us") for trace, step in zip(traces, steps))
    end = max(ns(trace, step, "end_us") for trace, step in zip(traces, steps))
    event_end = max(
        ns(trace, step, "start_us") + round(step["cuda_elapsed_ms"] * 1e6)
        for trace, step in zip(traces, steps)
    )
    rows = []
    for trace in traces:
        for record in trace["records"]:
            if record["iteration"] == iteration:
                rows.append(
                    dict(
                        **record,
                        start_ms=(ns(trace, record, "start_us") - start) / 1e6,
                        end_ms=(ns(trace, record, "end_us") - start) / 1e6,
                    )
                )
    return dict(
        duration_ms=(end - start) / 1e6,
        rank_count=len(traces),
        records=rows,
        logical_ops=sum(r["kind"] in ("F", "B") for r in rows),
        config=traces[0]["config"],
        steps=steps,
        origin_ns=start,
        end_ns=end,
        mode=traces[0]["mode"],
        cuda_event_panel_estimate_ms=(event_end - start) / 1e6,
        cuda_event_panel_discrepancy_ms=abs(event_end - end) / 1e6,
        transport=traces[0]["transport"],
    )


def render(
    baseline: dict,
    slackpipe: dict,
    *,
    png: Path,
    pdf: Path,
    title: str,
    caption: str,
    color_mode: str,
    annotate_saved: bool,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    fields = (
        "pp",
        "num_stages",
        "num_layers",
        "num_microbatches",
        "hidden_size",
        "heads",
        "seq_length",
        "micro_batch_size",
        "vocab_size",
        "seed",
        "learning_rate",
        "dtype",
        "tf32",
        "dropout",
        "tp",
        "dp",
        "cp",
        "model",
        "model_sha256",
    )
    if any(baseline["config"][key] != slackpipe["config"][key] for key in fields):
        raise ValueError("Panels do not use matching model/data settings")
    if baseline["mode"] != "baseline" or slackpipe["mode"] != "slackpipe":
        raise ValueError("Expected baseline on top and SlackPipe on bottom")
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42}
    )
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 5.0 + 0.45 * (baseline["rank_count"] - 2)), sharex=True
    )
    maximum = max(baseline["duration_ms"], slackpipe["duration_ms"])
    for panel_index, (axis, panel) in enumerate(zip(axes, (baseline, slackpipe))):
        n = panel["rank_count"]
        for record in panel["records"]:
            kind = record["kind"]
            if kind in ("F", "B"):
                fraction = record["microbatch"] / max(1, panel["config"]["num_microbatches"] - 1)
                level = 0.42 + fraction * 0.40 if color_mode == "microbatch" else 0.65
                color = plt.get_cmap("Blues" if kind == "F" else "Oranges")(level)
            else:
                color = "#858b91" if kind == "optimizer" else "#54a88b"
            width = record["end_ms"] - record["start_ms"]
            axis.add_patch(
                Rectangle(
                    (record["start_ms"], n - 1 - record["rank"] - 0.30),
                    width,
                    0.6,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.35,
                )
            )
        axis.set_yticks(list(range(n)), [f"GPU {rank}" for rank in reversed(range(n))])
        axis.set_ylim(-1.2, n - 0.3)
        axis.set_xlim(-0.012 * maximum, maximum * 1.025)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        axis.tick_params(axis="y", length=0)
        axis.annotate(
            "",
            xy=(panel["duration_ms"], -0.65),
            xytext=(0, -0.65),
            arrowprops=dict(arrowstyle="<->", linestyle="--", color="#303030", linewidth=1),
        )
        axis.text(
            panel["duration_ms"] / 2,
            -0.86,
            f"{panel['duration_ms']:.2f} ms",
            ha="center",
            va="top",
            fontsize=10,
        )
        axis.set_title(
            (
                "(a) Profiled trace of interleaved 1F1B."
                if panel_index == 0
                else "(b) Profiled trace of SlackPipe."
            ),
            loc="left",
            fontsize=11,
            pad=9,
        )
    saved = baseline["duration_ms"] - slackpipe["duration_ms"]
    if annotate_saved and saved > 0:
        left, right = slackpipe["duration_ms"], baseline["duration_ms"]
        y = slackpipe["rank_count"] - 0.5
        axes[1].annotate(
            "",
            xy=(left, y),
            xytext=(right, y),
            arrowprops=dict(arrowstyle="<->", color="#b62732", linewidth=1.2),
        )
        axes[1].text(
            right,
            y + 0.1,
            f"{saved:.2f} ms saved",
            color="#b62732",
            ha="right",
            va="bottom",
            fontsize=9,
        )
    axes[1].set_xlabel("Time from panel step start (ms)")
    fig.suptitle(title, fontsize=12, y=0.98)
    fig.legend(
        handles=[
            Patch(color=plt.get_cmap("Blues")(0.65), label="Forward"),
            Patch(color=plt.get_cmap("Oranges")(0.65), label="Backward"),
            Patch(color="#858b91", label="Optimizer"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
    )
    fig.text(0.5, 0.015, caption, ha="center", va="bottom", fontsize=8)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.83, bottom=0.13, hspace=0.56)
    for path in (png, pdf):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return dict(
        schema_version="slackpipe.figure_report.v1",
        baseline=baseline,
        slackpipe=slackpipe,
        time_saved_ms=saved,
        png=str(png),
        pdf=str(pdf),
        title=title,
        caption=caption,
        timing="GPU activity envelopes correlated to profiler ranges; step spans include unplotted work",
        warning="One profiled step; profiler perturbation means this is not benchmark evidence",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--slackpipe", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=5)
    parser.add_argument("--title", default="Heterogeneous LARGE: PP=2, 4 logical stages")
    parser.add_argument(
        "--caption",
        default="GPU logical-work envelopes from torch.profiler; white gaps are outside labeled work, not guaranteed hardware idle.",
    )
    parser.add_argument("--color-mode", choices=("direction", "microbatch"), default="microbatch")
    parser.add_argument("--annotate-time-saved", action="store_true")
    args = parser.parse_args()
    report = render(
        load_panel(args.baseline, args.iteration),
        load_panel(args.slackpipe, args.iteration),
        png=args.output_png,
        pdf=args.output_pdf,
        title=args.title,
        caption=args.caption,
        color_mode=args.color_mode,
        annotate_saved=args.annotate_time_saved,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "baseline_ms": report["baseline"]["duration_ms"],
                "slackpipe_ms": report["slackpipe"]["duration_ms"],
                "time_saved_ms": report["time_saved_ms"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
