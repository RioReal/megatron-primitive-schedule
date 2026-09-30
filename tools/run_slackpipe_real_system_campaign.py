# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Rotate fresh-process methods across repetitions, preserving every attempt."""

import argparse
import copy
import random
from pathlib import Path

from tools.run_slackpipe_eval import Experiment, argument_parser
from tools.slackpipe_eval_config import (
    COMPARISON_SCHEDULES,
    SCHEDULES,
    canonical_schedule,
    fingerprint,
    load_model,
)
from tools.slackpipe_eval_tables import export_tables
from tools.slackpipe_hybrid import write_json


def rotated_orders(schedules: list, repetitions: int, seed: int) -> list:
    order = list(schedules)
    random.Random(seed).shuffle(order)
    return [order[i % len(order) :] + order[: i % len(order)] for i in range(repetitions)]


def plot_campaign(root: Path, trace_receipts: dict) -> None:
    from tools.plot_schedule_trace import available_iterations, load_panel, render

    methods = [s for s in COMPARISON_SCHEDULES if s in trace_receipts]
    if not {"interleaved", "slackpipe"}.issubset(methods) or not all(
        trace_receipts[s]["status"] == "passed" for s in methods
    ):
        return
    directories = {s: root / trace_receipts[s]["directory"] for s in methods}
    # Immutable destination tied to the actual capture identities.
    destination = (
        root
        / "traces"
        / ("figure-" + fingerprint({s: r["directory"] for s, r in trace_receipts.items()})[:12])
    )
    # The report is written last. An interrupted render must not count as complete.
    if all(
        (destination / name).is_file() and (destination / name).stat().st_size > 0
        for name in ("timeline.png", "timeline.pdf", "figure_report.json")
    ):
        return
    candidates = sorted(
        set.intersection(*(available_iterations(p, cycle=0) for p in directories.values()))
    )
    if not candidates:
        raise ValueError("No common complete-rank capture/iteration for comparison figure")
    iteration = candidates[0]
    report = render(
        load_panel(directories["interleaved"], iteration, cycle=0),
        load_panel(directories["slackpipe"], iteration, cycle=0),
        **{
            name: (
                load_panel(directories[method], iteration, cycle=0)
                if method in directories
                else None
            )
            for name, method in (
                ("noninterleaved", "1f1b"),
                ("octopipe", "octopipe"),
                ("refined", "slackpipe-refined"),
            )
        },
        png=destination / "timeline.png",
        pdf=destination / "timeline.pdf",
        title=f"{root.parent.name} {root.name}: {len(methods)} scheduling methods",
        caption="White gaps: outside labeled work (schedule gaps/bubbles), not guaranteed hardware idle. Profiled step, not benchmark evidence.",
        color_mode="microbatch",
        annotate_saved=False,
    )
    report["selection"] = dict(
        iteration=iteration,
        cycle=0,
        candidates=candidates,
        rule="earliest common complete-rank global iteration in first cycle; all traces retained",
    )
    write_json(destination / "figure_report.json", report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", default="llama,nemotron_h")
    parser.add_argument("--sizes", default="4b,8b,16b,30b")
    parser.add_argument(
        "--schedules",
        default="1f1b,1f1b-interleave,octopipe,slackpipe,slackpipe-refined",
        help="Comma-separated methods (default: %(default)s); interleaved aliases 1f1b-interleave",
    )
    parser.add_argument("--configs", type=Path, default=Path("configs/slackpipe_eval"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    campaign, extra = parser.parse_known_args()
    schedules = [canonical_schedule(s.strip()) for s in campaign.schedules.split(",")]
    if not schedules or len(set(schedules)) != len(schedules) or set(schedules) - set(SCHEDULES):
        parser.error("Unknown/duplicate schedule")
    config_paths = [
        campaign.configs / f"{f.strip()}_{s.strip()}.json"
        for f in campaign.families.split(",")
        for s in campaign.sizes.split(",")
    ]
    configs = []
    seen_paths, seen_models, seen_roots = set(), set(), set()
    for path in config_paths:
        if not path.is_file():
            parser.error(f"Missing config: {path}")
        model = load_model(path)
        root = campaign.output / model["model_family"] / model["scale_label"]
        identity = fingerprint(model)
        if path.resolve() in seen_paths or identity in seen_models or root.resolve() in seen_roots:
            parser.error(f"Duplicate config/model or colliding output directory: {path}: {root}")
        seen_paths.add(path.resolve())
        seen_models.add(identity)
        seen_roots.add(root.resolve())
        configs.append((path, model, root))
    orders = []
    try:
        for path, model, root in configs:
            shared_receipts = set()
            base = argument_parser().parse_args(
                ["full", "--model-config", str(path), "--output", str(root), *extra]
            )
            ordering = rotated_orders(schedules, base.repetitions, base.seed)
            orders.append(dict(model=model["name"], orders=ordering))
            write_json(campaign.output / "execution_order.json", orders)
            successful = set()
            for rep, order in enumerate(ordering):
                for schedule in order:
                    args = copy.copy(base)
                    args.stage, args.schedule = "benchmark", schedule
                    args.logical_stages = args.pp if schedule == "1f1b" else base.logical_stages
                    args.run_index = rep
                    args.resume, args.force = (campaign.resume or rep > 0, campaign.force)
                    experiment = Experiment(args, shared_receipts=shared_receipts)
                    result = experiment.execute()
                    shared_receipts.update(
                        experiment.receipt_path(stage)
                        for stage, receipt in experiment.completed.items()
                        if receipt["status"] == "passed"
                    )
                    if result["status"] == "passed":
                        successful.add(schedule)
            traces = {}
            for schedule in schedules:
                if schedule not in successful:
                    continue
                args = copy.copy(base)
                args.stage, args.schedule = "trace", schedule
                args.logical_stages = args.pp if schedule == "1f1b" else base.logical_stages
                args.run_index, args.resume = None, True
                args.force = campaign.force
                traces[schedule] = Experiment(args).execute()
            plot_campaign(root, traces)
    finally:
        export_tables(campaign.output, config_paths)


if __name__ == "__main__":
    main()
