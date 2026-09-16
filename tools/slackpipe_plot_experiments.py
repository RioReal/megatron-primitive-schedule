#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION. All rights reserved.

"""Regenerate SlackPipe experiment plots from CSV artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args()
    root = args.experiment_dir
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise SystemExit(f"matplotlib is not available: {exc}") from exc

    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    partition = read_csv(root / "partition_schedule_ablation.csv")
    if partition:
        labels = [row["config"] for row in partition]
        x = range(len(labels))
        plt.figure(figsize=(max(8, len(labels) * 1.4), 4))
        plt.bar(
            [i - 0.25 for i in x],
            [100 * float(row["partition_benefit_mean"]) for row in partition],
            width=0.25,
            label="partition",
        )
        plt.bar(
            list(x),
            [100 * float(row["schedule_benefit_uniform_mean"]) for row in partition],
            width=0.25,
            label="schedule uniform",
        )
        plt.bar(
            [i + 0.25 for i in x],
            [100 * float(row["joint_benefit_mean"]) for row in partition],
            width=0.25,
            label="joint",
        )
        plt.xticks(list(x), labels, rotation=35, ha="right")
        plt.ylabel("speedup (%)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots / "partition_schedule_joint_speedup.png", dpi=160)
        plt.close()

    cost = read_csv(root / "cost_model_ablation.csv")
    if cost:
        by_model: dict[str, list[float]] = {}
        for row in cost:
            by_model.setdefault(row["cost_model"], []).append(float(row["measured_mean_ms"]))
        plt.figure(figsize=(7, 4))
        models = sorted(by_model)
        plt.boxplot([by_model[model] for model in models], labels=models)
        plt.ylabel("measured SlackPipe time (ms)")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(plots / "cost_model_ablation.png", dpi=160)
        plt.close()

    prediction = read_csv(root / "prediction_vs_measurement.csv")
    if prediction:
        plt.figure(figsize=(5, 4))
        plt.scatter(
            [float(row["predicted_makespan"]) for row in prediction],
            [float(row["measured_time_ms"]) for row in prediction],
        )
        plt.xlabel("predicted makespan")
        plt.ylabel("measured time (ms)")
        plt.tight_layout()
        plt.savefig(plots / "prediction_vs_measurement.png", dpi=160)
        plt.close()


if __name__ == "__main__":
    main()
