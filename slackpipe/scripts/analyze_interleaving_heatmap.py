#!/usr/bin/env python3
"""Analyze SlackPipe worker-local adjacent-operation interleaving CSVs."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Iterable, List

REQUIRED_COLUMNS = {
    "B",
    "N",
    "W",
    "L",
    "worker",
    "kind",
    "is_direction_switch",
    "prev_dir",
    "prev_b",
    "prev_n",
    "prev_stage_position",
    "prev_start",
    "prev_end",
    "next_dir",
    "next_b",
    "next_n",
    "next_stage_position",
    "next_start",
    "next_end",
    "prev_order_index",
    "next_order_index",
    "num_worker_ops",
    "prev_order_fraction",
    "next_order_fraction",
    "gap_ticks",
}
KINDS = ["F_to_F", "F_to_B", "B_to_F", "B_to_B"]
CONFIG_COLUMNS = ["B", "N", "W", "L"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input CSV path or glob. May be repeated.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--order-bins", type=int, default=10)
    args = parser.parse_args()
    if args.order_bins <= 0:
        parser.error("--order-bins must be positive")
    return args


def expand_inputs(patterns: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    seen = set()
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches:
            matches = [pattern]
        for match in matches:
            path = Path(match)
            if path in seen:
                continue
            seen.add(path)
            paths.append(path)
    return paths


def require_packages():
    try:
        import pandas as pd  # noqa: F401
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "required Python packages are missing: pandas and matplotlib are required"
        ) from exc


def load_inputs(paths: List[Path]):
    import pandas as pd

    frames = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"input file does not exist: {path}")
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if frame.empty:
            continue
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise SystemExit(f"{path} is missing required columns: {sorted(missing)}")
        frame = frame.copy()
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise SystemExit("no non-empty input CSV rows found")
    return pd.concat(frames, ignore_index=True)


def add_derived_columns(df, order_bins: int):
    import pandas as pd

    df = df.copy()
    numeric_columns = [
        "B",
        "N",
        "W",
        "L",
        "worker",
        "prev_b",
        "prev_n",
        "prev_stage_position",
        "prev_start",
        "prev_end",
        "next_b",
        "next_n",
        "next_stage_position",
        "next_start",
        "next_end",
        "prev_order_index",
        "next_order_index",
        "num_worker_ops",
        "prev_order_fraction",
        "next_order_fraction",
        "gap_ticks",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")
    df["delta_b"] = df["next_b"] - df["prev_b"]
    df["delta_n"] = df["next_n"] - df["prev_n"]
    df["config_key"] = df[CONFIG_COLUMNS].astype(str).agg("|".join, axis=1)
    df["prev_order_bin"] = (df["prev_order_fraction"].clip(0, 1) * order_bins).astype(int)
    df["next_order_bin"] = (df["next_order_fraction"].clip(0, 1) * order_bins).astype(int)
    df["prev_order_bin"] = df["prev_order_bin"].clip(upper=order_bins - 1)
    df["next_order_bin"] = df["next_order_bin"].clip(upper=order_bins - 1)
    return df


def write_summaries(df, output_dir: Path):
    stage = (
        df.groupby(CONFIG_COLUMNS + ["kind", "prev_stage_position", "next_stage_position"], dropna=False)
        .agg(
            count=("kind", "size"),
            num_workers=("worker", "nunique"),
            avg_gap_ticks=("gap_ticks", "mean"),
            min_gap_ticks=("gap_ticks", "min"),
            max_gap_ticks=("gap_ticks", "max"),
        )
        .reset_index()
        .sort_values(CONFIG_COLUMNS + ["kind", "prev_stage_position", "next_stage_position"])
    )
    stage.to_csv(output_dir / "stage_transition_summary.csv", index=False)

    per_config = (
        df.groupby(["config_key", "kind", "prev_stage_position", "next_stage_position"], dropna=False)
        .agg(count=("kind", "size"), avg_gap_ticks=("gap_ticks", "mean"), min_gap_ticks=("gap_ticks", "min"), max_gap_ticks=("gap_ticks", "max"))
        .reset_index()
    )
    global_stage = (
        per_config.groupby(["kind", "prev_stage_position", "next_stage_position"], dropna=False)
        .agg(
            count=("count", "sum"),
            num_configs=("config_key", "nunique"),
            avg_count_per_config=("count", "mean"),
            avg_gap_ticks=("avg_gap_ticks", "mean"),
            min_gap_ticks=("min_gap_ticks", "min"),
            max_gap_ticks=("max_gap_ticks", "max"),
        )
        .reset_index()
        .sort_values(["kind", "prev_stage_position", "next_stage_position"])
    )
    global_stage.to_csv(output_dir / "global_stage_transition_summary.csv", index=False)

    order = (
        df.groupby(["kind", "prev_order_bin", "next_order_bin"], dropna=False)
        .agg(
            count=("kind", "size"),
            avg_gap_ticks=("gap_ticks", "mean"),
            min_gap_ticks=("gap_ticks", "min"),
            max_gap_ticks=("gap_ticks", "max"),
        )
        .reset_index()
        .sort_values(["kind", "prev_order_bin", "next_order_bin"])
    )
    order.to_csv(output_dir / "order_bin_summary.csv", index=False)

    relation = (
        df.groupby(["kind", "delta_b", "delta_n", "prev_stage_position", "next_stage_position"], dropna=False)
        .agg(
            count=("kind", "size"),
            avg_gap_ticks=("gap_ticks", "mean"),
            min_gap_ticks=("gap_ticks", "min"),
            max_gap_ticks=("gap_ticks", "max"),
        )
        .reset_index()
        .sort_values(["kind", "delta_b", "delta_n", "prev_stage_position", "next_stage_position"])
    )
    relation.to_csv(output_dir / "relation_summary.csv", index=False)
    return stage, global_stage, order, relation


def draw_heatmap(matrix, title: str, xlabel: str, ylabel: str, path: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    image = ax.imshow(matrix.values, origin="lower", aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    fig.colorbar(image, ax=ax, label="count")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_heatmaps(df, output_dir: Path, order_bins: int):
    import pandas as pd

    for kind in KINDS:
        subset = df[df["kind"] == kind]
        if subset.empty:
            stage_matrix = pd.DataFrame([[0]])
        else:
            stage_matrix = subset.pivot_table(
                index="prev_stage_position",
                columns="next_stage_position",
                values="kind",
                aggfunc="size",
                fill_value=0,
            ).sort_index().sort_index(axis=1)
        draw_heatmap(
            stage_matrix,
            f"Stage transition heatmap {kind}",
            "next_stage_position",
            "prev_stage_position",
            output_dir / f"heatmap_stage_{kind}.png",
        )

        if subset.empty:
            order_matrix = pd.DataFrame([[0]], index=[0], columns=[0])
        else:
            order_matrix = subset.pivot_table(
                index="prev_order_bin",
                columns="next_order_bin",
                values="kind",
                aggfunc="size",
                fill_value=0,
            ).reindex(index=range(order_bins), columns=range(order_bins), fill_value=0)
        draw_heatmap(
            order_matrix,
            f"Order-position heatmap {kind}",
            "next_order_bin",
            "prev_order_bin",
            output_dir / f"heatmap_order_{kind}.png",
        )


def print_summary(df, stage, relation) -> None:
    total_rows = len(df)
    num_configs = df["config_key"].nunique()
    print(f"total rows: {total_rows}")
    print(f"number of configs: {num_configs}")
    print("counts by kind:")
    for kind, count in df["kind"].value_counts().sort_index().items():
        print(f"  {kind}: {count}")
    print("most common stage transitions:")
    top_stage = stage.sort_values("count", ascending=False).head(10)
    for _, row in top_stage.iterrows():
        print(
            "  "
            f"kind={row['kind']} prev={int(row['prev_stage_position'])} "
            f"next={int(row['next_stage_position'])} count={int(row['count'])}"
        )
    print("most common delta_b/delta_n relations:")
    top_relation = relation.sort_values("count", ascending=False).head(10)
    for _, row in top_relation.iterrows():
        print(
            "  "
            f"kind={row['kind']} delta_b={int(row['delta_b'])} "
            f"delta_n={int(row['delta_n'])} count={int(row['count'])}"
        )


def main() -> int:
    args = parse_args()
    require_packages()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = expand_inputs(args.input)
    df = load_inputs(paths)
    df = add_derived_columns(df, args.order_bins)
    stage, _global_stage, _order, relation = write_summaries(df, output_dir)
    write_heatmaps(df, output_dir, args.order_bins)
    print_summary(df, stage, relation)
    print(f"output dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
