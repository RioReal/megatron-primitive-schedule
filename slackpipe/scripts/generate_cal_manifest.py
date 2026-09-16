#!/usr/bin/env python3
"""Generate JSONL manifests for the SlackPipe CAL evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import cal_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=[
            "smoke",
            "smoke_big_1f1b",
            "main",
            "main_big_5min_1f1b",
            "main_big_5min_1f1b_uncapped",
            "solver_machinery_ablation_smoke",
            "solver_machinery_ablation",
            "solver_machinery_balance_pruning_smoke",
            "solver_machinery_balance_pruning",
            "ablation",
            "oracle",
            "all",
        ],
        required=True,
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--time-limit-seconds", type=float, default=None)
    parser.add_argument("--solver-threads", type=int, default=None)
    parser.add_argument("--worker-balance-tolerance-percent", type=float, default=None)
    parser.add_argument("--worker-balance-tolerance-layers", type=int, default=None)
    parser.add_argument(
        "--seeds",
        "--seed-list",
        dest="seeds",
        default=",".join(str(seed) for seed in cal_manifest.DEFAULT_SEEDS),
        help="comma-separated seed list for solver-backed non-smoke rows",
    )
    args = parser.parse_args(argv)

    rows, metadata = cal_manifest.generate_manifest(
        profile=args.profile,
        binary=args.binary,
        time_limit_seconds=args.time_limit_seconds,
        solver_threads=args.solver_threads,
        seeds=cal_manifest.parse_seed_list(args.seeds),
        worker_balance_tolerance_percent=args.worker_balance_tolerance_percent,
        worker_balance_tolerance_layers=args.worker_balance_tolerance_layers,
    )
    cal_manifest.write_manifest(rows, metadata, args.output)
    print(
        f"wrote {len(rows)} rows to {args.output} "
        f"(manifest_hash={metadata['manifest_hash']})"
    )
    print(f"metadata: {args.output.with_suffix(args.output.suffix + '.metadata.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
