#!/usr/bin/env python3
"""Worker-balance tolerance experiment runner for SlackPipe.

Runs unconstrained optimize-joint baselines and hard worker-balance-constrained
variants, then summarizes optimum preservation and timing changes.

Recommended three-phase workflow:

1. Baseline discovery:
   run ``--variants baseline --write-baseline-optimal-csv ...`` to identify
   configurations whose unconstrained joint CP-SAT result is proven OPTIMAL.
2. Constrained comparison:
   run ``--config-csv <baseline-optimal.csv> --tolerance-percents 0,1.6,3,5,10``
   with ``--skip-existing`` to compare only configurations with useful
   baselines.
3. Repeated timing:
   rerun the baseline-OPTIMAL subset with ``--repeats 3`` or higher and compare
   median wall times for baseline, tol0, tol3, and tol5.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_TOLERANCES = [0.0, 3.0, 5.0, 10.0]
TIMING_TOLERANCES = {0.0, 3.0, 5.0}
SUPPORTED_VARIANTS = ("baseline", "tol0", "tol3", "tol5", "tol10")
SUMMARY_COLUMNS = [
    "B",
    "N",
    "J",
    "L",
    "variant",
    "tolerance_percent",
    "tolerance_layers",
    "status",
    "objective",
    "best_bound",
    "wall_time_seconds",
    "deterministic_time",
    "branches",
    "conflicts",
    "deterministic_replay_accepted",
    "stage_partition",
    "worker_aggregate_layer_loads",
    "baseline_status",
    "baseline_objective",
    "preserves_unconstrained_optimum",
    "speedup_vs_baseline",
    "raw_json_path",
    "error_type",
    "error_message",
]


@dataclass(frozen=True)
class Config:
    B: int
    N: int
    J: int
    L: int

    @property
    def label(self) -> str:
        return f"B{self.B}_N{self.N}_J{self.J}_L{self.L}"


@dataclass(frozen=True)
class Variant:
    name: str
    tolerance_percent: Optional[float]


def default_configs() -> List[Config]:
    configs: List[Config] = []
    for n in [32, 48, 64, 80]:
        for b in [24, 32, 48, 64]:
            for layers_per_stage in [2, 4]:
                configs.append(Config(B=b, N=n, J=16, L=n * layers_per_stage))
    return configs


def smoke_configs() -> List[Config]:
    return [Config(B=8, N=8, J=4, L=64)]


def read_config_csv(path: Path) -> List[Config]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"B", "N", "J", "L"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"config CSV missing columns: {sorted(missing)}")
        return [
            Config(
                B=int(row["B"]),
                N=int(row["N"]),
                J=int(row["J"]),
                L=int(row["L"]),
            )
            for row in reader
        ]


def csv_json(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"))


def objective_from(data: Dict[str, Any]) -> Optional[int]:
    value = data.get("makespan_ticks", data.get("makespan"))
    if value is None:
        return None
    return int(value)


def best_bound_from(data: Dict[str, Any]) -> Optional[float]:
    value = data.get("best_bound_ticks", data.get("best_bound"))
    if value is None:
        return None
    return float(value)


def has_usable_result(data: Dict[str, Any]) -> bool:
    status = data.get("status")
    if not isinstance(status, str) or not status:
        return False
    if status in {"ERROR", "TIMEOUT", "SKIPPED_BASELINE_NOT_OPTIMAL"}:
        return True
    return objective_from(data) is not None or best_bound_from(data) is not None


def deterministic_replay_accepted(data: Dict[str, Any]) -> bool:
    stats = data.get("search_stats") or {}
    candidates = stats.get("candidate_schedules") or {}
    if candidates:
        return int(candidates.get("accepted", 0) or 0) > 0 and int(
            candidates.get("rejected", 0) or 0
        ) == 0
    errors = data.get("errors")
    if isinstance(errors, list):
        return len(errors) == 0
    return data.get("status") in {"OPTIMAL", "FEASIBLE"} and bool(data.get("operations"))


def tolerance_label(percent: float) -> str:
    text = f"{percent:.12g}"
    text = text.replace("-", "m").replace(".", "p")
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    return f"tol{text}"


def variant_for_tolerance(percent: float) -> Variant:
    return Variant(tolerance_label(percent), percent)


def variant_list() -> List[Variant]:
    return [Variant("baseline", None)] + [
        variant_for_tolerance(tol) for tol in DEFAULT_TOLERANCES
    ]


def parse_tolerance_percents(values: Optional[str]) -> List[float]:
    if values is None:
        return []
    percents: List[float] = []
    seen = set()
    for raw_value in values.split(","):
        value_text = raw_value.strip()
        if not value_text:
            continue
        try:
            value = float(value_text)
        except ValueError as exc:
            raise ValueError(f"invalid tolerance percent {value_text!r}") from exc
        if value < 0.0:
            raise ValueError("tolerance percents must be non-negative")
        key = f"{value:.12g}"
        if key not in seen:
            percents.append(value)
            seen.add(key)
    if values is not None and not percents:
        raise ValueError("--tolerance-percents selected no tolerances")
    return percents


def parse_variant_name(name: str) -> Optional[Variant]:
    if name == "baseline":
        return Variant("baseline", None)
    for variant in variant_list()[1:]:
        if name == variant.name:
            return variant
    if name.startswith("tol") and len(name) > 3:
        percent_text = name[3:].replace("p", ".")
        try:
            percent = float(percent_text)
        except ValueError:
            return None
        if percent < 0.0:
            return None
        return variant_for_tolerance(percent)
    return None


def selected_variants(names: Optional[str], tolerance_percents: Optional[str]) -> List[Variant]:
    selected: List[Variant] = []
    seen = set()

    if names is None and tolerance_percents is None:
        names = "baseline,tol0,tol3,tol5,tol10"
    elif names is None and tolerance_percents is not None:
        names = "baseline"

    if names is not None:
        for raw_name in names.split(","):
            name = raw_name.strip()
            if not name:
                continue
            variant = parse_variant_name(name)
            if variant is None:
                raise ValueError(
                    f"unsupported variant {name!r}; supported variants include "
                    f"{', '.join(SUPPORTED_VARIANTS)} or generated labels such as tol1p6"
                )
            if variant.name not in seen:
                selected.append(variant)
                seen.add(variant.name)

    for percent in parse_tolerance_percents(tolerance_percents):
        variant = variant_for_tolerance(percent)
        if variant.name not in seen:
            selected.append(variant)
            seen.add(variant.name)

    if not selected:
        raise ValueError("no variants selected")
    return selected


def command_for(
    slackpipe_bin: Path,
    config: Config,
    variant: Variant,
    output_prefix: Path,
    stats_json: Path,
    args: argparse.Namespace,
) -> List[str]:
    cmd = [
        str(slackpipe_bin),
        "--algorithm",
        "optimize-joint",
        "--B",
        str(config.B),
        "--N",
        str(config.N),
        "--J",
        str(config.J),
        "--L",
        str(config.L),
        "--min-layers",
        str(args.min_layers),
        "--ratio-num",
        str(args.ratio_num),
        "--ratio-den",
        str(args.ratio_den),
        "--num-workers",
        str(args.num_workers),
        "--time-limit-seconds",
        str(args.time_limit_seconds),
        "--random-seed",
        str(args.random_seed),
        "--require-optimal",
        "false",
        "--output-prefix",
        str(output_prefix),
        "--search-stats-json",
        str(stats_json),
    ]
    if variant.tolerance_percent is not None:
        cmd.extend([
            "--worker-balance-tolerance-percent",
            str(variant.tolerance_percent),
        ])
    return cmd


def command_text(cmd: Sequence[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in cmd)


def run_command(cmd: Sequence[str], timeout: float) -> Tuple[int, str, str, float]:
    started = time.monotonic()
    completed = subprocess.run(
        list(cmd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    elapsed = time.monotonic() - started
    return completed.returncode, completed.stdout, completed.stderr, elapsed


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def raw_name(config: Config, variant: Variant, repeat: int, repeats: int) -> str:
    stem = f"{config.label}_{variant.name}"
    if repeats > 1:
        stem += f"_rep{repeat + 1}"
    return stem


def make_error_row(
    config: Config,
    variant: Variant,
    raw_path: Path,
    message: str,
    baseline: Optional[Dict[str, Any]],
    status: str = "ERROR",
    error_type: str = "exception",
) -> Dict[str, Any]:
    baseline_status = (baseline or {}).get("status", "")
    baseline_objective = objective_from(baseline or {}) if baseline else None
    return {
        "B": config.B,
        "N": config.N,
        "J": config.J,
        "L": config.L,
        "variant": variant.name,
        "tolerance_percent": "" if variant.tolerance_percent is None else variant.tolerance_percent,
        "tolerance_layers": "",
        "status": status,
        "objective": "",
        "best_bound": "",
        "wall_time_seconds": "",
        "deterministic_time": "",
        "branches": "",
        "conflicts": "",
        "deterministic_replay_accepted": "false",
        "stage_partition": "",
        "worker_aggregate_layer_loads": "",
        "baseline_status": baseline_status,
        "baseline_objective": "" if baseline_objective is None else baseline_objective,
        "preserves_unconstrained_optimum": "false",
        "speedup_vs_baseline": "",
        "raw_json_path": str(raw_path),
        "error_type": error_type,
        "error_message": message,
    }


def make_skipped_row(
    config: Config,
    variant: Variant,
    baseline: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    baseline_status = (baseline or {}).get("status", "")
    baseline_objective = objective_from(baseline or {}) if baseline else None
    return {
        "B": config.B,
        "N": config.N,
        "J": config.J,
        "L": config.L,
        "variant": variant.name,
        "tolerance_percent": "" if variant.tolerance_percent is None else variant.tolerance_percent,
        "tolerance_layers": "",
        "status": "SKIPPED_BASELINE_NOT_OPTIMAL",
        "objective": "",
        "best_bound": "",
        "wall_time_seconds": "",
        "deterministic_time": "",
        "branches": "",
        "conflicts": "",
        "deterministic_replay_accepted": "false",
        "stage_partition": "",
        "worker_aggregate_layer_loads": "",
        "baseline_status": baseline_status,
        "baseline_objective": "" if baseline_objective is None else baseline_objective,
        "preserves_unconstrained_optimum": "false",
        "speedup_vs_baseline": "",
        "raw_json_path": "",
        "error_type": "baseline_not_optimal",
        "error_message": f"baseline status is {baseline_status or 'missing'}",
    }


def summarize_runs(
    config: Config,
    variant: Variant,
    run_jsons: List[Dict[str, Any]],
    raw_paths: List[Path],
    baseline_jsons: List[Dict[str, Any]],
) -> Dict[str, Any]:
    representative = run_jsons[0]
    wall_times = [float(row.get("wall_time_seconds", 0.0) or 0.0) for row in run_jsons]
    median_wall = statistics.median(wall_times) if wall_times else 0.0
    baseline = baseline_jsons[0] if baseline_jsons else None
    baseline_status = baseline.get("status", "") if baseline else ""
    baseline_objective = objective_from(baseline) if baseline else None
    baseline_wall_times = [
        float(row.get("wall_time_seconds", 0.0) or 0.0) for row in baseline_jsons
    ]
    baseline_median_wall = statistics.median(baseline_wall_times) if baseline_wall_times else 0.0
    objective = objective_from(representative)
    accepted = deterministic_replay_accepted(representative)
    status = str(representative.get("status", ""))
    preserves = False
    if variant.tolerance_percent is not None:
        preserves = (
            baseline_status == "OPTIMAL"
            and status == "OPTIMAL"
            and accepted
            and baseline_objective is not None
            and objective == baseline_objective
        )
    constraint = representative.get("worker_balance_constraint") or {}
    speedup = ""
    if variant.tolerance_percent in TIMING_TOLERANCES and baseline_median_wall > 0.0:
        speedup = baseline_median_wall / median_wall if median_wall > 0.0 else ""
    return {
        "B": config.B,
        "N": config.N,
        "J": config.J,
        "L": config.L,
        "variant": variant.name,
        "tolerance_percent": "" if variant.tolerance_percent is None else variant.tolerance_percent,
        "tolerance_layers": constraint.get("tolerance_layers", ""),
        "status": status,
        "objective": "" if objective is None else objective,
        "best_bound": "" if best_bound_from(representative) is None else best_bound_from(representative),
        "wall_time_seconds": median_wall,
        "deterministic_time": representative.get("deterministic_time", ""),
        "branches": representative.get("branches", ""),
        "conflicts": representative.get("conflicts", ""),
        "deterministic_replay_accepted": str(accepted).lower(),
        "stage_partition": csv_json(representative.get("stage_partition") or representative.get("split")),
        "worker_aggregate_layer_loads": csv_json(
            representative.get("worker_aggregate_layer_loads")
        ),
        "baseline_status": baseline_status,
        "baseline_objective": "" if baseline_objective is None else baseline_objective,
        "preserves_unconstrained_optimum": str(preserves).lower(),
        "speedup_vs_baseline": speedup,
        "raw_json_path": ";".join(str(path) for path in raw_paths),
        "error_type": "",
        "error_message": "",
    }


def load_or_run(
    config: Config,
    variant: Variant,
    repeat: int,
    args: argparse.Namespace,
    raw_dir: Path,
) -> Tuple[Optional[Dict[str, Any]], Path, Optional[Dict[str, str]]]:
    stem = raw_name(config, variant, repeat, args.repeats)
    prefix = raw_dir / stem
    raw_json = prefix.with_suffix(".json")
    stats_json = raw_dir / f"{stem}.search_stats.json"
    cmd = command_for(
        slackpipe_bin=Path(args.slackpipe_bin),
        config=config,
        variant=variant,
        output_prefix=prefix,
        stats_json=stats_json,
        args=args,
    )
    if args.dry_run:
        print(command_text(cmd))
        return None, raw_json, None
    if args.skip_existing and raw_json.exists():
        try:
            data = load_json(raw_json)
            if has_usable_result(data):
                data["runner_reused_existing"] = True
                return data, raw_json, None
        except Exception:
            pass
    try:
        rc, stdout, stderr, elapsed = run_command(cmd, timeout=args.timeout_seconds)
    except subprocess.TimeoutExpired:
        return None, raw_json, {
            "status": "TIMEOUT",
            "error_type": "subprocess_timeout",
            "message": (
                f"command timed out after {args.timeout_seconds}s: "
                f"{command_text(cmd)}"
            ),
        }
    if not raw_json.exists():
        return None, raw_json, {
            "status": "ERROR",
            "error_type": "missing_json",
            "message": (
                f"missing JSON rc={rc} elapsed={elapsed:.3f}s "
                f"cmd={command_text(cmd)} stdout={stdout!r} stderr={stderr!r}"
            ),
        }
    try:
        data = load_json(raw_json)
        data["runner_exit_code"] = rc
        data["runner_elapsed_seconds"] = elapsed
        data["runner_stdout"] = stdout
        data["runner_stderr"] = stderr
        with raw_json.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        return data, raw_json, None
    except Exception as exc:
        return None, raw_json, {
            "status": "ERROR",
            "error_type": "json_parse_or_update",
            "message": str(exc),
        }


def run_study(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.config_csv:
        configs = read_config_csv(Path(args.config_csv))
    elif args.smoke:
        configs = smoke_configs()
    else:
        configs = default_configs()
    if args.max_configs is not None:
        configs = configs[: args.max_configs]

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    if not args.dry_run:
        raw_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

    variants = selected_variants(args.variants, args.tolerance_percents)
    run_baseline_first = args.skip_variants_if_baseline_not_optimal and any(
        variant.name != "baseline" for variant in variants
    )
    execution_variants = variants
    baseline_probe_only = False
    if run_baseline_first and all(variant.name != "baseline" for variant in variants):
        execution_variants = [Variant("baseline", None)] + variants
        baseline_probe_only = True

    summary_rows: List[Dict[str, Any]] = []
    errors = 0
    timeouts = 0
    skipped_baselines = 0
    completed_runs = 0
    expected_runs = len(configs) * len(variants) * args.repeats

    for config in configs:
        baseline_jsons: List[Dict[str, Any]] = []
        baseline_status_known = False
        for variant in execution_variants:
            if (
                args.skip_variants_if_baseline_not_optimal
                and variant.name != "baseline"
                and baseline_status_known
                and (not baseline_jsons or baseline_jsons[0].get("status") != "OPTIMAL")
            ):
                summary_rows.append(
                    make_skipped_row(config, variant, baseline_jsons[0] if baseline_jsons else None)
                )
                continue

            run_jsons: List[Dict[str, Any]] = []
            raw_paths: List[Path] = []
            variant_had_error = False
            for repeat in range(args.repeats):
                data, raw_json, error = load_or_run(config, variant, repeat, args, raw_dir)
                if args.dry_run:
                    continue
                if error is None and data is not None:
                    run_jsons.append(data)
                    raw_paths.append(raw_json)
                    completed_runs += 1
                else:
                    errors += 1
                    if error and error["status"] == "TIMEOUT":
                        timeouts += 1
                    variant_had_error = True
                    raw_paths.append(raw_json)
                    summary_rows.append(
                        make_error_row(
                            config,
                            variant,
                            raw_json,
                            (error or {}).get("message", "unknown error"),
                            baseline_jsons[0] if baseline_jsons else None,
                            status=(error or {}).get("status", "ERROR"),
                            error_type=(error or {}).get("error_type", "exception"),
                        )
                    )
                    break

            if args.dry_run or variant_had_error:
                continue
            if variant.name == "baseline":
                baseline_jsons = run_jsons
                baseline_status_known = True
                if baseline_probe_only:
                    continue
            row = summarize_runs(config, variant, run_jsons, raw_paths, baseline_jsons)
            summary_rows.append(row)

        if baseline_jsons and baseline_jsons[0].get("status") != "OPTIMAL":
            skipped_baselines += 1

    summary_path = output_dir / "summary.csv"
    if not args.dry_run:
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        if args.write_baseline_optimal_csv:
            write_baseline_optimal_csv(summary_rows, Path(args.write_baseline_optimal_csv))
        print_final_summary(
            summary_rows,
            summary_path,
            raw_dir,
            len(configs),
            expected_runs,
            completed_runs,
            skipped_baselines,
            errors,
            timeouts,
        )
    return summary_rows


def median(values: List[float]) -> str:
    if not values:
        return "n/a"
    return f"{statistics.median(values):.6g}"


def print_final_summary(
    rows: Sequence[Dict[str, Any]],
    summary_path: Path,
    raw_dir: Path,
    total_configs: int,
    expected_runs: int,
    completed_runs: int,
    skipped_baselines: int,
    errors: int,
    timeouts: int,
) -> None:
    baseline_optimal = {
        (int(row["B"]), int(row["N"]), int(row["J"]), int(row["L"]))
        for row in rows
        if row["variant"] == "baseline" and row["status"] == "OPTIMAL"
    }
    status_counts = Counter(str(row["status"]) for row in rows)
    status_by_variant: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        status_by_variant[str(row["variant"])][str(row["status"])] += 1
    baseline_rows = [row for row in rows if row["variant"] == "baseline"]
    baseline_completed_non_optimal = sum(
        1
        for row in baseline_rows
        if row["status"] not in {"OPTIMAL", "ERROR", "TIMEOUT"}
    )
    baseline_timeout_error = sum(
        1 for row in baseline_rows if row["status"] in {"ERROR", "TIMEOUT"}
    )
    skipped_variants = status_counts.get("SKIPPED_BASELINE_NOT_OPTIMAL", 0)
    preservation_denominator = len(baseline_optimal)

    print(f"Total configs: {total_configs}")
    print(f"Expected runs: {expected_runs}")
    print(f"Completed runs: {completed_runs}")
    print("Status counts:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    print("Status counts by variant:")
    for variant in sorted(status_by_variant):
        detail = ", ".join(
            f"{status}={count}" for status, count in sorted(status_by_variant[variant].items())
        )
        print(f"  {variant}: {detail}")
    print(f"Baseline OPTIMAL configs: {len(baseline_optimal)}")
    print(f"Baseline completed but non-OPTIMAL configs: {baseline_completed_non_optimal}")
    print(f"Baseline TIMEOUT/ERROR configs: {baseline_timeout_error}")
    print(f"Skipped variants: {skipped_variants}")
    print(f"Preservation denominator: {preservation_denominator}")
    tolerance_rows = [
        row for row in rows if row["variant"] != "baseline" and row["tolerance_percent"] not in {"", None}
    ]
    tolerances = sorted({float(row["tolerance_percent"]) for row in tolerance_rows})
    for tol in tolerances:
        variant = tolerance_label(tol)
        preserved = sum(
            1
            for row in rows
            if row["variant"] == variant
            and row["preserves_unconstrained_optimum"] == "true"
        )
        print(f"Tolerance {tol:g}% preserved: {preserved}")
    for tol in tolerances:
        variant = tolerance_label(tol)
        speedups = [
            float(row["speedup_vs_baseline"])
            for row in rows
            if row["variant"] == variant and row["speedup_vs_baseline"] not in {"", None}
        ]
        print(f"Median speedup {tol:g}% vs baseline: {median(speedups)}")
    print(f"Summary CSV path: {summary_path}")
    print(f"Raw result directory: {raw_dir}")


def write_baseline_optimal_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    configs = []
    for row in rows:
        if row["variant"] != "baseline" or row["status"] != "OPTIMAL":
            continue
        key = (int(row["B"]), int(row["N"]), int(row["J"]), int(row["L"]))
        if key not in seen:
            seen.add(key)
            configs.append(key)
    configs.sort()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["B", "N", "J", "L"])
        writer.writerows(configs)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-csv", help="CSV with columns B,N,J,L")
    parser.add_argument("--output-dir", default="results/worker-balance-study")
    parser.add_argument("--slackpipe-bin", default="build/release/slackpipe_cli")
    parser.add_argument("--time-limit-seconds", type=float, default=300.0)
    parser.add_argument("--timeout-seconds", type=float, default=360.0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--min-layers", type=int, default=1)
    parser.add_argument("--ratio-num", type=int, default=2)
    parser.add_argument("--ratio-den", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--variants",
        help=(
            "comma-separated variants: baseline,tol0,tol3,tol5,tol10; "
            "generated labels like tol1p6 are also accepted"
        ),
    )
    parser.add_argument(
        "--tolerance-percents",
        help="comma-separated worker-balance tolerance percentages, e.g. 1.6 or 0,1.6,3,5,10",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-variants-if-baseline-not-optimal", action="store_true")
    parser.add_argument("--write-baseline-optimal-csv")
    parser.add_argument("--max-configs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.timeout_seconds < args.time_limit_seconds:
        parser.error("--timeout-seconds must be >= --time-limit-seconds")
    try:
        selected_variants(args.variants, args.tolerance_percents)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_study(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
