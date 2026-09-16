#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable


PRIMARY_W_VALUES = (2, 4, 8, 16, 32)
PRIMARY_VIRTUAL_STAGES_PER_WORKER = (1, 2, 4)
PRIMARY_MICROBATCH_RATIOS = (Fraction(1, 2), Fraction(1, 1), Fraction(2, 1), Fraction(4, 1))
PRIMARY_CORE_L_VALUES = (24, 32, 48, 64, 80, 96, 128)
PRIMARY_REAL_L_VALUES = (61, 94, 126)

PRIMARY_RATIO_NUM = 2
PRIMARY_RATIO_DEN = 1
PRIMARY_COMMUNICATION_TICKS = 0
PRIMARY_MIN_LAYERS = 1
PRIMARY_TIME_LIMIT_SECONDS = 600.0
PRIMARY_SOLVER_THREADS = 16
PRIMARY_RANDOM_SEED = 1
PRIMARY_DERIVE_SCHEDULE_ONLY_BEST = False
BUDGET_POLICY_VERSION = 1

PRIMARY_MODES = ("joint", "partition-only", "schedule-only-uniform")
SENSITIVITY_OUTPUT_DIR = Path("results/ablation_sensitivity")
PRIMARY_OUTPUT_DIR = Path("results/primary_ablation")


@dataclass(frozen=True)
class StructuralConfig:
    configuration_id: str
    depth_set: str
    B: int
    N: int
    W: int
    L: int
    microbatches_per_worker: Fraction
    stages_per_worker: int
    layers_per_stage: Fraction


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    structural: StructuralConfig
    optimization_mode: str
    fixed_partition_source: str
    fixed_order_partition_backend: str
    ratio_num: int
    ratio_den: int
    communication_ticks: int
    min_layers: int
    time_limit_seconds: float
    solver_threads: int
    random_seed: int
    use_bfs_hints: bool | None = None
    symmetry_break_f0_fifo: bool | None = None
    pressure_pruning: bool = False
    sensitivity_axis: str = "primary"


RESULT_COLUMNS = [
    "run_id",
    "configuration_id",
    "depth_set",
    "B",
    "N",
    "W",
    "L",
    "microbatches_per_worker",
    "stages_per_worker",
    "layers_per_stage",
    "B_over_W",
    "N_over_W",
    "L_over_N",
    "optimization_mode",
    "fixed_partition_source",
    "fixed_order_partition_backend",
    "fixed_order_partition_backend_requested",
    "fixed_order_partition_backend_effective",
    "schema_version",
    "budget_policy_version",
    "evaluation_method_version",
    "requested_method",
    "canonical_method",
    "method_contract_hash",
    "fixed_schedule_rule",
    "uniform_partition_rule",
    "actual_solver_path",
    "partition_decision",
    "schedule_decision",
    "full_partition_fixed",
    "worker_aggregate_loads_fixed",
    "partition_optimized",
    "schedule_optimized",
    "predecessor_candidate_restriction_requested",
    "predecessor_candidate_restriction_active",
    "predecessor_candidate_rule",
    "solver_status_raw",
    "reported_status",
    "fallback_reason",
    "returned_solution_source",
    "communication_model",
    "activation_model",
    "activation_units_per_layer",
    "explicit_stage_activation_units",
    "activation_cap_mode",
    "activation_cap_units_per_worker",
    "activation_cap_enforced",
    "activation_cap_enforcement_requested",
    "activation_cap_enforcement_mode",
    "activation_cap_solver_supported",
    "activation_cap_solver_support_level",
    "activation_cap_enforced_in_solver",
    "activation_cap_constraints_added",
    "activation_retained_interval_count",
    "activation_cumulative_constraint_count",
    "activation_variable_demand_count",
    "activation_fixed_demand_count",
    "activation_constraint_build_runtime_seconds",
    "incumbent_rejected_for_activation_cap",
    "activation_model_validation_agreement",
    "activation_model_disagreement_details",
    "activation_cap_formulation_version",
    "activation_cap_satisfied",
    "maximum_worker_peak_activation_units",
    "global_simultaneous_peak_activation_units",
    "activation_peak_ratio_to_uniform",
    "activation_cap_derivation_hash",
    "result_validation_passed",
    "result_validation_error",
    "executable_name",
    "pipeline_workers",
    "solver_threads",
    "time_limit_seconds",
    "random_seed",
    "ratio_num",
    "ratio_den",
    "communication_ticks",
    "min_layers",
    "sensitivity_axis",
    "solver_status",
    "proven_optimal",
    "incumbent_objective",
    "best_objective_bound",
    "replayed_makespan",
    "relative_optimality_gap",
    "utilization",
    "useful_work",
    "busy_time",
    "idle_time",
    "maximum_worker_load",
    "fill_time",
    "drain_time",
    "communication_blocked_time",
    "incumbent_budget_seconds",
    "incumbent_model_build_seconds",
    "incumbent_solve_seconds",
    "incumbent_status",
    "joint_budget_seconds",
    "joint_model_build_seconds",
    "joint_solve_seconds",
    "joint_status",
    "incumbent_method_requested",
    "incumbent_method_effective",
    "bfs_incumbent_method_requested",
    "bfs_incumbent_method_effective",
    "incumbent_source",
    "incumbent_feasible",
    "incumbent_primary_objective",
    "incumbent_hybrid_min_slack",
    "incumbent_baseline_primary_objective",
    "incumbent_baseline_hybrid_min_slack",
    "incumbent_improved_over_baseline",
    "incumbent_hybrid_stage_scores",
    "incumbent_hybrid_bottleneck_stages",
    "horizon_source",
    "hint_budget_seconds",
    "hint_elapsed_seconds",
    "hint_iterations",
    "hint_candidates_generated",
    "hint_candidates_simulated",
    "hint_partition_moves_accepted",
    "hint_interleaving_moves_accepted",
    "hint_deadline_reached",
    "hint_termination_reason",
    "hints_requested",
    "hints_effective",
    "hint_source",
    "hint_scope",
    "hint_complete_for_basic_model",
    "hint_complete_for_full_model",
    "hinted_layer_variable_count",
    "hinted_operation_variable_count",
    "hinted_scalar_variable_count",
    "hinted_auxiliary_variable_count",
    "hinted_total_variable_count",
    "auxiliary_variable_count",
    "fallback_available",
    "fallback_source",
    "solution_source",
    "fallback_used",
    "final_split",
    "final_worker_local_order",
    "uniform_split",
    "command",
    "return_code",
    "completed",
    "optimization_wall_time_seconds",
    "git_commit",
    "git_dirty",
    "output_json_path",
    "start_timestamp_utc",
    "end_timestamp_utc",
    "stdout_log_path",
    "stderr_log_path",
    "last_lifecycle_phase",
    "stdout",
    "stderr",
    "failure",
    "partition_order_validation_passed",
    "schedule_partition_validation_passed",
    "primary_protocol_validation_passed",
]

MANIFEST_COLUMNS = [
    "run_id",
    "configuration_id",
    "depth_set",
    "B",
    "N",
    "W",
    "L",
    "microbatches_per_worker",
    "stages_per_worker",
    "layers_per_stage",
    "B_over_W",
    "N_over_W",
    "L_over_N",
    "optimization_mode",
    "fixed_partition_source",
    "fixed_order_partition_backend",
    "pipeline_workers",
    "solver_threads",
    "time_limit_seconds",
    "random_seed",
    "ratio_num",
    "ratio_den",
    "communication_ticks",
    "min_layers",
    "sensitivity_axis",
]


def fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def fraction_float(value: Fraction) -> float:
    return float(value.numerator) / float(value.denominator)


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def stable_id(parts: Iterable[Any]) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def uniform_split(total_layers: int, stages: int, min_layers: int = PRIMARY_MIN_LAYERS) -> list[int]:
    minimum_total = stages * min_layers
    if total_layers < minimum_total:
        raise ValueError("uniform split requires L >= N * min_layers")
    remaining = total_layers - minimum_total
    q, remainder = divmod(remaining, stages)
    split = [min_layers + q for _ in range(stages)]
    for stage in range(remainder):
        split[stage] += 1
    if min(split) <= 0 or max(split) - min(split) > 1 or sum(split) != total_layers:
        raise ValueError("invalid deterministic uniform split")
    return split


def generate_structural_configs() -> list[StructuralConfig]:
    configs: list[StructuralConfig] = []
    for depth_set, layer_values in (
        ("core", PRIMARY_CORE_L_VALUES),
        ("real-depth-extension", PRIMARY_REAL_L_VALUES),
    ):
        for W in PRIMARY_W_VALUES:
            for stages_per_worker in PRIMARY_VIRTUAL_STAGES_PER_WORKER:
                N = W * stages_per_worker
                for microbatch_ratio in PRIMARY_MICROBATCH_RATIOS:
                    B_fraction = W * microbatch_ratio
                    if B_fraction.denominator != 1:
                        raise ValueError("generated non-integer microbatch count")
                    B = B_fraction.numerator
                    for L in layer_values:
                        if N % W != 0:
                            continue
                        if N > L:
                            continue
                        index = len(configs) + 1
                        configs.append(
                            StructuralConfig(
                                configuration_id=f"cfg-{index:04d}",
                                depth_set=depth_set,
                                B=B,
                                N=N,
                                W=W,
                                L=L,
                                microbatches_per_worker=microbatch_ratio,
                                stages_per_worker=stages_per_worker,
                                layers_per_stage=Fraction(L, N),
                            )
                        )
    return configs


def validate_primary_manifest(configs: list[StructuralConfig], runs: list[RunSpec]) -> None:
    core = sum(1 for config in configs if config.depth_set == "core")
    real = sum(1 for config in configs if config.depth_set == "real-depth-extension")
    if core != 360 or real != 160 or len(configs) != 520:
        raise ValueError(f"primary manifest count mismatch: core={core} real={real} total={len(configs)}")
    if len(runs) != 1560:
        raise ValueError(f"primary run count mismatch: {len(runs)}")
    if PRIMARY_DERIVE_SCHEDULE_ONLY_BEST:
        raise ValueError("primary manifest requires derive_schedule_only_best=false")
    config_ids = [config.configuration_id for config in configs]
    if len(set(config_ids)) != len(config_ids):
        raise ValueError("duplicate configuration IDs in primary manifest")
    for config in configs:
        if config.B <= 0 or config.N <= 0 or config.W <= 0 or config.L <= 0:
            raise ValueError(f"nonpositive structural value in {config.configuration_id}")
        if config.N % config.W != 0:
            raise ValueError(f"N % W rejection failed for {config.configuration_id}")
        if config.N > config.L:
            raise ValueError(f"N > L rejection failed for {config.configuration_id}")
    modes_by_config: dict[str, set[str]] = {}
    config_mode_keys: set[tuple[str, str]] = set()
    for run in runs:
        key = (run.structural.configuration_id, run.optimization_mode)
        if key in config_mode_keys:
            raise ValueError(f"duplicate configuration/mode key: {key}")
        config_mode_keys.add(key)
        modes_by_config.setdefault(run.structural.configuration_id, set()).add(run.optimization_mode)
        if run.sensitivity_axis != "primary":
            raise ValueError("primary manifest contains sensitivity row")
        if run.ratio_num != PRIMARY_RATIO_NUM or run.ratio_den != PRIMARY_RATIO_DEN:
            raise ValueError("primary manifest varies computation cost")
        if run.communication_ticks != PRIMARY_COMMUNICATION_TICKS:
            raise ValueError("primary manifest varies communication cost")
        if run.random_seed != PRIMARY_RANDOM_SEED:
            raise ValueError("primary manifest varies random seed")
        if run.solver_threads != PRIMARY_SOLVER_THREADS:
            raise ValueError("primary manifest varies solver thread count")
        if run.time_limit_seconds != PRIMARY_TIME_LIMIT_SECONDS:
            raise ValueError("primary manifest varies optimization budget")
        if run.optimization_mode == "schedule-only-uniform" and run.fixed_partition_source != "uniform":
            raise ValueError("primary schedule-only mode is not fixed to uniform")
        if run.optimization_mode != "schedule-only-uniform" and run.fixed_partition_source:
            raise ValueError("non-schedule primary row has a fixed partition source")
        if (
            run.optimization_mode == "partition-only"
            and run.fixed_order_partition_backend != "cpsat"
        ):
            raise ValueError("primary partition-only mode must request cpsat backend")
        if (
            run.optimization_mode != "partition-only"
            and run.fixed_order_partition_backend
        ):
            raise ValueError("non-partition primary row has a fixed-order backend")
        command = command_for_run(Path("build/release/slackpipe_cli"), Path("manifest-check"), run)
        if command[command.index("--J") + 1] != str(run.structural.W):
            raise ValueError("pipeline W is not mapped to --J")
        if command[command.index("--num-workers") + 1] != str(PRIMARY_SOLVER_THREADS):
            raise ValueError("solver_threads is not mapped to --num-workers")
        if command[command.index("--require-optimal") + 1] != "false":
            raise ValueError("primary manifest requires --require-optimal false")
    for config_id, modes in modes_by_config.items():
        if modes != set(PRIMARY_MODES):
            raise ValueError(f"primary modes mismatch for {config_id}: {sorted(modes)}")


def make_run_id(config: StructuralConfig, mode: str, sensitivity_axis: str = "primary", **kwargs: Any) -> str:
    parts = [
        config.configuration_id,
        config.B,
        config.N,
        config.W,
        config.L,
        mode,
        sensitivity_axis,
    ]
    for key in sorted(kwargs):
        parts.append(key)
        parts.append(kwargs[key])
    return stable_id(parts)


def primary_run_specs(configs: list[StructuralConfig]) -> list[RunSpec]:
    runs: list[RunSpec] = []
    for config in configs:
        for mode in PRIMARY_MODES:
            fixed_source = "uniform" if mode == "schedule-only-uniform" else ""
            runs.append(
                RunSpec(
                    run_id=make_run_id(config, mode),
                    structural=config,
                    optimization_mode=mode,
                    fixed_partition_source=fixed_source,
                    fixed_order_partition_backend="cpsat"
                    if mode == "partition-only"
                    else "",
                    ratio_num=PRIMARY_RATIO_NUM,
                    ratio_den=PRIMARY_RATIO_DEN,
                    communication_ticks=PRIMARY_COMMUNICATION_TICKS,
                    min_layers=PRIMARY_MIN_LAYERS,
                    time_limit_seconds=PRIMARY_TIME_LIMIT_SECONDS,
                    solver_threads=PRIMARY_SOLVER_THREADS,
                    random_seed=PRIMARY_RANDOM_SEED,
                )
            )
    validate_primary_manifest(configs, runs)
    return runs


def sensitivity_run_specs() -> list[RunSpec]:
    primary_configs = generate_structural_configs()
    subset = [
        config
        for config in primary_configs
        if config.W in {4, 16, 32}
        and config.stages_per_worker in {1, 4}
        and config.microbatches_per_worker in {Fraction(1, 2), Fraction(2, 1)}
        and config.L in {32, 61, 128}
    ][:24]
    runs: list[RunSpec] = []

    def add(config: StructuralConfig, mode: str, axis: str, **overrides: Any) -> None:
        fixed_source = overrides.pop("fixed_partition_source", "uniform" if mode.startswith("schedule-only") else "")
        run = RunSpec(
            run_id=make_run_id(config, mode, axis, fixed_partition_source=fixed_source, **overrides),
            structural=config,
            optimization_mode=mode,
            fixed_partition_source=fixed_source,
            fixed_order_partition_backend="cpsat"
            if mode == "partition-only"
            else "",
            ratio_num=overrides.get("ratio_num", PRIMARY_RATIO_NUM),
            ratio_den=overrides.get("ratio_den", PRIMARY_RATIO_DEN),
            communication_ticks=overrides.get("communication_ticks", PRIMARY_COMMUNICATION_TICKS),
            min_layers=PRIMARY_MIN_LAYERS,
            time_limit_seconds=overrides.get("time_limit_seconds", 60.0),
            solver_threads=overrides.get("solver_threads", PRIMARY_SOLVER_THREADS),
            random_seed=overrides.get("random_seed", PRIMARY_RANDOM_SEED),
            use_bfs_hints=overrides.get("use_bfs_hints"),
            symmetry_break_f0_fifo=overrides.get("symmetry_break_f0_fifo"),
            pressure_pruning=overrides.get("pressure_pruning", False),
            sensitivity_axis=axis,
        )
        runs.append(run)

    for config in subset:
        for mode in PRIMARY_MODES:
            add(config, mode, "baseline")
        for ratio_num, ratio_den in ((1, 1), (3, 1)):
            for mode in PRIMARY_MODES:
                add(config, mode, "compute_cost", ratio_num=ratio_num, ratio_den=ratio_den)
        for communication_ticks in (1, 4):
            for mode in PRIMARY_MODES:
                add(config, mode, "communication_cost", communication_ticks=communication_ticks)
        for random_seed in (2, 3):
            for mode in ("joint", "schedule-only-uniform"):
                add(config, mode, "random_seed", random_seed=random_seed)
        for time_limit_seconds in (60.0, 180.0):
            for mode in PRIMARY_MODES:
                add(config, mode, "time_limit", time_limit_seconds=time_limit_seconds)
        for solver_threads in (1, 4, 8):
            for mode in ("joint", "schedule-only-uniform"):
                add(config, mode, "solver_threads", solver_threads=solver_threads)
        for source in ("load-balanced", "partition-only"):
            add(
                config,
                "schedule-only-uniform",
                "schedule_partition_source",
                fixed_partition_source=source,
            )
        for use_bfs_hints in (False,):
            add(config, "joint", "hinting", use_bfs_hints=use_bfs_hints)
        for symmetry in (False,):
            add(config, "joint", "cp_sat_parameter", symmetry_break_f0_fifo=symmetry)
        add(config, "joint", "pruning", pressure_pruning=True)
    return runs


def select_pilot_configuration_ids(configs: list[StructuralConfig]) -> set[str]:
    targets = [
        (2, 1, Fraction(1, 2), 24),
        (2, 2, Fraction(1, 1), 64),
        (2, 4, Fraction(4, 1), 128),
        (4, 1, Fraction(2, 1), 24),
        (4, 4, Fraction(2, 1), 61),
        (8, 1, Fraction(2, 1), 32),
        (8, 2, Fraction(4, 1), 126),
        (8, 4, Fraction(1, 2), 128),
        (16, 2, Fraction(4, 1), 96),
        (16, 4, Fraction(1, 1), 128),
        (32, 1, Fraction(1, 2), 32),
        (32, 4, Fraction(4, 1), 128),
    ]
    by_key = {
        (config.W, config.stages_per_worker, config.microbatches_per_worker, config.L): config
        for config in configs
    }
    selected = []
    for target in targets:
        config = by_key.get(target)
        if config is None:
            raise ValueError(f"pilot target is not in primary manifest: {target}")
        selected.append(config.configuration_id)
    if len(set(selected)) != 12:
        raise ValueError("pilot selector produced duplicate configurations")
    return set(selected)


def select_pilot_runs(configs: list[StructuralConfig], runs: list[RunSpec]) -> list[RunSpec]:
    pilot_config_ids = select_pilot_configuration_ids(configs)
    selected = [run for run in runs if run.structural.configuration_id in pilot_config_ids]
    if len(selected) != 36:
        raise ValueError(f"pilot requires exactly 36 rows, selected {len(selected)}")
    if {run.optimization_mode for run in selected} != set(PRIMARY_MODES):
        raise ValueError("pilot does not cover every mode")
    if {run.structural.microbatches_per_worker for run in selected} != set(PRIMARY_MICROBATCH_RATIOS):
        raise ValueError("pilot does not cover every B/W value")
    if {run.structural.stages_per_worker for run in selected} != set(PRIMARY_VIRTUAL_STAGES_PER_WORKER):
        raise ValueError("pilot does not cover every N/W value")
    if 2 not in {run.structural.W for run in selected} or max(PRIMARY_W_VALUES) not in {run.structural.W for run in selected}:
        raise ValueError("pilot does not cover both minimum and maximum W")
    if not ({24, 64, 128} <= {run.structural.L for run in selected}):
        raise ValueError("pilot does not cover small, middle, and large L")
    return selected


def run_to_manifest_row(run: RunSpec) -> dict[str, Any]:
    config = run.structural
    return {
        "run_id": run.run_id,
        "configuration_id": config.configuration_id,
        "depth_set": config.depth_set,
        "B": config.B,
        "N": config.N,
        "W": config.W,
        "L": config.L,
        "microbatches_per_worker": fraction_text(config.microbatches_per_worker),
        "stages_per_worker": config.stages_per_worker,
        "layers_per_stage": fraction_text(config.layers_per_stage),
        "B_over_W": fraction_text(config.microbatches_per_worker),
        "N_over_W": config.stages_per_worker,
        "L_over_N": fraction_text(config.layers_per_stage),
        "optimization_mode": run.optimization_mode,
        "fixed_partition_source": run.fixed_partition_source,
        "fixed_order_partition_backend": run.fixed_order_partition_backend,
        "pipeline_workers": config.W,
        "solver_threads": run.solver_threads,
        "time_limit_seconds": run.time_limit_seconds,
        "random_seed": run.random_seed,
        "ratio_num": run.ratio_num,
        "ratio_den": run.ratio_den,
        "communication_ticks": run.communication_ticks,
        "min_layers": run.min_layers,
        "sensitivity_axis": run.sensitivity_axis,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def grouped_manifest_count_rows(manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in ("W", "B", "N", "L", "B_over_W", "N_over_W", "optimization_mode"):
        counts = Counter(str(row[field]) for row in manifest_rows)
        for value, count in sorted(
            counts.items(),
            key=lambda item: (
                float(Fraction(item[0])) if field in {"B_over_W", "N_over_W"} else float(item[0]) if item[0].isdigit() else item[0],
                item[0],
            ),
        ):
            rows.append({"group": field, "value": value, "rows": count})
    return rows


def write_manifest(output_dir: Path, runs: list[RunSpec]) -> None:
    manifest_rows = [run_to_manifest_row(run) for run in runs]
    write_csv(output_dir / "manifest.csv", manifest_rows, MANIFEST_COLUMNS)
    write_csv(output_dir / "config_manifest.csv", manifest_rows, MANIFEST_COLUMNS)
    write_csv(
        output_dir / "manifest_counts.csv",
        grouped_manifest_count_rows(manifest_rows),
        ["group", "value", "rows"],
    )
    configs: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        configs[row["configuration_id"]] = {
            key: row[key]
            for key in (
                "configuration_id",
                "depth_set",
                "B",
                "N",
                "W",
                "L",
                "microbatches_per_worker",
                "stages_per_worker",
                "layers_per_stage",
                "B_over_W",
                "N_over_W",
                "L_over_N",
            )
        }
    write_csv(
        output_dir / "configurations.csv",
        list(configs.values()),
        [
            "configuration_id",
            "depth_set",
            "B",
            "N",
            "W",
            "L",
            "microbatches_per_worker",
            "stages_per_worker",
            "layers_per_stage",
            "B_over_W",
            "N_over_W",
            "L_over_N",
        ],
    )
    metadata = {
        "manifest_rows": len(runs),
        "configurations": len(configs),
        "primary_expected_configurations": 520,
        "primary_expected_runs": 1560,
        "derive_schedule_only_best": PRIMARY_DERIVE_SCHEDULE_ONLY_BEST,
        "modes": sorted({run.optimization_mode for run in runs}),
    }
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def git_dirty() -> bool | None:
    try:
        output = subprocess.check_output(["git", "status", "--short"], text=True)
    except Exception:
        return None
    return bool(output.strip())


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as error:
        return f"unavailable: {error}"


def cache_value(cache_path: Path, key: str) -> str:
    if not cache_path.exists():
        return "unknown"
    for line in cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(f"{key}:"):
            _, value = line.split("=", 1)
            return value
    return "unknown"


def ortools_version() -> str:
    version_file = Path("/opt/or-tools/lib/cmake/ortools/ortoolsConfigVersion.cmake")
    if version_file.exists():
        for line in version_file.read_text(encoding="utf-8", errors="replace").splitlines():
            text = line.strip()
            if text.startswith("set(PACKAGE_VERSION"):
                return text.split('"')[1]
    return "unknown"


def lscpu_field(name: str) -> str:
    output = command_output(["lscpu"])
    for line in output.splitlines():
        if line.startswith(name + ":"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def logical_processor_count() -> int:
    try:
        return int(os.cpu_count() or 0)
    except Exception:
        return 0


def physical_core_count() -> str:
    cores = lscpu_field("Core(s) per socket")
    sockets = lscpu_field("Socket(s)")
    try:
        return str(int(cores) * int(sockets))
    except ValueError:
        return "unknown"


def memory_size() -> str:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                return " ".join(line.split()[1:3])
    return "unknown"


def write_environment_report(output_dir: Path, cli: Path, runner_command: list[str]) -> None:
    logical = logical_processor_count()
    if logical < PRIMARY_SOLVER_THREADS:
        raise RuntimeError(f"primary ablation requires at least {PRIMARY_SOLVER_THREADS} logical processors, found {logical}")
    cache_path = cli.parent / "CMakeCache.txt"
    build_type = cache_value(cache_path, "CMAKE_BUILD_TYPE")
    ortools_enabled = cache_value(cache_path, "SLACKPIPE_ENABLE_ORTOOLS")
    if build_type != "Release":
        raise RuntimeError(f"primary ablation requires a Release build, found {build_type} for {cli}")
    if ortools_enabled != "ON":
        raise RuntimeError(f"primary ablation requires OR-Tools enabled, found {ortools_enabled} for {cli}")
    compiler = cache_value(cache_path, "CMAKE_CXX_COMPILER")
    compiler_version = command_output([compiler, "--version"]).splitlines()[0] if compiler != "unknown" else command_output(["c++", "--version"]).splitlines()[0]
    lines = [
        "SlackPipe Primary Ablation Environment Report",
        f"recorded_utc: {utc_now_text()}",
        f"cpu_model: {lscpu_field('Model name')}",
        f"physical_core_count: {physical_core_count()}",
        f"logical_processor_count: {logical}",
        f"memory_size: {memory_size()}",
        f"operating_system: {platform.platform()}",
        f"compiler: {compiler}",
        f"compiler_version: {compiler_version}",
        f"cmake_build_type: {build_type}",
        f"ortools_enabled: {ortools_enabled}",
        f"ortools_version: {ortools_version()}",
        f"python_version: {sys.version.split()[0]}",
        f"git_commit: {git_commit()}",
        f"dirty_tree: {git_dirty()}",
        f"derive_schedule_only_best: {PRIMARY_DERIVE_SCHEDULE_ONLY_BEST}",
        f"runner_command: {shlex.join(runner_command)}",
    ]
    (output_dir / "environment_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def command_for_run(cli: Path, output_prefix: Path, run: RunSpec) -> list[str]:
    config = run.structural
    if run.optimization_mode == "joint":
        algorithm_args = ["--algorithm", "joint-unrestricted-no-overlap"]
    elif run.optimization_mode == "partition-only":
        algorithm_args = [
            "--algorithm",
            "partition-only-fixed-order",
            "--fixed-order-partition-backend",
            run.fixed_order_partition_backend or "cpsat",
        ]
    elif run.optimization_mode == "schedule-only-uniform":
        source = run.fixed_partition_source or "uniform"
        algorithm_args = [
            "--algorithm",
            "schedule-only-uniform",
            "--fixed-partition-source",
            source,
        ]
    else:
        raise ValueError(f"unknown optimization mode: {run.optimization_mode}")

    command = [
        str(cli),
        "--B",
        str(config.B),
        "--N",
        str(config.N),
        "--J",
        str(config.W),
        "--L",
        str(config.L),
        "--min-layers",
        str(run.min_layers),
        "--ratio-num",
        str(run.ratio_num),
        "--ratio-den",
        str(run.ratio_den),
        "--communication",
        str(run.communication_ticks),
        *algorithm_args,
        "--time-limit-seconds",
        str(run.time_limit_seconds),
        "--num-workers",
        str(run.solver_threads),
        "--random-seed",
        str(run.random_seed),
        "--require-optimal",
        "false",
        "--output-prefix",
        str(output_prefix),
    ]
    if run.use_bfs_hints is not None:
        command.extend(["--use-bfs-hints", "true" if run.use_bfs_hints else "false"])
    if run.symmetry_break_f0_fifo is not None:
        command.extend(
            [
                "--symmetry-break-f0-fifo",
                "true" if run.symmetry_break_f0_fifo else "false",
            ]
        )
    if run.pressure_pruning:
        command.append("--enable-pressure-pruning")
    return command


def normalize_tick_list(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return str(value)


def normalize_float_list(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ";".join(f"{float(item):.12g}" for item in value)
    return str(value)


def normalize_order(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        workers = []
        for index, worker in enumerate(value):
            if isinstance(worker, list):
                workers.append(f"w{index}:" + " ".join(str(item) for item in worker))
            else:
                workers.append(f"w{index}:{worker}")
        return "|".join(workers)
    return str(value)


def operation_name(stages: int, microbatch: int, chain_index: int) -> str:
    backward = chain_index >= stages
    stage = chain_index if chain_index < stages else 2 * stages - 1 - chain_index
    return f"{'B' if backward else 'F'}{stage}(b{microbatch})"


def worker_for_chain_index(stages: int, workers: int, chain_index: int) -> int:
    stage = chain_index if chain_index < stages else 2 * stages - 1 - chain_index
    return stage % workers


def canonical_worker_order(B: int, N: int, W: int) -> str:
    orders: list[list[tuple[tuple[int, int, int], str]]] = [[] for _ in range(W)]
    for microbatch in range(B):
        for chain_index in range(2 * N):
            worker = worker_for_chain_index(N, W, chain_index)
            rank = (microbatch + chain_index, -chain_index, microbatch)
            orders[worker].append((rank, operation_name(N, microbatch, chain_index)))
    pieces = []
    for worker, entries in enumerate(orders):
        entries.sort(key=lambda item: item[0])
        pieces.append(f"w{worker}:" + " ".join(name for _, name in entries))
    return "|".join(pieces)


def solver_json_path(output_prefix: Path) -> Path:
    return output_prefix.with_suffix(".json")


def diagnostic_json_path(output_prefix: Path) -> Path:
    return output_prefix.with_suffix(".diagnostic.json")


def last_lifecycle_phase(stderr_text: str) -> str:
    phase = ""
    for line in stderr_text.splitlines():
        if "SLACKPIPE_LIFECYCLE" not in line:
            continue
        for token in line.split():
            if token.startswith("phase="):
                phase = token.split("=", 1)[1]
                break
    return phase


def completed_run_ids(results_jsonl: Path, rerun_unsuccessful: bool) -> set[str]:
    completed: set[str] = set()
    if not results_jsonl.exists():
        return completed
    with results_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("completed"):
                continue
            if rerun_unsuccessful and row.get("solver_status") not in {"OPTIMAL", "FEASIBLE"}:
                continue
            if row.get("sensitivity_axis") == "primary":
                if row.get("primary_protocol_validation_passed") not in {True, "True", "true", "1", 1}:
                    continue
                if int(row.get("budget_policy_version", 0) or 0) != BUDGET_POLICY_VERSION:
                    continue
                if row.get("optimization_mode") == "partition-only" and row.get("solver_status") in {"OPTIMAL", "FEASIBLE"}:
                    if row.get("partition_order_validation_passed") not in {True, "True", "true", "1", 1}:
                        continue
                if row.get("optimization_mode") == "schedule-only-uniform" and row.get("solver_status") in {"OPTIMAL", "FEASIBLE"}:
                    if row.get("schedule_partition_validation_passed") not in {True, "True", "true", "1", 1}:
                        continue
            completed.add(str(row.get("run_id", "")))
    return completed


def row_from_solver_json(run: RunSpec, payload: dict[str, Any], output_json: Path) -> dict[str, Any]:
    config = run.structural
    canonical = payload.get("canonical_result", {})
    if not isinstance(canonical, dict):
        canonical = {}
    activation_metrics = canonical.get("activation_memory_metrics", {})
    if not isinstance(activation_metrics, dict):
        activation_metrics = {}
    activation_global = activation_metrics.get("global", {})
    if not isinstance(activation_global, dict):
        activation_global = {}
    activation_ratio = canonical.get("activation_peak_ratio_to_uniform", {})
    if not isinstance(activation_ratio, dict):
        activation_ratio = {}
    makespan = payload.get("simulated_iteration_time")
    if makespan is None:
        makespan = payload.get("makespan_ticks", payload.get("makespan", ""))
    incumbent = payload.get("makespan_ticks", makespan)
    best_bound = payload.get("best_bound_ticks", payload.get("best_objective_bound", ""))
    gap = ""
    try:
        makespan_float = float(makespan)
        bound_float = float(best_bound)
        if makespan_float > 0 and bound_float > 0:
            gap = max(0.0, (makespan_float - bound_float) / makespan_float)
    except (TypeError, ValueError):
        pass
    return {
        "solver_status": payload.get("status", "UNKNOWN"),
        "schema_version": payload.get("schema_version", canonical.get("schema_version", "")),
        "budget_policy_version": payload.get(
            "budget_policy_version",
            canonical.get("budget_policy_version", ""),
        ),
        "evaluation_method_version": payload.get(
            "evaluation_method_version",
            canonical.get("evaluation_method_version", ""),
        ),
        "requested_method": canonical.get("requested_method", ""),
        "canonical_method": payload.get("canonical_method", canonical.get("canonical_method", "")),
        "method_contract_hash": canonical.get("method_contract_hash", ""),
        "fixed_schedule_rule": canonical.get("fixed_schedule_rule", ""),
        "uniform_partition_rule": canonical.get("uniform_partition_rule", ""),
        "actual_solver_path": payload.get("actual_solver_path", canonical.get("actual_solver_path", "")),
        "partition_decision": canonical.get("partition_decision", ""),
        "schedule_decision": canonical.get("schedule_decision", ""),
        "full_partition_fixed": canonical.get("full_partition_fixed", ""),
        "worker_aggregate_loads_fixed": canonical.get("worker_aggregate_loads_fixed", ""),
        "partition_optimized": canonical.get("partition_optimized", ""),
        "schedule_optimized": canonical.get("schedule_optimized", ""),
        "predecessor_candidate_restriction_requested": canonical.get("predecessor_candidate_restriction_requested", ""),
        "predecessor_candidate_restriction_active": canonical.get("predecessor_candidate_restriction_active", ""),
        "predecessor_candidate_rule": canonical.get("predecessor_candidate_rule", ""),
        "fixed_order_partition_backend_requested": canonical.get(
            "fixed_order_partition_backend_requested",
            payload.get("fixed_order_partition_backend_requested", ""),
        ),
        "fixed_order_partition_backend_effective": canonical.get(
            "fixed_order_partition_backend_effective",
            payload.get("fixed_order_partition_backend_effective", ""),
        ),
        "solver_status_raw": canonical.get("solver_status_raw", ""),
        "reported_status": canonical.get("reported_status", ""),
        "fallback_reason": canonical.get("fallback_reason", ""),
        "returned_solution_source": canonical.get("returned_solution_source", ""),
        "communication_model": canonical.get("communication_model", ""),
        "activation_model": canonical.get("activation_model", ""),
        "activation_units_per_layer": canonical.get("activation_units_per_layer", ""),
        "explicit_stage_activation_units": normalize_tick_list(
            canonical.get("explicit_stage_activation_units")
        ),
        "activation_cap_mode": canonical.get("activation_cap_mode", ""),
        "activation_cap_units_per_worker": normalize_tick_list(
            canonical.get("activation_cap_units_per_worker")
        ),
        "activation_cap_enforced": canonical.get("activation_cap_enforced", ""),
        "activation_cap_enforcement_requested": canonical.get(
            "activation_cap_enforcement_requested", ""
        ),
        "activation_cap_enforcement_mode": canonical.get(
            "activation_cap_enforcement_mode", ""
        ),
        "activation_cap_solver_supported": canonical.get(
            "activation_cap_solver_supported", ""
        ),
        "activation_cap_solver_support_level": canonical.get(
            "activation_cap_solver_support_level", ""
        ),
        "activation_cap_enforced_in_solver": canonical.get(
            "activation_cap_enforced_in_solver", ""
        ),
        "activation_cap_constraints_added": canonical.get(
            "activation_cap_constraints_added", ""
        ),
        "activation_retained_interval_count": canonical.get(
            "activation_retained_interval_count", ""
        ),
        "activation_cumulative_constraint_count": canonical.get(
            "activation_cumulative_constraint_count", ""
        ),
        "activation_variable_demand_count": canonical.get(
            "activation_variable_demand_count", ""
        ),
        "activation_fixed_demand_count": canonical.get(
            "activation_fixed_demand_count", ""
        ),
        "activation_constraint_build_runtime_seconds": canonical.get(
            "activation_constraint_build_runtime_seconds", ""
        ),
        "incumbent_rejected_for_activation_cap": canonical.get(
            "incumbent_rejected_for_activation_cap", ""
        ),
        "activation_model_validation_agreement": canonical.get(
            "activation_model_validation_agreement", ""
        ),
        "activation_model_disagreement_details": canonical.get(
            "activation_model_disagreement_details", ""
        ),
        "activation_cap_formulation_version": canonical.get(
            "activation_cap_formulation_version", ""
        ),
        "activation_cap_satisfied": canonical.get("activation_cap_satisfied", ""),
        "maximum_worker_peak_activation_units": activation_global.get(
            "maximum_worker_peak_activation_units", ""
        ),
        "global_simultaneous_peak_activation_units": activation_global.get(
            "peak_simultaneous_activation_units_across_workers", ""
        ),
        "activation_peak_ratio_to_uniform": activation_ratio.get(
            "maximum_worker_peak_units_ratio", ""
        ),
        "activation_cap_derivation_hash": canonical.get(
            "activation_cap_derivation_hash", ""
        ),
        "result_validation_passed": canonical.get("result_validation_passed", ""),
        "result_validation_error": canonical.get("result_validation_error", ""),
        "executable_name": canonical.get("executable_name", ""),
        "proven_optimal": bool(payload.get("proven_optimal", False)),
        "incumbent_objective": incumbent,
        "best_objective_bound": best_bound,
        "replayed_makespan": makespan,
        "relative_optimality_gap": gap,
        "utilization": payload.get("pipeline_utilization", ""),
        "useful_work": payload.get("total_useful_work", ""),
        "busy_time": normalize_tick_list(payload.get("per_worker_busy_time")),
        "idle_time": normalize_tick_list(payload.get("per_worker_idle_time")),
        "maximum_worker_load": payload.get("maximum_worker_load", ""),
        "fill_time": payload.get("pipeline_fill_time", ""),
        "drain_time": payload.get("pipeline_drain_time", ""),
        "communication_blocked_time": payload.get("communication_blocked_time", ""),
        "incumbent_budget_seconds": payload.get("incumbent_budget_seconds", ""),
        "incumbent_model_build_seconds": payload.get("incumbent_model_build_seconds", ""),
        "incumbent_solve_seconds": payload.get("incumbent_solve_seconds", ""),
        "incumbent_status": payload.get("incumbent_status", ""),
        "joint_budget_seconds": payload.get("joint_budget_seconds", ""),
        "joint_model_build_seconds": payload.get("joint_model_build_seconds", ""),
        "joint_solve_seconds": payload.get("joint_solve_seconds", ""),
        "joint_status": payload.get("joint_status", ""),
        "incumbent_method_requested": payload.get("incumbent_method_requested", ""),
        "incumbent_method_effective": payload.get("incumbent_method_effective", ""),
        "bfs_incumbent_method_requested": payload.get("bfs_incumbent_method_requested", ""),
        "bfs_incumbent_method_effective": payload.get("bfs_incumbent_method_effective", ""),
        "incumbent_source": payload.get("incumbent_source", ""),
        "incumbent_feasible": payload.get("incumbent_feasible", ""),
        "incumbent_primary_objective": payload.get("incumbent_primary_objective", ""),
        "incumbent_hybrid_min_slack": payload.get("incumbent_hybrid_min_slack", ""),
        "incumbent_baseline_primary_objective": payload.get("incumbent_baseline_primary_objective", ""),
        "incumbent_baseline_hybrid_min_slack": payload.get("incumbent_baseline_hybrid_min_slack", ""),
        "incumbent_improved_over_baseline": payload.get("incumbent_improved_over_baseline", ""),
        "incumbent_hybrid_stage_scores": normalize_float_list(payload.get("incumbent_hybrid_stage_scores")),
        "incumbent_hybrid_bottleneck_stages": normalize_tick_list(payload.get("incumbent_hybrid_bottleneck_stages")),
        "horizon_source": payload.get("horizon_source", ""),
        "hint_budget_seconds": payload.get("hint_budget_seconds", ""),
        "hint_elapsed_seconds": payload.get("hint_elapsed_seconds", ""),
        "hint_iterations": payload.get("hint_iterations", ""),
        "hint_candidates_generated": payload.get("hint_candidates_generated", ""),
        "hint_candidates_simulated": payload.get("hint_candidates_simulated", ""),
        "hint_partition_moves_accepted": payload.get("hint_partition_moves_accepted", ""),
        "hint_interleaving_moves_accepted": payload.get("hint_interleaving_moves_accepted", ""),
        "hint_deadline_reached": payload.get("hint_deadline_reached", ""),
        "hint_termination_reason": payload.get("hint_termination_reason", ""),
        "hints_requested": payload.get("hints_requested", ""),
        "hints_effective": payload.get("hints_effective", ""),
        "hint_source": payload.get("hint_source", ""),
        "hint_scope": payload.get("hint_scope", ""),
        "hint_complete_for_basic_model": payload.get("hint_complete_for_basic_model", ""),
        "hint_complete_for_full_model": payload.get("hint_complete_for_full_model", ""),
        "hinted_layer_variable_count": payload.get("hinted_layer_variable_count", ""),
        "hinted_operation_variable_count": payload.get("hinted_operation_variable_count", ""),
        "hinted_scalar_variable_count": payload.get("hinted_scalar_variable_count", ""),
        "hinted_auxiliary_variable_count": payload.get("hinted_auxiliary_variable_count", ""),
        "hinted_total_variable_count": payload.get("hinted_total_variable_count", ""),
        "auxiliary_variable_count": payload.get("auxiliary_variable_count", ""),
        "fallback_available": payload.get("fallback_available", ""),
        "fallback_source": payload.get("fallback_source", ""),
        "solution_source": payload.get("solution_source", ""),
        "fallback_used": payload.get("fallback_used", ""),
        "final_split": normalize_tick_list(payload.get("split")),
        "final_worker_local_order": normalize_order(payload.get("worker_local_operation_order")),
        "uniform_split": normalize_tick_list(uniform_split(config.L, config.N, run.min_layers)),
        "output_json_path": str(output_json),
    }


def base_result_row(run: RunSpec, command: list[str], commit: str | None, dirty: bool | None) -> dict[str, Any]:
    row = run_to_manifest_row(run)
    row.update(
        {
            "pipeline_workers": run.structural.W,
            "command": shlex.join(command),
            "return_code": "",
            "completed": False,
            "optimization_wall_time_seconds": "",
            "git_commit": commit,
            "git_dirty": dirty,
            "solver_status": "NOT_RUN",
            "schema_version": "",
            "budget_policy_version": "",
            "requested_method": "",
            "canonical_method": "",
            "actual_solver_path": "",
            "partition_decision": "",
            "schedule_decision": "",
            "full_partition_fixed": "",
            "worker_aggregate_loads_fixed": "",
            "partition_optimized": "",
            "schedule_optimized": "",
            "predecessor_candidate_restriction_requested": "",
            "predecessor_candidate_restriction_active": "",
            "predecessor_candidate_rule": "",
            "fixed_order_partition_backend_requested": "",
            "fixed_order_partition_backend_effective": "",
            "solver_status_raw": "",
            "reported_status": "",
            "fallback_reason": "",
            "returned_solution_source": "",
            "communication_model": "",
            "activation_model": "",
            "activation_units_per_layer": "",
            "explicit_stage_activation_units": "",
            "activation_cap_mode": "",
            "activation_cap_units_per_worker": "",
            "activation_cap_enforced": "",
            "activation_cap_enforcement_requested": "",
            "activation_cap_enforcement_mode": "",
            "activation_cap_solver_supported": "",
            "activation_cap_solver_support_level": "",
            "activation_cap_enforced_in_solver": "",
            "activation_cap_constraints_added": "",
            "activation_retained_interval_count": "",
            "activation_cumulative_constraint_count": "",
            "activation_variable_demand_count": "",
            "activation_fixed_demand_count": "",
            "activation_constraint_build_runtime_seconds": "",
            "incumbent_rejected_for_activation_cap": "",
            "activation_model_validation_agreement": "",
            "activation_model_disagreement_details": "",
            "activation_cap_formulation_version": "",
            "activation_cap_satisfied": "",
            "maximum_worker_peak_activation_units": "",
            "global_simultaneous_peak_activation_units": "",
            "activation_peak_ratio_to_uniform": "",
            "activation_cap_derivation_hash": "",
            "result_validation_passed": "",
            "result_validation_error": "",
            "executable_name": "",
            "proven_optimal": False,
            "incumbent_objective": "",
            "best_objective_bound": "",
            "replayed_makespan": "",
            "relative_optimality_gap": "",
            "utilization": "",
            "useful_work": "",
            "busy_time": "",
            "idle_time": "",
            "maximum_worker_load": "",
            "fill_time": "",
            "drain_time": "",
            "communication_blocked_time": "",
            "incumbent_budget_seconds": "",
            "incumbent_model_build_seconds": "",
            "incumbent_solve_seconds": "",
            "incumbent_status": "",
            "joint_budget_seconds": "",
            "joint_model_build_seconds": "",
            "joint_solve_seconds": "",
            "joint_status": "",
            "incumbent_method_requested": "",
            "incumbent_method_effective": "",
            "bfs_incumbent_method_requested": "",
            "bfs_incumbent_method_effective": "",
            "incumbent_source": "",
            "incumbent_feasible": "",
            "incumbent_primary_objective": "",
            "incumbent_hybrid_min_slack": "",
            "incumbent_baseline_primary_objective": "",
            "incumbent_baseline_hybrid_min_slack": "",
            "incumbent_improved_over_baseline": "",
            "incumbent_hybrid_stage_scores": "",
            "incumbent_hybrid_bottleneck_stages": "",
            "horizon_source": "",
            "hint_budget_seconds": "",
            "hint_elapsed_seconds": "",
            "hint_iterations": "",
            "hint_candidates_generated": "",
            "hint_candidates_simulated": "",
            "hint_partition_moves_accepted": "",
            "hint_interleaving_moves_accepted": "",
            "hint_deadline_reached": "",
            "hint_termination_reason": "",
            "hints_requested": "",
            "hints_effective": "",
            "hint_source": "",
            "hint_scope": "",
            "hint_complete_for_basic_model": "",
            "hint_complete_for_full_model": "",
            "hinted_layer_variable_count": "",
            "hinted_operation_variable_count": "",
            "hinted_scalar_variable_count": "",
            "hinted_auxiliary_variable_count": "",
            "hinted_total_variable_count": "",
            "auxiliary_variable_count": "",
            "fallback_available": "",
            "fallback_source": "",
            "solution_source": "",
            "fallback_used": "",
            "final_split": "",
            "final_worker_local_order": "",
            "uniform_split": normalize_tick_list(
                uniform_split(run.structural.L, run.structural.N, run.min_layers)
            ),
            "output_json_path": "",
            "start_timestamp_utc": "",
            "end_timestamp_utc": "",
            "stdout_log_path": "",
            "stderr_log_path": "",
            "last_lifecycle_phase": "",
            "stdout": "",
            "stderr": "",
            "failure": "",
            "partition_order_validation_passed": "",
            "schedule_partition_validation_passed": "",
            "primary_protocol_validation_passed": "",
        }
    )
    return row


def validate_result_row(row: dict[str, Any]) -> None:
    row["partition_order_validation_passed"] = ""
    row["schedule_partition_validation_passed"] = ""
    row["primary_protocol_validation_passed"] = ""
    mode = row["optimization_mode"]
    successful = row["solver_status"] in {"OPTIMAL", "FEASIBLE"}
    if mode == "partition-only":
        row["partition_order_validation_passed"] = (
            not successful
            or row["final_worker_local_order"]
            == canonical_worker_order(int(row["B"]), int(row["N"]), int(row["W"]))
        )
        if successful and not row["partition_order_validation_passed"]:
            raise RuntimeError(f"partition-only changed canonical order for {row['run_id']}")
    if mode == "schedule-only-uniform":
        row["schedule_partition_validation_passed"] = row["final_split"] == row["uniform_split"]
        if successful and not row["schedule_partition_validation_passed"]:
            raise RuntimeError(f"schedule-only changed uniform split for {row['run_id']}")
    if successful:
        makespan = float(row["replayed_makespan"])
        useful = float(row["useful_work"])
        workers = float(row["W"])
        utilization = float(row["utilization"])
        expected_utilization = useful / (workers * makespan)
        if abs(utilization - expected_utilization) > 1e-6:
            raise RuntimeError(f"utilization mismatch for {row['run_id']}")
        incumbent = row.get("incumbent_objective", "")
        if incumbent not in {"", None} and abs(float(incumbent) - makespan) > 1e-6:
            raise RuntimeError(f"replayed makespan does not match incumbent objective for {row['run_id']}")
        if mode in {"joint", "schedule-only-uniform"}:
            fallback_used = row.get("fallback_used") in {True, "True", "true", "1", 1}
            fallback_available = row.get("fallback_available") in {True, "True", "true", "1", 1}
            hints_requested = row.get("hints_requested") in {True, "True", "true", "1", 1}
            hints_effective = row.get("hints_effective") in {True, "True", "true", "1", 1}
            solution_source = str(row.get("solution_source", ""))
            joint_status = str(row.get("joint_status", ""))
            if fallback_used and solution_source != "bfs_incumbent_fallback":
                raise RuntimeError(f"fallback provenance mismatch for {row['run_id']}")
            if solution_source == "bfs_incumbent_fallback" and not fallback_used:
                raise RuntimeError(f"fallback flag mismatch for {row['run_id']}")
            if solution_source.startswith("joint_cpsat") and joint_status not in {"OPTIMAL", "FEASIBLE"}:
                raise RuntimeError(f"joint solution source has non-feasible joint status for {row['run_id']}")
            if fallback_used and not fallback_available:
                raise RuntimeError(f"fallback used without available fallback for {row['run_id']}")
            if hints_effective and not hints_requested:
                raise RuntimeError(f"hints effective despite not being requested for {row['run_id']}")
            if hints_requested and not hints_effective:
                raise RuntimeError(f"hints requested but not effective for {row['run_id']}")
            if row["sensitivity_axis"] == "hinting" and (hints_requested or hints_effective):
                raise RuntimeError(f"no-hint sensitivity row used hints for {row['run_id']}")
    if row["sensitivity_axis"] == "primary":
        protocol_ok = (
            int(row["pipeline_workers"]) == int(row["W"])
            and int(row["solver_threads"]) == PRIMARY_SOLVER_THREADS
            and float(row["time_limit_seconds"]) == PRIMARY_TIME_LIMIT_SECONDS
            and int(row.get("budget_policy_version", 0) or 0) == BUDGET_POLICY_VERSION
            and int(row["random_seed"]) == PRIMARY_RANDOM_SEED
            and int(row["ratio_num"]) == PRIMARY_RATIO_NUM
            and int(row["ratio_den"]) == PRIMARY_RATIO_DEN
            and int(row["communication_ticks"]) == PRIMARY_COMMUNICATION_TICKS
            and mode in PRIMARY_MODES
            and mode != "schedule-only-best"
        )
        if mode == "schedule-only-uniform":
            protocol_ok = protocol_ok and row["fixed_partition_source"] == "uniform"
        else:
            protocol_ok = protocol_ok and row["fixed_partition_source"] == ""
        row["primary_protocol_validation_passed"] = protocol_ok
        if not protocol_ok:
            raise RuntimeError(f"primary protocol validation failed for {row['run_id']}")


def run_one(cli: Path, run: RunSpec, output_dir: Path, commit: str | None, dirty: bool | None, grace_seconds: float) -> dict[str, Any]:
    run_dir = output_dir / "run_outputs" / run.structural.configuration_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = run_dir / run.optimization_mode
    if run.fixed_partition_source and run.fixed_partition_source != "uniform":
        output_prefix = run_dir / f"{run.optimization_mode}_{run.fixed_partition_source}"
    command = command_for_run(cli, output_prefix, run)
    row = base_result_row(run, command, commit, dirty)
    stdout_log = output_prefix.with_suffix(".stdout.log")
    stderr_log = output_prefix.with_suffix(".stderr.log")
    row["stdout_log_path"] = str(stdout_log)
    row["stderr_log_path"] = str(stderr_log)
    row["start_timestamp_utc"] = utc_now_text()
    started = time.monotonic()
    try:
        with stdout_log.open("w", encoding="utf-8") as stdout_handle, stderr_log.open("w", encoding="utf-8") as stderr_handle:
            completed = subprocess.run(
                command,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                timeout=run.time_limit_seconds + grace_seconds,
            )
        row["return_code"] = completed.returncode
        row["stdout"] = stdout_log.read_text(encoding="utf-8", errors="replace").strip()
        row["stderr"] = stderr_log.read_text(encoding="utf-8", errors="replace").strip()
        row["last_lifecycle_phase"] = last_lifecycle_phase(row["stderr"])
        output_json = solver_json_path(output_prefix)
        diagnostic_json = diagnostic_json_path(output_prefix)
        if output_json.exists():
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            row.update(row_from_solver_json(run, payload, output_json))
        elif diagnostic_json.exists():
            payload = json.loads(diagnostic_json.read_text(encoding="utf-8"))
            row["solver_status"] = payload.get("status", "ERROR")
            row["failure"] = payload.get("message", "diagnostic JSON written without normal result")
            row["output_json_path"] = str(diagnostic_json)
        elif completed.returncode == 0:
            row["solver_status"] = "MISSING_OUTPUT"
            row["failure"] = "solver returned success but did not write JSON output"
        else:
            row["solver_status"] = "ERROR"
            row["failure"] = row["stderr"] or row["stdout"] or "solver exited nonzero"
    except subprocess.TimeoutExpired as error:
        row["solver_status"] = "ERROR"
        row["return_code"] = ""
        row["stdout"] = stdout_log.read_text(encoding="utf-8", errors="replace").strip() if stdout_log.exists() else ""
        row["stderr"] = stderr_log.read_text(encoding="utf-8", errors="replace").strip() if stderr_log.exists() else ""
        row["last_lifecycle_phase"] = last_lifecycle_phase(row["stderr"])
        row["failure"] = f"process exceeded timeout of {run.time_limit_seconds + grace_seconds} seconds"
    except Exception as error:
        row["solver_status"] = "ERROR"
        row["failure"] = str(error)
        row["stdout"] = stdout_log.read_text(encoding="utf-8", errors="replace").strip() if stdout_log.exists() else row["stdout"]
        row["stderr"] = stderr_log.read_text(encoding="utf-8", errors="replace").strip() if stderr_log.exists() else row["stderr"]
        row["last_lifecycle_phase"] = last_lifecycle_phase(row["stderr"])
    row["optimization_wall_time_seconds"] = time.monotonic() - started
    row["end_timestamp_utc"] = utc_now_text()
    row["completed"] = True
    validate_result_row(row)
    return row


def append_result(output_dir: Path, row: dict[str, Any]) -> None:
    jsonl_path = output_dir / "results.jsonl"
    for csv_path in (output_dir / "results.csv", output_dir / "raw_results.csv"):
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def latest_rows_from_jsonl(jsonl: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("completed"):
            latest[str(row.get("run_id"))] = row
    return list(latest.values())


def write_completeness_matrix(output_dir: Path, runs: list[RunSpec], seen: dict[str, dict[str, Any]]) -> None:
    by_config: dict[str, dict[str, Any]] = {}
    for run in runs:
        config = run.structural
        row = by_config.setdefault(
            config.configuration_id,
            {
                "configuration_id": config.configuration_id,
                "B": config.B,
                "N": config.N,
                "W": config.W,
                "L": config.L,
                "B_over_W": fraction_text(config.microbatches_per_worker),
                "N_over_W": config.stages_per_worker,
                "L_over_N": fraction_text(config.layers_per_stage),
            },
        )
        result = seen.get(run.run_id)
        mode = run.optimization_mode
        row[f"{mode}_present"] = result is not None
        row[f"{mode}_status"] = "" if result is None else result.get("solver_status", "")
        row[f"{mode}_successful"] = result is not None and result.get("solver_status") in {"OPTIMAL", "FEASIBLE"}
    fields = [
        "configuration_id",
        "B",
        "N",
        "W",
        "L",
        "B_over_W",
        "N_over_W",
        "L_over_N",
    ]
    for mode in PRIMARY_MODES:
        fields.extend([f"{mode}_present", f"{mode}_status", f"{mode}_successful"])
    write_csv(output_dir / "completeness_matrix.csv", list(by_config.values()), fields)


def validate_completed_results(output_dir: Path, runs: list[RunSpec], expected_run_ids: set[str] | None = None) -> dict[str, Any]:
    latest_rows = latest_rows_from_jsonl(output_dir / "results.jsonl")
    expected = {run.run_id: run for run in runs if expected_run_ids is None or run.run_id in expected_run_ids}
    all_expected = {run.run_id: run for run in runs}
    seen = {str(row.get("run_id")): row for row in latest_rows if row.get("completed")}
    scoped_seen = {run_id: row for run_id, row in seen.items() if run_id in expected}
    missing = sorted(set(expected) - set(scoped_seen))
    useful_by_config: dict[str, set[str]] = {}
    protocol_violations: list[str] = []
    invariant_violations: list[str] = []
    duplicate_config_mode: list[str] = []
    key_counts: Counter[tuple[str, str]] = Counter()
    for row in seen.values():
        if row.get("run_id") not in all_expected:
            continue
        key_counts[(str(row.get("configuration_id")), str(row.get("optimization_mode")))] += 1
        if row.get("useful_work") not in {"", None}:
            useful_by_config.setdefault(str(row["configuration_id"]), set()).add(str(row["useful_work"]))
        if row.get("sensitivity_axis") == "primary":
            if row.get("optimization_mode") == "schedule-only-best":
                protocol_violations.append(f"{row.get('run_id')}: primary results include schedule-only-best")
            if float(row["time_limit_seconds"]) != PRIMARY_TIME_LIMIT_SECONDS:
                protocol_violations.append(f"{row.get('run_id')}: primary row changed optimization budget")
            if int(row.get("budget_policy_version", 0) or 0) != BUDGET_POLICY_VERSION:
                protocol_violations.append(f"{row.get('run_id')}: incompatible budget policy version")
            if int(row["solver_threads"]) != PRIMARY_SOLVER_THREADS:
                protocol_violations.append(f"{row.get('run_id')}: primary row changed solver thread count")
            if int(row["pipeline_workers"]) != int(row["W"]):
                protocol_violations.append(f"{row.get('run_id')}: pipeline W was not mapped to --J")
            if row.get("primary_protocol_validation_passed") not in {True, "True", "true", "1", 1}:
                protocol_violations.append(f"{row.get('run_id')}: primary protocol validation flag is false")
        if row.get("solver_status") in {"OPTIMAL", "FEASIBLE"}:
            if row.get("optimization_mode") == "partition-only" and row.get("partition_order_validation_passed") not in {True, "True", "true", "1", 1}:
                invariant_violations.append(f"{row.get('run_id')}: partition-only order validation failed")
            if row.get("optimization_mode") == "schedule-only-uniform" and row.get("schedule_partition_validation_passed") not in {True, "True", "true", "1", 1}:
                invariant_violations.append(f"{row.get('run_id')}: schedule-only uniform split validation failed")
            try:
                makespan = float(row["replayed_makespan"])
                useful = float(row["useful_work"])
                utilization = float(row["utilization"])
                expected_util = useful / (float(row["W"]) * makespan)
                if abs(utilization - expected_util) > 1e-6:
                    invariant_violations.append(f"{row.get('run_id')}: utilization mismatch")
                if row.get("incumbent_objective") not in {"", None} and abs(float(row["incumbent_objective"]) - makespan) > 1e-6:
                    invariant_violations.append(f"{row.get('run_id')}: incumbent objective != replayed makespan")
                if row.get("optimization_mode") in {"joint", "schedule-only-uniform"}:
                    fallback_used = row.get("fallback_used") in {True, "True", "true", "1", 1}
                    fallback_available = row.get("fallback_available") in {True, "True", "true", "1", 1}
                    hints_requested = row.get("hints_requested") in {True, "True", "true", "1", 1}
                    hints_effective = row.get("hints_effective") in {True, "True", "true", "1", 1}
                    solution_source = str(row.get("solution_source", ""))
                    joint_status = str(row.get("joint_status", ""))
                    if fallback_used and solution_source != "bfs_incumbent_fallback":
                        invariant_violations.append(f"{row.get('run_id')}: fallback provenance mismatch")
                    if solution_source == "bfs_incumbent_fallback" and not fallback_used:
                        invariant_violations.append(f"{row.get('run_id')}: fallback flag mismatch")
                    if solution_source.startswith("joint_cpsat") and joint_status not in {"OPTIMAL", "FEASIBLE"}:
                        invariant_violations.append(f"{row.get('run_id')}: joint source with non-feasible joint status")
                    if fallback_used and not fallback_available:
                        invariant_violations.append(f"{row.get('run_id')}: fallback used without available fallback")
                    if hints_effective and not hints_requested:
                        invariant_violations.append(f"{row.get('run_id')}: hints effective despite not requested")
                    if hints_requested and not hints_effective:
                        invariant_violations.append(f"{row.get('run_id')}: hints requested but not effective")
                    if row.get("sensitivity_axis") == "hinting" and (hints_requested or hints_effective):
                        invariant_violations.append(f"{row.get('run_id')}: no-hint sensitivity row used hints")
            except Exception as error:
                invariant_violations.append(f"{row.get('run_id')}: metric validation error {error}")
    duplicate_config_mode = [f"{config_id}/{mode}: {count}" for (config_id, mode), count in key_counts.items() if count > 1]
    useful_mismatches = {
        config_id: sorted(values)
        for config_id, values in useful_by_config.items()
        if len(values) > 1
    }
    write_completeness_matrix(output_dir, runs, seen)
    summary = {
        "intended_rows": len(runs),
        "expected_rows": len(expected),
        "completed_rows": len(scoped_seen),
        "successful_rows": sum(1 for row in scoped_seen.values() if row.get("solver_status") in {"OPTIMAL", "FEASIBLE"}),
        "failed_rows": sum(1 for row in scoped_seen.values() if row.get("solver_status") not in {"OPTIMAL", "FEASIBLE"}),
        "missing_rows": len(missing),
        "missing_run_ids": missing[:100],
        "useful_work_mismatches": useful_mismatches,
        "protocol_violations": protocol_violations[:100],
        "invariant_violations": invariant_violations[:100],
        "duplicate_config_mode_keys": duplicate_config_mode[:100],
        "solver_status_counts": {},
        "validation_passed": not missing and not useful_mismatches and not protocol_violations and not invariant_violations and not duplicate_config_mode,
    }
    for row in scoped_seen.values():
        status = str(row.get("solver_status", "UNKNOWN"))
        summary["solver_status_counts"][status] = summary["solver_status_counts"].get(status, 0) + 1
    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SlackPipe primary structural ablation from a manifest.")
    parser.add_argument("--cli", type=Path, default=Path("build/release/slackpipe_cli"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--study", choices=["primary", "sensitivity"], default="primary")
    parser.add_argument("--dry-run", action="store_true", help="write manifests and exit")
    parser.add_argument("--no-resume", action="store_true", help="do not skip completed rows")
    parser.add_argument("--rerun-unsuccessful", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="run the fixed representative 36-row pilot subset")
    parser.add_argument("--max-runs", type=int, default=None, help="testing convenience; do not use for final primary aggregate")
    parser.add_argument("--timeout-grace-seconds", type=float, default=120.0)
    return parser.parse_args()


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args() if argv is None else parse_args_from(argv)
    runner_command = sys.argv if argv is None else [sys.executable, str(Path(__file__)), *argv]
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = PRIMARY_OUTPUT_DIR if args.study == "primary" else SENSITIVITY_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.study == "primary":
        configs = generate_structural_configs()
        runs = primary_run_specs(configs)
    else:
        runs = sensitivity_run_specs()
        configs = sorted({run.structural for run in runs}, key=lambda config: config.configuration_id)

    write_manifest(output_dir, runs)
    write_environment_report(output_dir, args.cli, runner_command)
    selected_runs = select_pilot_runs(configs, runs) if args.pilot and args.study == "primary" else runs
    expected_run_ids = {run.run_id for run in selected_runs}
    if args.pilot and args.study == "primary":
        (output_dir / "pilot_run_ids.txt").write_text(
            "\n".join(run.run_id for run in selected_runs) + "\n", encoding="utf-8"
        )
    if args.dry_run:
        summary = validate_completed_results(output_dir, runs, expected_run_ids if args.pilot else None)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return summary

    complete = set()
    if not args.no_resume:
        complete = completed_run_ids(output_dir / "results.jsonl", args.rerun_unsuccessful)
    if args.pilot and not args.rerun_unsuccessful:
        latest = {
            str(row.get("run_id")): row
            for row in latest_rows_from_jsonl(output_dir / "results.jsonl")
            if str(row.get("run_id")) in expected_run_ids
        }
        existing_failures = [
            row
            for row in latest.values()
            if row.get("solver_status") not in {"OPTIMAL", "FEASIBLE"}
        ]
        if existing_failures:
            summary = validate_completed_results(output_dir, runs, expected_run_ids)
            print(json.dumps(summary, indent=2, sort_keys=True))
            raise SystemExit(1)

    commit = git_commit()
    dirty = git_dirty()
    launched = 0
    for run in selected_runs:
        if run.run_id in complete:
            continue
        row = run_one(args.cli, run, output_dir, commit, dirty, args.timeout_grace_seconds)
        append_result(output_dir, row)
        launched += 1
        if args.pilot and row.get("solver_status") not in {"OPTIMAL", "FEASIBLE"}:
            summary = validate_completed_results(output_dir, runs, expected_run_ids)
            print(json.dumps(summary, indent=2, sort_keys=True))
            raise SystemExit(1)
        if args.max_runs is not None and launched >= args.max_runs:
            break

    summary = validate_completed_results(output_dir, runs, expected_run_ids if args.pilot else None)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args_from(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SlackPipe primary structural ablation from a manifest.")
    parser.add_argument("--cli", type=Path, default=Path("build/release/slackpipe_cli"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--study", choices=["primary", "sensitivity"], default="primary")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--rerun-unsuccessful", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--timeout-grace-seconds", type=float, default=120.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
