#!/usr/bin/env python3
"""Focused evaluation harness for SlackPipe.

This runner is intentionally small and resumable. It shells out to the existing
slackpipe_cli binary, stores a normalized JSON wrapper per run, and then emits
the summary CSVs and plots requested by the evaluation protocol.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import platform
import random
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


TERMINAL_STATUSES = {
    "OPTIMAL",
    "FEASIBLE",
    "INFEASIBLE",
    "UNKNOWN",
    "MODEL_INVALID",
    "UNAVAILABLE",
    "CRASHED",
}

TARGET_HORIZONS = [5, 15, 30, 60, 120, 300]
TARGET_TOLERANCES = [0.0, 0.01, 0.02, 0.05]
DEFAULT_RATIO_NUM = 2
DEFAULT_RATIO_DEN = 1
DEFAULT_MIN_LAYERS = 1
DEFAULT_NUM_WORKERS = 16
DEFAULT_TIME_LIMIT = 300.0
BUDGET_POLICY_VERSION = 1


@dataclasses.dataclass(frozen=True)
class InstanceSpec:
    instance_id: str
    B: int
    N: int
    J: int
    L: int
    layers_per_stage: int
    min_layers: int = DEFAULT_MIN_LAYERS
    ratio_num: int = DEFAULT_RATIO_NUM
    ratio_den: int = DEFAULT_RATIO_DEN
    symmetry_break_f0_fifo: bool = True


@dataclasses.dataclass(frozen=True)
class AlgorithmSpec:
    algorithm: str
    split_mode: str = ""
    worker_move_budget: Optional[int] = None
    per_worker_delta: Optional[int] = None

    @property
    def label(self) -> str:
        if self.algorithm == "optimize-joint":
            return "optimize-joint"
        if self.algorithm == "slackpipe":
            if self.split_mode == "worker-local":
                return f"slackpipe-worker-local-b{self.worker_move_budget}"
            return f"slackpipe-{self.split_mode}"
        return self.algorithm


def build_git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if out:
            return out
    except Exception:
        pass
    return None


def cpu_information() -> str:
    model = "unknown"
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    _, value = line.split(":", 1)
                    model = value.strip()
                    break
    except OSError:
        model = platform.processor() or "unknown"
    parts = [
        platform.machine() or "unknown",
        platform.processor() or "unknown",
        model,
    ]
    return " | ".join(part for part in parts if part and part != "unknown")


def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def instance_id(B: int, N: int, J: int, L: int) -> str:
    return f"n{N}_b{B}_l{L}_j{J}"


def validate_instance(spec: InstanceSpec) -> None:
    if spec.J != DEFAULT_NUM_WORKERS:
        raise ValueError(f"J must be {DEFAULT_NUM_WORKERS}: {spec.instance_id}")
    if spec.N % spec.J != 0:
        raise ValueError(f"N must be divisible by J: {spec.instance_id}")
    if spec.N <= spec.J:
        raise ValueError(f"N must be greater than J: {spec.instance_id}")
    if spec.B <= spec.J:
        raise ValueError(f"B must be greater than J: {spec.instance_id}")
    if spec.L < spec.N * spec.min_layers:
        raise ValueError(f"L must be at least N * min_layers: {spec.instance_id}")


def validate_algorithm_spec(spec: AlgorithmSpec) -> None:
    if spec.algorithm not in {"optimize-joint", "slackpipe"}:
        raise ValueError(f"unexpected algorithm: {spec.algorithm}")
    if spec.algorithm == "slackpipe":
        if spec.split_mode != "worker-local":
            raise ValueError("focused harness only allows worker-local SlackPipe")
        if spec.worker_move_budget not in {1, 2}:
            raise ValueError("worker-local budget must be 1 or 2")
        if spec.per_worker_delta is not None:
            raise ValueError("per-worker-delta must be disabled")


def candidate_instances() -> List[InstanceSpec]:
    out: List[InstanceSpec] = []
    for N in [32, 48, 64, 80]:
        for B in [24, 32, 48, 64]:
            for layers_per_stage in [2, 4]:
                L = N * layers_per_stage
                spec = InstanceSpec(
                    instance_id=instance_id(B, N, DEFAULT_NUM_WORKERS, L),
                    B=B,
                    N=N,
                    J=DEFAULT_NUM_WORKERS,
                    L=L,
                    layers_per_stage=layers_per_stage,
                )
                validate_instance(spec)
                out.append(spec)
    return out


def focused_algorithms() -> List[AlgorithmSpec]:
    algos = [
        AlgorithmSpec("optimize-joint"),
        AlgorithmSpec("slackpipe", split_mode="worker-local", worker_move_budget=1),
        AlgorithmSpec("slackpipe", split_mode="worker-local", worker_move_budget=2),
    ]
    for algo in algos:
        validate_algorithm_spec(algo)
    return algos


def shell_quote_args(args: Sequence[str]) -> str:
    return " ".join(subprocess.list2cmdline([arg]) for arg in args)


def worker_layer_totals(split: Sequence[int], workers: int) -> List[int]:
    totals = [0 for _ in range(workers)]
    for stage, layers in enumerate(split):
        totals[stage % workers] += int(layers)
    return totals


def worker_layer_differences(current: Sequence[int], baseline: Sequence[int]) -> List[int]:
    if len(current) != len(baseline):
        raise ValueError("worker total vectors must have the same length")
    return [int(c) - int(b) for c, b in zip(current, baseline)]


def worker_balance_l1(differences: Sequence[int]) -> int:
    return int(sum(abs(int(value)) for value in differences))


def worker_balance_max_deviation(differences: Sequence[int]) -> int:
    return int(max((abs(int(value)) for value in differences), default=0))


def split_to_csv(split: Optional[Sequence[int]]) -> str:
    if split is None:
        return ""
    return ",".join(str(int(value)) for value in split)


def trace_to_csv(trace: Sequence[Dict[str, Any]]) -> str:
    return json.dumps(trace, separators=(",", ":"))


def run_id_for(instance: InstanceSpec, algo: AlgorithmSpec, seed: int) -> str:
    parts = [instance.instance_id, algo.algorithm]
    if algo.split_mode:
        parts.append(algo.split_mode)
    if algo.worker_move_budget is not None:
        parts.append(f"b{algo.worker_move_budget}")
    parts.append(f"seed{seed}")
    return "__".join(parts)


def make_cli_command(
    cli_path: Path,
    instance: InstanceSpec,
    algo: AlgorithmSpec,
    seed: int,
    num_workers: int,
    time_limit_seconds: float,
    output_prefix: Path,
) -> List[str]:
    base = [
        str(cli_path),
        "--B",
        str(instance.B),
        "--N",
        str(instance.N),
        "--J",
        str(instance.J),
        "--L",
        str(instance.L),
        "--ratio-num",
        str(instance.ratio_num),
        "--ratio-den",
        str(instance.ratio_den),
        "--min-layers",
        str(instance.min_layers),
        "--symmetry-break-f0-fifo",
        "true",
        "--num-workers",
        str(num_workers),
        "--random-seed",
        str(seed),
        "--require-optimal",
        "false",
        "--output-prefix",
        str(output_prefix),
    ]
    if algo.algorithm == "optimize-joint":
        base.extend(["--algorithm", "optimize-joint", "--time-limit-seconds", str(time_limit_seconds)])
        return base
    if algo.algorithm == "slackpipe":
        base.extend(
            [
                "--algorithm",
                "slackpipe",
                "--split-mode",
                algo.split_mode,
                "--worker-move-budget",
                str(algo.worker_move_budget or 0),
                "--time-limit-seconds",
                str(time_limit_seconds),
            ]
        )
        return base
    raise ValueError(f"unsupported algorithm {algo.algorithm}")


def run_with_peak_memory(
    cmd: Sequence[str],
    timeout_seconds: float,
) -> Tuple[int, str, str, float, Optional[int]]:
    started = time.perf_counter()
    peak_kb: Optional[int] = None
    time_tool = shutil.which("/usr/bin/time")
    if time_tool:
        with tempfile.NamedTemporaryFile(prefix="slackpipe-time-", delete=False) as handle:
            peak_file = handle.name
        wrapper = [time_tool, "-f", "%M", "-o", peak_file] + list(cmd)
    else:
        peak_file = None
        wrapper = list(cmd)
    try:
        proc = subprocess.run(
            wrapper,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(timeout_seconds, 1.0) + 120.0,
        )
        elapsed = time.perf_counter() - started
        if peak_file and os.path.exists(peak_file):
            try:
                with open(peak_file, "r", encoding="utf-8") as handle:
                    text = handle.read().strip()
                    if text:
                        peak_kb = int(text)
            finally:
                try:
                    os.unlink(peak_file)
                except OSError:
                    pass
        return proc.returncode, proc.stdout, proc.stderr, elapsed, peak_kb
    except subprocess.TimeoutExpired as error:
        elapsed = time.perf_counter() - started
        if peak_file:
            try:
                os.unlink(peak_file)
            except OSError:
                pass
        raise RuntimeError(f"command timed out after {elapsed:.1f}s: {cmd[0]}") from error


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_split(value: Any) -> List[int]:
    return [int(v) for v in value] if value is not None else []


def parse_trace(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    trace: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            trace.append(
                {
                    "time_seconds": float(item["time_seconds"]),
                    "objective": int(item["objective"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    trace.sort(key=lambda item: (item["time_seconds"], item["objective"]))
    return trace


def objective_at_time(trace: Sequence[Dict[str, Any]], seconds: float) -> Optional[int]:
    objective: Optional[int] = None
    for item in trace:
        if float(item["time_seconds"]) <= seconds + 1e-12:
            objective = int(item["objective"])
        else:
            break
    return objective


def time_to_target(trace: Sequence[Dict[str, Any]], target: int) -> Optional[float]:
    for item in trace:
        if int(item["objective"]) <= target:
            return float(item["time_seconds"])
    return None


def candidate_from_result(data: Dict[str, Any]) -> Optional[int]:
    final = data.get("final_objective")
    if final is None:
        final = data.get("final_makespan_ticks")
    if final is None:
        final = data.get("makespan_ticks")
    if final is None:
        final = data.get("makespan")
    if final is None:
        return None
    return int(final)


def terminal_status(value: Any) -> bool:
    return isinstance(value, str) and value in TERMINAL_STATUSES


def normalize_run(
    instance: InstanceSpec,
    algo: AlgorithmSpec,
    seed: int,
    time_limit_seconds: float,
    num_workers: int,
    build_meta: Dict[str, Any],
    cli_json: Dict[str, Any],
    elapsed_seconds: float,
    peak_rss_kb: Optional[int],
) -> Dict[str, Any]:
    solver_output = dict(cli_json)
    canonical = solver_output.get("canonical_result", {})
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
    status = str(solver_output.get("status", "UNKNOWN"))
    if algo.algorithm == "optimize-joint":
        initial_split = parse_split(solver_output.get("bfs_incumbent_split"))
        final_split = parse_split(solver_output.get("split"))
        baseline_worker_layers = worker_layer_totals(initial_split, instance.J) if initial_split else []
        final_worker_layers = worker_layer_totals(final_split, instance.J) if final_split else []
        initial_objective = int(solver_output.get("bfs_incumbent_makespan_ticks", 0))
        final_objective = candidate_from_result(solver_output) or 0
        first_feasible = int(solver_output.get("first_feasible_objective", 0))
        trace = parse_trace(solver_output.get("incumbent_trace"))
        proven_optimal = bool(solver_output.get("proven_optimal", False))
        proven_global_optimal = proven_optimal
        global_certificate = ""
    else:
        initial_split = parse_split(solver_output.get("initial_uniform_split"))
        final_split = parse_split(
            solver_output.get("final_split", solver_output.get("split"))
        )
        baseline_worker_layers = parse_split(solver_output.get("baseline_worker_layers"))
        if not baseline_worker_layers and initial_split:
            baseline_worker_layers = worker_layer_totals(initial_split, instance.J)
        final_worker_layers = parse_split(solver_output.get("final_worker_layers"))
        if not final_worker_layers and final_split:
            final_worker_layers = worker_layer_totals(final_split, instance.J)
        initial_objective = int(solver_output.get("initial_uniform_bfs_makespan", 0))
        final_objective = candidate_from_result(solver_output) or 0
        first_feasible = int(solver_output.get("first_feasible_objective", 0))
        trace = parse_trace(solver_output.get("incumbent_trace"))
        proven_optimal = bool(solver_output.get("proven_depth_optimal", False))
        proven_global_optimal = bool(solver_output.get("proven_global_optimal", False))
        global_certificate = str(solver_output.get("global_certificate", ""))

    if not final_worker_layers and final_split:
        final_worker_layers = worker_layer_totals(final_split, instance.J)
    if not baseline_worker_layers and initial_split:
        baseline_worker_layers = worker_layer_totals(initial_split, instance.J)
    if final_worker_layers and baseline_worker_layers:
        diff = worker_layer_differences(final_worker_layers, baseline_worker_layers)
        worker_l1 = worker_balance_l1(diff)
        worker_max = worker_balance_max_deviation(diff)
    else:
        diff = []
        worker_l1 = 0
        worker_max = 0
    if not trace and final_objective > 0 and status in {"OPTIMAL", "FEASIBLE"}:
        trace = [{"time_seconds": float(time_limit_seconds), "objective": final_objective}]
    worker_move_budget = algo.worker_move_budget if algo.worker_move_budget is not None else 0
    per_worker_delta = algo.per_worker_delta

    result = {
        "run_id": run_id_for(instance, algo, seed),
        "instance_id": instance.instance_id,
        "algorithm": algo.algorithm,
        "configuration": algo.label,
        "B": instance.B,
        "N": instance.N,
        "J": instance.J,
        "L": instance.L,
        "layers_per_stage": instance.layers_per_stage,
        "min_layers": instance.min_layers,
        "ratio_num": instance.ratio_num,
        "ratio_den": instance.ratio_den,
        "random_seed": seed,
        "seed": seed,
        "split_mode": algo.split_mode,
        "worker_move_budget": worker_move_budget if algo.algorithm == "slackpipe" else "",
        "per_worker_delta": per_worker_delta if per_worker_delta is not None else "",
        "time_limit_seconds": time_limit_seconds,
        "num_workers": num_workers,
        "symmetry_break_f0_fifo": instance.symmetry_break_f0_fifo,
        "schema_version": solver_output.get("schema_version", canonical.get("schema_version", "")),
        "budget_policy_version": solver_output.get(
            "budget_policy_version",
            canonical.get("budget_policy_version", ""),
        ),
        "evaluation_method_version": solver_output.get(
            "evaluation_method_version",
            canonical.get("evaluation_method_version", ""),
        ),
        "requested_method": canonical.get("requested_method", ""),
        "canonical_method": solver_output.get("canonical_method", canonical.get("canonical_method", "")),
        "method_contract_hash": canonical.get("method_contract_hash", ""),
        "fixed_schedule_rule": canonical.get("fixed_schedule_rule", ""),
        "uniform_partition_rule": canonical.get("uniform_partition_rule", ""),
        "actual_solver_path": solver_output.get("actual_solver_path", canonical.get("actual_solver_path", "")),
        "partition_decision": canonical.get("partition_decision", ""),
        "schedule_decision": canonical.get("schedule_decision", ""),
        "full_partition_fixed": canonical.get("full_partition_fixed", ""),
        "worker_aggregate_loads_fixed": canonical.get("worker_aggregate_loads_fixed", ""),
        "partition_optimized": canonical.get("partition_optimized", ""),
        "schedule_optimized": canonical.get("schedule_optimized", ""),
        "predecessor_candidate_restriction_requested": canonical.get("predecessor_candidate_restriction_requested", ""),
        "predecessor_candidate_restriction_active": canonical.get("predecessor_candidate_restriction_active", ""),
        "solver_status_raw": canonical.get("solver_status_raw", ""),
        "reported_status": canonical.get("reported_status", ""),
        "fallback_reason": canonical.get("fallback_reason", ""),
        "returned_solution_source": canonical.get("returned_solution_source", ""),
        "communication_model": canonical.get("communication_model", ""),
        "activation_model": canonical.get("activation_model", ""),
        "activation_units_per_layer": canonical.get("activation_units_per_layer", ""),
        "explicit_stage_activation_units": split_to_csv(
            canonical.get("explicit_stage_activation_units")
        )
        if isinstance(canonical.get("explicit_stage_activation_units"), list)
        else "",
        "activation_cap_mode": canonical.get("activation_cap_mode", ""),
        "activation_cap_units_per_worker": split_to_csv(
            canonical.get("activation_cap_units_per_worker")
        )
        if isinstance(canonical.get("activation_cap_units_per_worker"), list)
        else "",
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
        "status": status,
        "terminal_status": status if terminal_status(status) else "",
        "completed": True,
        "proven_optimal": proven_optimal,
        "proven_global_optimal": proven_global_optimal,
        "global_certificate": global_certificate,
        "initial_split": initial_split,
        "final_split": final_split,
        "baseline_worker_layers": baseline_worker_layers,
        "final_worker_layers": final_worker_layers,
        "worker_layer_differences": diff,
        "worker_balance_l1": worker_l1,
        "worker_balance_max_deviation": worker_max,
        "initial_objective": initial_objective,
        "first_feasible_objective": first_feasible,
        "final_objective": final_objective,
        "makespan": final_objective,
        "objective": int(solver_output.get("solver_objective_ticks", final_objective)),
        "best_bound": float(
            solver_output.get(
                "best_bound_ticks",
                solver_output.get("best_bound", 0.0),
            )
        ),
        "time_to_first_feasible_seconds": float(
            solver_output.get("time_to_first_feasible_seconds", 0.0)
        ),
        "time_to_best_incumbent_seconds": float(
            solver_output.get("time_to_best_incumbent_seconds", 0.0)
        ),
        "incumbent_improvement_count": int(
            solver_output.get("incumbent_improvement_count", 0)
        ),
        "incumbent_trace": trace,
        "selected_stage_count": solver_output.get("selected_stage_count", ""),
        "slack_stages": solver_output.get("slack_stages", []),
        "cp_sat_models_solved": int(solver_output.get("cp_sat_models_solved", 0)),
        "solver_seconds": float(
            solver_output.get("solver_seconds", solver_output.get("total_seconds", 0.0))
        ),
        "ortools_wall_time_seconds": float(
            solver_output.get(
                "ortools_wall_time_seconds",
                solver_output.get("wall_time_seconds", 0.0),
            )
        ),
        "external_total_seconds": elapsed_seconds,
        "peak_rss_kb": peak_rss_kb if peak_rss_kb is not None else "",
        "git_commit": build_meta["git_commit"],
        "build_type": build_meta["build_type"],
        "hostname": build_meta["hostname"],
        "cpu_information": build_meta["cpu_information"],
        "solver_output": solver_output,
    }
    return result


def expected_instance_match(instance: InstanceSpec, seed: int, data: Dict[str, Any]) -> bool:
    return (
        int(data.get("B", instance.B)) == instance.B
        and int(data.get("N", instance.N)) == instance.N
        and int(data.get("J", instance.J)) == instance.J
        and int(data.get("L", instance.L)) == instance.L
        and int(data.get("random_seed", seed)) == seed
    )


def is_complete_run(
    path: Path,
    instance: InstanceSpec,
    algo: AlgorithmSpec,
    seed: int,
    time_limit_seconds: float,
    num_workers: int,
) -> bool:
    if not path.exists():
        return False
    try:
        data = load_json(path)
    except Exception:
        return False
    if data.get("run_id") != run_id_for(instance, algo, seed):
        return False
    if not expected_instance_match(instance, seed, data):
        return False
    if data.get("algorithm") != algo.algorithm:
        return False
    if data.get("seed") != seed or data.get("random_seed") != seed:
        return False
    if int(data.get("time_limit_seconds", time_limit_seconds)) != int(time_limit_seconds):
        return False
    if int(data.get("num_workers", num_workers)) != num_workers:
        return False
    if int(data.get("budget_policy_version", 0) or 0) != BUDGET_POLICY_VERSION:
        return False
    if not terminal_status(str(data.get("status", ""))):
        return False
    if algo.algorithm == "slackpipe":
        if data.get("split_mode") != algo.split_mode:
            return False
        if int(data.get("worker_move_budget", -1)) != int(algo.worker_move_budget or 0):
            return False
        if data.get("per_worker_delta", "") not in {"", None}:
            return False
    return True


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def summarize_trace_at_horizons(
    run: Dict[str, Any],
    horizons: Sequence[int] = TARGET_HORIZONS,
) -> Dict[int, Optional[int]]:
    trace = run.get("incumbent_trace", [])
    values: Dict[int, Optional[int]] = {}
    for horizon in horizons:
        values[horizon] = objective_at_time(trace, float(horizon))
    return values


def global_from_runs(rows: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    best_known: Dict[str, int] = {}
    best_joint_lb: Dict[str, int] = {}
    for row in rows:
        inst = row["instance_id"]
        final = int(row["final_objective"])
        status = str(row["status"])
        if status in {"OPTIMAL", "FEASIBLE"} and final > 0:
            best_known[inst] = min(best_known.get(inst, final), final)
        if row["algorithm"] == "optimize-joint":
            bound = int(float(row["best_bound"]))
            if bound > 0:
                best_joint_lb[inst] = max(best_joint_lb.get(inst, bound), bound)
    return best_known, best_joint_lb


def assign_analysis_fields(rows: List[Dict[str, Any]]) -> None:
    best_known, best_joint_lb = global_from_runs(rows)
    joint_final_by_pair: Dict[Tuple[str, int], int] = {}
    for row in rows:
        if row["algorithm"] == "optimize-joint" and row["status"] in {"OPTIMAL", "FEASIBLE"}:
            joint_final_by_pair[(row["instance_id"], int(row["seed"]))] = int(row["final_objective"])

    for row in rows:
        inst = row["instance_id"]
        final = int(row["final_objective"]) if row["final_objective"] else 0
        best = best_known.get(inst, final or 1)
        joint_lb = best_joint_lb.get(inst, 0)
        row["best_known_objective"] = best
        row["best_joint_lower_bound"] = joint_lb
        row["best_known_gap"] = 0.0 if best <= 0 else 100.0 * (final - best) / best
        row["certified_global_gap"] = (
            0.0 if final <= 0 else 100.0 * (final - joint_lb) / final
        )
        joint_final = joint_final_by_pair.get((inst, int(row["seed"])))
        if joint_final and row["algorithm"] == "slackpipe":
            row["joint_relative_gap"] = 100.0 * (final - joint_final) / joint_final
        elif row["algorithm"] == "optimize-joint":
            row["joint_relative_gap"] = 0.0
        else:
            row["joint_relative_gap"] = ""
        row["worker_layer_differences"] = row.get("worker_layer_differences", [])


def instance_summary_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["instance_id"], []).append(row)
    summaries: List[Dict[str, Any]] = []
    for instance_id_, group in sorted(grouped.items()):
        best_known = min(
            int(r["final_objective"]) for r in group if r["status"] in {"OPTIMAL", "FEASIBLE"}
        )
        joint_lb = max(
            int(r["best_bound"]) for r in group if r["algorithm"] == "optimize-joint"
        )
        best_row = min(
            (r for r in group if r["final_objective"]),
            key=lambda r: int(r["final_objective"]),
        )
        summaries.append(
            {
                "instance_id": instance_id_,
                "B": best_row["B"],
                "N": best_row["N"],
                "J": best_row["J"],
                "L": best_row["L"],
                "layers_per_stage": best_row["layers_per_stage"],
                "best_known_objective": best_known,
                "best_joint_lower_bound": joint_lb,
                "best_known_algorithm": best_row["algorithm"],
                "best_known_seed": best_row["seed"],
                "num_runs": len(group),
                "joint_runs": sum(1 for r in group if r["algorithm"] == "optimize-joint"),
                "worker_local_b1_runs": sum(
                    1
                    for r in group
                    if r["algorithm"] == "slackpipe" and int(r["worker_move_budget"] or 0) == 1
                ),
                "worker_local_b2_runs": sum(
                    1
                    for r in group
                    if r["algorithm"] == "slackpipe" and int(r["worker_move_budget"] or 0) == 2
                ),
            }
        )
    return summaries


def aggregate_summary_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    by_algo: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_algo.setdefault(row["configuration"], []).append(row)
    for key, group in sorted(by_algo.items()):
        finals = [int(r["final_objective"]) for r in group if r["final_objective"]]
        gaps = [float(r["best_known_gap"]) for r in group if r["final_objective"]]
        times = [float(r["external_total_seconds"]) for r in group if float(r["external_total_seconds"]) > 0]
        out.append(
            {
                "group": key,
                "count": len(group),
                "median_final_objective": median(finals),
                "iqr_final_objective": iqr(finals),
                "median_best_known_gap": median(gaps),
                "iqr_best_known_gap": iqr(gaps),
                "median_external_total_seconds": median(times),
                "iqr_external_total_seconds": iqr(times),
                "geomean_speedup_vs_joint_target": "",
            }
        )
    return out


def time_to_target_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    per_pair_joint: Dict[Tuple[str, int], Dict[str, Any]] = {}
    per_pair_target: Dict[Tuple[str, int], int] = {}
    for row in rows:
        key = (row["instance_id"], int(row["seed"]))
        if row["algorithm"] == "optimize-joint":
            per_pair_joint[key] = row
            per_pair_target[key] = int(row["final_objective"])
    for row in rows:
        key = (row["instance_id"], int(row["seed"]))
        trace = row["incumbent_trace"]
        final = int(row["final_objective"]) if row["final_objective"] else 0
        horizons = summarize_trace_at_horizons(row)
        for horizon in TARGET_HORIZONS:
            obj = horizons[horizon]
            out.append(
                {
                    "analysis_type": "timepoint",
                    "instance_id": row["instance_id"],
                    "seed": row["seed"],
                    "algorithm": row["algorithm"],
                    "worker_move_budget": row["worker_move_budget"],
                    "target_kind": f"objective_at_{horizon}s",
                    "target_objective": "",
                    "time_seconds": horizon,
                    "objective": obj if obj is not None else "",
                    "censored": obj is None,
                    "joint_time_seconds": "",
                    "local_time_seconds": "",
                    "speedup": "",
                    "speedup_lower_bound": "",
                }
            )

        best_known = int(row["best_known_objective"])
        if best_known > 0:
            exact = [0.0, 0.01, 0.02, 0.05]
            for tol in exact:
                target = math.floor((1.0 + tol) * best_known)
                tt = time_to_target(trace, target)
                out.append(
                    {
                        "analysis_type": "target",
                        "instance_id": row["instance_id"],
                        "seed": row["seed"],
                        "algorithm": row["algorithm"],
                        "worker_move_budget": row["worker_move_budget"],
                        "target_kind": f"best_known_plus_{int(tol * 100)}pct",
                        "target_objective": target,
                        "time_seconds": tt if tt is not None else "",
                        "objective": final,
                        "censored": tt is None,
                        "joint_time_seconds": "",
                        "local_time_seconds": "",
                        "speedup": "",
                        "speedup_lower_bound": "",
                    }
                )

        if row["algorithm"] == "slackpipe" and key in per_pair_joint:
            joint_row = per_pair_joint[key]
            target = int(joint_row["final_objective"])
            joint_time = time_to_target(joint_row["incumbent_trace"], target)
            local_time = time_to_target(trace, target)
            speedup = ""
            speedup_lower_bound = ""
            if joint_time is not None and local_time is not None and local_time > 0:
                speedup = joint_time / local_time
            elif joint_time is None and local_time is not None and local_time > 0:
                speedup_lower_bound = 300.0 / local_time
            out.append(
                {
                    "analysis_type": "joint_final_target",
                    "instance_id": row["instance_id"],
                    "seed": row["seed"],
                    "algorithm": row["algorithm"],
                    "worker_move_budget": row["worker_move_budget"],
                    "target_kind": "joint_final_objective",
                    "target_objective": target,
                    "time_seconds": local_time if local_time is not None else "",
                    "objective": final,
                    "censored": local_time is None,
                    "joint_time_seconds": joint_time if joint_time is not None else "",
                    "local_time_seconds": local_time if local_time is not None else "",
                    "speedup": speedup,
                    "speedup_lower_bound": speedup_lower_bound,
                }
            )
    return out


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    idx = p * (len(ordered) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)


def iqr(values: Sequence[float]) -> float:
    return percentile(values, 0.75) - percentile(values, 0.25)


def geometric_mean(values: Sequence[float]) -> float:
    positive = [float(v) for v in values if float(v) > 0]
    if not positive:
        return 0.0
    return math.exp(sum(math.log(v) for v in positive) / len(positive))


def bootstrap_ci(
    values: Sequence[float],
    stat_fn,
    *,
    repetitions: int = 2000,
    seed: int = 0,
) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    samples: List[float] = []
    data = list(values)
    for _ in range(repetitions):
        draw = [data[rng.randrange(len(data))] for _ in range(len(data))]
        samples.append(float(stat_fn(draw)))
    samples.sort()
    return percentile(samples, 0.025), percentile(samples, 0.975)


def write_run_manifest(path: Path, plans: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "instance_id",
        "seed",
        "order_index",
        "algorithm",
        "split_mode",
        "worker_move_budget",
        "run_id",
        "output_prefix",
    ]
    write_csv(path, plans, fieldnames)


def make_execution_manifest(
    instances: Sequence[InstanceSpec],
    seeds: Sequence[int],
    algorithms: Sequence[AlgorithmSpec],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    plans: List[Dict[str, Any]] = []
    for instance in instances:
        for seed in seeds:
            ordering = list(algorithms)
            rng = random.Random(stable_hash(f"{instance.instance_id}:{seed}"))
            rng.shuffle(ordering)
            for order_index, algo in enumerate(ordering):
                plans.append(
                    {
                        "instance_id": instance.instance_id,
                        "seed": seed,
                        "order_index": order_index,
                        "algorithm": algo.algorithm,
                        "split_mode": algo.split_mode,
                        "worker_move_budget": algo.worker_move_budget
                        if algo.worker_move_budget is not None
                        else "",
                        "run_id": run_id_for(instance, algo, seed),
                        "output_prefix": str(output_dir / "runs" / run_id_for(instance, algo, seed)),
                    }
                )
    return plans


def write_raw_runs_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "run_id",
        "instance_id",
        "algorithm",
        "configuration",
        "B",
        "N",
        "J",
        "L",
        "layers_per_stage",
        "min_layers",
        "ratio_num",
        "ratio_den",
        "random_seed",
        "seed",
        "split_mode",
        "worker_move_budget",
        "per_worker_delta",
        "time_limit_seconds",
        "num_workers",
        "symmetry_break_f0_fifo",
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
        "status",
        "terminal_status",
        "completed",
        "proven_optimal",
        "proven_global_optimal",
        "global_certificate",
        "initial_split",
        "final_split",
        "baseline_worker_layers",
        "final_worker_layers",
        "worker_layer_differences",
        "worker_balance_l1",
        "worker_balance_max_deviation",
        "initial_objective",
        "first_feasible_objective",
        "final_objective",
        "makespan",
        "objective",
        "best_bound",
        "joint_relative_gap",
        "best_known_objective",
        "best_known_gap",
        "best_joint_lower_bound",
        "certified_global_gap",
        "time_to_first_feasible_seconds",
        "time_to_best_incumbent_seconds",
        "incumbent_improvement_count",
        "incumbent_trace",
        "selected_stage_count",
        "slack_stages",
        "cp_sat_models_solved",
        "solver_seconds",
        "ortools_wall_time_seconds",
        "external_total_seconds",
        "peak_rss_kb",
        "git_commit",
        "build_type",
        "hostname",
        "cpu_information",
    ]
    write_csv(path, rows, fieldnames)


def write_instance_summary_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    summaries = instance_summary_rows(rows)
    fieldnames = [
        "instance_id",
        "B",
        "N",
        "J",
        "L",
        "layers_per_stage",
        "best_known_objective",
        "best_joint_lower_bound",
        "best_known_algorithm",
        "best_known_seed",
        "num_runs",
        "joint_runs",
        "worker_local_b1_runs",
        "worker_local_b2_runs",
    ]
    write_csv(path, summaries, fieldnames)


def write_aggregate_summary_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    summaries = aggregate_summary_rows(rows)
    fieldnames = [
        "group",
        "count",
        "median_final_objective",
        "iqr_final_objective",
        "median_best_known_gap",
        "iqr_best_known_gap",
        "median_external_total_seconds",
        "iqr_external_total_seconds",
        "geomean_speedup_vs_joint_target",
    ]
    write_csv(path, summaries, fieldnames)


def write_time_to_target_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "analysis_type",
        "instance_id",
        "seed",
        "algorithm",
        "worker_move_budget",
        "target_kind",
        "target_objective",
        "time_seconds",
        "objective",
        "censored",
        "joint_time_seconds",
        "local_time_seconds",
        "speedup",
        "speedup_lower_bound",
    ]
    write_csv(path, rows, fieldnames)


def write_failures_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = ["run_id", "instance_id", "algorithm", "seed", "message", "output_prefix"]
    write_csv(path, rows, fieldnames)


def maybe_write_wrapper(prefix: Path, wrapper: Dict[str, Any]) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_json(prefix.with_suffix(".json"), wrapper)


def load_retained_instances_from_pilot_csv(path: Path) -> List[InstanceSpec]:
    if not path.exists():
        raise FileNotFoundError(f"pilot screening file not found: {path}")
    retained: List[InstanceSpec] = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("retain", "")).lower() != "true":
                continue
            B = int(row["B"])
            N = int(row["N"])
            J = int(row["J"])
            L = int(row["L"])
            layers_per_stage = int(row["layers_per_stage"])
            spec = InstanceSpec(
                instance_id=row["instance_id"],
                B=B,
                N=N,
                J=J,
                L=L,
                layers_per_stage=layers_per_stage,
            )
            validate_instance(spec)
            retained.append(spec)
    return retained


def complete_or_run(
    cli_path: Path,
    instance: InstanceSpec,
    algo: AlgorithmSpec,
    seed: int,
    num_workers: int,
    time_limit_seconds: float,
    output_dir: Path,
    build_meta: Dict[str, Any],
) -> Dict[str, Any]:
    run_prefix = output_dir / "runs" / run_id_for(instance, algo, seed)
    wrapper_path = run_prefix.with_suffix(".json")
    if is_complete_run(wrapper_path, instance, algo, seed, time_limit_seconds, num_workers):
        return load_json(wrapper_path)

    temp_dir = output_dir / ".tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_prefix = temp_dir / run_id_for(instance, algo, seed)
    for suffix in [".json", ".csv", ".orders.txt", ".svg"]:
        try:
            temp_prefix.with_suffix(suffix).unlink()
        except OSError:
            pass

    cmd = make_cli_command(
        cli_path=cli_path,
        instance=instance,
        algo=algo,
        seed=seed,
        num_workers=num_workers,
        time_limit_seconds=time_limit_seconds,
        output_prefix=temp_prefix,
    )
    rc, stdout, stderr, elapsed, peak_rss_kb = run_with_peak_memory(
        cmd, timeout_seconds=time_limit_seconds
    )
    json_path = temp_prefix.with_suffix(".json")
    if not json_path.exists():
        raise RuntimeError(
            f"solver completed without producing JSON: run_id={run_id_for(instance, algo, seed)}\n"
            f"stdout={stdout}\nstderr={stderr}"
        )
    cli_json = load_json(json_path)
    wrapper = normalize_run(
        instance=instance,
        algo=algo,
        seed=seed,
        time_limit_seconds=time_limit_seconds,
        num_workers=num_workers,
        build_meta=build_meta,
        cli_json=cli_json,
        elapsed_seconds=elapsed,
        peak_rss_kb=peak_rss_kb,
    )
    wrapper["exit_code"] = rc
    wrapper["stdout"] = stdout
    wrapper["stderr"] = stderr
    maybe_write_wrapper(run_prefix, wrapper)
    return wrapper


def pilot_screening(
    cli_path: Path,
    output_dir: Path,
    build_meta: Dict[str, Any],
    instances: Sequence[InstanceSpec],
) -> Tuple[List[Dict[str, Any]], List[InstanceSpec]]:
    rows: List[Dict[str, Any]] = []
    retained: List[InstanceSpec] = []
    algo_map = {
        "joint": AlgorithmSpec("optimize-joint"),
        "local1": AlgorithmSpec("slackpipe", split_mode="worker-local", worker_move_budget=1),
        "local2": AlgorithmSpec("slackpipe", split_mode="worker-local", worker_move_budget=2),
    }
    for instance in instances:
        result_rows: Dict[str, Dict[str, Any]] = {}
        for name, algo in algo_map.items():
            run = complete_or_run(
                cli_path=cli_path,
                instance=instance,
                algo=algo,
                seed=0,
                num_workers=DEFAULT_NUM_WORKERS,
                time_limit_seconds=60.0,
                output_dir=output_dir / "pilot",
                build_meta=build_meta,
            )
            result_rows[name] = run
        joint = result_rows["joint"]
        local_1 = result_rows["local1"]
        local_2 = result_rows["local2"]
        def improved_at_60(run: Dict[str, Any]) -> Tuple[bool, float, Optional[int], Optional[int]]:
            trace = run.get("incumbent_trace", [])
            obj5 = objective_at_time(trace, 5.0)
            obj60 = objective_at_time(trace, 60.0)
            if obj60 is None:
                return False, 0.0, obj5, obj60
            if obj5 is None:
                return True, 0.0, obj5, obj60
            improvement = (obj5 - obj60) / max(1.0, float(obj5))
            return improvement >= 0.01, improvement, obj5, obj60

        local1_keep, local1_improvement, local1_obj5, local1_obj60 = improved_at_60(local_1)
        local2_keep, local2_improvement, local2_obj5, local2_obj60 = improved_at_60(local_2)
        joint_optimal = joint["status"] == "OPTIMAL"
        retain = (not joint_optimal) and (local1_keep or local2_keep)
        reason_bits = [
            f"joint_status={joint['status']}",
            f"joint_optimal={str(joint_optimal).lower()}",
            f"local1_5s={local1_obj5 if local1_obj5 is not None else 'none'}",
            f"local1_60s={local1_obj60 if local1_obj60 is not None else 'none'}",
            f"local1_improvement={local1_improvement:.4f}",
            f"local2_5s={local2_obj5 if local2_obj5 is not None else 'none'}",
            f"local2_60s={local2_obj60 if local2_obj60 is not None else 'none'}",
            f"local2_improvement={local2_improvement:.4f}",
        ]
        reason = "retained" if retain else "rejected"
        rows.append(
            {
                "instance_id": instance.instance_id,
                "B": instance.B,
                "N": instance.N,
                "J": instance.J,
                "L": instance.L,
                "layers_per_stage": instance.layers_per_stage,
                "joint_status": joint["status"],
                "joint_objective": joint["final_objective"],
                "joint_best_bound": joint["best_bound"],
                "worker_local_1_status": local_1["status"],
                "worker_local_1_obj_5s": local1_obj5 if local1_obj5 is not None else "",
                "worker_local_1_obj_60s": local1_obj60 if local1_obj60 is not None else "",
                "worker_local_1_improvement_fraction": local1_improvement,
                "worker_local_2_status": local_2["status"],
                "worker_local_2_obj_5s": local2_obj5 if local2_obj5 is not None else "",
                "worker_local_2_obj_60s": local2_obj60 if local2_obj60 is not None else "",
                "worker_local_2_improvement_fraction": local2_improvement,
                "retain": retain,
                "reason": reason,
                "details": "; ".join(reason_bits),
            }
        )
        if retain:
            retained.append(instance)
    write_csv(
        output_dir / "pilot_screening.csv",
        rows,
        [
            "instance_id",
            "B",
            "N",
            "J",
            "L",
            "layers_per_stage",
            "joint_status",
            "joint_objective",
            "joint_best_bound",
            "worker_local_1_status",
            "worker_local_1_obj_5s",
            "worker_local_1_obj_60s",
            "worker_local_1_improvement_fraction",
            "worker_local_2_status",
            "worker_local_2_obj_5s",
            "worker_local_2_obj_60s",
            "worker_local_2_improvement_fraction",
            "retain",
            "reason",
            "details",
        ],
    )
    return rows, retained


def run_main_experiments(
    cli_path: Path,
    output_dir: Path,
    build_meta: Dict[str, Any],
    retained_instances: Sequence[InstanceSpec],
    seeds: Sequence[int],
) -> List[Dict[str, Any]]:
    algorithms = focused_algorithms()
    manifest = make_execution_manifest(retained_instances, seeds, algorithms, output_dir)
    write_run_manifest(output_dir / "run_manifest.csv", manifest)
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for instance in retained_instances:
        for seed in seeds:
            ordered = list(algorithms)
            rng = random.Random(stable_hash(f"{instance.instance_id}:{seed}"))
            rng.shuffle(ordered)
            for algo in ordered:
                try:
                    run = complete_or_run(
                        cli_path=cli_path,
                        instance=instance,
                        algo=algo,
                        seed=seed,
                        num_workers=DEFAULT_NUM_WORKERS,
                        time_limit_seconds=DEFAULT_TIME_LIMIT,
                        output_dir=output_dir / "main",
                        build_meta=build_meta,
                    )
                    rows.append(run)
                except Exception as error:
                    failures.append(
                        {
                            "run_id": run_id_for(instance, algo, seed),
                            "instance_id": instance.instance_id,
                            "algorithm": algo.algorithm,
                            "seed": seed,
                            "message": str(error),
                            "output_prefix": str(
                                output_dir / "main" / "runs" / run_id_for(instance, algo, seed)
                            ),
                        }
                    )
    write_failures_csv(output_dir / "failures.csv", failures)
    return rows


def enrich_rows(rows: List[Dict[str, Any]]) -> None:
    assign_analysis_fields(rows)
    for row in rows:
        row["objective_at_5s"] = objective_at_time(row["incumbent_trace"], 5.0)
        row["objective_at_15s"] = objective_at_time(row["incumbent_trace"], 15.0)
        row["objective_at_30s"] = objective_at_time(row["incumbent_trace"], 30.0)
        row["objective_at_60s"] = objective_at_time(row["incumbent_trace"], 60.0)
        row["objective_at_120s"] = objective_at_time(row["incumbent_trace"], 120.0)
        row["objective_at_300s"] = objective_at_time(row["incumbent_trace"], 300.0)


def write_analysis_artifacts(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    write_raw_runs_csv(output_dir / "raw_runs.csv", rows)
    write_instance_summary_csv(output_dir / "instance_summary.csv", rows)
    write_aggregate_summary_csv(output_dir / "aggregate_summary.csv", rows)
    write_time_to_target_csv(output_dir / "time_to_target.csv", time_to_target_rows(rows))


def best_known_target_values(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    best_known, _ = global_from_runs(rows)
    return best_known


def plot_artifacts(output_dir: Path, rows: Sequence[Dict[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        raise RuntimeError(
            "matplotlib is required for plot generation"
        ) from error

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    best_known = best_known_target_values(rows)

    # Plot 1: objective gap versus time.
    timepoints = TARGET_HORIZONS
    groups = sorted({row["configuration"] for row in rows})
    fig, ax = plt.subplots(figsize=(9, 5))
    for group in groups:
        series: List[Tuple[int, float]] = []
        for t in timepoints:
            gaps = []
            for row in rows:
                if row["configuration"] != group:
                    continue
                obj = objective_at_time(row["incumbent_trace"], float(t))
                if obj is None:
                    continue
                best = best_known[row["instance_id"]]
                gaps.append(100.0 * (obj - best) / best)
            if gaps:
                series.append((t, statistics.median(gaps)))
        if series:
            ax.plot([x for x, _ in series], [y for _, y in series], marker="o", label=group)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("median gap to best-known (%)")
    ax.set_title("Objective gap versus time")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "objective_gap_vs_time.png", dpi=160)
    plt.close(fig)

    # Plot 2: ECDF of time to within 1% of best-known.
    fig, ax = plt.subplots(figsize=(8, 5))
    for group in groups:
        values = []
        for row in rows:
            if row["configuration"] != group:
                continue
            target = math.floor(1.01 * best_known[row["instance_id"]])
            tt = time_to_target(row["incumbent_trace"], target)
            if tt is not None:
                values.append(tt)
        if values:
            ordered = sorted(values)
            ys = [(i + 1) / len(ordered) for i in range(len(ordered))]
            ax.step(ordered, ys, where="post", label=group)
    ax.set_xlabel("time to within 1% of best-known (s)")
    ax.set_ylabel("ECDF")
    ax.set_title("Time-to-1% empirical CDF")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "time_to_within_1pct_ecdf.png", dpi=160)
    plt.close(fig)

    # Plot 3: scatter quality gap versus time-to-target speedup.
    fig, ax = plt.subplots(figsize=(8, 5))
    xs: List[float] = []
    ys: List[float] = []
    colors: List[str] = []
    palette = {"worker-local": "tab:blue"}
    for row in rows:
        if row["algorithm"] != "slackpipe":
            continue
        key = (row["instance_id"], int(row["seed"]))
        joint = next(
            (
                other
                for other in rows
                if other["instance_id"] == row["instance_id"]
                and int(other["seed"]) == int(row["seed"])
                and other["algorithm"] == "optimize-joint"
            ),
            None,
        )
        if joint is None:
            continue
        target = int(joint["final_objective"])
        local_time = time_to_target(row["incumbent_trace"], target)
        joint_time = time_to_target(joint["incumbent_trace"], target)
        if local_time is None or joint_time is None or local_time <= 0:
            continue
        speedup = joint_time / local_time
        xs.append(speedup)
        ys.append(float(row["best_known_gap"]))
        colors.append("tab:blue" if int(row["worker_move_budget"] or 0) == 1 else "tab:orange")
    if xs:
        ax.scatter(xs, ys, c=colors, alpha=0.7)
    ax.set_xlabel("speedup versus optimize-joint")
    ax.set_ylabel("final gap to best-known (%)")
    ax.set_title("Quality gap versus time-to-target speedup")
    fig.tight_layout()
    fig.savefig(plots_dir / "quality_gap_vs_speedup.png", dpi=160)
    plt.close(fig)

    # Plot 4: cactus plot of instances solved to each target.
    fig, ax = plt.subplots(figsize=(8, 5))
    for group in groups:
        solved_counts = []
        for t in timepoints:
            count = 0
            for row in rows:
                if row["configuration"] != group:
                    continue
                obj = objective_at_time(row["incumbent_trace"], float(t))
                if obj is not None and obj <= math.floor(1.01 * best_known[row["instance_id"]]):
                    count += 1
            solved_counts.append(count)
        ax.step(timepoints, solved_counts, where="post", label=group)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("instances within 1% of best-known")
    ax.set_title("Cactus plot")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "cactus_plot.png", dpi=160)
    plt.close(fig)

    # Plot 5: worker-local budget 1 vs 2 comparison.
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    budget1 = [
        row for row in rows if row["algorithm"] == "slackpipe" and int(row["worker_move_budget"] or 0) == 1
    ]
    budget2 = [
        row for row in rows if row["algorithm"] == "slackpipe" and int(row["worker_move_budget"] or 0) == 2
    ]
    by_key1 = {(r["instance_id"], int(r["seed"])): r for r in budget1}
    by_key2 = {(r["instance_id"], int(r["seed"])): r for r in budget2}
    common = sorted(set(by_key1) & set(by_key2))
    final_x = [float(by_key1[key]["best_known_gap"]) for key in common]
    final_y = [float(by_key2[key]["best_known_gap"]) for key in common]
    axes[0].scatter(final_x, final_y, alpha=0.7)
    lim = max(final_x + final_y + [1.0])
    axes[0].plot([0, lim], [0, lim], linestyle="--", color="gray")
    axes[0].set_xlabel("budget 1 final gap (%)")
    axes[0].set_ylabel("budget 2 final gap (%)")
    axes[0].set_title("Final quality")
    tt_x = []
    tt_y = []
    for key in common:
        joint = next(
            (
                row
                for row in rows
                if row["instance_id"] == key[0]
                and int(row["seed"]) == key[1]
                and row["algorithm"] == "optimize-joint"
            ),
            None,
        )
        if joint is None:
            continue
        target = int(joint["final_objective"])
        t1 = time_to_target(by_key1[key]["incumbent_trace"], target)
        t2 = time_to_target(by_key2[key]["incumbent_trace"], target)
        if t1 is None or t2 is None:
            continue
        tt_x.append(float(t1))
        tt_y.append(float(t2))
    axes[1].scatter(tt_x, tt_y, alpha=0.7)
    lim2 = max(tt_x + tt_y + [1.0])
    axes[1].plot([0, lim2], [0, lim2], linestyle="--", color="gray")
    axes[1].set_xlabel("budget 1 time-to-target (s)")
    axes[1].set_ylabel("budget 2 time-to-target (s)")
    axes[1].set_title("Time to joint target")
    fig.tight_layout()
    fig.savefig(plots_dir / "worker_local_b1_vs_b2.png", dpi=160)
    plt.close(fig)


def validate_outputs(rows: Sequence[Dict[str, Any]]) -> None:
    for row in rows:
        if row["J"] != DEFAULT_NUM_WORKERS:
            raise ValueError("unexpected worker count in output row")
        if row["N"] % row["J"] != 0:
            raise ValueError("unexpected N % J in output row")
        if row["N"] <= row["J"]:
            raise ValueError("unexpected N <= J in output row")
        if row["B"] <= row["J"]:
            raise ValueError("unexpected B <= J in output row")
        if row["algorithm"] not in {"optimize-joint", "slackpipe"}:
            raise ValueError("unexpected algorithm in output row")
        if row["algorithm"] == "slackpipe":
            if row["split_mode"] != "worker-local":
                raise ValueError("unexpected split mode in output row")
            if int(row["worker_move_budget"] or 0) not in {1, 2}:
                raise ValueError("unexpected worker move budget in output row")
            if row.get("per_worker_delta") not in {"", None}:
                raise ValueError("per-worker-delta must stay disabled")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, default=Path("build/release/slackpipe_cli"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--phase",
        choices=["validate", "pilot", "main", "all"],
        default="all",
        help="which parts of the focused evaluation to run",
    )
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--pilot-seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.pilot_seed != 0:
        raise ValueError("focused pilot screening is fixed to seed 0")

    if not args.cli.exists():
        raise FileNotFoundError(f"slackpipe_cli not found: {args.cli}")

    build_type: Optional[str] = "Release"
    if os.environ.get("SLACKPIPE_BUILD_TYPE"):
        build_type = os.environ["SLACKPIPE_BUILD_TYPE"]
    elif (args.cli.parent / "CMakeCache.txt").exists():
        build_type = None
    build_meta = {
        "git_commit": build_git_commit(),
        "build_type": build_type,
        "hostname": socket.gethostname(),
        "cpu_information": cpu_information(),
    }
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    instances = candidate_instances()
    algorithms = focused_algorithms()

    if args.phase == "validate":
        print("validation-ok")
        return 0

    retained_instances = instances
    pilot_file = output_dir / "pilot_screening.csv"
    if args.phase in {"pilot", "all"}:
        _, retained_instances = pilot_screening(
            cli_path=args.cli,
            output_dir=output_dir,
            build_meta=build_meta,
            instances=instances,
        )
        if args.phase == "pilot":
            return 0
    elif args.phase == "main":
        retained_instances = load_retained_instances_from_pilot_csv(pilot_file)

    if args.phase in {"main", "all"}:
        seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
        rows = run_main_experiments(
            cli_path=args.cli,
            output_dir=output_dir,
            build_meta=build_meta,
            retained_instances=retained_instances,
            seeds=seeds,
        )
        enrich_rows(rows)
        validate_outputs(rows)
        write_analysis_artifacts(output_dir, rows)
        if not args.dry_run:
            plot_artifacts(output_dir, rows)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
