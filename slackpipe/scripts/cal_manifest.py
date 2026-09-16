#!/usr/bin/env python3
"""CAL evaluation manifest and runner utilities for SlackPipe."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


MANIFEST_SCHEMA_VERSION = 1
EXPECTED_SCHEMA_VERSION = 1
EXPECTED_BUDGET_POLICY_VERSION = 1
EXPECTED_EVALUATION_METHOD_VERSION = 1
EXPECTED_VALIDATION_VERSION = 1
EXPECTED_ACTIVATION_ANALYSIS_VERSION = 1
EXPECTED_ACTIVATION_CAP_FORMULATION_VERSION = 1

DEFAULT_MAIN_TIME_LIMIT_SECONDS = 300.0
DEFAULT_BIG_TIME_LIMIT_SECONDS = 300.0
DEFAULT_SMOKE_TIME_LIMIT_SECONDS = 5.0
DEFAULT_SOLVER_THREADS = 1
DEFAULT_BIG_SOLVER_THREADS = 8
DEFAULT_MIN_LAYERS = 1
DEFAULT_SEEDS = [0, 1, 2]
SMOKE_SEEDS = [0]
BIG_ONE_F_ONE_B_PROFILES = {
    "main_big_5min_1f1b",
    "main_big_5min_1f1b_uncapped",
}

COMMUNICATION_TICKS = {
    "none": 0,
    "moderate": 2,
    "heavy": 8,
}


@dataclass(frozen=True)
class SolverMachineryVariant:
    variant_id: str
    label: str
    worker_balance_pruning: bool
    incumbent_method: str
    incumbent_bound: bool
    incumbent_hints: bool
    worker_balance_tolerance_percent: float | None = None
    worker_balance_tolerance_layers: int | None = None


UNIFORM_BREADTH_FIRST = "uniform-breadth-first"
UNIFORM_INTERLEAVED_ONE_F_ONE_B = "uniform-interleaved-1f1b"

MAIN_METHODS = [
    UNIFORM_BREADTH_FIRST,
    "partition-only-fixed-order",
    "schedule-only-uniform",
    "sequential-partition-then-schedule",
    "alternating-partition-schedule",
    "joint-unrestricted-no-overlap",
]

BIG_ONE_F_ONE_B_METHODS = [
    UNIFORM_BREADTH_FIRST,
    UNIFORM_INTERLEAVED_ONE_F_ONE_B,
    "partition-only-fixed-order",
    "schedule-only-uniform",
    "sequential-partition-then-schedule",
    "alternating-partition-schedule",
    "joint-unrestricted-no-overlap",
]

CONTRACTED_METHODS = list(dict.fromkeys(MAIN_METHODS + BIG_ONE_F_ONE_B_METHODS))
DETERMINISTIC_METHODS = {UNIFORM_BREADTH_FIRST, UNIFORM_INTERLEAVED_ONE_F_ONE_B}

SOLVER_BACKED_METHODS = {
    "partition-only-fixed-order",
    "schedule-only-uniform",
    "sequential-partition-then-schedule",
    "alternating-partition-schedule",
    "joint-unrestricted-no-overlap",
}

SOLVER_MACHINERY_METHOD = "joint-unrestricted-no-overlap"

SOLVER_MACHINERY_ABLATION_VARIANTS = [
    SolverMachineryVariant(
        "production-slackpipe",
        "Production SlackPipe",
        False,
        "slack",
        True,
        True,
    ),
    SolverMachineryVariant(
        "canonical-incumbent",
        "Canonical incumbent",
        False,
        "canonical",
        True,
        True,
    ),
    SolverMachineryVariant(
        "no-incumbent-bound",
        "No incumbent bound",
        False,
        "slack",
        False,
        True,
    ),
    SolverMachineryVariant(
        "no-incumbent-hints",
        "No incumbent hints",
        False,
        "slack",
        True,
        False,
    ),
    SolverMachineryVariant(
        "no-bound-or-hints",
        "No bound or hints",
        False,
        "slack",
        False,
        False,
    ),
    SolverMachineryVariant(
        "bare-joint-cpsat",
        "Bare joint CP-SAT",
        False,
        "none",
        False,
        False,
    ),
]


def solver_machinery_balance_pruning_variants(
    *,
    tolerance_percent: float | None = None,
    tolerance_layers: int | None = None,
) -> list[SolverMachineryVariant]:
    if tolerance_percent is None and tolerance_layers is None:
        raise ValueError(
            "solver-machinery balance-pruning profiles require an explicit "
            "worker-balance tolerance"
        )
    return [
        SolverMachineryVariant(
            "production-plus-balance-pruning",
            "Production + balance pruning",
            True,
            "slack",
            True,
            True,
            tolerance_percent,
            tolerance_layers,
        ),
        SolverMachineryVariant(
            "bare-joint-cpsat-plus-balance-pruning",
            "Bare joint CP-SAT + balance pruning",
            True,
            "none",
            False,
            False,
            tolerance_percent,
            tolerance_layers,
        ),
    ]

METHOD_REQUIRES_ORTOOLS = {
    UNIFORM_BREADTH_FIRST: False,
    UNIFORM_INTERLEAVED_ONE_F_ONE_B: False,
    "partition-only-fixed-order": False,
    "schedule-only-uniform": True,
    "sequential-partition-then-schedule": True,
    "alternating-partition-schedule": True,
    "joint-unrestricted-no-overlap": True,
}

FALLBACK_METHOD_CONTRACT_HASHES = {
    UNIFORM_BREADTH_FIRST: "f8e025383ceb9dcc",
    UNIFORM_INTERLEAVED_ONE_F_ONE_B: "0f2b9aaaa049fd5b",
    "partition-only-fixed-order": "9da321e1678f2ee8",
    "schedule-only-uniform": "d2cfc9e9b27a5f83",
    "sequential-partition-then-schedule": "fafb8155a28b6d79",
    "alternating-partition-schedule": "2077dac893d43b88",
    "joint-unrestricted-no-overlap": "5e1e081bea2c06ad",
}

OUTCOME_CLASSES = {
    "completed_valid",
    "completed_no_solution",
    "completed_invalid",
    "unavailable",
    "timeout_or_failed_process",
    "skipped_valid",
    "rerun_invalid",
    "schema_mismatch",
    "manifest_mismatch",
}

TERMINAL_FEASIBLE_STATUSES = {"OPTIMAL", "FEASIBLE"}
UNAVAILABLE_STATUSES = {"UNAVAILABLE", "NOT_AVAILABLE"}
NO_VALID_SOLUTION_STATUS = "NO_VALID_SOLUTION"
NO_SOLUTION_REASONS = {
    "activation_cap_replay_rejected_until_deadline",
    "solver_no_feasible_solution",
    "global_deadline_before_valid_solution",
    "replay_validation_failed",
}
RUNNER_MANIFEST_MARKER = ".slackpipe_cal_manifest.json"


@dataclass(frozen=True)
class Config:
    experiment_group: str
    configuration_id: str
    B: int
    N: int
    W: int
    L: int
    communication_profile: str
    communication_ticks: int


@dataclass
class CompatibilityResult:
    passed: bool
    outcome: str
    reasons: list[str]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def json_dumps_canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_hex(value: Any) -> str:
    return hashlib.sha256(json_dumps_canonical(value).encode("utf-8")).hexdigest()


def fnv1a_hex(text: str) -> str:
    value = 1469598103934665603
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def normalize_token(value: str) -> str:
    return value.replace("-", "_")


def cli_token(value: str) -> str:
    return value.replace("_", "-")


def csv_ints(values: Sequence[int] | None) -> str:
    if values is None:
        return ""
    return ",".join(str(int(value)) for value in values)


def current_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def current_git_dirty() -> bool | None:
    try:
        subprocess.check_call(
            ["git", "diff", "--quiet", "--ignore-submodules", "--"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.check_call(
            ["git", "diff", "--cached", "--quiet", "--ignore-submodules", "--"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return False
    except subprocess.CalledProcessError:
        return True
    except Exception:
        return None


def load_build_info(binary: Path | None) -> dict[str, Any]:
    if binary is None:
        return {}
    try:
        if not binary.exists():
            return {}
        output = subprocess.check_output(
            [str(binary), "build-info"], text=True, stderr=subprocess.DEVNULL
        )
        return json.loads(output)
    except Exception:
        return {}


def load_method_contracts(binary: Path | None) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    if binary is not None and binary.exists():
        for method in CONTRACTED_METHODS:
            try:
                output = subprocess.check_output(
                    [str(binary), "describe-method", method],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                payload = json.loads(output)
                contracts[method] = payload
            except Exception:
                pass
    for method in CONTRACTED_METHODS:
        contracts.setdefault(
            method,
            {
                "canonical_name": method,
                "method_contract_hash": FALLBACK_METHOD_CONTRACT_HASHES[method],
                "requires_ortools": METHOD_REQUIRES_ORTOOLS[method],
                "deterministic": method in DETERMINISTIC_METHODS,
            },
        )
    return contracts


def uniform_split(N: int, L: int, min_layers: int = DEFAULT_MIN_LAYERS) -> list[int]:
    minimum = N * min_layers
    if L < minimum:
        raise ValueError("uniform split requires L >= N * min_layers")
    remaining = L - minimum
    q, r = divmod(remaining, N)
    return [min_layers + q + (1 if stage < r else 0) for stage in range(N)]


def operation_stage(N: int, chain_index: int) -> int:
    return chain_index if chain_index < N else 2 * N - 1 - chain_index


def operation_worker(N: int, W: int, chain_index: int) -> int:
    return operation_stage(N, chain_index) % W


def encode_op(N: int, microbatch: int, chain_index: int) -> int:
    return microbatch * (2 * N) + chain_index


def decode_op(N: int, W: int, op_id: int) -> dict[str, int | bool]:
    chain_index = op_id % (2 * N)
    microbatch = op_id // (2 * N)
    stage = operation_stage(N, chain_index)
    return {
        "microbatch": microbatch,
        "chain_index": chain_index,
        "stage": stage,
        "worker": stage % W,
        "backward": chain_index >= N,
    }


def breadth_first_orders(B: int, N: int, W: int) -> list[list[int]]:
    orders: list[list[int]] = [[] for _ in range(W)]
    for b in range(B):
        for n in range(2 * N):
            worker = operation_worker(N, W, n)
            orders[worker].append(encode_op(N, b, n))

    def rank(op_id: int) -> tuple[int, int, int]:
        view = decode_op(N, W, op_id)
        return (
            int(view["microbatch"]) + int(view["chain_index"]),
            -int(view["chain_index"]),
            int(view["microbatch"]),
        )

    for worker in range(W):
        orders[worker].sort(key=rank)
    return orders


def evaluate_schedule_uniform_bfs(
    *,
    B: int,
    N: int,
    W: int,
    L: int,
    backward_ratio_num: int,
    backward_ratio_den: int,
    communication_ticks: int,
    min_layers: int = DEFAULT_MIN_LAYERS,
) -> dict[int, tuple[int, int, int, int]]:
    split = uniform_split(N, L, min_layers)
    op_count = B * 2 * N
    graph: list[list[tuple[int, int]]] = [[] for _ in range(op_count)]
    indegree = [0] * op_count
    duration = [0] * op_count

    def add_edge(src: int, dst: int, lag: int) -> None:
        graph[src].append((dst, lag))
        indegree[dst] += 1

    for op_id in range(op_count):
        view = decode_op(N, W, op_id)
        stage = int(view["stage"])
        layers = split[stage]
        duration[op_id] = layers * (
            backward_ratio_num if bool(view["backward"]) else backward_ratio_den
        )
        chain_index = int(view["chain_index"])
        b = int(view["microbatch"])
        if chain_index + 1 < 2 * N:
            nxt = encode_op(N, b, chain_index + 1)
            nxt_worker = int(decode_op(N, W, nxt)["worker"])
            lag = 0 if int(view["worker"]) == nxt_worker else communication_ticks
            add_edge(op_id, nxt, lag)
        if b > 0:
            add_edge(encode_op(N, b - 1, chain_index), op_id, 0)

    for order in breadth_first_orders(B, N, W):
        for previous, current in zip(order, order[1:]):
            prev_view = decode_op(N, W, previous)
            curr_view = decode_op(N, W, current)
            if (
                int(prev_view["worker"]) == int(curr_view["worker"])
                and int(prev_view["chain_index"]) != int(curr_view["chain_index"])
                and not (
                    int(curr_view["chain_index"]) > 0
                    and previous
                    == encode_op(
                        N,
                        int(curr_view["microbatch"]),
                        int(curr_view["chain_index"]) - 1,
                    )
                )
            ):
                add_edge(previous, current, 0)

    ready = [op for op, degree in enumerate(indegree) if degree == 0]
    start = [0] * op_count
    operations: dict[int, tuple[int, int, int, int]] = {}
    visited = 0
    while ready:
        current = ready.pop(0)
        visited += 1
        view = decode_op(N, W, current)
        end = start[current] + duration[current]
        operations[current] = (start[current], end, int(view["worker"]), duration[current])
        for nxt, lag in graph[current]:
            start[nxt] = max(start[nxt], end + lag)
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
    if visited != op_count:
        raise ValueError("cycle detected in Python uniform BFS evaluator")
    return operations


def derive_uniform_baseline_cap(row: Mapping[str, Any]) -> dict[str, Any]:
    split = uniform_split(int(row["N"]), int(row["L"]), DEFAULT_MIN_LAYERS)
    operations = evaluate_schedule_uniform_bfs(
        B=int(row["B"]),
        N=int(row["N"]),
        W=int(row["W"]),
        L=int(row["L"]),
        backward_ratio_num=int(row["backward_cost_ratio_num"]),
        backward_ratio_den=int(row["backward_cost_ratio_den"]),
        communication_ticks=int(row["communication_ticks"]),
    )
    live_events: list[tuple[int, int, int, int, int, int]] = []
    for b in range(int(row["B"])):
        for stage in range(int(row["N"])):
            forward = operations[encode_op(int(row["N"]), b, stage)]
            backward_index = 2 * int(row["N"]) - 1 - stage
            backward = operations[encode_op(int(row["N"]), b, backward_index)]
            start = forward[1]
            end = backward[0]
            if end <= start:
                continue
            if row["activation_model"] == "count":
                demand = 1
            elif row["activation_model"] == "linear_in_stage_layers":
                demand = int(row["activation_units_per_layer"]) * split[stage]
            else:
                explicit = row.get("explicit_stage_activation_units") or []
                demand = int(explicit[stage])
            worker = stage % int(row["W"])
            live_events.append((start, 1, worker, stage, b, demand))
            live_events.append((end, 0, worker, stage, b, demand))
    live_events.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
    live_units = [0] * int(row["W"])
    peak_units = [0] * int(row["W"])
    for _, event_kind, worker, _, _, demand in live_events:
        if event_kind == 0:
            live_units[worker] -= demand
        else:
            live_units[worker] += demand
            peak_units[worker] = max(peak_units[worker], live_units[worker])

    method_hash = row.get("method_contract_hash_expected_for_uniform") or (
        FALLBACK_METHOD_CONTRACT_HASHES[UNIFORM_BREADTH_FIRST]
    )
    seed_text = (
        f"activation-analysis-v{EXPECTED_ACTIVATION_ANALYSIS_VERSION}|"
        f"B={row['B']}|N={row['N']}|W={row['W']}|L={row['L']}|"
        f"min={DEFAULT_MIN_LAYERS}|"
        f"ratio={row['backward_cost_ratio_num']}/{row['backward_cost_ratio_den']}|"
        f"comm={row['communication_ticks']}|"
        f"model={row['activation_model']}|"
        f"units_per_layer={row['activation_units_per_layer']}|"
        f"stage_units={json_dumps_tick_vector(row.get('explicit_stage_activation_units') or [])}|"
        f"bytes=null|partition={json_dumps_tick_vector(split)}|"
        f"method_contract_hash={method_hash}"
    )
    cap_hash = fnv1a_hex(seed_text)
    return {
        "cap_units_per_worker": peak_units,
        "cap_derivation_hash": cap_hash,
        "baseline_partition": split,
        "baseline_run_id": "activation-uniform-" + cap_hash,
        "baseline_method_contract_hash": method_hash,
    }


def json_dumps_tick_vector(values: Sequence[int]) -> str:
    return "[" + ",".join(str(int(value)) for value in values) + "]"


def stage_to_worker_mapping(N: int, W: int) -> list[dict[str, int]]:
    return [{"stage": stage, "worker": stage % W} for stage in range(N)]


def configuration_id(config: Config) -> str:
    return config.configuration_id


def main_configs() -> list[Config]:
    configs: list[Config] = []
    for W in [2, 4, 8]:
        for stage_multiplier in [1, 2, 4]:
            N = W * stage_multiplier
            for ratio in [Fraction(1, 2), Fraction(1, 1), Fraction(2, 1)]:
                B = int(N * ratio)
                for profile, ticks in COMMUNICATION_TICKS.items():
                    L = 8 * N
                    config_id = (
                        f"W{W}_N{N}_B{B}_L{L}_comm-{profile}"
                    )
                    configs.append(
                        Config(
                            "main",
                            config_id,
                            B,
                            N,
                            W,
                            L,
                            profile,
                            ticks,
                        )
                    )
    return configs


def smoke_configs() -> list[Config]:
    return [
        Config("smoke", "smoke_W2_N2_B2_L4_comm-none", 2, 2, 2, 4, "none", 0)
    ]


def smoke_big_one_f_one_b_configs() -> list[Config]:
    return [
        Config(
            "smoke_big_1f1b",
            "smoke_big_W2_N4_B4_L16_comm-none",
            4,
            4,
            2,
            16,
            "none",
            0,
        )
    ]


def ablation_configs() -> list[Config]:
    return [
        Config("ablation", "ablation_small_W2_N2_B2_L4", 2, 2, 2, 4, "moderate", 2),
        Config("ablation", "ablation_medium_W4_N8_B8_L64", 8, 8, 4, 64, "moderate", 2),
        Config("ablation", "ablation_large_W8_N16_B16_L128", 16, 16, 8, 128, "moderate", 2),
    ]


def oracle_configs() -> list[Config]:
    return [
        Config("oracle", "oracle_W2_N2_B1_L4_comm-none", 1, 2, 2, 4, "none", 0),
        Config("oracle", "oracle_W2_N3_B2_L6_comm-moderate", 2, 3, 2, 6, "moderate", 2),
    ]


def seeds_for_method(method: str, seeds: Sequence[int], smoke: bool) -> list[int]:
    if method in DETERMINISTIC_METHODS:
        return [int(seeds[0])]
    if smoke:
        return [int(seeds[0])]
    return [int(seed) for seed in seeds]


def activation_modes(include_equal_memory: bool = True) -> list[str]:
    modes = ["uncapped"]
    if include_equal_memory:
        modes.append("equal_memory")
    return modes


def base_row(
    *,
    manifest_id: str,
    group: str,
    config: Config,
    method: str,
    activation_mode: str,
    seed: int,
    repetition_id: str,
    contracts: Mapping[str, Mapping[str, Any]],
    build_info: Mapping[str, Any],
    time_limit_seconds: float,
    solver_threads: int,
    fixed_order_partition_backend: str | None = "cpsat",
    alternating_max_rounds: int | None = None,
    solver_machinery_variant: SolverMachineryVariant | None = None,
) -> dict[str, Any]:
    is_equal_memory = activation_mode == "equal_memory"
    variant_id = (
        solver_machinery_variant.variant_id if solver_machinery_variant else None
    )
    variant_fragment = f"__{variant_id}" if variant_id else ""
    run_id = (
        f"{group}__{config.configuration_id}__{activation_mode}__"
        f"{method}{variant_fragment}__{repetition_id}"
    )
    rel_dir = Path(group) / config.configuration_id / activation_mode / method
    if variant_id:
        rel_dir = rel_dir / variant_id
    rel_dir = rel_dir / repetition_id
    method_contract = contracts[method]
    git_dirty = build_info.get("git_dirty")
    if git_dirty is None:
        git_dirty = current_git_dirty()
    git_commit = build_info.get("git_commit") or current_git_commit()
    row: dict[str, Any] = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_id": manifest_id,
        "manifest_hash": None,
        "run_id": run_id,
        "experiment_group": group,
        "configuration_id": config.configuration_id,
        "repetition_id": repetition_id,
        "method": method,
        "canonical_method_expected": method,
        "B": config.B,
        "N": config.N,
        "W": config.W,
        "L": config.L,
        "mapping_type": "cyclic_stage_mod_worker",
        "stage_to_worker_mapping": stage_to_worker_mapping(config.N, config.W),
        "forward_cost_ratio_num": 1,
        "forward_cost_ratio_den": 1,
        "backward_cost_ratio_num": 2,
        "backward_cost_ratio_den": 1,
        "communication_profile": config.communication_profile,
        "communication_ticks": config.communication_ticks,
        "communication_model": "constant_inter_worker_delay",
        "activation_model": "linear_in_stage_layers",
        "activation_units_per_layer": 1,
        "explicit_stage_activation_units": None,
        "activation_bytes_per_unit": None,
        "activation_cap_mode": "uniform_baseline" if is_equal_memory else "none",
        "activation_cap_units_per_worker": None,
        "activation_cap_source": "uniform_baseline" if is_equal_memory else None,
        "activation_cap_derivation_hash": None,
        "activation_cap_baseline_method": UNIFORM_BREADTH_FIRST
        if is_equal_memory
        else None,
        "enforce_activation_cap": is_equal_memory,
        "fixed_order_partition_backend": fixed_order_partition_backend
        if method in {
            "partition-only-fixed-order",
            "sequential-partition-then-schedule",
            "alternating-partition-schedule",
        }
        else None,
        "time_limit_seconds": float(time_limit_seconds),
        "random_seed": int(seed),
        "solver_threads": int(solver_threads),
        "alternating_max_rounds": alternating_max_rounds
        if method == "alternating-partition-schedule"
        else None,
        "output_json": str(rel_dir / "result.json"),
        "stdout_log": str(rel_dir / "stdout.log"),
        "stderr_log": str(rel_dir / "stderr.log"),
        "expected_schema_version": EXPECTED_SCHEMA_VERSION,
        "expected_budget_policy_version": EXPECTED_BUDGET_POLICY_VERSION,
        "expected_evaluation_method_version": EXPECTED_EVALUATION_METHOD_VERSION,
        "expected_validation_version": EXPECTED_VALIDATION_VERSION,
        "expected_activation_analysis_version": EXPECTED_ACTIVATION_ANALYSIS_VERSION,
        "expected_activation_cap_formulation_version": (
            EXPECTED_ACTIVATION_CAP_FORMULATION_VERSION
        ),
        "method_contract_hash_expected": method_contract["method_contract_hash"],
        "method_contract_hash_expected_for_uniform": contracts[
            UNIFORM_BREADTH_FIRST
        ]["method_contract_hash"],
        "git_commit_expected": git_commit,
        "git_dirty_policy": "allow_dirty_build" if git_dirty else "require_clean",
        "requires_ortools": bool(
            METHOD_REQUIRES_ORTOOLS.get(method, False)
            or row_requires_cpsat_backend(method, fixed_order_partition_backend)
        ),
        "production_or_smoke": "smoke"
        if group.startswith("smoke") or group.endswith("_smoke")
        else "production",
    }
    if solver_machinery_variant is not None:
        row.update(
            {
                "solver_machinery_variant": solver_machinery_variant.variant_id,
                "solver_machinery_variant_label": solver_machinery_variant.label,
                "worker_balance_pruning_requested": (
                    solver_machinery_variant.worker_balance_pruning
                ),
                "worker_balance_tolerance_percent": (
                    solver_machinery_variant.worker_balance_tolerance_percent
                ),
                "worker_balance_tolerance_layers": (
                    solver_machinery_variant.worker_balance_tolerance_layers
                ),
                "incumbent_method_requested": (
                    solver_machinery_variant.incumbent_method
                ),
                "incumbent_bound_requested": (
                    solver_machinery_variant.incumbent_bound
                ),
                "incumbent_hints_requested": (
                    solver_machinery_variant.incumbent_hints
                ),
                "incumbent_fallback_requested": (
                    solver_machinery_variant.incumbent_method != "none"
                ),
            }
        )
    return row


def row_requires_cpsat_backend(method: str, backend: str | None) -> bool:
    return method in {
        "partition-only-fixed-order",
        "sequential-partition-then-schedule",
        "alternating-partition-schedule",
    } and backend == "cpsat"


def materialize_equal_memory_caps(rows: list[dict[str, Any]]) -> None:
    by_config: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["activation_cap_mode"] != "uniform_baseline":
            continue
        config_id = row["configuration_id"]
        if config_id not in by_config:
            cap = derive_uniform_baseline_cap(row)
            by_config[config_id] = cap
        cap = by_config[config_id]
        row["activation_cap_units_per_worker"] = cap["cap_units_per_worker"]
        row["activation_cap_derivation_hash"] = cap["cap_derivation_hash"]
        row["activation_cap_baseline_partition"] = cap["baseline_partition"]
        row["activation_cap_baseline_run_id"] = cap["baseline_run_id"]
        row["activation_cap_baseline_method"] = UNIFORM_BREADTH_FIRST
        row["activation_cap_baseline_method_contract_hash"] = cap[
            "baseline_method_contract_hash"
        ]


def profile_rows(
    *,
    profile: str,
    manifest_id: str,
    contracts: Mapping[str, Mapping[str, Any]],
    build_info: Mapping[str, Any],
    time_limit_seconds: float | None = None,
    solver_threads: int | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    worker_balance_tolerance_percent: float | None = None,
    worker_balance_tolerance_layers: int | None = None,
) -> list[dict[str, Any]]:
    effective_solver_threads = (
        DEFAULT_BIG_SOLVER_THREADS
        if profile in BIG_ONE_F_ONE_B_PROFILES and solver_threads is None
        else (DEFAULT_SOLVER_THREADS if solver_threads is None else int(solver_threads))
    )
    include_equal_memory = True
    solver_machinery_variants: Sequence[SolverMachineryVariant | None] = [None]
    if profile == "smoke":
        configs = smoke_configs()
        methods = MAIN_METHODS
        smoke = True
        groups = ["smoke"]
        limit = time_limit_seconds or DEFAULT_SMOKE_TIME_LIMIT_SECONDS
    elif profile == "smoke_big_1f1b":
        configs = smoke_big_one_f_one_b_configs()
        methods = BIG_ONE_F_ONE_B_METHODS
        smoke = True
        groups = ["smoke_big_1f1b"]
        limit = time_limit_seconds or DEFAULT_SMOKE_TIME_LIMIT_SECONDS
    elif profile == "main":
        configs = main_configs()
        methods = MAIN_METHODS
        smoke = False
        groups = ["main"]
        limit = time_limit_seconds or DEFAULT_MAIN_TIME_LIMIT_SECONDS
    elif profile == "main_big_5min_1f1b":
        configs = main_configs()
        methods = BIG_ONE_F_ONE_B_METHODS
        smoke = False
        groups = ["main_big_5min_1f1b"]
        limit = time_limit_seconds or DEFAULT_BIG_TIME_LIMIT_SECONDS
    elif profile == "main_big_5min_1f1b_uncapped":
        configs = main_configs()
        methods = BIG_ONE_F_ONE_B_METHODS
        smoke = False
        groups = ["main_big_5min_1f1b_uncapped"]
        limit = time_limit_seconds or DEFAULT_BIG_TIME_LIMIT_SECONDS
        include_equal_memory = False
    elif profile == "solver_machinery_ablation_smoke":
        configs = ablation_configs()
        methods = [SOLVER_MACHINERY_METHOD]
        smoke = True
        groups = ["solver_machinery_ablation_smoke"]
        limit = time_limit_seconds or DEFAULT_SMOKE_TIME_LIMIT_SECONDS
        include_equal_memory = False
        solver_machinery_variants = SOLVER_MACHINERY_ABLATION_VARIANTS
    elif profile == "solver_machinery_ablation":
        configs = main_configs()
        methods = [SOLVER_MACHINERY_METHOD]
        smoke = False
        groups = ["solver_machinery_ablation"]
        limit = time_limit_seconds or DEFAULT_MAIN_TIME_LIMIT_SECONDS
        solver_machinery_variants = SOLVER_MACHINERY_ABLATION_VARIANTS
    elif profile == "solver_machinery_balance_pruning_smoke":
        configs = ablation_configs()
        methods = [SOLVER_MACHINERY_METHOD]
        smoke = True
        groups = ["solver_machinery_balance_pruning_smoke"]
        limit = time_limit_seconds or DEFAULT_SMOKE_TIME_LIMIT_SECONDS
        include_equal_memory = False
        solver_machinery_variants = solver_machinery_balance_pruning_variants(
            tolerance_percent=worker_balance_tolerance_percent,
            tolerance_layers=worker_balance_tolerance_layers,
        )
    elif profile == "solver_machinery_balance_pruning":
        configs = main_configs()
        methods = [SOLVER_MACHINERY_METHOD]
        smoke = False
        groups = ["solver_machinery_balance_pruning"]
        limit = time_limit_seconds or DEFAULT_MAIN_TIME_LIMIT_SECONDS
        solver_machinery_variants = solver_machinery_balance_pruning_variants(
            tolerance_percent=worker_balance_tolerance_percent,
            tolerance_layers=worker_balance_tolerance_layers,
        )
    elif profile == "ablation":
        configs = ablation_configs()
        methods = MAIN_METHODS
        smoke = False
        groups = ["ablation"]
        limit = time_limit_seconds or DEFAULT_MAIN_TIME_LIMIT_SECONDS
    elif profile == "oracle":
        configs = oracle_configs()
        methods = ["partition-only-fixed-order"]
        smoke = False
        groups = ["oracle"]
        limit = time_limit_seconds or DEFAULT_SMOKE_TIME_LIMIT_SECONDS
    elif profile == "all":
        rows: list[dict[str, Any]] = []
        for child in ["smoke", "main", "ablation", "oracle"]:
            rows.extend(
                profile_rows(
                    profile=child,
                    manifest_id=manifest_id,
                    contracts=contracts,
                    build_info=build_info,
                    time_limit_seconds=time_limit_seconds,
                    solver_threads=effective_solver_threads,
                    seeds=seeds,
                    worker_balance_tolerance_percent=worker_balance_tolerance_percent,
                    worker_balance_tolerance_layers=worker_balance_tolerance_layers,
                )
            )
        return rows
    else:
        raise ValueError(f"unknown profile: {profile}")

    rows = []
    for group in groups:
        for config in configs:
            for activation_mode in activation_modes(
                include_equal_memory=include_equal_memory
            ):
                variants: Sequence[SolverMachineryVariant | None]
                if group.startswith("solver_machinery"):
                    variants = solver_machinery_variants
                else:
                    variants = [None]
                for method in methods:
                    backend = "enumerate" if group == "oracle" else "cpsat"
                    for variant in variants:
                        for seed in seeds_for_method(
                            method, seeds, smoke or group == "oracle"
                        ):
                            repetition_id = (
                                "rep0"
                                if method in DETERMINISTIC_METHODS
                                else f"seed{seed}"
                            )
                            if group == "oracle":
                                repetition_id = f"oracle-seed{seed}"
                            rows.append(
                                base_row(
                                    manifest_id=manifest_id,
                                    group=group,
                                    config=config,
                                    method=method,
                                    activation_mode=activation_mode,
                                    seed=seed,
                                    repetition_id=repetition_id,
                                    contracts=contracts,
                                    build_info=build_info,
                                    time_limit_seconds=limit,
                                    solver_threads=effective_solver_threads,
                                    fixed_order_partition_backend=backend,
                                    alternating_max_rounds=1
                                    if group.startswith("smoke")
                                    else 4,
                                    solver_machinery_variant=variant,
                                )
                            )
    materialize_equal_memory_caps(rows)
    return rows


def manifest_hash_payload(
    rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    normalized_rows = []
    for row in rows:
        item = dict(row)
        item.pop("manifest_hash", None)
        normalized_rows.append(item)
    metadata_subset = {
        key: metadata[key]
        for key in [
            "manifest_schema_version",
            "manifest_name",
            "manifest_description",
            "matrix_definition",
            "total_run_count",
            "methods",
            "configurations",
            "seeds",
            "default_time_limits",
            "default_solver_threads",
            "ortools_requirement",
            "production_or_smoke",
            "communication_ticks",
        ]
        if key in metadata
    }
    return {"metadata": metadata_subset, "rows": normalized_rows}


def assign_manifest_hash(
    rows: list[dict[str, Any]], metadata: dict[str, Any]
) -> str:
    manifest_hash = sha256_hex(manifest_hash_payload(rows, metadata))
    for row in rows:
        row["manifest_hash"] = manifest_hash
    metadata["manifest_hash"] = manifest_hash
    return manifest_hash


def metadata_for_profile(
    *,
    profile: str,
    manifest_id: str,
    rows: Sequence[Mapping[str, Any]],
    build_info: Mapping[str, Any],
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    configurations = sorted({str(row["configuration_id"]) for row in rows})
    methods = sorted({str(row["method"]) for row in rows})
    seeds = sorted({int(row["random_seed"]) for row in rows})
    default_time_limits = {
        "smoke_seconds": DEFAULT_SMOKE_TIME_LIMIT_SECONDS,
        "main_seconds": DEFAULT_MAIN_TIME_LIMIT_SECONDS,
        "main_big_5min_1f1b_seconds": DEFAULT_BIG_TIME_LIMIT_SECONDS,
    }
    default_solver_threads = {
        "default": DEFAULT_SOLVER_THREADS,
        "main_big_5min_1f1b": DEFAULT_BIG_SOLVER_THREADS,
    }
    if profile == "main_big_5min_1f1b_uncapped":
        default_time_limits["main_big_5min_1f1b_uncapped_seconds"] = (
            DEFAULT_BIG_TIME_LIMIT_SECONDS
        )
        default_solver_threads["main_big_5min_1f1b_uncapped"] = (
            DEFAULT_BIG_SOLVER_THREADS
        )
    if profile.startswith("solver_machinery"):
        default_time_limits["solver_machinery_ablation_seconds"] = (
            DEFAULT_MAIN_TIME_LIMIT_SECONDS
        )
    metadata: dict[str, Any] = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_id": manifest_id,
        "manifest_hash": None,
        "generated_at_utc": generated_at_utc or utc_now(),
        "generator_git_commit": current_git_commit(),
        "generator_git_dirty": current_git_dirty(),
        "binary_git_commit": build_info.get("git_commit"),
        "binary_git_dirty": build_info.get("git_dirty"),
        "host_info": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "manifest_name": f"cal_{profile}",
        "manifest_description": (
            "SlackPipe CAL evaluation manifest; every JSONL row is one method run."
        ),
        "matrix_definition": matrix_definition(profile),
        "total_run_count": len(rows),
        "methods": methods,
        "configurations": configurations,
        "seeds": seeds,
        "default_time_limits": default_time_limits,
        "default_solver_threads": default_solver_threads,
        "ortools_requirement": (
            "required for solver-backed rows and solver-enforced equal-memory caps"
        ),
        "production_or_smoke": "smoke"
        if profile.startswith("smoke") or profile.endswith("_smoke")
        else "production",
        "communication_ticks": COMMUNICATION_TICKS,
    }
    return metadata


def matrix_definition(profile: str) -> dict[str, Any]:
    if profile == "main":
        return {
            "W": [2, 4, 8],
            "stage_multiplier": [1, 2, 4],
            "N": "W * stage_multiplier",
            "batch_ratio": [0.5, 1.0, 2.0],
            "B": "int(N * batch_ratio)",
            "L": "8 * N",
            "communication_profiles": COMMUNICATION_TICKS,
            "activation_modes": ["uncapped", "equal_memory"],
            "row_count_formula": "81 configurations * (2 activation modes * (1 deterministic + 5 solver-backed * 3 seeds)) = 2592",
        }
    if profile == "main_big_5min_1f1b":
        return {
            "W": [2, 4, 8],
            "stage_multiplier": [1, 2, 4],
            "N": "W * stage_multiplier",
            "batch_ratio": [0.5, 1.0, 2.0],
            "B": "int(N * batch_ratio)",
            "L": "8 * N",
            "communication_profiles": COMMUNICATION_TICKS,
            "activation_modes": ["uncapped", "equal_memory"],
            "methods": BIG_ONE_F_ONE_B_METHODS,
            "time_limit_seconds": DEFAULT_BIG_TIME_LIMIT_SECONDS,
            "solver_threads": DEFAULT_BIG_SOLVER_THREADS,
            "fixed_order_partition_backend": "cpsat",
            "row_count_formula": "81 configurations * 2 activation modes * (2 deterministic + 5 solver-backed * 3 seeds) = 2754",
        }
    if profile == "main_big_5min_1f1b_uncapped":
        return {
            "W": [2, 4, 8],
            "stage_multiplier": [1, 2, 4],
            "N": "W * stage_multiplier",
            "batch_ratio": [0.5, 1.0, 2.0],
            "B": "int(N * batch_ratio)",
            "L": "8 * N",
            "communication_profiles": COMMUNICATION_TICKS,
            "activation_modes": ["uncapped"],
            "methods": BIG_ONE_F_ONE_B_METHODS,
            "time_limit_seconds": DEFAULT_BIG_TIME_LIMIT_SECONDS,
            "solver_threads": DEFAULT_BIG_SOLVER_THREADS,
            "fixed_order_partition_backend": "cpsat",
            "row_count_formula": "81 configurations * (2 deterministic + 5 solver-backed * 3 seeds) = 1377",
        }
    if profile == "smoke":
        return {
            "configurations": [config.configuration_id for config in smoke_configs()],
            "activation_modes": ["uncapped", "equal_memory"],
            "methods": MAIN_METHODS,
            "row_count_formula": "1 configuration * 2 activation modes * 6 methods * 1 seed = 12",
        }
    if profile == "smoke_big_1f1b":
        return {
            "configurations": [
                config.configuration_id
                for config in smoke_big_one_f_one_b_configs()
            ],
            "activation_modes": ["uncapped", "equal_memory"],
            "methods": BIG_ONE_F_ONE_B_METHODS,
            "row_count_formula": "1 configuration * 2 activation modes * 7 methods * 1 seed = 14",
        }
    if profile == "solver_machinery_ablation_smoke":
        return {
            "configurations": [config.configuration_id for config in ablation_configs()],
            "activation_modes": ["uncapped"],
            "methods": [SOLVER_MACHINERY_METHOD],
            "solver_machinery_variants": [
                variant.variant_id for variant in SOLVER_MACHINERY_ABLATION_VARIANTS
            ],
            "row_count_formula": "3 configurations * 1 activation mode * 6 solver-machinery variants * 1 seed = 18",
        }
    if profile == "solver_machinery_ablation":
        return {
            "W": [2, 4, 8],
            "stage_multiplier": [1, 2, 4],
            "N": "W * stage_multiplier",
            "batch_ratio": [0.5, 1.0, 2.0],
            "B": "int(N * batch_ratio)",
            "L": "8 * N",
            "communication_profiles": COMMUNICATION_TICKS,
            "activation_modes": ["uncapped", "equal_memory"],
            "methods": [SOLVER_MACHINERY_METHOD],
            "solver_machinery_variants": [
                variant.variant_id for variant in SOLVER_MACHINERY_ABLATION_VARIANTS
            ],
            "row_count_formula": "81 configurations * 2 activation modes * 6 solver-machinery variants * 3 seeds = 2916",
        }
    if profile == "solver_machinery_balance_pruning_smoke":
        return {
            "configurations": [config.configuration_id for config in ablation_configs()],
            "activation_modes": ["uncapped"],
            "methods": [SOLVER_MACHINERY_METHOD],
            "solver_machinery_variants": [
                "production-plus-balance-pruning",
                "bare-joint-cpsat-plus-balance-pruning",
            ],
            "worker_balance_tolerance": "required explicitly at manifest generation",
            "row_count_formula": "3 configurations * 1 activation mode * 2 balance-pruning variants * 1 seed = 6",
        }
    if profile == "solver_machinery_balance_pruning":
        return {
            "W": [2, 4, 8],
            "stage_multiplier": [1, 2, 4],
            "N": "W * stage_multiplier",
            "batch_ratio": [0.5, 1.0, 2.0],
            "B": "int(N * batch_ratio)",
            "L": "8 * N",
            "communication_profiles": COMMUNICATION_TICKS,
            "activation_modes": ["uncapped", "equal_memory"],
            "methods": [SOLVER_MACHINERY_METHOD],
            "solver_machinery_variants": [
                "production-plus-balance-pruning",
                "bare-joint-cpsat-plus-balance-pruning",
            ],
            "worker_balance_tolerance": "required explicitly at manifest generation",
            "row_count_formula": "81 configurations * 2 activation modes * 2 balance-pruning variants * 3 seeds = 972",
        }
    if profile == "ablation":
        return {
            "configurations": [config.configuration_id for config in ablation_configs()],
            "policy": "contract-hashed canonical CAL baseline variants only",
        }
    if profile == "oracle":
        return {
            "configurations": [config.configuration_id for config in oracle_configs()],
            "methods": ["partition-only-fixed-order"],
            "fixed_order_partition_backend": "enumerate",
            "policy": "oracle rows are separate manifest rows, not hidden baseline objectives",
        }
    if profile == "all":
        return {"profiles": ["smoke", "main", "ablation", "oracle"]}
    raise ValueError(f"unknown profile: {profile}")


def generate_manifest(
    *,
    profile: str,
    binary: Path | None = None,
    time_limit_seconds: float | None = None,
    solver_threads: int | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    generated_at_utc: str | None = None,
    worker_balance_tolerance_percent: float | None = None,
    worker_balance_tolerance_layers: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    build_info = load_build_info(binary)
    contracts = load_method_contracts(binary)
    manifest_id = f"cal-{profile}"
    rows = profile_rows(
        profile=profile,
        manifest_id=manifest_id,
        contracts=contracts,
        build_info=build_info,
        time_limit_seconds=time_limit_seconds,
        solver_threads=solver_threads,
        seeds=seeds,
        worker_balance_tolerance_percent=worker_balance_tolerance_percent,
        worker_balance_tolerance_layers=worker_balance_tolerance_layers,
    )
    metadata = metadata_for_profile(
        profile=profile,
        manifest_id=manifest_id,
        rows=rows,
        build_info=build_info,
        generated_at_utc=generated_at_utc,
    )
    assign_manifest_hash(rows, metadata)
    return rows, metadata


def write_manifest(
    rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any], output: Path
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json_dumps_canonical(row) + "\n")
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if row.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
                raise ValueError(f"{path}:{line_number}: unsupported schema")
            rows.append(row)
    run_ids = [row["run_id"] for row in rows]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("manifest contains duplicate run_id values")
    return rows


def resolve_output_path(output_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return output_root / path


def is_oracle_enumeration_row(row: Mapping[str, Any]) -> bool:
    return (
        row.get("experiment_group") == "oracle"
        and row.get("method") == "partition-only-fixed-order"
        and row.get("fixed_order_partition_backend") == "enumerate"
    )


def oracle_enumeration_cap_mode(row: Mapping[str, Any], canonical: Mapping[str, Any]) -> str | None:
    if not is_oracle_enumeration_row(row) or not row.get("enforce_activation_cap"):
        return None
    mode = canonical.get("activation_cap_enforcement_mode")
    if mode in {"exact_enumeration", "partial_enumeration"}:
        return str(mode)
    return None


def oracle_exact_enumeration_cap_compatible(
    row: Mapping[str, Any], canonical: Mapping[str, Any]
) -> bool:
    return (
        oracle_enumeration_cap_mode(row, canonical) == "exact_enumeration"
        and canonical.get("activation_cap_enforcement_requested") is True
        and canonical.get("fixed_order_partition_backend_effective") == "enumerate"
        and canonical.get("activation_cap_enforced_by_enumeration") is True
        and canonical.get("activation_cap_enforced_in_solver") is not True
        and canonical.get("activation_cap_constraints_added") is not True
        and canonical.get("cp_sat_launched") is False
        and canonical.get("cp_sat_models_solved") == 0
        and canonical.get("enumeration_proved_optimal") is True
        and canonical.get("optimality_proof_source") == "exhaustive_enumeration"
    )


def oracle_partial_enumeration_cap_compatible(
    row: Mapping[str, Any], canonical: Mapping[str, Any]
) -> bool:
    return (
        oracle_enumeration_cap_mode(row, canonical) == "partial_enumeration"
        and canonical.get("activation_cap_enforcement_requested") is True
        and canonical.get("fixed_order_partition_backend_effective") == "enumerate"
        and canonical.get("activation_cap_enforced_by_enumeration") is True
        and canonical.get("activation_cap_enforced_in_solver") is not True
        and canonical.get("activation_cap_constraints_added") is not True
        and canonical.get("cp_sat_launched") is False
        and canonical.get("cp_sat_models_solved") == 0
        and canonical.get("enumeration_proved_optimal") is not True
        and canonical.get("optimal") is not True
        and canonical.get("reported_status") != "OPTIMAL"
    )


def command_for_row(row: Mapping[str, Any], binary: Path, output_root: Path) -> list[str]:
    output_json = resolve_output_path(output_root, str(row["output_json"]))
    output_prefix = output_json.with_suffix("")
    command = [
        str(binary),
        "--B",
        str(row["B"]),
        "--N",
        str(row["N"]),
        "--J",
        str(row["W"]),
        "--L",
        str(row["L"]),
        "--min-layers",
        str(DEFAULT_MIN_LAYERS),
        "--ratio-num",
        str(row["backward_cost_ratio_num"]),
        "--ratio-den",
        str(row["backward_cost_ratio_den"]),
        "--communication",
        str(row["communication_ticks"]),
        "--algorithm",
        str(row["method"]),
        "--time-limit-seconds",
        str(row["time_limit_seconds"]),
        "--num-workers",
        str(row["solver_threads"]),
        "--random-seed",
        str(row["random_seed"]),
        "--require-optimal",
        "false",
        "--symmetry-break-f0-fifo",
        "true",
        "--activation-model",
        cli_token(str(row["activation_model"])),
        "--activation-units-per-layer",
        str(row["activation_units_per_layer"]),
        "--activation-cap-mode",
        cli_token(str(row["activation_cap_mode"])),
        "--output-prefix",
        str(output_prefix),
    ]
    if row.get("activation_bytes_per_unit") is not None:
        command.extend(["--activation-bytes-per-unit", str(row["activation_bytes_per_unit"])])
    if row.get("explicit_stage_activation_units") is not None:
        command.extend(
            ["--activation-stage-units", csv_ints(row["explicit_stage_activation_units"])]
        )
    if row.get("activation_cap_units_per_worker") is not None:
        command.extend(
            ["--activation-cap-units", csv_ints(row["activation_cap_units_per_worker"])]
        )
    if row.get("activation_cap_derivation_hash"):
        command.extend(
            ["--activation-cap-derivation-hash", str(row["activation_cap_derivation_hash"])]
        )
    if row.get("enforce_activation_cap"):
        enforcement = (
            "exact-enumeration"
            if is_oracle_enumeration_row(row)
            else "solver"
        )
        command.extend(["--activation-cap-enforcement", enforcement])
    if row.get("fixed_order_partition_backend") is not None:
        command.extend(
            ["--fixed-order-partition-backend", str(row["fixed_order_partition_backend"])]
        )
    if row.get("solver_machinery_variant"):
        command.extend(
            [
                "--worker-balance-pruning",
                "on" if row.get("worker_balance_pruning_requested") else "off",
                "--incumbent-method",
                str(row["incumbent_method_requested"]),
                "--incumbent-bound",
                "on" if row.get("incumbent_bound_requested") else "off",
                "--incumbent-hints",
                "on" if row.get("incumbent_hints_requested") else "off",
            ]
        )
        if row.get("worker_balance_tolerance_percent") is not None:
            command.extend(
                [
                    "--worker-balance-tolerance-percent",
                    str(row["worker_balance_tolerance_percent"]),
                ]
            )
        if row.get("worker_balance_tolerance_layers") is not None:
            command.extend(
                [
                    "--worker-balance-tolerance-layers",
                    str(row["worker_balance_tolerance_layers"]),
                ]
            )
    if row.get("alternating_max_rounds") is not None:
        command.extend(["--alternating-max-rounds", str(row["alternating_max_rounds"])])
    return command


def canonical_result(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    result = payload.get("canonical_result")
    if isinstance(result, Mapping):
        return result
    if isinstance(payload.get("canonical"), Mapping):
        return payload["canonical"]  # type: ignore[index]
    return payload


def no_solution_compatibility_reasons(
    row: Mapping[str, Any], canonical: Mapping[str, Any]
) -> list[str]:
    reasons: list[str] = []
    if canonical.get("reported_status") != NO_VALID_SOLUTION_STATUS:
        reasons.append("no-solution row must report NO_VALID_SOLUTION")
    if canonical.get("feasible") is not False:
        reasons.append("no-solution row must set feasible=false")
    if canonical.get("optimal") is True:
        reasons.append("no-solution row must not claim optimal=true")
    if canonical.get("final_solution_available") is not False:
        reasons.append("no-solution row must set final_solution_available=false")
    if canonical.get("final_solution_source") != "none":
        reasons.append("no-solution row must set final_solution_source='none'")
    reason = canonical.get("no_solution_reason")
    if reason not in NO_SOLUTION_REASONS:
        reasons.append(
            "no-solution row missing supported no_solution_reason"
        )
    if canonical.get("makespan") is not None:
        reasons.append("no-solution row must not report a final makespan")
    if canonical.get("result_validation_passed") is False:
        reasons.append("no-solution row must not claim failed final validation")
    if (
        canonical.get("external_incumbent_available") is True
        and canonical.get("fallback_enabled") is True
    ):
        reasons.append(
            "no-solution row has a fallback-enabled external incumbent"
        )

    if reason == "activation_cap_replay_rejected_until_deadline":
        if canonical.get("solver_status_raw") not in {"OPTIMAL", "FEASIBLE"}:
            reasons.append(
                "activation-cap replay rejection requires a CP-SAT feasible status"
            )
        if canonical.get("solver_solution_available") is not True:
            reasons.append(
                "activation-cap replay rejection requires solver_solution_available=true"
            )
        if not (canonical.get("cp_sat_models_solved") or 0):
            reasons.append(
                "activation-cap replay rejection requires cp_sat_models_solved > 0"
            )
        if row.get("enforce_activation_cap") is not True:
            reasons.append(
                "activation-cap replay rejection requires an enforced cap row"
            )
        if canonical.get("activation_cap_enforcement_mode") != "solver":
            reasons.append(
                "activation-cap replay rejection must use solver cap enforcement"
            )
        if canonical.get("activation_cap_enforced_in_solver") is not True:
            reasons.append(
                "activation-cap replay rejection row was not solver-enforced"
            )
        if canonical.get("activation_cap_constraints_added") is not True:
            reasons.append(
                "activation-cap replay rejection row missing cap constraints"
            )

    if row.get("solver_machinery_variant") == "bare-joint-cpsat":
        expected = {
            "incumbent_found": False,
            "fallback_enabled": False,
            "external_incumbent_used_as_fallback": False,
            "incumbent_bound_effective": False,
            "incumbent_hints_effective": False,
        }
        for field, value in expected.items():
            if canonical.get(field) != value:
                reasons.append(
                    f"bare no-solution {field} mismatch: "
                    f"expected {value!r}, got {canonical.get(field)!r}"
                )
        hinted = canonical.get("incumbent_hint_count")
        if hinted not in {0, None}:
            reasons.append("bare no-solution row emitted incumbent hints")
    return reasons


def check_result_against_manifest(
    row: Mapping[str, Any], payload: Mapping[str, Any]
) -> CompatibilityResult:
    reasons: list[str] = []
    runner_outcome = payload.get("runner_outcome")
    canonical = canonical_result(payload)
    if runner_outcome == "unavailable":
        return CompatibilityResult(False, "unavailable", ["solver unavailable"])
    if payload.get("manifest_hash") != row.get("manifest_hash"):
        reasons.append("manifest_hash mismatch")
    if payload.get("manifest_run_id") != row.get("run_id"):
        reasons.append("manifest_run_id mismatch")
    if not canonical:
        reasons.append("canonical_result missing")
        return CompatibilityResult(False, "completed_invalid", reasons)

    reported_status = canonical.get("reported_status") or payload.get("status")
    solver_status = canonical.get("solver_status_raw") or reported_status
    if reported_status in UNAVAILABLE_STATUSES or solver_status in UNAVAILABLE_STATUSES:
        return CompatibilityResult(False, "unavailable", ["reported UNAVAILABLE"])
    is_no_solution = reported_status == NO_VALID_SOLUTION_STATUS
    if is_no_solution:
        reasons.extend(no_solution_compatibility_reasons(row, canonical))
    elif canonical.get("final_solution_available") is False:
        reasons.append(
            "final_solution_available=false requires NO_VALID_SOLUTION status"
        )

    expected_equalities = {
        "schema_version": "expected_schema_version",
        "budget_policy_version": "expected_budget_policy_version",
        "evaluation_method_version": "expected_evaluation_method_version",
        "activation_analysis_version": "expected_activation_analysis_version",
        "activation_cap_formulation_version": (
            "expected_activation_cap_formulation_version"
        ),
        "canonical_method": "canonical_method_expected",
        "method_contract_hash": "method_contract_hash_expected",
        "micro_batches": "B",
        "logical_stages": "N",
        "physical_workers": "W",
        "total_layers": "L",
        "mapping_type": "mapping_type",
        "forward_cost_ratio_numerator": "forward_cost_ratio_num",
        "forward_cost_ratio_denominator": "forward_cost_ratio_den",
        "backward_cost_ratio_numerator": "backward_cost_ratio_num",
        "backward_cost_ratio_denominator": "backward_cost_ratio_den",
        "communication_ticks": "communication_ticks",
        "activation_model": "activation_model",
        "activation_units_per_layer": "activation_units_per_layer",
        "activation_cap_mode": "activation_cap_mode",
        "requested_time_limit_seconds": "time_limit_seconds",
        "random_seed": "random_seed",
        "solver_threads": "solver_threads",
    }
    for result_field, row_field in expected_equalities.items():
        expected = row.get(row_field)
        actual = canonical.get(result_field)
        if expected is None and actual is None:
            continue
        if isinstance(expected, float) or isinstance(actual, float):
            try:
                if abs(float(expected) - float(actual)) <= 1e-9:
                    continue
            except Exception:
                pass
        if actual != expected:
            reasons.append(
                f"{result_field} mismatch: expected {expected!r}, got {actual!r}"
            )

    if row.get("activation_cap_mode") != "none":
        for result_field, row_field in {
            "activation_cap_units_per_worker": "activation_cap_units_per_worker",
            "activation_cap_source": "activation_cap_source",
            "activation_cap_derivation_hash": "activation_cap_derivation_hash",
        }.items():
            expected = row.get(row_field)
            actual = canonical.get(result_field)
            if actual != expected:
                reasons.append(
                    f"{result_field} mismatch: expected {expected!r}, got {actual!r}"
                )

    if (
        row.get("fixed_order_partition_backend") is not None
        and row.get("method") == "partition-only-fixed-order"
    ):
        if (
            canonical.get("fixed_order_partition_backend_requested")
                != row.get("fixed_order_partition_backend")
        ):
            reasons.append("fixed_order_partition_backend_requested mismatch")

    if row.get("solver_machinery_variant"):
        expected_mechanisms = {
            "worker_balance_pruning_requested": row.get(
                "worker_balance_pruning_requested"
            ),
            "incumbent_method_requested": row.get("incumbent_method_requested"),
            "incumbent_bound_requested": row.get("incumbent_bound_requested"),
            "incumbent_hints_requested": row.get("incumbent_hints_requested"),
            "fallback_enabled": row.get("incumbent_fallback_requested"),
        }
        if row.get("incumbent_method_requested") == "none":
            expected_mechanisms.update(
                {
                    "incumbent_method_effective": "none",
                    "incumbent_bound_effective": False,
                    "incumbent_hints_effective": False,
                    "incumbent_found": False,
                    "incumbent_valid": False,
                    "fallback_enabled": False,
                    "external_incumbent_available": False,
                    "external_incumbent_used_as_fallback": False,
                }
            )
        else:
            if canonical.get("incumbent_method_effective") not in {
                row.get("incumbent_method_requested"),
                "none",
            }:
                reasons.append(
                    "incumbent_method_effective mismatch: expected "
                    f"{row.get('incumbent_method_requested')!r} or 'none', "
                    f"got {canonical.get('incumbent_method_effective')!r}"
                )
        if row.get("worker_balance_pruning_requested") is False:
            expected_mechanisms["worker_balance_pruning_effective"] = False
        if row.get("incumbent_bound_requested") is False:
            expected_mechanisms["incumbent_bound_effective"] = False
        if row.get("incumbent_hints_requested") is False:
            expected_mechanisms["incumbent_hints_effective"] = False
        for field, expected in expected_mechanisms.items():
            actual = canonical.get(field)
            if actual != expected:
                reasons.append(
                    f"{field} mismatch: expected {expected!r}, got {actual!r}"
                )
        if row.get("worker_balance_pruning_requested"):
            tolerance = canonical.get("worker_balance_tolerance")
            if not isinstance(tolerance, Mapping):
                reasons.append("worker_balance_tolerance missing")
            else:
                for result_field, row_field in {
                    "percent": "worker_balance_tolerance_percent",
                    "layers": "worker_balance_tolerance_layers",
                }.items():
                    expected = row.get(row_field)
                    if expected is not None and tolerance.get(result_field) != expected:
                        reasons.append(
                            "worker_balance_tolerance "
                            f"{result_field} mismatch: expected {expected!r}, "
                            f"got {tolerance.get(result_field)!r}"
                        )
        if row.get("incumbent_hints_requested") is False:
            hinted = canonical.get("incumbent_hint_count")
            if hinted not in {0, None}:
                reasons.append(
                    "incumbent_hints_requested=false but hints were emitted"
                )

    expected_git = row.get("git_commit_expected")
    actual_git = canonical.get("git_commit")
    if expected_git and actual_git != expected_git:
        if row.get("git_dirty_policy") != "allow_dirty_build":
            reasons.append("git_commit mismatch")

    if row.get("experiment_group") == "oracle":
        if canonical.get("fixed_order_partition_backend_effective") != "enumerate":
            reasons.append("oracle row did not use enumerate backend")
        if canonical.get("optimal") is True:
            if canonical.get("reported_status") != "OPTIMAL":
                reasons.append("oracle row claims optimal without OPTIMAL status")
            if canonical.get("enumeration_proved_optimal") is not True:
                reasons.append("oracle optimal row lacks exhaustive enumeration proof")
            if (
                canonical.get("optimality_proof_source")
                != "exhaustive_enumeration"
            ):
                reasons.append("oracle optimal row lacks exhaustive proof source")
            if canonical.get("makespan") != canonical.get("best_objective_bound"):
                reasons.append("oracle optimal row bound does not match makespan")
            gap = canonical.get("relative_optimality_gap")
            try:
                gap_is_zero = gap is not None and abs(float(gap)) <= 1e-12
            except Exception:
                gap_is_zero = False
            if not gap_is_zero:
                reasons.append("oracle optimal row gap is not zero")
        elif canonical.get("reported_status") == "OPTIMAL":
            reasons.append("oracle OPTIMAL status lacks optimal=true")

    if row.get("enforce_activation_cap") and not is_no_solution:
        if row.get("method") in DETERMINISTIC_METHODS:
            if canonical.get("activation_cap_satisfied") is not True:
                reasons.append("equal-memory row is not cap-satisfied")
            if (
                canonical.get("activation_cap_enforcement_mode")
                != "deterministic_postconstruction_check"
            ):
                reasons.append(
                    "deterministic equal-memory row lacks deterministic cap check"
                )
        elif is_oracle_enumeration_row(row):
            exact_cap = oracle_exact_enumeration_cap_compatible(row, canonical)
            partial_cap = oracle_partial_enumeration_cap_compatible(row, canonical)
            if not exact_cap and not partial_cap:
                reasons.append(
                    "oracle equal-memory enumeration row lacks valid "
                    "enumeration cap enforcement"
                )
            if canonical.get("feasible") is True:
                if canonical.get("activation_cap_satisfied") is not True:
                    reasons.append("equal-memory row is not cap-satisfied")
            elif exact_cap:
                if canonical.get("reported_status") != "INFEASIBLE":
                    reasons.append(
                        "non-feasible exact enumeration oracle row is not "
                        "reported INFEASIBLE"
                    )
                if canonical.get("makespan") is not None:
                    reasons.append(
                        "non-feasible exact enumeration oracle row has a makespan"
                    )
        else:
            if canonical.get("activation_cap_satisfied") is not True:
                reasons.append("equal-memory row is not cap-satisfied")
            if canonical.get("activation_cap_enforcement_mode") != "solver":
                reasons.append(
                    "solver-backed capped row must use solver activation cap "
                    "enforcement"
                )
            if canonical.get("activation_cap_enforced_in_solver") is not True:
                reasons.append("solver-backed capped row not enforced in solver")
            if canonical.get("activation_cap_constraints_added") is not True:
                reasons.append("solver-backed capped row missing cap constraints")

    feasible = canonical.get("feasible")
    validation = canonical.get("result_validation")
    if isinstance(validation, Mapping):
        expected_validation = row.get("expected_validation_version")
        actual_validation = validation.get("validation_version")
        if actual_validation != expected_validation:
            reasons.append(
                "validation_version mismatch: expected "
                f"{expected_validation!r}, got {actual_validation!r}"
            )
    elif feasible is True:
        reasons.append("result_validation missing")
    if feasible is True and canonical.get("result_validation_passed") is not True:
        reasons.append("feasible row failed independent validation")

    if reasons:
        if any("_version" in reason for reason in reasons):
            outcome = "schema_mismatch"
        elif any("manifest_" in reason for reason in reasons):
            outcome = "manifest_mismatch"
        else:
            outcome = "completed_invalid"
        return CompatibilityResult(False, outcome, reasons)
    if is_no_solution:
        return CompatibilityResult(True, "completed_no_solution", [])
    return CompatibilityResult(True, "completed_valid", [])


def stamp_result(
    *,
    row: Mapping[str, Any],
    payload: dict[str, Any],
    command: Sequence[str],
    start_utc: str,
    end_utc: str,
    wall_time_seconds: float,
    exit_code: int,
    output_root: Path,
    outcome: str,
) -> dict[str, Any]:
    stamped = copy.deepcopy(payload)
    stamped.update(
        {
            "manifest_id": row["manifest_id"],
            "manifest_hash": row["manifest_hash"],
            "manifest_run_id": row["run_id"],
            "experiment_group": row["experiment_group"],
            "configuration_id": row["configuration_id"],
            "repetition_id": row["repetition_id"],
            "output_json": row["output_json"],
            "stdout_log": row["stdout_log"],
            "stderr_log": row["stderr_log"],
            "runner_start_utc": start_utc,
            "runner_end_utc": end_utc,
            "runner_wall_time_seconds": wall_time_seconds,
            "runner_exit_code": exit_code,
            "runner_command": list(command),
            "runner_command_string": shlex.join(command),
            "runner_hostname": socket.gethostname(),
            "runner_output_root": str(output_root),
            "runner_outcome": outcome,
        }
    )
    return stamped


def synthetic_unavailable_payload(reason: str) -> dict[str, Any]:
    return {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "reported_status": "UNAVAILABLE",
        "runner_outcome": "unavailable",
        "canonical_result": None,
        "diagnostic": reason,
    }


def existing_result_status(
    row: Mapping[str, Any], output_root: Path
) -> CompatibilityResult | None:
    output_json = resolve_output_path(output_root, str(row["output_json"]))
    if not output_json.exists():
        return None
    try:
        payload = json.loads(output_json.read_text(encoding="utf-8"))
    except Exception as exc:
        return CompatibilityResult(False, "completed_invalid", [str(exc)])
    return check_result_against_manifest(row, payload)


def should_run_row(
    row: Mapping[str, Any],
    output_root: Path,
    *,
    force: bool = False,
    no_rerun_invalid: bool = False,
) -> tuple[bool, str, list[str]]:
    if force:
        return True, "force", []
    existing = existing_result_status(row, output_root)
    if existing is None:
        return True, "missing", []
    if existing.passed:
        if existing.outcome == "completed_no_solution":
            return False, "completed_no_solution", []
        return False, "skipped_valid", []
    if no_rerun_invalid:
        return False, existing.outcome, existing.reasons
    return True, "rerun_invalid", existing.reasons


def run_row(
    row: Mapping[str, Any],
    *,
    binary: Path,
    output_root: Path,
    build_info: Mapping[str, Any],
    force: bool = False,
    no_rerun_invalid: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    command = command_for_row(row, binary, output_root)
    should_run, prior_outcome, prior_reasons = should_run_row(
        row,
        output_root,
        force=force,
        no_rerun_invalid=no_rerun_invalid,
    )
    output_json = resolve_output_path(output_root, str(row["output_json"]))
    stdout_log = resolve_output_path(output_root, str(row["stdout_log"]))
    stderr_log = resolve_output_path(output_root, str(row["stderr_log"]))
    for path in [output_json, stdout_log, stderr_log]:
        path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        return {
            "run_id": row["run_id"],
            "outcome": "dry_run",
            "command": command,
            "executed": False,
        }
    if not should_run:
        return {
            "run_id": row["run_id"],
            "outcome": prior_outcome,
            "reasons": prior_reasons,
            "command": command,
            "executed": False,
        }

    if row.get("requires_ortools") and build_info.get("ortools_enabled") is False:
        start_utc = utc_now()
        end_utc = start_utc
        payload = synthetic_unavailable_payload("OR-Tools is not enabled in binary")
        stamped = stamp_result(
            row=row,
            payload=payload,
            command=command,
            start_utc=start_utc,
            end_utc=end_utc,
            wall_time_seconds=0.0,
            exit_code=0,
            output_root=output_root,
            outcome="unavailable",
        )
        output_json.write_text(json.dumps(stamped, indent=2, sort_keys=True) + "\n")
        stdout_log.write_text("", encoding="utf-8")
        stderr_log.write_text("OR-Tools is not enabled in binary\n", encoding="utf-8")
        return {
            "run_id": row["run_id"],
            "outcome": "unavailable",
            "command": command,
            "executed": False,
        }

    start_utc = utc_now()
    started = time.time()
    with stdout_log.open("w", encoding="utf-8") as stdout_handle, stderr_log.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        process = subprocess.run(
            command,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            check=False,
        )
    wall_time = time.time() - started
    end_utc = utc_now()
    if output_json.exists():
        try:
            payload = json.loads(output_json.read_text(encoding="utf-8"))
        except Exception as exc:
            payload = {"canonical_result": None, "result_read_error": str(exc)}
    else:
        payload = {"canonical_result": None, "diagnostic": "output JSON missing"}

    preliminary = check_result_against_manifest(row, payload)
    if process.returncode != 0 and preliminary.outcome != "unavailable":
        outcome = "timeout_or_failed_process"
    elif preliminary.outcome == "unavailable":
        outcome = "unavailable"
    elif preliminary.outcome == "completed_no_solution":
        outcome = "completed_no_solution"
    else:
        outcome = "completed_valid"
    stamped = stamp_result(
        row=row,
        payload=dict(payload),
        command=command,
        start_utc=start_utc,
        end_utc=end_utc,
        wall_time_seconds=wall_time,
        exit_code=process.returncode,
        output_root=output_root,
        outcome=outcome,
    )
    output_json.write_text(json.dumps(stamped, indent=2, sort_keys=True) + "\n")
    final = check_result_against_manifest(row, stamped)
    if outcome in {"completed_valid", "completed_no_solution"} and final.passed:
        outcome = final.outcome
        stamped["runner_outcome"] = outcome
        output_json.write_text(json.dumps(stamped, indent=2, sort_keys=True) + "\n")
    elif outcome in {"completed_valid", "completed_no_solution"} and not final.passed:
        outcome = final.outcome
        stamped["runner_outcome"] = outcome
        output_json.write_text(json.dumps(stamped, indent=2, sort_keys=True) + "\n")
    return {
        "run_id": row["run_id"],
        "outcome": outcome,
        "reasons": final.reasons,
        "command": command,
        "executed": True,
        "exit_code": process.returncode,
    }


def filter_rows(
    rows: Iterable[dict[str, Any]],
    *,
    group: str | None = None,
    method: str | None = None,
    configuration_id: str | None = None,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if group and row["experiment_group"] != group:
            continue
        if method and row["method"] != method:
            continue
        if configuration_id and row["configuration_id"] != configuration_id:
            continue
        if run_id and row["run_id"] != run_id:
            continue
        selected.append(row)
    return selected


def run_manifest(
    *,
    rows: Sequence[dict[str, Any]],
    binary: Path,
    output_root: Path,
    jobs: int = 1,
    dry_run: bool = False,
    force: bool = False,
    force_new_manifest: bool = False,
    no_rerun_invalid: bool = False,
    fail_fast: bool = False,
) -> list[dict[str, Any]]:
    build_info = load_build_info(binary)
    output_root.mkdir(parents=True, exist_ok=True)
    resource_summary = runner_resource_summary(rows, jobs)
    if resource_summary["oversubscribed"]:
        print(
            "warning: CAL runner requested "
            f"jobs * solver_threads = "
            f"{resource_summary['estimated_concurrent_solver_threads']} "
            f"({resource_summary['effective_jobs']} jobs * "
            f"{resource_summary['max_solver_threads_per_row']} solver threads), "
            f"which exceeds available cores "
            f"{resource_summary['available_cores']}; reduce --jobs for fair timing.",
            file=sys.stderr,
        )
    guard_output_root_manifest(
        rows=rows,
        output_root=output_root,
        force_new_manifest=force_new_manifest,
        dry_run=dry_run,
        runner_resource_summary=resource_summary,
    )
    worker = lambda row: run_row(
        row,
        binary=binary,
        output_root=output_root,
        build_info=build_info,
        force=force,
        no_rerun_invalid=no_rerun_invalid,
        dry_run=dry_run,
    )
    results: list[dict[str, Any]] = []
    if jobs <= 1:
        for row in rows:
            result = worker(row)
            results.append(result)
            if fail_fast and result["outcome"] not in {
                "completed_valid",
                "completed_no_solution",
                "skipped_valid",
                "unavailable",
                "dry_run",
            }:
                break
        return results
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        future_to_row = {executor.submit(worker, row): row for row in rows}
        for future in concurrent.futures.as_completed(future_to_row):
            result = future.result()
            results.append(result)
            if fail_fast and result["outcome"] not in {
                "completed_valid",
                "completed_no_solution",
                "skipped_valid",
                "unavailable",
                "dry_run",
            }:
                for pending in future_to_row:
                    pending.cancel()
                break
    results.sort(key=lambda item: str(item["run_id"]))
    return results


def guard_output_root_manifest(
    *,
    rows: Sequence[Mapping[str, Any]],
    output_root: Path,
    force_new_manifest: bool = False,
    dry_run: bool = False,
    runner_resource_summary: Mapping[str, Any] | None = None,
) -> None:
    manifest_hashes = {
        str(row.get("manifest_hash")) for row in rows if row.get("manifest_hash")
    }
    if len(manifest_hashes) > 1:
        raise ValueError(
            "selected rows contain multiple manifest hashes: "
            + ", ".join(sorted(manifest_hashes))
        )
    if not manifest_hashes:
        return
    manifest_hash = next(iter(manifest_hashes))
    existing_hashes = existing_output_root_manifest_hashes(output_root)
    if existing_hashes and existing_hashes != {manifest_hash} and not force_new_manifest:
        raise RuntimeError(
            "output root already contains CAL results for manifest hash(es) "
            f"{sorted(existing_hashes)}; refusing to append manifest hash "
            f"{manifest_hash!r}. Use --force-new-manifest to intentionally reuse "
            "this output root."
        )
    if dry_run:
        return
    marker = output_root / RUNNER_MANIFEST_MARKER
    marker.write_text(
        json.dumps(
            {
                "manifest_hash": manifest_hash,
                "manifest_ids": sorted(
                    {str(row.get("manifest_id")) for row in rows if row.get("manifest_id")}
                ),
                "row_count": len(rows),
                "runner_effective_jobs": (
                    runner_resource_summary or {}
                ).get("effective_jobs"),
                "runner_max_solver_threads_per_row": (
                    runner_resource_summary or {}
                ).get("max_solver_threads_per_row"),
                "runner_estimated_concurrent_solver_threads": (
                    runner_resource_summary or {}
                ).get("estimated_concurrent_solver_threads"),
                "runner_available_cores": (
                    runner_resource_summary or {}
                ).get("available_cores"),
                "runner_oversubscribed": (
                    runner_resource_summary or {}
                ).get("oversubscribed"),
                "updated_at_utc": utc_now(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def runner_resource_summary(
    rows: Sequence[Mapping[str, Any]], jobs: int
) -> dict[str, Any]:
    effective_jobs = max(1, int(jobs))
    max_solver_threads = 1
    for row in rows:
        try:
            max_solver_threads = max(max_solver_threads, int(row.get("solver_threads") or 1))
        except Exception:
            max_solver_threads = max(max_solver_threads, 1)
    available_cores = os.cpu_count()
    estimated = effective_jobs * max_solver_threads
    oversubscribed = (
        available_cores is not None
        and available_cores > 0
        and estimated > available_cores
    )
    return {
        "effective_jobs": effective_jobs,
        "max_solver_threads_per_row": max_solver_threads,
        "estimated_concurrent_solver_threads": estimated,
        "available_cores": available_cores,
        "oversubscribed": oversubscribed,
    }


def existing_output_root_manifest_hashes(output_root: Path) -> set[str]:
    hashes: set[str] = set()
    marker = output_root / RUNNER_MANIFEST_MARKER
    if marker.exists():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            value = payload.get("manifest_hash")
            if value:
                hashes.add(str(value))
        except Exception:
            hashes.add("<unreadable-marker>")
    if output_root.exists():
        for path in output_root.rglob("*.json"):
            if path == marker:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            value = payload.get("manifest_hash") if isinstance(payload, Mapping) else None
            if value:
                hashes.add(str(value))
    return hashes


def summarize_outcomes(results: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        outcome = str(result["outcome"])
        counts[outcome] = counts.get(outcome, 0) + 1
    return counts


def parse_seed_list(text: str) -> list[int]:
    return [int(token) for token in text.split(",") if token != ""]
