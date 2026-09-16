from __future__ import annotations

import csv
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import pytest

from scripts import run_primary_ablation as primary


def test_primary_manifest_counts_and_modes() -> None:
    configs = primary.generate_structural_configs()
    runs = primary.primary_run_specs(configs)

    assert len(configs) == 520
    assert Counter(config.depth_set for config in configs) == {
        "core": 360,
        "real-depth-extension": 160,
    }
    assert len(runs) == 1560
    assert Counter(run.optimization_mode for run in runs) == {
        "joint": 520,
        "partition-only": 520,
        "schedule-only-uniform": 520,
    }
    assert all(config.N % config.W == 0 for config in configs)
    assert all(config.N <= config.L for config in configs)
    assert any(config.L % config.N != 0 for config in configs)
    assert all(run.optimization_mode != "schedule-only-best" for run in runs)


def test_uniform_partition_allows_nondivisible_layer_depths() -> None:
    split = primary.uniform_split(total_layers=61, stages=16)
    assert sum(split) == 61
    assert min(split) > 0
    assert max(split) - min(split) <= 1


def test_cli_command_maps_pipeline_workers_and_solver_threads_separately(tmp_path: Path) -> None:
    config = next(
        item
        for item in primary.generate_structural_configs()
        if item.W == 32 and item.N == 32 and item.L == 32
    )
    run = next(
        item
        for item in primary.primary_run_specs(primary.generate_structural_configs())
        if item.structural == config and item.optimization_mode == "schedule-only-uniform"
    )
    command = primary.command_for_run(
        Path("build/release/slackpipe_cli"), tmp_path / "case", run
    )

    assert command[command.index("--J") + 1] == "32"
    assert command[command.index("--num-workers") + 1] == "16"
    assert command[command.index("--fixed-partition-source") + 1] == "uniform"
    assert "--algorithm" in command
    assert command[command.index("--algorithm") + 1] == "schedule-only-uniform"


def test_dry_run_writes_complete_primary_manifest(tmp_path: Path) -> None:
    primary.main(["--dry-run", "--output-dir", str(tmp_path)])

    manifest_rows = list(csv.DictReader((tmp_path / "manifest.csv").open()))
    config_rows = list(csv.DictReader((tmp_path / "configurations.csv").open()))
    summary = json.loads((tmp_path / "manifest_summary.json").read_text())

    assert len(manifest_rows) == 1560
    assert len(config_rows) == 520
    assert summary["primary_expected_configurations"] == 520
    assert summary["primary_expected_runs"] == 1560
    assert Counter(row["optimization_mode"] for row in manifest_rows) == {
        "joint": 520,
        "partition-only": 520,
        "schedule-only-uniform": 520,
    }


def run_joint_cli(
    tmp_path: Path,
    *,
    name: str,
    B: int,
    N: int,
    J: int,
    L: int,
    time_limit_seconds: float,
    num_workers: int,
) -> tuple[subprocess.CompletedProcess[str], dict]:
    cli = Path("build/release/slackpipe_cli")
    if not cli.exists():
        pytest.skip("release CLI is not built")
    output_prefix = tmp_path / name
    command = [
        str(cli),
        "--B",
        str(B),
        "--N",
        str(N),
        "--J",
        str(J),
        "--L",
        str(L),
        "--min-layers",
        "1",
        "--ratio-num",
        "2",
        "--ratio-den",
        "1",
        "--communication",
        "0",
        "--algorithm",
        "joint",
        "--time-limit-seconds",
        str(time_limit_seconds),
        "--num-workers",
        str(num_workers),
        "--random-seed",
        "1",
        "--require-optimal",
        "false",
        "--output-prefix",
        str(output_prefix),
    ]
    started = time.monotonic()
    completed = subprocess.run(command, text=True, capture_output=True, timeout=20)
    elapsed = time.monotonic() - started

    assert completed.returncode == 0, completed.stderr
    assert elapsed < 20
    payload = json.loads(output_prefix.with_suffix(".json").read_text())
    return completed, payload


def test_reduced_problematic_regime_short_limit_returns_structured_record(tmp_path: Path) -> None:
    completed, payload = run_joint_cli(
        tmp_path,
        name="reduced-joint",
        B=16,
        N=16,
        J=8,
        L=48,
        time_limit_seconds=0.25,
        num_workers=4,
    )
    assert payload["status"] in {"OPTIMAL", "FEASIBLE"}
    assert payload["simulated_iteration_time"] > 0
    assert "SLACKPIPE_LIFECYCLE" in completed.stderr
    assert "phase=SOLVE_START" in completed.stderr
    assert "phase=CLI_EXIT" in completed.stderr


def test_joint_budget_policy_reserves_time_and_records_fallback(tmp_path: Path) -> None:
    total_budget = 2.0
    completed, payload = run_joint_cli(
        tmp_path,
        name="budget-transfer",
        B=4,
        N=4,
        J=2,
        L=8,
        time_limit_seconds=total_budget,
        num_workers=2,
    )
    if "budget_policy_version" not in payload:
        pytest.skip("release CLI predates budget policy schema")
    assert payload["budget_policy_version"] == primary.BUDGET_POLICY_VERSION
    assert payload["incumbent_budget_seconds"] <= 0.10 * total_budget + 0.01
    assert payload["incumbent_solve_seconds"] < total_budget
    assert payload["joint_budget_seconds"] >= 0.90 * total_budget - 0.05
    assert payload["incumbent_solve_seconds"] + payload["joint_budget_seconds"] <= total_budget + 0.05
    assert payload["solution_source"] in {"joint_cpsat", "joint_cpsat_improved_bfs"}
    assert payload["fallback_used"] is False
    assert "requested_solver_limit_seconds=" in completed.stderr
    assert "effective_solver_limit_seconds=" in completed.stderr

    tiny_budget = 0.05
    completed, payload = run_joint_cli(
        tmp_path,
        name="budget-fallback",
        B=16,
        N=16,
        J=8,
        L=48,
        time_limit_seconds=tiny_budget,
        num_workers=4,
    )
    assert payload["incumbent_budget_seconds"] <= 0.10 * tiny_budget + 0.005
    assert payload["incumbent_solve_seconds"] < tiny_budget
    assert payload["incumbent_solve_seconds"] + payload["joint_budget_seconds"] <= tiny_budget + 0.05
    assert payload["joint_status"] in {"UNKNOWN", "NOT_RUN"}
    if payload["joint_status"] == "NOT_RUN":
        assert payload["cp_sat_models_solved"] == 0
    assert payload["solution_source"] == "bfs_incumbent_fallback"
    assert payload["fallback_used"] is True
    assert "solution_source=bfs_incumbent_fallback" in completed.stdout
