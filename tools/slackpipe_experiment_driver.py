#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION. All rights reserved.

"""Run controlled SlackPipe measured-cost ablation experiments.

This is a host-side orchestrator.  Megatron work is always launched inside the
existing container, while OR-Tools solver work uses a separately supplied image.
SLACKPIPE_REPO, SLACKPIPE_CONTAINER_REPO, SLACKPIPE_CONTAINER, and
SLACKPIPE_ORTOOLS_IMAGE override the local development defaults.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO = Path(os.environ.get("SLACKPIPE_REPO", Path(__file__).resolve().parents[1])).resolve()
CONTAINER_REPO = Path(os.environ.get("SLACKPIPE_CONTAINER_REPO", "/workspace/Megatron-LM"))
SLACKPIPE_DIR = REPO / "slackpipe"
CONTAINER_SLACKPIPE_DIR = CONTAINER_REPO / "slackpipe"
MEGATRON_CONTAINER = os.environ.get("SLACKPIPE_CONTAINER", "slackpipe-dev")
ORTOOLS_IMAGE = os.environ.get("SLACKPIPE_ORTOOLS_IMAGE", "slackpipe-ortools-runtime:local")
STAGE_TO_WORKER = [0, 1, 0, 1]


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    seq_length: int
    num_microbatches: int
    micro_batch_size: int = 1
    vocab_size: int = 1024
    dropout: float = 0.0
    pp: int = 2
    vpp: int = 2
    tp: int = 1
    dp: int = 1
    cp: int = 1
    dtype: str = "fp32"


DEFAULT_CONFIGS = [
    ExperimentConfig("S_b4_seq64", 16, 64, 4, 64, 4),
    ExperimentConfig("S_b8_seq64", 16, 64, 4, 64, 8),
    ExperimentConfig("S_b16_seq64", 16, 64, 4, 64, 16),
    ExperimentConfig("S_b8_seq128", 16, 64, 4, 128, 8),
    ExperimentConfig("S_b8_seq256", 16, 64, 4, 256, 8),
    ExperimentConfig("M_b8_seq128", 16, 128, 4, 128, 8),
    ExperimentConfig("L_b8_seq128", 24, 128, 4, 128, 8),
]

SMOKE_CONFIGS = [ExperimentConfig("S_b8_seq64", 16, 64, 4, 64, 8)]


def json_dumps(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json_dumps(data), encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def stable_digest(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def host_path(container_path: str | Path) -> Path:
    path = Path(container_path)
    if path.is_absolute() and str(path).startswith(str(CONTAINER_REPO)):
        return REPO / path.relative_to(CONTAINER_REPO)
    return path


def container_path(path: Path) -> str:
    return str(CONTAINER_REPO / path.resolve().relative_to(REPO))


def run(cmd: list[str], *, cwd: Path = REPO, log_path: Path | None = None) -> str:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"[run] {printable}", flush=True)
    started = time.time()
    proc = subprocess.run(
        cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    )
    output = proc.stdout
    if log_path is not None:
        write_text(log_path, output)
    print(f"[run] exit={proc.returncode} elapsed={time.time() - started:.1f}s", flush=True)
    if proc.returncode != 0:
        tail = "\n".join(output.splitlines()[-80:])
        raise RuntimeError(f"command failed ({proc.returncode}): {printable}\n{tail}")
    return output


def docker_exec(script: str, *, log_path: Path | None = None) -> str:
    return run(["docker", "exec", MEGATRON_CONTAINER, "bash", "-lc", script], log_path=log_path)


def chown_from_container(path: Path) -> None:
    if not path.exists():
        return
    run(
        [
            "docker",
            "exec",
            "--user",
            "root",
            MEGATRON_CONTAINER,
            "bash",
            "-lc",
            f"chown -R {os.getuid()}:{os.getgid()} {shlex.quote(container_path(path))}",
        ]
    )


def ortools_run(script: str, *, log_path: Path | None = None) -> str:
    uid = f"{os.getuid()}:{os.getgid()}"
    return run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "/bin/bash",
            "--user",
            uid,
            "-v",
            f"{REPO}:{CONTAINER_REPO}",
            "-w",
            str(CONTAINER_SLACKPIPE_DIR),
            ORTOOLS_IMAGE,
            "-lc",
            script,
        ],
        log_path=log_path,
    )


def stage_done(meta_path: Path, metadata: dict[str, Any], outputs: list[Path], force: bool) -> bool:
    if force or not meta_path.exists():
        return False
    if not all(path.exists() for path in outputs):
        return False
    try:
        return read_json(meta_path) == metadata
    except Exception:
        return False


def mark_done(meta_path: Path, metadata: dict[str, Any]) -> None:
    write_json(meta_path, metadata)


def split_uniform(num_layers: int) -> list[int]:
    split = [num_layers // 4] * 4
    for idx in range(num_layers % 4):
        split[idx] += 1
    return split


def layout_from_split(split: list[int]) -> str:
    specs = []
    for stage, count in enumerate(split):
        spec = ""
        if stage == 0:
            spec += "E"
        spec += "t" if count == 1 else f"t*{count}"
        if stage == len(split) - 1:
            spec += "L"
        specs.append(spec)
    return "|".join(specs)


def make_plan_json(
    path: Path,
    cfg: ExperimentConfig,
    split: list[int],
    operations: list[list[dict[str, int | str]]],
    *,
    status: str = "SYNTHETIC",
    makespan: float = 0.0,
    forward_costs: list[float] | None = None,
    backward_costs: list[float] | None = None,
) -> None:
    payload = {
        "schema_version": "slackpipe.plan.v1",
        "num_microbatches": cfg.num_microbatches,
        "num_stages": 4,
        "num_workers": 2,
        "num_layers": cfg.num_layers,
        "layer_split": split,
        "stage_to_worker": STAGE_TO_WORKER,
        "operations": operations,
        "solver_status": status,
        "predicted_makespan": makespan,
        "forward_costs": forward_costs or split,
        "backward_costs": backward_costs or split,
    }
    write_json(path, payload)


def breadth_first_operations(
    cfg: ExperimentConfig, split: list[int]
) -> list[list[dict[str, int | str]]]:
    workers: list[list[dict[str, int | str]]] = [[], []]
    for b in range(cfg.num_microbatches):
        for s in range(4):
            workers[STAGE_TO_WORKER[s]].append({"kind": "F", "microbatch": b, "stage": s})
    for b in range(cfg.num_microbatches):
        for s in reversed(range(4)):
            workers[STAGE_TO_WORKER[s]].append({"kind": "B", "microbatch": b, "stage": s})
    return workers


def write_calibration_seed_plan(path: Path, cfg: ExperimentConfig, force: bool) -> None:
    split = split_uniform(cfg.num_layers)
    meta = {"kind": "calibration_seed_plan", "config": asdict(cfg), "split": split}
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if stage_done(meta_path, meta, [path], force):
        return
    make_plan_json(
        path, cfg, split, breadth_first_operations(cfg, split), status="CALIBRATION_SEED"
    )
    mark_done(meta_path, meta)


def build_solver(force: bool, root: Path) -> None:
    meta = {"kind": "ortools_build", "repo": str(SLACKPIPE_DIR)}
    meta_path = root / "solver" / "build.meta.json"
    binary = SLACKPIPE_DIR / "build" / "solver-ortools-e2e" / "slackpipe_cli"
    if stage_done(meta_path, meta, [binary], force):
        return
    script = (
        "set -euo pipefail; "
        "export LD_LIBRARY_PATH=/opt/or-tools/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; "
        "cmake --build build/solver-ortools-e2e -j2"
    )
    ortools_run(script, log_path=root / "logs" / "build_solver.log")
    mark_done(meta_path, meta)


def solver_command(
    cfg: ExperimentConfig,
    algorithm: str,
    output_prefix: Path,
    plan_path: Path,
    *,
    cost_profile: Path | None,
    time_limit: float,
    require_optimal: bool,
) -> str:
    parts = [
        "build/solver-ortools-e2e/slackpipe_cli",
        "--algorithm",
        algorithm,
        "--B",
        str(cfg.num_microbatches),
        "--N",
        "4",
        "--J",
        "2",
        "--L",
        str(cfg.num_layers),
        "--time-limit-seconds",
        str(time_limit),
        "--require-optimal",
        "true" if require_optimal else "false",
        "--output-prefix",
        container_path(output_prefix),
        "--emit-plan",
        container_path(plan_path),
    ]
    if cost_profile is None:
        parts += ["--ratio-num", "1", "--ratio-den", "1"]
    else:
        parts += ["--cost-profile", container_path(cost_profile)]
    return " ".join(shlex.quote(part) for part in parts)


def run_solver(
    cfg: ExperimentConfig,
    root: Path,
    name: str,
    algorithm: str,
    *,
    cost_profile: Path | None,
    force: bool,
    time_limit: float,
    require_optimal: bool = False,
) -> Path:
    prefix = root / "solver" / name / name
    plan = root / "plans" / f"{name}.plan.json"
    meta = {
        "kind": "solver",
        "name": name,
        "algorithm": algorithm,
        "config": asdict(cfg),
        "cost_profile_digest": file_digest(cost_profile) if cost_profile else None,
        "time_limit": time_limit,
        "require_optimal": require_optimal,
    }
    meta_path = plan.with_suffix(plan.suffix + ".meta.json")
    if stage_done(meta_path, meta, [plan, prefix.with_suffix(".json")], force):
        return plan
    script = (
        "set -euo pipefail; "
        f"mkdir -p {shlex.quote(container_path(prefix.parent))} {shlex.quote(container_path(plan.parent))}; "
        "export LD_LIBRARY_PATH=/opt/or-tools/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; "
        f"{solver_command(cfg, algorithm, prefix, plan, cost_profile=cost_profile, time_limit=time_limit, require_optimal=require_optimal)}"
    )
    ortools_run(script, log_path=root / "logs" / f"solver_{name}.log")
    mark_done(meta_path, meta)
    return plan


def file_digest(path: Path | None) -> str | None:
    if path is None:
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def validate_plan(plan: Path, root: Path, name: str) -> None:
    script = (
        f"cd {CONTAINER_REPO} && python - <<'PY'\n"
        "from megatron.core.pipeline_parallel.slackpipe.plan import load_slackpipe_plan\n"
        f"plan = load_slackpipe_plan({str(container_path(plan))!r}, pipeline_model_parallel_size=2)\n"
        "print(plan.schema_version, plan.num_microbatches, plan.num_stages, list(plan.layer_split), plan.predicted_makespan)\n"
        "print(sum(len(ops) for ops in plan.operations))\n"
        "PY"
    )
    docker_exec(script, log_path=root / "logs" / f"validate_plan_{name}.log")


def calibrate(
    cfg: ExperimentConfig, root: Path, seed_plan: Path, force: bool, warmups: int, iterations: int
) -> Path:
    profile = root / "cost_profiles" / "M3_full.cost_profile.json"
    events = root / "calibration" / "calibration_events.json"
    meta = {
        "kind": "calibration",
        "config": asdict(cfg),
        "seed_plan_digest": file_digest(seed_plan),
        "warmups": warmups,
        "iterations": iterations,
    }
    meta_path = profile.with_suffix(profile.suffix + ".meta.json")
    if stage_done(meta_path, meta, [profile, events], force):
        return profile
    script = (
        f"cd {CONTAINER_REPO} && "
        f"rm -rf {container_path(root / 'calibration')} && "
        f"mkdir -p {container_path(root / 'calibration')} {container_path(root / 'cost_profiles')} && "
        "CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run "
        f"--master_port {next_port(cfg.name, 'calibration')} --nproc_per_node=2 "
        "tests/unit_tests/pipeline_parallel/slackpipe_perf_benchmark.py "
        f"--plan {container_path(seed_plan)} "
        f"--output-dir {container_path(root / 'calibration')} "
        "--warmup-iterations 5 --iterations 20 "
        f"--hidden-size {cfg.hidden_size} --num-attention-heads {cfg.num_attention_heads} "
        f"--seq-length {cfg.seq_length} --micro-batch-size {cfg.micro_batch_size} "
        f"--vocab-size {cfg.vocab_size} --learning-rate 0 "
        f"--emit-cost-profile {container_path(profile)} "
        f"--calibration-warmup-iterations {warmups} --calibration-iterations {iterations} "
        "--shared-slope-estimator min"
    )
    docker_exec(script, log_path=root / "logs" / "calibration.log")
    chown_from_container(root / "calibration")
    chown_from_container(root / "cost_profiles")
    mark_done(meta_path, meta)
    return profile


def derive_cost_profiles(full_profile: Path, root: Path, force: bool) -> dict[str, Path | None]:
    full = read_json(full_profile)
    ratio = float(full["measured_backward_forward_ratio"])
    base = {
        "schema_version": "slackpipe.cost_profile.v1",
        "model_config": full["model_config"],
        "parallel_config": full["parallel_config"],
        "microbatch_config": full["microbatch_config"],
        "calibration_partition": full["calibration_partition"],
        "estimator": "derived",
        "percentile": None,
        "observed_stages": full["observed_stages"],
        "units": "milliseconds",
    }
    profiles: dict[str, Path | None] = {"M0_equal": None}
    variants = {
        "M1_global_ratio": {
            **base,
            "a_fwd": 1.0,
            "a_bwd": ratio,
            "measured_backward_forward_ratio": ratio,
            "bias_fwd": [0.0] * 4,
            "bias_bwd": [0.0] * 4,
            "derived_from": str(full_profile),
            "derivation": "global_bf_ratio_only",
        },
        "M2_shared_slopes": {
            **base,
            "a_fwd": full["a_fwd"],
            "a_bwd": full["a_bwd"],
            "measured_backward_forward_ratio": ratio,
            "bias_fwd": [0.0] * 4,
            "bias_bwd": [0.0] * 4,
            "derived_from": str(full_profile),
            "derivation": "shared_slopes_no_bias",
        },
    }
    for name, payload in variants.items():
        path = root / "cost_profiles" / f"{name}.cost_profile.json"
        meta = {
            "kind": "derived_cost_profile",
            "name": name,
            "source_digest": file_digest(full_profile),
        }
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if not stage_done(meta_path, meta, [path], force):
            write_json(path, payload)
            mark_done(meta_path, meta)
        profiles[name] = path
    profiles["M3_full"] = full_profile
    return profiles


def next_port(config_name: str, label: str, rep: int = 0) -> int:
    digest = int(hashlib.sha1(f"{config_name}:{label}:{rep}".encode()).hexdigest()[:6], 16)
    return 20000 + digest % 30000


def benchmark_once(
    cfg: ExperimentConfig,
    root: Path,
    plan: Path,
    label: str,
    only_mode: str,
    rep: int,
    warmups: int,
    iterations: int,
    timeout_seconds: int,
    force: bool,
) -> Path:
    out_dir = root / "benchmark" / label / f"rep_{rep:02d}"
    summary = out_dir / "benchmark_summary.json"
    meta = {
        "kind": "benchmark",
        "config": asdict(cfg),
        "plan_digest": file_digest(plan),
        "label": label,
        "only_mode": only_mode,
        "rep": rep,
        "warmups": warmups,
        "iterations": iterations,
        "timeout_seconds": timeout_seconds,
    }
    meta_path = summary.with_suffix(summary.suffix + ".meta.json")
    if stage_done(meta_path, meta, [summary], force):
        return summary
    script = (
        f"cd {CONTAINER_REPO} && rm -rf {container_path(out_dir)} && "
        f"timeout --kill-after=30s {timeout_seconds}s "
        "env CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run "
        f"--master_port {next_port(cfg.name, label, rep)} --nproc_per_node=2 "
        "tests/unit_tests/pipeline_parallel/slackpipe_perf_benchmark.py "
        f"--plan {container_path(plan)} --output-dir {container_path(out_dir)} "
        f"--warmup-iterations {warmups} --iterations {iterations} "
        f"--hidden-size {cfg.hidden_size} --num-attention-heads {cfg.num_attention_heads} "
        f"--seq-length {cfg.seq_length} --micro-batch-size {cfg.micro_batch_size} "
        f"--vocab-size {cfg.vocab_size} --learning-rate 0 "
        "--disable-slackpipe-nvtx --slackpipe-runtime fast "
        f"--only-mode {only_mode}"
    )
    try:
        docker_exec(script, log_path=root / "logs" / f"benchmark_{label}_rep_{rep:02d}.log")
    except RuntimeError:
        cleanup_script = (
            "pkill -TERM -f '[s]lackpipe_perf_benchmark.py' "
            "|| true; sleep 1; "
            "pkill -KILL -f '[s]lackpipe_perf_benchmark.py' || true"
        )
        try:
            docker_exec(cleanup_script)
        except RuntimeError:
            pass
        raise
    chown_from_container(out_dir)
    check_trace_if_present(out_dir)
    mark_done(meta_path, meta)
    return summary


def check_trace_if_present(out_dir: Path) -> None:
    for rank in (0, 1):
        path = out_dir / f"slackpipe_trace.rank{rank}.json"
        if path.exists() and not read_json(path).get("matched_plan", False):
            raise RuntimeError(f"trace did not match plan: {path}")


def extract_global_times(summary_path: Path, mode_name: str) -> tuple[list[float], dict[str, Any]]:
    summary = read_json(summary_path)
    stats = summary["modes"][mode_name]
    rank_times = [rank["iteration_cuda_times_ms"] for rank in stats["rank_results"]]
    return [max(values) for values in zip(*rank_times)], summary


def summarize_samples(samples_by_rep: list[list[float]]) -> dict[str, float | int | str]:
    samples = [value for rep in samples_by_rep for value in rep]
    rep_means = [statistics.fmean(rep) for rep in samples_by_rep if rep]
    mean = statistics.fmean(samples)
    stddev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    rep_stddev = statistics.stdev(rep_means) if len(rep_means) > 1 else 0.0
    ci95 = 1.96 * rep_stddev / math.sqrt(len(rep_means)) if len(rep_means) > 1 else 0.0
    return {
        "mean_ms": mean,
        "median_ms": statistics.median(samples),
        "stddev_ms": stddev,
        "rep_mean_stddev_ms": rep_stddev,
        "ci95_ms": ci95,
        "cv": stddev / mean if mean else 0.0,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "num_iterations": len(samples),
        "num_repetitions": len(samples_by_rep),
    }


def speedup_ci(
    numerator_reps: list[list[float]], denominator_reps: list[list[float]]
) -> tuple[float, float, str]:
    ratios = []
    for num, den in zip(numerator_reps, denominator_reps):
        if num and den:
            ratios.append((statistics.fmean(den) - statistics.fmean(num)) / statistics.fmean(den))
    mean = statistics.fmean(ratios) if ratios else 0.0
    ci = 0.0
    if len(ratios) > 1:
        ci = 1.96 * statistics.stdev(ratios) / math.sqrt(len(ratios))
    label = "inconclusive" if mean - ci <= 0.0 <= mean + ci else ("win" if mean > 0 else "loss")
    return mean, ci, label


def load_plan_info(path: Path) -> dict[str, Any]:
    data = read_json(path)
    return {
        "split": data["layer_split"],
        "predicted_makespan": data["predicted_makespan"],
        "solver_status": data["solver_status"],
        "forward_costs": data["forward_costs"],
        "backward_costs": data["backward_costs"],
    }


def run_regressions(root: Path, suffix: str) -> dict[str, str]:
    commands = {
        "python_plan_cost": (
            f"cd {CONTAINER_REPO} && python -m pytest -q "
            "tests/unit_tests/pipeline_parallel/test_slackpipe_plan.py "
            "tests/unit_tests/pipeline_parallel/test_slackpipe_cost_profile.py --capture=fd"
        ),
        "pp1": (
            f"cd {CONTAINER_REPO} && CUDA_VISIBLE_DEVICES=0 python -m pytest -q "
            "tests/unit_tests/pipeline_parallel/test_slackpipe_model_construction.py::test_slackpipe_args_derive_layout_from_plan "
            "tests/unit_tests/pipeline_parallel/test_slackpipe_model_construction.py::test_slackpipe_pp1_constructs_logical_vpp_chunks "
            "tests/unit_tests/pipeline_parallel/test_slackpipe_model_construction.py::test_slackpipe_pp1_numerical_equivalence --capture=fd"
        ),
        "communicator": (
            f"cd {CONTAINER_REPO} && CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run "
            f"--master_port {next_port('regression', 'communicator_' + suffix)} --nproc_per_node=2 "
            "-m pytest -q tests/unit_tests/pipeline_parallel/test_slackpipe_communication.py --capture=fd"
        ),
        "pp2": (
            f"cd {CONTAINER_REPO} && CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run "
            f"--master_port {next_port('regression', 'pp2_' + suffix)} --nproc_per_node=2 "
            "-m pytest -q tests/unit_tests/pipeline_parallel/test_slackpipe_model_construction.py::test_slackpipe_pp2_numerical_equivalence --capture=fd"
        ),
    }
    results = {}
    for name, script in commands.items():
        docker_exec(script, log_path=root / "logs" / f"regression_{suffix}_{name}.log")
        results[name] = "passed"
    ortools_run(
        "set -euo pipefail; export LD_LIBRARY_PATH=/opt/or-tools/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; "
        "ctest --test-dir build/solver-ortools-e2e --output-on-failure",
        log_path=root / "logs" / f"regression_{suffix}_ctest_or.log",
    )
    results["ctest_or"] = "passed"
    run(
        ["ctest", "--test-dir", "build/no-or", "--output-on-failure"],
        cwd=SLACKPIPE_DIR,
        log_path=root / "logs" / f"regression_{suffix}_ctest_no_or.log",
    )
    results["ctest_no_or"] = "passed"
    return results


def run_config(cfg: ExperimentConfig, args: argparse.Namespace) -> dict[str, Any]:
    config_id = cfg.name
    root = args.output_dir / config_id
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "config.json", asdict(cfg))

    seed_plan = root / "plans" / "calibration_seed.plan.json"
    write_calibration_seed_plan(seed_plan, cfg, args.force)
    validate_plan(seed_plan, root, "calibration_seed")

    full_profile = calibrate(
        cfg, root, seed_plan, args.force, args.calibration_warmups, args.calibration_iterations
    )
    profiles = derive_cost_profiles(full_profile, root, args.force)

    plans: dict[str, dict[str, Path]] = {}
    for model_name, profile in profiles.items():
        plans[model_name] = {}
        plans[model_name]["joint"] = run_solver(
            cfg,
            root,
            f"{model_name}_joint",
            "slackpipe",
            cost_profile=profile,
            force=args.force,
            time_limit=args.solver_time_limit,
        )
        validate_plan(plans[model_name]["joint"], root, f"{model_name}_joint")
        plans[model_name]["uniform_schedule"] = run_solver(
            cfg,
            root,
            f"{model_name}_uniform_schedule",
            "schedule-only-uniform",
            cost_profile=profile,
            force=args.force,
            time_limit=args.solver_time_limit,
        )
        validate_plan(plans[model_name]["uniform_schedule"], root, f"{model_name}_uniform_schedule")

    rng = random.Random(args.seed + int(stable_digest(asdict(cfg)), 16))
    primary_jobs = [
        ("A_uniform_conventional", plans["M3_full"]["joint"], "A_uniform_interleaved_1f1b"),
        (
            "B_opt_partition_conventional",
            plans["M3_full"]["joint"],
            "B_solver_split_interleaved_1f1b",
        ),
        ("C_uniform_slackpipe", plans["M3_full"]["uniform_schedule"], "C_slackpipe_solver_order"),
        ("D_joint_slackpipe", plans["M3_full"]["joint"], "C_slackpipe_solver_order"),
    ]
    ablation_jobs = [
        (f"cost_{name}_joint_slackpipe", plan_set["joint"], "C_slackpipe_solver_order")
        for name, plan_set in plans.items()
        if name != "M3_full"
    ]
    jobs = primary_jobs + ablation_jobs

    benchmark_results: dict[str, list[Path]] = {label: [] for label, _, _ in jobs}
    for rep in range(args.repetitions):
        order = jobs[:]
        rng.shuffle(order)
        for label, plan, mode in order:
            summary = benchmark_once(
                cfg,
                root,
                plan,
                label,
                mode,
                rep,
                args.benchmark_warmups,
                args.benchmark_iterations,
                args.benchmark_timeout,
                args.force,
            )
            benchmark_results[label].append(summary)

    label_summaries: dict[str, dict[str, Any]] = {}
    for label, summaries in benchmark_results.items():
        samples_by_rep = []
        mode_name = next(mode for job_label, _, mode in jobs if job_label == label)
        last_summary = None
        for summary_path in summaries:
            samples, parsed = extract_global_times(summary_path, mode_name)
            samples_by_rep.append(samples)
            last_summary = parsed
        stats = summarize_samples(samples_by_rep)
        if last_summary is not None:
            mode_stats = last_summary["modes"][mode_name]
            stats["samples_per_second"] = (
                cfg.num_microbatches * cfg.micro_batch_size / (float(stats["mean_ms"]) / 1000.0)
            )
            stats["tokens_per_second"] = (
                cfg.num_microbatches
                * cfg.micro_batch_size
                * cfg.seq_length
                / (float(stats["mean_ms"]) / 1000.0)
            )
            stats["peak_allocated_bytes"] = mode_stats["peak_allocated_bytes"]
            stats["peak_reserved_bytes"] = mode_stats["peak_reserved_bytes"]
            stats["parameter_count"] = mode_stats["parameter_count"]
        label_summaries[label] = {"stats": stats, "samples_by_rep": samples_by_rep}

    primary = label_summaries
    A = primary["A_uniform_conventional"]["samples_by_rep"]
    B = primary["B_opt_partition_conventional"]["samples_by_rep"]
    C = primary["C_uniform_slackpipe"]["samples_by_rep"]
    D = primary["D_joint_slackpipe"]["samples_by_rep"]
    partition_mean, partition_ci, partition_label = speedup_ci(B, A)
    schedule_uniform_mean, schedule_uniform_ci, schedule_uniform_label = speedup_ci(C, A)
    schedule_opt_mean, schedule_opt_ci, schedule_opt_label = speedup_ci(D, B)
    joint_mean, joint_ci, joint_label = speedup_ci(D, A)

    profile = read_json(full_profile)
    config_summary = {
        "config": asdict(cfg),
        "paths": {
            "root": str(root),
            "profile": str(full_profile),
            "plans": {
                model: {kind: str(path) for kind, path in plan_set.items()}
                for model, plan_set in plans.items()
            },
        },
        "calibration": profile,
        "plans": {
            model: {kind: load_plan_info(path) for kind, path in plan_set.items()}
            for model, plan_set in plans.items()
        },
        "benchmarks": {label: summary["stats"] for label, summary in label_summaries.items()},
        "effects": {
            "partition_benefit": {
                "mean": partition_mean,
                "ci95": partition_ci,
                "label": partition_label,
            },
            "schedule_benefit_uniform": {
                "mean": schedule_uniform_mean,
                "ci95": schedule_uniform_ci,
                "label": schedule_uniform_label,
            },
            "schedule_benefit_optimized": {
                "mean": schedule_opt_mean,
                "ci95": schedule_opt_ci,
                "label": schedule_opt_label,
            },
            "joint_benefit": {"mean": joint_mean, "ci95": joint_ci, "label": joint_label},
            "interaction": joint_mean - partition_mean - schedule_uniform_mean,
        },
    }
    write_json(root / "summary.json", config_summary)
    return config_summary


def rows_from_summaries(summaries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    experiment_rows = []
    partition_rows = []
    cost_rows = []
    calibration_rows = []
    prediction_rows = []

    for summary in summaries:
        cfg = summary["config"]
        config_name = cfg["name"]
        profile = summary["calibration"]
        for stage in profile["observed_stages"]:
            pred_f = profile["a_fwd"] * stage["layer_count"] + profile["bias_fwd"][stage["stage"]]
            pred_b = profile["a_bwd"] * stage["layer_count"] + profile["bias_bwd"][stage["stage"]]
            calibration_rows.append(
                {
                    "config": config_name,
                    "stage": stage["stage"],
                    "layer_count": stage["layer_count"],
                    "forward_ms_per_op": stage["forward_ms_per_op"],
                    "backward_ms_per_op": stage["backward_ms_per_op"],
                    "bf_ratio": stage["backward_forward_ratio"],
                    "a_fwd": profile["a_fwd"],
                    "a_bwd": profile["a_bwd"],
                    "slope_bf_ratio": profile["measured_backward_forward_ratio"],
                    "bias_fwd": profile["bias_fwd"][stage["stage"]],
                    "bias_bwd": profile["bias_bwd"][stage["stage"]],
                    "forward_residual_ms": stage["forward_ms_per_op"] - pred_f,
                    "backward_residual_ms": stage["backward_ms_per_op"] - pred_b,
                }
            )

        for label, bench in summary["benchmarks"].items():
            row = {"config": config_name, "variant": label, **bench}
            experiment_rows.append(row)

        effects = summary["effects"]
        partition_rows.append(
            {
                "config": config_name,
                "partition_benefit_mean": effects["partition_benefit"]["mean"],
                "partition_benefit_ci95": effects["partition_benefit"]["ci95"],
                "partition_benefit_label": effects["partition_benefit"]["label"],
                "schedule_benefit_uniform_mean": effects["schedule_benefit_uniform"]["mean"],
                "schedule_benefit_uniform_ci95": effects["schedule_benefit_uniform"]["ci95"],
                "schedule_benefit_uniform_label": effects["schedule_benefit_uniform"]["label"],
                "schedule_benefit_optimized_mean": effects["schedule_benefit_optimized"]["mean"],
                "schedule_benefit_optimized_ci95": effects["schedule_benefit_optimized"]["ci95"],
                "schedule_benefit_optimized_label": effects["schedule_benefit_optimized"]["label"],
                "joint_benefit_mean": effects["joint_benefit"]["mean"],
                "joint_benefit_ci95": effects["joint_benefit"]["ci95"],
                "joint_benefit_label": effects["joint_benefit"]["label"],
                "interaction": effects["interaction"],
            }
        )

        for model in ["M0_equal", "M1_global_ratio", "M2_shared_slopes", "M3_full"]:
            label = "D_joint_slackpipe" if model == "M3_full" else f"cost_{model}_joint_slackpipe"
            bench = summary["benchmarks"][label]
            plan = summary["plans"][model]["joint"]
            cost_rows.append(
                {
                    "config": config_name,
                    "cost_model": model,
                    "split": " ".join(str(x) for x in plan["split"]),
                    "solver_status": plan["solver_status"],
                    "predicted_makespan": plan["predicted_makespan"],
                    "measured_mean_ms": bench["mean_ms"],
                    "measured_ci95_ms": bench["ci95_ms"],
                    "prediction_error_raw": plan["predicted_makespan"] - bench["mean_ms"],
                }
            )
            prediction_rows.append(
                {
                    "config": config_name,
                    "model_variant": model,
                    "schedule_variant": "D_joint_slackpipe",
                    "predicted_makespan": plan["predicted_makespan"],
                    "measured_time_ms": bench["mean_ms"],
                }
            )

        for variant, model, plan_kind in [
            ("A_uniform_conventional", "M3_full", "uniform_schedule"),
            ("C_uniform_slackpipe", "M3_full", "uniform_schedule"),
            ("D_joint_slackpipe", "M3_full", "joint"),
        ]:
            prediction_rows.append(
                {
                    "config": config_name,
                    "model_variant": model,
                    "schedule_variant": variant,
                    "predicted_makespan": summary["plans"][model][plan_kind]["predicted_makespan"],
                    "measured_time_ms": summary["benchmarks"][variant]["mean_ms"],
                }
            )
    return {
        "experiment_summary": experiment_rows,
        "partition_schedule_ablation": partition_rows,
        "cost_model_ablation": cost_rows,
        "calibration_profiles": calibration_rows,
        "prediction_vs_measurement": prediction_rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        write_text(path, "")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_artifacts(root: Path, summaries: list[dict[str, Any]]) -> None:
    write_json(root / "experiment_summary.json", {"configs": summaries})
    tables = rows_from_summaries(summaries)
    for name, rows in tables.items():
        write_csv(root / f"{name}.csv", rows)
    maybe_write_plots(root, tables)


def maybe_write_plots(root: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    rows = tables["partition_schedule_ablation"]
    if rows:
        labels = [r["config"] for r in rows]
        x = range(len(labels))
        plt.figure(figsize=(max(8, len(labels) * 1.4), 4))
        plt.bar(
            [i - 0.25 for i in x],
            [100 * r["partition_benefit_mean"] for r in rows],
            width=0.25,
            label="partition",
        )
        plt.bar(
            x,
            [100 * r["schedule_benefit_uniform_mean"] for r in rows],
            width=0.25,
            label="schedule uniform",
        )
        plt.bar(
            [i + 0.25 for i in x],
            [100 * r["joint_benefit_mean"] for r in rows],
            width=0.25,
            label="joint",
        )
        plt.xticks(list(x), labels, rotation=35, ha="right")
        plt.ylabel("speedup (%)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots / "partition_schedule_joint_speedup.png", dpi=160)
        plt.close()

    rows = tables["cost_model_ablation"]
    if rows:
        by_model: dict[str, list[float]] = {}
        for row in rows:
            by_model.setdefault(row["cost_model"], []).append(row["measured_mean_ms"])
        plt.figure(figsize=(7, 4))
        plt.boxplot([by_model[k] for k in sorted(by_model)], labels=sorted(by_model))
        plt.ylabel("measured SlackPipe time (ms)")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(plots / "cost_model_ablation.png", dpi=160)
        plt.close()

    rows = tables["prediction_vs_measurement"]
    if rows:
        plt.figure(figsize=(5, 4))
        plt.scatter([r["predicted_makespan"] for r in rows], [r["measured_time_ms"] for r in rows])
        plt.xlabel("predicted makespan")
        plt.ylabel("measured time (ms)")
        plt.tight_layout()
        plt.savefig(plots / "prediction_vs_measurement.png", dpi=160)
        plt.close()


def select_configs(args: argparse.Namespace) -> list[ExperimentConfig]:
    configs = SMOKE_CONFIGS if args.config_set == "smoke" else DEFAULT_CONFIGS
    if args.config:
        wanted = set(args.config)
        configs = [cfg for cfg in configs if cfg.name in wanted]
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPO / "slackpipe_experiments")
    parser.add_argument("--config-set", choices=["default", "smoke"], default="default")
    parser.add_argument("--config", action="append", help="Run only the named config; repeatable.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--benchmark-warmups", type=int, default=5)
    parser.add_argument("--benchmark-iterations", type=int, default=50)
    parser.add_argument("--benchmark-timeout", type=int, default=600)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--calibration-warmups", type=int, default=5)
    parser.add_argument("--calibration-iterations", type=int, default=10)
    parser.add_argument("--solver-time-limit", type=float, default=30.0)
    parser.add_argument("--skip-regressions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    configs = select_configs(args)
    if not configs:
        raise SystemExit("no configurations selected")

    run(["git", "rev-parse", "HEAD"], log_path=args.output_dir / "logs" / "outer_commit.log")
    run(["git", "status", "--short"], log_path=args.output_dir / "logs" / "outer_status.log")
    run(
        ["git", "-C", "slackpipe", "rev-parse", "HEAD"],
        log_path=args.output_dir / "logs" / "nested_commit.log",
    )
    run(
        ["git", "-C", "slackpipe", "status", "--short"],
        log_path=args.output_dir / "logs" / "nested_status.log",
    )

    build_solver(args.force, args.output_dir)
    regressions: dict[str, Any] = {}
    if not args.skip_regressions:
        regressions["start"] = run_regressions(args.output_dir, "start")

    summaries = []
    for cfg in configs:
        summaries.append(run_config(cfg, args))
        write_artifacts(args.output_dir, summaries)

    if not args.skip_regressions:
        regressions["end"] = run_regressions(args.output_dir, "end")
    write_json(args.output_dir / "regression_results.json", regressions)
    write_artifacts(args.output_dir, summaries)
    print(f"Wrote study artifacts under {args.output_dir}")


if __name__ == "__main__":
    main()
