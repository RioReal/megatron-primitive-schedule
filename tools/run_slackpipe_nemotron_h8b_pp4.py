# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Explicit, fail-closed stages for a random-init Nemotron-H 8B four-GPU run."""

import argparse
import hashlib
import importlib
import json
import os
import shlex
import signal
import statistics
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from megatron.core.pipeline_parallel.slackpipe.cost_profile import profile_fingerprint
from megatron.core.pipeline_parallel.slackpipe.hybrid import nemotron_h_8b_config
from megatron.core.pipeline_parallel.slackpipe.manifest import (
    build_model_manifest,
    validate_plan_model,
)
from megatron.core.pipeline_parallel.slackpipe.plan import (
    load_slackpipe_plan,
    validate_plan_parallel_layout,
)
from tools.slackpipe_hybrid import launch_command, write_json

ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    "env",
    "pp4-correctness",
    "baseline-smoke",
    "calibrate",
    "solve",
    "slackpipe-smoke",
    "benchmark",
    "trace",
    "memory",
)
REQUIRES = {
    "env": (),
    "pp4-correctness": ("env",),
    "baseline-smoke": ("pp4-correctness",),
    "calibrate": ("baseline-smoke",),
    "solve": ("calibrate",),
    "slackpipe-smoke": ("solve",),
    "benchmark": ("slackpipe-smoke",),
    "trace": ("benchmark",),
    "memory": ("slackpipe-smoke",),
}
DETERMINISTIC = dict(
    OMP_NUM_THREADS="1",
    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0",
    MAMBA_DETERMINISTIC="1",
    TRITON_CACHE_AUTOTUNING="0",
    NVTE_ALLOW_NONDETERMINISTIC_ALGO="0",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity() -> dict:
    return dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        diff=hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=ROOT)
        ).hexdigest(),
    )


def context(args) -> dict:
    return dict(
        **source_identity(),
        seq_length=args.seq_length,
        transport=args.transport,
        precision="bf16",
        pp=4,
        stages=8,
        microbatches=8,
    )


def require_receipt(output: Path, stage: str, expected: dict) -> None:
    path = output / "receipts" / f"{stage}.json"
    if not path.exists():
        raise RuntimeError(f"Missing successful prerequisite: {stage}; run that stage first")
    receipt = json.loads(path.read_text())
    if receipt["context"] != expected:
        raise RuntimeError(f"Stale prerequisite {stage}: code/config changed")
    for name, checksum in receipt["artifacts"].items():
        artifact = output / name
        if not artifact.is_file() or digest(artifact) != checksum:
            raise RuntimeError(f"Changed/missing prerequisite artifact: {name}")
    for parent in REQUIRES[stage]:
        require_receipt(output, parent, expected)


def invalidate(output: Path, stage: str) -> None:
    # A failed rerun must not leave a previous success or its descendants valid.
    invalid = {stage}
    for candidate in STAGES:
        if any(p in invalid for p in REQUIRES[candidate]):
            invalid.add(candidate)
    for name in invalid:
        (output / "receipts" / f"{name}.json").unlink(missing_ok=True)


def run(command: list, log: Path, timeout: int, env: dict) -> None:
    print(shlex.join(map(str, command)), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        process = subprocess.Popen(
            list(map(str, command)),
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if code:
        raise RuntimeError(f"Command failed ({code}); stopped. See {log}")


def preflight(output: Path, transport: str, minimum_gib: float) -> dict:
    errors = []
    metadata = dict(
        **source_identity(),
        torch=torch.__version__,
        python=sys.version,
        executable=sys.executable,
        deterministic_environment=DETERMINISTIC,
        visible_device_selection=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cuda=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),
        visible_gpus=torch.cuda.device_count(),
        nccl_available=dist.is_nccl_available(),
        target_image="nvcr.io/nvidia/pytorch:26.01-py3",
        transport=transport,
        dependencies={},
        devices=[],
    )
    if not metadata["cuda_available"] or metadata["visible_gpus"] < 4:
        errors.append("requires at least 4 CUDA GPUs")
    if not metadata["nccl_available"]:
        errors.append("NCCL unavailable")
    metadata["nccl"] = torch.cuda.nccl.version() if metadata["nccl_available"] else None
    try:
        metadata["topology"] = subprocess.check_output(
            ["nvidia-smi", "topo", "-m"], text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError) as exc:
        errors.append(f"GPU topology query failed: {exc}")
    for i in range(min(metadata["visible_gpus"], 4)):
        prop = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        metadata["devices"].append(
            dict(
                rank=i,
                name=prop.name,
                uuid=str(getattr(prop, "uuid", "unknown")),
                total_bytes=total,
                free_bytes=free,
            )
        )
        if "A100" not in prop.name or "SXM" not in prop.name:
            errors.append(f"GPU {i} is not an A100 SXM: {prop.name}")
        if free < minimum_gib * 1024**3:
            errors.append(f"GPU {i} needs >= {minimum_gib} GiB free; not an OOM guarantee")
        if not torch.cuda.is_bf16_supported():
            errors.append("BF16 unsupported")
    for name in ("transformer_engine", "mamba_ssm", "causal_conv1d", "mamba_ssm.utils.determinism"):
        try:
            module = importlib.import_module(name)
            metadata["dependencies"][name] = getattr(module, "__version__", "imported")
        except ImportError as exc:
            errors.append(f"{name}: {exc}")
    try:
        symm = importlib.import_module("torch.distributed._symmetric_memory")
        metadata["rma_available"] = metadata["nccl"] >= (2, 29, 0) and all(
            hasattr(symm, n) for n in ("put_signal", "wait_signal", "rendezvous")
        )
    except (ImportError, TypeError):
        metadata["rma_available"] = False
    if transport == "nccl-rma" and not metadata["rma_available"]:
        errors.append("RMA requires NCCL >=2.29 and PyTorch symmetric-memory RMA bindings")
    metadata["errors"] = errors
    write_json(output / "run_metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)
    if errors:
        raise RuntimeError("Hardware/dependency preflight failed; no expensive stage authorized")
    return metadata


def nccl_sanity() -> None:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(
        "nccl",
        timeout=timedelta(seconds=60),
        device_id=torch.device("cuda", torch.cuda.current_device()),
    )
    try:
        if dist.get_world_size() != 4:
            raise RuntimeError("sanity requires four ranks")
        for dtype in (torch.float32, torch.bfloat16):
            value = torch.tensor(float(dist.get_rank() + 1), device="cuda", dtype=dtype)
            dist.all_reduce(value)
            if value.item() != 10:
                raise RuntimeError("NCCL all-reduce failed")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def validate_target_plan(path: Path, profile_path: Path, manifest_path: Path, seq_length: int):
    config = nemotron_h_8b_config(
        params_dtype=torch.bfloat16, pipeline_dtype=torch.bfloat16, bf16=True
    )
    manifest = json.loads(manifest_path.read_text())
    if manifest != build_model_manifest(config):
        raise ValueError("Calibration manifest is not the authoritative Nemotron-H model")
    profile = json.loads(profile_path.read_text())
    if profile.get("cost_profile_hash") != profile_fingerprint(profile):
        raise ValueError("Invalid profile fingerprint")
    if (
        profile["model_config"].get("dtype") != "bf16"
        or profile["model_config"].get("sequence_length") != seq_length
    ):
        raise ValueError("Profile precision/sequence does not match execution")
    plan = load_slackpipe_plan(path, pipeline_model_parallel_size=4)
    validate_plan_parallel_layout(plan, 4)
    validate_plan_model(plan, config)
    if (plan.schema_version, plan.num_stages, plan.num_microbatches, plan.num_layers) != (
        "slackpipe.plan.v2",
        8,
        8,
        config.num_layers,
    ):
        raise ValueError("Target requires plan.v2 W4/N8/B8 and authoritative layer count")
    if (
        plan.model_manifest_hash != manifest["manifest_hash"]
        or plan.cost_profile_hash != profile["cost_profile_hash"]
    ):
        raise ValueError("Plan provenance mismatch")
    if Path(plan.cost_profile_path).resolve() != profile_path.resolve():
        raise ValueError("Plan references a different cost profile")
    if plan.stage_to_worker != tuple(s % 4 for s in range(8)):
        raise ValueError("Noncyclic placement")
    if any(count < 1 for count in plan.layer_split):
        raise ValueError("Hybrid stages require nonempty contiguous ranges")
    if sum(len(plan.worker_operations(r)) for r in range(4)) != 128:
        raise ValueError("Target requires exactly 128 F/B operations")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(*STAGES, "nccl-sanity"))
    parser.add_argument(
        "--output", type=Path, default=Path("/workspace/slackpipe-runs/nemotron-h8b")
    )
    parser.add_argument(
        "--transport", "--slackpipe-transport", choices=("nccl-p2p", "nccl-rma"), default="nccl-p2p"
    )
    parser.add_argument("--seq-length", type=int, choices=(1024, 2048, 4096, 8192), default=1024)
    parser.add_argument(
        "--solver", type=Path, default=ROOT / "slackpipe/build/ortools/slackpipe_cli"
    )
    parser.add_argument("--solver-seconds", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--minimum-free-gib", type=float, default=35)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--profiler-wait", type=int, default=2)
    parser.add_argument("--profiler-warmup", type=int, default=2)
    parser.add_argument("--profiler-active", type=int, default=3)
    parser.add_argument("--profiler-repeat", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--runs", type=int, default=3, help="Independent fresh-process benchmark repetitions"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.stage == "nccl-sanity":
        nccl_sanity()
        return
    if (
        args.warmups < 1
        or args.iterations < 10
        or args.timeout < 1
        or args.solver_seconds < 1
        or args.runs < 1
    ):
        parser.error("Require warmups>=1, iterations>=10 and positive timeouts")
    args.output = args.output.resolve()
    output, stage = args.output, args.stage
    expected = context(args)
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    stage=stage,
                    requires=REQUIRES[stage],
                    context=expected,
                    output=str(output),
                    dry_run=True,
                ),
                indent=2,
            )
        )
        return
    invalidate(output, stage)
    for parent in REQUIRES[stage]:
        require_receipt(output, parent, expected)
    if stage != "env":
        metadata = json.loads((output / "run_metadata.json").read_text())
        if torch.cuda.device_count() < 4 or metadata["torch"] != torch.__version__:
            raise RuntimeError("Hardware/runtime changed since preflight")
        uuids = [
            str(getattr(torch.cuda.get_device_properties(i), "uuid", "unknown")) for i in range(4)
        ]
        if uuids != [d["uuid"] for d in metadata["devices"]]:
            raise RuntimeError("GPU assignment changed since preflight")
        if (
            tuple(metadata["nccl"]) != torch.cuda.nccl.version()
            or metadata["cuda"] != torch.version.cuda
        ):
            raise RuntimeError("CUDA/NCCL changed since preflight")
        for name, version in metadata["dependencies"].items():
            if getattr(importlib.import_module(name), "__version__", "imported") != version:
                raise RuntimeError(f"Dependency changed since preflight: {name}")
    env = dict(os.environ, **DETERMINISTIC)
    env.pop("SLACKPIPE_HYBRID_PLAN", None)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    torchrun = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=4"]
    logs = output / "logs"
    artifacts = []

    def execute(command, label):
        log = logs / f"{label}.log"
        run(command, log, args.timeout, env)
        artifacts.append(log)

    profile = output / "calibration/cost_profile.json"
    manifest = output / "calibration/model_manifest.json"
    plan_path = output / "solve/slackpipe.plan.json"
    if stage == "env":
        preflight(output, args.transport, args.minimum_free_gib)
        execute(
            torchrun + ["-m", "tools.run_slackpipe_nemotron_h8b_pp4", "nccl-sanity"], "nccl-sanity"
        )
        artifacts.append(output / "run_metadata.json")
    elif stage == "pp4-correctness":
        env["SLACKPIPE_TEST_TRANSPORT"] = args.transport
        env["SLACKPIPE_CORRECTNESS_OUTPUT"] = str(output / "correctness")
        for precision in ("fp32", "bf16"):
            for rank in range(4):
                (output / "correctness" / f"{precision}.rank{rank}.json").unlink(missing_ok=True)
            execute(
                torchrun
                + [
                    "-m",
                    "pytest",
                    f"tests/unit_tests/pipeline_parallel/test_slackpipe_hybrid.py::test_hybrid_numerical_equivalence[{precision}-4]",
                    "-vv",
                    "-s",
                    "-ra",
                ],
                f"correctness-{precision}",
            )
            for rank in range(4):
                result = output / "correctness" / f"{precision}.rank{rank}.json"
                if not result.exists() or json.loads(result.read_text())["pp"] != 4:
                    raise RuntimeError(
                        "Correctness test skipped or did not produce all four rank results"
                    )
                artifacts.append(result)
    elif stage in ("baseline-smoke", "slackpipe-smoke"):
        if stage == "slackpipe-smoke":
            validate_target_plan(plan_path, profile, manifest, args.seq_length)
        options = SimpleNamespace(
            action="baseline" if stage == "baseline-smoke" else "slackpipe",
            seq_length=args.seq_length,
            output=output / stage,
            plan=plan_path,
            transport=args.transport,
        )
        execute(launch_command(options), stage)
    elif stage == "solve":
        info = json.loads(subprocess.check_output([str(args.solver), "build-info"], text=True))
        if not info["ortools_enabled"]:
            raise RuntimeError("Joint CP-SAT requires an OR-Tools-enabled solver")
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        execute(
            [
                args.solver,
                "--algorithm",
                "joint-unrestricted-no-overlap",
                "--B",
                "8",
                "--N",
                "8",
                "--J",
                "4",
                "--L",
                str(nemotron_h_8b_config().num_layers),
                "--ratio-num",
                "1",
                "--ratio-den",
                "1",
                "--time-limit-seconds",
                str(args.solver_seconds),
                "--num-workers",
                "2",
                "--random-seed",
                "1",
                "--require-optimal",
                "false",
                "--cost-profile",
                profile,
                "--output-prefix",
                plan_path.parent / "solver",
                "--emit-plan",
                plan_path,
            ],
            stage,
        )
        validate_target_plan(plan_path, profile, manifest, args.seq_length)
        artifacts.append(plan_path)
    else:
        if stage != "calibrate":
            validate_target_plan(plan_path, profile, manifest, args.seq_length)
        modes = (
            ("baseline",)
            if stage == "calibrate"
            else (
                ("baseline", "partition", "slackpipe")
                if stage == "benchmark"
                else ("baseline", "slackpipe")
            )
        )
        jobs = [
            (mode, rep) for rep in range(args.runs if stage == "benchmark" else 1) for mode in modes
        ]
        for mode, rep in jobs:
            directory = output / "calibration" if stage == "calibrate" else output / stage / mode
            if stage == "benchmark":
                directory = directory / f"run{rep:03d}"
            execute(
                torchrun
                + [
                    "-m",
                    "tools.slackpipe_nemotron_worker",
                    stage,
                    "--output",
                    directory,
                    "--mode",
                    mode,
                    "--plan",
                    plan_path,
                    "--transport",
                    args.transport,
                    "--seq-length",
                    str(args.seq_length),
                    "--warmups",
                    str(args.warmups),
                    "--iterations",
                    str(args.iterations),
                    "--profiler-wait",
                    str(args.profiler_wait),
                    "--profiler-warmup",
                    str(args.profiler_warmup),
                    "--profiler-active",
                    str(args.profiler_active),
                    "--profiler-repeat",
                    str(args.profiler_repeat),
                ],
                f"{stage}-{mode}-run{rep:03d}",
            )
            artifacts.extend(directory.rglob("*.json"))
        if stage == "trace":
            execute(
                [
                    sys.executable,
                    "-m",
                    "tools.plot_schedule_trace",
                    "--baseline",
                    output / "trace/baseline",
                    "--slackpipe",
                    output / "trace/slackpipe",
                    "--output-png",
                    output / "trace/timeline.png",
                    "--output-pdf",
                    output / "trace/timeline.pdf",
                    "--report",
                    output / "trace/figure_report.json",
                    "--iteration",
                    str(args.warmups + args.profiler_wait + args.profiler_warmup),
                    "--title",
                    "Nemotron-H 8B: PP=4, N=8, B=8",
                ],
                "plot",
            )
        if stage == "benchmark":
            summary = {}
            for mode in modes:
                from megatron.core.pipeline_parallel.slackpipe.collection import (
                    summarize_rank_samples,
                )

                runs = []
                for rep in range(args.runs):
                    results = [
                        json.loads(
                            (
                                output / stage / mode / f"run{rep:03d}" / f"result.rank{r}.json"
                            ).read_text()
                        )
                        for r in range(4)
                    ]
                    if any(len(r["iteration_ms"]) != args.iterations for r in results):
                        raise RuntimeError("Incomplete benchmark samples")
                    runs.append(summarize_rank_samples(results))
                means = [r["mean_ms"] for r in runs]
                summary[mode] = dict(
                    runs=runs,
                    run_means_ms=means,
                    mean_ms=statistics.fmean(means),
                    run_mean_stddev_ms=statistics.stdev(means) if len(means) > 1 else None,
                    within_run_stddev_ms=[r["stddev_ms"] for r in runs],
                )
            path = output / stage / "summary.json"
            write_json(path, dict(context=expected, modes=summary, warmups_excluded=args.warmups))
            artifacts.append(path)
    if not artifacts:
        raise RuntimeError("No artifacts produced; refusing success receipt")
    write_json(
        output / "receipts" / f"{stage}.json",
        dict(
            context=expected,
            collection_policy=dict(
                warmups=args.warmups,
                iterations=args.iterations,
                runs=args.runs,
                wait=args.profiler_wait,
                profiler_warmup=args.profiler_warmup,
                active=args.profiler_active,
                repeat=args.profiler_repeat,
            ),
            artifacts={str(p.relative_to(output)): digest(p) for p in artifacts},
        ),
    )
    print(f"Passed {stage}. Artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
