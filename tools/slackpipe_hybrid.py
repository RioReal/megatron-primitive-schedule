# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Offline Nemotron-H architecture, class fitting, and future four-GPU launchers."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from megatron.core.pipeline_parallel.slackpipe.cost_profile import build_heterogeneous_cost_profile
from megatron.core.pipeline_parallel.slackpipe.hybrid import (
    NEMOTRON_H_8B_MAX_SEQUENCE_LENGTH,
    NEMOTRON_H_8B_PATTERN,
    NEMOTRON_H_8B_SOURCE,
    NEMOTRON_H_8B_VOCAB_SIZE,
    nemotron_h_8b_config,
    partition_hybrid_pattern,
)
from megatron.core.pipeline_parallel.slackpipe.manifest import (
    build_model_manifest,
    validate_plan_model,
)
from megatron.core.pipeline_parallel.slackpipe.plan import (
    load_slackpipe_plan,
    validate_plan_parallel_layout,
)


def write_json(path: Path, payload: object) -> None:
    """Atomically publish a completed workflow artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def launch_command(args):
    config = nemotron_h_8b_config(
        params_dtype=torch.bfloat16, pipeline_dtype=torch.bfloat16, bf16=True
    )
    if not 1 <= args.seq_length <= NEMOTRON_H_8B_MAX_SEQUENCE_LENGTH:
        raise ValueError("Base-8K sequence length must be in [1, 8192]")
    extra = []
    if args.action == "slackpipe":
        if not args.plan:
            raise ValueError("--plan is required for SlackPipe")
        plan = load_slackpipe_plan(args.plan, pipeline_model_parallel_size=4)
        validate_plan_parallel_layout(plan, 4)
        if (plan.num_stages, plan.num_microbatches, plan.num_layers) != (8, 8, config.num_layers):
            raise ValueError("Nemotron-H target requires N=8, B=8, L=52")
        validate_plan_model(plan, config)
        extra = [
            "--pipeline-schedule",
            "slackpipe",
            "--slackpipe-plan",
            str(args.plan),
            "--slackpipe-transport",
            args.transport,
        ]
    else:
        cuts = [config.num_layers * s // 8 for s in range(9)]
        plan = SimpleNamespace(
            num_layers=config.num_layers, stage_layer_ranges=tuple(zip(cuts, cuts[1:]))
        )
    pattern = partition_hybrid_pattern(NEMOTRON_H_8B_PATTERN, plan)
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        "pretrain_hybrid.py",
        "--bf16",
        "--pipeline-model-parallel-size",
        "4",
        "--tensor-model-parallel-size",
        "1",
        "--context-parallel-size",
        "1",
        "--hybrid-layer-pattern",
        pattern,
        "--num-layers",
        str(config.num_layers),
        "--hidden-size",
        str(config.hidden_size),
        "--ffn-hidden-size",
        str(config.ffn_hidden_size),
        "--num-attention-heads",
        str(config.num_attention_heads),
        "--group-query-attention",
        "--num-query-groups",
        str(config.num_query_groups),
        "--mamba-state-dim",
        str(config.mamba_state_dim),
        "--mamba-head-dim",
        str(config.mamba_head_dim),
        "--mamba-num-groups",
        str(config.mamba_num_groups),
        "--mamba-num-heads",
        str(config.mamba_num_heads),
        "--normalization",
        "RMSNorm",
        "--norm-epsilon",
        str(config.layernorm_epsilon),
        "--squared-relu",
        "--disable-bias-linear",
        "--no-bias-gelu-fusion",
        "--hidden-dropout",
        "0",
        "--attention-dropout",
        "0",
        "--position-embedding-type",
        "none",
        "--untie-embeddings-and-output-weights",
        "--spec",
        "megatron.core.models.hybrid.hybrid_layer_specs",
        "hybrid_stack_spec",
        "--seq-length",
        str(args.seq_length),
        "--max-position-embeddings",
        str(NEMOTRON_H_8B_MAX_SEQUENCE_LENGTH),
        "--micro-batch-size",
        "1",
        "--global-batch-size",
        "8",
        "--train-iters",
        "1",
        "--optimizer",
        "sgd",
        "--lr",
        "0.000001",
        "--lr-decay-style",
        "constant",
        "--sgd-momentum",
        "0",
        "--no-gradient-accumulation-fusion",
        "--no-overlap-p2p-communication",
        "--mock-data",
        "--tokenizer-type",
        "NullTokenizer",
        "--vocab-size",
        str(NEMOTRON_H_8B_VOCAB_SIZE),
        "--data-cache-path",
        str(args.output / "data_cache"),
        "--eval-iters",
        "0",
        "--eval-interval",
        "1000",
        "--num-workers",
        "0",
        "--log-interval",
        "1",
        "--seed",
        "1234",
        "--no-one-logger",
        *extra,
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("manifest", "fit", "baseline", "slackpipe"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--observations", type=Path, help="stage-level rows from ordinary Megatron calibration"
    )
    parser.add_argument(
        "--manifest", type=Path, help="actual model_manifest.v1, defaults to Nemotron-H Base-8K"
    )
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--transport", choices=("nccl-p2p", "nccl-rma"), default="nccl-p2p")
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument(
        "--calibration-pp", type=int, help="physical PP used to collect fit observations"
    )
    parser.add_argument(
        "--calibration-vpp", type=int, help="chunks per rank used to collect fit observations"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = (
        json.loads(args.manifest.read_text())
        if args.manifest
        else build_model_manifest(nemotron_h_8b_config())
    )
    if args.action in ("manifest", "fit"):
        payload = manifest
        if args.action == "fit":
            if not args.observations:
                parser.error("fit requires --observations")
            if (
                not args.calibration_pp
                or args.calibration_pp < 1
                or not args.calibration_vpp
                or args.calibration_vpp < 1
            ):
                parser.error("fit requires positive --calibration-pp and --calibration-vpp")
            payload = build_heterogeneous_cost_profile(
                model_manifest=manifest,
                observed_stage_rows=json.loads(args.observations.read_text()),
                model_config=manifest["model_config"],
                parallel_config={
                    "pp": args.calibration_pp,
                    "vpp": args.calibration_vpp,
                    "tp": 1,
                    "dp": 1,
                    "cp": 1,
                },
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        source = args.manifest or NEMOTRON_H_8B_SOURCE
        print(f"Wrote {args.output}; architecture source: {source}")
        return
    command = launch_command(args)
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        if torch.cuda.device_count() != 4:
            parser.error("requires 4 CUDA devices; use CUDA_VISIBLE_DEVICES to select exactly four")
        args.output.mkdir(parents=True, exist_ok=True)
        env = dict(
            os.environ,
            OMP_NUM_THREADS="1",
            TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0",
            MAMBA_DETERMINISTIC="1",
            TRITON_CACHE_AUTOTUNING="0",
            NVTE_ALLOW_NONDETERMINISTIC_ALGO="0",
        )
        subprocess.run(command, check=True, env=env, cwd=Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
