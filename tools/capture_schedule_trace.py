"""Capture selected steady-state steps; run with two-rank torchrun in slackpipe-dev."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.slackpipe.collection import CollectionOptions, collect_steps
from megatron.core.pipeline_parallel.slackpipe.plan import load_slackpipe_plan
from megatron.core.pipeline_parallel.slackpipe.schedule import shutdown_slackpipe_runtime
from tests.unit_tests.pipeline_parallel import slackpipe_perf_benchmark as bench


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slackpipe-profiler-output", type=Path, required=True)
    parser.add_argument(
        "--slackpipe-profile-start-step", "--warmup-iterations", type=int, default=20
    )
    parser.add_argument("--slackpipe-profile-num-steps", "--profiler-active", type=int, default=3)
    parser.add_argument(
        "--collection-mode", choices=("benchmark", "timeline", "memory"), default="timeline"
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--profiler-wait", type=int, default=2)
    parser.add_argument("--profiler-warmup", type=int, default=2)
    parser.add_argument("--profiler-repeat", type=int, default=3)
    parser.add_argument("--run-id")
    parser.add_argument("--memory-history-entries", type=int, default=100000)
    parser.add_argument(
        "--slackpipe-profile-kind", choices=("baseline", "slackpipe"), required=True
    )
    parser.add_argument(
        "--slackpipe-profile-format",
        choices=("torch", "compact", "both"),
        default="both",
        help="Legacy option; all profiling modes now retain both raw and compact files for auditability",
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--heterogeneous-config", type=Path)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-attention-heads", type=int, default=16)
    parser.add_argument("--seq-length", type=int, default=256)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--learning-rate", type=float, default=1e-9)
    parser.add_argument(
        "--slackpipe-transport", choices=("nccl-rma", "nccl-p2p"), default="nccl-rma"
    )
    args = parser.parse_args()
    if args.slackpipe_profile_start_step < 1 or args.slackpipe_profile_num_steps < 1:
        parser.error("Use at least one warmup and one captured step")
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        parser.error("This real-model capture driver requires PP=2; no PP4 support is added")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan = load_slackpipe_plan(args.plan, pipeline_model_parallel_size=2)
    if plan.num_stages != 4:
        parser.error("The existing model harness supports N=4 for this figure demo")
    args.num_layers, args.num_microbatches = plan.num_layers, plan.num_microbatches
    args.slackpipe_runtime, args.disable_slackpipe_nvtx = "fast", True
    kind = args.slackpipe_profile_kind
    split = (
        [plan.num_layers // plan.num_stages] * plan.num_stages
        if kind == "baseline"
        else list(plan.layer_split)
    )
    if kind == "baseline" and plan.num_layers % plan.num_stages:
        parser.error("Uniform baseline requires layers divisible by logical stages")
    mode = dict(
        name=kind,
        schedule="default" if kind == "baseline" else "slackpipe",
        layout=bench._layout_from_split(split),
    )
    output = args.slackpipe_profiler_output
    bench.Utils.initialize_model_parallel(
        1, 2, virtual_pipeline_model_parallel_size=plan.num_stages // 2
    )
    rank = dist.get_rank()
    config = dict(
        pp=2,
        num_stages=plan.num_stages,
        num_layers=plan.num_layers,
        layer_split=split,
        num_microbatches=plan.num_microbatches,
        hidden_size=args.hidden_size,
        heads=args.num_attention_heads,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        vocab_size=args.vocab_size,
        seed=args.seed,
        learning_rate=args.learning_rate,
        dtype="fp32",
        tf32=False,
        dropout=0,
        tp=1,
        dp=1,
        cp=1,
        warmup_steps=args.slackpipe_profile_start_step,
        plan=str(args.plan),
        model=str(args.heterogeneous_config),
        host=os.uname().nodename,
        model_sha256=(
            hashlib.sha256(args.heterogeneous_config.read_bytes()).hexdigest()
            if args.heterogeneous_config
            else "homogeneous"
        ),
        plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(),
    )
    try:
        model = bench._build_model(args, mode)
        optimizer = torch.optim.SGD(bench._logical_params(model).values(), lr=args.learning_rate)
        batches = bench._make_batches(args)
        params = bench._logical_params(model)
        result = collect_steps(
            step=bench.make_training_step(args, mode, model, optimizer, batches),
            optimizer=optimizer,
            options=CollectionOptions(
                mode=args.collection_mode,
                warmup=args.slackpipe_profile_start_step,
                iterations=args.iterations,
                wait=args.profiler_wait,
                profiler_warmup=args.profiler_warmup,
                active=args.slackpipe_profile_num_steps,
                repeat=args.profiler_repeat,
                run_id=args.run_id,
                history_entries=args.memory_history_entries,
            ),
            output=output,
            method=kind,
            transport="megatron-p2p" if kind == "baseline" else args.slackpipe_transport,
            config=config,
            plan=plan if kind == "slackpipe" else None,
            validate=lambda losses: bench._assert_finite(losses, params),
        )
        print(
            json.dumps(
                dict(
                    rank=rank,
                    run_id=result["run_id"],
                    captures=len(result["captures"]),
                    summary=result["summary"],
                )
            ),
            flush=True,
        )
    finally:
        shutdown_slackpipe_runtime()
        parallel_state.destroy_model_parallel()
        bench.Utils.inited = False
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
