# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native hybrid calibration, benchmark and trace worker (launched by torchrun).

The default architecture is the released random-init 8B model. --tiny is solely
for local harness tests; it cannot produce a certificate for a real pod run.
"""

import argparse
import gc
import hashlib
import json
import os
import traceback
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.pipeline_parallel.schedules import (
    begin_slackpipe_cost_calibration_iteration,
    end_slackpipe_cost_calibration_iteration,
    get_forward_backward_func,
)
from megatron.core.pipeline_parallel.slackpipe.collection import (
    CollectionOptions,
    collect_steps,
    environment_metadata,
    memory_sample,
    profiler_settings,
)
from megatron.core.pipeline_parallel.slackpipe.cost_profile import (
    aggregate_stage_costs,
    build_heterogeneous_cost_profile,
    profile_fingerprint,
    stage_role,
    write_cost_profile,
)
from megatron.core.pipeline_parallel.slackpipe.hybrid import (
    NEMOTRON_H_8B_MAX_SEQUENCE_LENGTH,
    NEMOTRON_H_8B_VOCAB_SIZE,
    nemotron_h_8b_config,
    partition_hybrid_pattern,
)
from megatron.core.pipeline_parallel.slackpipe.manifest import (
    build_model_manifest,
    validate_chunk_layers,
    validate_plan_model,
)
from megatron.core.pipeline_parallel.slackpipe.plan import load_slackpipe_plan, parse_slackpipe_plan
from megatron.core.pipeline_parallel.slackpipe.schedule import (
    shutdown_slackpipe_runtime,
    slackpipe_transport_statistics,
)
from tests.unit_tests.pipeline_parallel.slackpipe_perf_benchmark import _make_batches
from tests.unit_tests.pipeline_parallel.test_slackpipe_model_construction import (
    _batch_iterator,
    _build_model,
    _forward_step_func,
    _logical_named_parameters,
)
from tools.slackpipe_hybrid import write_json


def uniform_cuts(layers: int, stages: int) -> tuple:
    if layers < stages:
        raise ValueError("At least one layer per stage is required")
    return tuple(layers * s // stages for s in range(stages + 1))


def calibration_partitions(manifest: dict, stages: int, minimum: int = 6) -> tuple:
    """Choose full-rank, near-uniform designs without enumerating L choose N.

    Moving each boundary by at most two layers keeps calibration memory close
    to the smoke-tested uniform partition. Refuse unidentifiable architectures.
    """
    classes = [r["config_class"] for r in manifest["layers"]]
    keys = sorted(set(classes))
    base = uniform_cuts(len(classes), stages)
    candidates = {base}
    for s in range(1, stages):
        for shift in (-2, -1, 1, 2):
            cuts = list(base)
            cuts[s] += shift
            if all(a < b for a, b in zip(cuts, cuts[1:])):
                candidates.add(tuple(cuts))

    def design(cuts):
        return np.asarray(
            [
                [classes[a:b].count(k) for k in keys]
                + [int(stage_role(s, stages) == r) for r in ("first", "middle", "last")]
                for s, (a, b) in enumerate(zip(cuts, cuts[1:]))
            ],
            dtype=float,
        )

    selected = [base]
    columns = len(keys) + 3
    while True:
        matrix = np.concatenate([design(c) for c in selected])
        rank = np.linalg.matrix_rank(matrix)
        if rank == columns and len(selected) >= minimum:
            return selected, dict(
                rank=int(rank), columns=columns, condition=float(np.linalg.cond(matrix))
            )
        remaining = candidates - set(selected)
        if not remaining:
            raise ValueError("Rank-deficient calibration design; refusing to fit")

        def score(c):
            aug = np.concatenate([matrix, design(c)])
            return (-np.linalg.matrix_rank(aug), np.linalg.cond(aug), c)

        selected.append(min(remaining, key=score))


def construction_plan(cuts: tuple, pp: int, microbatches: int, manifest: dict):
    """Structural construction fixture, never a replacement for solver ordering."""
    stages = len(cuts) - 1
    operations = [[] for _ in range(pp)]
    for kind, order in (("F", range(stages)), ("B", range(stages - 1, -1, -1))):
        for b in range(microbatches):
            for s in order:
                operations[s % pp].append(dict(kind=kind, microbatch=b, stage=s))
    return parse_slackpipe_plan(
        dict(
            schema_version="slackpipe.plan.v2",
            num_layers=cuts[-1],
            num_workers=pp,
            num_stages=stages,
            num_microbatches=microbatches,
            layer_cuts=cuts,
            stage_to_worker=[s % pp for s in range(stages)],
            operations=operations,
            model_manifest_hash=manifest["manifest_hash"],
        )
    )


def initialize_parameters(parameters: dict, seed: int, layers: int) -> None:
    """Partition-independent mock weights; preserve stable Mamba state scales."""
    with torch.no_grad():
        for name, param in parameters.items():
            generator = torch.Generator(device=param.device)
            generator.manual_seed(
                seed + int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
            )
            if name.endswith("A_log"):
                param.uniform_(1, 16, generator=generator).log_()
            elif name.endswith("dt_bias"):
                dt = (
                    torch.empty_like(param, dtype=torch.float32)
                    .uniform_(-6.907755, -2.302585, generator=generator)
                    .exp_()
                )
                param.copy_(dt + torch.log(-torch.expm1(-dt)))
            elif name.endswith(".D") or (param.ndim == 1 and name.endswith("weight")):
                param.fill_(1)
            elif name.endswith("bias"):
                param.zero_()
            else:
                std = 0.02 / (
                    (2 * layers) ** 0.5
                    if name.endswith(("out_proj.weight", "linear_proj.weight", "linear_fc2.weight"))
                    else 1
                )
                param.normal_(0, std, generator=generator)


def run_partition(args, config, manifest, plan, output: Path, collect: str) -> list:
    rank = dist.get_rank()
    schedule = "slackpipe" if args.mode == "slackpipe" else "default"
    config.pipeline_schedule = schedule
    config.deallocate_pipeline_outputs = schedule == "default"
    pattern = partition_hybrid_pattern(config.slackpipe_hybrid_pattern, plan)

    def provider(
        pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None
    ):
        return HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=args.vocab_size,
            max_sequence_length=args.seq_length,
            hybrid_layer_pattern=pattern,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
            pg_collection=pg_collection,
            position_embedding_type="none",
            share_embeddings_and_output_weights=False,
        )

    model = _build_model(
        config,
        pipeline_schedule=schedule,
        pp_size=args.pp,
        vpp=args.stages // args.pp,
        provider=provider,
    )
    records = validate_chunk_layers(plan, model, rank)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / f"layers.rank{rank}.json", records)
    params = _logical_named_parameters(model)
    initialize_parameters(params, args.seed, config.num_layers)
    optimizer = torch.optim.SGD(params.values(), lr=args.learning_rate)
    batches = _make_batches(args)
    fb = (
        get_forward_backward_func(
            pipeline_schedule="slackpipe",
            slackpipe_plan_path=str(args.plan),
            slackpipe_transport=args.transport,
            slackpipe_runtime="fast",
            slackpipe_enable_nvtx=collect in ("trace", "timeline", "memory"),
        )
        if schedule == "slackpipe"
        else get_forward_backward_func(pp_size=args.pp, vp_size=args.stages // args.pp)
    )

    def iteration():
        optimizer.zero_grad(set_to_none=True)
        losses = fb(
            forward_step_func=_forward_step_func,
            data_iterator=[_batch_iterator(batches) for _ in model],
            model=model,
            num_microbatches=args.num_microbatches,
            seq_length=args.seq_length,
            micro_batch_size=1,
            forward_only=False,
        )
        optimizer.step()
        return losses

    def assert_finite(losses):
        flags = [torch.isfinite(p).all() for p in params.values()]
        flags += [torch.isfinite(p.grad).all() for p in params.values() if p.grad is not None]
        flags += [torch.isfinite(l["loss"]).all() for l in losses]
        finite = torch.stack(flags).all().to(torch.int32)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise RuntimeError("Non-finite loss/gradient/parameter")

    events, measured = [], {}
    if collect == "calibrate":
        before_warmup = memory_sample()
        for _ in range(args.warmups):
            losses = iteration()
        assert_finite(losses)
        measured = dict(
            collection_mode="calibration",
            run_id=args.run_id or str(args.output),
            rank=rank,
            partition_output=str(output.relative_to(args.output)),
            stage_layer_ranges=list(plan.stage_layer_ranges),
            seed=args.seed,
            optimizer=dict(name="SGD", learning_rate=args.learning_rate),
            environment=environment_metadata(),
            profiler=profiler_settings(CollectionOptions()),
            warmup_iterations=list(range(args.warmups)),
            measurement_iterations=list(range(args.warmups, args.warmups + args.iterations)),
            memory_before_warmup=before_warmup,
            memory_after_warmup=memory_sample(),
        )
        for i in measured["measurement_iterations"]:
            begin_slackpipe_cost_calibration_iteration(i)
            try:
                losses = iteration()
            finally:
                events.extend(end_slackpipe_cost_calibration_iteration())
            assert_finite(losses)
        measured["memory_end"] = memory_sample()
    else:
        metadata = dict(
            pp=args.pp,
            num_stages=args.stages,
            num_layers=config.num_layers,
            num_microbatches=args.num_microbatches,
            layer_split=list(plan.layer_split),
            hidden_size=config.hidden_size,
            heads=config.num_attention_heads,
            seq_length=args.seq_length,
            micro_batch_size=1,
            vocab_size=args.vocab_size,
            seed=args.seed,
            learning_rate=args.learning_rate,
            dtype=args.precision,
            tf32=False,
            dropout=0,
            tp=1,
            dp=1,
            cp=1,
            model="Nemotron-H-8B-Base-8K" if not args.tiny else "tiny-hybrid",
            model_sha256=manifest["manifest_hash"],
            host=os.uname().nodename,
            plan_sha256=(
                hashlib.sha256(args.plan.read_bytes()).hexdigest()
                if args.mode != "baseline"
                else None
            ),
        )
        measured = collect_steps(
            step=lambda i: iteration(),
            optimizer=optimizer,
            options=CollectionOptions(
                mode="timeline" if collect == "trace" else collect,
                warmup=args.warmups,
                iterations=args.iterations,
                wait=args.profiler_wait,
                profiler_warmup=args.profiler_warmup,
                active=args.profiler_active,
                repeat=args.profiler_repeat,
                run_id=args.run_id,
                history_entries=args.memory_history_entries,
            ),
            output=output,
            method=args.mode,
            transport=args.transport if schedule == "slackpipe" else "megatron-p2p",
            config=metadata,
            plan=plan if schedule == "slackpipe" else None,
            validate=assert_finite,
        )
        measured["iteration_ms"] = [s["cuda_elapsed_ms"] for s in measured["samples"]]
        measured["wall_ms"] = measured["continuous_wall_ms"]
    write_json(output / f"result.rank{rank}.json", measured)
    gathered = [None] * args.pp
    dist.all_gather_object(gathered, events)
    shutdown_slackpipe_runtime()
    if slackpipe_transport_statistics():
        raise RuntimeError("Transport state remains after shutdown")
    optimizer = params = model = batches = fb = None
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    return [event for worker in gathered for event in worker]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("calibrate", "benchmark", "trace", "timeline", "memory"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("baseline", "partition", "slackpipe"), default="baseline"
    )
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--transport", choices=("nccl-p2p", "nccl-rma"), default="nccl-p2p")
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--profiler-wait", type=int, default=2)
    parser.add_argument("--profiler-warmup", type=int, default=2)
    parser.add_argument("--profiler-active", type=int, default=3)
    parser.add_argument("--profiler-repeat", type=int, default=3)
    parser.add_argument("--memory-history-entries", type=int, default=100000)
    parser.add_argument("--run-id")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    if list(args.output.glob("result.rank*.json")) or (args.output / "cost_profile.json").exists():
        parser.error("Refusing to overwrite an existing collection; use a fresh --output")
    args.pp = int(os.environ["WORLD_SIZE"])
    if (not args.tiny and args.pp != 4) or args.pp not in (2, 4):
        parser.error("real 8B requires four ranks; --tiny supports two or four")
    if (
        args.warmups < 1
        or args.iterations < 1
        or not 1 <= args.seq_length <= NEMOTRON_H_8B_MAX_SEQUENCE_LENGTH
    ):
        parser.error("invalid warmup/iteration/sequence length")
    if args.action == "calibrate" and (args.iterations < 10 or args.mode != "baseline"):
        parser.error("calibration requires ordinary baseline and >=10 iterations")
    args.stages, args.num_microbatches, args.micro_batch_size, args.seed, args.learning_rate = (
        2 * args.pp,
        8,
        1,
        1234,
        1e-6,
    )
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    if args.tiny:
        from tests.unit_tests.pipeline_parallel.test_slackpipe_hybrid import tiny_config

        config = tiny_config(args.pp, 2, dtype=dtype)
        args.vocab_size = 128
    else:
        config = nemotron_h_8b_config(
            pipeline_model_parallel_size=args.pp,
            virtual_pipeline_model_parallel_size=2,
            params_dtype=dtype,
            pipeline_dtype=dtype,
            bf16=dtype == torch.bfloat16,
            batch_p2p_comm=True,
            overlap_p2p_comm=False,
        )
        args.vocab_size = NEMOTRON_H_8B_VOCAB_SIZE
    manifest = build_model_manifest(config)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    dist.all_reduce(torch.zeros(1, device="cuda"))
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=args.pp,
        virtual_pipeline_model_parallel_size=2,
    )
    rank = dist.get_rank()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        if rank == 0:
            write_json(args.output / "model_manifest.json", manifest)
        if args.action == "calibrate":
            partitions, diagnostics = calibration_partitions(manifest, args.stages)
        else:
            partitions, diagnostics = [uniform_cuts(config.num_layers, args.stages)], None
        observed, raw = [], []
        for i, cuts in enumerate(partitions):
            plan = construction_plan(cuts, args.pp, args.num_microbatches, manifest)
            if args.mode != "baseline":
                if not args.plan:
                    raise ValueError("--plan required for optimized partition or SlackPipe")
                plan = load_slackpipe_plan(args.plan, pipeline_model_parallel_size=args.pp)
                validate_plan_model(plan, config)
                if (plan.num_stages, plan.num_microbatches) != (args.stages, args.num_microbatches):
                    raise ValueError("Plan N/B mismatch")
            output = args.output / f"partition{i}" if args.action == "calibrate" else args.output
            events = run_partition(args, config, manifest, plan, output, args.action)
            if args.action == "calibrate" and rank == 0:
                raw.append(dict(cuts=cuts, events=events))
                rows = aggregate_stage_costs(
                    events,
                    layer_split=plan.layer_split,
                    num_microbatches=args.num_microbatches,
                    iteration_start=args.warmups,
                    iteration_end=args.warmups + args.iterations - 1,
                )
                for s, row in enumerate(rows):
                    row.update(
                        partition=i,
                        stage_layer_range=list(plan.stage_layer_ranges[s]),
                        stage_role=stage_role(s, args.stages),
                    )
                observed.extend(rows)
                write_json(args.output / "calibration_events.json", raw)
        if args.action == "calibrate" and rank == 0:
            profile = build_heterogeneous_cost_profile(
                model_manifest=manifest,
                observed_stage_rows=observed,
                model_config={
                    **manifest["model_config"],
                    "dtype": args.precision,
                    "sequence_length": args.seq_length,
                    "vocab_size": args.vocab_size,
                    "micro_batch_size": 1,
                },
                parallel_config=dict(pp=args.pp, vpp=2, tp=1, dp=1, cp=1),
            )
            profile["collection"] = dict(
                collection_mode="calibration",
                rank_partition_metadata=[
                    json.loads(path.read_text())
                    for path in sorted(args.output.glob("partition*/result.rank*.json"))
                ],
            )
            profile["cost_profile_hash"] = profile_fingerprint(profile)
            write_cost_profile(args.output / "cost_profile.json", profile)
            write_json(args.output / "observations.json", observed)
            write_json(
                args.output / "fit_diagnostics.json",
                dict(
                    design=diagnostics,
                    fit=profile["fit"],
                    partitions=partitions,
                    warmups=args.warmups,
                    iterations=args.iterations,
                    num_microbatches=args.num_microbatches,
                ),
            )
    except Exception:
        # Collective teardown can deadlock if a peer is still in model code.
        # Report the original failure and let torchrun terminate all peers.
        traceback.print_exc()
        os._exit(1)
    else:
        shutdown_slackpipe_runtime()
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
