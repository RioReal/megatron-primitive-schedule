# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Torchrun adapter sharing the established Nemotron full-step collector."""

import json
import os
import traceback
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.slackpipe.cost_profile import (
    aggregate_stage_costs,
    build_cost_profile,
    build_heterogeneous_cost_profile,
    profile_fingerprint,
    stage_role,
    write_cost_profile,
)
from megatron.core.pipeline_parallel.slackpipe.manifest import (
    build_model_manifest,
    validate_plan_model,
)
from megatron.core.pipeline_parallel.slackpipe.plan import (
    derive_pipeline_model_parallel_layout,
    load_slackpipe_plan,
)
from megatron.core.pipeline_parallel.slackpipe.schedule import shutdown_slackpipe_runtime
from tools.slackpipe_eval_config import (
    fingerprint,
    load_model,
    parameter_breakdown,
    schedule_topology,
    transformer_config,
)
from tools.slackpipe_hybrid import write_json
from tools.slackpipe_nemotron_worker import (
    calibration_partitions,
    construction_plan,
    run_partition,
    uniform_cuts,
)


def validate_plan_provenance(plan_path: Path, profile_path: Path, profile: dict) -> None:
    """v1 exporter predates embedded hashes; bind its unchanged bytes with a sidecar."""
    from tools.run_slackpipe_nemotron_h8b_pp4 import digest

    raw = json.loads(plan_path.read_text())
    referenced = Path(raw.get("cost_model", {}).get("path", ""))
    if not referenced.is_absolute():
        referenced = plan_path.parent / referenced
    if referenced.resolve() != profile_path.resolve():
        raise ValueError("Plan references a different calibrated profile")
    if raw["schema_version"] == "slackpipe.plan.v2":
        if raw.get("cost_profile_hash") != profile["cost_profile_hash"]:
            raise ValueError("Plan profile provenance mismatch")
    else:
        binding = json.loads(plan_path.with_suffix(".binding.json").read_text())
        if binding != dict(
            schema_version="slackpipe.eval_plan_binding.v1",
            plan_sha256=digest(plan_path),
            profile_sha256=digest(profile_path),
        ):
            raise ValueError("v1 plan/profile binding mismatch")


def validate_profile(
    profile: dict,
    model: dict,
    topology: dict,
    precision: str,
    seq_length: int,
    micro_batch_size: int,
) -> None:
    expected = dict(
        model_config_hash=fingerprint(model),
        dtype=precision,
        sequence_length=seq_length,
        micro_batch_size=micro_batch_size,
    )
    if any(profile["model_config"].get(k) != v for k, v in expected.items()):
        raise ValueError("Calibration model hash/precision/sequence/microbatch mismatch")
    if profile.get("cost_profile_hash") != profile_fingerprint(profile):
        raise ValueError("Calibration profile fingerprint mismatch")
    if profile["parallel_config"] != {k: topology[k] for k in ("pp", "vpp", "tp", "dp", "cp")}:
        raise ValueError("Calibration topology mismatch")
    if model["model_family"] == "nemotron_h":
        manifest = build_model_manifest(transformer_config(model, topology, precision))
        if profile.get("model_manifest_hash") != manifest["manifest_hash"]:
            raise ValueError("Calibration manifest mismatch")


def run_worker(args) -> None:
    model = load_model(args.model_config)
    topology = schedule_topology(args.schedule, args.pp, args.logical_stages, args.microbatches)
    args.stages, args.vpp, args.num_microbatches = (
        topology["num_stages"],
        topology["vpp"],
        args.microbatches,
    )
    args.mode = (
        "slackpipe"
        if args.schedule == "slackpipe"
        else ("partition" if args.schedule == "optimized_interleaved" else "baseline")
    )
    args.vocab_size, args.hidden_size = model["vocab_size"], model["hidden_size"]
    args.exact_parameter_count = parameter_breakdown(model)["exact_parameter_count"]
    args.tiny, args.verify_update = False, args.action == "smoke"
    config = transformer_config(model, topology, args.precision)
    manifest = build_model_manifest(config)
    if args.mode != "baseline":
        if not args.profile or not args.plan:
            raise ValueError("Optimized execution requires the calibrated profile and solver plan")
        profile = json.loads(args.profile.read_text())
        validate_profile(
            profile, model, topology, args.precision, args.seq_length, args.micro_batch_size
        )
        plan = load_slackpipe_plan(args.plan, pipeline_model_parallel_size=args.pp)
        validate_plan_model(plan, config)
        if (plan.num_stages, plan.num_microbatches) != (args.stages, args.microbatches):
            raise ValueError("Plan B/N mismatch")
        validate_plan_provenance(args.plan, args.profile, profile)
    if args.action == "calibrate" and args.mode != "baseline":
        raise ValueError("Calibration must use native Megatron")
    if args.output.exists():
        raise ValueError("Refusing to overwrite worker output")
    if int(os.environ["WORLD_SIZE"]) != args.pp:
        raise ValueError("Worker world size != PP; TP/DP/CP are unsupported")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    dist.all_reduce(torch.zeros(1, device="cuda"))
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=args.pp,
        virtual_pipeline_model_parallel_size=args.vpp,
    )
    rank = dist.get_rank()
    hybrid = model["model_family"] == "nemotron_h"
    partitions = [uniform_cuts(model["num_layers"], args.stages)]
    if args.action == "calibrate" and hybrid:
        partitions, _ = calibration_partitions(manifest, args.stages)
    observations, raw = [], []
    for i, cuts in enumerate(partitions):
        if args.mode == "baseline":
            plan = construction_plan(cuts, args.pp, args.microbatches, manifest)
        args.layout = None if hybrid else derive_pipeline_model_parallel_layout(plan)
        config = transformer_config(model, topology, args.precision, args.layout)
        args.eval_metadata = dict(
            **topology,
            schedule=args.schedule,
            model=model["name"],
            model_config_hash=fingerprint(model),
            model_sha256=fingerprint(model),
            exact_parameter_count=args.exact_parameter_count,
            stage_layer_ranges=plan.stage_layer_ranges,
            precision=args.precision,
            global_batch_size=args.microbatches * args.micro_batch_size,
            transport=args.transport if args.mode == "slackpipe" else "megatron-p2p",
            initialization="partition-independent per-logical-parameter seed",
            optimizer="unwrapped SGD; no momentum/master weights",
        )
        output = args.output / f"partition{i}" if args.action == "calibrate" else args.output
        events = run_partition(
            args,
            config,
            manifest,
            plan,
            output,
            "benchmark" if args.action == "smoke" else args.action,
        )
        if rank == 0 and args.action == "calibrate":
            raw.extend(events)
            rows = aggregate_stage_costs(
                events,
                layer_split=plan.layer_split,
                num_microbatches=args.microbatches,
                iteration_start=args.warmups,
                iteration_end=args.warmups + args.iterations - 1,
            )
            for s, row in enumerate(rows):
                row.update(
                    partition=i,
                    stage_layer_range=list(plan.stage_layer_ranges[s]),
                    stage_role=stage_role(s, args.stages),
                )
            observations.extend(rows)
    if rank == 0:
        write_json(args.output / "model_manifest.json", manifest)
        if args.action == "calibrate":
            common = dict(
                model_config={
                    **manifest["model_config"],
                    "model_config_hash": fingerprint(model),
                    "dtype": args.precision,
                    "sequence_length": args.seq_length,
                    "micro_batch_size": args.micro_batch_size,
                    "vocab_size": args.vocab_size,
                },
                parallel_config={k: topology[k] for k in ("pp", "vpp", "tp", "dp", "cp")},
            )
            if hybrid:
                profile = build_heterogeneous_cost_profile(
                    model_manifest=manifest, observed_stage_rows=observations, **common
                )
            else:
                profile = build_cost_profile(
                    events=raw,
                    layer_split=plan.layer_split,
                    num_microbatches=args.microbatches,
                    iteration_start=args.warmups,
                    iteration_end=args.warmups + args.iterations - 1,
                    **common,
                )
            profile["collection"] = dict(
                collection_mode="calibration",
                rank_partition_metadata=[
                    json.loads(p.read_text())
                    for p in sorted(args.output.glob("partition*/result.rank*.json"))
                ],
            )
            profile["cost_profile_hash"] = profile_fingerprint(profile)
            write_cost_profile(args.output / "cost_profile.json", profile)
            write_json(args.output / "observations.json", observations)
            write_json(args.output / "calibration_events.json", raw)
    shutdown_slackpipe_runtime()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def main() -> None:
    from tools.run_slackpipe_eval import argument_parser, normalize_args

    parser = argument_parser()
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--run-id", required=True)
    args = normalize_args(parser.parse_args())
    args.action = args.stage
    args.memory_history_entries = 100000
    try:
        run_worker(args)
    except Exception:
        traceback.print_exc()
        os._exit(1)


if __name__ == "__main__":
    main()
