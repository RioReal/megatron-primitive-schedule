"""Two-rank, communication-only lifecycle probe; no model or solver required."""

import argparse
import csv
import gc
import json
import os
import resource
import weakref
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.slackpipe.communication_rma import SlackPipeRMACommunicator
from megatron.core.pipeline_parallel.slackpipe.plan import parse_slackpipe_plan
from megatron.core.pipeline_parallel.slackpipe.schedule import (
    _get_slackpipe_runtime,
    clear_slackpipe_plan_cache,
    shutdown_slackpipe_runtime,
    slackpipe_transport_statistics,
)
from tests.unit_tests.pipeline_parallel.slackpipe_test_utils import managed_model_parallel_groups
from tests.unit_tests.test_utilities import Utils


def tiny_plan():
    operations = [[], []]
    for b in range(4):
        for kind, stages in (("F", range(4)), ("B", reversed(range(4)))):
            for s in stages:
                operations[s % 2].append(dict(kind=kind, microbatch=b, stage=s))
    return parse_slackpipe_plan(
        dict(
            num_microbatches=4,
            num_stages=4,
            num_workers=2,
            num_layers=8,
            layer_split=[2] * 4,
            stage_to_worker=[0, 1, 0, 1],
            operations=operations,
        ),
        pipeline_model_parallel_size=2,
    )


def iteration(comm, index):
    shape, dtype, device = (64, 1, 128), torch.float32, torch.device("cuda")
    sends, receives = [], []
    for s in range(3):
        for direction in ("forward", "backward"):
            sender = (s if direction == "forward" else s + 1) % 2
            for b in range(comm.num_microbatches):
                (sends if sender == dist.get_rank() else receives).append(
                    (direction, (s, s + 1), b)
                )
    comm.prepare_iteration(
        receive_specs=receives, send_specs=sends, shape=shape, dtype=dtype, device=device
    )
    for direction, edge, b in sends:
        value = index * 1000 + edge[0] * 100 + (50 if direction == "backward" else 0) + b
        comm._send(direction, edge, torch.full(shape, value, dtype=dtype, device=device), b)
    comm.drain_sends()
    dist.barrier(group=comm.control)
    for direction, edge, b in receives:
        value = index * 1000 + edge[0] * 100 + (50 if direction == "backward" else 0) + b
        actual = comm._recv(direction, edge, b, shape, dtype, device)
        assert torch.equal(actual, torch.full_like(actual, value))
    comm.finalize_iteration()
    comm.assert_no_outstanding_work()


def planned_iteration(runtime, index):
    """Exercise the exact worker operation order, with deterministic edge data."""
    comm = runtime.communicator
    runtime.prepare_iteration()

    def value(direction, edge, b):
        return index * 1000 + edge[0] * 100 + (50 if direction == "backward" else 0) + b

    for op in runtime.expected_operations:
        b, s = op.microbatch, op.stage
        direction = "forward" if op.kind == "F" else "backward"
        incoming = (s - 1, s) if op.kind == "F" else (s, s + 1)
        outgoing = (s, s + 1) if op.kind == "F" else (s - 1, s)
        if 0 <= incoming[0] < 3:
            actual = comm._recv(
                direction,
                incoming,
                b,
                runtime.pipeline_tensor_shape,
                runtime.pipeline_tensor_dtype,
                runtime.pipeline_tensor_device,
            )
            assert torch.equal(actual, torch.full_like(actual, value(direction, incoming, b)))
        if 0 <= outgoing[0] < 3:
            tensor = torch.full(
                runtime.pipeline_tensor_shape,
                value(direction, outgoing, b),
                dtype=runtime.pipeline_tensor_dtype,
                device=runtime.pipeline_tensor_device,
            )
            comm._send(direction, outgoing, tensor, b)
    comm.drain_sends()
    comm.finalize_iteration()
    comm.assert_no_outstanding_work()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=50)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--recreate-megatron", action="store_true")
    parser.add_argument("--managed-groups", action="store_true")
    parser.add_argument("--plans", type=Path, nargs="+")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    dist.all_reduce(torch.zeros(1, device="cuda"))
    plan = tiny_plan()
    rows, references = [], []
    comm = None
    output = args.output.with_name(f"{args.output.stem}.rank{dist.get_rank()}")
    output.parent.mkdir(parents=True, exist_ok=True)

    def record(cycle, phase):
        torch.cuda.synchronize()
        gc.collect()
        live = [ref() for ref in references if ref() is not None]
        row = dict(
            cycle=cycle,
            phase=phase,
            allocated_bytes=torch.cuda.memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved(),
            device_used_bytes=torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0],
            live_contexts=len(live),
            live_mailboxes=sum(len(c.channels) for c in live),
            live_windows=sum(2 * len(c.channels) for c in live),
            registered_process_groups=len(dist.distributed_c10d._world.pg_map),
            rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        rows.append(row)
        Path(str(output) + ".json").write_text(json.dumps(rows, indent=2) + "\n")
        with Path(str(output) + ".csv").open("w") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        if dist.get_rank() == 0:
            print(json.dumps(row), flush=True)

    context = managed_model_parallel_groups() if args.managed_groups else nullcontext()
    with context:
        try:
            record(-1, "baseline")
            if args.plans:
                Utils.initialize_model_parallel(pipeline_model_parallel_size=2)
            for cycle in range(args.cycles):
                if args.recreate_megatron:
                    Utils.initialize_model_parallel(pipeline_model_parallel_size=2)
                    dist.all_reduce(
                        torch.zeros(1, device="cuda"),
                        group=parallel_state.get_tensor_model_parallel_group(),
                    )
                if args.plans:
                    path = args.plans[cycle % len(args.plans)]
                    from megatron.core.pipeline_parallel.slackpipe.plan import load_slackpipe_plan

                    runtime = _get_slackpipe_runtime(
                        str(path),
                        pp_rank=dist.get_rank(),
                        pipeline_tensor_shape=(64, 1, 128),
                        pipeline_tensor_dtype=torch.float32,
                        pipeline_tensor_device=torch.device("cuda"),
                        num_microbatches=load_slackpipe_plan(
                            path, pipeline_model_parallel_size=2
                        ).num_microbatches,
                        forward_only=False,
                        enable_fast_path=True,
                        transport="nccl-rma",
                    )
                    comm = runtime.communicator
                    if not any(ref() is comm for ref in references):
                        references.append(weakref.ref(comm))
                    assert len(slackpipe_transport_statistics()) == 1
                    planned_iteration(runtime, cycle)
                    del runtime
                elif comm is None:
                    comm = SlackPipeRMACommunicator(plan)
                    references.append(weakref.ref(comm))
                if not args.plans:
                    iteration(comm, cycle)
                record(cycle, "idle")
                if args.plans:
                    clear_slackpipe_plan_cache()
                if not args.reuse or cycle == args.cycles - 1:
                    if args.plans:
                        shutdown_slackpipe_runtime()
                        assert not slackpipe_transport_statistics()
                    else:
                        comm.close()
                    comm = None
                    if args.recreate_megatron:
                        Utils.destroy_model_parallel()
                    record(cycle, "closed")
            plateau = [row for row in rows if row["phase"] == ("idle" if args.reuse else "closed")]
            if not args.recreate_megatron or args.managed_groups:
                for metric, tolerance in (
                    ("allocated_bytes", 1024 * 1024),
                    ("device_used_bytes", 16 * 1024 * 1024),
                ):
                    assert (
                        max(row[metric] for row in plateau) - min(row[metric] for row in plateau)
                        <= tolerance
                    ), metric
        finally:
            shutdown_slackpipe_runtime()
            if comm is not None:
                comm.close()
                comm = None
    record(args.cycles, "final")
    assert not any(ref() is not None for ref in references)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
