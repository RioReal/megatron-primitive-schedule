# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""SlackPipe schedule prototype."""

import contextlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_model_config, nvtx_range_pop, nvtx_range_push

from .communication import SlackPipeCommunicator
from .plan import (
    SLACKPIPE_PLAN_SCHEMA_VERSION,
    SlackPipeOperation,
    load_slackpipe_plan,
    validate_cyclic_placement,
)

_RUNTIME_CACHE: Dict[tuple, "_SlackPipeRuntime"] = {}
_RMA_TRANSPORT_CACHE: Dict[tuple, object] = {}


def _distributed_context():
    if not torch.distributed.is_initialized():
        return None
    return (torch.distributed.group.WORLD, parallel_state.get_pipeline_model_parallel_group())


def _get_rma_transport(plan, shape, dtype, device):
    from .communication_rma import SlackPipeRMACommunicator

    context = _distributed_context()
    if any(key[0] != context for key in _RMA_TRANSPORT_CACHE):
        raise RuntimeError("Call shutdown_slackpipe_runtime before changing distributed groups")
    edges = tuple(
        (s, s + 1, plan.stage_to_worker[s], plan.stage_to_worker[s + 1])
        for s in range(plan.num_stages - 1)
        if plan.stage_to_worker[s] != plan.stage_to_worker[s + 1]
    )
    key = (
        context,
        torch.distributed.get_world_size(),
        plan.num_workers,
        torch.distributed.get_rank(),
        edges,
        tuple(shape),
        dtype,
        plan.num_microbatches,
        "nccl-rma",
        device.type,
        torch.cuda.current_device() if device.index is None else device.index,
    )
    if key not in _RMA_TRANSPORT_CACHE:
        _RMA_TRANSPORT_CACHE[key] = SlackPipeRMACommunicator(plan)
    return _RMA_TRANSPORT_CACHE[key]


def forward_backward_slackpipe(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,  # unused
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,
    p2p_communicator: Optional[P2PCommunicator] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
    force_all_reduce: Optional[bool] = False,
    slackpipe_plan_path: Optional[str] = None,
    slackpipe_trace_path: Optional[str] = None,
    slackpipe_profile_path: Optional[str] = None,
    slackpipe_runtime: str = "debug",
    slackpipe_enable_nvtx: bool = True,
    slackpipe_transport: str = "nccl-p2p",
):
    """Run a validated SlackPipe plan on the local pipeline worker.

    Stage tensors are keyed by (microbatch_id, stage_id) so the JSON plan can
    choose any dependency-valid operation order.
    """

    del decoder_seq_length, p2p_communicator, force_all_reduce

    if slackpipe_plan_path is None:
        raise ValueError("--slackpipe-plan is required when --pipeline-schedule=slackpipe")
    if adjust_tensor_shapes_fn is not None:
        raise ValueError("SlackPipe prototype does not support adjust_tensor_shapes_fn")
    if slackpipe_runtime not in ("debug", "fast"):
        raise ValueError(f"Unknown SlackPipe runtime: {slackpipe_runtime}")
    if slackpipe_transport not in ("nccl-p2p", "nccl-rma"):
        raise ValueError(f"Unknown SlackPipe transport: {slackpipe_transport}")

    _validate_supported_parallelism()

    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    model_chunks = model if isinstance(model, list) else [model]
    data_iterators = data_iterator if isinstance(data_iterator, list) else [data_iterator]
    config = get_model_config(model_chunks[0])
    _validate_unsupported_features(config, forward_only)

    cp_group_size = _get_cp_group_size(pg_collection)
    pipeline_tensor_shape = (seq_length, micro_batch_size, config.hidden_size)
    pipeline_tensor_dtype = config.pipeline_dtype or config.params_dtype
    pipeline_tensor_device = torch.device("cuda")
    runtime = _get_slackpipe_runtime(
        slackpipe_plan_path,
        pp_rank=pp_rank,
        pipeline_tensor_shape=pipeline_tensor_shape,
        pipeline_tensor_dtype=pipeline_tensor_dtype,
        pipeline_tensor_device=pipeline_tensor_device,
        num_microbatches=num_microbatches,
        forward_only=forward_only,
        enable_fast_path=slackpipe_runtime == "fast",
        transport=slackpipe_transport,
    )
    plan = runtime.plan
    if (
        plan.model_manifest_hash or plan.cost_profile_hash
        or plan.schema_version == "slackpipe.plan.v2"
        or getattr(config, "slackpipe_hybrid_pattern", None)
    ):
        from .manifest import validate_chunk_layers, validate_plan_model

        for chunk in model_chunks:
            validate_plan_model(plan, get_model_config(chunk))
        validate_chunk_layers(plan, model_chunks, pp_rank)
    if len(model_chunks) != len(runtime.local_stages):
        raise ValueError(
            "SlackPipe prototype requires one local model chunk per local logical stage "
            f"({len(model_chunks)} != {len(runtime.local_stages)})"
        )
    if len(data_iterators) != len(model_chunks):
        raise ValueError(
            "SlackPipe prototype requires one data iterator per local model chunk "
            f"({len(data_iterators)} != {len(model_chunks)})"
        )

    forward_data_store = []
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    input_tensors: Dict[tuple[int, int], object] = {}
    output_tensors: Dict[tuple[int, int], object] = {}
    forwarded_output_tensors: Dict[tuple[int, int], object] = {}
    input_tensor_grads: Dict[tuple[int, int], object] = {}

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    executed_operations: List[SlackPipeOperation] = []
    profile = _SlackPipeProfile(slackpipe_profile_path, pp_rank)
    profile.start()
    runtime.prepare_iteration()
    no_sync_func = config.no_sync_func or contextlib.nullcontext
    with no_sync_func():
        for compiled_op in runtime.compiled_operations:
            op = compiled_op.operation
            if slackpipe_enable_nvtx:
                nvtx_range_push(compiled_op.nvtx_name)
            profile.start_operation(op)
            try:
                executed_operations.append(op)
                key = compiled_op.key
                if compiled_op.is_forward:
                    from megatron.core.pipeline_parallel.schedules import (
                        check_first_val_step,
                        forward_step,
                    )

                    if op.stage == 0:
                        input_tensor = None
                    else:
                        if not compiled_op.input_is_remote:
                            if compiled_op.prev_key not in forwarded_output_tensors:
                                raise ValueError(
                                    f"SlackPipe plan runs F({op.microbatch},{op.stage}) before "
                                    f"F({op.microbatch},{op.stage - 1}) produced its input"
                                )
                            input_tensor = forwarded_output_tensors[compiled_op.prev_key]
                        else:
                            assert runtime.communicator is not None
                            wait_started = time.perf_counter()
                            profile.start_current_recv_wait()
                            input_tensor = runtime.communicator.recv_forward(
                                op.stage,
                                runtime.pipeline_tensor_shape,
                                runtime.pipeline_tensor_dtype,
                                runtime.pipeline_tensor_device,
                                requires_grad=not forward_only,
                                microbatch=op.microbatch,
                            )
                            profile.add_current_recv_wait(time.perf_counter() - wait_started)

                    input_tensors[key] = input_tensor
                    output_tensor, num_tokens = forward_step(
                        forward_step_func,
                        data_iterators[compiled_op.local_index],
                        model_chunks[compiled_op.local_index],
                        num_microbatches,
                        input_tensor,
                        forward_data_store,
                        config,
                        cp_group_size=cp_group_size,
                        collect_non_loss_data=collect_non_loss_data,
                        is_first_microbatch=check_first_val_step(
                            first_val_step, forward_only, op.microbatch == 0
                        ),
                        current_microbatch=op.microbatch,
                        vp_stage=getattr(model_chunks[compiled_op.local_index], "vp_stage", None),
                        is_last_stage=op.stage == plan.num_stages - 1,
                    )
                    output_tensors[key] = output_tensor
                    if op.stage != plan.num_stages - 1:
                        detached_output = _detach_pipeline_tensor(output_tensor)
                        if not compiled_op.output_is_remote:
                            forwarded_output_tensors[key] = detached_output
                        else:
                            assert runtime.communicator is not None
                            runtime.communicator.send_forward(
                                op.stage, detached_output, microbatch=op.microbatch
                            )
                    total_num_tokens += num_tokens
                elif forward_only:
                    continue
                else:
                    from megatron.core.pipeline_parallel.schedules import backward_step

                    if key not in output_tensors:
                        raise ValueError(
                            f"SlackPipe plan runs B({op.microbatch},{op.stage}) before its forward pass"
                        )
                    if op.stage == plan.num_stages - 1:
                        output_tensor_grad = None
                    else:
                        if not compiled_op.grad_input_is_remote:
                            if compiled_op.next_key not in input_tensor_grads:
                                raise ValueError(
                                    f"SlackPipe plan runs B({op.microbatch},{op.stage}) before "
                                    f"B({op.microbatch},{op.stage + 1}) produced its gradient"
                                )
                            output_tensor_grad = input_tensor_grads.pop(compiled_op.next_key)
                        else:
                            assert runtime.communicator is not None
                            wait_started = time.perf_counter()
                            profile.start_current_recv_wait()
                            output_tensor_grad = runtime.communicator.recv_backward(
                                op.stage,
                                runtime.pipeline_tensor_shape,
                                runtime.pipeline_tensor_dtype,
                                runtime.pipeline_tensor_device,
                                microbatch=op.microbatch,
                            )
                            profile.add_current_recv_wait(time.perf_counter() - wait_started)

                    input_tensor_grad = backward_step(
                        input_tensors[key], output_tensors[key], output_tensor_grad, config
                    )
                    if op.stage != 0:
                        if not compiled_op.grad_output_is_remote:
                            input_tensor_grads[key] = input_tensor_grad
                        else:
                            assert runtime.communicator is not None
                            runtime.communicator.send_backward(
                                op.stage, input_tensor_grad, microbatch=op.microbatch
                            )
                    del input_tensors[key]
                    del output_tensors[key]
                    forwarded_output_tensors.pop(key, None)
            finally:
                profile.end_operation()
                if slackpipe_enable_nvtx:
                    nvtx_range_pop()

    expected_operations = runtime.expected_operations
    if tuple(executed_operations) != expected_operations:
        raise RuntimeError(
            "SlackPipe runtime operation trace does not match the plan for rank " f"{pp_rank}"
        )
    _write_trace(slackpipe_trace_path, pp_rank, expected_operations, executed_operations)

    if runtime.communicator is not None:
        runtime.communicator.drain_sends()
        runtime.communicator.finalize_iteration()
        runtime.communicator.assert_no_outstanding_work()

    if not forward_only:
        _assert_no_tensor_state(
            input_tensors, output_tensors, forwarded_output_tensors, input_tensor_grads
        )

    if config.finalize_model_grads_func is not None and not forward_only:
        config.finalize_model_grads_func(
            model_chunks,
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    profile.finish(plan, expected_operations, executed_operations)

    return forward_data_store


@dataclass(frozen=True)
class _CompiledOperation:
    operation: SlackPipeOperation
    key: tuple[int, int]
    prev_key: Optional[tuple[int, int]]
    next_key: Optional[tuple[int, int]]
    local_index: int
    previous_worker: Optional[int]
    next_worker: Optional[int]
    previous_edge: Optional[tuple[int, int]]
    next_edge: Optional[tuple[int, int]]
    is_forward: bool
    input_is_remote: bool
    output_is_remote: bool
    grad_input_is_remote: bool
    grad_output_is_remote: bool
    nvtx_name: str


class _SlackPipeRuntime:
    def __init__(
        self,
        *,
        plan,
        pp_rank: int,
        pipeline_tensor_shape: Tuple[int, int, int],
        pipeline_tensor_dtype: torch.dtype,
        pipeline_tensor_device: torch.device,
        forward_only: bool,
        enable_fast_path: bool,
        transport: str = "nccl-p2p",
    ):
        self.plan = plan
        self.pp_rank = pp_rank
        self.pipeline_tensor_shape = pipeline_tensor_shape
        self.pipeline_tensor_dtype = pipeline_tensor_dtype
        self.pipeline_tensor_device = pipeline_tensor_device
        self.forward_only = forward_only
        self.enable_fast_path = enable_fast_path
        self.local_stages = tuple(
            stage for stage, worker in enumerate(plan.stage_to_worker) if worker == pp_rank
        )
        self.stage_to_local_index = {stage: index for index, stage in enumerate(self.local_stages)}
        self.expected_operations = plan.worker_operations(pp_rank)
        self.compiled_operations = tuple(self._compile(op) for op in self.expected_operations)
        self.receive_specs, self.send_specs = self._compile_message_specs()
        self.persistent_transport = transport == "nccl-rma"
        self.communicator = None
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            self.communicator = (
                _get_rma_transport(
                    plan, pipeline_tensor_shape, pipeline_tensor_dtype, pipeline_tensor_device
                )
                if self.persistent_transport
                else SlackPipeCommunicator(plan)
            )

    def close(self) -> None:
        if self.communicator is not None and not self.persistent_transport:
            self.communicator.close()

    def prepare_iteration(self) -> None:
        if self.communicator is None:
            return
        self.communicator.prepare_iteration(
            receive_specs=self.receive_specs,
            send_specs=self.send_specs,
            shape=self.pipeline_tensor_shape,
            dtype=self.pipeline_tensor_dtype,
            device=self.pipeline_tensor_device,
        )

    def _compile(self, op: SlackPipeOperation) -> _CompiledOperation:
        stage = op.stage
        microbatch = op.microbatch
        is_forward = op.kind == "F"
        prev_key = (microbatch, stage - 1) if stage > 0 else None
        next_key = (microbatch, stage + 1) if stage + 1 < self.plan.num_stages else None
        previous_is_remote = stage > 0 and self.plan.stage_to_worker[stage - 1] != self.pp_rank
        next_is_remote = (
            stage + 1 < self.plan.num_stages
            and self.plan.stage_to_worker[stage + 1] != self.pp_rank
        )
        return _CompiledOperation(
            operation=op,
            key=(microbatch, stage),
            prev_key=prev_key,
            next_key=next_key,
            local_index=self.stage_to_local_index[stage],
            previous_worker=self.plan.stage_to_worker[stage - 1] if prev_key else None,
            next_worker=self.plan.stage_to_worker[stage + 1] if next_key else None,
            previous_edge=(stage - 1, stage) if prev_key else None,
            next_edge=(stage, stage + 1) if next_key else None,
            is_forward=is_forward,
            input_is_remote=previous_is_remote,
            output_is_remote=next_is_remote,
            grad_input_is_remote=next_is_remote,
            grad_output_is_remote=previous_is_remote,
            nvtx_name=f"SlackPipe/{op.kind}/b{microbatch}/s{stage}",
        )

    def _compile_message_specs(self):
        receive_specs = []
        send_specs = []
        for compiled_op in self.compiled_operations:
            op = compiled_op.operation
            if compiled_op.is_forward:
                if compiled_op.input_is_remote:
                    receive_specs.append(("forward", (op.stage - 1, op.stage), op.microbatch))
                if compiled_op.output_is_remote:
                    send_specs.append(("forward", (op.stage, op.stage + 1), op.microbatch))
            elif not self.forward_only:
                if compiled_op.grad_input_is_remote:
                    receive_specs.append(("backward", (op.stage, op.stage + 1), op.microbatch))
                if compiled_op.grad_output_is_remote:
                    send_specs.append(("backward", (op.stage - 1, op.stage), op.microbatch))
        return tuple(receive_specs), tuple(send_specs)


def _get_slackpipe_runtime(
    plan_path: str,
    *,
    pp_rank: int,
    pipeline_tensor_shape: Tuple[int, int, int],
    pipeline_tensor_dtype: torch.dtype,
    pipeline_tensor_device: torch.device,
    num_microbatches: int,
    forward_only: bool,
    enable_fast_path: bool,
    transport: str = "nccl-p2p",
) -> _SlackPipeRuntime:
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    normalized_plan_path = str(Path(plan_path).expanduser().resolve())
    plan_stat = Path(normalized_plan_path).stat()
    cache_key = (
        normalized_plan_path,
        (plan_stat.st_mtime_ns, plan_stat.st_size),
        _distributed_context(),
        num_microbatches,
        pipeline_model_parallel_size,
        pp_rank,
        pipeline_tensor_shape,
        str(pipeline_tensor_dtype),
        str(pipeline_tensor_device),
        torch.cuda.current_device() if pipeline_tensor_device.type == "cuda" else None,
        forward_only,
        enable_fast_path,
        transport,
    )
    runtime = _RUNTIME_CACHE.get(cache_key)
    if runtime is not None:
        return runtime

    plan = load_slackpipe_plan(
        normalized_plan_path, pipeline_model_parallel_size=pipeline_model_parallel_size
    )
    validate_cyclic_placement(plan)
    if plan.num_microbatches != num_microbatches:
        raise ValueError(
            f"SlackPipe plan num_microbatches must match runtime num_microbatches "
            f"({plan.num_microbatches} != {num_microbatches})"
        )
    runtime = _SlackPipeRuntime(
        plan=plan,
        pp_rank=pp_rank,
        pipeline_tensor_shape=pipeline_tensor_shape,
        pipeline_tensor_dtype=pipeline_tensor_dtype,
        pipeline_tensor_device=pipeline_tensor_device,
        forward_only=forward_only,
        enable_fast_path=enable_fast_path,
        transport=transport,
    )
    _RUNTIME_CACHE[cache_key] = runtime
    return runtime


def clear_slackpipe_plan_cache() -> None:
    """Release compiled plans; keep compatible RMA resources until shutdown."""
    for communicator in _RMA_TRANSPORT_CACHE.values():
        communicator.assert_no_outstanding_work()
    for runtime in list(_RUNTIME_CACHE.values()):
        runtime.close()
    _RUNTIME_CACHE.clear()


def shutdown_slackpipe_runtime() -> None:
    """Collectively release SlackPipe resources before destroying parent groups.

    All ranks must call this while idle, in the same configuration/order.
    Active iterations are rejected rather than silently discarding GPU work.
    """
    clear_slackpipe_plan_cache()
    for key in list(_RMA_TRANSPORT_CACHE):
        _RMA_TRANSPORT_CACHE[key].close()
        del _RMA_TRANSPORT_CACHE[key]
    assert not _RUNTIME_CACHE and not _RMA_TRANSPORT_CACHE


def clear_slackpipe_runtime_cache() -> None:
    """Backward-compatible collective full shutdown."""
    shutdown_slackpipe_runtime()


def slackpipe_transport_statistics() -> list[dict]:
    """Return local cached RMA allocation statistics for benchmark reports."""
    return [communicator.statistics() for communicator in _RMA_TRANSPORT_CACHE.values()]


class _SlackPipeProfile:
    def __init__(self, profile_path: Optional[str], pp_rank: int):
        self.profile_path = profile_path
        self.pp_rank = pp_rank
        self.enabled = profile_path is not None
        self.step_start_event = None
        self.step_end_event = None
        self.step_start_cpu = 0.0
        self.step_end_cpu = 0.0
        self.operations = []
        self.current = None

    def start(self) -> None:
        if not self.enabled:
            return
        torch.cuda.reset_peak_memory_stats()
        self.step_start_cpu = time.perf_counter()
        self.step_start_event = torch.cuda.Event(enable_timing=True)
        self.step_end_event = torch.cuda.Event(enable_timing=True)
        self.step_start_event.record()

    def start_operation(self, op: SlackPipeOperation) -> None:
        if not self.enabled:
            return
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        self.current = {
            "kind": op.kind,
            "microbatch": op.microbatch,
            "stage": op.stage,
            "rank": self.pp_rank,
            "start_event": start_event,
            "end_event": end_event,
            "start_time_seconds": time.perf_counter(),
            "end_time_seconds": None,
            "recv_wait_seconds": 0.0,
        }

    def start_current_recv_wait(self) -> None:
        if self.enabled and self.current is not None:
            self.current["recv_start"] = torch.cuda.Event(enable_timing=True)
            self.current["recv_end"] = torch.cuda.Event(enable_timing=True)
            self.current["recv_start"].record()

    def add_current_recv_wait(self, seconds: float) -> None:
        if self.enabled and self.current is not None:
            self.current["recv_wait_seconds"] += seconds
            self.current["recv_end"].record()

    def end_operation(self) -> None:
        if not self.enabled or self.current is None:
            return
        self.current["end_time_seconds"] = time.perf_counter()
        self.current["end_event"].record()
        self.operations.append(self.current)
        self.current = None

    def finish(
        self,
        plan,
        expected_operations: Tuple[SlackPipeOperation, ...],
        executed_operations: List[SlackPipeOperation],
    ) -> None:
        if not self.enabled:
            return
        self.step_end_cpu = time.perf_counter()
        self.step_end_event.record()
        torch.cuda.synchronize()

        op_records = []
        for op in self.operations:
            elapsed_ms = op["start_event"].elapsed_time(op["end_event"])
            record = {
                "kind": op["kind"],
                "microbatch": op["microbatch"],
                "stage": op["stage"],
                "rank": op["rank"],
                "start_time_seconds": op["start_time_seconds"] - self.step_start_cpu,
                "end_time_seconds": op["end_time_seconds"] - self.step_start_cpu,
                "duration_ms": elapsed_ms,
                "recv_wait_ms": op["recv_wait_seconds"] * 1000.0,
                "recv_gpu_wait_ms": (
                    op["recv_start"].elapsed_time(op["recv_end"]) if "recv_start" in op else 0.0
                ),
            }
            if op["kind"] == "F":
                record["forward_duration_ms"] = elapsed_ms
                record["backward_duration_ms"] = 0.0
            else:
                record["forward_duration_ms"] = 0.0
                record["backward_duration_ms"] = elapsed_ms
            op_records.append(record)

        profile = {
            "schema_version": SLACKPIPE_PLAN_SCHEMA_VERSION,
            "rank": self.pp_rank,
            "matched_plan": tuple(executed_operations) == expected_operations,
            "num_microbatches": plan.num_microbatches,
            "num_stages": plan.num_stages,
            "num_workers": plan.num_workers,
            "layer_split": list(plan.layer_split),
            "stage_to_worker": list(plan.stage_to_worker),
            "solver_status": plan.solver_status,
            "predicted_makespan": plan.predicted_makespan,
            "step_cpu_wall_time_ms": (self.step_end_cpu - self.step_start_cpu) * 1000.0,
            "step_cuda_time_ms": self.step_start_event.elapsed_time(self.step_end_event),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "operations": op_records,
        }
        path = _rank_trace_path(self.profile_path, self.pp_rank)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as profile_file:
            json.dump(profile, profile_file, indent=2, sort_keys=True)


def _validate_supported_parallelism() -> None:
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    cp_size = parallel_state.get_context_parallel_world_size()
    dp_size = parallel_state.get_data_parallel_world_size(with_context_parallel=True)
    if pp_size < 1:
        raise ValueError(
            f"SlackPipe requires positive pipeline model parallel size, got {pp_size}"
        )
    if tp_size != 1:
        raise ValueError(
            f"SlackPipe prototype requires tensor model parallel size 1, got {tp_size}"
        )
    if cp_size != 1:
        raise ValueError(f"SlackPipe prototype requires context parallel size 1, got {cp_size}")
    if dp_size != 1:
        raise ValueError(f"SlackPipe prototype does not support data parallelism, got {dp_size}")


def _assert_no_tensor_state(*stores: Dict[tuple[int, int], object]) -> None:
    leaked = [store for store in stores if store]
    if leaked:
        sizes = [len(store) for store in leaked]
        raise RuntimeError(f"SlackPipe schedule retained tensor state after completion: {sizes}")


def _write_trace(
    trace_path: Optional[str],
    pp_rank: int,
    expected_operations: Tuple[SlackPipeOperation, ...],
    executed_operations: List[SlackPipeOperation],
) -> None:
    if trace_path is None:
        return

    path = _rank_trace_path(trace_path, pp_rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as trace_file:
        json.dump(
            {
                "schema_version": SLACKPIPE_PLAN_SCHEMA_VERSION,
                "rank": pp_rank,
                "matched_plan": tuple(executed_operations) == expected_operations,
                "operations": [
                    {"kind": op.kind, "microbatch": op.microbatch, "stage": op.stage}
                    for op in executed_operations
                ],
            },
            trace_file,
            indent=2,
            sort_keys=True,
        )


def _rank_trace_path(trace_path: str, pp_rank: int) -> Path:
    path = Path(trace_path)
    if path.suffix:
        return path.with_name(f"{path.stem}.rank{pp_rank}{path.suffix}")
    return path / f"rank{pp_rank}.json"


def _validate_unsupported_features(config, forward_only: bool) -> None:
    if config.params_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("SlackPipe supports FP32 and BF16 only")
    if config.pipeline_dtype not in (None, config.params_dtype):
        raise ValueError("SlackPipe pipeline dtype must match model parameter dtype")
    if getattr(config, "recompute_granularity", None) is not None:
        raise ValueError("SlackPipe does not support activation recomputation")
    if getattr(config, "num_moe_experts", None) or getattr(config, "mtp_num_layers", None):
        raise ValueError("SlackPipe does not support MoE or MTP")
    if getattr(config, "num_microbatches_with_partial_activation_checkpoints", None) is not None:
        raise ValueError("SlackPipe prototype does not support activation checkpointing")
    if (
        getattr(config, "overlap_p2p_comm", False)
        or getattr(config, "batch_p2p_comm", False) is False
    ):
        raise ValueError("SlackPipe prototype does not support communication overlap")
    if getattr(config, "tp_comm_overlap", False):
        raise ValueError(
            "SlackPipe prototype does not support tensor-parallel communication overlap"
        )
    if (
        getattr(config, "enable_cuda_graph", False)
        or getattr(config, "cuda_graph_impl", "none") != "none"
    ):
        raise ValueError("SlackPipe prototype does not support CUDA graphs")
    if getattr(config, "hybrid_context_parallel", False):
        raise ValueError("SlackPipe prototype does not support context parallelism")
    if getattr(config, "overlap_moe_expert_parallel_comm", False) and not forward_only:
        raise ValueError("SlackPipe prototype does not support expert communication overlap")


def _get_cp_group_size(pg_collection: Optional[ProcessGroupCollection]) -> int:
    if pg_collection is not None and hasattr(pg_collection, "cp"):
        return pg_collection.cp.size()
    return parallel_state.get_context_parallel_world_size()


def _detach_pipeline_tensor(tensor):
    """Detach a local logical-stage boundary as P2P communication would."""

    if tensor is None:
        return None
    if isinstance(tensor, list):
        return [_detach_pipeline_tensor(item) for item in tensor]
    if isinstance(tensor, dict):
        return {key: _detach_pipeline_tensor(value) for key, value in tensor.items()}
    assert isinstance(tensor, torch.Tensor), f"expected Tensor, found {type(tensor).__name__}"
    detached_tensor = tensor.detach()
    if tensor.requires_grad:
        detached_tensor.requires_grad_(True)
    return detached_tensor
