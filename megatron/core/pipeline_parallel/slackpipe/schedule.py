# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""SlackPipe schedule prototype."""

import contextlib
from typing import Callable, Dict, Iterator, List, Optional, Union

import torch

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_model_config

from .communication import SlackPipeCommunicator
from .plan import load_slackpipe_plan, validate_cyclic_placement


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

    _validate_supported_parallelism()

    plan = load_slackpipe_plan(
        slackpipe_plan_path,
        pipeline_model_parallel_size=parallel_state.get_pipeline_model_parallel_world_size(),
    )
    validate_cyclic_placement(plan)
    if plan.num_microbatches != num_microbatches:
        raise ValueError(
            f"SlackPipe plan num_microbatches must match runtime num_microbatches "
            f"({plan.num_microbatches} != {num_microbatches})"
        )

    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    local_stages = [
        stage for stage, worker in enumerate(plan.stage_to_worker) if worker == pp_rank
    ]
    stage_to_local_index = {stage: index for index, stage in enumerate(local_stages)}
    model_chunks = model if isinstance(model, list) else [model]
    data_iterators = data_iterator if isinstance(data_iterator, list) else [data_iterator]
    if len(model_chunks) != len(local_stages):
        raise ValueError(
            "SlackPipe prototype requires one local model chunk per local logical stage "
            f"({len(model_chunks)} != {len(local_stages)})"
        )
    if len(data_iterators) != len(model_chunks):
        raise ValueError(
            "SlackPipe prototype requires one data iterator per local model chunk "
            f"({len(data_iterators)} != {len(model_chunks)})"
        )

    config = get_model_config(model_chunks[0])
    _validate_unsupported_features(config, forward_only)

    cp_group_size = _get_cp_group_size(pg_collection)
    communicator = (
        SlackPipeCommunicator(plan)
        if parallel_state.get_pipeline_model_parallel_world_size() > 1
        else None
    )
    pipeline_tensor_shape = (seq_length, micro_batch_size, config.hidden_size)
    pipeline_tensor_dtype = config.pipeline_dtype or torch.float32
    pipeline_tensor_device = torch.device("cuda")
    forward_data_store = []
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    input_tensors: Dict[tuple[int, int], object] = {}
    output_tensors: Dict[tuple[int, int], object] = {}
    forwarded_output_tensors: Dict[tuple[int, int], object] = {}
    input_tensor_grads: Dict[tuple[int, int], object] = {}

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    no_sync_func = config.no_sync_func or contextlib.nullcontext
    with no_sync_func():
        for op in plan.worker_operations(pp_rank):
            key = (op.microbatch, op.stage)
            if op.kind == "F":
                from megatron.core.pipeline_parallel.schedules import (
                    check_first_val_step,
                    forward_step,
                )

                if op.stage == 0:
                    input_tensor = None
                else:
                    prev_key = (op.microbatch, op.stage - 1)
                    if plan.stage_to_worker[op.stage - 1] == pp_rank:
                        if prev_key not in forwarded_output_tensors:
                            raise ValueError(
                                f"SlackPipe plan runs F({op.microbatch},{op.stage}) before "
                                f"F({op.microbatch},{op.stage - 1}) produced its input"
                            )
                        input_tensor = forwarded_output_tensors[prev_key]
                    else:
                        assert communicator is not None
                        input_tensor = communicator.recv_forward(
                            op.stage,
                            pipeline_tensor_shape,
                            pipeline_tensor_dtype,
                            pipeline_tensor_device,
                            requires_grad=not forward_only,
                        )

                input_tensors[key] = input_tensor
                local_index = stage_to_local_index[op.stage]
                output_tensor, num_tokens = forward_step(
                    forward_step_func,
                    data_iterators[local_index],
                    model_chunks[local_index],
                    num_microbatches,
                    input_tensor,
                    forward_data_store,
                    config,
                    cp_group_size=cp_group_size,
                    collect_non_loss_data=collect_non_loss_data,
                    is_first_microbatch=check_first_val_step(
                        first_val_step,
                        forward_only,
                        op.microbatch == 0,
                    ),
                    current_microbatch=op.microbatch,
                    vp_stage=getattr(model_chunks[local_index], "vp_stage", None),
                    is_last_stage=op.stage == plan.num_stages - 1,
                )
                output_tensors[key] = output_tensor
                if op.stage != plan.num_stages - 1:
                    detached_output = _detach_pipeline_tensor(output_tensor)
                    if plan.stage_to_worker[op.stage + 1] == pp_rank:
                        forwarded_output_tensors[key] = detached_output
                    else:
                        assert communicator is not None
                        communicator.send_forward(op.stage, detached_output)
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
                    next_key = (op.microbatch, op.stage + 1)
                    if plan.stage_to_worker[op.stage + 1] == pp_rank:
                        if next_key not in input_tensor_grads:
                            raise ValueError(
                                f"SlackPipe plan runs B({op.microbatch},{op.stage}) before "
                                f"B({op.microbatch},{op.stage + 1}) produced its gradient"
                            )
                        output_tensor_grad = input_tensor_grads.pop(next_key)
                    else:
                        assert communicator is not None
                        output_tensor_grad = communicator.recv_backward(
                            op.stage,
                            pipeline_tensor_shape,
                            pipeline_tensor_dtype,
                            pipeline_tensor_device,
                        )

                input_tensor_grad = backward_step(
                    input_tensors[key],
                    output_tensors[key],
                    output_tensor_grad,
                    config,
                )
                if op.stage != 0:
                    if plan.stage_to_worker[op.stage - 1] == pp_rank:
                        input_tensor_grads[key] = input_tensor_grad
                    else:
                        assert communicator is not None
                        communicator.send_backward(op.stage, input_tensor_grad)
                del input_tensors[key]
                del output_tensors[key]
                forwarded_output_tensors.pop(key, None)

    if communicator is not None:
        communicator.drain_sends()
        communicator.assert_no_outstanding_work()

    if not forward_only:
        _assert_no_tensor_state(
            input_tensors,
            output_tensors,
            forwarded_output_tensors,
            input_tensor_grads,
        )

    if config.finalize_model_grads_func is not None and not forward_only:
        config.finalize_model_grads_func(
            model_chunks,
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    return forward_data_store


def _validate_supported_parallelism() -> None:
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    cp_size = parallel_state.get_context_parallel_world_size()
    dp_size = parallel_state.get_data_parallel_world_size(with_context_parallel=True)
    if pp_size not in (1, 2):
        raise ValueError(
            f"SlackPipe prototype requires pipeline model parallel size 1 or 2, got {pp_size}"
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


def _validate_unsupported_features(config, forward_only: bool) -> None:
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
