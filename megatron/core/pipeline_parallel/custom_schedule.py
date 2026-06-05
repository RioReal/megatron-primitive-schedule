# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure Python helpers for custom table-driven pipeline schedules.

This module intentionally does not integrate with Megatron runtime scheduling. It only builds a
static task table from the DP recurrence used by the prototype custom pipeline schedule.
"""

from dataclasses import dataclass
from math import inf
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class CustomPipelineTask:
    """One logical forward or backward task in the generated static schedule."""

    microbatch_id: int
    op_index: int
    is_forward: bool
    logical_stage_idx: int
    pp_rank: int
    vp_rank: int
    start_time: int
    end_time: int
    weight: int


@dataclass(frozen=True)
class CustomPipelineSchedule:
    """Generated global and per-physical-rank task lists."""

    global_tasks: List[CustomPipelineTask]
    per_rank_tasks: Dict[int, List[CustomPipelineTask]]


def _task_sort_key(task: CustomPipelineTask) -> Tuple[int, int, int, int, int, int]:
    return (
        task.start_time,
        task.end_time,
        task.microbatch_id,
        task.op_index,
        task.logical_stage_idx,
        int(not task.is_forward),
    )


def _validate_inputs(
    num_microbatches: int,
    num_logical_stages: int,
    num_physical_workers: int,
    stage_layer_counts: Sequence[int],
) -> None:
    if num_microbatches <= 0:
        raise ValueError("num_microbatches must be positive")
    if num_logical_stages <= 0:
        raise ValueError("num_logical_stages must be positive")
    if num_physical_workers <= 0:
        raise ValueError("num_physical_workers must be positive")
    if num_logical_stages % num_physical_workers != 0:
        raise ValueError("num_logical_stages must be divisible by num_physical_workers")
    if len(stage_layer_counts) != num_logical_stages:
        raise ValueError("stage_layer_counts length must equal num_logical_stages")
    if any(not isinstance(count, int) or count <= 0 for count in stage_layer_counts):
        raise ValueError("stage_layer_counts entries must all be positive integers")


def _get_weights(num_logical_stages: int, stage_layer_counts: Sequence[int]) -> List[int]:
    weights = []
    for op_index in range(2 * num_logical_stages):
        if op_index < num_logical_stages:
            weights.append(stage_layer_counts[op_index])
        else:
            weights.append(2 * stage_layer_counts[2 * num_logical_stages - op_index - 1])
    return weights


def _get_prev_b_dependency(
    microbatch_id: int,
    op_index: int,
    num_microbatches: int,
    num_logical_stages: int,
    num_physical_workers: int,
) -> Tuple[int, int]:
    if op_index == num_logical_stages - 1:
        if microbatch_id == 0:
            return num_microbatches - 1, op_index - num_physical_workers
        return microbatch_id - 1, num_logical_stages

    if op_index == num_logical_stages:
        return microbatch_id, op_index - 1

    if microbatch_id > 0:
        return microbatch_id - 1, op_index

    dep_microbatch_id = num_microbatches - 1
    if op_index < num_logical_stages - 1 or op_index >= num_logical_stages + num_physical_workers:
        dep_op_index = op_index - num_physical_workers
    else:
        dep_op_index = 2 * num_logical_stages - op_index - 1
    return dep_microbatch_id, dep_op_index


def _dependency_is_known(
    table: List[List[float]],
    microbatch_id: int,
    op_index: int,
    num_microbatches: int,
    num_ops: int,
) -> bool:
    if not (0 <= microbatch_id < num_microbatches and 0 <= op_index < num_ops):
        raise RuntimeError(
            "custom pipeline schedule dependency is out of bounds "
            f"({microbatch_id=}, {op_index=})"
        )
    return table[microbatch_id][op_index] != inf


def _resolve_completion_table(
    num_microbatches: int,
    num_logical_stages: int,
    num_physical_workers: int,
    weights: Sequence[int],
) -> List[List[int]]:
    num_ops = 2 * num_logical_stages
    table = [[inf for _ in range(num_ops)] for _ in range(num_microbatches)]
    fixed_cells = set()

    for microbatch_id in range(num_microbatches):
        table[microbatch_id][0] = (microbatch_id + 1) * weights[0]
        fixed_cells.add((microbatch_id, 0))

    for op_index in range(1, num_physical_workers):
        table[0][op_index] = sum(weights[: op_index + 1])
        fixed_cells.add((0, op_index))

    while True:
        progress = False
        unresolved = 0

        for microbatch_id in range(num_microbatches):
            for op_index in range(num_ops):
                if table[microbatch_id][op_index] != inf:
                    continue
                unresolved += 1

                if not _dependency_is_known(
                    table, microbatch_id, op_index - 1, num_microbatches, num_ops
                ):
                    continue
                dep_microbatch_id, dep_op_index = _get_prev_b_dependency(
                    microbatch_id,
                    op_index,
                    num_microbatches,
                    num_logical_stages,
                    num_physical_workers,
                )
                if not _dependency_is_known(
                    table, dep_microbatch_id, dep_op_index, num_microbatches, num_ops
                ):
                    continue

                table[microbatch_id][op_index] = (
                    max(table[dep_microbatch_id][dep_op_index], table[microbatch_id][op_index - 1])
                    + weights[op_index]
                )
                progress = True

        if unresolved == 0:
            return [[int(value) for value in row] for row in table]
        if not progress:
            fixed = sorted(fixed_cells)
            raise RuntimeError(
                "custom pipeline schedule dependency resolution made no progress "
                f"({fixed=})"
            )


def validate_custom_pipeline_schedule(
    global_tasks: Sequence[CustomPipelineTask],
    num_logical_stages: int,
    forward_only: bool = False,
) -> None:
    """Validate local ordering and communication matching for a generated task list.

    The task list contains compute tasks only. Sends and receives are implicit:
    forward task ``s`` sends activations to forward task ``s + 1`` for the same
    microbatch, and backward task ``s`` sends gradients to backward task ``s - 1``.
    """

    if num_logical_stages <= 0:
        raise ValueError("num_logical_stages must be positive")

    expected_global_order = sorted(global_tasks, key=_task_sort_key)
    if list(global_tasks) != expected_global_order:
        raise ValueError("global task list must be sorted deterministically")

    forward_tasks = {}
    backward_tasks = {}
    per_rank_tasks = {}
    seen_tasks = set()

    for task in global_tasks:
        if not (0 <= task.logical_stage_idx < num_logical_stages):
            raise ValueError(f"task logical_stage_idx is out of range ({task=})")

        identity = (task.microbatch_id, task.op_index, task.is_forward)
        if identity in seen_tasks:
            raise ValueError(f"duplicate task found ({identity=})")
        seen_tasks.add(identity)

        per_rank_tasks.setdefault(task.pp_rank, []).append(task)
        key = (task.microbatch_id, task.logical_stage_idx)
        if task.is_forward:
            if key in forward_tasks:
                raise ValueError(f"duplicate forward task found ({key=})")
            forward_tasks[key] = task
        else:
            if forward_only:
                raise ValueError(f"forward-only schedule contains backward task ({task=})")
            if key in backward_tasks:
                raise ValueError(f"duplicate backward task found ({key=})")
            backward_tasks[key] = task

    for pp_rank, local_tasks in per_rank_tasks.items():
        if local_tasks != sorted(local_tasks, key=_task_sort_key):
            raise ValueError(f"local task order must be deterministic ({pp_rank=})")

    for key, forward_task in forward_tasks.items():
        microbatch_id, logical_stage_idx = key
        if not forward_only:
            backward_task = backward_tasks.get(key)
            if backward_task is None:
                raise ValueError(f"missing backward task for forward task ({key=})")
            if backward_task.start_time < forward_task.end_time:
                raise ValueError(f"backward task starts before forward task completes ({key=})")

        if logical_stage_idx < num_logical_stages - 1:
            recv_key = (microbatch_id, logical_stage_idx + 1)
            recv_task = forward_tasks.get(recv_key)
            if recv_task is None:
                raise ValueError(f"missing matching forward recv task ({key=}, {recv_key=})")
            if recv_task.start_time < forward_task.end_time:
                raise ValueError(
                    "forward recv task starts before producer forward send completes "
                    f"({key=}, {recv_key=})"
                )

    if not forward_only:
        for key, backward_task in backward_tasks.items():
            microbatch_id, logical_stage_idx = key
            if key not in forward_tasks:
                raise ValueError(f"backward task has no corresponding forward task ({key=})")

            if logical_stage_idx > 0:
                recv_key = (microbatch_id, logical_stage_idx - 1)
                recv_task = backward_tasks.get(recv_key)
                if recv_task is None:
                    raise ValueError(f"missing matching backward recv task ({key=}, {recv_key=})")
                if recv_task.start_time < backward_task.end_time:
                    raise ValueError(
                        "backward recv task starts before producer backward send completes "
                        f"({key=}, {recv_key=})"
                    )


def generate_custom_pipeline_schedule(
    num_microbatches: int,
    num_logical_stages: int,
    num_physical_workers: int,
    stage_layer_counts: Sequence[int],
    forward_only: bool = False,
) -> CustomPipelineSchedule:
    """Generate a static custom pipeline task schedule from the DP recurrence.

    Args:
        num_microbatches: Number of microbatches, B.
        num_logical_stages: Number of logical forward stages, N.
        num_physical_workers: Number of physical pipeline workers, J.
        stage_layer_counts: Per-logical-stage forward weights, ``t_split``.

    Returns:
        A ``CustomPipelineSchedule`` with globally sorted tasks and per-physical-rank task lists.
    """

    _validate_inputs(
        num_microbatches,
        num_logical_stages,
        num_physical_workers,
        stage_layer_counts,
    )
    weights = _get_weights(num_logical_stages, stage_layer_counts)
    table = _resolve_completion_table(
        num_microbatches,
        num_logical_stages,
        num_physical_workers,
        weights,
    )

    global_tasks = []
    for microbatch_id in range(num_microbatches):
        for op_index, end_time in enumerate(table[microbatch_id]):
            if forward_only and op_index >= num_logical_stages:
                continue
            is_forward = op_index < num_logical_stages
            if is_forward:
                logical_stage_idx = op_index
            else:
                logical_stage_idx = 2 * num_logical_stages - op_index - 1
            global_tasks.append(
                CustomPipelineTask(
                    microbatch_id=microbatch_id,
                    op_index=op_index,
                    is_forward=is_forward,
                    logical_stage_idx=logical_stage_idx,
                    pp_rank=logical_stage_idx % num_physical_workers,
                    vp_rank=logical_stage_idx // num_physical_workers,
                    start_time=end_time - weights[op_index],
                    end_time=end_time,
                    weight=weights[op_index],
                )
            )

    global_tasks.sort(
        key=lambda task: (task.start_time, task.end_time, task.microbatch_id, task.op_index)
    )
    validate_custom_pipeline_schedule(global_tasks, num_logical_stages, forward_only=forward_only)
    per_rank_tasks = {rank: [] for rank in range(num_physical_workers)}
    for task in global_tasks:
        per_rank_tasks[task.pp_rank].append(task)

    return CustomPipelineSchedule(global_tasks=global_tasks, per_rank_tasks=per_rank_tasks)
