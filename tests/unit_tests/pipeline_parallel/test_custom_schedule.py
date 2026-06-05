# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import replace

import pytest

from megatron.core.pipeline_parallel.custom_schedule import (
    generate_custom_pipeline_schedule,
    validate_custom_pipeline_schedule,
)


def _task_by_index(schedule):
    return {(task.microbatch_id, task.op_index): task for task in schedule.global_tasks}


def _deterministic_sort(tasks):
    return sorted(
        tasks,
        key=lambda task: (
            task.start_time,
            task.end_time,
            task.microbatch_id,
            task.op_index,
            task.logical_stage_idx,
            int(not task.is_forward),
        ),
    )


def test_custom_pipeline_schedule_generates_all_tasks():
    schedule = generate_custom_pipeline_schedule(
        num_microbatches=4,
        num_logical_stages=4,
        num_physical_workers=2,
        stage_layer_counts=[1, 2, 3, 2],
    )

    assert len(schedule.global_tasks) == 4 * 2 * 4
    assert sum(len(tasks) for tasks in schedule.per_rank_tasks.values()) == len(
        schedule.global_tasks
    )


def test_custom_pipeline_schedule_maps_pp_and_vp_ranks():
    schedule = generate_custom_pipeline_schedule(
        num_microbatches=4,
        num_logical_stages=4,
        num_physical_workers=2,
        stage_layer_counts=[1, 2, 3, 2],
    )

    for task in schedule.global_tasks:
        assert task.pp_rank == task.logical_stage_idx % 2
        assert task.vp_rank == task.logical_stage_idx // 2
        if task.is_forward:
            assert task.logical_stage_idx == task.op_index
        else:
            assert task.logical_stage_idx == 2 * 4 - task.op_index - 1


def test_custom_pipeline_schedule_forward_only_generates_forward_tasks():
    schedule = generate_custom_pipeline_schedule(
        num_microbatches=4,
        num_logical_stages=4,
        num_physical_workers=2,
        stage_layer_counts=[1, 2, 3, 2],
        forward_only=True,
    )

    assert len(schedule.global_tasks) == 4 * 4
    assert all(task.is_forward for task in schedule.global_tasks)
    assert sum(len(tasks) for tasks in schedule.per_rank_tasks.values()) == 4 * 4
    for task in schedule.global_tasks:
        assert task.logical_stage_idx == task.op_index
        assert task.pp_rank == task.logical_stage_idx % 2
        assert task.vp_rank == task.logical_stage_idx // 2

    validate_custom_pipeline_schedule(
        schedule.global_tasks, num_logical_stages=4, forward_only=True
    )


def test_custom_pipeline_schedule_uses_last_forward_special_dependency():
    num_microbatches = 4
    num_logical_stages = 4
    num_physical_workers = 2
    schedule = generate_custom_pipeline_schedule(
        num_microbatches=num_microbatches,
        num_logical_stages=num_logical_stages,
        num_physical_workers=num_physical_workers,
        stage_layer_counts=[1, 2, 3, 2],
    )
    tasks = _task_by_index(schedule)

    special_task = tasks[(0, num_logical_stages - 1)]
    wrapped_dependency = tasks[
        (num_microbatches - 1, num_logical_stages - 1 - num_physical_workers)
    ]
    previous_op = tasks[(0, num_logical_stages - 2)]

    assert wrapped_dependency.end_time > previous_op.end_time
    assert special_task.start_time == wrapped_dependency.end_time


@pytest.mark.parametrize(
    "num_microbatches,num_logical_stages,num_physical_workers,stage_layer_counts",
    [
        (1, 2, 1, [1, 1]),
        (2, 2, 1, [1, 2]),
        (4, 4, 2, [1, 2, 3, 2]),
        (3, 6, 3, [1, 1, 2, 2, 3, 3]),
    ],
)
def test_custom_pipeline_schedule_has_no_deadlock_for_valid_small_examples(
    num_microbatches,
    num_logical_stages,
    num_physical_workers,
    stage_layer_counts,
):
    schedule = generate_custom_pipeline_schedule(
        num_microbatches=num_microbatches,
        num_logical_stages=num_logical_stages,
        num_physical_workers=num_physical_workers,
        stage_layer_counts=stage_layer_counts,
    )

    assert len(schedule.global_tasks) == num_microbatches * 2 * num_logical_stages
    assert all(task.start_time < task.end_time for task in schedule.global_tasks)


def test_custom_pipeline_schedule_rejects_logical_stage_count_not_divisible_by_workers():
    with pytest.raises(ValueError, match="divisible"):
        generate_custom_pipeline_schedule(
            num_microbatches=4,
            num_logical_stages=3,
            num_physical_workers=2,
            stage_layer_counts=[1, 2, 3],
        )


def test_custom_pipeline_schedule_validator_accepts_generated_tasks():
    schedule = generate_custom_pipeline_schedule(
        num_microbatches=4,
        num_logical_stages=4,
        num_physical_workers=2,
        stage_layer_counts=[1, 2, 3, 2],
    )

    validate_custom_pipeline_schedule(schedule.global_tasks, num_logical_stages=4)


def test_custom_pipeline_schedule_validator_rejects_missing_forward_recv():
    schedule = generate_custom_pipeline_schedule(
        num_microbatches=4,
        num_logical_stages=4,
        num_physical_workers=2,
        stage_layer_counts=[1, 2, 3, 2],
    )
    corrupted_tasks = [
        task
        for task in schedule.global_tasks
        if not (task.microbatch_id == 0 and task.is_forward and task.logical_stage_idx == 1)
    ]

    with pytest.raises(ValueError, match="missing matching forward recv"):
        validate_custom_pipeline_schedule(corrupted_tasks, num_logical_stages=4)


def test_custom_pipeline_schedule_validator_rejects_bad_backward_recv_order():
    schedule = generate_custom_pipeline_schedule(
        num_microbatches=4,
        num_logical_stages=4,
        num_physical_workers=2,
        stage_layer_counts=[1, 2, 3, 2],
    )
    tasks = []
    for task in schedule.global_tasks:
        if task.microbatch_id == 0 and not task.is_forward and task.logical_stage_idx == 2:
            tasks.append(replace(task, start_time=15, end_time=16))
        elif task.microbatch_id == 0 and not task.is_forward and task.logical_stage_idx == 1:
            tasks.append(replace(task, start_time=15, end_time=17))
        else:
            tasks.append(task)

    with pytest.raises(ValueError, match="backward recv task starts before producer"):
        validate_custom_pipeline_schedule(_deterministic_sort(tasks), num_logical_stages=4)


def test_custom_pipeline_schedule_validator_rejects_backward_before_forward():
    schedule = generate_custom_pipeline_schedule(
        num_microbatches=4,
        num_logical_stages=4,
        num_physical_workers=2,
        stage_layer_counts=[1, 2, 3, 2],
    )
    tasks = []
    for task in schedule.global_tasks:
        if task.microbatch_id == 0 and not task.is_forward and task.logical_stage_idx == 0:
            tasks.append(replace(task, start_time=0, end_time=1))
        else:
            tasks.append(task)

    with pytest.raises(ValueError, match="backward task starts before forward"):
        validate_custom_pipeline_schedule(_deterministic_sort(tasks), num_logical_stages=4)


def test_custom_pipeline_schedule_validator_rejects_nondeterministic_global_order():
    schedule = generate_custom_pipeline_schedule(
        num_microbatches=4,
        num_logical_stages=4,
        num_physical_workers=2,
        stage_layer_counts=[1, 2, 3, 2],
    )
    corrupted_tasks = list(schedule.global_tasks)
    corrupted_tasks[0], corrupted_tasks[1] = corrupted_tasks[1], corrupted_tasks[0]

    with pytest.raises(ValueError, match="sorted deterministically"):
        validate_custom_pipeline_schedule(corrupted_tasks, num_logical_stages=4)
