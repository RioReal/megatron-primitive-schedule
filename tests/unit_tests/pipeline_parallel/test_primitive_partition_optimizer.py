# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.core.pipeline_parallel.custom_schedule import generate_custom_pipeline_schedule
from tools.primitive_partition_optimizer import (
    primitive_profile_guided_makespan,
    solve_primitive_profile_guided_partition,
    t_split_to_megatron_vpp_string,
)


def _old_model_makespan(B, N, J, t_split):
    schedule = generate_custom_pipeline_schedule(
        num_microbatches=B,
        num_logical_stages=N,
        num_physical_workers=J,
        stage_layer_counts=t_split,
    )
    return max(task.end_time for task in schedule.global_tasks)


def test_profile_guided_optimizer_matches_old_weight_model():
    solution = solve_primitive_profile_guided_partition(
        B=4,
        N=4,
        J=2,
        total_layers=8,
        fwd_per_layer=[1, 1, 1, 1],
        bwd_per_layer=[2, 2, 2, 2],
    )

    assert solution.makespan == _old_model_makespan(4, 4, 2, solution.t_split)
    assert solution.makespan == primitive_profile_guided_makespan(
        B=4,
        N=4,
        J=2,
        t_split=solution.t_split,
        fwd_per_layer=[1, 1, 1, 1],
        bwd_per_layer=[2, 2, 2, 2],
    )


def test_profile_guided_optimizer_t_split_length_matches_pp_times_vpp():
    pp_size = 2
    vpp_size = 2
    solution = solve_primitive_profile_guided_partition(
        B=4,
        N=pp_size * vpp_size,
        J=pp_size,
        total_layers=8,
        fwd_per_layer=[1.0, 1.2, 0.9, 1.1],
        bwd_per_layer=[2.0, 2.4, 1.8, 2.2],
    )

    assert len(solution.t_split) == pp_size * vpp_size
    assert sum(solution.t_split) == 8
    assert all(layers >= 1 for layers in solution.t_split)


def test_t_split_to_megatron_vpp_string_uses_logical_stage_mapping():
    partition = t_split_to_megatron_vpp_string([1, 2, 3, 4], pp_size=2, vpp_size=2)

    rows = partition.split(";")
    assert partition == "1,3;2,4"
    assert len(rows) == 2
    assert all(len(row.split(",")) == 2 for row in rows)
