# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from megatron.core.pipeline_parallel.custom_schedule import generate_custom_pipeline_schedule
from tools.profile_and_suggest_partition import estimate_single_run_shared_slope_and_bias
from tools.primitive_partition_optimizer import (
    _solve_with_cp_sat,
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


def test_shared_slope_stage_bias_recovers_known_bias():
    slope, bias = estimate_single_run_shared_slope_and_bias(
        layers=[2, 4, 6, 8],
        costs=[6, 17, 18, 29],
    )

    assert slope == 3
    assert bias == [0.0, 5.0, 0.0, 5.0]


def test_exact_optimizer_preserves_total_layers():
    solution = solve_primitive_profile_guided_partition(
        B=4,
        N=4,
        J=2,
        total_layers=8,
        fwd_per_layer=[1.0, 1.2, 0.9, 1.1],
        bwd_per_layer=[2.0, 2.4, 1.8, 2.2],
    )

    assert solution.search_method == "exact_enumeration"
    assert sum(solution.t_split) == 8
    assert all(layers >= 1 for layers in solution.t_split)


def test_exact_optimizer_is_no_worse_than_current_partition():
    current = [1, 2, 3, 2]
    fwd_per_layer = [1.0, 1.2, 0.9, 1.1]
    bwd_per_layer = [2.0, 2.4, 1.8, 2.2]
    current_makespan = primitive_profile_guided_makespan(
        B=4,
        N=4,
        J=2,
        t_split=current,
        fwd_per_layer=fwd_per_layer,
        bwd_per_layer=bwd_per_layer,
    )

    solution = solve_primitive_profile_guided_partition(
        B=4,
        N=4,
        J=2,
        total_layers=sum(current),
        fwd_per_layer=fwd_per_layer,
        bwd_per_layer=bwd_per_layer,
    )

    assert solution.makespan <= current_makespan


def test_cp_sat_optimizer_preserves_total_layers():
    pytest.importorskip("ortools.sat.python.cp_model")

    solution = _solve_with_cp_sat(
        B=4,
        N=4,
        J=2,
        total_layers=8,
        fwd=[1.0, 1.2, 0.9, 1.1],
        bwd=[2.0, 2.4, 1.8, 2.2],
        fixed_forward=[0.0, 0.0, 0.0, 0.0],
        fixed_backward=[0.0, 0.0, 0.0, 0.0],
        mins=[1, 1, 1, 1],
        maxes=[None, None, None, None],
    )

    assert solution is not None
    t_split, _score, search_method = solution
    assert search_method.startswith("cp_sat")
    assert sum(t_split) == 8
    assert all(layers >= 1 for layers in t_split)


def test_t_split_to_megatron_vpp_string_uses_logical_stage_mapping():
    partition = t_split_to_megatron_vpp_string([1, 2, 3, 4], pp_size=2, vpp_size=2)

    rows = partition.split(";")
    assert partition == "1,3;2,4"
    assert len(rows) == 2
    assert all(len(row.split(",")) == 2 for row in rows)
