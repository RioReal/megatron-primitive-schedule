# Copyright (c) 2026 NVIDIA CORPORATION. All rights reserved.

import json

import pytest

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.slackpipe.plan import (
    SLACKPIPE_PLAN_SCHEMA_VERSION,
    SLACKPIPE_PLAN_SCHEMA_VERSION_V2,
    SlackPipeOperation,
    derive_pipeline_model_parallel_layout,
    load_slackpipe_plan,
    parse_slackpipe_plan,
    validate_cyclic_placement,
    validate_plan_parallel_layout,
)
from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout


def _valid_plan():
    return {
        "num_microbatches": 2,
        "num_stages": 2,
        "num_workers": 1,
        "num_layers": 4,
        "layer_split": [2, 2],
        "stage_to_worker": [0, 0],
        "operations": [
            [
                {"kind": "F", "microbatch": 0, "stage": 0},
                {"kind": "F", "microbatch": 0, "stage": 1},
                {"kind": "B", "microbatch": 0, "stage": 1},
                {"kind": "B", "microbatch": 0, "stage": 0},
                {"kind": "F", "microbatch": 1, "stage": 0},
                {"kind": "F", "microbatch": 1, "stage": 1},
                {"kind": "B", "microbatch": 1, "stage": 1},
                {"kind": "B", "microbatch": 1, "stage": 0},
            ]
        ],
    }


def test_parse_valid_plan():
    plan = parse_slackpipe_plan(_valid_plan(), pipeline_model_parallel_size=1)

    assert plan.schema_version == SLACKPIPE_PLAN_SCHEMA_VERSION
    assert plan.num_microbatches == 2
    assert plan.num_stages == 2
    assert plan.num_workers == 1
    assert plan.num_layers == 4
    assert plan.layer_split == (2, 2)
    assert plan.stage_layer_ranges == ((0, 2), (2, 4))
    assert plan.stage_to_worker == (0, 0)
    assert plan.local_operations[0] == SlackPipeOperation("F", 0, 0)


def test_parse_versioned_solver_metadata():
    payload = _valid_plan()
    payload.update(
        {
            "schema_version": SLACKPIPE_PLAN_SCHEMA_VERSION,
            "solver_status": "optimal",
            "predicted_makespan": 12.5,
            "forward_costs": [1.0, 2.0],
            "backward_costs": [3.0, 4.0],
        }
    )

    plan = parse_slackpipe_plan(payload, pipeline_model_parallel_size=1)

    assert plan.solver_status == "optimal"
    assert plan.predicted_makespan == 12.5
    assert plan.forward_costs == (1.0, 2.0)
    assert plan.backward_costs == (3.0, 4.0)


def test_parse_plan_v2_with_stage_layer_ranges():
    payload = _valid_plan()
    payload.pop("layer_split")
    payload.update(
        {
            "schema_version": SLACKPIPE_PLAN_SCHEMA_VERSION_V2,
            "stage_layer_ranges": [{"begin": 0, "end": 1}, {"begin": 1, "end": 4}],
            "model_manifest_hash": "model-sha",
            "cost_profile_hash": "profile-sha",
            "cost_profile_version": "slackpipe.cost_profile.v2",
        }
    )

    plan = parse_slackpipe_plan(payload, pipeline_model_parallel_size=1)

    assert plan.schema_version == SLACKPIPE_PLAN_SCHEMA_VERSION_V2
    assert plan.stage_layer_ranges == ((0, 1), (1, 4))
    assert plan.layer_split == (1, 3)
    assert plan.stage_layer_ids(0) == (0,)
    assert plan.stage_layer_ids(1) == (1, 2, 3)
    assert derive_pipeline_model_parallel_layout(plan) == "Et|t*3L"
    assert plan.model_manifest_hash == "model-sha"
    assert plan.cost_profile_hash == "profile-sha"
    assert plan.cost_profile_version == "slackpipe.cost_profile.v2"


def test_parse_plan_v2_with_layer_cuts():
    payload = _valid_plan()
    payload.pop("layer_split")
    payload.update({"schema_version": SLACKPIPE_PLAN_SCHEMA_VERSION_V2, "layer_cuts": [0, 3, 4]})

    plan = parse_slackpipe_plan(payload, pipeline_model_parallel_size=1)

    assert plan.stage_layer_ranges == ((0, 3), (3, 4))
    assert plan.layer_split == (3, 1)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("stage_layer_ranges", [[0, 1], [2, 4]], "stage_layer_ranges must be contiguous"),
        ("stage_layer_ranges", [[0, 0], [0, 4]], "stage_layer_ranges must be non-empty"),
        ("stage_layer_ranges", [[0, 1], [1, 3]], "stage_layer_ranges must cover num_layers"),
        ("layer_cuts", [0, 2], "layer_cuts must have num_stages \\+ 1 entries"),
    ],
)
def test_plan_v2_rejects_invalid_ranges(field, value, error):
    payload = _valid_plan()
    payload.pop("layer_split")
    payload["schema_version"] = SLACKPIPE_PLAN_SCHEMA_VERSION_V2
    payload[field] = value

    with pytest.raises(ValueError, match=error):
        parse_slackpipe_plan(payload, pipeline_model_parallel_size=1)


def test_plan_v2_rejects_mismatched_compat_layer_split():
    payload = _valid_plan()
    payload.update(
        {
            "schema_version": SLACKPIPE_PLAN_SCHEMA_VERSION_V2,
            "stage_layer_ranges": [[0, 1], [1, 4]],
            "layer_split": [2, 2],
        }
    )

    with pytest.raises(ValueError, match="layer_split must match stage_layer_ranges"):
        parse_slackpipe_plan(payload, pipeline_model_parallel_size=1)


def test_plan_v2_ranges_match_megatron_layout_global_layer_ids():
    payload = {
        "schema_version": SLACKPIPE_PLAN_SCHEMA_VERSION_V2,
        "num_microbatches": 1,
        "num_stages": 4,
        "num_workers": 2,
        "num_layers": 8,
        "stage_layer_ranges": [
            {"begin": 0, "end": 1},
            {"begin": 1, "end": 3},
            {"begin": 3, "end": 4},
            {"begin": 4, "end": 8},
        ],
        "stage_to_worker": [0, 1, 0, 1],
        "operations": [
            [
                {"kind": "F", "microbatch": 0, "stage": 0},
                {"kind": "F", "microbatch": 0, "stage": 2},
                {"kind": "B", "microbatch": 0, "stage": 2},
                {"kind": "B", "microbatch": 0, "stage": 0},
            ],
            [
                {"kind": "F", "microbatch": 0, "stage": 1},
                {"kind": "F", "microbatch": 0, "stage": 3},
                {"kind": "B", "microbatch": 0, "stage": 3},
                {"kind": "B", "microbatch": 0, "stage": 1},
            ],
        ],
    }
    plan = parse_slackpipe_plan(payload, pipeline_model_parallel_size=2)
    layout = PipelineParallelLayerLayout(
        derive_pipeline_model_parallel_layout(plan), pipeline_model_parallel_size=2
    )

    parallel_state.set_virtual_pipeline_model_parallel_world_size(2)
    try:
        for stage in range(plan.num_stages):
            pp_rank = plan.stage_to_worker[stage]
            vp_stage = stage // plan.num_workers
            assert tuple(
                layout.get_layer_id_list(LayerType.decoder, vp_stage=vp_stage, pp_rank=pp_rank)
            ) == plan.stage_layer_ids(stage)
    finally:
        parallel_state.set_virtual_pipeline_model_parallel_world_size(None)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("schema_version", "slackpipe.plan.v999", "Unsupported SlackPipe plan schema_version"),
        ("predicted_makespan", -1.0, "predicted_makespan.*non-negative"),
        ("forward_costs", [1.0], "forward_costs.*num_stages entries"),
        ("backward_costs", [1.0, -2.0], "backward_costs.*non-negative"),
    ],
)
def test_plan_rejects_invalid_schema_metadata(field, value, error):
    payload = _valid_plan()
    payload[field] = value

    with pytest.raises(ValueError, match=error):
        parse_slackpipe_plan(payload, pipeline_model_parallel_size=1)


def test_load_plan_from_json(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_valid_plan()), encoding="utf-8")

    plan = load_slackpipe_plan(plan_path, pipeline_model_parallel_size=1)

    assert len(plan.local_operations) == 8


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("num_workers", 2, "num_workers must equal pipeline model parallel size"),
        ("layer_split", [1, 2], "layer_split must sum to num_layers"),
        ("stage_to_worker", [0], "stage_to_worker must have num_stages entries"),
    ],
)
def test_plan_top_level_validation(field, value, error):
    payload = _valid_plan()
    payload[field] = value

    with pytest.raises(ValueError, match=error):
        parse_slackpipe_plan(payload, pipeline_model_parallel_size=1)


def test_plan_requires_every_forward_once():
    payload = _valid_plan()
    payload["operations"][0].pop(1)

    with pytest.raises(ValueError, match="missing operations"):
        parse_slackpipe_plan(payload, pipeline_model_parallel_size=1)


def test_plan_rejects_duplicate_backward():
    payload = _valid_plan()
    payload["operations"][0][-1] = {"kind": "B", "microbatch": 1, "stage": 1}

    with pytest.raises(ValueError, match="duplicate operations"):
        parse_slackpipe_plan(payload, pipeline_model_parallel_size=1)


@pytest.mark.parametrize(
    "operation,error",
    [
        ({"kind": "F", "microbatch": 2, "stage": 0}, "invalid microbatch index"),
        ({"kind": "F", "microbatch": 0, "stage": 2}, "invalid stage index"),
        ({"kind": "X", "microbatch": 0, "stage": 0}, "kind must be 'F' or 'B'"),
    ],
)
def test_plan_rejects_invalid_operation(operation, error):
    payload = _valid_plan()
    payload["operations"][0][0] = operation

    with pytest.raises(ValueError, match=error):
        parse_slackpipe_plan(payload, pipeline_model_parallel_size=1)


def test_plan_rejects_operation_on_wrong_worker():
    payload = _valid_plan()
    payload["num_workers"] = 2
    payload["stage_to_worker"] = [0, 1]
    payload["operations"] = {
        "0": [
            {"kind": "F", "microbatch": 0, "stage": 0},
            {"kind": "F", "microbatch": 0, "stage": 1},
            {"kind": "B", "microbatch": 0, "stage": 0},
            {"kind": "F", "microbatch": 1, "stage": 0},
            {"kind": "B", "microbatch": 1, "stage": 0},
        ],
        "1": [
            {"kind": "B", "microbatch": 0, "stage": 1},
            {"kind": "F", "microbatch": 1, "stage": 1},
            {"kind": "B", "microbatch": 1, "stage": 1},
        ],
    }

    with pytest.raises(ValueError, match="wrong worker"):
        parse_slackpipe_plan(payload, pipeline_model_parallel_size=2)


def test_validate_cyclic_placement_rejects_non_cyclic_plan():
    payload = _valid_plan()
    payload["num_workers"] = 2
    payload["stage_to_worker"] = [0, 0]
    payload["operations"] = [payload["operations"][0], []]
    plan = parse_slackpipe_plan(payload, pipeline_model_parallel_size=2)

    with pytest.raises(ValueError, match="cyclic placement"):
        validate_cyclic_placement(plan)


def test_plan_derives_megatron_layout_and_vpp():
    plan = parse_slackpipe_plan(_valid_plan(), pipeline_model_parallel_size=1)

    assert derive_pipeline_model_parallel_layout(plan) == "Et*2|t*2L"
    assert validate_plan_parallel_layout(plan, pipeline_model_parallel_size=1) == 2


def test_validate_plan_parallel_layout_rejects_nondivisible_plan():
    payload = {
        "num_microbatches": 1,
        "num_stages": 3,
        "num_workers": 2,
        "num_layers": 3,
        "layer_split": [1, 1, 1],
        "stage_to_worker": [0, 1, 0],
        "operations": [
            [
                {"kind": "F", "microbatch": 0, "stage": 0},
                {"kind": "F", "microbatch": 0, "stage": 2},
                {"kind": "B", "microbatch": 0, "stage": 2},
                {"kind": "B", "microbatch": 0, "stage": 0},
            ],
            [
                {"kind": "F", "microbatch": 0, "stage": 1},
                {"kind": "B", "microbatch": 0, "stage": 1},
            ],
        ],
    }
    plan = parse_slackpipe_plan(payload, pipeline_model_parallel_size=2)

    with pytest.raises(ValueError, match="num_stages must be divisible by num_workers"):
        validate_plan_parallel_layout(plan, pipeline_model_parallel_size=2)


def test_plan_rejects_dependency_cycle():
    payload = _valid_plan()
    payload["operations"][0] = [
        {"kind": "F", "microbatch": 0, "stage": 1},
        {"kind": "F", "microbatch": 0, "stage": 0},
        {"kind": "B", "microbatch": 0, "stage": 1},
        {"kind": "B", "microbatch": 0, "stage": 0},
        {"kind": "F", "microbatch": 1, "stage": 0},
        {"kind": "F", "microbatch": 1, "stage": 1},
        {"kind": "B", "microbatch": 1, "stage": 1},
        {"kind": "B", "microbatch": 1, "stage": 0},
    ]

    with pytest.raises(ValueError, match="dependency graph contains a cycle"):
        parse_slackpipe_plan(payload, pipeline_model_parallel_size=1)


def test_plan_rejects_fifo_violation():
    payload = _valid_plan()
    payload["operations"][0] = [
        {"kind": "F", "microbatch": 1, "stage": 0},
        {"kind": "F", "microbatch": 0, "stage": 0},
        {"kind": "F", "microbatch": 0, "stage": 1},
        {"kind": "B", "microbatch": 0, "stage": 1},
        {"kind": "B", "microbatch": 0, "stage": 0},
        {"kind": "F", "microbatch": 1, "stage": 1},
        {"kind": "B", "microbatch": 1, "stage": 1},
        {"kind": "B", "microbatch": 1, "stage": 0},
    ]

    with pytest.raises(ValueError, match="violates FIFO microbatch order"):
        parse_slackpipe_plan(payload, pipeline_model_parallel_size=1)
