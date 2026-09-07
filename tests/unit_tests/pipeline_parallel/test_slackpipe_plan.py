# Copyright (c) 2026 NVIDIA CORPORATION. All rights reserved.

import json

import pytest

from megatron.core.pipeline_parallel.slackpipe.plan import (
    SlackPipeOperation,
    load_slackpipe_plan,
    parse_slackpipe_plan,
    validate_cyclic_placement,
)


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

    assert plan.num_microbatches == 2
    assert plan.num_stages == 2
    assert plan.num_workers == 1
    assert plan.num_layers == 4
    assert plan.layer_split == (2, 2)
    assert plan.stage_to_worker == (0, 0)
    assert plan.local_operations[0] == SlackPipeOperation("F", 0, 0)


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
