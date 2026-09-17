# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Offline preparation and fail-closed gates; no full 8B allocation."""

import json
from dataclasses import asdict, replace

import pytest
import torch

from megatron.core.pipeline_parallel.slackpipe.cost_profile import (
    build_heterogeneous_cost_profile,
    profile_fingerprint,
)
from megatron.core.pipeline_parallel.slackpipe.hybrid import nemotron_h_8b_config
from megatron.core.pipeline_parallel.slackpipe.manifest import (
    build_model_manifest,
    validate_plan_model,
)
from megatron.core.pipeline_parallel.slackpipe.schedule import _validate_unsupported_features
from tests.unit_tests.pipeline_parallel.test_slackpipe_hybrid import tiny_config
from tools.run_slackpipe_nemotron_h8b_pp4 import (
    digest,
    invalidate,
    preflight,
    require_receipt,
    validate_target_plan,
)
from tools.slackpipe_nemotron_worker import calibration_partitions, construction_plan, write_json


def test_nemotron_calibration_design():
    manifest = build_model_manifest(nemotron_h_8b_config())
    partitions, diagnostics = calibration_partitions(manifest, 8)
    assert diagnostics["rank"] == diagnostics["columns"] == 6
    assert len(partitions) >= 6
    for cuts in partitions:
        plan = construction_plan(cuts, 4, 8, manifest)
        assert plan.num_layers == 52
        assert sum(map(len, plan.operations)) == 128
        assert list(range(52)) == [i for s in range(8) for i in plan.stage_layer_ids(s)]
    bad = {"layers": [{"config_class": "only"}] * 8}
    with pytest.raises(ValueError, match="Rank-deficient"):
        calibration_partitions(bad, 8)


def test_receipts_fail_closed(tmp_path):
    expected = dict(commit="abc", seq_length=1024)
    with pytest.raises(RuntimeError, match="Missing"):
        require_receipt(tmp_path, "env", expected)
    artifact = tmp_path / "run_metadata.json"
    write_json(artifact, {"gpus": 4})
    write_json(
        tmp_path / "receipts/env.json",
        dict(context=expected, artifacts={artifact.name: digest(artifact)}),
    )
    require_receipt(tmp_path, "env", expected)
    with pytest.raises(RuntimeError, match="Stale"):
        require_receipt(tmp_path, "env", dict(commit="changed"))
    write_json(artifact, {"gpus": 2})
    with pytest.raises(RuntimeError, match="Changed/missing"):
        require_receipt(tmp_path, "env", expected)
    for stage in ("pp4-correctness", "baseline-smoke", "solve", "benchmark"):
        write_json(tmp_path / f"receipts/{stage}.json", {})
    invalidate(tmp_path, "pp4-correctness")
    assert list((tmp_path / "receipts").glob("*.json")) == [tmp_path / "receipts/env.json"]


def test_target_plan_provenance(tmp_path):
    manifest = build_model_manifest(nemotron_h_8b_config())
    partitions, _ = calibration_partitions(manifest, 8)
    rows = []
    costs = {"mamba": 0.3, "attention": 0.4, "mlp": 0.2}
    for cuts in partitions:
        for s, (a, b) in enumerate(zip(cuts, cuts[1:])):
            cost = sum(costs[r["layer_type"]] for r in manifest["layers"][a:b]) + 0.01
            rows.append(
                dict(
                    begin=a,
                    end=b,
                    stage_role="first" if s == 0 else "last" if s == 7 else "middle",
                    forward_ms_per_op=cost,
                    backward_ms_per_op=3 * cost,
                )
            )
    profile = build_heterogeneous_cost_profile(
        model_manifest=manifest,
        observed_stage_rows=rows,
        model_config={"dtype": "bf16", "sequence_length": 1024},
        parallel_config={"pp": 4, "vpp": 2},
    )
    profile_path, manifest_path, path = [
        tmp_path / name for name in ("profile.json", "manifest.json", "plan.json")
    ]
    write_json(profile_path, profile)
    write_json(manifest_path, manifest)
    payload = asdict(construction_plan(partitions[0], 4, 8, manifest))
    for name in ("forward_costs", "backward_costs", "cost_profile_path"):
        payload.pop(name)
    payload.update(
        cost_profile_hash=profile["cost_profile_hash"],
        cost_profile_version=profile["schema_version"],
        cost_model={"path": str(profile_path)},
    )
    write_json(path, payload)
    assert validate_target_plan(path, profile_path, manifest_path, 1024).num_microbatches == 8
    with pytest.raises(ValueError, match="precision/sequence"):
        validate_target_plan(path, profile_path, manifest_path, 2048)
    payload["model_manifest_hash"] = "wrong"
    write_json(path, payload)
    with pytest.raises(ValueError, match="manifest_hash"):
        validate_target_plan(path, profile_path, manifest_path, 1024)
    profile["model_config"]["dtype"] = "fp32"
    write_json(profile_path, profile)
    with pytest.raises(ValueError, match="fingerprint"):
        validate_target_plan(path, profile_path, manifest_path, 1024)


def test_preflight_records_two_gpu_rejection(tmp_path, monkeypatch):
    # CPU-only simulation avoids device allocation while preserving fail-closed output.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="preflight failed"):
        preflight(tmp_path, "nccl-p2p", 35)
    metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    assert "requires at least 4 CUDA GPUs" in metadata["errors"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_runtime_dtype_and_profile_validation(tmp_path, dtype):
    config = tiny_config(dtype=dtype)
    _validate_unsupported_features(config, False)
    manifest = build_model_manifest(config)
    plan = construction_plan((0, 2, 5, 9, 12), 2, 8, manifest)
    profile = dict(
        schema_version="slackpipe.cost_profile.v2",
        model_manifest_hash=manifest["manifest_hash"],
        model_config={"dtype": "bf16" if dtype == torch.bfloat16 else "fp32"},
    )
    profile["cost_profile_hash"] = profile_fingerprint(profile)
    path = tmp_path / "profile.json"
    write_json(path, profile)
    linked = replace(
        plan,
        cost_profile_hash=profile["cost_profile_hash"],
        cost_profile_path=str(path),
        cost_profile_version=profile["schema_version"],
    )
    validate_plan_model(linked, config)
    config.params_dtype = torch.float32 if dtype == torch.bfloat16 else torch.bfloat16
    with pytest.raises(ValueError, match="dtype"):
        validate_plan_model(linked, config)
    with pytest.raises(ValueError, match="dtype"):
        _validate_unsupported_features(config, False)
    config.params_dtype = torch.float16
    with pytest.raises(ValueError, match="FP32 and BF16"):
        _validate_unsupported_features(config, False)
