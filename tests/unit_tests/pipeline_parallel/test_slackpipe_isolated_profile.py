# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Isolated class costs, quality, receipts and actual no-communication autograd."""

import copy
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.slackpipe.cost_profile import profile_fingerprint
from megatron.core.pipeline_parallel.slackpipe.isolated_profile import (
    ESTIMATOR,
    LEGACY_ESTIMATOR,
    IsolatedPolicy,
    build_profile,
    class_mapping,
    summarize,
)
from megatron.core.pipeline_parallel.slackpipe.manifest import manifest_fingerprint
from megatron.core.pipeline_parallel.slackpipe.profile_quality import (
    ProfilingQualityError,
    require_profile_quality,
)
from tools.run_slackpipe_eval import Experiment, argument_parser
from tools.slackpipe_eval_config import transformer_config
from tools.slackpipe_isolated_profile import make_unit, no_communication
from tools.slackpipe_profile_quality import collect_with_retry


def fixture_profile(
    kinds=("mamba", "mlp", "mamba", "attention"),
    unstable=False,
    manifest=None,
    execution=None,
    warmups=20,
    iterations=30,
    max_cv=0.15,
):
    manifest = manifest or dict(
        schema_version="slackpipe.model_manifest.v1",
        num_layers=len(kinds),
        model_config=dict(hidden_size=64),
        layers=[dict(layer_id=i, layer_type=k, config_class=k) for i, k in enumerate(kinds)],
    )
    manifest["manifest_hash"] = manifest_fingerprint(manifest)
    policy = IsolatedPolicy(max_cv=max_cv)
    context = dict(
        execution=execution or dict(precision="fp32", hidden_size=64),
        policy=asdict(policy),
        warmups=warmups,
        iterations=iterations,
    )
    classes = class_mapping(manifest, context["execution"])
    units = []
    for i, key in enumerate([c["class_signature"] for c in classes] + ["first", "last"]):
        f = [float(i + 1)] * iterations
        b = [float(11 + i * i)] * iterations
        if unstable:
            b = [1.0 if j % 2 == 0 else 5.0 for j in range(iterations)]
        units.append(
            dict(
                class_signature=key,
                forward_gpu_ms=f,
                backward_gpu_ms=b,
                samples=[
                    dict(iteration=warmups + j, forward_gpu_ms=x, backward_gpu_ms=y)
                    for j, (x, y) in enumerate(zip(f, b))
                ],
            )
        )
    return build_profile(
        manifest=manifest,
        classes=classes,
        units=units,
        context=context,
        model_config={},
        parallel_config=dict(pp=2, vpp=2, tp=1, cp=1, dp=1),
        policy=policy,
    )


def test_homogeneous_and_heterogeneous_expansion():
    homogeneous = fixture_profile(("decoder",) * 8)
    assert len(homogeneous["class_costs_us"]) == 1
    assert homogeneous["prefix_forward_us"] == [1000.0 * i for i in range(9)]
    p = fixture_profile()
    assert [x["forward_us"] for x in p["layer_costs_us"]] == [1000, 2000, 1000, 3000]
    assert [x["backward_us"] for x in p["layer_costs_us"]] == [11000, 12000, 11000, 15000]
    for i, j in ((0, 1), (0, 4), (1, 3), (2, 4)):
        for phase in ("forward", "backward"):
            assert p[f"prefix_{phase}_us"][j] - p[f"prefix_{phase}_us"][i] == sum(
                x[phase + "_us"] for x in p["layer_costs_us"][i:j]
            )
    assert p["stage_role_bias_us"]["middle"] == dict(forward=0, backward=0)
    assert p["stage_role_bias_us"]["first"]["forward"] == 4000
    assert p["stage_role_bias_us"]["last"]["forward"] == 5000
    require_profile_quality(json.loads(json.dumps(p)))


def test_signature_is_complete_stable_and_instance_sensitive():
    p = fixture_profile(("decoder",) * 4)
    manifest = dict(p["isolated"]["manifest"], schema_version="slackpipe.model_manifest.v1")
    execution = p["isolated"]["context"]["execution"]
    first = class_mapping(manifest, execution)
    assert class_mapping(copy.deepcopy(manifest), copy.deepcopy(execution)) == first
    assert class_mapping(manifest, dict(execution, precision="bf16")) != first
    manifest["layers"][2]["block_config"] = dict(ffn_hidden_size=128)
    manifest["manifest_hash"] = manifest_fingerprint(manifest)
    assert len(class_mapping(manifest, execution)) == 2


def test_single_stage_charges_both_boundaries_once():
    p = fixture_profile()
    data = p["isolated"]
    single = build_profile(
        manifest=dict(data["manifest"], schema_version="slackpipe.model_manifest.v1"),
        classes=data["classes"],
        units=data["units"],
        context=data["context"],
        model_config={},
        parallel_config=dict(pp=1, vpp=None),
        policy=IsolatedPolicy(),
    )
    assert single["stage_role_bias_us"]["first"]["forward"] == 9000
    assert single["prefix_forward_us"] == p["prefix_forward_us"]
    require_profile_quality(single)


def test_optional_pipeline_comparison_is_not_an_equality_gate():
    from tools.slackpipe_compare_isolated import compare

    p = fixture_profile()
    pipeline = dict(
        manifest=p["model_manifest_hash"],
        metadata={},
        events=[
            dict(
                calibration_group_id=f"partition{g}",
                worker=0,
                logical_stage=0,
                stage_layer_range=[0, 2],
                stage_role="middle",
                phase="backward_compute",
                iteration=i,
                elapsed_ms=value,
            )
            for g, value in enumerate((46, 92))
            for i in range(3)
        ],
    )
    rows = compare(p, pipeline)
    assert [r["isolated_layers_ms"] for r in rows] == [23, 23]
    assert [r["pipeline_wall_ms"] for r in rows] == [46, 92]
    assert all(len(r["iteration_means_ms"]) == 3 for r in rows)
    pipeline["manifest"] = "different"
    with pytest.raises(ValueError, match="manifest"):
        compare(p, pipeline)


def test_quality_retains_raw_catastrophe_and_rejects_instability():
    p = IsolatedPolicy()
    assert summarize([1.0] * 30, p)["passed"]
    report = summarize([1.0] * 29 + [100.0], p)
    assert report["passed"] and report["excluded_indices"] == [29]
    assert report["raw"]["max_ms"] == 100 and report["accepted"]["count"] == 29
    assert not summarize([1.0, 3.0] * 15, p)["passed"]
    assert not summarize([1.0] * 8, p)["passed"]
    with pytest.raises(ProfilingQualityError):
        require_profile_quality(fixture_profile(unstable=True))


@pytest.mark.parametrize("corruption", ["prefix", "raw", "policy", "class", "summary"])
def test_quality_gate_recomputes_not_just_pass_flag(corruption):
    p = fixture_profile()
    if corruption == "prefix":
        p["prefix_backward_us"][2] += 100
    if corruption == "raw":
        p["isolated"]["units"][0]["samples"][0]["forward_gpu_ms"] += 1
    if corruption == "policy":
        p["quality"]["thresholds"]["max_cv"] = 0.9
    if corruption == "class":
        p["isolated"]["classes"][0]["member_layers"] = [0]
    if corruption == "summary":
        p["isolated"]["summaries"]["first"]["forward"]["selected_ms"] += 1
    p["cost_profile_hash"] = profile_fingerprint(p)
    with pytest.raises(ProfilingQualityError):
        require_profile_quality(p)


@pytest.mark.parametrize("second_passes", [True, False])
def test_retry_at_most_once_and_no_solver_profile_after_failure(tmp_path, second_passes):
    calls = []

    def collect(folder, attempt):
        calls.append(attempt)
        folder.mkdir()
        p = fixture_profile(unstable=not (attempt == 1 and second_passes))
        for name, data in (
            ("cost_profile.candidate.json", p),
            ("cost_profile.quality.json", p["quality"]),
        ):
            (folder / name).write_text(json.dumps(data))
        if p["quality"]["status"] == "passed":
            (folder / "cost_profile.json").write_text(json.dumps(p))

    if second_passes:
        path = collect_with_retry(tmp_path, collect, dict(policy=dict(estimator=ESTIMATOR)))
        assert "attempt1" in str(path)
    else:
        with pytest.raises(ProfilingQualityError):
            collect_with_retry(tmp_path, collect, {})
        assert not list(tmp_path.glob("attempt*/worker/cost_profile.json"))
    assert calls == [0, 1]


def test_estimator_policy_context_scoped_to_calibration(tmp_path, monkeypatch):
    config = Path("configs/slackpipe_eval/llama_8b.json")
    args = argument_parser().parse_args(
        ["calibrate", "--model-config", str(config), "--output", str(tmp_path)]
    )
    monkeypatch.setattr(Experiment, "_base_context", lambda self: {})
    exp = Experiment(args)
    first = exp._context("calibrate", {})
    benchmark = exp._context("benchmark", {})
    assert first["policy"]["estimator"] == ESTIMATOR
    assert first == exp._context("calibrate", {})  # Compatible resume identity.
    exp.args.isolated_profile_iterations += 1
    assert exp._context("calibrate", {}) != first
    assert exp._context("benchmark", {}) == benchmark
    exp.args.calibration_estimator = LEGACY_ESTIMATOR
    assert exp._context("calibrate", {})["policy"]["estimator"] == LEGACY_ESTIMATOR


def test_receipt_rejects_wrong_sample_count(tmp_path):
    def collect(folder, attempt):
        folder.mkdir()
        p = fixture_profile()
        for name, value in (
            ("cost_profile", p),
            ("cost_profile.candidate", p),
            ("cost_profile.quality", p["quality"]),
        ):
            (folder / (name + ".json")).write_text(json.dumps(value))

    with pytest.raises(ProfilingQualityError, match="iterations mismatch"):
        collect_with_retry(tmp_path, collect, dict(policy=dict(estimator=ESTIMATOR, iterations=40)))


def test_isolated_worker_dispatch_is_single_process(tmp_path, monkeypatch):
    args = argument_parser().parse_args(
        [
            "calibrate",
            "--model-config",
            "configs/slackpipe_eval/llama_8b.json",
            "--output",
            str(tmp_path),
        ]
    )
    exp = Experiment(args)
    commands = []
    monkeypatch.setattr(exp, "_execute", lambda command, *unused: commands.append(command))
    exp._worker("calibrate", tmp_path / "attempt0" / "worker")
    assert commands[-1][1:3] == ["-m", "tools.slackpipe_isolated_profile"]
    assert "torch.distributed.run" not in commands[-1]
    exp.args.calibration_estimator = LEGACY_ESTIMATOR
    exp._worker("calibrate", tmp_path / "legacy" / "worker")
    assert "torch.distributed.run" in commands[-1]
    assert "tools.slackpipe_eval_worker" in commands[-1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_actual_units_finite_repeatable_and_no_pipeline(tmp_path):
    from unittest.mock import patch

    from megatron.core.pipeline_parallel import schedules
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model = json.loads(Path("configs/slackpipe_eval/nemotron_h_4b.json").read_text())
    model.update(
        num_layers=6,
        hidden_size=64,
        ffn_hidden_size=128,
        num_attention_heads=4,
        num_query_groups=2,
        vocab_size=128,
        hybrid_pattern="M*-M*-",
        mamba_state_dim=16,
        mamba_head_dim=16,
        mamba_num_groups=1,
        mamba_num_heads=8,
    )
    assert not dist.is_initialized()
    dist.init_process_group(
        "gloo", init_method="file://" + str(tmp_path / "store"), rank=0, world_size=1
    )
    try:
        parallel_state.initialize_model_parallel()
        config = transformer_config(model, dict(pp=1, vpp=None), "fp32")
        model_parallel_cuda_manual_seed(1234)
        with (
            patch.object(
                schedules, "get_forward_backward_func", side_effect=AssertionError("No pipeline")
            ),
            no_communication(),
        ):
            for kind in ("mamba", "attention", "mlp", "decoder", "first", "last"):
                if kind == "decoder":
                    model = dict(model, model_family="llama", position_embedding="rope")
                    config = transformer_config(
                        {k: v for k, v in model.items() if not k.startswith("mamba_")},
                        dict(pp=1, vpp=None),
                        "fp32",
                    )
                unit = dict(class_signature=kind, layer_type=kind, representative_layer=0)
                module, hidden, forward = make_unit(model, config, unit, 32, 1)
                out = forward()
                grad = torch.ones_like(out)
                out.backward(grad)
                reference = out.detach().clone()
                grads = [p.grad.clone() for p in module.parameters()]
                module.zero_grad(set_to_none=True)
                if hidden is not None:
                    hidden.grad = None
                out = forward()
                out.backward(grad)
                torch.testing.assert_close(out, reference, rtol=0, atol=0)
                for p, g in zip(module.parameters(), grads):
                    assert torch.isfinite(p.grad).all()
                    torch.testing.assert_close(p.grad, g, rtol=0, atol=0)
                del module, hidden, forward, out, reference, grads, grad
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
