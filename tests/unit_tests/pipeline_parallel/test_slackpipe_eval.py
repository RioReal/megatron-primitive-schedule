# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import json
from pathlib import Path

import pytest

from tools.run_slackpipe_eval import benchmark_summary, memory_preflight, receipt_valid
from tools.run_slackpipe_real_system_campaign import rotated_orders
from tools.slackpipe_eval_config import (
    fingerprint,
    load_model,
    parameter_breakdown,
    schedule_topology,
    transformer_config,
    validate_model,
)
from tools.slackpipe_eval_tables import benefits, export_tables, latex_model_table
from tools.slackpipe_eval_worker import validate_profile
from tools.slackpipe_nemotron_worker import construction_plan, uniform_cuts

CONFIGS = Path(__file__).resolve().parents[3] / "configs/slackpipe_eval"


def tiny_model(family):
    model = load_model(CONFIGS / f"{family}_4b.json")
    model.update(
        name=f"test_{family}",
        scale_label="fixture",
        official_or_scaled="scaled_research",
        num_layers=12 if family == "nemotron_h" else 8,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=4,
        num_query_groups=2,
        vocab_size=128,
        max_sequence_length=128,
        defaults=dict(seq_length=32, micro_batch_size=1, global_batch_size=4, precision="fp32"),
    )
    if family == "nemotron_h":
        model.update(
            hybrid_pattern="M*-M-M*-M-M-",
            mamba_state_dim=16,
            mamba_head_dim=32,
            mamba_num_heads=8,
            mamba_num_groups=1,
        )
    return validate_model(model)


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.json")), ids=lambda p: p.stem)
def test_all_configs(path):
    model = load_model(path)
    counts = parameter_breakdown(model)
    assert abs(counts["relative_difference"]) < 0.06
    assert (
        counts["exact_parameter_count"]
        == sum(counts["layers"]) + counts["embedding"] + counts["output"] + counts["final_norm"]
    )
    assert (model["official_or_scaled"] == "official_architecture") == (
        model["name"] == "nemotron_h_8b"
    )
    assert model["architecture_source"] and model["scaling_rule"]


def test_schema_rejects_missing_and_mislabelled():
    model = tiny_model("llama")
    del model["activation"]
    with pytest.raises(ValueError, match="Missing"):
        validate_model(model)
    model = tiny_model("nemotron_h")
    model["official_or_scaled"] = "official_architecture"
    with pytest.raises(ValueError, match="authoritative"):
        validate_model(model)


@pytest.mark.parametrize("b", [4, 8, 16])
@pytest.mark.parametrize("n", [4, 8, 12])
def test_generic_topology_and_coverage(b, n):
    topology = schedule_topology("slackpipe", 4, n, b)
    assert topology["vpp"] == (None if n == 4 else n // 4)
    model = load_model(CONFIGS / "llama_4b.json")
    from megatron.core.pipeline_parallel.slackpipe.manifest import build_model_manifest

    manifest = build_model_manifest(transformer_config(model, topology, "fp32"))
    plan = construction_plan(uniform_cuts(model["num_layers"], n), 4, b, manifest)
    assert sum(len(plan.worker_operations(r)) for r in range(4)) == 2 * b * n
    assert plan.stage_to_worker == tuple(s % 4 for s in range(n))


def test_schedule_routing():
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    assert (
        get_forward_backward_func(pp_size=4, vp_size=None).__name__
        == "forward_backward_pipelining_without_interleaving"
    )
    assert (
        get_forward_backward_func(pp_size=4, vp_size=2).__name__
        == "forward_backward_pipelining_with_interleaving"
    )
    assert (
        get_forward_backward_func(
            pipeline_schedule="slackpipe", slackpipe_plan_path="unused"
        ).func.__name__
        == "forward_backward_slackpipe"
    )
    with pytest.raises(ValueError):
        schedule_topology("1f1b", 4, 8, 8)
    with pytest.raises(ValueError):
        schedule_topology("slackpipe", 4, 10, 8)


def test_receipts_and_campaign_order(tmp_path):
    from tools.run_slackpipe_nemotron_h8b_pp4 import digest

    p = tmp_path / "artifact.json"
    p.write_text("{}")
    receipt = dict(context={"seed": 1}, artifacts={p.name: digest(p)})
    assert receipt_valid(receipt, {"seed": 1}, tmp_path)
    receipt["context"]["nccl"] = [2, 29, 7]
    assert receipt_valid(receipt, {"seed": 1, "nccl": (2, 29, 7)}, tmp_path)
    del receipt["context"]["nccl"]
    assert not receipt_valid(receipt, {"seed": 2}, tmp_path)
    p.write_text("changed")
    assert not receipt_valid(receipt, {"seed": 1}, tmp_path)
    orders = rotated_orders(["1f1b", "interleaved", "slackpipe"], 3, 10)
    assert orders == rotated_orders(["1f1b", "interleaved", "slackpipe"], 3, 10)
    assert len({o[0] for o in orders}) == 3


def test_memory_gate_does_not_shrink():
    model = load_model(CONFIGS / "llama_30b.json")
    original = copy.deepcopy(model)
    report = memory_preflight(
        model,
        schedule_topology("slackpipe", 4, 8, 8),
        "bf16",
        1024,
        1,
        [dict(free_bytes=6 * 1024**3)] * 4,
    )
    assert report["status"] == "skipped_memory_capacity"
    assert model == original


def test_profile_rejects_mismatch():
    from megatron.core.pipeline_parallel.slackpipe.cost_profile import profile_fingerprint

    model = tiny_model("llama")
    topology = schedule_topology("interleaved", 2, 4, 4)
    profile = dict(
        model_config=dict(
            model_config_hash=fingerprint(model),
            dtype="fp32",
            sequence_length=32,
            micro_batch_size=1,
        ),
        parallel_config={k: topology[k] for k in ("pp", "vpp", "tp", "dp", "cp")},
    )
    profile["cost_profile_hash"] = profile_fingerprint(profile)
    validate_profile(profile, model, topology, "fp32", 32, 1)
    for precision, seq, micro in (("bf16", 32, 1), ("fp32", 64, 1), ("fp32", 32, 2)):
        with pytest.raises(ValueError, match="mismatch"):
            validate_profile(profile, model, topology, precision, seq, micro)


def test_v1_binding_rejects_changed_profile(tmp_path):
    from tools.run_slackpipe_nemotron_h8b_pp4 import digest
    from tools.slackpipe_eval_worker import validate_plan_provenance

    profile = tmp_path / "profile.json"
    profile.write_text("{}")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(dict(schema_version="slackpipe.plan.v1", cost_model=dict(path=str(profile))))
    )
    plan.with_suffix(".binding.json").write_text(
        json.dumps(
            dict(
                schema_version="slackpipe.eval_plan_binding.v1",
                plan_sha256=digest(plan),
                profile_sha256=digest(profile),
            )
        )
    )
    validate_plan_provenance(plan, profile, {})
    profile.write_text('{"different":true}')
    with pytest.raises(ValueError, match="binding"):
        validate_plan_provenance(plan, profile, {})


def test_experiment_resume_never_launches_matching_stage(tmp_path, monkeypatch):
    from tools.run_slackpipe_eval import Experiment, argument_parser
    from tools.run_slackpipe_nemotron_h8b_pp4 import digest

    config = tmp_path / "config.json"
    config.write_text(json.dumps(tiny_model("llama")))
    args = argument_parser().parse_args(
        [
            "env",
            "--model-config",
            str(config),
            "--output",
            str(tmp_path),
            "--schedule",
            "1f1b",
            "--pp",
            "2",
            "--resume",
        ]
    )
    exp = Experiment(args)
    monkeypatch.setattr(exp, "_execute", lambda *args: pytest.fail("resume launched a subprocess"))
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    context = exp._context("env", {})
    receipt = dict(
        schema_version="slackpipe.eval_receipt.v2",
        status="passed",
        stage="env",
        context=context,
        context_hash=fingerprint(context),
        artifacts={"config.json": digest(config)},
    )
    (receipts / "1f1b.env.json").write_text(json.dumps(receipt))
    assert fingerprint(exp.ensure("env")) == fingerprint(receipt)
    exp.completed.clear()
    config.write_text("changed")
    with pytest.raises(RuntimeError, match="stale"):
        exp.ensure("env")


def test_three_panel_plot_and_capture_policy(tmp_path):
    from megatron.core.pipeline_parallel.slackpipe.figure_trace import write_compact_trace
    from tests.unit_tests.pipeline_parallel.test_slackpipe_figure_trace import compact
    from tools.plot_schedule_trace import load_panel, render

    for rank in range(2):
        write_compact_trace(tmp_path / f"rank{rank}_trace.json", compact(rank))
    panel = load_panel(tmp_path, 5)
    panel["config"].update(
        num_layers=8,
        hidden_size=128,
        heads=4,
        seq_length=32,
        micro_batch_size=1,
        vocab_size=128,
        seed=1234,
        learning_rate=0.001,
        dtype="fp32",
        tf32=False,
        dropout=0,
        tp=1,
        dp=1,
        cp=1,
        model="fixture",
        model_sha256="same",
    )
    panel["collection"] = dict(
        profiler={},
        active_iterations=[5],
        cycle=0,
        environment={
            k: None for k in ("allocator_environment", "allocator_backend", "torch", "cuda", "nccl")
        },
    )
    native, inter, slack = [copy.deepcopy(panel) for _ in range(3)]
    native["mode"], inter["mode"], slack["mode"] = "1f1b", "interleaved", "slackpipe"
    inter["config"]["num_stages"] = slack["config"]["num_stages"] = 4
    options = dict(
        png=tmp_path / "plot.png",
        pdf=tmp_path / "plot.pdf",
        title="fixture",
        caption="unlabeled gaps are not proven idle",
        color_mode="microbatch",
        annotate_saved=False,
    )
    report = render(inter, slack, noninterleaved=native, **options)
    assert report["noninterleaved"]["iteration"] == 5
    assert (tmp_path / "plot.pdf").stat().st_size > 0
    native["collection"]["cycle"] = 1
    with pytest.raises(ValueError, match="capture"):
        render(inter, slack, noninterleaved=native, **options)


def test_tables_are_from_configs(tmp_path):
    paths = sorted(CONFIGS.glob("*.json"))
    table = latex_model_table([load_model(p) for p in paths])
    assert "Homo." in table and "Hetero." in table
    assert "131072" in table and "21504" not in table  # column is FFN type, not width
    export_tables(tmp_path, paths)
    assert len(json.loads((tmp_path / "model_configs.json").read_text())) == 8
    assert (tmp_path / "memory_summary.csv").is_file()
    assert benefits({"1f1b": 10, "interleaved": 8, "slackpipe": 6}) == dict(
        slackpipe_vs_1f1b=0.4, interleaving_benefit=0.2, slackpipe_vs_interleaved=0.25
    )


def test_benchmark_summary_raw_agreement():
    def result(rank, values):
        return dict(
            rank=rank,
            run_id="r",
            method="1f1b",
            options={},
            config={"pp": 2},
            samples=[dict(iteration=i + 5, cuda_elapsed_ms=v) for i, v in enumerate(values)],
            continuous_wall_ms=40,
            memory_boundaries={"end": dict(peak_allocated_bytes=100, peak_reserved_bytes=200)},
        )

    summary = benchmark_summary([[result(0, [10, 20]), result(1, [12, 18])]], 8, 1024)
    assert summary["all_samples_by_run"] == [[12, 20]]
    assert summary["mean_ms"] == 16 and summary["ci95_ms"] is None
    assert summary["tokens_per_second"] == 8 * 1024 * 1000 / 20


@pytest.mark.parametrize("family", ["llama", "nemotron_h"])
def test_exact_count_matches_instantiated_model(family):
    import os

    import torch

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("parameter-count reference construction requires one rank")
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from tests.unit_tests.pipeline_parallel.test_slackpipe_model_construction import _build_model
    from tests.unit_tests.test_utilities import Utils

    model = tiny_model(family)
    config = transformer_config(model, schedule_topology("1f1b", 1, 1, 4), "fp32")
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    def provider(
        pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None
    ):
        common = dict(
            config=config,
            vocab_size=model["vocab_size"],
            max_sequence_length=32,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
            pg_collection=pg_collection,
            share_embeddings_and_output_weights=False,
        )
        if family == "llama":
            return GPTModel(
                **common,
                transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
                position_embedding_type="rope",
            )
        return HybridModel(
            **common,
            hybrid_stack_spec=hybrid_stack_spec,
            hybrid_layer_pattern=model["hybrid_pattern"],
            position_embedding_type="none",
        )

    try:
        chunks = _build_model(config, pp_size=1, provider=provider)
        actual = sum(p.numel() for c in chunks for p in c.parameters())
        assert actual == parameter_breakdown(model)["exact_parameter_count"]
    finally:
        Utils.destroy_model_parallel()
