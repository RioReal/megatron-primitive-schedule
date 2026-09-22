# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Strict research model specifications and allocation-free exact shape counts.

The count mirrors GPT TE and hybrid_stack_spec at TP=1, untied embeddings,
RMSNorm, bias-free projections; Mamba retains its depthwise convolution bias.
Unsupported architecture options are rejected rather than approximately counted.
"""

import hashlib
import json
from pathlib import Path

SCHEMA = "slackpipe.eval_model.v1"
SCHEDULES = ("1f1b", "interleaved", "slackpipe", "optimized_interleaved")


def fingerprint(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_model(value: dict) -> dict:
    required = (
        "schema_version name model_family scale_label official_or_scaled architecture_source "
        "num_layers hidden_size ffn_hidden_size num_attention_heads num_query_groups vocab_size "
        "max_sequence_length normalization activation position_embedding attention_type ffn_type "
        "expected_parameter_scale defaults scaling_rule"
    ).split()
    missing = sorted(set(required) - value.keys())
    if missing:
        raise ValueError(f"Missing model fields: {missing}")
    if value["schema_version"] != SCHEMA:
        raise ValueError("Unsupported model schema")
    for key in required:
        if value[key] is None or value[key] == "":
            raise ValueError(f"Empty model field: {key}")
    for key in (
        "num_layers",
        "hidden_size",
        "ffn_hidden_size",
        "num_attention_heads",
        "num_query_groups",
        "vocab_size",
        "max_sequence_length",
        "expected_parameter_scale",
    ):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(f"Expected positive integer: {key}")
    h, heads, groups = (
        value[k] for k in ("hidden_size", "num_attention_heads", "num_query_groups")
    )
    if h % heads or heads % groups:
        raise ValueError("Incompatible attention dimensions")
    if value["normalization"] != "RMSNorm" or value["attention_type"] != "GQA":
        raise ValueError("Only bias-free RMSNorm/GQA adapters are defined")
    family = value["model_family"]
    expected = {
        "llama": ("silu", "rope", "SwiGLU"),
        "nemotron_h": ("squared_relu", "none", "ReLU2"),
    }
    if (
        family not in expected
        or tuple(value[k] for k in ("activation", "position_embedding", "ffn_type"))
        != expected[family]
    ):
        raise ValueError("Unsupported model family/activation/position/FFN combination")
    if value["official_or_scaled"] not in ("scaled_research", "official_architecture"):
        raise ValueError("Invalid official/scaled designation")
    if family == "llama" and value["official_or_scaled"] != "scaled_research":
        raise ValueError("LLaMA-style presets are research architectures, not released checkpoints")
    if family == "nemotron_h":
        pattern = value.get("hybrid_pattern", "")
        if len(pattern) != value["num_layers"] or set(pattern) != set("M*-"):
            raise ValueError("Hybrid sequence must specify every M/attention/MLP global block")
        for key in ("mamba_state_dim", "mamba_head_dim", "mamba_num_groups", "mamba_num_heads"):
            if type(value.get(key)) is not int or value[key] <= 0:
                raise ValueError(f"Missing/invalid {key}")
        if value["mamba_num_heads"] % value["mamba_num_groups"]:
            raise ValueError("Mamba heads must be divisible by groups")
        if value["official_or_scaled"] == "official_architecture":
            from megatron.core.pipeline_parallel.slackpipe.hybrid import (
                NEMOTRON_H_8B_PATTERN,
                NEMOTRON_H_8B_SOURCE,
            )

            exact = dict(
                num_layers=52,
                hidden_size=4096,
                ffn_hidden_size=21504,
                num_attention_heads=32,
                num_query_groups=8,
                vocab_size=131072,
                max_sequence_length=8192,
                mamba_state_dim=128,
                mamba_head_dim=64,
                mamba_num_groups=8,
                mamba_num_heads=128,
                hybrid_pattern=NEMOTRON_H_8B_PATTERN,
                architecture_source=NEMOTRON_H_8B_SOURCE,
            )
            if any(value[k] != v for k, v in exact.items()):
                raise ValueError(
                    "Official designation requires exact authoritative Base-8K architecture"
                )
    defaults = value["defaults"]
    for key in ("seq_length", "micro_batch_size", "global_batch_size"):
        if type(defaults.get(key)) is not int or defaults[key] < 1:
            raise ValueError(f"Invalid default {key}")
    if defaults["seq_length"] > value["max_sequence_length"] or defaults.get("precision") not in (
        "fp32",
        "bf16",
    ):
        raise ValueError("Invalid default sequence/precision")
    return value


def load_model(path: Path) -> dict:
    return validate_model(json.loads(Path(path).read_text()))


def parameter_breakdown(model: dict) -> dict:
    """Count all trainable scalar elements, including norms and Mamba vectors."""
    validate_model(model)
    h, f = model["hidden_size"], model["ffn_hidden_size"]
    heads, groups = model["num_attention_heads"], model["num_query_groups"]
    attention = h * (h + 2 * groups * (h // heads)) + h * h + h
    mlp = h * f * (3 if model["ffn_type"] == "SwiGLU" else 2) + h
    if model["model_family"] == "llama":
        layers = [attention + mlp] * model["num_layers"]
    else:
        n = model["mamba_num_heads"]
        inner = n * model["mamba_head_dim"]
        conv = inner + 2 * model["mamba_num_groups"] * model["mamba_state_dim"]
        mamba = h * (inner + conv + n) + h + conv * 5 + 3 * n + inner + inner * h
        layers = [{"M": mamba, "*": attention, "-": mlp}[c] for c in model["hybrid_pattern"]]
    embedding = h * model["vocab_size"]
    total = 2 * embedding + h + sum(layers)
    target = model["expected_parameter_scale"]
    return dict(
        exact_parameter_count=total,
        target_parameter_count=target,
        difference_from_target=total - target,
        relative_difference=(total - target) / target,
        embedding=embedding,
        output=embedding,
        final_norm=h,
        layers=layers,
    )


def schedule_topology(schedule: str, pp: int, stages: int | None, microbatches: int) -> dict:
    if schedule not in SCHEDULES or pp not in (1, 2, 4) or microbatches < 1:
        raise ValueError("Unsupported schedule, PP or B")
    stages = stages if stages is not None else (pp if schedule == "1f1b" else 2 * pp)
    if stages < pp or stages % pp:
        raise ValueError("Cyclic placement requires N % PP == 0")
    if schedule == "1f1b" and stages != pp:
        raise ValueError("Native 1F1B requires N=PP and VPP=None")
    if schedule in ("interleaved", "optimized_interleaved") and (stages == pp or pp == 1):
        raise ValueError("Native interleaving requires physical PP>1 and N>PP")
    if schedule in ("interleaved", "optimized_interleaved") and microbatches % pp:
        raise ValueError("Native interleaving requires B divisible by PP")
    return dict(
        pp=pp,
        num_stages=stages,
        vpp=None if stages == pp else stages // pp,
        num_microbatches=microbatches,
        tp=1,
        dp=1,
        cp=1,
    )


def transformer_config(model: dict, topology: dict, precision: str, layout=None):
    import torch

    from megatron.core.activations import squared_relu
    from megatron.core.pipeline_parallel.slackpipe.hybrid import bind_hybrid_config
    from megatron.core.transformer.transformer_config import TransformerConfig

    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    config = TransformerConfig(
        **{
            k: model[k]
            for k in (
                "num_layers",
                "hidden_size",
                "ffn_hidden_size",
                "num_attention_heads",
                "num_query_groups",
            )
        },
        **{k: v for k, v in model.items() if k.startswith("mamba_")},
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        add_bias_linear=False,
        activation_func=(
            torch.nn.functional.silu if model["model_family"] == "llama" else squared_relu
        ),
        gated_linear_unit=model["model_family"] == "llama",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        pipeline_model_parallel_size=topology["pp"],
        virtual_pipeline_model_parallel_size=topology["vpp"],
        pipeline_model_parallel_layout=layout,
        params_dtype=dtype,
        pipeline_dtype=dtype,
        bf16=precision == "bf16",
        use_cpu_initialization=True,
        batch_p2p_comm=True,
        overlap_p2p_comm=False,
    )
    if model["model_family"] == "nemotron_h":
        bind_hybrid_config(config, model["hybrid_pattern"])
    return config
