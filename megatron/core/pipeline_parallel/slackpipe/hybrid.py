# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exact-range adapters for the pinned Megatron HybridModel implementation."""

from collections import Counter

from megatron.core.models.hybrid.hybrid_layer_allocation import parse_hybrid_pattern

HYBRID_TYPES = {"M": "mamba", "*": "attention", "-": "mlp"}


def global_hybrid_pattern(pattern):
    parsed = parse_hybrid_pattern(pattern)
    if parsed.mtp_pattern:
        raise ValueError("SlackPipe hybrid MTP is unsupported")
    sequence = parsed.main_pattern.replace("|", "")
    if not sequence or set(sequence) - HYBRID_TYPES.keys():
        raise ValueError("SlackPipe hybrid supports only Mamba, attention and MLP blocks")
    return sequence


def partition_hybrid_pattern(pattern, plan):
    sequence = global_hybrid_pattern(pattern)
    if len(sequence) != plan.num_layers:
        raise ValueError("SlackPipe hybrid pattern length must match plan num_layers")
    partitioned = "|".join(sequence[begin:end] for begin, end in plan.stage_layer_ranges)
    if "|" in pattern and pattern != partitioned:
        raise ValueError("Explicit hybrid pipeline segments do not match SlackPipe plan ranges")
    return partitioned


def stage_layer_types(plan, manifest, stage):
    return tuple(manifest["layers"][i]["layer_type"] for i in plan.stage_layer_ids(stage))


def stage_class_counts(plan, manifest, stage):
    return dict(Counter(manifest["layers"][i]["config_class"] for i in plan.stage_layer_ids(stage)))


def bind_hybrid_config(config, pattern):
    """Attach the authoritative unpartitioned sequence, independent of PP cuts."""
    sequence = global_hybrid_pattern(pattern)
    if len(sequence) != config.num_layers:
        raise ValueError("Hybrid config num_layers does not match its block sequence")
    config.slackpipe_hybrid_pattern = sequence
    return config


NEMOTRON_H_8B_SOURCE = (
    "https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K/blob/"
    "253e00241ff77421b6d811c971e9cee1b2d824ad/config.json"
)
NEMOTRON_H_8B_PATTERN = "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
NEMOTRON_H_8B_VOCAB_SIZE = 131072
NEMOTRON_H_8B_MAX_SEQUENCE_LENGTH = 8192


def nemotron_h_8b_config(**overrides):
    """Authoritative offline random-init architecture, not a weight loader.

    The released Base-8K config is the source of dimensions and block order.
    Pinned MambaMixer defaults supply conv=4, expand=2 and chunk_size=256.
    HybridModel must use position_embedding_type='none' and untied embeddings.
    """
    import torch

    from megatron.core.activations import squared_relu
    from megatron.core.transformer.transformer_config import TransformerConfig

    settings = dict(
        num_layers=52,
        hidden_size=4096,
        ffn_hidden_size=21504,
        num_attention_heads=32,
        num_query_groups=8,
        kv_channels=128,
        mamba_state_dim=128,
        mamba_head_dim=64,
        mamba_num_groups=8,
        mamba_num_heads=128,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        activation_func=squared_relu,
        gated_linear_unit=False,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        init_method_std=0.02,
        pipeline_dtype=torch.float32,
        params_dtype=torch.float32,
        use_cpu_initialization=True,
    )
    settings.update(overrides)
    return bind_hybrid_config(TransformerConfig(**settings), NEMOTRON_H_8B_PATTERN)
