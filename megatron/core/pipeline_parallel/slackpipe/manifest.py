# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""SlackPipe model manifest helpers."""

import hashlib
import json
from dataclasses import asdict
from typing import Mapping

SLACKPIPE_MODEL_MANIFEST_SCHEMA_VERSION = "slackpipe.model_manifest.v1"


def canonical_json(data: Mapping[str, object]) -> str:
    """Return deterministic JSON used for SlackPipe fingerprints."""

    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def manifest_fingerprint(manifest: Mapping[str, object]) -> str:
    """Hash a manifest after removing any existing hash field."""

    payload = dict(manifest)
    payload.pop("manifest_hash", None)
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def build_model_manifest(config) -> dict[str, object]:
    """Build a SlackPipe manifest from a Megatron transformer config.

    The authoritative layer identity is the zero-based global decoder layer ID.
    Heterogeneous configs expose ``per_block_parameters``; homogeneous configs
    fall back to a single repeated class.
    """

    num_layers = int(config.num_layers)
    layers = []
    pattern = getattr(config, "slackpipe_hybrid_pattern", None)
    if pattern is not None:
        from .hybrid import HYBRID_TYPES, global_hybrid_pattern

        sequence = global_hybrid_pattern(pattern)
        if len(sequence) != num_layers:
            raise ValueError("Hybrid manifest sequence length does not match num_layers")
        for layer_id, symbol in enumerate(sequence):
            layers.append(
                {
                    "layer_id": layer_id,
                    "global_layer_id": layer_id,
                    "layer_type": HYBRID_TYPES[symbol],
                    "config_class": HYBRID_TYPES[symbol],
                }
            )
    elif getattr(config, "heterogeneous_block_specs", False):
        block_configs = getattr(config, "per_block_parameters")
        if len(block_configs) != num_layers:
            raise ValueError(
                "heterogeneous config per_block_parameters must match num_layers "
                f"({len(block_configs)} != {num_layers})"
            )
        class_ids: dict[str, str] = {}
        for layer_id, block_config in enumerate(block_configs):
            block_payload = asdict(block_config)
            class_key = canonical_json(block_payload)
            class_id = class_ids.setdefault(class_key, f"class_{len(class_ids)}")
            layers.append(
                {
                    "layer_id": layer_id,
                    "layer_type": "decoder",
                    "config_class": class_id,
                    "block_config": block_payload,
                }
            )
    else:
        for layer_id in range(num_layers):
            layers.append(
                {
                    "layer_id": layer_id,
                    "layer_type": "decoder",
                    "config_class": "homogeneous_decoder",
                }
            )

    manifest = {
        "schema_version": SLACKPIPE_MODEL_MANIFEST_SCHEMA_VERSION,
        "num_layers": num_layers,
        "layers": layers,
        "model_config": {
            name: getattr(config, name)
            for name in (
                "hidden_size",
                "num_attention_heads",
                "num_query_groups",
                "kv_channels",
                "ffn_hidden_size",
                "normalization",
                "layernorm_epsilon",
                "gated_linear_unit",
                "add_bias_linear",
                "hidden_dropout",
                "attention_dropout",
            )
        },
    }
    if pattern is not None:
        manifest["model_config"].update(
            {
                name: getattr(config, name)
                for name in (
                    "mamba_state_dim",
                    "mamba_head_dim",
                    "mamba_num_groups",
                    "mamba_num_heads",
                )
            }
        )
        manifest["model_config"]["activation_func"] = config.activation_func.__name__
    manifest["manifest_hash"] = manifest_fingerprint(manifest)
    return manifest


def validate_plan_model(plan, config) -> None:
    """Validate v2 model/profile provenance before executing any operations."""
    if plan.schema_version != "slackpipe.plan.v2":
        return
    actual = build_model_manifest(config)
    if plan.model_manifest_hash and actual["manifest_hash"] != plan.model_manifest_hash:
        raise ValueError("SlackPipe model_manifest_hash does not match the actual Megatron config")
    if plan.cost_profile_hash:
        from pathlib import Path

        from .cost_profile import profile_fingerprint

        if not plan.cost_profile_path:
            raise ValueError("SlackPipe cost_profile_hash requires cost_model.path for validation")
        profile = json.loads(Path(plan.cost_profile_path).read_text(encoding="utf-8"))
        digest = profile_fingerprint(profile)
        if digest != plan.cost_profile_hash or profile.get("cost_profile_hash") != digest:
            raise ValueError("SlackPipe cost_profile_hash does not match the actual cost profile")
        if profile.get("model_manifest_hash") != actual["manifest_hash"]:
            raise ValueError("SlackPipe cost profile model_manifest_hash does not match the model")
        if profile.get("schema_version") != plan.cost_profile_version:
            raise ValueError("SlackPipe cost profile version does not match the plan")


def validate_chunk_layers(plan, chunks, worker: int) -> list[dict]:
    """Check exact global IDs and instantiated heterogeneous modules for TP=1."""
    from megatron.core.transformer.attention import SelfAttention
    from megatron.core.transformer.identity_op import IdentityOp
    from megatron.core.utils import unwrap_model

    chunks = unwrap_model(chunks)
    stages = [s for s, w in enumerate(plan.stage_to_worker) if w == worker]
    if len(stages) != len(chunks):
        raise ValueError("SlackPipe model chunk count does not match plan")
    records = []
    for stage, chunk in zip(stages, chunks):
        manifest = build_model_manifest(chunk.config)
        ids = [layer.layer_number - 1 for layer in chunk.decoder.layers]
        if ids != list(plan.stage_layer_ids(stage)):
            raise ValueError(f"SlackPipe stage {stage} global layer IDs do not match plan: {ids}")
        for layer, layer_id in zip(chunk.decoder.layers, ids):
            if getattr(chunk.config, "slackpipe_hybrid_pattern", None):
                from megatron.core.ssm.mamba_layer import MambaLayer
                from megatron.core.ssm.mlp_layer import MLPLayer
                from megatron.core.transformer.transformer_layer import TransformerLayer

                expected_type = manifest["layers"][layer_id]["layer_type"]
                classes = {"mamba": MambaLayer, "attention": TransformerLayer, "mlp": MLPLayer}
                if type(layer) is not classes[expected_type]:
                    raise ValueError(f"SlackPipe hybrid layer {layer_id} type mismatch")
                if expected_type == "attention" and (
                    not isinstance(layer.self_attention, SelfAttention)
                    or not isinstance(layer.mlp, IdentityOp)
                ):
                    raise ValueError(f"SlackPipe hybrid layer {layer_id} attention class mismatch")
                if layer.config is not chunk.config:
                    raise ValueError(f"SlackPipe hybrid layer {layer_id} configuration mismatch")
                config = chunk.config
                if expected_type == "mamba":
                    mixer = layer.mixer
                    heads = (
                        config.mamba_num_heads or 2 * config.hidden_size // config.mamba_head_dim
                    )
                    if (mixer.d_state, mixer.headdim, mixer.ngroups, mixer.nheads) != (
                        config.mamba_state_dim,
                        config.mamba_head_dim,
                        config.mamba_num_groups,
                        heads,
                    ):
                        raise ValueError(
                            f"SlackPipe hybrid layer {layer_id} Mamba configuration mismatch"
                        )
                elif expected_type == "mlp":
                    width = config.ffn_hidden_size * (2 if config.gated_linear_unit else 1)
                    if layer.mlp.linear_fc1.weight.shape != (width, config.hidden_size):
                        raise ValueError(
                            f"SlackPipe hybrid layer {layer_id} MLP configuration mismatch"
                        )
                else:
                    width = (
                        config.num_attention_heads + 2 * config.num_query_groups
                    ) * config.kv_channels
                    if layer.self_attention.linear_qkv.weight.shape != (width, config.hidden_size):
                        raise ValueError(
                            f"SlackPipe hybrid layer {layer_id} attention configuration mismatch"
                        )
            if getattr(chunk.config, "heterogeneous_block_specs", False):
                block = chunk.config.per_block_parameters[layer_id]
                if block.attention.no_op != isinstance(layer.self_attention, IdentityOp):
                    raise ValueError(f"SlackPipe layer {layer_id} attention class mismatch")
                if not block.attention.no_op and not block.attention.replace_with_linear:
                    if not isinstance(layer.self_attention, SelfAttention):
                        raise ValueError(f"SlackPipe layer {layer_id} attention module mismatch")
                    if layer.config.num_query_groups != block.attention.num_query_groups:
                        raise ValueError(
                            f"SlackPipe layer {layer_id} attention configuration mismatch"
                        )
                if block.mlp.ffn_hidden_size is not None:
                    width = block.mlp.ffn_hidden_size
                    multiplier = 2 if layer.config.gated_linear_unit else 1
                    if (
                        layer.config.ffn_hidden_size != width
                        or layer.mlp.linear_fc1.weight.shape[0] != width * multiplier
                    ):
                        raise ValueError(f"SlackPipe layer {layer_id} MLP configuration mismatch")
            records.append(
                {
                    "stage": stage,
                    "layer_id": layer_id,
                    "config_class": manifest["layers"][layer_id]["config_class"],
                    "layer_type": manifest["layers"][layer_id]["layer_type"],
                }
            )
    return records
