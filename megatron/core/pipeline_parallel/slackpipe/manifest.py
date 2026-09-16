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
    if getattr(config, "heterogeneous_block_specs", False):
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
                }
            )
    return records
