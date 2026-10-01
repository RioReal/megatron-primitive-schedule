# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Isolated measured-unit costs, deliberately independent of partition consensus."""

import hashlib
import math
import statistics
from dataclasses import asdict, dataclass

from .cost_profile import profile_fingerprint
from .manifest import canonical_json, manifest_fingerprint

ESTIMATOR = "isolated-layer-compute-v1"
LEGACY_ESTIMATOR = "existing-stage-wall-time-v1"
QUALITY_SCHEMA = "slackpipe.isolated_profile_quality.v1"


@dataclass(frozen=True)
class IsolatedPolicy:
    min_samples: int = 10
    max_cv: float = 0.15
    catastrophic_ratio: float = 10.0
    max_excluded: int = 1
    min_samples_for_exclusion: int = 30

    def __post_init__(self):
        if self.min_samples < 2 or not 0 < self.max_cv < 1 or self.catastrophic_ratio <= 1:
            raise ValueError("Invalid isolated quality thresholds")
        if self.max_excluded not in (0, 1) or self.min_samples_for_exclusion < 30:
            raise ValueError(
                "Isolated exclusion allows at most one catastrophic sample out of >=30"
            )


def class_mapping(manifest: dict, execution: dict) -> list[dict]:
    """Group only identical manifest units under the complete execution signature."""
    if manifest.get("manifest_hash") != manifest_fingerprint(manifest):
        raise ValueError("Invalid model manifest hash")
    classes = {}
    for i, layer in enumerate(manifest["layers"]):
        if layer["layer_id"] != i:
            raise ValueError("Manifest layers must be contiguous and ordered")
        signature = dict(
            execution=execution,
            manifest_model_config=manifest["model_config"],
            unit={k: v for k, v in layer.items() if k not in ("layer_id", "global_layer_id")},
        )
        key = hashlib.sha256(canonical_json(signature).encode()).hexdigest()
        classes.setdefault(
            key,
            dict(
                class_signature=key,
                signature=signature,
                layer_type=layer["layer_type"],
                representative_layer=i,
                member_layers=[],
            ),
        )["member_layers"].append(i)
    return list(classes.values())


def summarize(values: list[float], policy: IsolatedPolicy) -> dict:
    """Keep all samples; optionally exclude one >10x-median catastrophe for selection/CV."""
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Isolated timings must be positive finite samples")
    median = statistics.median(values)
    candidates = [i for i, v in enumerate(values) if v > policy.catastrophic_ratio * median]
    excluded = (
        candidates
        if len(values) >= policy.min_samples_for_exclusion
        and len(candidates) <= policy.max_excluded
        else []
    )
    accepted = [v for i, v in enumerate(values) if i not in excluded]

    def stats(v):
        mean = statistics.fmean(v)
        return dict(
            mean_ms=mean,
            median_ms=statistics.median(v),
            stddev_ms=statistics.pstdev(v),
            cv=statistics.pstdev(v) / mean,
            min_ms=min(v),
            max_ms=max(v),
            count=len(v),
        )

    raw, clean = stats(values), stats(accepted)
    return dict(
        raw=raw,
        accepted=clean,
        excluded_indices=excluded,
        selected_ms=clean["median_ms"],
        passed=len(accepted) >= policy.min_samples and clean["cv"] <= policy.max_cv,
    )


def build_profile(
    *,
    manifest: dict,
    classes: list[dict],
    units: list[dict],
    context: dict,
    model_config: dict,
    parallel_config: dict,
    policy: IsolatedPolicy,
) -> dict:
    """Expand measured medians to ordered layers and the existing [i,j) v2 prefixes."""
    measured = {u["class_signature"]: u for u in units}
    if len(measured) != len(units) or set(measured) != {c["class_signature"] for c in classes} | {
        "first",
        "last",
    }:
        raise ValueError("Missing/duplicate/unexpected profiling units")
    summaries = {
        key: {
            phase: summarize(unit[phase + "_gpu_ms"], policy) for phase in ("forward", "backward")
        }
        for key, unit in measured.items()
    }
    layers = [None] * manifest["num_layers"]
    for c in classes:
        for i in c["member_layers"]:
            if not 0 <= i < len(layers) or layers[i] is not None:
                raise ValueError("Invalid class expansion")
            layers[i] = dict(
                layer_id=i,
                config_class=manifest["layers"][i]["config_class"],
                class_signature=c["class_signature"],
                **{
                    p + "_us": 1000 * summaries[c["class_signature"]][p]["selected_ms"]
                    for p in ("forward", "backward")
                },
            )
    if None in layers:
        raise ValueError("Class expansion does not cover every layer")
    prefixes = {}
    for phase in ("forward", "backward"):
        values = [0.0]
        for layer in layers:
            values.append(values[-1] + layer[phase + "_us"])
        prefixes["prefix_" + phase + "_us"] = values
    issues = [
        dict(
            class_signature=k,
            phase=p,
            cv=v["accepted"]["cv"],
            threshold=policy.max_cv,
            samples=v["accepted"]["count"],
            min_samples=policy.min_samples,
        )
        for k, s in summaries.items()
        for p, v in s.items()
        if not v["passed"]
    ]
    profile = dict(
        schema_version="slackpipe.cost_profile.v2",
        estimator=ESTIMATOR,
        units="microseconds",
        model_manifest_hash=manifest["manifest_hash"],
        model_config=model_config,
        parallel_config=parallel_config,
        measurement_definition=dict(
            measurement_mode="isolated",
            includes_pipeline_wait=False,
            includes_p2p=False,
            includes_communication=False,
            statistic="median of accepted GPU event samples",
            timing="synchronized current-stream events; isolated implementation time including launch gaps, not pure FLOP time",
            backward="real autograd with representative upstream gradient; parameter/input grads reset before each F/B pair",
            optimizer_included=False,
        ),
        isolated=dict(
            context=context, manifest=manifest, classes=classes, units=units, summaries=summaries
        ),
        layer_costs_us=layers,
        class_costs_us={
            c["class_signature"]: {
                p: 1000 * summaries[c["class_signature"]][p]["selected_ms"]
                for p in ("forward", "backward")
            }
            for c in classes
        },
        stage_role_bias_us={
            r: {
                p: 0.0 if r == "middle" else 1000 * summaries[r][p]["selected_ms"]
                for p in ("forward", "backward")
            }
            for r in ("first", "middle", "last")
        },
        quality=dict(
            quality_schema_version=QUALITY_SCHEMA,
            status="rerun_required" if issues else "passed",
            thresholds=asdict(policy),
            issues=issues,
        ),
        **prefixes,
    )
    # Nested manifest must not expose a second schema_version to the C++ reader.
    profile["isolated"]["manifest"] = {k: v for k, v in manifest.items() if k != "schema_version"}
    if parallel_config["pp"] * (parallel_config.get("vpp") or 1) == 1:
        for phase in ("forward", "backward"):
            profile["stage_role_bias_us"]["first"][phase] += profile["stage_role_bias_us"]["last"][
                phase
            ]
    profile["cost_profile_hash"] = profile_fingerprint(profile)
    return profile


def require_quality(profile: dict) -> None:
    """Recompute selection and expansion, rather than trusting a 'passed' flag."""
    from .profile_quality import MESSAGE, ProfilingQualityError

    if profile.get("cost_profile_hash") != profile_fingerprint(profile):
        raise ProfilingQualityError("Isolated profile hash mismatch")
    try:
        data = profile["isolated"]
        if data["context"]["policy"] != profile["quality"]["thresholds"]:
            raise ValueError("Isolated quality policy changed after collection")
        for unit in data["units"]:
            samples = unit["samples"]
            context = data["context"]
            if [s["iteration"] for s in samples] != list(
                range(context["warmups"], context["warmups"] + context["iterations"])
            ):
                raise ValueError("Incomplete isolated samples")
            for phase in ("forward", "backward"):
                if unit[phase + "_gpu_ms"] != [s[phase + "_gpu_ms"] for s in samples]:
                    raise ValueError("Isolated raw samples and selected input differ")
        manifest = dict(data["manifest"], schema_version="slackpipe.model_manifest.v1")
        expected_classes = class_mapping(manifest, data["context"]["execution"])
        if data["classes"] != expected_classes:
            raise ValueError("Class mapping differs from execution/manifest")
        rebuilt = build_profile(
            manifest=manifest,
            classes=expected_classes,
            units=data["units"],
            context=data["context"],
            model_config=profile["model_config"],
            parallel_config=profile["parallel_config"],
            policy=IsolatedPolicy(**profile["quality"]["thresholds"]),
        )
        if rebuilt["isolated"]["summaries"] != data["summaries"]:
            raise ValueError("Isolated summaries differ from raw samples")
        for key in (
            "model_manifest_hash",
            "layer_costs_us",
            "class_costs_us",
            "prefix_forward_us",
            "prefix_backward_us",
            "stage_role_bias_us",
            "quality",
        ):
            if rebuilt[key] != profile[key]:
                raise ValueError(f"Invalid isolated {key}")
        if rebuilt["quality"]["status"] != "passed":
            raise ValueError("Unstable isolated measurements")
    except (KeyError, TypeError, ValueError) as e:
        raise ProfilingQualityError(f"{MESSAGE} {e}") from e


def add_arguments(parser) -> None:
    parser.add_argument(
        "--calibration-estimator", choices=(ESTIMATOR, LEGACY_ESTIMATOR), default=ESTIMATOR
    )
    parser.add_argument("--isolated-profile-warmups", type=int, default=20)
    parser.add_argument("--isolated-profile-iterations", type=int, default=30)
    parser.add_argument("--isolated-profile-max-cv", type=float, default=0.15)


def execution_signature(
    model: dict, *, seq_length: int, micro_batch_size: int, precision: str
) -> dict:
    """Conservative key for the strict eval-model adapter; unknown models are rejected by it."""
    return dict(
        model={k: v for k, v in model.items() if k != "schema_version"},
        sequence_length=seq_length,
        micro_batch_size=micro_batch_size,
        precision=precision,
        tp=1,
        dp=1,
        cp=1,
        recompute=None,
        implementation="Megatron TE GPT/hybrid_stack_spec; native untied head/loss; adapter-v1",
    )
