# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Synthetic raw timings model the reported LLaMA spikes, not GPU measurements."""

import copy
import json
from dataclasses import replace
from unittest.mock import Mock

import pytest

from megatron.core.pipeline_parallel.slackpipe.cost_profile import build_cost_profile
from megatron.core.pipeline_parallel.slackpipe.profile_quality import (
    ProfilingQualityError,
    QualityThresholds,
    assess_profile,
    publish_profile,
    require_profile_quality,
    tag_calibration_events,
)
from tools.slackpipe_profile_quality import collect_with_retry


def raw_profile(
    root, spikes=(), *, stages=8, iterations=30, microbatches=8, thresholds=None, normal=None
):
    """Generate one tagged calibration group with independent stage/phase clocks."""
    events = []
    overrides = {(s, p, i): v for s, p, i, v in spikes}
    for stage in range(stages):
        local = []
        for phase in ("forward", "backward"):
            for i in range(iterations * microbatches):
                value = (
                    normal(stage, phase, i) if normal else (10.0 if phase == "forward" else 20.0)
                )
                local.append(
                    dict(
                        logical_stage=stage,
                        phase=phase + "_compute",
                        iteration=5 + i // microbatches,
                        microbatch=i % microbatches,
                        rank=stage % 4,
                        elapsed_ms=overrides.get((stage, phase, i), value),
                    )
                )
        tag_calibration_events(
            local,
            group="partition0",
            ranges=[[s, s + 1] for s in range(stages)],
            worker=stage % 4,
            configuration={"model": "fixture"},
            raw_path=root / "raw.json",
        )
        events.extend(local)
    original = copy.deepcopy(events)
    profile = build_cost_profile(
        events=events,
        model_config={},
        parallel_config={},
        layer_split=[1] * stages,
        num_microbatches=microbatches,
        iteration_start=5,
        iteration_end=5 + iterations - 1,
        quality_thresholds=thresholds or QualityThresholds(),
    )
    assert events == original
    return profile, events


def test_stable_data_preserves_estimator(tmp_path):
    profile, _ = raw_profile(tmp_path)
    report = assess_profile(profile)
    assert report["status"] == "passed" and report["samples_discarded"] == 0
    assert profile["a_fwd"] == 10 and profile["a_bwd"] == 20
    assert profile["bias_fwd"] == [0] * 8
    assert profile["observed_stages"][0]["forward_diagnostics"]["samples_ms"] == [80] * 30


def test_one_spike_before_iteration_mean_and_raw_unchanged(tmp_path):
    profile, events = raw_profile(tmp_path, [(4, "forward", 13, 356)])
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(events))
    original = raw.read_bytes()
    report = publish_profile(tmp_path / "cost_profile.json", profile)
    assert raw.read_bytes() == original
    group = next(g for g in report["sample_groups"] if g["stage"] == 4 and g["phase"] == "forward")
    assert (
        group["raw_sample_count"],
        group["accepted_sample_count"],
        group["samples_discarded"],
    ) == (240, 239, 1)
    assert group["discarded_fraction"] == pytest.approx(1 / 240)
    assert group["raw_cv"] > 0.5 and group["cleaned_cv"] == 0
    assert report["status"] == "passed" and report["samples_discarded"] == 1
    rejected = report["rejected_samples"][0]
    assert (rejected["iteration"], rejected["microbatch"], rejected["elapsed_ms"]) == (6, 5, 356)
    assert rejected["median"] == 10 and rejected["MAD"] == 0
    assert rejected["robust_sigma"] == 0 and rejected["threshold_ms"] == 30
    assert rejected["raw_data_path"] == str(raw.resolve())
    row = profile["observed_stages"][4]
    assert row["forward_ms_per_op"] == 10
    assert row["forward_diagnostics"]["samples_ms"][1] == 80
    assert row["forward_diagnostics"]["accepted_counts_by_iteration"][1] == 7
    require_profile_quality(profile)


def test_reported_llama_pattern_passes_without_retry(tmp_path):
    spikes = [
        (4, "forward", 13, 356),
        (2, "forward", 50, 313),
        (5, "forward", 57, 306),
        (3, "forward", 98, 286),
    ]
    calls = []

    def collect(worker, attempt):
        calls.append(attempt)
        profile, _ = raw_profile(worker, spikes)
        publish_profile(worker / "cost_profile.json", profile, attempt=attempt)

    selected = collect_with_retry(tmp_path, collect, {})
    assert calls == [0]
    profile = json.loads(selected.read_text())
    assert profile["quality"]["samples_discarded"] == 4
    assert not profile["quality"]["issues"]
    assert all(g["cleaned_cv"] == 0 for g in profile["quality"]["sample_groups"])
    assert all(
        g["samples_discarded"] == 0
        for g in profile["quality"]["sample_groups"]
        if g["phase"] == "backward"
    )


def test_rare_multiple_spikes_and_nonzero_mad(tmp_path):
    profile, _ = raw_profile(
        tmp_path,
        [(0, "forward", i, 310) for i in (13, 50, 98, 170)],
        normal=lambda s, p, i: (10 if p == "forward" else 20) + (i % 3 - 1) * 0.1,
    )
    report = assess_profile(profile)
    assert report["status"] == "passed" and report["samples_discarded"] == 4
    assert all(r["MAD"] > 0 for r in report["rejected_samples"])


def test_moderate_variation_is_not_trimmed(tmp_path):
    profile, _ = raw_profile(tmp_path, normal=lambda s, p, i: 9 + (i // 8 % 5))
    report = assess_profile(profile)
    assert report["samples_discarded"] == 0
    assert any(i["rule"] == "within_group_cv" for i in report["issues"])
    assert report["status"] == "rerun_required"


@pytest.mark.parametrize("spike,spread", [(25, 0), (35, 2)])
def test_both_catastrophic_thresholds_must_be_exceeded(tmp_path, spike, spread):
    profile, _ = raw_profile(
        tmp_path,
        [(0, "forward", 13, spike)],
        normal=lambda s, p, i: 10 + (spread if i % 2 else -spread),
    )
    # 25 exceeds MAD-only (10), not ratio (30); 35 exceeds ratio,
    # not MAD-only (39.652). Neither may be removed.
    assert assess_profile(profile)["samples_discarded"] == 0


@pytest.mark.parametrize(
    "spikes,kwargs,rule",
    [
        ([1, 25, 49, 73, 97], {}, "discarded_fraction"),
        ([13, 14], {}, "max_consecutive_outliers"),
        ([7, 8], {}, "max_consecutive_outliers"),
        ([0, 100, 200, 300, 400, 500, 600], {"microbatches": 100}, "affected_iteration_fraction"),
    ],
)
def test_systemic_patterns_rejected(tmp_path, spikes, kwargs, rule):
    profile, _ = raw_profile(tmp_path, [(0, "forward", i, 310) for i in spikes], **kwargs)
    report = assess_profile(profile)
    assert report["status"] == "rerun_required"
    assert any(issue["rule"] == rule for issue in report["issues"])


@pytest.mark.parametrize("retry_passes", [True, False])
def test_systemic_failure_retries_only_once(tmp_path, retry_passes):
    calls, solver = [], Mock()

    def collect(worker, attempt):
        calls.append(attempt)
        spikes = [13] if attempt and retry_passes else [13, 14]
        profile, _ = raw_profile(worker, [(0, "forward", i, 310) for i in spikes])
        publish_profile(worker / "cost_profile.json", profile, attempt=attempt)

    with pytest.warns(RuntimeWarning, match="Profiling measurements are inconsistent"):
        if retry_passes:
            solver(collect_with_retry(tmp_path, collect, {}))
        else:
            with pytest.raises(ProfilingQualityError, match="retry exhausted"):
                solver(collect_with_retry(tmp_path, collect, {}))
    assert calls == [0, 1] and solver.call_count == int(retry_passes)


def test_stage_and_phase_isolation(tmp_path):
    profile, _ = raw_profile(
        tmp_path,
        [(0, "forward", 13, 310)],
        normal=lambda s, p, i: 100 if s == 7 or p == "backward" else 10,
    )
    report = assess_profile(profile)
    assert report["samples_discarded"] == 1 and report["status"] == "passed"
    assert profile["observed_stages"][7]["forward_ms_per_op"] == 100
    assert profile["observed_stages"][0]["backward_ms_per_op"] == 100


def test_changed_policy_cannot_reuse_filtered_observations(tmp_path):
    profile, _ = raw_profile(tmp_path, [(0, "forward", 13, 310)])
    report = assess_profile(profile, replace(QualityThresholds(), outlier_median_multiplier=4))
    assert any(i["rule"] == "outlier_policy_match" for i in report["issues"])


def test_insufficient_raw_samples_never_filter(tmp_path):
    profile, _ = raw_profile(tmp_path, [(0, "forward", 1, 310)], iterations=3, microbatches=4)
    report = assess_profile(profile)
    assert report["samples_discarded"] == 0 and report["status"] == "rerun_required"


@pytest.mark.parametrize(
    "policy",
    [
        {"outlier_median_multiplier": 1},
        {"outlier_mad_multiplier": float("nan")},
        {"max_outlier_fraction": -0.1},
        {"max_outlier_iteration_fraction": 1.1},
        {"max_consecutive_outliers": 0},
        {"min_outlier_samples": 2},
    ],
)
def test_invalid_policy(policy):
    with pytest.raises(ValueError):
        QualityThresholds(**policy)
