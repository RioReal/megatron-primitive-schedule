# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Synthetic repeated partitions; not measurements of the reported 4B run."""

import copy
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from megatron.core.pipeline_parallel.slackpipe.cost_profile import (
    aggregate_stage_costs,
    build_heterogeneous_cost_profile,
)
from megatron.core.pipeline_parallel.slackpipe.profile_quality import (
    ProfilingQualityError,
    QualityThresholds,
    assess_profile,
    group_consensus,
    isolate_timing_spikes,
    publish_profile,
    require_profile_quality,
    tag_calibration_events,
)
from tools.slackpipe_profile_quality import collect_with_retry


def group_profile(root, values, *, raw_spike=False, noisy_group=None, thresholds=None):
    thresholds = thresholds or QualityThresholds()
    manifest = dict(
        manifest_hash="one-class-fixture",
        layers=[dict(layer_id=i, config_class="A") for i in range(6)],
    )
    rows = []
    for g, value in enumerate(values):
        for p, cuts in enumerate(((0, 1, 3, 6), (0, 2, 4, 6), (0, 3, 5, 6))):
            events = []
            for s in range(3):
                local = []
                for phase in ("forward", "backward"):
                    v = (cuts[s + 1] - cuts[s]) * (1 if phase == "forward" else 2)
                    v += ([0.5, 0.25, 0.75] if phase == "forward" else [17, 1, 2])[s]
                    if phase == "backward" and s == p == 0:
                        v = value
                    for i in range(30):
                        for b in range(8):
                            elapsed = v
                            if raw_spike and (g, p, s, phase, i, b) == (0, 0, 0, "forward", 6, 5):
                                elapsed = 310
                            if g == noisy_group and p == s == 0 and phase == "backward":
                                elapsed *= 0.7 if i % 2 else 1.3
                            local.append(
                                dict(
                                    logical_stage=s,
                                    phase=phase + "_compute",
                                    iteration=i,
                                    microbatch=b,
                                    elapsed_ms=elapsed,
                                    rank=s % 2,
                                )
                            )
                tag_calibration_events(
                    local,
                    group=f"g{g}-p{p}",
                    worker=s % 2,
                    ranges=list(zip(cuts, cuts[1:])),
                    configuration={"fixture": 1},
                    raw_path=root / f"g{g}-p{p}.raw.json",
                )
                events.extend(local)
            rows.extend(
                aggregate_stage_costs(
                    events,
                    layer_split=[b - a for a, b in zip(cuts, cuts[1:])],
                    num_microbatches=8,
                    iteration_start=0,
                    iteration_end=29,
                    quality_thresholds=thresholds,
                )
            )
    original = copy.deepcopy(rows)
    profile = build_heterogeneous_cost_profile(
        model_manifest=manifest,
        observed_stage_rows=rows,
        model_config={},
        parallel_config={},
        quality_thresholds=thresholds,
    )
    assert rows == original
    return profile


@pytest.mark.parametrize(
    "values,rejected",
    [
        ([19, 19.2, 18.9, 19.1, 13.5], 1),
        ([19, 19.2, 18.9, 13.5, 13.8], 2),
        ([19, 19.1, 18.9, 19], 0),
        ([19.05, 18.8, 20.04, 13.75], 1),
        ([19, 19.2, 18.9, 19.1, 26], 1),
    ],
)
def test_clear_consensus_uses_cleaned_fit(tmp_path, values, rejected):
    profile = group_profile(tmp_path, values)
    report = publish_profile(tmp_path / "cost_profile.json", profile)
    assert report["status"] == "passed"
    assert report["rejected_group_count"] == rejected
    assert all(r["phase"] == "backward" for r in report["rejected_groups"])
    comparison = next(
        c
        for c in report["group_filter"]["comparisons"]
        if c["phase"] == "backward" and c["stage_layer_range"] == [0, 1]
    )
    assert comparison["accepted_group_count"] >= 3
    assert comparison["cleaned_median_shift"] < 0.1
    assert comparison["cleaned_cv"] < 0.03
    assert comparison["consensus_median_ms"] == pytest.approx(19, abs=0.06)
    inputs = profile["group_filter"]["fit_observations"]["backward"]
    assert len(inputs) == 9
    assert inputs[0]["backward_ms_per_op"] == comparison["consensus_median_ms"]
    if rejected:
        assert comparison["raw_median_shift"] > 0.3
        assert comparison["raw_cv"] > comparison["cleaned_cv"]
        assert len(inputs[0]["source_row_indices"]) == len(values) - rejected
    # Prefix/role coefficients consumed by C++ are actually refitted, not only approved.
    assert profile["prefix_backward_us"][1] + profile["stage_role_bias_us"]["first"][
        "backward"
    ] == pytest.approx(19000, abs=70)
    require_profile_quality(profile)


@pytest.mark.parametrize("values", [[14, 16, 18, 20, 22], [14, 14.2, 20, 20.1]])
def test_broad_or_ambiguous_groups_fail_without_trimming(tmp_path, values):
    report = assess_profile(group_profile(tmp_path, values))
    assert report["status"] == "rerun_required"
    assert report["rejected_group_count"] == 0
    assert any(i["rule"] == "cross_group_median_shift" for i in report["issues"])


def test_two_level_filtering_and_surviving_cv_gate(tmp_path):
    profile = group_profile(tmp_path, [19, 19.2, 18.9, 19.1, 13.5], raw_spike=True)
    report = assess_profile(profile)
    assert report["status"] == "passed"
    assert report["samples_discarded"] == 1 and report["rejected_group_count"] == 1
    assert report["rejected_samples"][0]["elapsed_ms"] == 310
    assert report["rejected_groups"][0]["estimate_ms"] == 13.5
    assert report["effective_accepted_sample_count"] == report["accepted_sample_count"] - 240
    # CV alone is never a reason to exclude a consensus member.
    profile = group_profile(tmp_path, [19, 19.2, 18.9, 19.1, 13.5], noisy_group=0)
    report = assess_profile(profile)
    assert report["status"] == "rerun_required"
    assert any(i["rule"] == "within_group_cv" for i in report["issues"])


@pytest.mark.parametrize("median,spike", [(5.9342, 20.7578), (5.4485, 20.9894)])
def test_reported_raw_spikes_remain_recorded(median, spike):
    events = [
        dict(
            logical_stage=0,
            phase="forward_compute",
            iteration=i // 8,
            microbatch=i % 8,
            elapsed_ms=median,
        )
        for i in range(240)
    ]
    events[13]["elapsed_ms"] = spike
    accepted, diagnostics = isolate_timing_spikes(events, QualityThresholds())
    assert len(accepted) == 239 and diagnostics["samples_discarded"] == 1
    assert diagnostics["rejected_samples"][0]["elapsed_ms"] == spike
    assert events[13]["elapsed_ms"] == spike


@pytest.mark.parametrize(
    "key,value",
    [
        ("stage", 7),
        ("stage_layer_range", [1, 2]),
        ("stage_role", "middle"),
        ("worker", 3),
        ("measurement_context", "other"),
        ("layer_composition", ["Mamba"]),
        ("model_manifest_hash", "other-model"),
    ],
)
def test_exact_comparability(tmp_path, key, value):
    profile = group_profile(tmp_path, [19, 19.2, 18.9, 19.1, 13.5])
    rows = profile["observed_stages"]
    rows[36][key] = value
    selected = group_consensus(rows, QualityThresholds())
    assert not selected["rejected_groups"]


@pytest.mark.parametrize(
    "policy", [dict(group_min_survivors=4), dict(group_max_discarded_fraction=0.2)]
)
def test_minimum_survivors_and_discard_limit(tmp_path, policy):
    t = replace(QualityThresholds(), **policy)
    profile = group_profile(tmp_path, [19, 19.2, 18.9, 13.5, 13.8], thresholds=t)
    report = assess_profile(profile, t)
    assert report["status"] == "rerun_required"
    assert report["rejected_group_count"] == 0
    assert any(i["rule"] == "group_consensus" for i in report["issues"])


def test_revalidation_cannot_change_policy_or_selection(tmp_path):
    profile = group_profile(tmp_path, [19, 19.2, 18.9, 19.1, 13.5])
    report = assess_profile(profile, replace(QualityThresholds(), group_mad_multiplier=4))
    assert any(i["rule"] == "group_consensus_provenance" for i in report["issues"])
    publish_profile(tmp_path / "profile.json", profile)
    profile["group_filter"]["accepted_row_indices"]["backward"].append(36)
    with pytest.raises(ProfilingQualityError):
        require_profile_quality(profile)


def test_cannot_omit_group_selection_or_launder_manifest(tmp_path):
    profile = group_profile(tmp_path, [19, 19.2, 18.9, 19.1, 13.5])
    del profile["group_filter"]
    assert any(i["rule"] == "missing_group_analysis" for i in assess_profile(profile)["issues"])
    with pytest.raises(ValueError, match="composition"):
        build_heterogeneous_cost_profile(
            model_manifest=dict(
                manifest_hash="different",
                layers=[dict(layer_id=i, config_class="Mamba") for i in range(6)],
            ),
            observed_stage_rows=profile["observed_stages"],
            model_config={},
            parallel_config={},
        )


def test_reported_five_estimates_do_not_force_consensus(tmp_path):
    profile = group_profile(tmp_path, [19.053, 18.799, 20.045, 13.755, 15.995])
    report = assess_profile(profile)
    assert report["status"] == "rerun_required"
    assert not report["rejected_groups"]
    assert any(i["rule"] == "cross_group_median_shift" for i in report["issues"])


@pytest.mark.parametrize("stable", [True, False])
def test_group_quality_retry_and_solver_export(tmp_path, stable):
    calls = []

    def collect(worker, attempt):
        calls.append(attempt)
        profile = group_profile(
            worker, [19, 19.2, 18.9, 19.1, 13.5] if stable else [14, 16, 18, 20, 22]
        )
        publish_profile(worker / "cost_profile.json", profile, attempt=attempt)

    if not stable:
        with (
            pytest.warns(RuntimeWarning),
            pytest.raises(ProfilingQualityError, match="retry exhausted"),
        ):
            collect_with_retry(tmp_path, collect, {})
        assert calls == [0, 1]
        assert (
            json.loads((tmp_path / "profiling_attempts.json").read_text())["selected_profile"]
            is None
        )
        return
    selected = collect_with_retry(tmp_path, collect, {})
    assert calls == [0]
    binary = Path(__file__).resolve().parents[3] / "slackpipe/build/no-or/slackpipe_cli"
    if not binary.exists():
        pytest.skip("requires no-OR C++ build")
    result = subprocess.run(
        [
            str(binary),
            "--algorithm",
            "uniform-breadth-first",
            "--B",
            "4",
            "--N",
            "3",
            "--J",
            "1",
            "--L",
            "6",
            "--cost-profile",
            str(selected),
            "--output-prefix",
            str(tmp_path / "solver"),
            "--emit-plan",
            str(tmp_path / "plan.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "plan.json").exists()
