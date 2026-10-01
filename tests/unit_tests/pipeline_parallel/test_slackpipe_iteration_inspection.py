# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Evidence replay tests, not validation of an enabled iteration filter."""

import copy
import hashlib
import json
import sys

import pytest

from megatron.core.pipeline_parallel.slackpipe.profile_quality import assess_profile
from tests.unit_tests.pipeline_parallel.test_slackpipe_outliers import raw_profile
from tools.slackpipe_inspect_iterations import inspect_iterations, main


def fixture(root, pattern, *, spikes=()):
    profile, events = raw_profile(
        root,
        stages=2,
        iterations=len(pattern),
        spikes=spikes,
        normal=lambda stage, phase, i: (
            pattern[i // 8] if stage == 0 and phase == "forward" else 20.0
        ),
    )
    return profile, assess_profile(profile), events


def test_stable_no_failures_or_mutation(tmp_path):
    inputs = fixture(tmp_path, [10.0] * 30)
    before = copy.deepcopy(inputs)
    result = inspect_iterations(*inputs)
    assert inputs == before
    assert result["failures"] == [] and result["profile_status"] == "passed"
    assert not result["iteration_filter_enabled"]


@pytest.mark.parametrize(
    "pattern",
    [
        [10.0] * 15 + [20.0] + [10.0] * 14,
        [10.0] * 15 + [3.0] + [10.0] * 14,
        [10.0] * 15 + [20.0] + [10.0] * 15 + [20.0] + [10.0] * 28,
        [20.0 if i % 5 == 0 else 10.0 for i in range(30)],
        [10.0] * 15 + [20.0] * 15,
        [10.0 + i for i in range(30)],
    ],
    ids=["one-slow", "one-fast", "two-isolated", "twenty-percent", "two-clusters", "drift"],
)
def test_preserves_time_order_and_does_not_authorize_cleaning(tmp_path, pattern):
    inputs = fixture(tmp_path, pattern)
    result = inspect_iterations(*inputs)
    assert result["profile_status"] == "rerun_required"
    assert result["selected_profile"] is None
    (failure,) = result["failures"]
    assert [i["mean_ms"] for i in failure["iterations"]] == pattern
    assert [i["iteration"] for i in failure["iterations"]] == list(range(5, 5 + len(pattern)))
    assert failure["rejected_iteration_count"] == 0
    assert failure["sample_cleaned_iteration_cv"] > 0.1
    assert failure["adjacent_changes_ms"] == [b - a for a, b in zip(pattern, pattern[1:])]
    assert failure["cause"].startswith("unclassified")


def test_sample_filter_replay_and_bundle_order(tmp_path):
    profile, quality, events = fixture(
        tmp_path, [10.0] * 15 + [20.0] + [10.0] * 14, spikes=[(0, "forward", 3, 400)]
    )
    # Partition bundle order and event order must not supply group identity.
    raw = [
        dict(events=list(reversed(events[len(events) // 2 :]))),
        dict(events=events[: len(events) // 2]),
    ]
    result = inspect_iterations(profile, quality, raw)
    (failure,) = result["failures"]
    first = failure["iterations"][0]
    assert first["accepted_microbatch_count"] == 7
    assert first["mean_ms"] == first["median_ms"] == first["min_ms"] == first["max_ms"] == 10
    assert first["aggregate_ms"] == 80 and first["raw_aggregate_ms"] == 470
    assert len(result["rejected_samples"]) == 1
    assert failure["raw_iteration_cv"] > failure["sample_cleaned_iteration_cv"]
    assert first["robust_z"] is None  # Zero MAD is not infinity or an automatic rejection.


def test_unrelated_range_group_and_context_never_enter_iteration_mean(tmp_path):
    profile, quality, events = fixture(tmp_path, [10.0] * 15 + [20.0] + [10.0] * 14)
    expected = inspect_iterations(profile, quality, events)
    for field, value in (
        ("calibration_group_id", "other-group"),
        ("measurement_context", "other-configuration"),
        ("stage_layer_range", [8, 9]),
        ("attempt", 1),
        ("worker", 3),
    ):
        other = copy.deepcopy(events)
        for event in other:
            event[field] = value
            event["elapsed_ms"] *= 100
        assert inspect_iterations(profile, quality, other + events) == expected


@pytest.mark.parametrize(
    "corruption", ["duplicate", "missing", "context", "range", "worker", "timing", "report"]
)
def test_mismatched_evidence_fails_closed(tmp_path, corruption):
    profile, quality, events = fixture(tmp_path, [10.0] * 15 + [20.0] + [10.0] * 14)
    if corruption == "duplicate":
        events[0]["microbatch"] = 1
    elif corruption == "missing":
        events.pop(0)
    elif corruption in ("context", "range", "worker"):
        field = dict(context="measurement_context", range="stage_layer_range", worker="worker")[
            corruption
        ]
        events[0][field] = dict(context="different", range=[1, 2], worker=99)[corruption]
    elif corruption == "timing":
        events[0]["elapsed_ms"] = 11
    else:
        quality["issues"][0]["value"] += 0.01
    with pytest.raises(ValueError):
        inspect_iterations(profile, quality, events)


def test_cli_hashes_inputs_and_refuses_overwrite(tmp_path, monkeypatch):
    inputs = fixture(tmp_path, [10.0] * 15 + [20.0] + [10.0] * 14)
    paths = [tmp_path / f"{name}.json" for name in ("profile", "quality", "events")]
    for path, value in zip(paths, inputs):
        path.write_text(json.dumps(value))
    before = [p.read_bytes() for p in paths]
    output = tmp_path / "inspection.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inspect",
            "--profile",
            str(paths[0]),
            "--quality-report",
            str(paths[1]),
            "--events",
            str(paths[2]),
            "--output",
            str(output),
        ],
    )
    assert main() == 0
    report = json.loads(output.read_text())
    assert report["diagnostic_only"] and report["profile_status"] == "rerun_required"
    for key, data in zip(("profile", "quality", "events"), before):
        assert report["inputs"][key]["sha256"] == hashlib.sha256(data).hexdigest()
    saved = output.read_bytes()
    assert main() == 2
    assert output.read_bytes() == saved
    assert [p.read_bytes() for p in paths] == before


def test_no_output_on_failed_replay(tmp_path, monkeypatch):
    profile, quality, events = fixture(tmp_path, [10.0] * 15 + [20.0] + [10.0] * 14)
    events.pop(0)
    paths = [tmp_path / f"{name}.json" for name in ("profile", "quality", "events")]
    for path, value in zip(paths, (profile, quality, events)):
        path.write_text(json.dumps(value))
    output = tmp_path / "inspection.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inspect",
            "--profile",
            str(paths[0]),
            "--quality-report",
            str(paths[1]),
            "--events",
            str(paths[2]),
            "--output",
            str(output),
        ],
    )
    assert main() == 2
    assert not output.exists()
