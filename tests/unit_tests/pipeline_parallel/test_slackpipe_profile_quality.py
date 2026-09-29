# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Offline quality and retry tests; synthetic timings are not GPU measurements."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from megatron.core.pipeline_parallel.slackpipe.cost_profile import aggregate_stage_costs
from megatron.core.pipeline_parallel.slackpipe.profile_quality import (
    MESSAGE,
    ProfilingQualityError,
    QualityThresholds,
    assess_profile,
    publish_profile,
    require_profile_quality,
    tag_calibration_events,
)
from tools.slackpipe_profile_quality import collect_with_retry


def stable_profile():
    rows = []
    for group in range(6):
        for stage in range(2):
            rows.append(
                dict(
                    calibration_group_id=f"partition{group}",
                    stage=stage,
                    stage_layer_range=[stage * 8, (stage + 1) * 8],
                    stage_role="first" if stage == 0 else "last",
                    worker=stage,
                    measurement_context="same-hardware-model-and-timing",
                    raw_data_path=f"/fixture/partition{group}/events.rank{stage}.json",
                    layer_count=8,
                    forward_ms_per_op=8.0,
                    backward_ms_per_op=24.8,
                    forward_diagnostics=dict(samples_ms=[64.0, 64.0, 64.0]),
                    backward_diagnostics=dict(samples_ms=[198.4, 198.4, 198.4]),
                )
            )
    return dict(
        schema_version="slackpipe.cost_profile.v1",
        observed_stages=rows,
        a_fwd=1.0,
        a_bwd=3.1,
        bias_fwd=[0.0, 0.0],
        bias_bwd=[0.0, 0.0],
    )


def shifted_profile():
    profile = stable_profile()
    for row in profile["observed_stages"]:
        if row["calibration_group_id"] in ("partition4", "partition5"):
            row["backward_ms_per_op"] = 55.0
            row["backward_diagnostics"]["samples_ms"] = [440.0] * 3
    return profile


def test_stable_and_systematic_shift():
    assert assess_profile(stable_profile())["status"] == "passed"
    report = assess_profile(shifted_profile())
    assert report["status"] == "rerun_required"
    assert not any(m["rule"] == "within_group_cv" for m in report["issues"])
    shifts = [m for m in report["issues"] if m["rule"] == "cross_group_median_shift"]
    assert len(shifts) == 2
    assert shifts[0]["value"] == pytest.approx(55 / 24.8 - 1)
    assert len(shifts[0]["measurements"]) == 6
    assert shifts[0]["measurements"][0]["raw_data_path"]
    assert any(m["rule"] == "fit_relative_rmse" for m in report["issues"])


@pytest.mark.parametrize(
    "key,value",
    [
        ("stage_layer_range", [0, 16]),
        ("stage_role", "middle"),
        ("worker", 3),
        ("measurement_context", "different-model-or-hardware"),
    ],
)
def test_only_compare_identical_workloads(key, value):
    profile = shifted_profile()
    for row in profile["observed_stages"]:
        if row["calibration_group_id"] in ("partition4", "partition5"):
            row[key] = value
    issues = assess_profile(profile)["issues"]
    assert not any(m["rule"] == "cross_group_median_shift" for m in issues)


def test_cv_fit_and_sample_requirements():
    profile = stable_profile()
    profile["observed_stages"][0]["forward_diagnostics"]["samples_ms"] = [20.0, 64.0, 108.0]
    assert any(m["rule"] == "within_group_cv" for m in assess_profile(profile)["issues"])
    profile = stable_profile()
    profile["a_fwd"] = 2
    assert any(m["rule"] == "fit_relative_rmse" for m in assess_profile(profile)["issues"])
    assert assess_profile(profile, QualityThresholds(relative_rmse=1.1))["status"] == "passed"
    profile = stable_profile()
    profile["observed_stages"][0]["forward_diagnostics"]["samples_ms"] = [64.0, 64.0]
    assert assess_profile(profile)["status"] == "rerun_required"


def test_v2_checks_actual_prefix_and_role_predictions():
    profile = stable_profile()
    profile.update(
        schema_version="slackpipe.cost_profile.v2",
        prefix_forward_us=[i * 1000.0 for i in range(17)],
        prefix_backward_us=[i * 3100.0 for i in range(17)],
        stage_role_bias_us={
            r: dict(forward=0.0, backward=0.0) for r in ("first", "middle", "last")
        },
    )
    assert assess_profile(profile)["status"] == "passed"
    for row in profile["observed_stages"]:
        if row["calibration_group_id"] in ("partition4", "partition5"):
            row["stage_layer_range"] = [0, 4] if row["stage"] == 0 else [4, 16]
            count = row["stage_layer_range"][1] - row["stage_layer_range"][0]
            row["layer_count"] = count
            row["forward_ms_per_op"] = float(count)
            row["backward_ms_per_op"] = count * 3.1
            row["forward_diagnostics"]["samples_ms"] = [count * 8.0] * 3
            row["backward_diagnostics"]["samples_ms"] = [count * 3.1 * 8] * 3
    assert assess_profile(profile)["status"] == "passed"
    profile["stage_role_bias_us"]["first"]["backward"] = 30000.0
    assert any(m["rule"] == "fit_relative_rmse" for m in assess_profile(profile)["issues"])


@pytest.mark.parametrize("second_passes", [False, True])
def test_one_retry_and_fail_closed(tmp_path, second_passes):
    calls, original = [], shifted_profile()
    solver = Mock()

    def collect(worker, attempt):
        calls.append(attempt)
        profile = stable_profile() if attempt == 1 and second_passes else copy.deepcopy(original)
        publish_profile(
            worker / "cost_profile.json",
            profile,
            attempt=attempt,
            command="python -m tools.run_slackpipe_eval calibrate --output fresh",
        )

    with pytest.warns(RuntimeWarning, match=MESSAGE):
        if second_passes:
            selected = collect_with_retry(tmp_path, collect, {"fixture": True})
            solver(selected)
            assert selected == tmp_path / "attempt1/worker/cost_profile.json"
        else:
            with pytest.raises(ProfilingQualityError, match="retry exhausted"):
                solver(collect_with_retry(tmp_path, collect, {"fixture": True}))
    assert calls == [0, 1]
    assert solver.call_count == int(second_passes)
    report = json.loads((tmp_path / "profiling_attempts.json").read_text())
    assert len(report["attempts"]) == 2
    assert bool(report["selected_profile"]) == second_passes
    for attempt in range(2):
        directory = tmp_path / f"attempt{attempt}/worker"
        assert (directory / "cost_profile.candidate.json").is_file()
        assert (directory / "cost_profile.quality.json").is_file()
        assert (directory / "cost_profile.json").exists() == (attempt == 1 and second_passes)


def test_stable_collects_once_and_revalidates(tmp_path):
    calls = []

    def collect(worker, attempt):
        calls.append(attempt)
        publish_profile(worker / "cost_profile.json", stable_profile())

    selected = collect_with_retry(tmp_path, collect, {})
    assert calls == [0]
    profile = json.loads(selected.read_text())
    require_profile_quality(profile)
    profile["a_fwd"] *= 10
    with pytest.raises(ProfilingQualityError):
        require_profile_quality(profile)
    with pytest.raises(ProfilingQualityError):
        require_profile_quality(stable_profile())
    with pytest.raises(FileExistsError):
        publish_profile(selected, stable_profile())


def test_retry_configuration_cannot_change(tmp_path):
    def collect(worker, attempt):
        profile = shifted_profile() if not attempt else stable_profile()
        if attempt:
            profile["observed_stages"][0]["worker"] = 3
        publish_profile(worker / "cost_profile.json", profile)

    with pytest.warns(RuntimeWarning), pytest.raises(ProfilingQualityError, match="configuration"):
        collect_with_retry(tmp_path, collect, {})
    assert (
        json.loads((tmp_path / "profiling_attempts.json").read_text())["selected_profile"] is None
    )


def test_raw_group_identity_survives_aggregation(tmp_path):
    events = [
        dict(logical_stage=s, phase=p, iteration=i, elapsed_ms=1.0)
        for s in range(2)
        for p in ("forward_compute", "backward_compute")
        for i in range(3)
    ]
    tag_calibration_events(
        events,
        group="group7",
        ranges=[[0, 1], [1, 2]],
        worker=0,
        configuration={"model": "fixture"},
        raw_path=tmp_path / "raw.json",
        attempt=1,
    )
    rows = aggregate_stage_costs(
        events, layer_split=[1, 1], num_microbatches=1, iteration_start=0, iteration_end=2
    )
    assert all(r["calibration_group_id"] == "group7" and r["attempt"] == 1 for r in rows)
    assert all(e["calibration_group_id"] == "group7" for e in events)
    events[0]["calibration_group_id"] = "other"
    with pytest.raises(ValueError, match="different calibration groups"):
        aggregate_stage_costs(
            events, layer_split=[1, 1], num_microbatches=1, iteration_start=0, iteration_end=2
        )


def test_cli_retries_with_fresh_processes(tmp_path, monkeypatch):
    from tools.slackpipe_profile_quality import main

    # A real subprocess collector without CUDA; the launcher must rewrite only outputs/attempt.
    script = tmp_path / "collector.py"
    script.write_text(
        '''import argparse, json, os
from pathlib import Path
from megatron.core.pipeline_parallel.slackpipe.profile_quality import publish_profile
from tests.unit_tests.pipeline_parallel.test_slackpipe_profile_quality import stable_profile, shifted_profile
p = argparse.ArgumentParser()
p.add_argument("--output", type=Path)
p.add_argument("--profiling-attempt", type=int)
a = p.parse_args()
profile = stable_profile() if a.profiling_attempt else shifted_profile()
publish_profile(a.output / "cost_profile.json", profile, attempt=a.profiling_attempt)
(a.output / "pid.json").write_text(json.dumps(os.getpid()))
if not a.profiling_attempt:
    raise SystemExit(2)
'''
    )
    output = tmp_path / "collection"
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[3]))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quality",
            "--collect-output",
            str(output),
            "--",
            sys.executable,
            str(script),
            "--output",
            str(output),
        ],
    )
    main()
    pids = [json.loads((output / f"attempt{i}/worker/pid.json").read_text()) for i in range(2)]
    assert len(set(pids + [os.getpid()])) == 3
    selected = json.loads((output / "cost_profile.json").read_text())
    assert selected["quality"]["attempt"] == 1
    assert selected["quality"]["status"] == "passed"


@pytest.mark.parametrize("version", [1, 2])
def test_quality_profile_cpp_compatibility(tmp_path, version):
    binary = Path(__file__).resolve().parents[3] / "slackpipe/build/no-or/slackpipe_cli"
    if not binary.is_file():
        pytest.skip("requires the no-OR C++ build")
    profile = stable_profile()
    profile["calibration_partition"] = [8, 8]
    if version == 2:
        profile.update(
            schema_version="slackpipe.cost_profile.v2",
            prefix_forward_us=[i * 1000.0 for i in range(17)],
            prefix_backward_us=[i * 3100.0 for i in range(17)],
            stage_role_bias_us={
                r: dict(forward=0.0, backward=0.0) for r in ("first", "middle", "last")
            },
        )
    path = tmp_path / "cost_profile.json"
    publish_profile(path, profile)
    assert path.read_text().count('"schema_version"') == 1
    result = subprocess.run(
        [
            str(binary),
            "--algorithm",
            "uniform-breadth-first",
            "--B",
            "4",
            "--N",
            "2",
            "--J",
            "2",
            "--L",
            "16",
            "--cost-profile",
            str(path),
            "--output-prefix",
            str(tmp_path / "solver"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
