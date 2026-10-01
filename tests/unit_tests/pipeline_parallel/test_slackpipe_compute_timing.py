# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Diagnostic timing boundaries and evidence alignment, not estimator validation."""

import copy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.pipeline_parallel import schedules
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.slackpipe.timing_diagnostic import (
    ComputeTimingDiagnostic,
    DiagnosticP2PCommunicator,
)
from tools.slackpipe_compare_compute_timing import compare_runs, load_run


def test_event_resolution_is_deferred_and_pairs_reused(monkeypatch):
    events = []

    def event(**kwargs):
        result = Mock()
        result.elapsed_time.return_value = 2.5
        events.append(result)
        return result

    monkeypatch.setattr(torch.cuda, "Event", event)
    sync = Mock()
    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=7))
    probe = ComputeTimingDiagnostic(2)
    sync.assert_called_once()  # Preinitialization before warmup only.
    sync.reset_mock()
    for i in (20, 21):
        probe.begin(i)
        for j in range(2):
            assert probe.compute(lambda: 123, dict(phase="backward_compute", microbatch=j)) == 123
            sync.assert_not_called()
        rows = probe.finish()
        sync.assert_called_once()
        assert [r["compute_gpu_ms"] for r in rows] == [2.5, 2.5]
        assert [r["compute_position"] for r in rows] == [0, 1]
        assert all(r["existing_stage_wall_ms"] is None for r in rows)
        sync.reset_mock()
    assert len(events) == 4  # No per-iteration event allocation.
    assert events[0].elapsed_time.call_count == 2


def test_schedule_hook_bypasses_only_legacy_probe_synchronization(monkeypatch):
    recorder = Mock()
    recorder.compute.side_effect = lambda compute, identity: compute()
    recorder.finish.return_value = [{"compute_gpu_ms": 1}]
    sync = Mock()
    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    monkeypatch.setattr(schedules.parallel_state, "get_pipeline_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        schedules.parallel_state, "get_pipeline_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    schedules.begin_slackpipe_cost_calibration_iteration(30, diagnostic=recorder)
    try:
        value = schedules._run_with_slackpipe_cost_calibration(
            phase="backward_compute", microbatch=2, model_chunk_id=1, compute=lambda: "result"
        )
        assert value == "result"
        assert recorder.compute.call_args.args[1]["logical_stage"] == 2
        sync.assert_not_called()
    finally:
        assert schedules.end_slackpipe_cost_calibration_iteration() == [{"compute_gpu_ms": 1}]
    assert schedules._SLACKPIPE_TIMING_DIAGNOSTIC is None
    assert schedules._SLACKPIPE_COST_CALIBRATION_EVENTS is None


def test_legacy_probe_still_has_completion_boundary(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(schedules.parallel_state, "get_pipeline_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        schedules.parallel_state, "get_pipeline_model_parallel_world_size", lambda: 2
    )
    schedules.begin_slackpipe_cost_calibration_iteration(30)
    try:
        schedules._run_with_slackpipe_cost_calibration(
            phase="backward_compute",
            microbatch=2,
            model_chunk_id=1,
            compute=lambda: calls.append("compute"),
        )
    finally:
        rows = schedules.end_slackpipe_cost_calibration_iteration()
    assert calls == ["sync", "compute", "sync"]
    assert "elapsed_ms" in rows[0] and "compute_gpu_ms" not in rows[0]


def test_native_p2p_requests_and_wait_policy_unchanged(monkeypatch):
    native = Mock(return_value=("activation", "gradient", {"work": object()}))
    monkeypatch.setattr(P2PCommunicator, "_communicate", native)
    communicator = object.__new__(DiagnosticP2PCommunicator)
    communicator.config = SimpleNamespace(batch_p2p_comm=True, batch_p2p_sync=True)
    communicator.diagnostic = SimpleNamespace(active=True, position=3, p2p=[])
    kwargs = dict(
        tensor_send_next=None,
        tensor_send_prev=object(),
        recv_prev=True,
        recv_next=False,
        tensor_shape=(2, 4, 8),
        wait_on_reqs=False,
    )
    result = communicator._communicate(**kwargs)
    assert result is native.return_value
    native.assert_called_once_with(**kwargs)
    (row,) = communicator.diagnostic.p2p
    assert row["schedule_position"] == 3 and row["native_batch_sync"]
    assert row["wait_synchronization_ms"] is None
    assert communicator.diagnostic.position == 4


def make_run(root, diagnostic):
    root.mkdir()
    (root / "model_manifest.json").write_text(
        json.dumps(dict(manifest_hash="fixture", layers=[dict(config_class="mamba")]))
    )
    events = []
    for g in range(2):
        group = f"partition{g}"
        folder = root / group
        folder.mkdir()
        metadata = dict(
            rank=0,
            stage_layer_ranges=[[0, 1]],
            measurement_iterations=[20, 21, 22],
            warmup_iterations=list(range(20)),
            calibration_configuration=dict(manifest="fixture", pp=1, stages=1, microbatches=2),
            timing_diagnostic=dict(iterations=[]) if diagnostic else None,
        )
        (folder / "result.rank0.json").write_text(json.dumps(metadata))
        for phase in ("forward_compute", "backward_compute"):
            for i in (20, 21, 22):
                for b in (0, 1):
                    event = dict(
                        calibration_group_id=group,
                        worker=0,
                        rank=0,
                        logical_stage=0,
                        iteration=i,
                        microbatch=b,
                        phase=phase,
                        stage_role="first",
                        stage_layer_range=[0, 1],
                        layer_composition=["mamba"],
                    )
                    event["measurement_context"] = hashlib.sha256(
                        json.dumps(
                            dict(
                                **metadata["calibration_configuration"],
                                timing=(
                                    "compute-stream-events-diagnostic-v1"
                                    if diagnostic
                                    else "synchronized-stage-wall-time-v1"
                                ),
                            ),
                            sort_keys=True,
                            default=str,
                        ).encode()
                    ).hexdigest()
                    event.update(
                        dict(compute_gpu_ms=10, cpu_wall_ms=1)
                        if diagnostic
                        else dict(elapsed_ms=10 + g * 5)
                    )
                    events.append(event)
    (root / ("timing_events.json" if diagnostic else "calibration_events.json")).write_text(
        json.dumps(events)
    )
    return root


def test_comparison_uses_all_raw_samples_and_exact_context(tmp_path):
    old = load_run(make_run(tmp_path / "old", False), diagnostic=False)
    new = load_run(make_run(tmp_path / "new", True), diagnostic=True)
    result = compare_runs(old, new)
    assert not result["estimator_validated"]
    assert result["samples_discarded"] == 0
    for row in result["comparisons"]:
        assert row["cross_partition_median_shift"]["compute_gpu_ms"] == 0
        assert row["cross_partition_median_shift"]["existing_stage_wall_ms"] == 0.5
    altered = copy.deepcopy(new)
    altered["metadata"][("partition0", 0)]["calibration_configuration"]["microbatches"] = 3
    with pytest.raises(ValueError, match="Incomparable"):
        compare_runs(old, altered)


@pytest.mark.parametrize(
    "corruption", ["missing", "duplicate", "range", "composition", "rank", "context"]
)
def test_corrupt_raw_comparison_fails(tmp_path, corruption):
    root = make_run(tmp_path / "new", True)
    path = root / "timing_events.json"
    events = json.loads(path.read_text())
    if corruption == "missing":
        events.pop()
    elif corruption == "duplicate":
        events.append(events[0])
    elif corruption == "rank":
        (root / "partition0/result.rank0.json").unlink()
    elif corruption == "context":
        events[0]["measurement_context"] = "wrong-context"
    else:
        events[0]["stage_layer_range" if corruption == "range" else "layer_composition"] = [999]
    path.write_text(json.dumps(events))
    with pytest.raises(ValueError):
        load_run(root, diagnostic=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_cuda_events_resolve_after_full_step():
    probe = ComputeTimingDiagnostic(2)
    x = torch.ones(128, 128, device="cuda", requires_grad=True)
    reference = torch.ones_like(x, requires_grad=True)
    (reference @ reference).sum().backward()
    with torch.no_grad():
        reference.add_(reference.grad, alpha=-0.01)
    probe.begin(0)
    loss = probe.compute(lambda: (x @ x).sum(), dict(phase="forward_compute"))
    probe.compute(loss.backward, dict(phase="backward_compute"))
    with torch.no_grad():
        x.add_(x.grad, alpha=-0.01)
    rows = probe.finish()
    assert all(r["compute_gpu_ms"] > 0 and r["cpu_wall_ms"] > 0 for r in rows)
    assert torch.isfinite(x).all() and torch.isfinite(x.grad).all()
    torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
    torch.testing.assert_close(x, reference, rtol=0, atol=0)
