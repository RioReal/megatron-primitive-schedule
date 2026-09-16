"""Compact GPU timing contracts without requiring a profiler-capable GPU."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from megatron.core.pipeline_parallel.slackpipe.figure_trace import (
    compact_profiler_trace,
    label_logical_operations,
    validate_compact_trace,
    write_compact_trace,
)
from tools.plot_schedule_trace import load_panel


def raw_trace(rank=0):
    events = []
    for kind, start, end in (("step", 0, 100), ("F", 5, 25), ("B", 40, 65), ("optimizer", 75, 90)):
        stage, b = (rank, 0) if kind in ("F", "B") else (-1, -1)
        events.append(
            dict(
                ph="X",
                cat="user_annotation",
                name=f"ScheduleTrace/{kind}/i5/s{stage}/b{b}",
                ts=start,
                dur=end - start,
            )
        )
    for external, cpu, gpu, duration in ((1, 6, 10, 15), (2, 41, 50, 15), (3, 76, 80, 5)):
        events.extend(
            [
                dict(
                    ph="X",
                    cat="cpu_op",
                    name="aten::operation",
                    ts=cpu,
                    dur=1,
                    args={"External id": external},
                ),
                dict(
                    ph="X",
                    cat="kernel",
                    name="kernel",
                    ts=gpu,
                    dur=duration,
                    args={"External id": external},
                ),
            ]
        )
    # This is a GPU mirror emitted by Kineto, not another logical operation.
    events.append(
        dict(
            ph="X", cat="gpu_user_annotation", name=f"ScheduleTrace/F/i5/s{rank}/b0", ts=10, dur=15
        )
    )
    return dict(baseTimeNanoseconds=1_000_000_000 + rank * 5000, traceEvents=events)


def compact(rank=0):
    return compact_profiler_trace(
        raw_trace(rank),
        rank=rank,
        mode="baseline",
        transport="megatron-p2p",
        measured_steps={"5": dict(cuda_elapsed_ms=0.080)},
        config=dict(pp=2, num_stages=2, num_microbatches=1, host="same-host"),
    )


def test_gpu_envelopes_format_and_roundtrip(tmp_path):
    trace = compact()
    assert len(trace["records"]) == 3
    assert trace["records"][0]["start_us"] == 10
    assert trace["records"][0]["cpu_start_us"] == 5
    assert trace["steps"][0]["end_us"] == 85
    write_compact_trace(tmp_path / "rank0_trace.json", trace)
    assert json.loads((tmp_path / "rank0_trace.json").read_text()) == trace
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("error", ["missing", "duplicate", "overlap", "negative", "nonfinite"])
def test_reject_invalid_envelopes(error):
    trace = compact()
    if error == "missing":
        trace["records"].pop(0)
    elif error == "duplicate":
        trace["records"].append(copy.deepcopy(trace["records"][0]))
    elif error == "overlap":
        trace["records"][1]["start_us"] = 20
    elif error == "negative":
        trace["records"][0]["end_us"] = 0
    else:
        trace["records"][0]["start_us"] = float("nan")
    with pytest.raises(ValueError):
        validate_compact_trace(trace)


def test_missing_gpu_clock_and_gpu_activity_rejected():
    for field in ("clock", "gpu"):
        raw = raw_trace()
        if field == "clock":
            raw.pop("baseTimeNanoseconds")
        else:
            raw["traceEvents"] = [e for e in raw["traceEvents"] if e["cat"] != "kernel"]
        with pytest.raises(ValueError):
            compact_profiler_trace(
                raw,
                rank=0,
                mode="baseline",
                transport="p2p",
                measured_steps={"5": dict(cuda_elapsed_ms=0.08)},
                config=dict(pp=2, num_stages=2, num_microbatches=1),
            )


def test_panel_preserves_rank_clock_offset(tmp_path):
    for rank in range(2):
        write_compact_trace(tmp_path / f"rank{rank}_trace.json", compact(rank))
    panel = load_panel(tmp_path, 5)
    assert panel["duration_ms"] == pytest.approx(0.080)
    assert panel["logical_ops"] == 4
    assert next(r["start_ms"] for r in panel["records"] if r["rank"] == 1) == pytest.approx(0.005)


def test_plan_order_is_checked():
    trace = compact()
    plan = SimpleNamespace(
        worker_operations=lambda rank: [
            SimpleNamespace(kind="B", microbatch=0, stage=rank),
            SimpleNamespace(kind="F", microbatch=0, stage=rank),
        ]
    )
    with pytest.raises(ValueError, match="order differs"):
        validate_compact_trace(trace, plan)


def test_multiple_selected_steps_are_retained():
    raw = raw_trace()
    following = copy.deepcopy(raw["traceEvents"])
    for event in following:
        event["ts"] += 200
        event["name"] = event["name"].replace("/i5/", "/i6/")
        if "args" in event:
            event["args"]["External id"] += 100
    raw["traceEvents"].extend(following)
    trace = compact_profiler_trace(
        raw,
        rank=0,
        mode="baseline",
        transport="p2p",
        measured_steps={str(i): dict(cuda_elapsed_ms=0.08) for i in (5, 6)},
        config=dict(pp=2, num_stages=2, num_microbatches=1),
    )
    assert [step["iteration"] for step in trace["steps"]] == [5, 6]
    assert len(trace["records"]) == 6


def test_label_scope_preserves_numerics_and_restores_helpers(monkeypatch):
    from megatron.core.pipeline_parallel import schedules

    weight = torch.nn.Parameter(torch.tensor(3.0))
    optimizer = torch.optim.SGD([weight], lr=0.1)

    def forward(model, current_microbatch, vp_stage):
        return weight.square(), 0

    def backward(output_tensor):
        output_tensor.backward()

    monkeypatch.setattr(schedules, "forward_step", forward)
    monkeypatch.setattr(schedules, "backward_step", backward)
    with label_logical_operations(5, 0, 2, optimizer):
        loss, _ = schedules.forward_step(None, 0, 0)
        schedules.backward_step(loss)
        optimizer.step()
    assert weight.grad.item() == 6.0
    assert weight.item() == pytest.approx(2.4)
    assert schedules.forward_step is forward and schedules.backward_step is backward
    with pytest.raises(RuntimeError, match="unmatched"):
        with label_logical_operations(6, 0, 2, optimizer):
            schedules.forward_step(None, 0, 0)
    assert schedules.forward_step is forward
