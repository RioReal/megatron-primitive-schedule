"""Opt-in profiler labels and GPU envelope export; never enabled by training."""

import bisect
import inspect
import json
import math
import re
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch

LABEL = re.compile(r"ScheduleTrace/(F|B|optimizer|step)/i(\d+)/s(-?\d+)/b(-?\d+)$")
SCHEMA = "slackpipe.figure_trace.v2"


def interval_union(intervals) -> list[list[float]]:
    """Union across streams, not a sum of potentially overlapping durations."""
    merged = []
    for start, end in sorted(intervals):
        if not all(math.isfinite(t) for t in (start, end)) or end < start:
            raise ValueError("Invalid activity interval")
        if start == end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def activity_metrics(events: list[dict]) -> dict:
    union = interval_union((e["ts"], e["ts"] + e["dur"]) for e in events)
    duration = sum(end - start for start, end in union)
    envelope = union[-1][1] - union[0][0] if union else 0.0
    return dict(
        gpu_activity_union_us=duration,
        gpu_envelope_us=envelope,
        internal_gap_us=envelope - duration,
        gpu_union_intervals_us=union,
    )


@contextmanager
def label_logical_operations(iteration: int, rank: int, pp_size: int, optimizer):
    """Temporarily label existing helpers, associating B with its F output object.

    No schedule order, tensor values, synchronization, or gradient logic changes.
    This scope is single-threaded at the schedule-dispatch level, as are the MVP
    schedulers. It must not wrap concurrent training invocations in one process.
    """
    from megatron.core.pipeline_parallel import schedules

    original_forward, original_backward = schedules.forward_step, schedules.backward_step
    forward_signature = inspect.signature(original_forward)
    backward_signature = inspect.signature(original_backward)
    pending = {}

    def region(kind, stage=-1, microbatch=-1):
        return torch.profiler.record_function(
            f"ScheduleTrace/{kind}/i{iteration}/s{stage}/b{microbatch}"
        )

    def forward(*args, **kwargs):
        values = forward_signature.bind(*args, **kwargs).arguments
        microbatch = values["current_microbatch"]
        vp_stage = values.get("vp_stage") or 0
        stage = vp_stage * pp_size + rank
        if microbatch is None:
            raise ValueError("Figure capture requires explicit microbatch IDs")
        with region("F", stage, microbatch):
            result = original_forward(*args, **kwargs)
        identity = id(result[0])
        if identity in pending:
            raise RuntimeError("Duplicate live forward output during trace capture")
        pending[identity] = (stage, microbatch)
        return result

    def backward(*args, **kwargs):
        values = backward_signature.bind(*args, **kwargs).arguments
        stage, microbatch = pending.pop(id(values["output_tensor"]))
        with region("B", stage, microbatch):
            return original_backward(*args, **kwargs)

    original_step = optimizer.step

    def step(*args, **kwargs):
        with region("optimizer"):
            return original_step(*args, **kwargs)

    with (
        patch.object(schedules, "forward_step", forward),
        patch.object(schedules, "backward_step", backward),
        patch.object(optimizer, "step", step),
    ):
        yield
        if pending:
            raise RuntimeError("Trace capture ended with unmatched forwards")


def compact_profiler_trace(
    raw: dict, *, rank: int, mode: str, transport: str, measured_steps: dict, config: dict
) -> dict:
    """Correlate CUDA activities to CPU ranges via Kineto external IDs.

    GPU bounds are first/last correlated activity, not CPU range timestamps and
    not a sum of kernel durations. Absolute timestamp offsets preserve cross-rank
    skew on the SAME HOST. Multi-node clock alignment is deliberately unsupported.
    """
    events = raw["traceEvents"]
    labels = []
    for event in events:
        match = LABEL.fullmatch(event.get("name", ""))
        if match and event.get("ph") == "X" and event.get("cat") == "user_annotation":
            kind, iteration, stage, microbatch = match.groups()
            labels.append(
                (
                    event,
                    dict(
                        kind=kind,
                        iteration=int(iteration),
                        stage=int(stage),
                        microbatch=int(microbatch),
                    ),
                )
            )
    if not labels:
        raise ValueError("No logical profiler labels found")
    cpu_by_external = {}
    runtime_by_correlation = {}
    for event in events:
        external = event.get("args", {}).get("External id")
        if event.get("cat") == "cpu_op" and external is not None:
            cpu_by_external.setdefault(external, []).append(event)
        correlation = event.get("args", {}).get("correlation")
        if event.get("cat") in ("cuda_runtime", "cuda_driver") and correlation is not None:
            runtime_by_correlation.setdefault(correlation, []).append(event)
    gpu = [
        e
        for e in events
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    ]
    if not gpu:
        raise ValueError("Profiler captured no CUDA activity (check CUPTI permissions)")
    # Kineto ts is microseconds relative to this trace's base, not necessarily
    # the same base on each rank. Keep base separate to avoid float precision loss.
    base_ns = raw.get("baseTimeNanoseconds")
    if not isinstance(base_ns, int):
        raise ValueError("Trace lacks baseTimeNanoseconds; cannot align ranks safely")
    # Autograd launches CUDA work on an engine thread, outside the dispatch
    # thread's annotation stack. Associate only engine-node intervals nested in
    # the synchronous B call, not arbitrary activity from other CPU threads.
    engine_intervals = {}
    engine_nodes = [
        e
        for e in events
        if e.get("cat") == "cpu_op"
        and e.get("name", "").startswith("autograd::engine::evaluate_function:")
    ]
    for label, fields in labels:
        if fields["kind"] != "B":
            continue
        grouped = {}
        for e in engine_nodes:
            if (
                e.get("pid") == label.get("pid")
                and label["ts"] <= e["ts"] < label["ts"] + label["dur"]
            ):
                grouped.setdefault((e.get("pid"), e.get("tid")), []).append(
                    (e["ts"], min(e["ts"] + e["dur"], label["ts"] + label["dur"]))
                )
        engine_intervals[label["name"]] = {k: interval_union(v) for k, v in grouped.items()}

    def contained(owner, label, same_thread=True):
        if not label["ts"] <= owner["ts"] < label["ts"] + label["dur"]:
            return False
        if (
            owner.get("pid") is not None
            and label.get("pid") is not None
            and owner["pid"] != label["pid"]
        ):
            return False
        if not same_thread or all(
            label.get(k) is None or owner.get(k) is None or label[k] == owner[k]
            for k in ("pid", "tid")
        ):
            return True
        intervals = engine_intervals.get(label["name"], {}).get(
            (owner.get("pid"), owner.get("tid")), []
        )
        index = bisect.bisect_right(intervals, [owner["ts"], float("inf")]) - 1
        return index >= 0 and intervals[index][0] <= owner["ts"] < intervals[index][1]

    def owners(activity):
        args = activity.get("args", {})
        return runtime_by_correlation.get(args.get("correlation")) or cpu_by_external.get(
            args.get("External id"), []
        )

    allocation_calls = [
        e
        for e in events
        if e.get("ph") == "X"
        and e.get("cat") in ("cuda_runtime", "cuda_driver")
        and e.get("name", "").startswith(("cudaMalloc", "cuMemAlloc"))
    ]
    details = []
    for index, event in enumerate(gpu):
        fields = [
            f
            for label, f in labels
            if f["kind"] == "step"
            and any(contained(o, label, same_thread=False) for o in owners(event))
        ]
        details.append(
            dict(
                activity_id=index,
                name=event["name"],
                category=event["cat"],
                pid=event.get("pid"),
                tid=event.get("tid"),
                args=event.get("args", {}),
                start_us=event["ts"],
                end_us=event["ts"] + event["dur"],
                communication="nccl" in event["name"].lower(),
                iteration=fields[0]["iteration"] if len(fields) == 1 else None,
            )
        )
    records, steps = [], []
    for label, fields in labels:
        start, end = label["ts"], label["ts"] + label["dur"]
        activities = []
        activity_ids = [
            i
            for i, activity in enumerate(gpu)
            if any(
                contained(owner, label, same_thread=fields["kind"] != "step")
                for owner in owners(activity)
            )
        ]
        activities = [gpu[i] for i in activity_ids]
        if not activities:
            raise ValueError(f"No correlated CUDA activity for {label['name']}")
        first = min(e["ts"] for e in activities)
        last = max(e["ts"] + e["dur"] for e in activities)
        record = dict(
            **fields,
            rank=rank,
            mode=mode,
            transport=transport,
            start_us=first,
            end_us=last,
            cpu_start_us=start,
            cpu_end_us=end,
            gpu_activity_count=len(activities),
            activity_ids=activity_ids,
            profiler_range=label["name"],
            **activity_metrics(activities),
        )
        allocations = [
            e for e in allocation_calls if contained(e, label, same_thread=fields["kind"] != "step")
        ]
        allocation_union = interval_union((e["ts"], e["ts"] + e["dur"]) for e in allocations)
        communication = [e for e in activities if "nccl" in e["name"].lower()]
        record.update(
            cuda_malloc_count=len(allocations),
            cuda_malloc_cpu_union_us=sum(b - a for a, b in allocation_union),
            cuda_malloc_cpu_sum_us=sum(e["dur"] for e in allocations),
            communication_activity_count=len(communication),
            communication_union_us=activity_metrics(communication)["gpu_activity_union_us"],
        )
        if fields["kind"] == "step":
            measured = measured_steps[str(fields["iteration"])]
            record.update(measured)
            difference = abs((last - first) / 1000 - measured["cuda_elapsed_ms"])
            # Event brackets include CPU dispatch before the first kernel and
            # after the last. Report that boundary slack rather than hiding it.
            record["cuda_event_boundary_slack_ms"] = difference
            tolerance = max(2.0, measured["cuda_elapsed_ms"] * 0.05)
            record["cuda_event_boundary_warning"] = difference > tolerance
            steps.append(record)
        else:
            records.append(record)
    result = dict(
        schema_version=SCHEMA,
        rank=rank,
        mode=mode,
        transport=transport,
        clock="Kineto same-host clock; base_ns + start_us * 1000",
        base_time_ns=base_ns,
        raw_clock_metadata={
            k: raw[k]
            for k in (
                "trace_id",
                "displayTimeUnit",
                "baseTimeNanoseconds",
                "cuda_driver_version",
                "cuda_runtime_version",
                "cupti_version",
            )
            if k in raw
        },
        gpu_activities=details,
        cuda_allocation_calls=[
            dict(
                e,
                iteration=next(
                    (
                        f["iteration"]
                        for label, f in labels
                        if f["kind"] == "step" and contained(e, label, same_thread=False)
                    ),
                    None,
                ),
            )
            for e in allocation_calls
        ],
        unassigned_gpu_activity_count=sum(d["iteration"] is None for d in details),
        communication_classification="kernel name contains nccl; raw API events retained separately",
        correlation_method="CUDA launch correlation, then CPU external ID fallback; synchronous B calls include nested autograd engine-thread nodes",
        timing_source="first/last CUDA activity correlated to labeled CPU ranges",
        config=config,
        records=sorted(records, key=lambda r: (r["iteration"], r["cpu_start_us"])),
        steps=sorted(steps, key=lambda r: r["iteration"]),
    )
    validate_compact_trace(result)
    return result


def validate_compact_trace(trace: dict, plan=None) -> None:
    if trace.get("schema_version") not in (SCHEMA, "slackpipe.figure_trace.v1"):
        raise ValueError("Unsupported compact trace schema")
    config, rank = trace["config"], trace["rank"]
    expected = {
        (kind, b, s)
        for kind in ("F", "B")
        for b in range(config["num_microbatches"])
        for s in range(config["num_stages"])
        if s % config["pp"] == rank
    }
    iterations = [step["iteration"] for step in trace["steps"]]
    if not iterations or len(set(iterations)) != len(iterations):
        raise ValueError("Missing or duplicate step envelopes")
    for record in trace["records"] + trace["steps"]:
        if (
            record["rank"] != rank
            or record["iteration"] not in iterations
            or not all(math.isfinite(record[k]) for k in ("start_us", "end_us"))
            or record["end_us"] < record["start_us"]
        ):
            raise ValueError("Invalid envelope identity or duration")
        if "gpu_envelope_us" in record and not math.isclose(
            record["end_us"] - record["start_us"], record["gpu_envelope_us"], abs_tol=0.01
        ):
            raise ValueError("Envelope inconsistent with activity intervals")
    for step in trace["steps"]:
        ops = [
            r
            for r in trace["records"]
            if r["iteration"] == step["iteration"] and r["kind"] in ("F", "B")
        ]
        keys = [(r["kind"], r["microbatch"], r["stage"]) for r in ops]
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError("Logical operation count/coverage mismatch")
        for previous, current in zip(ops, ops[1:]):
            # GPU envelopes may overlap across streams; CPU dispatch order is the plan order.
            if previous["cpu_end_us"] > current["cpu_start_us"] + 0.001:
                raise ValueError("Overlapping or out-of-order logical CPU ranges")
        if any(r["start_us"] < step["start_us"] or r["end_us"] > step["end_us"] for r in ops):
            raise ValueError("Operation outside step envelope")
        if plan is not None:
            desired = [(op.kind, op.microbatch, op.stage) for op in plan.worker_operations(rank)]
            if keys != desired:
                raise ValueError("SlackPipe trace order differs from plan")


def write_compact_trace(path: Path, trace: dict) -> None:
    validate_compact_trace(trace)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(trace, indent=2) + "\n")
    temporary.replace(path)
