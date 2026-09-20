"""Experiment-only collection. Never imported by ordinary training schedules."""

import hashlib
import importlib.metadata
import json
import math
import os
import re
import statistics
import subprocess
import time
import uuid
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from .figure_trace import (
    compact_profiler_trace,
    label_logical_operations,
    validate_compact_trace,
    write_compact_trace,
)

SCHEMA = "slackpipe.collection.v1"
TIMING = {
    "step": "zero_grad, iterator creation/consumption, forward/backward, pipeline communication, "
    "gradient processing performed by the schedule, optimizer.step; resident mock batches prepared before warmup",
    "excluded": "model/batch construction, validation collectives, aggregation, serialization, profiler.step",
    "cuda": "current compute stream event brackets; supported schedulers must join required communication "
    "before returning. P2P Work.wait joins the caller stream; RMA drains puts and joins receive streams. "
    "Not a guarantee for arbitrary user-created streams or asynchronous optimizers",
    "continuous": "barrier and device synchronization before window; all-device synchronization and "
    "rank barrier at end; includes dispatch gaps and collection bookkeeping. In profiling modes also "
    "includes profiler transitions/export pauses; these windows are NOT benchmark samples",
    "dispatch": "CPU enqueue duration, not CPU-to-GPU completion latency",
}


@dataclass(frozen=True)
class CollectionOptions:
    mode: str = "benchmark"
    warmup: int = 20
    iterations: int = 20
    wait: int = 2
    profiler_warmup: int = 2
    active: int = 3
    repeat: int = 3
    run_id: str | None = None
    history_entries: int = 100000

    def validate(self):
        if self.mode not in ("benchmark", "timeline", "memory"):
            raise ValueError("Unknown collection mode")
        if min(self.warmup, self.iterations, self.active, self.repeat, self.history_entries) < 1:
            raise ValueError("Warmup, iterations, active, repeat and history size must be positive")
        if min(self.wait, self.profiler_warmup) < 0:
            raise ValueError("Profiler wait/warmup must be nonnegative")
        if self.run_id is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", self.run_id):
            raise ValueError("run_id must be a filename-safe identifier")

    @property
    def steps(self):
        return (
            self.iterations
            if self.mode == "benchmark"
            else (self.wait + self.profiler_warmup + self.active) * self.repeat
        )


def capture_windows(options: CollectionOptions) -> list[list[int]]:
    options.validate()
    period = options.wait + options.profiler_warmup + options.active
    return [
        list(
            range(
                options.warmup + c * period + options.wait + options.profiler_warmup,
                options.warmup + (c + 1) * period,
            )
        )
        for c in range(options.repeat)
    ]


def profiler_settings(options: CollectionOptions) -> dict:
    enabled = options.mode != "benchmark"
    diagnostic = options.mode == "memory"
    return dict(
        enabled=enabled,
        activities=["CPU", "CUDA"] if enabled else [],
        record_shapes=diagnostic,
        with_stack=diagnostic,
        profile_memory=diagnostic,
        with_flops=False,
        with_modules=False,
        acc_events=False,
        schedule=(
            dict(
                wait=options.wait,
                warmup=options.profiler_warmup,
                active=options.active,
                repeat=options.repeat,
            )
            if enabled
            else None
        ),
        allocation_history=(
            dict(
                enabled=diagnostic,
                context="all",
                stacks="all",
                max_entries=options.history_entries,
                scope="after model/event setup, before training warmup through measurement; bounded ring history",
            )
            if diagnostic
            else None
        ),
    )


def memory_sample() -> dict:
    return dict(
        allocated_bytes=torch.cuda.memory_allocated(),
        reserved_bytes=torch.cuda.memory_reserved(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    )


def environment_metadata() -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()

    try:
        source = dict(
            commit=git("rev-parse", "HEAD"),
            diff_sha256=hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
        )
        root = Path(git("rev-parse", "--show-toplevel"))
        source["untracked_source_sha256"] = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in git("ls-files", "--others", "--exclude-standard").splitlines()
            if Path(name).suffix in (".py", ".cc", ".h", ".sh") and (root / name).is_file()
        }
    except (OSError, subprocess.CalledProcessError):
        source = dict(commit=None, diff_sha256=None)
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    packages = {}
    for name in ("transformer_engine", "mamba-ssm", "causal-conv1d", "triton"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(
        source=source,
        host=os.uname().nodename,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        nccl=torch.cuda.nccl.version(),
        gpu=prop.name,
        gpu_uuid=str(getattr(prop, "uuid", "unknown")),
        allocator_backend=torch.cuda.memory.get_allocator_backend(),
        allocator_environment={
            k: os.environ.get(k)
            for k in (
                "PYTORCH_CUDA_ALLOC_CONF",
                "PYTORCH_ALLOC_CONF",
                "PYTORCH_NO_CUDA_MEMORY_CACHING",
                "CUDA_MODULE_LOADING",
                "CUDA_DEVICE_MAX_CONNECTIONS",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        deterministic_environment={
            k: os.environ.get(k)
            for k in (
                "MAMBA_DETERMINISTIC",
                "TRITON_CACHE_AUTOTUNING",
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                "CUBLAS_WORKSPACE_CONFIG",
            )
        },
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32,
        packages=packages,
    )


def sample_summary(values: list[float]) -> dict:
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Expected finite nonnegative timing samples")
    mean = statistics.fmean(values)
    half = max(1, len(values) // 2)
    first, last = statistics.median(values[:half]), statistics.median(values[-half:])
    return dict(
        count=len(values),
        mean_ms=mean,
        median_ms=statistics.median(values),
        stddev_ms=statistics.pstdev(values),
        min_ms=min(values),
        max_ms=max(values),
        cv=statistics.pstdev(values) / mean if mean else 0,
        first_half_median_ms=first,
        last_half_median_ms=last,
        last_to_first_ratio=last / first if first else None,
        samples_discarded=0,
    )


def summarize_rank_samples(results: list[dict]) -> dict:
    """Maximum across ranks for each matching global iteration; never truncate zip."""
    if not results or len({r["rank"] for r in results}) != len(results):
        raise ValueError("Missing or duplicate rank results")
    first = results[0]
    if sorted(r["rank"] for r in results) != list(range(first["config"]["pp"])):
        raise ValueError("Incomplete rank results")
    ids = [s["iteration"] for s in first["samples"]]
    for r in results:
        if (
            r["run_id"],
            r["method"],
            r["options"],
            r["config"],
            [s["iteration"] for s in r["samples"]],
        ) != (first["run_id"], first["method"], first["options"], first["config"], ids):
            raise ValueError("Rank/run/iteration alignment mismatch")
    values = [max(r["samples"][i]["cuda_elapsed_ms"] for r in results) for i in range(len(ids))]
    return dict(
        iterations=ids,
        max_rank_iteration_ms=values,
        **sample_summary(values),
        continuous_wall_ms=max(r["continuous_wall_ms"] for r in results),
    )


def _write_new(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def collect_steps(
    *,
    step,
    optimizer,
    options: CollectionOptions,
    output: Path,
    method: str,
    transport: str,
    config: dict,
    plan=None,
    validate=None,
) -> dict:
    """Collect an unchanged full step; only profiler mode inserts label contexts.

    All ranks must call with identical options. Profile export barriers are at
    cycle boundaries, never inside a training step. Raw export happens in the
    callback; expensive correlation/plotting happens after the entire window.
    """
    options.validate()
    rank, world = dist.get_rank(), dist.get_world_size()
    identities = [options.run_id or uuid.uuid4().hex]
    dist.broadcast_object_list(identities, src=0)
    run_id = identities[0]
    settings = profiler_settings(options)
    metadata = dict(
        schema_version=SCHEMA,
        run_id=run_id,
        method=method,
        rank=rank,
        transport=transport,
        collection_mode=options.mode,
        options=asdict(options),
        profiler=settings,
        environment=environment_metadata(),
        timing_definitions=TIMING,
        config=config,
        training_warmup_iterations=list(range(options.warmup)),
        measurement_iterations=list(range(options.warmup, options.warmup + options.steps)),
    )
    configs = [None] * world
    dist.all_gather_object(configs, (asdict(options), method, config))
    if any(c != configs[0] for c in configs):
        raise ValueError("Collection options/model config differ across ranks")
    directory = output / run_id / method
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{run_id}.{method}.rank{rank}"
    # Exclusive reservation prevents reusing an old run ID from overwriting traces.
    _write_new(directory / f"{stem}.metadata.json", metadata)
    snapshots = dict(before_event_setup=memory_sample())
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(options.warmup + options.steps)
    ]
    for start, end in events:
        start.record()
        end.record()
    torch.cuda.synchronize()  # Initialize lazy event handles before training warmup.
    history = options.mode == "memory"
    if history:
        torch.cuda.memory._record_memory_history(
            enabled="all", context="all", stacks="all", max_entries=options.history_entries
        )
    warmup, samples, captures = [], [], []
    windows = capture_windows(options)
    last_result = None
    try:
        for iteration in range(options.warmup):
            start, end = events[iteration]
            start.record()
            dispatch = time.perf_counter()
            last_result = step(iteration)
            elapsed = (time.perf_counter() - dispatch) * 1000
            end.record()
            warmup.append(
                dict(iteration=iteration, cpu_dispatch_ms=elapsed, memory=memory_sample())
            )
        torch.cuda.synchronize()
        for row in warmup:
            start, end = events[row["iteration"]]
            row["cuda_elapsed_ms"] = start.elapsed_time(end)
        if validate is not None:
            validate(last_result)
        snapshots["after_training_warmup"] = memory_sample()
        torch.cuda.reset_peak_memory_stats()
        snapshots["measurement_start"] = memory_sample()

        def on_trace_ready(prof):
            cycle = len(captures)
            started = time.perf_counter()
            raw_path = directory / f"{stem}.cycle{cycle:03d}.torch.json"
            if raw_path.exists():
                raise FileExistsError(raw_path)
            prof.export_chrome_trace(str(raw_path))
            exported = time.perf_counter()
            # Serialization perturbs the next window. Align ranks after export;
            # the next cycle still executes its configured wait and warmup.
            dist.barrier()
            captures.append(
                dict(
                    cycle=cycle,
                    capture_id=f"{run_id}:cycle{cycle:03d}",
                    active_iterations=windows[cycle],
                    raw_path=str(raw_path),
                    export_ms=(exported - started) * 1000,
                    export_and_rank_alignment_ms=(time.perf_counter() - started) * 1000,
                    memory=memory_sample(),
                )
            )

        profiler = None
        if options.mode != "benchmark":
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(**settings["schedule"]),
                on_trace_ready=on_trace_ready,
                **{
                    k: settings[k]
                    for k in (
                        "record_shapes",
                        "with_stack",
                        "profile_memory",
                        "with_flops",
                        "with_modules",
                        "acc_events",
                    )
                },
            )
        dist.barrier()
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        with profiler if profiler is not None else nullcontext():
            for iteration in metadata["measurement_iterations"]:
                start, end = events[iteration]
                labels = (
                    label_logical_operations(iteration, rank, world, optimizer)
                    if profiler
                    else nullcontext()
                )
                region = (
                    torch.profiler.record_function(f"ScheduleTrace/step/i{iteration}/s-1/b-1")
                    if profiler
                    else nullcontext()
                )
                with labels, region:
                    start.record()
                    dispatch = time.perf_counter()
                    last_result = step(iteration)
                    elapsed = (time.perf_counter() - dispatch) * 1000
                    end.record()
                samples.append(
                    dict(iteration=iteration, cpu_dispatch_ms=elapsed, memory=memory_sample())
                )
                if profiler is not None:
                    profiler.step()
        torch.cuda.synchronize()
        dist.barrier()
        continuous_ms = (time.perf_counter() - wall_start) * 1000
        snapshots["measurement_end"] = memory_sample()
        for row in samples:
            start, end = events[row["iteration"]]
            row["cuda_elapsed_ms"] = start.elapsed_time(end)
        if validate is not None:
            validate(last_result)
        if history:
            snapshot_path = directory / f"{stem}.memory.pickle"
            torch.cuda.memory._dump_snapshot(str(snapshot_path))
            metadata["allocation_snapshot"] = str(snapshot_path)
    finally:
        if history:
            torch.cuda.memory._record_memory_history(enabled=None)
    result = dict(
        **metadata,
        warmup_samples=warmup,
        samples=samples,
        captures=captures,
        memory_boundaries=snapshots,
        continuous_wall_ms=continuous_ms,
        summary=sample_summary([r["cuda_elapsed_ms"] for r in samples]),
        warmup_summary=sample_summary([r["cuda_elapsed_ms"] for r in warmup]),
    )
    _write_new(directory / f"{stem}.samples.json", result)
    if profiler is not None and len(captures) != options.repeat:
        raise RuntimeError("Profiler did not export every requested cycle")
    for capture in captures:
        trace = compact_profiler_trace(
            json.loads(Path(capture["raw_path"]).read_text()),
            rank=rank,
            mode=method,
            transport=transport,
            measured_steps={str(r["iteration"]): r for r in samples},
            config=config,
        )
        if [s["iteration"] for s in trace["steps"]] != capture["active_iterations"]:
            raise RuntimeError("Profiler global iteration coverage differs from requested capture")
        trace["collection"] = dict(
            run_id=run_id,
            capture_id=capture["capture_id"],
            cycle=capture["cycle"],
            profiler=settings,
            timing_definitions=TIMING,
            environment=metadata["environment"],
            active_iterations=capture["active_iterations"],
        )
        validate_compact_trace(trace, plan)
        write_compact_trace(directory / f"{stem}.cycle{capture['cycle']:03d}.compact.json", trace)
    _write_new(directory / f"{stem}.complete.json", dict(valid=True, cycles=len(captures)))
    return result
