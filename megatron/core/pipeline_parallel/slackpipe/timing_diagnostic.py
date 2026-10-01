# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Opt-in compute-stream timing evidence, never a solver cost estimator."""

import time

import torch

from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator


class ComputeTimingDiagnostic:
    """Reuse preinitialized event pairs; resolve only at full-step completion.

    An event interval is a stream envelope, not summed kernel time. It can
    include host launch gaps and dependencies inserted inside the region. Work
    on other streams is covered only if it rejoins the recorded compute stream.
    """

    def __init__(self, operation_capacity: int):
        if operation_capacity < 1:
            raise ValueError("Positive operation capacity required")
        self.pairs = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(operation_capacity)
        ]
        # CUDA events are lazily initialized. Pay that cost before training warmup.
        for pair in self.pairs:
            for event in pair:
                event.record()
        torch.cuda.synchronize()
        self.active = False
        self.iterations = []

    def begin(self, iteration: int) -> None:
        if self.active:
            raise RuntimeError("Timing iteration already active")
        self.active = True
        self.iteration = iteration
        self.records = []
        self.p2p = []
        self.position = 0

    def compute(self, compute, identity: dict):
        if not self.active or len(self.records) >= len(self.pairs):
            raise RuntimeError("Inactive diagnostic or exhausted preallocated event capacity")
        start, end = self.pairs[len(self.records)]
        stream = torch.cuda.current_stream()
        position = self.position
        self.position += 1
        start.record(stream)
        before = time.perf_counter()
        result = compute()
        cpu_ms = (time.perf_counter() - before) * 1000
        end.record(stream)
        self.records.append(
            dict(
                **identity,
                iteration=self.iteration,
                schedule_position=position,
                compute_position=len(self.records),
                compute_stream_id=stream.cuda_stream,
                cpu_wall_ms=cpu_ms,
                existing_stage_wall_ms=None,
                wait_synchronization_ms=None,
            )
        )
        return result

    def finish(self) -> list:
        """One device completion boundary, after optimizer; no per-op waits."""
        if not self.active:
            raise RuntimeError("No timing iteration active")
        try:
            before = time.perf_counter()
            torch.cuda.synchronize()
            completion_ms = (time.perf_counter() - before) * 1000
            for record, (start, end) in zip(self.records, self.pairs):
                record["compute_gpu_ms"] = start.elapsed_time(end)
            self.iterations.append(
                dict(
                    iteration=self.iteration,
                    completion_wait_ms=completion_ms,
                    p2p_wall_ms=sum(e["p2p_wall_ms"] for e in self.p2p),
                    p2p_calls=self.p2p,
                    compute_operation_count=len(self.records),
                )
            )
            return self.records
        finally:
            self.active = False


class DiagnosticP2PCommunicator(P2PCommunicator):
    """Observe the native call boundary without changing its requests or waits."""

    def __init__(self, pp_group, config, diagnostic: ComputeTimingDiagnostic):
        super().__init__(pp_group, config)
        self.diagnostic = diagnostic

    def _communicate(self, **kwargs):
        diagnostic = self.diagnostic
        if not diagnostic.active:
            return super()._communicate(**kwargs)
        position = diagnostic.position
        diagnostic.position += 1
        before = time.perf_counter()
        result = super()._communicate(**kwargs)
        elapsed = (time.perf_counter() - before) * 1000
        diagnostic.p2p.append(
            dict(
                schedule_position=position,
                p2p_wall_ms=elapsed,
                # This inclusive host duration cannot be decomposed into NCCL
                # device service and dependency wait without a separate trace.
                wait_synchronization_ms=None,
                recv_prev=kwargs["recv_prev"],
                recv_next=kwargs["recv_next"],
                send_prev=kwargs["tensor_send_prev"] is not None,
                send_next=kwargs["tensor_send_next"] is not None,
                wait_on_reqs=kwargs.get("wait_on_reqs", True),
                native_batch_sync=bool(self.config.batch_p2p_comm and self.config.batch_p2p_sync),
            )
        )
        return result
