# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Experimental NCCL 2.29 mailbox transport using PyTorch symmetric memory.

Each (logical edge, direction) owns a communicator and a persistent allocation.
Slots are never reused within a step. A step-end compute event and a CPU group
barrier prevent the next step from overwriting activations still used by autograd.
There are no matching receives or registration calls in the operation loop.
"""

import math
import time
from datetime import timedelta
from typing import Sequence

import torch
import torch.distributed as dist

from .plan import SlackPipePlan, validate_cyclic_placement
from .topology import logical_edges


class SlackPipeRMACommunicator:
    """Persistent, direction-isolated FP32 mailboxes on logical edge subgroups."""

    def __init__(self, plan: SlackPipePlan):
        if not dist.is_initialized() or dist.get_world_size() != plan.num_workers:
            raise ValueError("SlackPipe nccl-rma world size must match plan workers")
        if torch.cuda.nccl.version() < (2, 29, 0):
            raise RuntimeError("SlackPipe nccl-rma requires NCCL >= 2.29")
        validate_cyclic_placement(plan)
        import torch.distributed._symmetric_memory as symm

        if not all(hasattr(symm, name) for name in ("put_signal", "wait_signal", "rendezvous")):
            raise RuntimeError("This PyTorch build lacks NCCL symmetric-memory RMA bindings")
        self.symm = symm
        self.num_microbatches = plan.num_microbatches
        self.stage_to_worker = plan.stage_to_worker
        self.edges = logical_edges(plan)
        self.rank = dist.get_rank()
        self.channels = {}
        self.pending = []
        self.active = False
        self.closed = False
        self.completed_iterations = 0
        self.metadata = None
        self.initialization_seconds = 0.0
        self.mailbox_bytes = 0
        self.slot_elements = 0
        self.device_memory_delta_bytes = 0
        self.expected_sends = set()
        self.expected_receives = set()
        self.control = dist.new_group(backend="gloo", timeout=timedelta(seconds=60))

    def _initialize(self, shape, dtype, device):
        if dtype != torch.float32 or device.type != "cuda":
            raise ValueError("SlackPipe nccl-rma supports fixed CUDA FP32 tensors only")
        started = time.perf_counter()
        free_before, _ = torch.cuda.mem_get_info()
        self.symm.set_backend("NCCL")
        self.shape = tuple(shape)
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.slot_elements = ((math.prod(shape) * 4 + 4095) // 4096) * 1024
        # PyTorch registers one data window and one internal signal-pad window
        # per allocation/group. Rendezvous of subviews reuses those windows.
        for edge in self.edges:
            for direction in ("forward", "backward"):
                group = dist.new_group(ranks=list(edge.ranks), backend="nccl")
                if self.rank not in edge.ranks:
                    continue
                dist.all_reduce(torch.zeros(1, device=self.device), group=group)
                storage = self.symm.empty(
                    (self.num_microbatches, self.slot_elements), dtype=dtype, device=self.device
                )
                # This PyTorch rendezvous uses storage.data_ptr(), ignoring a
                # tensor's storage_offset. DLPack rebases storage without a copy
                # and retains the parent allocation through its deleter.
                slots = [
                    torch.from_dlpack(storage[b, : math.prod(shape)].view(shape))
                    for b in range(self.num_microbatches)
                ]
                handles = [self.symm.rendezvous(slot, group) for slot in slots]
                # Initialize NCCL's lazy RMA resources collectively, outside the
                # training loop. Only the channel sender writes its source slot.
                key = edge.channel(direction)
                _, _, sender, receiver = key
                peer = receiver if self.rank == sender else sender
                peer_index = edge.ranks.index(peer)
                slots[0].zero_()
                torch.cuda.current_stream().synchronize()
                dist.barrier(group=group)
                if self.rank == sender:
                    self.symm.put_signal(slots[0], handles[0], peer_index)
                else:
                    self.symm.wait_signal(handles[0], peer_index)
                torch.cuda.current_stream().synchronize()
                self.channels[key] = {
                    "group": group,
                    "storage": storage,
                    "slots": slots,
                    "handles": handles,
                    "stream": torch.cuda.Stream(),
                    "sender": sender,
                    "peer_index": peer_index,
                    "position": 0,
                }
                self.mailbox_bytes += storage.numel() * storage.element_size()
        torch.cuda.current_stream().synchronize()
        self.initialization_seconds = time.perf_counter() - started
        # Includes driver/NCCL allocations, which PyTorch's caching allocator
        # statistics do not see. This is a device-wide measurement.
        self.device_memory_delta_bytes = free_before - torch.cuda.mem_get_info()[0]

    def statistics(self) -> dict:
        return {
            "transport": "nccl-rma",
            "channel_count": len(self.channels),
            "data_window_count": len(self.channels),
            "data_window_bytes": self.num_microbatches * self.slot_elements * 4,
            "signal_window_count": len(self.channels),
            "signal_window_bytes": self.symm.get_signal_pad_size(),
            "mailbox_bytes": self.mailbox_bytes,
            "initialization_seconds": self.initialization_seconds,
            "device_memory_delta_bytes": self.device_memory_delta_bytes,
            "completed_iterations": self.completed_iterations,
            "outstanding_puts": len(self.pending),
            "active_iteration": self.active,
        }

    def prepare_iteration(
        self,
        *,
        receive_specs: Sequence[tuple],
        send_specs: Sequence[tuple],
        shape: Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if self.closed:
            raise RuntimeError("SlackPipe RMA communicator is closed")
        if self.active or self.pending:
            raise RuntimeError("SlackPipe RMA previous iteration is still active")
        metadata = (tuple(shape), dtype, device)
        if self.metadata is None:
            self._initialize(shape, dtype, device)
            self.metadata = metadata
        elif metadata != self.metadata:
            raise ValueError("SlackPipe RMA mailbox tensor metadata changed")
        self.expected_sends = set(send_specs)
        self.expected_receives = set(receive_specs)
        if len(self.expected_sends) != len(send_specs) or len(self.expected_receives) != len(
            receive_specs
        ):
            raise ValueError("SlackPipe RMA duplicate message specification")
        for channel in self.channels.values():
            channel["position"] = 0
        self.active = True

    def _channel(self, direction, edge, microbatch, sending):
        key = (direction, edge, microbatch)
        expected = self.expected_sends if sending else self.expected_receives
        if not self.active or key not in expected:
            raise RuntimeError(f"Unexpected SlackPipe RMA message {key}")
        source, destination = (self.stage_to_worker[s] for s in edge)
        if direction == "backward":
            source, destination = destination, source
        channel = self.channels[(edge, direction, source, destination)]
        if (channel["sender"] == self.rank) != sending:
            raise RuntimeError(f"Wrong SlackPipe RMA channel direction: {key}")
        if microbatch != channel["position"]:
            raise RuntimeError(f"SlackPipe RMA FIFO violation: {key}")
        channel["position"] += 1
        expected.remove(key)
        return channel

    def _send(self, direction, edge, tensor, microbatch):
        if (
            tuple(tensor.shape) != self.shape
            or tensor.dtype != torch.float32
            or tensor.device != self.device
        ):
            raise ValueError("SlackPipe RMA send tensor metadata mismatch")
        channel = self._channel(direction, edge, microbatch, True)
        source = tensor.detach().contiguous()
        stream = channel["stream"]
        # Producer compute -> this channel's put. No put -> unrelated compute edge.
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            staging = channel["slots"][microbatch]
            staging.copy_(source)
            self.symm.put_signal(staging, channel["handles"][microbatch], channel["peer_index"])
            done = torch.cuda.Event()
            done.record()
        self.pending.append((done, source))

    def _recv(
        self, direction, edge, microbatch, shape, dtype, device, requires_grad=False, tensor=None
    ):
        if tensor is not None or (tuple(shape), dtype, device) != self.metadata:
            raise ValueError("SlackPipe RMA receive tensor metadata mismatch")
        channel = self._channel(direction, edge, microbatch, False)
        # This API consumes ONE further signal, advancing NCCL's cumulative
        # counter. Passing b+1 on every wait would incorrectly overcount.
        with torch.cuda.stream(channel["stream"]):
            self.symm.wait_signal(channel["handles"][microbatch], channel["peer_index"])
        torch.cuda.current_stream().wait_stream(channel["stream"])
        # Give autograd ordinary PyTorch-owned storage. Models may retain their
        # last input after backward/cache teardown; such references must not
        # defer window deregistration beyond communicator destruction.
        return channel["slots"][microbatch].clone().detach().requires_grad_(requires_grad)

    def send_forward(self, stage: int, tensor: torch.Tensor, microbatch: int | None = None) -> None:
        self._send("forward", (stage, stage + 1), tensor, microbatch)

    def send_backward(
        self, stage: int, tensor: torch.Tensor, microbatch: int | None = None
    ) -> None:
        self._send("backward", (stage - 1, stage), tensor, microbatch)

    def recv_forward(
        self,
        stage: int,
        shape: Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
        requires_grad: bool,
        tensor: torch.Tensor | None = None,
        microbatch: int | None = None,
    ) -> torch.Tensor:
        return self._recv(
            "forward", (stage - 1, stage), microbatch, shape, dtype, device, requires_grad, tensor
        )

    def recv_backward(
        self,
        stage: int,
        shape: Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
        tensor: torch.Tensor | None = None,
        microbatch: int | None = None,
    ) -> torch.Tensor:
        return self._recv(
            "backward", (stage, stage + 1), microbatch, shape, dtype, device, tensor=tensor
        )

    def drain_sends(self) -> None:
        for done, source in self.pending:
            done.synchronize()
        self.pending.clear()

    def finalize_iteration(self) -> None:
        if self.expected_sends or self.expected_receives:
            raise RuntimeError("SlackPipe RMA iteration has missing sends or unconsumed slots")
        # Backward has released all schedule activation dictionaries before this
        # call. Complete their GPU uses before allowing peer slot reuse.
        torch.cuda.current_stream().synchronize()
        dist.barrier(group=self.control)
        self.active = False
        self.completed_iterations += 1

    def assert_no_outstanding_work(self) -> None:
        if self.pending or self.active or self.expected_sends or self.expected_receives:
            raise RuntimeError("SlackPipe RMA has outstanding iteration state")

    def close(self) -> None:
        if self.closed:
            return
        self.assert_no_outstanding_work()
        for channel in self.channels.values():
            channel["stream"].synchronize()
            channel["handles"].clear()
            channel["slots"].clear()
            del channel["storage"]
            dist.destroy_process_group(channel["group"])
        self.channels.clear()
        dist.destroy_process_group(self.control)
        self.control = None
        self.mailbox_bytes = 0
        self.metadata = None
        self.closed = True
