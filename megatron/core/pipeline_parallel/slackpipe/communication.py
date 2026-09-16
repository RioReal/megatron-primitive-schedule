# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""SlackPipe point-to-point communication helpers."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from .plan import SlackPipePlan


@dataclass
class _OutstandingSend:
    work: dist.Work
    tensor: torch.Tensor
    description: str


class SlackPipeCommunicator:
    """NCCL communicators for SlackPipe logical stage edges.

    Each cross-worker logical edge gets a distinct process group, even when
    several logical edges map onto the same physical rank pair.
    """

    def __init__(self, plan: SlackPipePlan):
        if not dist.is_available() or not dist.is_initialized():
            raise ValueError("SlackPipe distributed communication requires torch.distributed")

        self.plan = plan
        self.rank = dist.get_rank()
        self.edge_groups: Dict[Tuple[int, int], dist.ProcessGroup] = {}
        self._outstanding_sends: List[_OutstandingSend] = []
        self._outstanding_receives: List[dist.Work] = []
        self._expected_send_orders: Dict[Tuple[str, Tuple[int, int]], List[int]] = {}
        self._send_positions: Dict[Tuple[str, Tuple[int, int]], int] = {}
        self._max_outstanding_sends = 1024

        for stage in range(plan.num_stages - 1):
            src_worker = plan.stage_to_worker[stage]
            dst_worker = plan.stage_to_worker[stage + 1]
            if src_worker == dst_worker:
                continue
            ranks = sorted((src_worker, dst_worker))
            self.edge_groups[(stage, stage + 1)] = dist.new_group(ranks=ranks, backend="nccl")
        self._warm_up_edge_groups()

    def has_remote_forward_edge(self, stage: int) -> bool:
        return (stage, stage + 1) in self.edge_groups

    def has_remote_backward_edge(self, stage: int) -> bool:
        return (stage - 1, stage) in self.edge_groups

    def send_forward(
        self, stage: int, tensor: torch.Tensor, microbatch: Optional[int] = None
    ) -> None:
        self._validate_send_order("forward", (stage, stage + 1), microbatch)
        self._send(
            edge=(stage, stage + 1),
            dst=self.plan.stage_to_worker[stage + 1],
            tensor=tensor,
            description=f"F stage {stage}->{stage + 1}",
        )

    def recv_forward(
        self,
        stage: int,
        shape: Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
        requires_grad: bool,
        tensor: Optional[torch.Tensor] = None,
        microbatch: Optional[int] = None,
    ) -> torch.Tensor:
        edge = (stage - 1, stage)
        tensor = self._recv(
            edge=edge,
            src=self.plan.stage_to_worker[stage - 1],
            shape=shape,
            dtype=dtype,
            device=device,
            description=f"F stage {stage - 1}->{stage}",
            tensor=tensor,
        )
        if requires_grad:
            tensor.requires_grad_(True)
        return tensor

    def send_backward(
        self, stage: int, tensor: torch.Tensor, microbatch: Optional[int] = None
    ) -> None:
        self._validate_send_order("backward", (stage - 1, stage), microbatch)
        self._send(
            edge=(stage - 1, stage),
            dst=self.plan.stage_to_worker[stage - 1],
            tensor=tensor,
            description=f"B stage {stage}->{stage - 1}",
        )

    def recv_backward(
        self,
        stage: int,
        shape: Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
        tensor: Optional[torch.Tensor] = None,
        microbatch: Optional[int] = None,
    ) -> torch.Tensor:
        edge = (stage, stage + 1)
        return self._recv(
            edge=edge,
            src=self.plan.stage_to_worker[stage + 1],
            shape=shape,
            dtype=dtype,
            device=device,
            description=f"B stage {stage + 1}->{stage}",
            tensor=tensor,
        )

    def prepare_iteration(
        self,
        *,
        receive_specs: Sequence[Tuple[str, Tuple[int, int], int]],
        send_specs: Sequence[Tuple[str, Tuple[int, int], int]],
        shape: Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
        receive_window: Optional[int] = None,
    ) -> None:
        self._build_expected_send_orders(send_specs)
        del receive_specs, shape, dtype, device, receive_window

    def finalize_iteration(self) -> None:
        for order_key, expected in self._expected_send_orders.items():
            sent = self._send_positions.get(order_key, 0)
            if sent != len(expected):
                raise RuntimeError(
                    f"SlackPipe sent {sent}/{len(expected)} messages for {order_key}"
                )
        self._expected_send_orders.clear()
        self._send_positions.clear()

    def drain_sends(self) -> None:
        for outstanding in self._outstanding_sends:
            outstanding.work.wait()
        self._outstanding_sends.clear()

    def assert_no_outstanding_work(self) -> None:
        if self._outstanding_sends:
            raise RuntimeError(
                f"SlackPipe has {len(self._outstanding_sends)} outstanding send requests"
            )
        if self._outstanding_receives:
            raise RuntimeError(
                f"SlackPipe has {len(self._outstanding_receives)} outstanding receive requests"
            )

    def close(self) -> None:
        self.assert_no_outstanding_work()
        for group in self.edge_groups.values():
            dist.destroy_process_group(group)
        self.edge_groups.clear()

    def _build_expected_send_orders(
        self, send_specs: Sequence[Tuple[str, Tuple[int, int], int]]
    ) -> None:
        orders: Dict[Tuple[str, Tuple[int, int]], List[int]] = {}
        for direction, edge, microbatch in send_specs:
            orders.setdefault((direction, edge), []).append(microbatch)
        for order_key, microbatches in orders.items():
            if microbatches != sorted(microbatches):
                raise RuntimeError(
                    f"SlackPipe send order for {order_key} is not FIFO: {microbatches}"
                )
        self._expected_send_orders = orders
        self._send_positions = {order_key: 0 for order_key in orders}

    def _validate_send_order(
        self, direction: str, edge: Tuple[int, int], microbatch: Optional[int]
    ) -> None:
        if microbatch is None or not self._expected_send_orders:
            return
        order_key = (direction, edge)
        expected = self._expected_send_orders[order_key]
        position = self._send_positions[order_key]
        if position >= len(expected) or expected[position] != microbatch:
            raise RuntimeError(
                f"SlackPipe sent {direction}:{edge}:b{microbatch} out of FIFO order; "
                f"expected microbatch {expected[position] if position < len(expected) else 'none'}"
            )
        self._send_positions[order_key] = position + 1

    def _send(
        self, *, edge: Tuple[int, int], dst: int, tensor: torch.Tensor, description: str
    ) -> None:
        if edge not in self.edge_groups:
            raise ValueError(f"SlackPipe edge {edge} is not a remote logical edge")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"SlackPipe can only send Tensor payloads, got {type(tensor).__name__}")
        send_tensor = tensor.detach().contiguous()
        work = dist.batch_isend_irecv(
            [dist.P2POp(dist.isend, send_tensor, dst, group=self.edge_groups[edge])]
        )[0]
        self._outstanding_sends.append(_OutstandingSend(work, send_tensor, description))
        self._drain_completed_sends()
        while len(self._outstanding_sends) > self._max_outstanding_sends:
            oldest = self._outstanding_sends.pop(0)
            oldest.work.wait()

    def _recv(
        self,
        *,
        edge: Tuple[int, int],
        src: int,
        shape: Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
        description: str,
        tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if edge not in self.edge_groups:
            raise ValueError(f"SlackPipe edge {edge} is not a remote logical edge")
        if tensor is None:
            tensor = torch.empty(tuple(shape), dtype=dtype, device=device)
        elif (
            tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype or tensor.device != device
        ):
            raise ValueError(
                "SlackPipe preallocated receive buffer does not match expected tensor metadata"
            )
        work = dist.batch_isend_irecv(
            [dist.P2POp(dist.irecv, tensor, src, group=self.edge_groups[edge])]
        )[0]
        self._outstanding_receives.append(work)
        work.wait()
        self._outstanding_receives.remove(work)
        return tensor

    def _drain_completed_sends(self) -> None:
        pending = []
        for outstanding in self._outstanding_sends:
            if outstanding.work.is_completed():
                outstanding.work.wait()
            else:
                pending.append(outstanding)
        self._outstanding_sends = pending

    def _warm_up_edge_groups(self) -> None:
        for edge, group in self.edge_groups.items():
            device = torch.device("cuda", torch.cuda.current_device())
            send_tensor = torch.zeros((), dtype=torch.float32, device=device)
            recv_tensor = torch.empty((), dtype=torch.float32, device=device)
            peer = self.plan.stage_to_worker[
                edge[1] if self.rank == self.plan.stage_to_worker[edge[0]] else edge[0]
            ]
            works = dist.batch_isend_irecv(
                [
                    dist.P2POp(dist.isend, send_tensor, peer, group=group),
                    dist.P2POp(dist.irecv, recv_tensor, peer, group=group),
                ]
            )
            for work in works:
                work.wait()
