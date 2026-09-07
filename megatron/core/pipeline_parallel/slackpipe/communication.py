# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""SlackPipe point-to-point communication helpers."""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

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

        for stage in range(plan.num_stages - 1):
            src_worker = plan.stage_to_worker[stage]
            dst_worker = plan.stage_to_worker[stage + 1]
            if src_worker == dst_worker:
                continue
            ranks = sorted((src_worker, dst_worker))
            self.edge_groups[(stage, stage + 1)] = dist.new_group(ranks=ranks, backend="nccl")

    def has_remote_forward_edge(self, stage: int) -> bool:
        return (stage, stage + 1) in self.edge_groups

    def has_remote_backward_edge(self, stage: int) -> bool:
        return (stage - 1, stage) in self.edge_groups

    def send_forward(self, stage: int, tensor: torch.Tensor) -> None:
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
    ) -> torch.Tensor:
        tensor = self._recv(
            edge=(stage - 1, stage),
            src=self.plan.stage_to_worker[stage - 1],
            shape=shape,
            dtype=dtype,
            device=device,
            description=f"F stage {stage - 1}->{stage}",
        )
        if requires_grad:
            tensor.requires_grad_(True)
        return tensor

    def send_backward(self, stage: int, tensor: torch.Tensor) -> None:
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
    ) -> torch.Tensor:
        return self._recv(
            edge=(stage, stage + 1),
            src=self.plan.stage_to_worker[stage + 1],
            shape=shape,
            dtype=dtype,
            device=device,
            description=f"B stage {stage + 1}->{stage}",
        )

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

    def _send(
        self,
        *,
        edge: Tuple[int, int],
        dst: int,
        tensor: torch.Tensor,
        description: str,
    ) -> None:
        if edge not in self.edge_groups:
            raise ValueError(f"SlackPipe edge {edge} is not a remote logical edge")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"SlackPipe can only send Tensor payloads, got {type(tensor).__name__}")
        send_tensor = tensor.detach().contiguous()
        work = dist.isend(send_tensor, dst=dst, group=self.edge_groups[edge])
        self._outstanding_sends.append(_OutstandingSend(work, send_tensor, description))

    def _recv(
        self,
        *,
        edge: Tuple[int, int],
        src: int,
        shape: Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
        description: str,
    ) -> torch.Tensor:
        if edge not in self.edge_groups:
            raise ValueError(f"SlackPipe edge {edge} is not a remote logical edge")
        tensor = torch.empty(tuple(shape), dtype=dtype, device=device)
        work = dist.irecv(tensor, src=src, group=self.edge_groups[edge])
        self._outstanding_receives.append(work)
        work.wait()
        self._outstanding_receives.remove(work)
        return tensor
