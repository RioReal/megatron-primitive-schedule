# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pure logical topology shared by transport construction and structural tests."""

from dataclasses import dataclass

from .plan import validate_plan_parallel_layout


@dataclass(frozen=True)
class LogicalEdge:
    stages: tuple[int, int]
    workers: tuple[int, int]

    @property
    def ranks(self):
        return tuple(sorted(self.workers))

    def channel(self, direction):
        source, destination = self.workers
        if direction == "backward":
            source, destination = destination, source
        elif direction != "forward":
            raise ValueError(f"Invalid direction: {direction}")
        return self.stages, direction, source, destination


def logical_edges(plan):
    validate_plan_parallel_layout(plan, plan.num_workers)
    return tuple(
        LogicalEdge((s, s + 1), (plan.stage_to_worker[s], plan.stage_to_worker[s + 1]))
        for s in range(plan.num_stages - 1)
        if plan.stage_to_worker[s] != plan.stage_to_worker[s + 1]
    )


def local_chunk(plan, stage):
    validate_plan_parallel_layout(plan, plan.num_workers)
    if not 0 <= stage < plan.num_stages:
        raise ValueError(f"Invalid logical stage: {stage}")
    return stage // plan.num_workers
