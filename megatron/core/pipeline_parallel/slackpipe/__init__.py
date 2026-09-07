# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from .plan import SlackPipeOperation, SlackPipePlan, load_slackpipe_plan

__all__ = [
    "SlackPipeOperation",
    "SlackPipePlan",
    "forward_backward_slackpipe",
    "load_slackpipe_plan",
]


def __getattr__(name):
    if name == "forward_backward_slackpipe":
        from .schedule import forward_backward_slackpipe

        return forward_backward_slackpipe
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
