# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from .communication import SlackPipeCommunicator
from .plan import (
    SLACKPIPE_PLAN_SCHEMA_VERSION,
    SLACKPIPE_PLAN_SCHEMA_VERSION_V2,
    SlackPipeOperation,
    SlackPipePlan,
    derive_pipeline_model_parallel_layout,
    load_slackpipe_plan,
    validate_plan_parallel_layout,
)

__all__ = [
    "SLACKPIPE_PLAN_SCHEMA_VERSION",
    "SLACKPIPE_PLAN_SCHEMA_VERSION_V2",
    "SlackPipeOperation",
    "SlackPipePlan",
    "SlackPipeCommunicator",
    "derive_pipeline_model_parallel_layout",
    "forward_backward_slackpipe",
    "load_slackpipe_plan",
    "shutdown_slackpipe_runtime",
    "validate_plan_parallel_layout",
]


def __getattr__(name):
    if name == "shutdown_slackpipe_runtime":
        from .schedule import shutdown_slackpipe_runtime

        return shutdown_slackpipe_runtime
    if name == "forward_backward_slackpipe":
        from .schedule import forward_backward_slackpipe

        return forward_backward_slackpipe
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
