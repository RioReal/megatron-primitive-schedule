"""Explicit ownership for process groups created by SlackPipe regression tests."""

from contextlib import contextmanager

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.slackpipe.schedule import shutdown_slackpipe_runtime
from tests.unit_tests.test_utilities import Utils


@contextmanager
def managed_model_parallel_groups():
    """Release test-owned groups, not just Megatron's references to them.

    PyTorch's private registry is inspected only in this test harness. Megatron
    destroy_model_parallel deliberately does not destroy most NCCL groups.
    Preserve the default group and any groups owned by the enclosing fixture.
    """
    original = Utils.initialize_model_parallel
    owned = []

    def cleanup():
        shutdown_slackpipe_runtime()
        if owned:
            torch.cuda.synchronize()
            parallel_state.destroy_model_parallel()
            for group in reversed(owned):
                if group in dist.distributed_c10d._world.pg_map:
                    dist.destroy_process_group(group)
            owned.clear()
            Utils.inited = False

    def initialize(*args, **kwargs):
        cleanup()
        before = set(dist.distributed_c10d._world.pg_map)
        try:
            return original(*args, **kwargs)
        finally:
            owned.extend(
                group
                for group in dist.distributed_c10d._world.pg_map
                if group not in before and group is not dist.group.WORLD
            )

    Utils.initialize_model_parallel = staticmethod(initialize)
    try:
        yield
    finally:
        cleanup()
        Utils.initialize_model_parallel = staticmethod(original)
