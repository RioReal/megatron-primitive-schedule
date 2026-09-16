"""Cache contract tests without symmetric-memory or multi-rank allocation."""

from dataclasses import replace

import pytest
import torch

from megatron.core.pipeline_parallel.slackpipe import communication_rma, schedule
from tests.unit_tests.pipeline_parallel.slackpipe_rma_lifecycle_stress import tiny_plan


def test_compatible_transport_cache_and_shutdown(monkeypatch):
    class Communicator:
        def __init__(self, plan):
            self.closed = False
            self.active = False

        def assert_no_outstanding_work(self):
            if self.active:
                raise RuntimeError("active")

        def close(self):
            self.assert_no_outstanding_work()
            self.closed = True

    monkeypatch.setattr(communication_rma, "SlackPipeRMACommunicator", Communicator)
    context = (object(), object())
    monkeypatch.setattr(schedule, "_distributed_context", lambda: context)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    plan = tiny_plan()

    def get(p=plan, shape=(64, 1, 128), dtype=torch.float32):
        return schedule._get_rma_transport(p, shape, dtype, torch.device("cuda"))

    first = get()
    assert get(replace(plan, layer_split=(1, 2, 2, 3))) is first
    schedule.clear_slackpipe_plan_cache()
    assert get() is first and not first.closed
    assert get(shape=(128, 1, 128)) is not first
    assert get(replace(plan, num_microbatches=8)) is not first
    first.active = True
    with pytest.raises(RuntimeError, match="active"):
        schedule.shutdown_slackpipe_runtime()
    first.active = False
    context = (object(), object())
    with pytest.raises(RuntimeError, match="shutdown_slackpipe_runtime"):
        get()
    schedule.shutdown_slackpipe_runtime()
    assert first.closed
    assert not schedule._RUNTIME_CACHE and not schedule._RMA_TRANSPORT_CACHE
    schedule.shutdown_slackpipe_runtime()
