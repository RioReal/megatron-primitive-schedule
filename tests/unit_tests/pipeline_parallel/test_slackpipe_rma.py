# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from megatron.core.pipeline_parallel.slackpipe.communication_rma import SlackPipeRMACommunicator
from megatron.core.pipeline_parallel.slackpipe.plan import load_slackpipe_plan, parse_slackpipe_plan
from tests.unit_tests.test_utilities import Utils


def test_m1_fixture_preserves_delayed_match():
    path = Path(__file__).parent / "slackpipe_fixtures" / "m1_delayed_match.plan.json"
    plan = load_slackpipe_plan(path, pipeline_model_parallel_size=2)
    assert plan.layer_split == (1, 7, 7, 1)
    assert plan.stage_to_worker == (0, 1, 0, 1)
    assert sum(len(ops) for ops in plan.operations) == 64
    sender = plan.worker_operations(1)[12]
    consumer = plan.worker_operations(0)[24]
    assert (sender.kind, sender.microbatch, sender.stage) == ("B", 0, 1)
    assert (consumer.kind, consumer.microbatch, consumer.stage) == ("B", 0, 0)


def test_rma_puts_complete_before_any_consumer_waits():
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("requires torchrun with two GPUs")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    Utils.initialize_model_parallel(pipeline_model_parallel_size=2)
    operations = [[], []]
    for b in range(4):
        for kind, stages in (("F", range(4)), ("B", reversed(range(4)))):
            for s in stages:
                operations[s % 2].append(dict(kind=kind, microbatch=b, stage=s))
    plan = parse_slackpipe_plan(
        dict(
            num_microbatches=4,
            num_stages=4,
            num_workers=2,
            num_layers=8,
            layer_split=[2] * 4,
            stage_to_worker=[0, 1, 0, 1],
            operations=operations,
        ),
        pipeline_model_parallel_size=2,
    )
    comm = SlackPipeRMACommunicator(plan)
    shape, dtype, device = (8, 1, 16), torch.float32, torch.device("cuda")
    sends, receives = [], []
    for s in range(3):
        for direction in ("forward", "backward"):
            sender = (s if direction == "forward" else s + 1) % 2
            for b in range(4):
                (sends if sender == dist.get_rank() else receives).append(
                    (direction, (s, s + 1), b)
                )
    for iteration in range(3):
        comm.prepare_iteration(
            receive_specs=receives, send_specs=sends, shape=shape, dtype=dtype, device=device
        )
        assert len({id(c["group"]) for c in comm.channels.values()}) == 6
        pointers = tuple(c["storage"].data_ptr() for c in comm.channels.values())
        if iteration == 0:
            original_pointers = pointers
        assert pointers == original_pointers
        direction, edge, b = sends[1]
        with pytest.raises(RuntimeError, match="FIFO"):
            comm._send(direction, edge, torch.zeros(shape, device=device), b)
        for direction, edge, b in sends:
            value = iteration * 1000 + edge[0] * 100 + (50 if direction == "backward" else 0) + b
            comm._send(direction, edge, torch.full(shape, value, dtype=dtype, device=device), b)
        # This must finish BEFORE either rank issues a wait_signal. Two-sided
        # send/recv cannot satisfy this test regardless of receive preposting.
        comm.drain_sends()
        dist.barrier(group=comm.control)
        for direction, edge, b in receives:
            actual = comm._recv(direction, edge, b, shape, dtype, device)
            expected = iteration * 1000 + edge[0] * 100 + (50 if direction == "backward" else 0) + b
            assert torch.equal(actual, torch.full_like(actual, expected))
        comm.finalize_iteration()
        comm.assert_no_outstanding_work()
        assert comm.completed_iterations == iteration + 1
    print(
        f"RMA rank={dist.get_rank()} isolated channels=6 iterations=3 mailbox_bytes={comm.mailbox_bytes} init_seconds={comm.initialization_seconds}"
    )
    comm.close()
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("external", [False, True], ids=["pp2-equivalence", "m1-delayed-match"])
def test_rma_numerical_equivalence(tmp_path, monkeypatch, external):
    from tests.unit_tests.pipeline_parallel.test_slackpipe_model_construction import (
        test_slackpipe_pp2_numerical_equivalence,
    )

    monkeypatch.setenv("SLACKPIPE_TEST_TRANSPORT", "nccl-rma")
    monkeypatch.delenv("SLACKPIPE_EXTERNAL_PP2_PLAN", raising=False)
    artifact_root = os.environ.get("SLACKPIPE_RMA_ARTIFACT_DIR")
    if artifact_root:
        monkeypatch.setenv(
            "SLACKPIPE_EXTERNAL_TRACE_DIR", str(Path(artifact_root) / ("M1" if external else "pp2"))
        )
    if external:
        path = Path(__file__).parent / "slackpipe_fixtures" / "m1_delayed_match.plan.json"
        monkeypatch.setenv("SLACKPIPE_EXTERNAL_PP2_PLAN", str(path))
    test_slackpipe_pp2_numerical_equivalence(tmp_path)


@pytest.mark.parametrize(
    "plan_name", ["M0_equal_joint", "M2_shared_slopes_joint", "M3_full_joint", "measured"]
)
def test_rma_existing_solver_plans(tmp_path, monkeypatch, plan_name):
    from tests.unit_tests.pipeline_parallel.test_slackpipe_model_construction import (
        test_slackpipe_pp2_numerical_equivalence,
    )

    root = Path(__file__).resolve().parents[3]
    if plan_name == "measured":
        path = root / "slackpipe_plans/perf/b8-n4-w2-l16-measured-cost-slackpipe.plan.json"
    else:
        path = root / "slackpipe_experiments_smoke/S_b8_seq64/plans" / f"{plan_name}.plan.json"
    if not path.exists():
        pytest.skip("requires generated solver-plan artifacts")
    monkeypatch.setenv("SLACKPIPE_TEST_TRANSPORT", "nccl-rma")
    monkeypatch.setenv("SLACKPIPE_EXTERNAL_PP2_PLAN", str(path))
    artifact_root = os.environ.get("SLACKPIPE_RMA_ARTIFACT_DIR")
    if artifact_root:
        monkeypatch.setenv("SLACKPIPE_EXTERNAL_TRACE_DIR", str(Path(artifact_root) / plan_name))
    print(f"Validating RMA solver plan {plan_name}")
    test_slackpipe_pp2_numerical_equivalence(tmp_path)
