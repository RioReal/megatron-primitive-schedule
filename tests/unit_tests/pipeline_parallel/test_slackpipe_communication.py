# Copyright (c) 2026 NVIDIA CORPORATION. All rights reserved.

import os

import pytest
import torch

from megatron.core.pipeline_parallel.slackpipe.communication import SlackPipeCommunicator
from megatron.core.pipeline_parallel.slackpipe.plan import parse_slackpipe_plan
from tests.unit_tests.test_utilities import Utils


def _pp2_plan():
    return parse_slackpipe_plan(
        {
            "num_microbatches": 1,
            "num_stages": 4,
            "num_workers": 2,
            "num_layers": 8,
            "layer_split": [1, 2, 2, 3],
            "stage_to_worker": [0, 1, 0, 1],
            "operations": [
                [
                    {"kind": "F", "microbatch": 0, "stage": 0},
                    {"kind": "F", "microbatch": 0, "stage": 2},
                    {"kind": "B", "microbatch": 0, "stage": 2},
                    {"kind": "B", "microbatch": 0, "stage": 0},
                ],
                [
                    {"kind": "F", "microbatch": 0, "stage": 1},
                    {"kind": "F", "microbatch": 0, "stage": 3},
                    {"kind": "B", "microbatch": 0, "stage": 3},
                    {"kind": "B", "microbatch": 0, "stage": 1},
                ],
            ],
        },
        pipeline_model_parallel_size=2,
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_slackpipe_logical_edge_communicators_are_isolated():
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("run with torchrun --nproc-per-node 2")

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
    )

    communicator = SlackPipeCommunicator(_pp2_plan())
    assert list(communicator.edge_groups) == [(0, 1), (1, 2), (2, 3)]
    assert len({id(group) for group in communicator.edge_groups.values()}) == 3

    rank = torch.distributed.get_rank()
    device = torch.device("cuda")
    shape = (1,)
    dtype = torch.float32

    if rank == 0:
        communicator.send_forward(0, torch.tensor([10.0], device=device))
        communicator.send_forward(2, torch.tensor([30.0], device=device))
        forward_from_stage_1 = communicator.recv_forward(2, shape, dtype, device, False)
        backward_from_stage_1 = communicator.recv_backward(0, shape, dtype, device)
        communicator.send_backward(2, torch.tensor([200.0], device=device))
        backward_from_stage_3 = communicator.recv_backward(2, shape, dtype, device)

        assert forward_from_stage_1.item() == 20.0
        assert backward_from_stage_1.item() == 100.0
        assert backward_from_stage_3.item() == 300.0
    else:
        forward_from_stage_0 = communicator.recv_forward(1, shape, dtype, device, False)
        forward_from_stage_2 = communicator.recv_forward(3, shape, dtype, device, False)
        communicator.send_forward(1, torch.tensor([20.0], device=device))
        communicator.send_backward(1, torch.tensor([100.0], device=device))
        backward_from_stage_2 = communicator.recv_backward(1, shape, dtype, device)
        communicator.send_backward(3, torch.tensor([300.0], device=device))

        assert forward_from_stage_0.item() == 10.0
        assert forward_from_stage_2.item() == 30.0
        assert backward_from_stage_2.item() == 200.0

    communicator.drain_sends()
    communicator.assert_no_outstanding_work()
    Utils.destroy_model_parallel()
