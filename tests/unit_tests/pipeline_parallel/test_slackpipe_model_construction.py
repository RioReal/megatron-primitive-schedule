# Copyright (c) 2026 NVIDIA CORPORATION. All rights reserved.

import json
import os
import re
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec as gpt_te_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.global_vars import set_args
from megatron.training.training import get_model
from tests.unit_tests.test_utilities import Utils, clear_nvte_env_vars

NUM_LAYERS = 8
SLACKPIPE_LAYOUT = "Et|t*2|t|t*4L"
SLACKPIPE_SPLIT = [1, 2, 1, 4]
SLACKPIPE_PP2_LAYOUT = "Et|t*2|t*2|t*3L"
SLACKPIPE_PP2_SPLIT = [1, 2, 2, 3]


def _set_minimal_training_args(**overrides):
    args = SimpleNamespace(
        create_all_gather_group=False,
        distributed_timeout_minutes=30,
        load=None,
        init_model_with_meta_device=False,
        use_torch_fsdp2=False,
        use_cpu_initialization=True,
        use_megatron_fsdp=False,
        fp16=False,
        bf16=False,
        pipeline_schedule="slackpipe",
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=4,
        pipeline_model_parallel_layout=SLACKPIPE_LAYOUT,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    set_args(args)
    return args


def _gpt_model_provider(
    pre_process=True,
    post_process=True,
    vp_stage=None,
    config=None,
    pg_collection=None,
):
    return GPTModel(
        config=config,
        transformer_layer_spec=gpt_te_spec(),
        vocab_size=128,
        max_sequence_length=8,
        pre_process=pre_process,
        post_process=post_process,
        position_embedding_type="rope",
        vp_stage=vp_stage,
        pg_collection=pg_collection,
        share_embeddings_and_output_weights=False,
    )


def _gpt_local_model_provider(
    pre_process=True,
    post_process=True,
    vp_stage=None,
    config=None,
    pg_collection=None,
):
    return GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=128,
        max_sequence_length=8,
        pre_process=pre_process,
        post_process=post_process,
        position_embedding_type="rope",
        vp_stage=vp_stage,
        pg_collection=pg_collection,
        share_embeddings_and_output_weights=False,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_slackpipe_pp1_constructs_logical_vpp_chunks():
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("run PP=1 SlackPipe construction test without torchrun")

    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    parallel_state.set_virtual_pipeline_model_parallel_world_size(4)
    _set_minimal_training_args()
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    config = TransformerConfig(
        num_layers=NUM_LAYERS,
        hidden_size=128,
        num_attention_heads=8,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=4,
        pipeline_model_parallel_layout=SLACKPIPE_LAYOUT,
        pipeline_schedule="slackpipe",
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )

    model = get_model(
        _gpt_model_provider,
        model_type=ModelType.encoder_or_decoder,
        wrap_with_ddp=False,
        config=config,
    )

    assert len(model) == 4
    assert [chunk.vp_stage for chunk in model] == [0, 1, 2, 3]
    assert [len(chunk.decoder.layers) for chunk in model] == [1, 2, 1, 4]
    assert [[layer.layer_number for layer in chunk.decoder.layers] for chunk in model] == [
        [1],
        [2, 3],
        [4],
        [5, 6, 7, 8],
    ]
    assert [chunk.pre_process for chunk in model] == [True, False, False, False]
    assert [chunk.post_process for chunk in model] == [False, False, False, True]

    Utils.destroy_model_parallel()
    parallel_state.set_virtual_pipeline_model_parallel_world_size(None)


def _make_equivalence_config(pipeline_schedule="default", vpp=None, layout=None):
    return TransformerConfig(
        num_layers=NUM_LAYERS,
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=vpp,
        pipeline_model_parallel_layout=layout,
        pipeline_schedule=pipeline_schedule,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )


def _make_pp2_equivalence_config(pipeline_schedule="default"):
    return TransformerConfig(
        num_layers=NUM_LAYERS,
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
        pipeline_model_parallel_layout=SLACKPIPE_PP2_LAYOUT,
        pipeline_schedule=pipeline_schedule,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        batch_p2p_comm=True,
        overlap_p2p_comm=False,
    )


def _build_model(
    config,
    pipeline_schedule="default",
    pp_size=1,
    vpp=None,
    layout=None,
    provider=None,
):
    _set_minimal_training_args(
        pipeline_schedule=pipeline_schedule,
        pipeline_model_parallel_size=pp_size,
        virtual_pipeline_model_parallel_size=vpp,
        pipeline_model_parallel_layout=layout,
    )
    torch.manual_seed(456)
    model_parallel_cuda_manual_seed(456)
    return get_model(
        provider or _gpt_model_provider,
        model_type=ModelType.encoder_or_decoder,
        wrap_with_ddp=False,
        config=config,
    )


def _batch_iterator(batches):
    for batch in batches:
        yield {key: value.clone() for key, value in batch.items()}


def _make_batches(num_microbatches=4):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(789)
    batches = []
    for _ in range(num_microbatches):
        tokens = torch.randint(0, 128, (1, 8), generator=generator, device="cpu").cuda()
        labels = torch.randint(0, 128, (1, 8), generator=generator, device="cpu").cuda()
        position_ids = torch.arange(8, device="cuda").view(1, -1)
        batches.append({"tokens": tokens, "labels": labels, "position_ids": position_ids})
    return batches


def _forward_step_func(data_iterator, model):
    batch = next(data_iterator)
    output_tensor = model(
        batch["tokens"],
        batch["position_ids"],
        None,
        labels=batch["labels"],
    )

    def loss_func(output_tensor):
        loss = output_tensor.float().mean()
        return loss, {"loss": loss.detach()}

    return output_tensor, loss_func


def _write_slackpipe_plan(path):
    operations = []
    for microbatch in range(4):
        operations.extend(
            {"kind": "F", "microbatch": microbatch, "stage": stage} for stage in range(4)
        )
        operations.extend(
            {"kind": "B", "microbatch": microbatch, "stage": stage}
            for stage in reversed(range(4))
        )
    path.write_text(
        json.dumps(
            {
                "num_microbatches": 4,
                "num_stages": 4,
                "num_workers": 1,
                "num_layers": NUM_LAYERS,
                "layer_split": SLACKPIPE_SPLIT,
                "stage_to_worker": [0, 0, 0, 0],
                "operations": [operations],
            }
        ),
        encoding="utf-8",
    )


def _write_slackpipe_pp2_plan(path):
    worker_operations = [[], []]
    for microbatch in range(4):
        worker_operations[0].extend(
            [
                {"kind": "F", "microbatch": microbatch, "stage": 0},
                {"kind": "F", "microbatch": microbatch, "stage": 2},
                {"kind": "B", "microbatch": microbatch, "stage": 2},
                {"kind": "B", "microbatch": microbatch, "stage": 0},
            ]
        )
        worker_operations[1].extend(
            [
                {"kind": "F", "microbatch": microbatch, "stage": 1},
                {"kind": "F", "microbatch": microbatch, "stage": 3},
                {"kind": "B", "microbatch": microbatch, "stage": 3},
                {"kind": "B", "microbatch": microbatch, "stage": 1},
            ]
        )
    path.write_text(
        json.dumps(
            {
                "num_microbatches": 4,
                "num_stages": 4,
                "num_workers": 2,
                "num_layers": NUM_LAYERS,
                "layer_split": SLACKPIPE_PP2_SPLIT,
                "stage_to_worker": [0, 1, 0, 1],
                "operations": worker_operations,
            }
        ),
        encoding="utf-8",
    )


def _logical_named_parameters(model):
    chunks = model if isinstance(model, list) else [model]
    logical_params = {}
    decoder_layer_pattern = re.compile(r"^decoder\.layers\.(\d+)\.(.*)$")
    for chunk in chunks:
        local_to_global = {
            local_idx: layer.layer_number - 1
            for local_idx, layer in enumerate(chunk.decoder.layers)
        }
        for name, param in chunk.named_parameters():
            match = decoder_layer_pattern.match(name)
            if match:
                local_idx = int(match.group(1))
                name = f"decoder.layers.{local_to_global[local_idx]}.{match.group(2)}"
            assert name not in logical_params, f"duplicate logical parameter {name}"
            logical_params[name] = param
    return logical_params


def _max_abs_parameter_diff(left, right):
    assert set(left) == set(right)
    return max((left[name].detach() - right[name].detach()).abs().max().item() for name in left)


def _max_abs_gradient_diff(left, right):
    assert set(left) == set(right)
    max_diff = 0.0
    for name in left:
        left_grad = left[name].grad
        right_grad = right[name].grad
        assert (left_grad is None) == (right_grad is None), name
        if left_grad is not None:
            max_diff = max(max_diff, (left_grad - right_grad).abs().max().item())
    return max_diff


def _copy_parameters(source, destination):
    assert set(destination).issubset(set(source))
    for name, param in destination.items():
        param.data.copy_(source[name].data)


def _max_abs_distributed(value):
    tensor = torch.tensor([value], dtype=torch.float32, device="cuda")
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return tensor.item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_slackpipe_pp1_numerical_equivalence(tmp_path):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("run PP=1 SlackPipe equivalence test without torchrun")

    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    parallel_state.set_virtual_pipeline_model_parallel_world_size(None)
    baseline_model = _build_model(_make_equivalence_config(), provider=_gpt_local_model_provider)

    parallel_state.set_virtual_pipeline_model_parallel_world_size(4)
    slackpipe_model = _build_model(
        _make_equivalence_config("slackpipe", vpp=4, layout=SLACKPIPE_LAYOUT),
        pipeline_schedule="slackpipe",
        vpp=4,
        layout=SLACKPIPE_LAYOUT,
        provider=_gpt_local_model_provider,
    )

    baseline_params = _logical_named_parameters(baseline_model)
    slackpipe_params = _logical_named_parameters(slackpipe_model)
    initial_param_diff = _max_abs_parameter_diff(baseline_params, slackpipe_params)

    baseline_optimizer = torch.optim.SGD(baseline_params.values(), lr=0.01)
    slackpipe_optimizer = torch.optim.SGD(slackpipe_params.values(), lr=0.01)
    baseline_optimizer.zero_grad(set_to_none=True)
    slackpipe_optimizer.zero_grad(set_to_none=True)

    batches = _make_batches()
    baseline_losses = get_forward_backward_func()(
        forward_step_func=_forward_step_func,
        data_iterator=_batch_iterator(batches),
        model=baseline_model,
        num_microbatches=4,
        seq_length=8,
        micro_batch_size=1,
        forward_only=False,
    )

    plan_path = tmp_path / "slackpipe_plan.json"
    _write_slackpipe_plan(plan_path)
    slackpipe_losses = get_forward_backward_func(
        pipeline_schedule="slackpipe",
        slackpipe_plan_path=str(plan_path),
    )(
        forward_step_func=_forward_step_func,
        data_iterator=[_batch_iterator(batches) for _ in range(4)],
        model=slackpipe_model,
        num_microbatches=4,
        seq_length=8,
        micro_batch_size=1,
        forward_only=False,
    )

    loss_diff = max(
        (baseline_loss["loss"] - slackpipe_loss["loss"]).abs().max().item()
        for baseline_loss, slackpipe_loss in zip(baseline_losses, slackpipe_losses)
    )
    grad_diff = _max_abs_gradient_diff(baseline_params, slackpipe_params)

    baseline_optimizer.step()
    slackpipe_optimizer.step()
    post_step_param_diff = _max_abs_parameter_diff(baseline_params, slackpipe_params)

    print(
        "SlackPipe PP=1 equivalence max abs diffs: "
        f"initial_params={initial_param_diff:.8e}, "
        f"loss={loss_diff:.8e}, "
        f"grads={grad_diff:.8e}, "
        f"post_step_params={post_step_param_diff:.8e}"
    )

    assert initial_param_diff == 0.0
    assert loss_diff == 0.0
    assert grad_diff == 0.0
    assert post_step_param_diff == 0.0

    Utils.destroy_model_parallel()
    parallel_state.set_virtual_pipeline_model_parallel_world_size(None)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_slackpipe_pp2_numerical_equivalence(tmp_path):
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("run with torchrun --nproc-per-node 2")

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    clear_nvte_env_vars()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    baseline_model = _build_model(_make_equivalence_config(), provider=_gpt_local_model_provider)
    baseline_params = _logical_named_parameters(baseline_model)
    baseline_optimizer = torch.optim.SGD(baseline_params.values(), lr=0.01)
    baseline_optimizer.zero_grad(set_to_none=True)
    batches = _make_batches()
    baseline_losses = []
    baseline_module = baseline_model[0] if isinstance(baseline_model, list) else baseline_model
    for batch in batches:
        output_tensor = baseline_module(
            batch["tokens"],
            batch["position_ids"],
            None,
            labels=batch["labels"],
        )
        loss = output_tensor.float().mean()
        baseline_losses.append({"loss": (loss / 4).detach()})
        (loss / 4).backward()

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
    )

    try:
        slackpipe_model = _build_model(
            _make_pp2_equivalence_config("slackpipe"),
            pipeline_schedule="slackpipe",
            pp_size=2,
            vpp=2,
            layout=SLACKPIPE_PP2_LAYOUT,
            provider=_gpt_local_model_provider,
        )

        slackpipe_params = _logical_named_parameters(slackpipe_model)
        _copy_parameters(baseline_params, slackpipe_params)
        baseline_subset = {name: baseline_params[name] for name in slackpipe_params}
        initial_param_diff = _max_abs_parameter_diff(baseline_subset, slackpipe_params)
        slackpipe_optimizer = torch.optim.SGD(slackpipe_params.values(), lr=0.01)
        slackpipe_optimizer.zero_grad(set_to_none=True)

        plan_path = tmp_path / "slackpipe_pp2_plan.json"
        _write_slackpipe_pp2_plan(plan_path)
        slackpipe_losses = get_forward_backward_func(
            pipeline_schedule="slackpipe",
            slackpipe_plan_path=str(plan_path),
        )(
            forward_step_func=_forward_step_func,
            data_iterator=[_batch_iterator(batches) for _ in range(2)],
            model=slackpipe_model,
            num_microbatches=4,
            seq_length=8,
            micro_batch_size=1,
            forward_only=False,
        )

        local_loss_diff = 0.0
        if slackpipe_losses:
            assert len(baseline_losses) == len(slackpipe_losses)
            local_loss_diff = max(
                (baseline_loss["loss"] - slackpipe_loss["loss"]).abs().max().item()
                for baseline_loss, slackpipe_loss in zip(baseline_losses, slackpipe_losses)
            )
        loss_diff = _max_abs_distributed(local_loss_diff)
        grad_diff = _max_abs_distributed(
            _max_abs_gradient_diff(baseline_subset, slackpipe_params)
        )

        baseline_optimizer.step()
        slackpipe_optimizer.step()
        post_step_param_diff = _max_abs_distributed(
            _max_abs_parameter_diff(baseline_subset, slackpipe_params)
        )
        initial_param_diff = _max_abs_distributed(initial_param_diff)

        if torch.distributed.get_rank() == 0:
            print(
                "SlackPipe PP=2 equivalence max abs diffs: "
                f"initial_params={initial_param_diff:.8e}, "
                f"loss={loss_diff:.8e}, "
                f"grads={grad_diff:.8e}, "
                f"post_step_params={post_step_param_diff:.8e}"
            )

        assert initial_param_diff == 0.0
        assert loss_diff == 0.0
        assert grad_diff == 0.0
        assert post_step_param_diff == 0.0
    finally:
        parallel_state.destroy_model_parallel()
        Utils.inited = False
        parallel_state.set_virtual_pipeline_model_parallel_world_size(None)
