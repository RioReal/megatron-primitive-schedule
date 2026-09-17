#!/usr/bin/env bash
# Run inside nvcr.io/nvidia/pytorch:26.01-py3 with the project's dependencies.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
"${PYTHON:-python}" - <<'PY'
import torch
import transformer_engine
import mamba_ssm
import causal_conv1d
from mamba_ssm.utils.determinism import use_deterministic_mode
import torch.distributed._symmetric_memory as symm
assert torch.cuda.device_count() == 4, "requires 4 CUDA devices"
for rank in range(4):
    name = torch.cuda.get_device_name(rank)
    print(rank, name, torch.cuda.get_device_properties(rank).total_memory)
    assert "A100" in name, "validation target is A100 SXM x4"
print("torch", torch.__version__, "CUDA", torch.version.cuda, "NCCL", torch.cuda.nccl.version())
print("TE", transformer_engine.__version__, "Mamba", mamba_ssm.__version__, "conv", causal_conv1d.__version__)
print("RMA available", torch.cuda.nccl.version() >= (2, 29, 0) and all(hasattr(symm, n) for n in ("put_signal", "wait_signal", "rendezvous")))
PY
nvidia-smi topo -m
