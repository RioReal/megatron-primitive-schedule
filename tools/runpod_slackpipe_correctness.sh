#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash tools/runpod_slackpipe_verify.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export MAMBA_DETERMINISTIC=1 TRITON_CACHE_AUTOTUNING=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export SLACKPIPE_TEST_TRANSPORT="${SLACKPIPE_TEST_TRANSPORT:-nccl-p2p}"
exec "${PYTHON:-python}" -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m pytest 'tests/unit_tests/pipeline_parallel/test_slackpipe_hybrid.py::test_hybrid_numerical_equivalence[4]' -vv -s -ra
