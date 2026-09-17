<div align="center">

Megatron-LM and Megatron Core
=============================

<h4>GPU-optimized library for training transformer models at scale</h4>

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)
[![version](https://img.shields.io/badge/release-0.15.0-green)](./CHANGELOG.md)
[![license](https://img.shields.io/badge/license-Apache-blue)](./LICENSE)

<div align="left">

## SlackPipe Integration

This personal research repository hosts two generations of scheduling work:
the earlier **megatron-primitive-schedule** work on
[`feature/schedule`](https://github.com/RioReal/megatron-primitive-schedule/tree/feature/schedule),
and the current **SlackPipe integration** on `slackpipe/mvp`. The earlier branch
is maintained separately; this branch does not merge or replace it. SlackPipe
is research code built on NVIDIA Megatron-LM, **not official NVIDIA functionality**.
Upstream attribution, documentation, and installation guidance are preserved below.

SlackPipe connects offline pipeline partition/schedule optimization to execution:

```text
ordinary Megatron calibration -> cost profile -> ./slackpipe C++ CP-SAT optimizer
    -> slackpipe.plan -> Megatron SlackPipe execution
```

### Current functionality

- Physical PP=1 and PP=2, logical/VPP stages, cyclic stage-to-worker placement,
  nonuniform partitions, and solver-defined worker-local F/B operation orders.
- Plan parsing validates coverage, FIFO ordering, worker ownership, and the
  combined computation/worker-order DAG. Plans determine the custom Megatron
  layout and VPP size; a conflicting explicit layout is rejected.
- Measured forward/backward cost calibration, homogeneous affine costs, and
  heterogeneous sequential decoder layers with exact contiguous global-layer
  ranges. Layer identities and compatible hidden-state interfaces are preserved
  across chunks. Class-cost fitting and model/profile hashes support provenance
  validation for the range-cost path.
- `nccl-p2p` transport and **experimental** `nccl-rma`, with persistent transport
  caching and explicit lifecycle cleanup. The RMA implementation is validated
  only in the local two-GPU setup, not as a general deployment guarantee.
- Optional runtime traces, NVTX labels, and selected-step torch.profiler captures
  with compact per-rank GPU envelopes and paper-style PNG/PDF schedule figures.

The current correctness baseline is fixed-shape FP32, TP=DP=CP=1, dropout=0,
and untied input/output embeddings. Targeted homogeneous and heterogeneous PP=1
and PP=2 tests compare loss, every logical parameter gradient, and post-SGD
parameters against ordinary Megatron; supported tested cases have matched exactly.
This is a research checkpoint, not a claim of validation for arbitrary models.

### Monorepo layout

| Path | Purpose |
| --- | --- |
| `megatron/core/pipeline_parallel/slackpipe/` | Runtime, transports, plan/cost schemas, manifests, tracing |
| `tests/unit_tests/pipeline_parallel/test_slackpipe_*.py` | Correctness and regression tests |
| `tests/unit_tests/pipeline_parallel/slackpipe_perf_benchmark.py` | Model construction, calibration and controlled benchmark harness |
| `tools/slackpipe_experiment_driver.py` | Homogeneous cost-model ablation orchestration |
| `tools/slackpipe_heterogeneous_experiment.py` | Heterogeneous class/range calibration |
| `tools/slackpipe_plot_experiments.py` | Experiment plotting |
| `tools/capture_schedule_trace.py`, `tools/plot_schedule_trace.py` | Opt-in profiler capture and schedule figures |
| `slackpipe/` | C++ CP-SAT optimizer, DAG evaluator, plan exporter, tests, and evaluation tools |

The runtime and optimizer are now published together. `slackpipe/` is a normal
tracked directory, not a nested repository or submodule. It imports solver
checkpoint `415ae8da60ede4aa69c5703cbecb6dafabec040b` and the calibrated/v2
range-cost work required by this runtime. Build products, dependencies, and
generated results remain excluded. Existing plans can be consumed without
installing OR-Tools. See the [solver guide](slackpipe/README.md) and
[import scope](slackpipe/docs/monorepo.md).

### Build the optimizer

CMake >=3.24, a C++20 compiler, and Ninja are required for these commands. No
new build system or host Megatron installation is needed. From the monorepo
root, use the prepared development container:

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev \
  cmake -S slackpipe -B slackpipe/build/no-or -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSLACKPIPE_ENABLE_ORTOOLS=OFF
docker exec -w /workspace/Megatron-LM slackpipe-dev \
  cmake --build slackpipe/build/no-or -j2
docker exec -w /workspace/Megatron-LM slackpipe-dev \
  ctest --test-dir slackpipe/build/no-or --output-on-failure
```

This dependency-light build includes the deterministic baselines, evaluator,
exporter, and tests; it does not run CP-SAT. The optional solver-only image can
be built with `docker build -t slackpipe-cpp:local -f slackpipe/Dockerfile slackpipe`.
For CP-SAT, use the established local OR-Tools image (or an equivalent environment
providing the OR-Tools CMake package and shared libraries):

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace/Megatron-LM" -w /workspace/Megatron-LM \
  --entrypoint /bin/bash slackpipe-ortools-runtime:local -lc \
  'BUILD_DIR="$PWD/slackpipe/build/release" BUILD_JOBS=2 \
   SLACKPIPE_OR_TEST_FILTER=".*" bash slackpipe/scripts/validate_ortools_evaluation.sh'
```

The script configures OR-Tools ON with `ORTOOLS_PREFIX` (default `/opt/or-tools`),
builds, checks capabilities, runs CTest, and exercises tiny validated CLI cases.
OR-Tools and container images remain external system dependencies, not vendored
binaries. The image name can be replaced by your locally provisioned equivalent.

### Schemas

| Schema | Role |
| --- | --- |
| `slackpipe.plan.v1` | Dimensions, split, placement, ordered worker operations, solver metadata and abstract costs |
| `slackpipe.plan.v2` | Exact half-open global-layer ranges/cuts, model/profile provenance, with compatible split metadata |
| `slackpipe.cost_profile.v1` | Homogeneous forward/backward per-layer slopes and stage-role biases |
| `slackpipe.cost_profile.v2` | Heterogeneous class costs expanded into per-layer/prefix costs for exact contiguous cuts |
| `slackpipe.model_manifest.v1` | Global decoder layer identities, configuration classes, and model fingerprint |

See the authoritative [plan parser](megatron/core/pipeline_parallel/slackpipe/plan.py),
[cost-profile helpers](megatron/core/pipeline_parallel/slackpipe/cost_profile.py),
and [manifest validation](megatron/core/pipeline_parallel/slackpipe/manifest.py).
Keep a v2 plan's referenced cost profile available at its `cost_model.path`;
the runtime verifies its hash and model identity.

### Docker and usage

Development uses image `megatron-slackpipe:0.18.2`, persistent container
`slackpipe-dev`, and this checkout mounted at `/workspace/Megatron-LM`.
Megatron execution and tests run inside that container; no host Megatron
dependencies are required. The commands below assume the existing prepared
environment and `/opt/venv/bin/python` interpreter.

```bash
docker exec -it -w /workspace/Megatron-LM slackpipe-dev bash
```

Add the following to an otherwise configured Megatron training launch (model,
data, optimizer and distributed launch arguments are still required):

```text
--pipeline-schedule slackpipe
--slackpipe-plan path/to/plan.json
--pipeline-model-parallel-size 2
--tensor-model-parallel-size 1
--context-parallel-size 1
--slackpipe-transport nccl-p2p
--slackpipe-runtime fast
```

Physical PP must match the plan's worker count; cyclic stage count must be
divisible by PP. Do not supply a duplicate custom layout. Normal Megatron
schedule selection remains unchanged unless SlackPipe is explicitly selected.
Use `--slackpipe-trace path/to/trace.json` for per-rank operation order and
`--slackpipe-disable-nvtx` to disable default SlackPipe NVTX labels.

Select `--slackpipe-transport nccl-rma` only on a compatible environment:
it requires NCCL >=2.29 and PyTorch's private NCCL symmetric-memory APIs.
See [RMA requirements and lifecycle](megatron/core/pipeline_parallel/slackpipe/README.rma.md).
Call `shutdown_slackpipe_runtime()` collectively before destroying distributed
groups when embedding the runtime; the provided harnesses perform this cleanup.

### Calibration and plan generation

**Calibration is intentionally intrusive:** synchronized profiling measures
primitive compute costs during ordinary Megatron execution. **Performance runs
must disable calibration profiling**, and should keep diagnostic operation
profiling separate from measured iterations. Profiler figures are not benchmarks.

For a small heterogeneous calibration, run from the repository root:

```bash
docker exec -w /workspace/Megatron-LM -e PYTHONPATH=. slackpipe-dev \
  /opt/venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tools/slackpipe_heterogeneous_experiment.py \
  --output-dir slackpipe_profiles/heterogeneous \
  --warmup-iterations 5 --iterations 10
```

Use the OR-enabled environment to export an evaluator-validated plan. The mounted
checkout path is kept identical so the plan's cost-profile reference is also
accessible to Megatron:

```bash
docker exec --user "$(id -u):$(id -g)" -w /workspace/Megatron-LM \
  slackpipe-dev mkdir -p slackpipe_plans
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace/Megatron-LM" -w /workspace/Megatron-LM \
  --entrypoint ./slackpipe/build/release/slackpipe_cli slackpipe-ortools-runtime:local \
  --algorithm slackpipe --split-mode global --B 4 --N 4 --J 2 --L 12 \
  --cost-profile slackpipe_profiles/heterogeneous/cost_profile.json \
  --time-limit-seconds 60 --num-workers 1 --require-optimal false \
  --output-prefix slackpipe_plans/heterogeneous \
  --emit-plan slackpipe_plans/heterogeneous.plan.json
```

This exports through the canonical SlackPipe predecessor-validation path;
`--algorithm optimize-joint` selects the unrestricted joint solver directly.
The reusable homogeneous host
orchestrator supports `SLACKPIPE_REPO`, `SLACKPIPE_CONTAINER_REPO`,
`SLACKPIPE_CONTAINER`, and `SLACKPIPE_ORTOOLS_IMAGE` overrides. Its default solver
image is a local development image, not a public dependency supplied here.
Consult each tool's `--help` before launching an experiment campaign.

For trace capture and plotting, see the
[figure workflow](megatron/core/pipeline_parallel/slackpipe/README.figure_trace.md).
Generated plans, profiles, experiment outputs, and raw traces are ignored and
must be regenerated; the small solver-order plans under test fixtures are
intentional regression inputs, not benchmark artifacts.

### Targeted tests

All applicable targeted tests pass in the validated two-GPU FP32 environment;
configuration-specific tests are exercised in their corresponding PP/transport
runs. Run the focused suite with one rank, then two ranks with each transport.
World-size-specific tests skip only in the inapplicable invocation. Solver-order
regressions use checked-in fixtures, not optional campaign artifacts. See the
[validation matrix and skip audit](docs/slackpipe_validation_matrix.md).
Manifest tests are in `test_slackpipe_heterogeneous.py`.

```bash
docker exec -w /workspace/Megatron-LM -e PYTHONPATH=. \
  -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 slackpipe-dev bash -c \
  '/opt/venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 \
  -m pytest tests/unit_tests/pipeline_parallel/test_slackpipe_*.py -q -ra'

docker exec -w /workspace/Megatron-LM -e PYTHONPATH=. \
  -e CUDA_VISIBLE_DEVICES=0,1 -e SLACKPIPE_TEST_TRANSPORT=nccl-p2p \
  -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 slackpipe-dev bash -c \
  '/opt/venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m pytest tests/unit_tests/pipeline_parallel/test_slackpipe_*.py -q -ra'

docker exec -w /workspace/Megatron-LM -e PYTHONPATH=. \
  -e CUDA_VISIBLE_DEVICES=0,1 -e SLACKPIPE_TEST_TRANSPORT=nccl-rma \
  -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 slackpipe-dev bash -c \
  '/opt/venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m pytest tests/unit_tests/pipeline_parallel/test_slackpipe_*.py -q -ra'
```

These cover plan v1/v2, cost v1/v2, manifests, PP=1/PP=2 construction and numerical
equivalence, P2P isolation, RMA delayed matching, lifecycle caching, and trace
export. The RMA tests require the compatible stack described above. Additional
reuse/teardown stress is available in `slackpipe_rma_lifecycle_stress.py` with
`--cycles`, `--reuse`, and `--managed-groups` options.

### Current limitations

- PP>2 is not validated; the current runtime is restricted to PP=1/PP=2.
- SlackPipe TP/DP/CP integration, distributed optimizer, activation recomputation,
  CUDA graphs, variable sequence lengths, and communication-overlap optimizations
  are unsupported. BF16/FP16 are not validated in this development path.
- Heterogeneous layers must be sequential decoder blocks with compatible
  hidden-state interfaces. Mamba, MoE, and MTP topologies are not supported here.
- RMA multi-node deployment and multi-host profiler clock alignment are unsupported.
- P2P retains matching-receive dependencies: DAG/FIFO validation alone does not
  guarantee progress for every solver order. Delayed-matching schedules are
  exercised with experimental RMA; validate a plan/transport pair before use.
- Current heterogeneous construction is exercised through the supplied model
  harness, not a universal model-provider adapter.

---

## About

This repository contains two components: **Megatron-LM** and **Megatron Core**.

**Megatron-LM** is a reference example that includes Megatron Core plus pre-configured training scripts. Best for research teams, learning distributed training, and quick experimentation.

**Megatron Core** is a composable library with GPU-optimized building blocks for custom training frameworks. It provides transformer building blocks, advanced parallelism strategies (TP, PP, DP, EP, CP), mixed precision support (FP16, BF16, FP8, FP4), and model architectures. Best for framework developers and ML engineers building custom training pipelines.

**[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)** provides bidirectional Hugging Face ↔ Megatron checkpoint conversion with production-ready recipes.

## Getting Started

**Install from PyPI:**

```bash
uv pip install megatron-core
```

**Or clone and install from source:**

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
uv pip install -e .
```

> **Note:** Building from source can use a lot of memory. If the build runs out of memory, limit parallel compilation jobs by setting `MAX_JOBS` (e.g. `MAX_JOBS=4 uv pip install -e .`).

For NGC container setup and all installation options, see the **[Installation Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/install.html)**.

- **[Your First Training Run](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/quickstart.html)** - End-to-end training examples with data preparation
- **[Parallelism Strategies](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)** - Scale training across GPUs with TP, PP, DP, EP, and CP
- **[Contribution Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/developer/contribute.html)** - How to contribute to Megatron Core

# Latest News

- **[2026/03]** **Deprecating Python 3.10 support:** We're officially dropping Python 3.10 support with the upcoming 0.17.0 release. Downstream applications must raise their lower boundary to 3.12 to stay compatible with MCore.
- **[2026/01]** **[Dynamic Context Parallelism](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/)** - Up to 1.48x speedup for variable-length sequence training with adaptive CP sizing.
- **[2025/12]** **Megatron Core development has moved to GitHub!** All development and CI now happens in the open. We welcome community contributions.
- **[2025/10]** **[Megatron Dev Branch](https://github.com/NVIDIA/Megatron-LM/tree/dev)** - early access branch with experimental features.
- **[2025/10]** **[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)** - Bidirectional converter for interoperability between Hugging Face and Megatron checkpoints, featuring production-ready recipes for popular models.
- **[2025/08]** **[MoE Q3-Q4 2025 Roadmap](https://github.com/NVIDIA/Megatron-LM/issues/1729)** - Comprehensive roadmap for MoE features including DeepSeek-V3, Qwen3, advanced parallelism strategies, FP8 optimizations, and Blackwell performance enhancements.
- **[2025/08]** **[GPT-OSS Model](https://github.com/NVIDIA/Megatron-LM/issues/1739)** - Advanced features including YaRN RoPE scaling, attention sinks, and custom activation functions are being integrated into Megatron Core.
- **[2025/06]** **[Megatron MoE Model Zoo](https://github.com/yanring/Megatron-MoE-ModelZoo)** - Best practices and optimized configurations for training DeepSeek-V3, Mixtral, and Qwen3 MoE models with performance benchmarking and checkpoint conversion tools.
- **[2025/05]** Megatron Core v0.11.0 brings new capabilities for multi-data center LLM training ([blog](https://developer.nvidia.com/blog/turbocharge-llm-training-across-long-haul-data-center-networks-with-nvidia-nemo-framework/)).

<details>
<summary>Previous News</summary>

- **[2024/07]** Megatron Core v0.7 improves scalability and training resiliency and adds support for multimodal training ([blog](https://developer.nvidia.com/blog/train-generative-ai-models-more-efficiently-with-new-nvidia-Megatron-Core-functionalities/)).
- **[2024/06]** Megatron Core added supports for Mamba-based models. Check out our paper [An Empirical Study of Mamba-based Language Models](https://arxiv.org/pdf/2406.07887) and [code example](https://github.com/NVIDIA/Megatron-LM/tree/ssm/examples/mamba).
- **[2024/01 Announcement]** NVIDIA has released the core capabilities in **Megatron-LM** into [**Megatron Core**](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) in this repository. Megatron Core expands upon Megatron-LM's GPU-optimized techniques with more cutting-edge innovations on system-level optimizations, featuring composable and modular APIs.

</details>

# Project Structure

```
Megatron-LM/
├── megatron/
│   ├── core/                    # Megatron Core (kernels, parallelism, building blocks)
│   │   ├── models/              # Transformer models
│   │   ├── transformer/         # Transformer building blocks
│   │   ├── tensor_parallel/     # Tensor parallelism
│   │   ├── pipeline_parallel/   # Pipeline parallelism
│   │   ├── distributed/         # Distributed training (FSDP, DDP)
│   │   ├── optimizer/           # Optimizers
│   │   ├── datasets/            # Dataset loaders
│   │   ├── inference/           # Inference engines and server
│   │   └── export/              # Model export (e.g. TensorRT-LLM)
│   ├── training/                # Training scripts
│   ├── legacy/                  # Legacy components
│   ├── post_training/           # Post-training (quantization, distillation, pruning, etc.)
│   └── rl/                      # Reinforcement learning (RLHF, etc.)
├── examples/                    # Ready-to-use training examples
├── tools/                       # Utility tools
├── tests/                       # Comprehensive test suite
└── docs/                        # Documentation
```

# Performance Benchmarking

For our latest performance benchmarking results, please refer to [NVIDIA Megatron Bridge Performance Summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html).

Our codebase efficiently trains models from 2B to 462B parameters across thousands of GPUs, achieving up to **47% Model FLOP Utilization (MFU)** on H100 clusters.

![Model table](images/model_table.png)

**Benchmark Configuration:**

- **Vocabulary size**: 131,072 tokens
- **Sequence length**: 4096 tokens
- **Model scaling**: Varied hidden size, attention heads, and layers to achieve target parameter counts
- **Communication optimizations**: Fine-grained overlapping with DP (`--overlap-grad-reduce`, `--overlap-param-gather`), TP (`--tp-comm-overlap`), and PP (enabled by default)

**Key Results:**

- **6144 H100 GPUs**: Successfully benchmarked 462B parameter model training
- **Superlinear scaling**: MFU increases from 41% to 47-48% with model size
- **End-to-end measurement**: Throughputs include all operations (data loading, optimizer steps, communication, logging)
- **Production ready**: Full training pipeline with checkpointing and fault tolerance
- *Note: Performance results measured without training to convergence*

## Weak Scaling Results

Our weak scaled results show superlinear scaling (MFU increases from 41% for the smallest model considered to 47-48% for the largest models); this is because larger GEMMs have higher arithmetic intensity and are consequently more efficient to execute.

![Weak scaling](images/weak_scaling.png)

## Strong Scaling Results

We also strong scaled the standard GPT-3 model (our version has slightly more than 175 billion parameters due to larger vocabulary size) from 96 H100 GPUs to 4608 GPUs, using the same batch size of 1152 sequences throughout. Communication becomes more exposed at larger scale, leading to a reduction in MFU from 47% to 42%.

![Strong scaling](images/strong_scaling.png)

# Roadmaps

- **[MoE Roadmap](https://github.com/NVIDIA/Megatron-LM/issues/1729)** - DeepSeek-V3, Qwen3, advanced parallelism, FP8 optimizations, and Blackwell enhancements

# Resources

## Getting Help

- 📖 **[Documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)** - Official documentation
- 🐛 **[Issues](https://github.com/NVIDIA/Megatron-LM/issues)** - Bug reports and feature requests

## Contributing

We ❤️ contributions! Ways to contribute:

- 🐛 **Report bugs** - Help us improve reliability
- 💡 **Suggest features** - Shape the future of Megatron Core
- 📝 **Improve docs** - Make Megatron Core more accessible
- 🔧 **Submit PRs** - Contribute code improvements

**→ [Contributing Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/developer/contribute.html)**

## Citation

If you use Megatron in your research or project, we appreciate that you use the following citations:

```bibtex
@article{megatron-lm,
  title={Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism},
  author={Shoeybi, Mohammad and Patwary, Mostofa and Puri, Raul and LeGresley, Patrick and Casper, Jared and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:1909.08053},
  year={2019}
}
```
