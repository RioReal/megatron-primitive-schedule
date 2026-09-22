# Multi-model Real-system Evaluation

This is a random-initialized, fixed-token, unwrapped-SGD scheduling experiment,
not a pretrained-model quality evaluation or production optimizer benchmark.
Both model families use the same full training step and collector. No solver,
transport, native 1F1B, or native interleaved scheduling semantics are changed.

## Architecture Audit

| Existing component | Reuse in the generic workflow |
| --- | --- |
| `tools/run_slackpipe_nemotron_h8b_pp4.py` | Subprocess timeouts/process-group cleanup, source hashing, deterministic environment. Old CLI remains unchanged. |
| `tools/runpod_*.sh` | Existing Nemotron acceptance wrappers remain usable. |
| `tools/slackpipe_nemotron_worker.py` | Global-name initialization, full training step, finite checks, class-identifiable calibration partitions, collector and transport teardown. Model provider, VPP and metadata now accept the generic adapter. |
| `slackpipe_perf_benchmark.py` / construction helpers | Existing resident mock batches, Megatron get_model and forward/loss helpers. |
| `collection.py` / `figure_trace.py` | Unprofiled measurement, synchronized continuous wall window, rank samples, multi-cycle raw/compact traces, allocation diagnostics and operation correlation. |
| `cost_profile.py` / heterogeneous experiment tools | Existing homogeneous shared-slope/stage-bias fitting and heterogeneous class/range fitting; neither is replaced. |
| `slackpipe_experiment_driver.py` | Sample/run aggregation and run-level confidence interval definition. |
| `plot_schedule_trace.py` | Existing correlation, rank/capture alignment and plotting; optional native 1F1B third panel. |

The new `tools/slackpipe_eval_config.py` defines the strict
`slackpipe.eval_model.v1` adapter schema. `slackpipe_eval_worker.py` selects model
construction and calibration family; `run_slackpipe_eval.py` gates immutable
attempts with acceptance receipts; `run_slackpipe_real_system_campaign.py`
rotates independent method runs; `slackpipe_eval_tables.py` derives tables.
The old specialized driver retains its defaults, separate from this preset.

## Model Specifications

All eight files live in `configs/slackpipe_eval/`. Required fields include model
identity/family/scale/source/designation/scaling rule; all layer, hidden, FFN,
attention, vocabulary and sequence dimensions; normalization/activation/position
and layer types; target count; and sequence/micro/global-batch/precision defaults.
Hybrid files include the complete final global block sequence and Mamba dimensions.
Only the explicitly supported bias-free RMSNorm/GQA architectures are counted.

| Config | Exact trainable parameters | Difference from label | Designation |
| --- | ---: | ---: | --- |
| `llama_4b.json` | 4,026,731,520 | +26,731,520 | LLaMA-style research |
| `llama_8b.json` | 8,053,329,920 | +53,329,920 | LLaMA-style research |
| `llama_16b.json` | 16,221,926,400 | +221,926,400 | LLaMA-style research |
| `llama_30b.json` | 30,677,882,880 | +677,882,880 | LLaMA-style research |
| `nemotron_h_4b.json` | 4,127,801,280 | +127,801,280 | Nemotron-H-like scaled research |
| `nemotron_h_8b.json` | 8,100,852,736 | +100,852,736 | Authoritative Base-8K architecture, random weights |
| `nemotron_h_16b.json` | 16,566,348,008 | +566,348,008 | Nemotron-H-like scaled research |
| `nemotron_h_30b.json` | 30,504,556,160 | +504,556,160 | Nemotron-H-like scaled research |

Counts enumerate the pinned TE GPT/hybrid parameter shapes: untied embedding and
output, final RMSNorm, attention and FFN projections/norms, plus Mamba projections,
conv weights/biases, A_log, D, dt_bias and gated norm. They do not count buffers as
parameters. Every worker also sums actual constructed parameters across ranks and
requires exact equality. Small instantiated models independently test both formulas.
The 16B LLaMA-style depth is 44 rather than the initially considered 48 blocks to
bring the actual count closer to 16B. Scaled hybrid sequences repeat the existing
52-block sequence and take the specified prefix; widths are target-count choices.
Only the 8B hybrid claims an official architecture; no official weights are loaded.

## Setup and Commands

Use the existing project container, not host Python. Follow the README C++ build
instructions for an OR-Tools-enabled `slackpipe_cli`; its shared libraries must be
visible in the same container via its normal loader path or `LD_LIBRARY_PATH`.
The old RunPod setup and dependency requirements still apply. Four-GPU real-size
acceptance requires four suitable GPUs; the two-GPU development host cannot
certify those experiments. Use `nccl-rma` only with the documented NCCL/symmetric
memory capabilities. Arbitrary delayed-match solver plans can still stall the
existing `nccl-p2p` transport; timeouts fail the receipt, never imply success.

Commands below run **inside** the container at the checkout root. From the host,
replace the leading `python` with
`docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python`.

```bash
python -m tools.run_slackpipe_eval full \
  --model-config configs/slackpipe_eval/llama_8b.json \
  --schedule interleaved --pp 4 --logical-stages 8 --microbatches 8 \
  --seq-length 1024 --precision bf16 --transport nccl-p2p \
  --output slackpipe_eval/llama/8b

python -m tools.run_slackpipe_eval full \
  --model-config configs/slackpipe_eval/nemotron_h_8b.json \
  --schedule slackpipe --pp 4 --logical-stages 8 --microbatches 8 \
  --solver slackpipe/build/release/slackpipe_cli --transport nccl-rma \
  --output slackpipe_eval/nemotron_h/8b

python -m tools.run_slackpipe_real_system_campaign \
  --families llama,nemotron_h --sizes 4b,8b,16b,30b \
  --schedules 1f1b,interleaved,slackpipe --pp 4 --logical-stages 8 \
  --microbatches 8 --seq-length 1024 --precision bf16 \
  --warmups 5 --iterations 50 --repetitions 3 \
  --solver slackpipe/build/release/slackpipe_cli --output slackpipe_eval --resume
```

For individual stages replace `full` with `env`, `correctness`, `smoke`,
`calibrate`, `solve`, `benchmark` or `trace`. Prerequisites run automatically.
Native `1f1b` requires `--logical-stages 4` at PP4 (or omit it for automatic N=PP).
Native `interleaved` requires N>PP, PP>1 and B divisible by PP. SlackPipe accepts
cyclic N divisible by PP; no source edits or manual layouts are required.
The generic calibrated multi-stage campaign is intended for PP2/PP4; existing
standalone PP1 SlackPipe runtime tests remain available. Native PP1 is also usable.

| Method | Main topology | Actual scheduler |
| --- | --- | --- |
| `1f1b` | PP4, N4, VPP=None | `forward_backward_pipelining_without_interleaving` |
| `interleaved` | PP4, N8, VPP2 | `forward_backward_pipelining_with_interleaving` |
| `slackpipe` | PP4, N8, VPP2 | `forward_backward_slackpipe`, exported solver orders |
| `optimized_interleaved` (optional) | PP4, N8, VPP2 | Native interleaving with solver partition |

1F1B versus SlackPipe is a **system-level comparison**, with different N/VPP.
Interleaved versus SlackPipe is the better-controlled schedule comparison, though
optimized partition effects remain. The optional fourth method separates the
partition contribution from scheduling contribution.

## Calibration and Acceptance

```text
env -> small numerical-equivalence gate -> target-model native smoke
    -> native calibration -> C++ solve -> target-model SlackPipe smoke
    -> independent unprofiled benchmarks -> separate trace capture
```

Native methods skip calibration/solve unless explicitly requested. The current
correctness gate runs the existing small hybrid equivalence fixture at the
requested PP/precision/transport, refuses skips, and labels this scope explicitly;
it is not evidence of full-size numerical equivalence. Target-model smoke checks
exact parameter count, contiguous global layer ownership, expected loss count,
finite parameters/loss/all gradients, a representable optimizer update on every
rank, and empty transport state after collective teardown. No expensive tracing
or update probes are enabled by the benchmark step.

Each size/family gets its own calibration. LLaMA-style uses the existing
cost_profile.v1 shared slope and per-stage biases, not hybrid regression. Hybrid
models use native class-identifiable partitions and cost_profile.v2 exact ranges.
Costs remain synchronized stage-call wall times including dispatch/allocation
stalls, excluding P2P outside the stage and SGD. Per-iteration observations,
variability and review flags remain available; this is not a sum-of-kernels model.
Inspect unstable fits and separate memory diagnosis before drawing conclusions.

Profile hash, model-config hash, manifest hash (hybrid), precision, sequence,
microbatch shape and topology must match. The existing homogeneous C++ exporter
emits compatible plan.v1 without a profile hash; a separate
`slackpipe.eval_plan_binding.v1` binds unchanged plan/profile file SHA256s.
Hybrid plan.v2 retains embedded profile/manifest hashes. Worker lists always come
from C++ validated schedule export, never reconstructed from a heuristic.

`slackpipe.eval_receipt.v1` receipts bind source and environment, configuration, collection policy,
prerequisite receipts and artifact hashes. `--resume` reuses exact matches;
stale receipts are rejected. `--force` creates new immutable attempts for the
requested stage and stale prerequisites; matching prerequisites can be reused.
`full --force` reruns the complete chain. Failed logs and older attempts remain.
Changing software, model, precision, shape, policy or seed invalidates acceptance.

## Measurement and Traces

Defaults: five full-step warmups, 50 measured steps, three independent fresh
process runs. Campaign order is seeded and rotated across repetitions. Seeds,
data and SGD learning rate are identical across methods; repetitions are
independent timing runs, not different random training problems.
Use `--warmups 20` consistently across methods for a stability diagnostic.
All samples, including slow ones, are retained. Benchmarking enables neither the
profiler nor synchronized stage calibration. The shared collector times batch
clones/preparation, zero_grad, forward/backward/P2P, and SGD; no data loader or
distributed optimizer is implied. CUDA samples and a synchronized continuous
wall window are both retained; file output/aggregation are outside that window.

Trace uses the same step after training warmup, with separate profiler
`wait=2,warmup=2,active=3,repeat=3`. Override using `--profiler-*` flags. Each
completed cycle has CPU+CUDA raw and compact output, actual global iteration IDs,
run/method/rank/cycle identity, and NVTX ranges. Stack/shape/memory profiling stays
off in this light trace. Existing standalone memory-diagnosis commands are in
[the collection guide](slackpipe_collection.md); allocator experiments remain
separate from primary comparisons.

The campaign generates a three-panel PNG/PDF once all three traces exist. The
first common complete-rank global iteration in cycle zero is selected
deterministically, with candidates retained. No synthetic average, timestamp
rescaling, or independent rank shifting is used. White gaps are outside labeled
envelopes, **not proven hardware idle**. For another recorded iteration:

```bash
python -m tools.plot_schedule_trace \
  --noninterleaved TRACE_1F1B --baseline TRACE_INTERLEAVED --slackpipe TRACE_SLACKPIPE \
  --iteration 9 --cycle 0 --output-png timeline.png --output-pdf timeline.pdf \
  --report figure_report.json
```

## Outputs and Schemas

```text
slackpipe_eval/FAMILY/SIZE/
  config.json
  receipts/METHOD.STAGE[.runNNN].json
  calibration/calibrate-ATTEMPT/worker/{cost_profile,model_manifest,...}.json
  solver/solve-ATTEMPT/{slackpipe.plan.json,solver.*,*.binding.json}
  1f1b/  interleaved/  slackpipe/  [optimized_interleaved/]
    benchmark-ATTEMPT/runNNN/result.rankR.json
    benchmark-ATTEMPT/summary.json
  traces/trace-ATTEMPT/run000/RUN/METHOD/*.{torch,compact}.json
  traces/figure-CAPTURE/{timeline.png,timeline.pdf,figure_report.json}
```

Raw samples retain `slackpipe.collection.v1` and traces `slackpipe.figure_trace.v2`.
`slackpipe.eval_benchmark.v1` adds configuration (PP/N/VPP/TP/DP/CP/B, micro/global
batch, sequence, precision, exact count, model hash, schedule, partition/ranges,
transport), all samples by run, rank aggregation, mean/median/stddev/CV, run means
and within-run variability, 95% CI across independent run means (normal
approximation; null for one run), wall-window means, throughput and peak allocated/
reserved memory across collection phase boundaries (including warmup). Raw
`memory_boundaries.measurement_end` separately records measurement-window peaks.
Collection metadata preserves profiler/allocator/software/GPU
identities and timing definitions.

The campaign exports `model_configs.csv/json`, `experiment_summary.csv/json`,
`benchmark_summary.csv`, `calibration_summary.csv`, `solver_summary.csv`,
`memory_summary.csv`, and `model_configs.tex`. Regenerate model tables without
training using `python -m tools.slackpipe_eval_tables --output slackpipe_eval`.
LaTeX groups homogeneous/hybrid configs and derives actual count, L, V, H, FFN and
attention types. Benefits use synchronized continuous step means:
`(T_1f1b-T_slackpipe)/T_1f1b`, `(T_1f1b-T_interleaved)/T_1f1b`,
`(T_interleaved-T_slackpipe)/T_interleaved`, plus optional partition/scheduling
contributions. Missing methods do not produce invented benefit values.
Report transport alongside these contributions: SlackPipe RMA versus native P2P
is not a pure scheduling-only attribution.

## Capacity and Validation Limits

Preflight reports exact parameter+gradient storage by physical worker and an
advisory activation estimate against free GPU memory. If parameter+gradient bytes
alone exceed 90% of available memory, it records `skipped_memory_capacity` and
does not allocate the model. Fewer GPUs record `skipped_hardware_capacity`.
No automatic sequence/model shrinking or repeated OOM probing occurs. The estimate
is not a fit guarantee: live activations, optimized cuts, allocator fragmentation,
workspace and RMA buffers can dominate. In particular 30B needs real capacity
planning; current SGD estimates must not be reused for Adam/master weights.

Tests: `tests/unit_tests/pipeline_parallel/test_slackpipe_eval.py`, the existing
`test_slackpipe_*.py` suites at PP1/PP2 with both transports, plus a tiny two-family
campaign exercising native 1F1B/interleaving, separate v1/v2 calibration, real
CP-SAT, solver-plan execution, multiple capture windows, three-panel figures and
receipt reuse. Run Megatron tests only in `slackpipe-dev` as documented in README.
Full 4B/8B/16B/30B allocation and PP4 runtime remain hardware acceptance tasks, not
claims inferred from a successful configuration parser or small-model run.

### Local Acceptance Record

On the two-GPU development container, the added 29 tests pass as part of the
full SlackPipe regression suite: PP1 **130 passed / 15 configuration skips**;
PP2 P2P and PP2 RMA each **135 passed / 10 configuration skips per rank**.
PP1/PP2-specific skips are covered by the alternate invocation; two PP4 precision
cases still require four GPUs. Both existing C++ CTest builds pass **4/4**.

A separate tiny FP32 campaign completed both families and all three schedules,
two independent benchmark runs per method, real v1/v2 calibration and CP-SAT,
solver-plan RMA execution, and two capture cycles per rank/method. It produced
42 successful receipts, 24 raw plus 24 compact trace files, and two three-panel
PNG/PDF pairs. Reconstructing all 12 summaries from raw rank samples matched
exactly. Repeating the campaign with `--resume` launched no training subprocess.
The eight production specifications also produced JSON/CSV/LaTeX configuration
tables without allocating the large models. Actual 30B/two-GPU and PP4 preflights
recorded capacity skips without attempting large allocations. These are harness
acceptance checks, not real-size performance results.
Additional generic BF16 smoke checks passed for tiny LLaMA-style interleaving
and tiny Nemotron-H-like native 1F1B, including finite gradients and SGD updates.
