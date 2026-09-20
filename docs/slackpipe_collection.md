# Reproducible SlackPipe experiment collection

These are experiment tools, not modifications to ordinary Megatron schedulers.
Run Megatron in `slackpipe-dev`, using `/opt/venv/bin/python`. Preserve old output
directories and use a new run ID/output root for each experiment. Never compare
profiler timings to an unprofiled benchmark as if they were equivalent samples.

## Collection and warmup policy

`collection.py` is shared by the homogeneous/heterogeneous benchmark harness,
`capture_schedule_trace.py`, and the native hybrid `slackpipe_nemotron_worker.py`.
The callable passed to it is the same complete training step in all three modes:

| Mode | Profiler | Shapes / stacks / memory | Allocation history |
| --- | --- | --- | --- |
| `benchmark` | Not constructed | Disabled | Disabled |
| `timeline` | CPU and CUDA | All disabled | Disabled |
| `memory` | CPU and CUDA | All enabled | Bounded CUDA allocator history + snapshot |

The measured step includes zeroing gradients, creating/consuming iterators over
resident mock batches, F/B, pipeline communication, schedule gradient processing,
and SGD. Model and resident batch creation, finite-check collectives, profiler
transitions, aggregation, and file output are outside its timing brackets.
This is not data-loader, checkpoint, or production distributed-optimizer throughput.

Use **20 full training warmups initially**, identical across compared methods.
Warmups use the exact partition, schedule, microbatches, dtype, transport, model,
optimizer and step callable used during measurement. Timing events are allocated
and their lazy handles initialized before training warmup. No model/optimizer
reconstruction or `empty_cache()` occurs between warmup and measurement. Peak
counter reset does not free cached memory. Profiler creation is an intentional
transition in diagnostic modes, followed by a distinct profiler warmup.

Every warmup and measured sample is retained. Inspect first/last-half medians,
CV, memory counters and independent-run variation before choosing a longer
warmup for **all** compared methods. Twenty is a starting policy, not proof of
steady state. There is no zero-allocation convergence requirement or slow-sample
filter. With SGD, different measurement lengths also change the optimization
trajectory, so keep seed, LR, warmup and measurement length equal for comparisons.

### Timing and communication completion

CUDA events bracket each step on the caller compute stream. This relies on the
supported synchronous completion contracts: ordinary non-overlapped P2P and
SlackPipe P2P join Work requests to that stream; RMA joins receives and drains
send events at schedule finalization. It is not a promise for arbitrary custom
streams, asynchronous optimizers, or communication-overlap configurations.
No per-layer/microbatch synchronization was added to simplify timing.

The continuous window starts after rank/device synchronization and ends after
device synchronization and a rank barrier. It includes host dispatch gaps and
lightweight collection bookkeeping. Aggregation and file output occur afterward
in benchmark mode. Profiling windows necessarily include callback/export pauses;
their continuous wall times are labeled diagnostic, never benchmark throughput.
Per-rank CPU dispatch times are enqueue durations, not device completion times.
Per-iteration summaries use the maximum aligned rank duration, while the
synchronized window remains a separate measurement.

## Capture cycles and identity

Default profiler schedule: `wait=2, warmup=2, active=3, repeat=3`, **after** training
warmup. Zero-based global training iterations are therefore 24-26, 31-33, 38-40
for the three captures. `prof.step()` executes once after every post-warmup
training iteration, outside its timing brackets. Each callback exports a fresh
raw file; `acc_events=False`. No final single-file export substitutes for cycles.

Serialization duration and serialization-plus-rank-alignment duration are
recorded. Ranks align after export, outside the step, then execute the next
cycle's wait and profiler warmup. This reduces phase mismatch but does not erase
instrumentation/cache/thermal perturbations; all intervening samples are kept.

## Commands

The following examples use an existing parser-valid PP2/N4/B4/L8 homogeneous
plan at `slackpipe_plans/tiny.plan.json`. Generate it through the documented
solver workflow if absent. Substitute the actual plan/model for a research run.
Use identical dimension, seed, optimizer and allocator settings on both methods.

Unprofiled benchmark (A=uniform ordinary, B=plan-partition ordinary, C=SlackPipe):

```bash
docker exec -w /workspace/Megatron-LM -e OMP_NUM_THREADS=1 \
  -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 slackpipe-dev \
  /opt/venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m tests.unit_tests.pipeline_parallel.slackpipe_perf_benchmark \
  --plan slackpipe_plans/tiny.plan.json --output-dir slackpipe_traces/new-benchmark-r1 \
  --run-id r1 --warmup-iterations 20 --iterations 30 --strict-fp32 \
  --hidden-size 64 --num-attention-heads 4 --seq-length 16 \
  --micro-batch-size 1 --vocab-size 128 --learning-rate 0.000001 \
  --slackpipe-transport nccl-rma --disable-slackpipe-nvtx
```

Lightweight timeline (repeat separately with `--slackpipe-profile-kind baseline`
and a distinct run ID; supply `--heterogeneous-config` for heterogeneous models):

```bash
docker exec -w /workspace/Megatron-LM -e OMP_NUM_THREADS=1 \
  -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 slackpipe-dev \
  /opt/venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m tools.capture_schedule_trace --collection-mode timeline \
  --slackpipe-profile-kind slackpipe --run-id timeline-r1 \
  --slackpipe-profiler-output slackpipe_traces/new-captures \
  --plan slackpipe_plans/tiny.plan.json --warmup-iterations 20 \
  --profiler-wait 2 --profiler-warmup 2 --profiler-active 3 --profiler-repeat 3 \
  --hidden-size 64 --num-attention-heads 4 --seq-length 16 \
  --micro-batch-size 1 --vocab-size 128 --learning-rate 0.000001 \
  --slackpipe-transport nccl-rma
```

For a separate short allocation run, use the same timeline command with
`--collection-mode memory --run-id memory-r1 --profiler-wait 1
--profiler-warmup 1 --profiler-active 1 --profiler-repeat 1
--memory-history-entries 100000`. Keep the 20 training warmups. Increasing history
size increases host overhead. The private PyTorch allocator-history API must
exist in the pinned container; failure is explicit, not a silent downgrade.

```bash
docker exec -w /workspace/Megatron-LM -e OMP_NUM_THREADS=1 \
  -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 slackpipe-dev \
  /opt/venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m tools.capture_schedule_trace --collection-mode memory \
  --slackpipe-profile-kind slackpipe --run-id memory-r1 \
  --slackpipe-profiler-output slackpipe_traces/new-captures \
  --plan slackpipe_plans/tiny.plan.json --warmup-iterations 20 \
  --profiler-wait 1 --profiler-warmup 1 --profiler-active 1 --profiler-repeat 1 \
  --memory-history-entries 100000 --hidden-size 64 --num-attention-heads 4 \
  --seq-length 16 --micro-batch-size 1 --vocab-size 128 --learning-rate 0.000001 \
  --slackpipe-transport nccl-rma
```

Repeat benchmarks in independent torchrun processes with fresh output roots and
run IDs. `slackpipe_experiment_driver.py --repetitions 3` and the native
Nemotron workflow's `--runs 3` retain run means, between-run standard deviation
and within-run variation separately. Pooled iteration variance is not an estimate
of independent-run uncertainty. Do not reuse completed outputs under `--force`.
For performance comparisons, prefer fresh processes for each method too: pass
`--only-mode A_uniform_interleaved_1f1b`, `B_solver_split_interleaved_1f1b` or
`C_slackpipe_solver_order` to the benchmark harness. Its all-three invocation is
convenient for correctness smoke checks but retains process-level kernel caches
between methods. The experiment driver already launches one mode per process.

## Output contracts

Under `OUTPUT/RUN/METHOD/`, filenames include `RUN.METHOD.rankR`:

| Suffix | Contents |
| --- | --- |
| `.metadata.json` | `slackpipe.collection.v1`; actual mode/options, profiler config, allocator/backend/env, software/source/GPU identity, model/plan hashes, warmup/measurement IDs, timing definitions |
| `.samples.json` | Same metadata plus every warmup/measured sample, memory boundaries, summaries, continuous wall time, captures and export durations |
| `.cycleCCC.torch.json` | Unfiltered Kineto Chrome trace: CUDA APIs, kernels, memory transfers, communication, stream/correlation IDs, raw clock metadata |
| `.cycleCCC.compact.json` | `slackpipe.figure_trace.v2`; correlated F/B/optimizer/step records, flat activities/allocation calls, capture identity, metrics and raw clock base |
| `.memory.pickle` | Diagnostic allocator snapshot, including bounded history and allocation stacks where available |
| `.complete.json` | Written only after all requested captures pass coverage and operation-order validation |

Each sample has `iteration`, `cuda_elapsed_ms`, `cpu_dispatch_ms`, and `memory`
(`allocated_bytes`, `reserved_bytes`, `peak_allocated_bytes`, `peak_reserved_bytes`).
Each capture has `cycle`, `capture_id`, exact `active_iterations`, `raw_path`,
`export_ms`, `export_and_rank_alignment_ms`, and memory counters. Compact files
carry the same run/capture IDs across ranks. Exclusive metadata creation rejects
an existing run/method/rank; failed processing leaves raw evidence intact.

Memory history starts after model construction/event setup, before training
warmup. It cannot reconstruct call stacks for earlier allocations and its bounded
ring can overwrite old history. Snapshots describe the PyTorch allocator, not all
NCCL/driver/external allocations. Kineto warnings about preexisting allocations
are meaningful limitations. Allocator environment and backend are always recorded.
`cudaMallocAsync` is **not** enabled by this change: test it only as an independent
matched experiment on both methods, with fresh processes and correctness checks.

## Derived metrics and plotting

For each operation/step, envelope is last GPU activity end minus first start;
union is the merged wall-clock intervals of its correlated GPU activities,
including overlaps across streams only once; internal gap is envelope minus
union. It is **not hardware idle time** or summed kernel service time.
`cuda_malloc_count` counts recorded CPU API calls (`cudaMalloc*`, `cuMemAlloc*`);
CPU union and sum are separate because runtime/driver calls can nest. Counts do
not necessarily equal physical allocations. Communication union currently uses
NCCL-named GPU activities: a useful subset, **not total RMA traffic**. Raw driver,
CUDA, copy and stream information is retained for deeper attribution.

CUDA launch correlation is preferred, with CPU external-ID fallback. Synchronous
backward includes nested autograd engine-thread nodes; unrelated threads are not
assigned to backward merely because their timestamps overlap. GPU envelopes may
overlap legitimately. Validation checks CPU dispatch order and plan coverage,
not artificial GPU serialization. Event/envelope discrepancies are retained as
warnings, never used to discard slow iterations.

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python \
  -m tools.plot_schedule_trace \
  --baseline slackpipe_traces/new-captures/baseline-r1/baseline \
  --slackpipe slackpipe_traces/new-captures/timeline-r1/slackpipe \
  --iteration 24 --detail --title 'PP2 diagnostic capture' \
  --output-png slackpipe_traces/new-figure.png \
  --output-pdf slackpipe_traces/new-figure.pdf --report slackpipe_traces/new-figure.json
```

Use any recorded global iteration, optionally `--cycle`, `--baseline-run-id` and
`--slackpipe-run-id`. All ranks must match run, capture, iteration, configuration
and same-host clock domain. Ambiguous/missing ranks fail. Without `--iteration`,
the reproducible rule is earliest common complete-rank iteration; all candidate
IDs remain in the report. Detailed plots show context envelopes and separate GPU,
NCCL and CPU allocation lanes. No averaged timestamps or benchmark-based rescaling.

`tools/reprocess_schedule_trace.py --input OLD_DIR --output NEW_DIR` reprocesses
legacy/raw captures without modifying originals, records input SHA256 and refuses
an existing or nested output directory. Missing historic profiler settings remain
unknown; they are not retroactively inferred from today's defaults.

## Calibration provenance

The existing stage-call hooks synchronize CUDA and measure wall time. They are
not pure CUDA-event kernel timings: they include dispatch, allocations and compute
completion within the call, excluding explicit P2P outside it and the optimizer.
The exporter now records this definition, every per-iteration stage total, range,
CV and a review flag above CV 0.1. Allocation attribution is explicitly unknown
unless separately investigated. Neither low CV nor zero allocation calls proves
the cost is stable. Slow observations are retained; medians are a visible fitting
choice, not deletion of evidence. Heterogeneous class fits and exact layer-prefix
costs remain unchanged. No solver/search logic or cost schema version was changed.

Before accepting a profile, inspect variability and a matched memory/timeline run;
record whether envelope stalls are legitimate execution overhead for the modeled
system or a collection artifact. Do not silently reinterpret stalled envelopes as
stable compute, or substitute summed kernel durations, which omit dispatch and
overlap semantics. Validate the chosen costs by solver-to-runtime correctness and
independent unprofiled measurements. New metadata is descriptive, not an automatic
causal diagnosis or a guarantee of predictive accuracy.

## Validation scope

Focused regression tests cover interval unions, nested allocation APIs, engine
thread correlation, per-cycle export, complete rank/iteration identity, preserved
outliers, no profiler construction in benchmarks and independent-run statistics.
The local two-GPU checks exercise benchmark, multi-cycle timeline and short memory
collection, plus numerical equivalence. Supplied historical raw traces are
reprocessed into new directories with input-hash verification. PP4/full Nemotron
8B, multi-node clocks and alternative allocator backends need separate hardware
and validation. Unplotted gaps, RMA attribution and allocator-external memory are
still diagnostic questions, not explained stalls.

### Local verification record

Validation used the existing two-GPU development container. These are correctness
and collection smoke checks, not new controlled performance results. Logs and
derived artifacts are in the ignored `slackpipe_traces/collection-audit-20260920/`.

| Check | Result |
| --- | --- |
| Single-rank targeted plan/cost/manifest/collector/figure/model/lifecycle/workflow suite | 100 passed; 4 two-rank cases run separately, 2 four-GPU cases unavailable |
| Two-rank P2P numerical/edge-isolation suite | 11 passed per rank; PP1 cases covered separately |
| Two-rank RMA delayed-match/solver-plan suite | 9 passed per rank |
| C++ CTest, no-OR and OR-enabled | 4/4 passed in each build; OR CLI smoke passed |
| Homogeneous calibration | Live v1 profile, both ranks, warmup 0-19 and measurements 20-29 retained |
| Native tiny hybrid calibration -> CP-SAT -> parser -> PP2 BF16/RMA | Six partitions; FEASIBLE validated plan; initial/loss/gradient/post-step maximum differences all 0 |
| Three-method benchmark, two fresh process runs | All per-rank samples retained; aggregate means match raw samples; run/iteration variability kept separate |
| Baseline and SlackPipe timeline | Three cycles per rank, active global IDs 24-26, 31-33, 38-40; all raw and compact files present |
| Short memory diagnosis | Both ranks completed, allocator snapshots include stacks; 100,000-entry history capacity reached, so earlier history can be truncated |
| Native hybrid timeline | Both ranks completed; actual F/B sequence validates against exported plan |
| Four supplied historical raw traces | Re-correlated offline, original SHA256 unchanged; same-rank/capture/iteration plots validated |
| Formatting/static checks | isort, black, Ruff F checks and `git diff --check` passed |

The supplied selected historical iteration contains no recorded `cudaMalloc*` or
`cuMemAlloc*` calls. It cannot establish allocator stalls as the cause of gaps.
CPU dispatch, communication waits, missing RMA attribution, and profiler
perturbation remain possible contributors requiring more evidence. The new small
benchmark retained constant reserved memory across measurement and near-equal
first/last-half medians; that is not proof that a large-model run is warmed up.
No alternate allocator backend, full 8B model, PP4 or multi-node experiment was
performed. Failed development attempts remain in distinct output directories;
only runs with completion receipts are treated as completed captures.
