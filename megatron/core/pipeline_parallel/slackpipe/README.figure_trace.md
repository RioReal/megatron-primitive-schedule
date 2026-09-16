# Paper-style schedule traces

This is an opt-in **diagnostic capture**, not a benchmark. No normal training
entry point imports or enables it. Scheduler/solver implementations and tensor
execution are unchanged. The standalone driver reuses the existing real-model
benchmark harness and temporarily wraps `forward_step`, `backward_step`, and
the optimizer's `step` only inside the selected profiler iteration(s). Wrappers
restore the original functions on both success and exceptions. They do not
change tensor values, gradients, communication, or operation order.

## Capture

The demo uses the validated heterogeneous LARGE configuration: PP=2, N=4,
12 layers, pattern AABCABCABCAA, hidden=512, heads=16, sequence=256, vocabulary=128,
microbatch size=1, four microbatches, FP32/TF32 off, dropout=0, untied embeddings,
seed=1234, SGD lr=1e-9. TP/DP/CP=1. The existing two-GPU machine cannot capture PP4;
this task adds no PP4 scheduling support. The plotting format supports rank lanes
generically, but clock alignment is explicitly single-host only.

Baseline uses ordinary interleaved 1F1B with uniform split [3,3,3,3]. SlackPipe
uses the unchanged LARGE solver plan, split [1,3,5,3], with nccl-rma. All other
model/data settings are identical. Five normal, unprofiled optimizer steps warm
the runtime, then zero-based iteration 5 is captured. Initial RMA setup is outside
the capture. Baseline/SlackPipe are separate fresh distributed runs.

The example plan/model and recorded outputs below are generated artifacts, not
included in Git. To prepare new inputs, follow the top-level README calibration
and solver workflow with hidden size 512, 16 heads, sequence length 256, and
learning rate 1e-9, then pass the resulting plan/model paths to this driver.
The optimized split may differ on another run or machine. Build the optimizer
in `slackpipe/` with its external OR-Tools dependencies as described in README.

From the repository root, run each capture separately (substitute `slackpipe`
for both occurrences of `baseline` in the second run):

```bash
docker exec -e PYTHONPATH=/workspace/Megatron-LM \
  -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 -e OMP_NUM_THREADS=1 slackpipe-dev \
  /opt/venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tools/capture_schedule_trace.py \
  --slackpipe-profiler-output slackpipe_traces/figure_demo/baseline \
  --slackpipe-profile-kind baseline \
  --slackpipe-profile-start-step 5 --slackpipe-profile-num-steps 1 \
  --slackpipe-profile-format both \
  --plan slackpipe_experiments/heterogeneous_scale_sweep/LARGE/joint.plan.json \
  --heterogeneous-config slackpipe_experiments/heterogeneous_scale_sweep/LARGE/model.json
```

The driver exposes dimension/seed/LR/transport options; defaults above match the
known stable LARGE case. `--slackpipe-profile-format torch` keeps raw traces and
validation/capture metadata without writing compact timing files. `compact`
and `both` write compact timing too; raw traces are always retained for audit.
CPU and CUDA profiler activities are enabled only during the selected window.
No calibration mode, stack capture, memory profiling, or benchmark sweep runs.

## Timing definition

- Raw CPU ranges are named `ScheduleTrace/F/i5/s0/b0`, similarly for B, optimizer,
  and the complete step. Backward identity comes directly from its retained
  forward output object, not a guessed FIFO counter or textual solver order.
- The compact exporter associates CUDA kernel/memcpy/memset events with their
  originating CPU operations using Kineto `External id`, then with the enclosing
  labeled CPU range. A GPU envelope starts at its **first correlated CUDA
  activity** and ends at its **last correlated CUDA activity**. CPU launch range
  bounds are retained separately. Mirrored `gpu_user_annotation` records are not
  counted as new logical operations.
- Thus the plot is neither CPU launch latency nor the union/sum of raw kernels.
  Each colored envelope may include internal dispatch gaps. White gaps mean no
  plotted envelope is active; unplotted communication/control/finite-check work
  can still execute there. They are not a measurement of hardware idle percentage.
- Step envelopes use all correlated CUDA activity in the complete training step,
  including finite checks and SGD. Arrows use `max(step end over ranks) -
  min(step start over ranks)`. The raw time base is retained as integer
  `baseTimeNanoseconds`, with relative `start_us/end_us`. Different rank trace
  bases are converted before subtracting the common panel origin. Ranks are
  **not** independently shifted to their first forward operation.
- CUDA events independently bracket the full step. Their elapsed times include
  small boundary dispatch gaps outside the first/last GPU activity. The exporter
  reports this discrepancy and rejects differences above max(2 ms, 5%). The
  panel report also gives an event-duration estimate anchored at each rank's
  first activity; it is an estimate, not a separately synchronized global event
  clock. The plotted duration is exactly the GPU-envelope span by definition.
- Profiling adds overhead and changes timing. A positive red annotation is the
  difference between these **profiled steps**, not a statistically established
  speedup. Do not substitute it for the controlled benchmark results.

## Compact format and validation

`rankR_trace.json` uses `slackpipe.figure_trace.v1`: `config`, rank/mode/transport,
clock/base, `records`, and `steps`. Each envelope has iteration, rank, mode,
transport, kind, stage, microbatch, start/end microseconds, CPU range bounds,
GPU activity count, and the originating profiler label. Step/optimizer use -1
for stage/microbatch. Raw traces live in `torch_profiler/rankR_trace.json`.
`rankR_capture.json` retains settings and independent measurements;
`rankR_validation.json` records successful checks.

Validation rejects missing/duplicate operations, invalid or overlapping compute
envelopes, missing CUDA activity/clock metadata, timing-bracket mismatches, and
SlackPipe order differing from the production plan parser's operations.

## Plot

```bash
docker exec -e PYTHONPATH=/workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python \
  tools/plot_schedule_trace.py \
  --baseline slackpipe_traces/figure_demo/baseline \
  --slackpipe slackpipe_traces/figure_demo/slackpipe \
  --output-png slackpipe_traces/figure_demo/figures/schedule_trace.png \
  --output-pdf slackpipe_traces/figure_demo/figures/schedule_trace.pdf \
  --report slackpipe_traces/figure_demo/report.json \
  --iteration 5 --color-mode microbatch --annotate-time-saved
```

`--title` and `--caption` customize figure text; `--color-mode direction` disables
microbatch shading. Forward is blue, backward orange, optimizer gray. Both panels
share millisecond units and limits. Negative savings are reported numerically
in JSON but do not get a misleading positive savings arrow.

Focused tests: inside `slackpipe-dev`, run `python -m pytest
tests/unit_tests/pipeline_parallel/test_slackpipe_figure_trace.py -q`.

## Recorded demo

Artifacts are under `slackpipe_traces/figure_demo/`. The generated figure is
`figures/schedule_trace.png` (300 dpi) and `figures/schedule_trace.pdf` (vector).
`report.json` contains the configuration, all plotted records, and measurements.

| Profiled iteration 5 | Baseline | SlackPipe |
| --- | ---: | ---: |
| GPU step envelope (ms) | 181.819208 | 168.446197 |
| Event-based panel estimate discrepancy (ms) | 0.262305 | 1.811160 |
| F/B operations across both ranks | 32 | 32 |

The positive annotation is **13.373011 ms**. This is a profiled diagnostic,
not benchmark evidence. Both ranks pass count, coverage, nonnegative duration,
nonoverlap, and timing-bracket validation; SlackPipe additionally matches the
plan order exactly. `tests.log` records 40 passing tests: 11 trace tests and
29 plan tests. The five frozen runtime files match the existing checkpoint
hashes; no scheduler, transport, or solver file was changed for this task.
