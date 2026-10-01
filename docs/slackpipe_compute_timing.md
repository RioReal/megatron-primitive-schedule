# Compute Timing Diagnostics

This diagnostic does not change the solver estimator or any quality threshold.
The supplied 30-iteration Nemotron-H summary has no raw data mounted in this
environment. Its recurring differences have not been explained or fixed here.

## Actual Boundaries

`existing-stage-wall-time-v1` in the generic campaign names the synchronized
wall-time collector in `schedules._run_with_slackpipe_cost_calibration`:

1. Device-wide CUDA synchronization **before** starting the timer.
2. `forward_step` or `backward_step` on the current compute stream.
3. Device-wide CUDA synchronization **inside** the interval, then stop the timer.

It is not unsynchronized CPU launch timing. Forward includes last-stage loss
handling. Backward includes autograd/custom Mamba backward launches. Pipeline
receives/sends, input selection and explicit pre/post gradient-sync bookkeeping
are outside these closures. Autograd-internal dependencies, allocations, host
dispatch and any work caught by the completion boundary are not identifiable
separately from the original elapsed duration.

Legacy records now also expose `cpu_wall_ms`, `pre_compute_sync_ms`, and
`completion_sync_ms`; `existing_stage_wall_ms` aliases `elapsed_ms`, whose meaning
and use as the ordinary calibration cost remain unchanged. Two extra host clock
reads add small, unquantified overhead, not new synchronization boundaries.

Native P2P waits on Work handles and, when `batch_p2p_comm && batch_p2p_sync`,
performs its own device synchronization. Those existing behaviors are unchanged.
Calibration constructs no torch profiler. There is no profiler wait/active/repeat
or flush transition in this path. Each partition receives full-step warmup;
cache clearing occurs between partitions, outside timed compute closures.

## Opt-In Probe

`--diagnose-compute-timing` is available on `tools.slackpipe_eval_worker` and
`tools.slackpipe_nemotron_worker`, only for native interleaved calibration
(VPP >= 2). It shares model construction, partition generation, data, schedule,
backward and SGD with ordinary collection.

- Preallocate and initialize one CUDA event pair per local F/B operation before
  training warmup; reuse pairs across iterations.
- Record on the current compute stream around the compute closure. The probe
  adds **no per-operation device synchronization or event wait**.
- Drain the device once after the full step including SGD, then resolve events.
  No new layer/microbatch barriers are inserted.
- Observe inclusive native P2P host-call duration with a communicator subclass.
  Requests, return objects and native wait/sync policy are unchanged.
- Keep global iteration, microbatch, stage, worker, exact range and ordered
  composition. `compute_position` counts compute calls; diagnostic
  `schedule_position` also includes P2P calls, exposing adjacency.

An event envelope is **not proven intrinsic compute time**: it can include host
launch gaps and waits inserted inside the region. Auxiliary-stream work is
included only if it joins the compute stream before the end event. The final
device drain does not retroactively extend individual event intervals.
P2P host duration, GPU envelopes and drain time cannot be added as disjoint work.
Fine-grained NCCL wait, allocator and kernel attribution still requires a trace
or additional focused instrumentation.

Probe records contain `compute_gpu_ms`, `cpu_wall_ms`, stream ID and operation
positions. `existing_stage_wall_ms` and per-operation `wait_synchronization_ms`
are **null**, not zero: recreating the old device-wide boundary in the same
operation would perturb the probe. Compare a separate matched legacy run.
Inclusive P2P host duration is not NCCL GPU service time; its wait component is
not separately measured. Full-step `completion_wait_ms` is measured separately.

## Target Reproduction

Run both commands on the same fixed checkout/container and idle GPUs, with the
original precision, seed, optimizer, sequence and batch settings. Do not edit or
stage source or run competing GPU tests during the pair. This four-GPU example
requires adequately sized target GPUs, not this host's two 6 GiB A2000s. Adjust
the common arguments consistently to match the original experiment. Bash:

```bash
launcher=(docker exec -w /workspace/Megatron-LM slackpipe-dev env
  OMP_NUM_THREADS=1 MAMBA_DETERMINISTIC=1 TRITON_CACHE_AUTOTUNING=0
  NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
  /opt/venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4
  -m tools.slackpipe_eval_worker calibrate)
common=(--model-config configs/slackpipe_eval/nemotron_h_4b.json
  --schedule interleaved --pp 4 --logical-stages 8 --microbatches 8
  --micro-batch-size 1 --seq-length 1024 --precision bf16
  --seed 1234 --learning-rate 1e-6 --warmups 20 --iterations 30)

"${launcher[@]}" "${common[@]}" --run-id wall-reference \
  --output slackpipe_experiments/timing-pair/legacy
"${launcher[@]}" "${common[@]}" --run-id event-probe \
  --output slackpipe_experiments/timing-pair/events --diagnose-compute-timing

docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python \
  -m tools.slackpipe_compare_compute_timing \
  --legacy slackpipe_experiments/timing-pair/legacy \
  --diagnostic slackpipe_experiments/timing-pair/events \
  --output slackpipe_experiments/timing-pair/comparison.json
```

A legacy quality failure returns worker exit 2 (torchrun reports failure).
Keep the raw data and receipt; never bypass it to launch the solver. The separate
event diagnostic writes **no cost profile**, does no fitting/filtering, and
cannot authorize the campaign. Exit 0 only means diagnostic collection completed;
it does not trigger quality retries. Always use fresh output directories.

For a local mechanism check, use `tools.slackpipe_nemotron_worker calibrate
--tiny --precision fp32 --seq-length 32 --warmups 20 --iterations 30` under
two-rank torchrun with the same environment above. Run without/with
`--diagnose-compute-timing` using different fresh `--output` directories. This is
a tiny hybrid with Mamba, attention and MLP blocks, not Nemotron-H 4B.

## Output and Comparison

- `timing_events.rank*.json` per partition and merged `timing_events.json`: all
  raw diagnostic operations. They deliberately lack `elapsed_ms`, preventing
  the current cost aggregator from interpreting them as calibrated costs.
- `result.rank*.json`: configuration, source, hardware/software/allocator,
  profiler settings (disabled), warmup/measurement IDs and boundary memory
  samples. `timing_diagnostic` uses `slackpipe.compute_timing_diagnostic.v1` and
  includes iteration-level P2P calls and full-step completion waits.
- `slackpipe.compute_timing_comparison.v1`: input hashes, raw records, all
  stage/group/phase iteration means and CVs, cross-partition median shifts.
  CV = population stddev/mean; shift = max(group medians)/min(group medians)-1,
  or null for fewer than two groups. Nothing is filtered or rescaled.

Comparison requires identical manifests, complete ranks/operation grids,
configuration including hardware/software/source, partition sets and iteration
IDs. It compares identical stage/range/ordered-composition/role/worker/phase, not
different workloads. Old references without comparable metadata must be
recollected rather than silently matched.

Inspect backward ranges [13,19], [26,32], [39,45], [6,13], [34,39] and [19,28]
in `groups` and `comparisons` on the target run. Lower event CV alone is not a
root-cause diagnosis. The report deliberately sets `estimator_validated: false`;
any permanent estimator change requires subsequent empirical review.

## Validation on 2026-10-01

The two-rank tiny-hybrid pair above completed six partitions, 20 warmup and 30
measured iterations per partition, B=8, N=4, FP32, sequence length 32, SGD
learning rate 1e-6. Each run retained 11,520 F/B records. The event probe retained
7,020 P2P call records and emitted no cost profile. Collection was sequential,
with no source edits or other GPU tests during the pair. The comparison verified
identical per-worker configuration/source/hardware across both runs and all
partitions, plus complete operation grids and event timing-context hashes.

Raw data, logs and `comparison-context-verified.json` are retained locally under
the ignored `slackpipe_experiments/compute_timing_20261001/`. Both runs used the
pre-commit worktree based on `30032681d0b510dd5853ae41089e797af445047a`, with tracked
diff SHA-256 `fe00811d2b42f1da95b0f81ad14ba5863f19df0d358cb4dcb3dcf68e870c6bdd`.
Per-file untracked source hashes are retained in each result's environment
metadata. The comparison records hashes of all input evidence. These are new
**tiny-hybrid diagnostic measurements**, not Nemotron-H 4B results.

Backward examples, same exact range/stage/worker/composition across partitions:

| Range | Groups | Wall medians (ms) | Event medians (ms) | Wall shift | Event shift |
| --- | --- | --- | --- | --- | --- |
| [3,6] | 0,1,3 | 12.0651, 9.2245, 12.0390 | 11.8397, 9.1358, 11.9864 | 30.79% | 31.20% |
| [6,9] | 0,2,4 | 7.9368, 5.8270, 3.5200 | 7.8252, 5.8116, 3.5358 | 125.48% | 121.31% |
| [0,3] | 0,1,3,5 | 8.0467, 8.1127, 7.6749, 3.5116 | 7.8498, 8.1330, 7.6977, 3.5131 | 131.02% | 131.50% |

For [3,6], the wall CVs are 0.0082/0.0899/0.0161 and event CVs are
0.0422/0.0504/0.0054 in the same group order. Across all 48 stage/phase/group
estimates (no diagnostic filtering), five wall CVs and ten event CVs exceed 0.1.
This run **does not support switching to events as a stable intrinsic cost**.
The separate protocols alter synchronization, so these are not paired samples
from identical GPU execution, nor a controlled throughput comparison.

The legacy reference's existing quality gate returned `rerun_required`: zero
raw sample exclusions, two group exclusions, five surviving within-group CV
failures, plus consensus/cross-group/fit failures. `selected_profile` remains
null; its worker exit 2 was expected quality rejection after complete collection.
No solver or benchmark was launched from these measurements. No iteration filter
or threshold adjustment was made, and no diagnostic retry was mislabeled as a
successful cost calibration.

Runtime inspection followed `MambaMixer` into the installed mamba-ssm 2.3.2.post1
`MambaSplitConv1dScanCombinedFn.backward`: it recomputes convolution intermediates,
allocates scratch tensors and launches the scan/convolution backward kernels.
There is no explicit device synchronization or stream-switch call in that Python
function. This is existing kernel-internal work, not newly enabled Megatron
activation recomputation. The native communication calls in the probe all retain
batch synchronization. Host closure durations closely track the observed event
envelopes, but that does not separate launch gaps, allocator/cache behavior,
kernel selection, or device execution. The precise cause remains unresolved;
kernel/runtime traces are needed before changing an estimator or buffer lifetime.

Regression validation: 165 profiling/cost/receipt/collection/timing tests passed
before the pair, including one real-CUDA event test. Two distributed PP=2 FP32
numerical-equivalence tests (homogeneous and hybrid) passed on both ranks after
the pair. The generic eval-worker diagnostic CLI also completed a 30-iteration,
PP=2/N=4/B=4 tiny GPT smoke, retaining 960 operations and no cost profile under
`generic-events/` in the same ignored root. This checks the generic entry point,
not 4B behavior. No target 4B or PP=4 run was possible on the two 6 GiB GPUs.

Final combined regression: **184 passed**, including the added context-mismatch
check and real-CUDA gradient/SGD equivalence with and without the event probe.
Black, isort and `git diff --check` passed. Generated measurement files and logs
are not part of the source commit.
