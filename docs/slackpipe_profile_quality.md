# Calibration Quality Gate

Calibration is a separate synchronized diagnostic workload, not benchmark timing.
Its stage-call wall time includes CPU dispatch, allocation/cache stalls, compute,
and the completion wait; explicit P2P calls and optimizer time are excluded from
the fitted stage costs. Training warmup still executes the complete training step.
Only rare catastrophic raw timing spikes can be excluded from aggregation as
described below. This does not substitute kernel sums, change the fitter, or
establish that allocation stalls are absent. Raw event files are never edited.

## Isolated Timing Spikes

Before averaging microbatches, analyze each calibration group/context, logical
stage and phase independently. With median `m`, `MAD = median(abs(x - m))` and
`sigma = 1.4826 * MAD`, a raw sample is a candidate only if **both**
`x > m + K * sigma` and `x > R * m`. These are strict upper-tail tests; `MAD=0`
requires no division and still requires the ratio test. There is no percentile
clipping, fixed-fraction trimming, or removal of moderately slow samples.

At least 20 raw samples are required by default; below that count nothing is
filtered, and the existing quality checks apply to the unfiltered data.
Accepted microbatches are averaged within each iteration, then the median across
iterations gives the per-operation stage cost, preserving the existing estimator.
For compatibility, diagnostics `samples_ms` store these means multiplied by B;
`raw_samples_ms` store original iteration totals. A partial accepted sum is never
divided by the original B. An iteration with no accepted samples fails quality.

Filtering is not permission to accept an unstable run. For **each stage/phase**,
exceeding any frequency, affected-iteration or consecutive-sample limit below
requires a retry even if cleaned CV is small. Consecutive means adjacent samples
in `(global iteration, microbatch)` order, including iteration boundaries.
Cleaned CV, cross-group consistency and fit error must also pass unchanged.
Isolated spikes alone do not cause a retry or `ProfilingQualityError`.

## Empirical Checks

All limits are configurable heuristics, not theoretical guarantees:

| Flag | Default | Definition |
| --- | --- | --- |
| `--quality-cv` | 0.10 | Population standard deviation / mean of accepted iteration per-operation means (equivalently B-normalized totals). This retains the existing 10% CV convention. |
| `--quality-median-shift` | 0.30 | Maximum / minimum comparable group median minus one. Each group cost is the median iteration total divided by its fixed microbatch count. |
| `--quality-relative-rmse` | 0.15 | `sqrt(mean((prediction - observation)^2)) / mean(observation)`, independently for forward and backward, over all stage/group rows. |
| `--quality-min-samples` | 3 | Minimum complete measured iterations per stage/group/phase; cannot be reduced below three. |
| `--quality-outlier-mad-multiplier` | 10 | K in the robust upper-tail test. |
| `--quality-outlier-median-multiplier` | 3 | R in the simultaneous median-ratio test. |
| `--quality-min-outlier-samples` | 20 | Minimum raw sample count for detection in one stage/group/phase. |
| `--quality-max-outlier-fraction` | 0.02 | Reject when discarded / raw sample count exceeds this fraction. |
| `--quality-max-outlier-iteration-fraction` | 0.20 | Reject when iterations with any rejected sample / all measured iterations exceeds this fraction. |
| `--quality-max-consecutive-outliers` | 1 | Reject when the longest consecutive rejected-sample run exceeds this count. |

Cross-group comparisons require at least two groups with identical contiguous
layer ranges, stage roles, physical workers, and measurement-context hashes.
The context binds model, microbatches, precision, input shape, warmup/measurement
settings, seed, optimizer learning rate, software, GPU identity, allocator, and
timing definition. Different partitions are compared only where those keys match.
Missing provenance or inadequate samples fail closed. If there are no comparable
groups (for example, the single-partition v1 collector), diagnostics explicitly
record `not_assessed_no_comparable_groups`; no cross-group consistency is claimed.
Fit checks require at least two observations per phase. For v2, predictions use
the actual heterogeneous prefix costs and role biases; for v1, the shared slope
and stage biases. A v1 exact fit is not independent evidence of generalization.

## Retry and Artifacts

`tools.run_slackpipe_eval calibrate` and its campaign callers launch the entire
partition set in fresh torchrun processes. On a quality failure, they repeat the
entire calibration **once**, with the same configuration and thresholds. A second
failure stops the pipeline before solver invocation. Retry configuration and
hardware identity must match the first attempt. There is no fastest-attempt
selection, raw-sample deletion, threshold relaxation, or unbounded retry.

Each calibration stage stores:

```text
profiling_attempts.json             # configuration, timestamps, attempts, selection
attempt0/worker/
  calibration_events.json           # all raw events, with explicit group/attempt IDs
  observations.json                # stage/group rows and all iteration samples
  cost_profile.candidate.json       # retained even when rejected
  cost_profile.quality.json         # metrics, issues, thresholds, rerun command
  cost_profile.json                 # created only when this attempt passes
attempt1/worker/                    # created only after a quality failure
```

The quality receipt is `slackpipe.profile_quality.v2`. Its status is `passed` or
`rerun_required`; issues carry rule, threshold, value, phase, group, stage, layer
range, worker, observed costs, context hash and original data path. Profiles embed
the receipt and are rehashed. The attempt ledger identifies the only selected
profile. Original attempt files are never replaced. Group IDs are attached to
events before gathering/aggregation, not inferred from event order.
The embedded receipt uses `quality_schema_version` instead of `schema_version`
to remain compatible with the existing C++ cost reader's first-key lookup.

New fields include global `raw_sample_count`, `accepted_sample_count`,
`samples_discarded`, `discarded_fraction`, and per-stage/phase `sample_groups`
with the same counts plus `raw_cv`, `cleaned_cv`, affected iterations and longest
outlier run. CV here is across iteration means, **not** across raw microbatches.
`rejected_samples` retain group/attempt/context, stage, layer range/role, worker,
rank, global iteration, microbatch, original `elapsed_ms`, median, MAD,
robust sigma, both thresholds, rejection reason and `raw_data_path`.
The source event JSON retains every sample, including rejected ones.

Each observation records `outlier_filter.policy` and accepted counts per global
iteration. Quality policy changes require reaggregation, not simply reapproving
previously filtered observations. All policy flags and the quality schema enter
the generic calibration context/hash; downstream receipts are invalidated when
they change. The existing receipt/context-v2 schema is unchanged. Legacy quality
v1 profiles can still be explicitly validated under their stored policy but are
not silently reused in a new v2 calibration context. Reports from old aggregated
rows mark `raw_sample_status=unavailable_or_partial_legacy_aggregation`; zero
reported discards there does not certify that original raw samples were checked.

Synthetic example: 239 samples of 10 ms and one of 356 ms, B=8, 30 iterations,
give 240 raw / 239 accepted / 1 discarded (0.4167%) for that stage/phase.
Raw iteration CV is about 0.679 and cleaned CV is zero; stage cost remains
10 ms. The profile passes without retry and can continue to solver/benchmark.
This is a regression fixture, not an 8B GPU result.

Example warning (synthetic illustration, not a new GPU result):

```text
Profiling measurements are inconsistent. Rerun profiling before using this cost model.
rule=cross_group_median_shift phase=backward threshold=0.30 value=1.21774
comparable group medians: 24.8 ms, 55.0 ms
```

## Usage

Run the generic calibration inside the existing container; thresholds also enter
the immutable calibration receipt context:

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python \
  -m tools.run_slackpipe_eval calibrate \
  --model-config configs/slackpipe_eval/llama_8b.json --pp 2 --logical-stages 4 \
  --output slackpipe_experiments/quality-run \
  --calibration-warmups 20 --calibration-iterations 10 \
  --quality-cv 0.10 --quality-median-shift 0.30 --quality-relative-rmse 0.15
```

Apply the same two-attempt wrapper to a standalone collector (small heterogeneous
fixture shown; full Nemotron requires the hardware specified by its own runner):

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python \
  -m tools.slackpipe_profile_quality --collect-output slackpipe_experiments/quality-fixture -- \
  /opt/venv/bin/python -m torch.distributed.run --standalone --nproc-per-node 2 \
  tools/slackpipe_heterogeneous_experiment.py \
  --output-dir slackpipe_experiments/quality-fixture \
  --warmup-iterations 20 --iterations 10
```

The legacy Nemotron driver and homogeneous experiment driver also use this
wrapper and gate their solver calls. Intentional M1/M2 ablations in
the homogeneous driver validate the source calibration and its digest; they are
not relabeled as accurate measured fits.

Standalone workers do not recursively spawn torchrun. They retain rejected
candidates, emit `rerun_required` diagnostics and a fresh-output collection
command, withhold the default profile, and exit nonzero after cleanup. Offline `tools.slackpipe_hybrid fit`
requires `--profiling-rerun-command '<original collection command with fresh output>'`:
refitting the same observations is not a profiling retry. Old profiles without
explicit event provenance/quality receipts are not automatically authorized.
The gate does not rewrite historical measurements or constrain manual direct
C++ solver usage outside these Python/shell experiment entry points.

The checks diagnose inconsistency, not its physical cause. GPU contention,
allocation/cache effects, thermal/clock changes and other stalls still require
separate allocation/timeline diagnostics.

The generic collector's `run_partition(..., collect="calibrate")` does not
construct a torch profiler: profiler windows and trace serialization run only
in the separate timeline/memory path. In `_run_with_slackpipe_cost_calibration`,
the pre-call CUDA synchronization precedes the timer; the completion
synchronization is inside it; event bookkeeping follows the stop timestamp.
Thus no trace flush or profiler transition is evident in this calibration path.
Without the failing run's raw traces, the reported 280-360 ms spikes cannot be
attributed to those mechanisms or ruled out as GPU/system stalls. Timing
boundaries and profiler functionality are unchanged.

Regression checks (existing container, no 8B model allocation):

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python -m pytest \
  tests/unit_tests/pipeline_parallel/test_slackpipe_outliers.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_profile_quality.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_eval_receipts.py -q
```
