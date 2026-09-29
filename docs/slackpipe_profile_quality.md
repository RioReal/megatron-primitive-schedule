# Calibration Quality Gate

Calibration is a separate synchronized diagnostic workload, not benchmark timing.
Its stage-call wall time includes CPU dispatch, allocation/cache stalls, compute,
and the completion wait; explicit P2P calls and optimizer time are excluded from
the fitted stage costs. Training warmup still executes the complete training step.
This change does not substitute kernel sums, filter samples, change the fitter,
or establish that allocation stalls are absent.

## Empirical Checks

All limits are configurable heuristics, not theoretical guarantees:

| Flag | Default | Definition |
| --- | --- | --- |
| `--quality-cv` | 0.10 | Population standard deviation / mean of measured iteration totals for each stage and phase. This retains the existing 10% CV review convention. |
| `--quality-median-shift` | 0.30 | Maximum / minimum comparable group median minus one. Each group cost is the median iteration total divided by its fixed microbatch count. |
| `--quality-relative-rmse` | 0.15 | `sqrt(mean((prediction - observation)^2)) / mean(observation)`, independently for forward and backward, over all stage/group rows. |
| `--quality-min-samples` | 3 | Minimum complete measured iterations per stage/group/phase; cannot be reduced below three. |

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
selection, sample deletion, threshold relaxation, or unbounded retry.

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

The quality receipt is `slackpipe.profile_quality.v1`. Its status is `passed` or
`rerun_required`; issues carry rule, threshold, value, phase, group, stage, layer
range, worker, observed costs, context hash and original data path. Profiles embed
the receipt and are rehashed. The attempt ledger identifies the only selected
profile. Original attempt files are never replaced. Group IDs are attached to
events before gathering/aggregation, not inferred from event order.
The embedded receipt uses `quality_schema_version` instead of `schema_version`
to remain compatible with the existing C++ cost reader's first-key lookup.

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
