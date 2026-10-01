# Isolated Compute Calibration

The generic evaluation campaign defaults to `isolated-layer-compute-v1`.
`existing-stage-wall-time-v1` remains an explicit diagnostic/ablation option.
Neither estimator measures pure hardware FLOP time, and isolated costs do not
claim to predict every pipeline-context latency or end-to-end iteration time.
See the [validation report](slackpipe_isolated_validation.md) for executed checks,
actual collected costs and hardware limitations.

## Dataflow and Units

Previously, candidate partitions ran under ordinary interleaved Megatron;
stage wall times were filtered, checked across comparable groups, and fitted.
The new path constructs the model manifest, groups equivalent partitionable
units, measures real isolated forward/backward operations, expands costs in
manifest order, and writes the existing `slackpipe.cost_profile.v2` prefixes.
The C++ optimizer, evaluator, plan format, communication model and runtime are
unchanged. Communication remains separate (zero in the default solver model).

For the strict evaluation model configurations, a LLaMA-style unit is one real
Megatron Transformer Engine decoder layer. Nemotron units are the actual Mamba,
attention-only and MLP-only layers selected by `hybrid_stack_spec`, in the exact
manifest order. The adapter uses the same `transformer_config` as training.
It does not load pretrained weights: both calibration and the existing synthetic
training campaign use seeded random parameters and representative random inputs.

Only one representative unit is resident at a time. A SHA-256 class signature
includes the complete supported model configuration, manifest unit metadata,
sequence length, micro batch size, precision, TP/DP/CP, recomputation setting and
implementation version. Layer IDs are excluded; differing per-unit configuration
is not merged. Each class records its representative and all member layer IDs.
Unknown model configurations are rejected by the existing strict adapter.
This is not a general adapter for arbitrary heterogeneous block implementations.

Embedding (including configured position embedding) is a separate `first` unit.
Final RMSNorm, untied output projection and native vocabulary loss plus scalar
mean are a separate `last` unit. Their F/B costs become first/last role biases,
never costs on every layer. For one logical stage, both biases are charged once
through the solver's first-role precedence. RoPE table preparation is outside
the measurement; its application within attention is inside.

## Measurement and Quality

Each class uses real autograd with a fixed representative upstream gradient,
independent F and B CUDA-event measurements, and a device synchronization before
and after each measured region. Host wall times are retained too. Gradient reset,
input/upstream construction and finite-output/gradient checks are outside the
events. Every warmup runs the same F/B pair. No optimizer is timed or constructed;
this is a compute cost model, not a full-step benchmark. Kernel dispatch gaps and
implementation overhead inside the isolated interval remain included.

There is no pipeline scheduler, VPP execution, NCCL group or P2P work. Single-rank
Gloo groups supply rank/size APIs required by Megatron layers. Native loss's TP=1
identity reductions are elided in a scoped guard; unexpected communication fails.
The full model is never constructed. Activations, gradients and modules are freed
between classes; allocator cleanup is between units only, not between samples.
An individual layer or vocabulary head still has to fit on the calibration GPU.

Defaults: 20 warmup pairs, 30 measured pairs, at least 10 accepted samples, and
population CV (`population_stddev / mean`) <= 0.15, independently for F and B.
The selected cost is the accepted median. With at least 30 raw samples, at most
one sample strictly above 10 times the raw median may be excluded from selection
and CV; its raw timing and index remain recorded. More such samples are not
discarded. These are empirical quality settings, not theoretical guarantees.
There is no partition/group consensus or regression fitting in this path.

`--isolated-profile-warmups`, `--isolated-profile-iterations` and
`--isolated-profile-max-cv` configure this policy. A failed check emits
"Profiling measurements are inconsistent. Rerun profiling before using this cost model."
The campaign/quality wrapper launches at most one complete retry in a fresh
process, retaining both attempts. A second failure blocks automatic solver use.
Direct collector invocation reports failure but does not itself retry.

### Diagnosing a Rejected Attempt

`collect_with_retry()` is a shared retry runner, not the legacy statistical
evaluator. For isolated results it validates the candidate with the isolated
validator directly. The pipeline evaluator (including group consensus and
cross-group median shifts) is only used for legacy profiles. The campaign no
longer constructs or forwards legacy `--quality-*` settings for isolated runs.
`--quality-cv` is not mapped onto `--isolated-profile-max-cv`.

Each `cost_profile.quality.json` and the corresponding entry in
`profiling_attempts.json` now includes `estimator`, `sampling`, `raw_data_path`
and `units`. A unit records class signature, layer type, member layer indices,
representative layer and configuration signature. Independent `forward` and
`backward` records contain raw/accepted samples, raw statistics, accepted
mean/median/stddev/CV/min/max, counts, discarded indices/fraction, selected cost,
failed rules and `passed`/`rerun_required` status. Boundary units are also checked.
No pipeline group metrics are generated. The unchanged catastrophic policy
allows at most one exclusion with >=30 raw samples, bounding the rejected
fraction by 1/30. Every omitted sample remains in the raw arrays.

Failure warnings and the final exhausted-retry exception identify the attempt,
class, phase, rule, actual CV, limit and raw-data path. For example, a synthetic
alternating 1/5 ms series produces `phase=backward rules=isolated_cv
CV=0.666667 limit=0.15 accepted=30/30 discarded=0`. Do not interpret the generic
`ProfilingQualityError` class or shared wrapper filename as evidence of pipeline
quality leakage; inspect the actual receipt's estimator, schema and failed rules.

The collector uses only isolated warmup/iteration flags. Compatibility
`--warmups 5 --iterations 10` cannot override `--isolated-profile-warmups 20
--isolated-profile-iterations 30`. Each raw unit records
`executed: {warmup_pairs: 20, measured_pairs: 30, initialization_forwards: 1}`;
the extra untimed forward prepares output shape/upstream gradient, and is not a
measured sample or warmup pair. Global sample indices are 20 through 49.
Schedule/PP/VPP/transport flags describe the target topology, not isolated runtime
execution. No partition0/partition1 collections or interleaved runtime are used.

## Commands

Run from the repository root with the existing `slackpipe-dev` container. Only
one visible GPU is needed for isolated collection; `--pp` and `--logical-stages`
describe the target plan topology, not the collector's process count.

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev env \
  OMP_NUM_THREADS=1 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
  MAMBA_DETERMINISTIC=1 TRITON_CACHE_AUTOTUNING=0 \
  NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
  /opt/venv/bin/python -m tools.slackpipe_profile_quality \
  --collect-output slackpipe_profiles/isolated-nemotron4b -- \
  /opt/venv/bin/python -m tools.slackpipe_isolated_profile calibrate \
  --model-config configs/slackpipe_eval/nemotron_h_4b.json \
  --pp 4 --logical-stages 8 --microbatches 8 \
  --seq-length 1024 --micro-batch-size 1 --precision bf16 \
  --isolated-profile-warmups 20 --isolated-profile-iterations 30 \
  --output unused
```

Use a fresh output path for each independent collection. Substitute
`configs/slackpipe_eval/llama_8b.json` for homogeneous profiling. No torchrun is
needed for the collector. Full runtime evaluation still needs the target GPUs.

For receipt-gated calibration and the normal downstream solver/runtime workflow:

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev \
  /opt/venv/bin/python -m tools.run_slackpipe_eval calibrate \
  --model-config configs/slackpipe_eval/nemotron_h_4b.json \
  --output slackpipe_experiments/isolated-campaign --precision bf16 \
  --calibration-estimator isolated-layer-compute-v1 --resume
```

Replace `calibrate` with `full` in the normal OR-enabled/four-GPU campaign
environment. Its native smoke prerequisites still require the target hardware.
The multi-model campaign forwards the same estimator and isolated policy flags.
Use `--calibration-estimator existing-stage-wall-time-v1` explicitly to retain
the old partition-based calibration and its unchanged quality checks.

Optional read-only comparison against a complete legacy calibration directory:

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev \
  /opt/venv/bin/python -m tools.slackpipe_compare_isolated \
  --profile slackpipe_profiles/isolated-nemotron4b/cost_profile.json \
  --pipeline LEGACY_CALIBRATION_DIRECTORY --output comparison.json
```

This verifies manifest, shapes and precision and reports exact `[i,j)` range
sums beside all raw pipeline per-iteration means (median across iterations).
It neither fits nor gates equality. Pipeline waits, overlap, resource contention
and wrapper work can legitimately make the two costs differ.

## Output and Receipts

`isolated_profile_raw.json` retains every measured pair with iteration IDs,
attempt, timestamp, host/GPU timings, configuration, environment and class keys.
`isolated_profile_summary.json` records classes, raw/accepted statistics, selected
costs, peak memory and collection runtime. `cost_profile.candidate.json` and
`cost_profile.quality.json` exist even when quality fails. Only passing collection
writes `cost_profile.json`. `profiling_attempts.json` identifies the selected
attempt; the full raw attempt directories remain intact.

The cost profile retains the existing schema (abbreviated field example below,
not experimental numbers):

```text
schema_version: slackpipe.cost_profile.v2
estimator: isolated-layer-compute-v1
units: microseconds
model_manifest_hash, cost_profile_hash
measurement_definition:
  measurement_mode: isolated
  includes_pipeline_wait: false
  includes_p2p: false
  includes_communication: false
layer_costs_us: [{layer_id, config_class, class_signature, forward_us, backward_us}, ...]
class_costs_us: {signature: {forward, backward}, ...}
prefix_forward_us: [0, F0, F0+F1, ...]
prefix_backward_us: [0, B0, B0+B1, ...]
stage_role_bias_us: {first: {forward, backward}, middle: {forward: 0, backward: 0}, last: ...}
isolated: {context, manifest, classes, units, summaries}
quality: {quality_schema_version, status, thresholds, issues}
```

Costs use half-open ranges: `prefix[j] - prefix[i]`. B is independently measured,
not a multiplier of F. The reader recomputes sample selection and class expansion
before accepting a profile. The nested manifest omits its schema marker for
compatibility with the existing C++ JSON reader; `model_manifest.json` is complete.

Calibration receipt/context v2 binds estimator, model/manifest hashes, execution
signature, class signatures, warmups, iterations, statistic and quality policy.
Changing them invalidates calibration and dependent optimized runs. Compatible
`--resume` reuses the profile. Isolated options do not invalidate unrelated native
benchmark contexts. Software/source/hardware provenance uses existing receipts.

## Limits

TP=DP=CP=1, no recomputation, no fused loss, no layer-index-dependent attention
scaling, no arbitrary per-block architecture overrides, and untied embeddings.
The generic model adapter's existing restrictions remain in force. A sum of
isolated layer medians is an additive compute approximation, not validation of
full-model prediction accuracy. Optimizer work, communication, contention,
pipeline scheduling and wrapper overhead require end-to-end evaluation.
Results on a small GPU must not be substituted for calibration on A100 targets.

Targeted checks:

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python -m pytest \
  tests/unit_tests/pipeline_parallel/test_slackpipe_isolated_profile.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_eval_receipts.py -q
```

Most cost/receipt tests use deterministic synthetic samples; the actual unit
autograd test needs one CUDA GPU and validates all six real unit types. It is not
a full-model numerical-equivalence or performance result.
