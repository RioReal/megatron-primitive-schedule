# Isolated Calibration Validation, 2026-10-01

This records implementation validation, not an A100 performance claim. Commands
ran in the existing `slackpipe-dev` and OR-Tools containers. Hardware was two
RTX A2000 6 GiB GPUs; each isolated collection used only GPU 0. Software was
PyTorch 2.12.0a0+0291f960b6.nv26.04.48445190, CUDA 13.2, TE 2.16.0+b9d690e0,
Mamba-SSM 2.3.2.post1. Deterministic environment matches the
[collector commands](slackpipe_isolated_calibration.md). No dependencies installed.

## Scope and Changes

- `isolated_profile.py`: full class signatures, ordered expansion, independent
  F/B prefixes, boundary biases, simple quality policy and raw-data revalidation.
- `tools/slackpipe_isolated_profile.py`: real TE/GPT/hybrid modules, no pipeline
  communication, one-unit memory lifecycle, synchronized timing and raw artifacts.
- `tools/run_slackpipe_eval.py`: default estimator, single-process dispatch and
  estimator-scoped v2 contexts. Legacy estimator remains explicitly selectable.
- Both `profile_quality.py` modules: isolated quality dispatch and at-most-one
  fresh-process retry; requested sample policy/class identity checked on receipt.
- `tools/slackpipe_eval_worker.py`: route isolated calibration to its own collector;
  check v2 manifest identity for homogeneous as well as hybrid models.
- `tools/slackpipe_compare_isolated.py`: optional hash-bound, exact-range diagnostic
  comparison with legacy wall-time records, without an equality gate.
- Tests and README/evaluation/calibration guides cover the new flow and limits.

Old flow: candidate pipeline partitions -> effective stage wall samples ->
group checks/fitting -> costs. New flow: manifest -> equivalent real layer units
-> isolated F/B event samples -> median costs -> ordered expansion -> v2 prefixes.
No solver, predecessor, placement, transport or runtime algorithms were changed.

The profiling unit, gradient construction, timing boundaries, signature fields,
boundary handling and schema example are specified in the calibration guide.
Notably, optimizer work is excluded; CUDA-event times include isolated dispatch
gaps and are not pure FLOP time. TP=1 native loss identity reductions are elided;
only Gloo rank/size metadata groups exist during collection.

## Collected Results

All collections used 20 warmup F/B pairs and 30 measured pairs per unit. Tiny
models used seq=32, micro batch=1, FP32. Large presets used seq=1024, micro batch=1,
BF16, target PP4/N8/B8 metadata (no PP4 execution during calibration). Full models
were **never constructed**. These are seeded synthetic-training configurations,
not pretrained-weight evaluations.

| Model | Layer classes + boundary units | Collection seconds | Peak allocated bytes | Peak reserved bytes |
|---|---:|---:|---:|---:|
| Tiny LLaMA, L8/H128 | 1 + 2 | 1.581 | 23,180,800 | 46,137,344 |
| Tiny hybrid, L12/H64 | 3 + 2 | 3.373 | 21,509,632 | 44,040,192 |
| LLaMA-style 8B | 1 + 2 | 19.875 | 4,870,400,000 | 5,681,184,768 |
| Nemotron-H-style 4B | 3 + 2 | 16.527 | 3,354,677,760 | 3,802,136,576 |

Collection time includes group setup, module construction, warmup, measured pairs,
finite checks, intermediate raw writes and cleanup. It excludes Python startup
and final serialization. It is not training iteration time or solver time.

| Model/unit | F median ms | F CV % | B median ms | B CV % |
|---|---:|---:|---:|---:|
| Tiny decoder | 1.56416 | 1.427 | 2.17242 | 11.373 |
| Tiny Mamba | 1.85859 | 1.734 | 2.55072 | 1.043 |
| Tiny attention | 1.00488 | 1.693 | 1.60461 | 4.552 |
| Tiny MLP | 0.47899 | 3.987 | 1.05216 | 10.248 |
| 8B decoder | 16.95597 | 0.736 | 35.54042 | 0.708 |
| 8B embedding | 0.14376 | 2.356 | 4.50661 | 0.211 |
| 8B norm/head/loss | 54.95493 | 0.680 | 81.98451 | 1.529 |
| 4B Mamba | 5.11619 | 0.214 | 11.29827 | 0.203 |
| 4B MLP | 6.00798 | 0.733 | 12.15846 | 0.853 |
| 4B attention | 2.19749 | 3.798 | 5.29510 | 0.703 |
| 4B embedding | 0.11109 | 3.188 | 3.16403 | 0.176 |
| 4B norm/head/loss | 44.66304 | 0.737 | 57.76947 | 0.971 |

Large final collections passed attempt 0 with no exclusions. The initial tiny
hybrid collection failed quality (MLP F CV 71.2%, one ~2.415 ms observation versus
~0.43 ms baseline, below the catastrophic exclusion ratio). One fresh recollection
passed. Both are retained; no thresholds changed. An initial constructor error
(missing RoPE fraction) was fixed before the first successful tiny run. Initial
large runs also passed; they were repeated after releasing finite-check tensor
references earlier in the loop. No improvement claim is made from those reruns.

Large class signatures and zero-based member layer IDs:

```text
8B decoder: 5d7c6af841962082bc89a918e4bb2ed19845735f012b325503490a0e2923af25
  0..31
4B Mamba: 7142c060afdb16e0aafe2d7b66ad4313c5a67ea9286c5edbea7d6395ec738781
  0,2,4,6,9,11,13,15,17,20,22,24,26,28,31,33,35,37,39,42,44,46,48,50
4B MLP: e3e77ab4795dbb753d0eb9d8f0b3f48d39f38f8a208160d72444175c5e24c200
  1,3,5,8,10,12,14,16,19,21,23,25,27,30,32,34,36,38,41,43,45,47,49,51
4B attention: 1ff537f815312ffe28414d8b15ff799f8be2cfbc32b3af6edeff82af875b6521
  7,18,29,40
```

## Previously Anomalous Ranges

Each requested half-open range contains 3 Mamba, 2 MLP and 1 attention units in
this manifest, so additive isolated predictions are identical by construction.
They exclude boundary biases (all three ranges are internal).

| Range | Isolated F ms, A2000 | Isolated B ms, A2000 | Previous pipeline B ms, as supplied by user |
|---|---:|---:|---|
| [13,19) | 29.56203 | 63.50685 | 15.9485 / 15.9581 |
| [26,32) | 29.56203 | 63.50685 | 22.7214 / 22.7077 |
| [39,45) | 29.56203 | 63.50685 | 22.6753 / 22.9022 |

The previous numbers were not revalidated from raw data, and are **not a matched
hardware/run comparison**. No ratio, accuracy claim or quality rejection is
derived from this table. Neither overlap/contention nor intrinsic-kernel causes
of the previous pipeline variation were established by this implementation.

## Provenance and Consumption

Raw local artifacts are under the ignored `slackpipe_experiments/isolated_20261001/`:
`llama-v2`, `hybrid`, `hybrid-retry`, `llama8b-final`, `nemotron4b-final`.
Earlier large attempts remain in `llama8b` and `nemotron4b`. They are not bundled
into the source commit. Each raw/profile artifact records source base
`8d528c69b18ef8e3179b01ddd6d22b190df2a704`, working-diff hash and untracked source
hashes for the code actually executed, plus software/hardware/configuration.

Final profile content hashes:

```text
LLaMA 8B: 50a38b28f085607fef056fc25a599340baaf90cb5a0fd75d93d68cdf9d357b06
Nemotron 4B: b14c4795a3fa3b164c8b7fe4cba56e71670babfd3010f76c468f4f46be73cb09
```

Both final profiles were consumed by the unchanged OR-enabled CLI, with
`--algorithm slackpipe --B 8 --N 8 --J 4 --communication 0
--time-limit-seconds 5 --num-workers 1 --require-optimal false`, L=32/52 and their
respective `--cost-profile`. Both returned **FEASIBLE**, source `joint_cpsat`,
no incumbent fallback, not proven globally optimal. Evaluator-approved exports
contained 128 operations, and the production Python parser verified v2 manifest
and profile provenance. Splits were `[3,4,5,4,6,5,4,1]` and `[6,7,8,5,9,9,7,1]`.
Predicted makespans were 4161.169 and 2215.213 ms, respectively; these are model
predictions from A2000 costs, not measured iteration times or A100 predictions.

The actual measured tiny homogeneous profile also passed no-OR baseline export
and PP2/N4/B4 execution via `tools.slackpipe_eval_worker smoke`, including finite
loss/gradients and optimizer update checks. No full-size PP4 training was run.

## Validation and Limits

- No-OR and OR-enabled CTest: 4/4 each. The OR validation script also validated
  14 canonical CLI results. An existing root-owned build directory was unusable
  as the normal user; validation used a fresh `slackpipe/build/isolated-or` tree.
- Targeted Python: **201 passed, 6 skipped** in the single-process suite; skipped
  cases require two or four ranks. Two-rank tests were separately launched.
  Coverage includes isolated class/quality/serialization, raw tamper rejection,
  dispatch, retries, v2 receipts/resume, legacy outlier/group/fit checks, plan/cost
  parsers and manifest/construction checks. Actual real-unit F/B repeatability
  tests cover decoder, Mamba, attention, MLP, embedding and output/loss.
- Tiny homogeneous and hybrid PP1/PP2 equivalence: zero maximum differences for
  initial parameters, loss, gradients and post-SGD parameters; hybrid includes
  BF16 as well as FP32. Four-GPU cases remain explicitly skipped on this host.
- A synthetic isolated-v2 fixture matching the hybrid equivalence model also
  passed C++ export -> PP2 numerical equivalence with all differences zero.
  This checks schema/runtime integration, not measured-cost prediction accuracy.
  A mistaken input/output filename collision in the first invocation was caught
  by runtime profile-hash validation; separate input/output paths passed.
- The staged-only source snapshot passed all 88 isolated/receipt tests without
  the unrelated worktree diagnostics. Formatting, focused Ruff checks and
  `git diff --check` passed.
- No A100 calibration, full-model prediction-accuracy test, PP4 runtime run,
  16B/30B/32B memory validation, or matched legacy-vs-isolated hardware comparison.
  Individual large layers/heads must still fit; sums of isolated medians omit
  communication, wrapper overhead, optimizer and context-dependent contention.
- The untracked scale-sweep tools and prior overlap diagnostic changes are
  intentionally excluded from this commit. No paper or historical data changed.
