# PP4 and Hybrid Readiness

## Audit Before Edits

- `schedule._validate_supported_parallelism` rejected PP outside {1,2}; two
  SlackPipe-only guards in `training.arguments.validate_args` did the same.
  These now accept positive PP while keeping TP/DP/CP=1 restrictions.
- RMA required exactly two world ranks, created every NCCL group with `[0,1]`,
  and addressed all puts/waits with `1-rank`. Channels now derive their ordered
  source/destination and sorted subgroup ranks from each logical edge.
- RMA initialization used a world barrier per channel. Nonmembers now skip
  allocations; warmup synchronizes the subgroup. The persistent world Gloo
  completion barrier remains necessary before mailbox reuse.
- P2P already created distinct logical-edge groups, but warmup and destruction
  assumed membership in every group. Every rank still creates every group in
  order; only members retain, warm and destroy them.
- Runtime local-stage enumeration and message extraction were already generic.
  Compiled operations now also expose previous/next workers and edge IDs.
- RMA cache keys already include world/PP group identities, world size, plan W,
  rank, all logical edges/placement, B, shape, dtype, transport and device. N is
  determined by the full topology. Compatible splits/orders intentionally reuse
  mailboxes. Compiled-plan keys additionally include resolved path and file
  mtime/size, PP rank, execution mode, shape and dtype. New tests distinguish
  PP2/PP4, B, device, rank, shape and dtype contexts.
- Plan validation already handled arbitrary positive W, exact ranges, DAG and
  FIFO dependencies. The existing cyclic/divisibility validator is reused.
- Existing GPU tests intentionally gate on PP1/PP2 and the isolation fixture
  expects three edges. They remain regressions, not generalized coverage claims.
- The full training CLI had duplicate dataclass/manual registration of three
  SlackPipe flags. They are now excluded from automatic registration and both
  future launch commands pass the real argument parser and validator.
- CLI-to-core config conversion enabled pseudo-deallocation unconditionally,
  but SlackPipe retains full outputs for arbitrary backward order. The config
  adapter now disables that optimization only for SlackPipe. The native hybrid
  pretraining entry explicitly closes persistent SlackPipe resources on success.

## Four-Worker Topology

Rank 0 owns stages 0,4; rank 1 owns 1,5; rank 2 owns 2,6; rank 3 owns 3,7.
Local chunk is `stage // W`; worker is `stage % W`. Seven logical edges map to
`(0,1), (1,2), (2,3), (3,0), (0,1), (1,2), (2,3)`. Repeated rank pairs retain
distinct communicators. RMA has fourteen directional channels globally and
6,8,8,6 local channels on ranks 0,1,2,3. Peer indices are positions in the sorted
two-member subgroup, including the wraparound edge. Groups/windows are created
outside the operation loop and released by collective explicit shutdown.

## Pinned Hybrid Mechanism

- `pretrain_hybrid.py` uses `hybrid_builders.hybrid_builder` and the legacy
  `training.get_model` VPP construction path.
- `core/models/hybrid/hybrid_model.py` parses the hybrid pattern and calls
  `hybrid_layer_allocation.select_pipeline_segment`. Segment index is
  `vp_stage * pp_size + pp_rank`; its global offset sums all preceding segments.
- `hybrid_block.HybridStack` instantiates layers with one-based global
  `layer_number`, without adding a second PP offset.
- `hybrid_layer_specs.hybrid_stack_spec` maps `M` to `MambaLayer`, `*` to a
  `TransformerLayer` with `SelfAttention` and an identity MLP, and `-` to
  `MLPLayer`. Pinned symbols also include GDN, DSA and MoE, but these and MTP
  are deliberately rejected by the SlackPipe hybrid adapter.
- The newer `training.models.hybrid.HybridModelBuilder` still rejects VPP;
  these launchers do not use that path. No upstream HybridModel or ordinary
  scheduler algorithm was changed.

Hybrid manifests derive from the actual config and unpartitioned pattern.
Partition boundaries are not part of model identity. Stage IDs, types and class
counts describe exact half-open global ranges. Runtime checks unwrap Megatron
wrappers before inspecting native layers. The tiny fixture is `M-*MM-*-M-*-`,
12 blocks, hidden size 128, sequence length 64, eight microbatches. Its PP4 cuts
are `[0,1,3,4,6,7,9,10,12]`. PP1/PP2 use four nonuniform chunks.

## Solver Proof

A synthetic (not measured) class-cost profile was fitted through the existing
`build_heterogeneous_cost_profile` regression and expanded to all twelve global
layers. The unchanged OR-enabled CLI ran `joint-unrestricted-no-overlap` with
`--B 8 --N 8 --J 4 --L 12 --ratio-num 1 --ratio-den 1
--time-limit-seconds 15 --num-workers 2 --random-seed 1 --require-optimal false`.
The evaluator validated the result before plan export. Result: OPTIMAL, split
`[1,2,1,2,2,1,2,1]`, 128 operations, predicted abstract makespan 2520.
The production parser accepted plan.v2 and validated model/profile fingerprints.
This proves schema/solver compatibility, not GPU execution or measured speed.

Actual tiny-hybrid stage calibration also completed on PP2 with the ordinary
interleaved schedule: six partitions, 24 aggregated stage observations, all
three native block classes, and regression rank 6/6. Its measured profile was
consumed by the unchanged joint solver, then the exported plan.v2 executed with
RMA and all four numerical differences zero. Calibration and solver artifacts
remain under `/tmp`, not in the repository.

## Local Regression Record

- The expanded suite contains 83 tests: PP1 runs 71 and skips 12; each PP2
  transport selection runs 77 and skips 6 per rank. All applicable cases pass
  with deterministic hybrid kernels. Skips retain explicit world/device reasons.
- Native hybrid initial parameters, loss, every gradient and post-SGD parameters
  match exactly at PP1 and PP2. Strict comparisons were not loosened. Enable
  `MAMBA_DETERMINISTIC=1 TRITON_CACHE_AUTOTUNING=0
  NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` before imports: nondeterministic kernel
  runs showed repeat-dependent differences, documented in the validation matrix.
- C++ no-OR and OR-enabled CTest pass 4/4 each; OR validation additionally passes
  its 14 tiny CLI export/evaluator smoke cases. No C++ semantics changed.
- Ten persistent-cache reuse cycles and ten Megatron-context recreation cycles
  pass with the measured solver plan. Both ranks end with zero live contexts,
  mailboxes and windows; the only remaining process group is WORLD before exit.
- Tiny full `pretrain_hybrid.py` CLI training completed on P2P and RMA, using
  mock tokens and native three-type blocks, never 8B weights. The RMA launch
  takes a nonzero-learning-rate SGD step and closes its SlackPipe resources.
  The upstream CLI still warns that WORLD is not explicitly destroyed at exit;
  targeted lifecycle harnesses explicitly destroy their process groups.
- Full baseline/SlackPipe Nemotron argument parsing, config/manifest identity,
  Python import ordering, formatting of new helpers, shell syntax, and
  `git diff --check` pass. Four-GPU tests were intentionally not executed.

## RunPod Acceptance

The real four-rank hybrid test checks ownership, VPP indices, pre/post roles,
global IDs/types, complete logical parameter coverage, initial values, loss,
every gradient, one SGD step, exact traces and empty transport state after
shutdown. Run it separately with P2P and RMA via the documented scripts.
On fewer than four visible devices it explicitly skips, never reports PASS.
The two-GPU host cannot establish PP4 NCCL progress, subgroup initialization,
RMA addressing or numerical correctness. Run those checks on A100 SXM x4 before
attempting the offline adapter's full 8B one-step smoke launch.

No weights were downloaded. The Base-8K architecture adapter records the exact
released source revision. FP32 allocation/activation capacity, eight-stage 8B
execution, long sequences, BF16 and multi-node remain unvalidated. P2P's known
delayed-matching progress limitation is unchanged; arbitrary optimized orders
must also be validated with the experimental RMA path.

## Changed Files

Paths below are relative to the repository root; `runtime` denotes
`megatron/core/pipeline_parallel/slackpipe/`.

| Files | Change |
| --- | --- |
| `runtime/topology.py` | Pure logical edges, directional channel identities, cyclic local chunks |
| `runtime/communication.py` | Deterministic group creation with nonmember-safe retention |
| `runtime/communication_rma.py` | Generic endpoint groups and subgroup-local RMA peers |
| `runtime/hybrid.py` | Pattern/range helpers and sourced offline 8B configuration |
| `runtime/manifest.py` | Hybrid identity, native type/config/shape validation, wrapper unwrapping |
| `runtime/schedule.py` | Positive PP, compiled neighbors/edges, hybrid validation and feature guards |
| `megatron/training/arguments.py` | Generic PP guards, hybrid plan layout, duplicate CLI registration fix |
| `megatron/training/argument_utils.py` | SlackPipe-specific output retention configuration |
| `hybrid_builders.py`, `pretrain_hybrid.py` | Bind the global pattern and explicitly close SlackPipe after training |
| `tests/unit_tests/pipeline_parallel/test_slackpipe_topology.py` | Structural topology/cache/range/profile and full CLI tests |
| `tests/unit_tests/pipeline_parallel/test_slackpipe_hybrid.py` | Native strict PP1/PP2 and gated PP4 equivalence fixture |
| `tools/slackpipe_heterogeneous_experiment.py` | Ordinary tiny-hybrid stage calibration option |
| `tools/slackpipe_hybrid.py` | Offline manifest, generic class fitting and validated launch commands |
| `tools/runpod_slackpipe_verify.sh`, `tools/runpod_slackpipe_correctness.sh` | Pod capability checks and gated four-rank correctness launch |
| `tools/runpod_nemotron_baseline.sh`, `tools/runpod_nemotron_slackpipe.sh` | Future random-init 8B one-step launchers |
| `README.md`, `runtime/README.rma.md` | Workflow, topology, prerequisites and limitations |
| `docs/slackpipe_validation_matrix.md`, this report | Validation boundary, audit and results |

The C++ source tree and ordinary Megatron schedules are unchanged. The two
untracked campaign scripts are preserved, and generated artifacts are excluded.
