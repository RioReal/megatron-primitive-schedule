# Experimental NCCL RMA transport

Select `--pipeline-schedule slackpipe --slackpipe-plan PLAN --slackpipe-transport nccl-rma`.
The default is `nccl-p2p`. The ordinary Megatron schedules do not use this option.
The solver plan and worker operation order are identical for both transports.

This prototype requires two ranks, cyclic placement, fixed CUDA FP32 tensor
shapes, and NCCL >= 2.29. It uses the installed PyTorch private
`torch.distributed._symmetric_memory` NCCL backend. It does not require nccl4py,
a local compiled extension, or host dependencies. The capability and tests were
checked with PyTorch 2.12.0a0, CUDA 13.2, and NCCL/header 2.29.7. Other builds must
pass the isolation test before use. Selecting the NCCL symmetric-memory backend
is a process-wide setting; mixing symmetric-memory backends is unsupported.

## Storage and initialization

Each cross-worker logical edge gets separate forward and backward process
groups. For PP=2/N=4 there are six groups, all containing ranks `[0, 1]`, created
in increasing edge order and then forward/backward order. A separate Gloo group
provides step-boundary CPU synchronization, never tensor transport.

Each channel allocates a persistent `ncclMemAlloc` buffer on both GPUs. Its slots
are indexed by microbatch, with stride `align_up(tensor_bytes, 4096)`. The sender
uses its local slots for staging; the receiver uses its peer slots as mailboxes.
All microbatches have independent slots. In this PyTorch build each allocation
registers one data window and one 9,216-byte internal signal window. Thus there
are 12 registrations per rank (24 across the two ranks), once at initialization.
Actual allocation sizes can exceed requested sizes due to VMM granularity.

Two installed API details matter:

- NCCL 2.29 requires the put source to be in a registered symmetric window too.
  The producer tensor is copied into its registered staging slot before the put.
- PyTorch rendezvous uses a tensor's storage base, ignoring `storage_offset`.
  Each slot is rebased with `torch.from_dlpack` without a data copy. The DLPack
  deleter retains the parent allocation. Slot handles share the underlying
  registration but carry different byte offsets.

A one-message initialization warmup on each channel initializes NCCL's lazy RMA
state collectively. It is fully consumed before training begins. No training
receives are preposted and no windows are registered in the operation loop.

## Streams and lifetime

```text
producer compute -> event -> channel stream: staging copy -> put -> completion event
                                     (source retained until completion)

channel stream: wait for next signal -> event -> consumer stream: copy -> F/B compute

end of step: finish puts + finish consumer stream -> CPU barrier -> allow slot reuse
```

There is no send-completion dependency on unrelated compute. Only the consuming
operation enqueues a signal wait. The installed wrapper's `opCnt=1` consumes one
additional signal; NCCL maintains the cumulative count. FIFO assertions enforce
microbatch order on every channel, and forward/backward never share counters.

The consumer copy returns an ordinary PyTorch-owned tensor, detached and marked
`requires_grad` for forward inputs. This prevents model-held input references
from delaying window deregistration until after communicator destruction.
Source references remain held through put completion. At the step boundary all
send events complete, all expected messages have been consumed, and the compute
stream completes before the peer can reuse any slot. There is no device-wide
synchronization in the transport operation loop. This implementation favors
lifetime correctness over memory or copy optimization.

Call `shutdown_slackpipe_runtime()` collectively before destroying distributed
groups. `clear_slackpipe_runtime_cache()` remains a backward-compatible alias.
Megatron's `destroy_global_state()` and the benchmark/test cleanup do this.
Registered allocations and signal handles are released before channel groups.

Compiled plans and persistent transports have separate caches. The RMA key
includes the default and PP process-group identities, world/PP sizes, rank,
logical edges and placement, shape, dtype, microbatch count, transport, and CUDA
device. Compatible plans (including different splits/orders) reuse the same
registrations. `clear_slackpipe_plan_cache()` drops plans without destroying RMA.
The communicator retains dimensions/placement, not the old plan or model.
Changing distributed groups while the cache is live is rejected; shutdown must
precede group destruction. Changing shape or microbatch count uses a distinct
transport allocation, released at explicit shutdown.

The repeated-context regression exposed a test ownership issue: Megatron's
`parallel_state.destroy_model_parallel()` clears most NCCL references without
destroying their PyTorch-registered groups. The SlackPipe-only test fixture now
tracks and explicitly destroys groups it created. Production RMA does not
destroy groups owned by Megatron or another caller, and does not inspect private
PyTorch process-group registries. `empty_cache()` is not a lifecycle fix.

## Validation and profiling

Run inside `slackpipe-dev`, using `/opt/venv/bin/python -m torch.distributed.run
--standalone --nproc-per-node 2 -m pytest` with
`tests/unit_tests/pipeline_parallel/test_slackpipe_rma.py`.

The communication regression completes all puts on both ranks before either
issues its first consumer wait, checks all six channels/slot offsets, exercises
FIFO rejection, and reuses the same registrations across three iterations.
The checked-in M1 fixture preserves the exact B(0,1) index 12 to B(0,0) index 24
delay and 64-operation schedule. Numerical tests compare every logical parameter,
loss, gradient, and post-SGD value to an ordinary full-model reference.

The existing performance harness accepts `--slackpipe-transport`. Its per-rank
results include initialization duration, requested mailbox/window sizes, and a
device-wide free-memory delta including allocations outside PyTorch's allocator.
The optional SlackPipe operation profile adds `recv_gpu_wait_ms`, measured on
the consumer stream around the receive call (including the RMA consumer copy).
`recv_wait_ms` remains the original CPU call duration. These are diagnostic
measurements from a separate profiled iteration; measured benchmark steps need
not enable per-operation timing.

Still unsupported: PP>2, TP/DP/CP, BF16/FP16, variable shapes, distributed
optimizer, recomputation, CUDA graphs, communication-overlap optimizations,
mailbox reuse within a step, and untested multi-node deployment. The tested
transport is true NCCL RMA, not CUDA IPC or two-sided receive preposting.

NCCL semantics: <https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2297/user-guide/docs/api/p2p.html>.
