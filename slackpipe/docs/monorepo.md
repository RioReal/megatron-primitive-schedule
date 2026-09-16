# Monorepo Development

`slackpipe/` contains the C++ optimizer, deterministic DAG/predecessor evaluator,
result validator, and Megatron plan exporter. It is a normal directory in the
same Git repository as the runtime. No separate clone or submodule is needed.

## Source Import

The import preserves solver source from historical commit
`415ae8da60ede4aa69c5703cbecb6dafabec040b` (branch `eval`) plus the working-tree
plan export, calibrated costs, heterogeneous prefix/range costs, joint cut
optimization, evaluator consistency, and validation changes used by the runtime.
Packaging does not change search or schedule semantics. The original repository
history and complete dirty checkout were backed up outside the monorepo before
its Git metadata was moved.

- `apps/`: CLI and benchmark driver.
- `include/slackpipe/`, `src/`: solver, evaluator, native operations, cost models,
  schema export, and validation.
- `tests/`: C++ tests and reusable C++-CLI/evaluation Python tests.
- `scripts/`, `configs/`: reproducible evaluation runners, analysis, and inputs.
- `third_party/gtest/`: small source-only test fallback used when GTest is absent.
- `docs/`, `EVALUATION.md`, `paper/`: evaluation protocol and paper source.

Legacy Python simulator scripts/tests and packaging referring to modules absent
from this checkout are not imported. Neither are agent-install tooling, local
prompt/debug archives, generated manifests/results/paper tables, binaries,
OR-Tools dependencies, environments, or build caches. Paper inputs must be
regenerated; see [`../paper/ARTIFACTS.md`](../paper/ARTIFACTS.md).

## Build and Validate

Use the [root Docker recipes](../../README.md#build-the-optimizer). The existing
`slackpipe-dev` container provides the no-OR C++ toolchain and Megatron test
environment. OR-enabled tests use `slackpipe-ortools-runtime:local` with
`/opt/or-tools`, mounting the entire monorepo. The optional solver-only
`Dockerfile` supplies build tools, not OR-Tools or Megatron dependencies.

From the monorepo root in a suitable build environment:

```bash
cmake -S slackpipe -B slackpipe/build/no-or -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSLACKPIPE_ENABLE_ORTOOLS=OFF
cmake --build slackpipe/build/no-or -j2
ctest --test-dir slackpipe/build/no-or --output-on-failure
BUILD_DIR="$PWD/slackpipe/build/release" BUILD_JOBS=2 \
  SLACKPIPE_OR_TEST_FILTER='.*' \
  bash slackpipe/scripts/validate_ortools_evaluation.sh
```

Never reuse a CMake cache configured at another checkout path. Set
`ORTOOLS_PREFIX` when the external C++ package is elsewhere. CP-SAT coverage
requires the OR-enabled build; no-OR tests alone do not exercise search.

When a container runs under a different UID, Git may reject the mounted checkout
as dubious ownership. Run as the checkout owner where possible. For this trusted
development mount only, `docker exec slackpipe-dev git config --global --add
safe.directory /workspace/Megatron-LM` permits provenance lookup; do not use a
wildcard trust rule. Missing Git metadata is valid for source archives, so tests
of commit matching use an explicit expected commit rather than the host checkout.

Schema/cost changes also require the production Megatron parser and a PP=2
solver-to-runtime numerical-equivalence test inside `slackpipe-dev`. Keep
generated cost profiles/plans/traces under the ignored artifact directories
documented in the root README. Stage explicit source paths only.
