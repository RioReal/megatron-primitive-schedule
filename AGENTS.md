# Repository Guidelines

## Skills

The `skills/` directory contains structured guides for common tasks (running
tests, building containers, managing dependencies, submitting SLURM jobs, etc.).
**Always read the relevant `SKILL.md` before starting any task it covers —
skills are mandatory context, not optional background reading.**

**Workflow — mandatory order for every task:**
1. **Pull information first.** Read the commit, PR, error log, file, or
   whatever artifact the task is about. Do not reason about it yet.
2. **Select and invoke the skill.** Based on what you just read, identify
   the relevant skill and invoke it before forming any answer or plan.
3. **Answer or implement.** Only after the skill is loaded, use its context
   to reason, diagnose, or write code.

Never skip or reorder these steps. Do not wait for the user to name the right
skill keyword — infer it from the artifact you read.

## SlackPipe Monorepo

- `slackpipe/` is the tracked C++ optimizer/evaluator/exporter, not a separate
  Git checkout. Never initialize a nested repository there.
- Megatron execution and tests run in the existing `slackpipe-dev` container
  (checkout at `/workspace/Megatron-LM`, Python `/opt/venv/bin/python`). Do not
  install Megatron dependencies on the host. This machine has two visible GPUs.
- Solver changes require no-OR and OR-enabled C++ tests. Configure the no-OR
  build with `cmake -S slackpipe -B slackpipe/build/no-or -G Ninja
  -DSLACKPIPE_ENABLE_ORTOOLS=OFF`, build, and run CTest inside the container.
- The existing `slackpipe-ortools-runtime:local` image supplies OR-Tools.
  Mount the whole monorepo and run `slackpipe/scripts/validate_ortools_evaluation.sh`
  with `SLACKPIPE_OR_TEST_FILTER=.*`; see README for the complete command.
- Plan/cost changes also require parser/manifest tests and a solver-to-PP=2
  numerical-equivalence smoke test. Do not change solver semantics as part of
  packaging or build-workflow changes.
- Keep build trees, environments, generated plans/results/profiler output, and
  paper build products out of commits. Stage explicit source paths only.

## Contributing

### Pull Requests

- All PRs must be created as **drafts**. Use `gh pr create --draft` or the GitHub UI draft option.
- Never push branches directly to `https://github.com/NVIDIA/Megatron-LM`. You must push your branch to a personal fork (e.g. `https://github.com/<your-username>/Megatron-LM`), then open a PR from the fork's branch against `NVIDIA/Megatron-LM`.
- Read @docs/developer/contribute.md for the full contribution policy, including code style, commit message conventions, and issue guidelines.

### Code Quality

- After editing imports in any Python files, always run `uv run isort` on those files to fix import order before committing.
