# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real orchestration/receipt IO with synthetic workers, never an 8B allocation."""

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from tools import run_slackpipe_real_system_campaign as campaign
from tools.run_slackpipe_eval import Experiment, argument_parser
from tools.run_slackpipe_nemotron_h8b_pp4 import digest
from tools.slackpipe_eval_config import COMPARISON_SCHEDULES, fingerprint, parameter_breakdown
from tools.slackpipe_eval_receipts import (
    EMPTY_DIFF,
    LEGACY_COMMIT,
    RECEIPT_SCHEMA,
    assess_receipt,
    legacy_source_compatible,
)
from tools.slackpipe_hybrid import write_json

CONFIG = Path(__file__).resolve().parents[3] / "configs/slackpipe_eval/llama_8b.json"


def args_for(root, stage="solve", **updates):
    args = argument_parser().parse_args(
        [
            stage,
            "--model-config",
            str(CONFIG),
            "--output",
            str(root),
            "--resume",
            "--calibration-estimator",
            "existing-stage-wall-time-v1",
        ]
    )
    for key, value in updates.items():
        setattr(args, key, value)
    return args


@pytest.fixture
def synthetic(monkeypatch):
    calls = Counter()

    def base(self):
        a = self.args
        return dict(
            model_config_hash=fingerprint(self.model),
            topology=self.topology,
            precision=a.precision,
            seq_length=a.seq_length,
            micro_batch_size=a.micro_batch_size,
            transport=a.transport,
            source=dict(commit="fixture", diff=EMPTY_DIFF),
            environment={"fixture": True},
        )

    monkeypatch.setattr(Experiment, "_base_context", base)

    def execute(self, command, directory, name):
        calls[name] += 1
        if name == "correctness":
            (directory / "correctness.xml").write_text("<testsuite><testcase/></testsuite>")
        elif name == "solve":
            a = self.args
            n, b, pp = self.topology["num_stages"], a.microbatches, a.pp
            cuts = [self.model["num_layers"] * s // n for s in range(n + 1)]
            ops = [[] for _ in range(pp)]
            for mb in range(b):
                for kind, stages in (("F", range(n)), ("B", reversed(range(n)))):
                    for s in stages:
                        ops[s % pp].append(dict(kind=kind, microbatch=mb, stage=s))
            write_json(
                directory / "slackpipe.plan.json",
                dict(
                    schema_version="slackpipe.plan.v1",
                    num_microbatches=b,
                    num_stages=n,
                    num_workers=pp,
                    num_layers=self.model["num_layers"],
                    layer_split=[y - x for x, y in zip(cuts, cuts[1:])],
                    stage_to_worker=[s % pp for s in range(n)],
                    operations=ops,
                    cost_model=dict(path=str(self.output / self.completed["calibrate"]["profile"])),
                ),
            )
        else:
            pytest.fail(f"unexpected command: {command}")

    def worker(self, stage, directory, **kwargs):
        calls[f"{self.args.schedule}.{stage}"] += 1
        calls[(self.args.schedule, stage, self.args.run_index)] += 1
        directory.mkdir(parents=True)
        a = self.args
        if stage == "calibrate":
            from megatron.core.pipeline_parallel.slackpipe.profile_quality import publish_profile
            from tests.unit_tests.pipeline_parallel.test_slackpipe_profile_quality import (
                stable_profile,
            )

            profile = dict(
                schema_version="slackpipe.cost_profile.v1",
                model_config=dict(
                    model_config_hash=fingerprint(self.model),
                    dtype=a.precision,
                    sequence_length=a.seq_length,
                    micro_batch_size=a.micro_batch_size,
                ),
                parallel_config={k: self.topology[k] for k in ("pp", "vpp", "tp", "dp", "cp")},
            )
            profile.update(stable_profile())
            if a.calibration_estimator == "isolated-layer-compute-v1":
                from megatron.core.pipeline_parallel.slackpipe.isolated_profile import (
                    execution_signature,
                )
                from megatron.core.pipeline_parallel.slackpipe.manifest import build_model_manifest
                from tests.unit_tests.pipeline_parallel.test_slackpipe_isolated_profile import (
                    fixture_profile,
                )
                from tools.slackpipe_eval_config import transformer_config

                isolated = fixture_profile(
                    warmups=a.isolated_profile_warmups,
                    iterations=a.isolated_profile_iterations,
                    max_cv=a.isolated_profile_max_cv,
                    manifest=build_model_manifest(
                        transformer_config(self.model, self.topology, a.precision)
                    ),
                    execution=execution_signature(
                        self.model,
                        seq_length=a.seq_length,
                        micro_batch_size=a.micro_batch_size,
                        precision=a.precision,
                    ),
                )
                isolated.update(
                    model_config=profile["model_config"], parallel_config=profile["parallel_config"]
                )
                from megatron.core.pipeline_parallel.slackpipe.cost_profile import (
                    profile_fingerprint,
                )

                isolated["cost_profile_hash"] = profile_fingerprint(isolated)
                write_json(directory / "cost_profile.json", isolated)
                write_json(directory / "cost_profile.candidate.json", isolated)
                write_json(directory / "cost_profile.quality.json", isolated["quality"])
                return
            publish_profile(
                directory / "cost_profile.json", profile, attempt=kwargs.get("attempt", 0)
            )
        else:
            config = dict(
                self.topology,
                model_config_hash=fingerprint(self.model),
                exact_parameter_count=parameter_breakdown(self.model)["exact_parameter_count"],
                micro_batch_size=a.micro_batch_size,
                global_batch_size=a.microbatches * a.micro_batch_size,
                seq_length=a.seq_length,
                precision=a.precision,
                seed=a.seed,
                learning_rate=a.learning_rate,
                tf32=False,
                dropout=0,
                schedule=a.schedule,
            )
            for rank in range(a.pp):
                write_json(
                    directory / f"result.rank{rank}.json",
                    dict(
                        rank=rank,
                        run_id=directory.name,
                        method=a.schedule,
                        config=config,
                        options=dict(warmup=a.warmups, iterations=a.iterations),
                        samples=[
                            dict(iteration=a.warmups + i, cuda_elapsed_ms=10 + rank + i)
                            for i in range(a.iterations)
                        ],
                        continuous_wall_ms=100 * a.iterations,
                        memory_boundaries={
                            "end": dict(peak_allocated_bytes=100, peak_reserved_bytes=200)
                        },
                    ),
                )

    monkeypatch.setattr(Experiment, "_execute", execute)
    monkeypatch.setattr(Experiment, "_worker", worker)
    import torch

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(
        "tools.run_slackpipe_eval.memory_preflight", lambda *a: dict(status="passed")
    )
    monkeypatch.setattr(campaign, "plot_campaign", lambda *a: None)
    return calls


def execute(root, stage="trace", **kwargs):
    exp = Experiment(args_for(root, stage, **kwargs))
    assert exp.execute()["status"] == "passed"
    return exp


def directories(exp):
    return {k: r["directory"] for k, r in exp.completed.items()}


def test_exact_standalone_three_campaign_one_reproduction(tmp_path, synthetic):
    first = execute(tmp_path, "solve", repetitions=3, warmups=5, iterations=50)
    before = synthetic.copy()
    second = execute(tmp_path, "solve", repetitions=1, run_index=0)
    assert directories(first) == directories(second)
    assert synthetic == before


@pytest.mark.parametrize(
    "field,value,changed",
    [
        ("iterations", 100, {"benchmark"}),
        ("warmups", 10, {"benchmark"}),
        ("repetitions", 5, {"benchmark"}),
        ("calibration_iterations", 20, {"calibrate", "solve", "smoke", "benchmark", "trace"}),
        ("calibration_warmups", 8, {"calibrate", "solve", "smoke", "benchmark", "trace"}),
        (
            "quality_outlier_mad_multiplier",
            12,
            {"calibrate", "solve", "smoke", "benchmark", "trace"},
        ),
        (
            "quality_outlier_median_multiplier",
            4,
            {"calibrate", "solve", "smoke", "benchmark", "trace"},
        ),
        (
            "quality_max_outlier_fraction",
            0.01,
            {"calibrate", "solve", "smoke", "benchmark", "trace"},
        ),
        (
            "quality_max_outlier_iteration_fraction",
            0.1,
            {"calibrate", "solve", "smoke", "benchmark", "trace"},
        ),
        (
            "quality_max_consecutive_outliers",
            2,
            {"calibrate", "solve", "smoke", "benchmark", "trace"},
        ),
        ("quality_min_outlier_samples", 30, {"calibrate", "solve", "smoke", "benchmark", "trace"}),
        ("quality_group_mad_multiplier", 4, {"calibrate", "solve", "smoke", "benchmark", "trace"}),
        (
            "quality_group_relative_deviation",
            0.2,
            {"calibrate", "solve", "smoke", "benchmark", "trace"},
        ),
        (
            "quality_group_max_discarded_fraction",
            0.3,
            {"calibrate", "solve", "smoke", "benchmark", "trace"},
        ),
        ("quality_group_min_survivors", 4, {"calibrate", "solve", "smoke", "benchmark", "trace"}),
        (
            "quality_group_consensus_median_shift",
            0.05,
            {"calibrate", "solve", "smoke", "benchmark", "trace"},
        ),
        ("solver_seconds", 600, {"solve", "smoke", "benchmark", "trace"}),
        ("profiler_active", 4, {"trace"}),
        ("trace_warmups", 9, {"trace"}),
    ],
)
def test_stage_invalidation(tmp_path, synthetic, field, value, changed):
    first = execute(tmp_path, "full")
    old = directories(first)
    second = execute(tmp_path, "full", **{field: value})
    new = directories(second)
    assert {s for s in old if old[s] != new[s]} == changed
    assert all((tmp_path / p).is_dir() for p in old.values())
    assert list((tmp_path / "receipts/history").glob("*.json"))


@pytest.mark.parametrize("systemic", [False, True])
@pytest.mark.parametrize("group_level", [False, True])
def test_raw_spikes_campaign_continuation(tmp_path, synthetic, monkeypatch, systemic, group_level):
    from megatron.core.pipeline_parallel.slackpipe.profile_quality import (
        ProfilingQualityError,
        publish_profile,
        thresholds_from_args,
    )
    from tests.unit_tests.pipeline_parallel.test_slackpipe_group_quality import group_profile
    from tests.unit_tests.pipeline_parallel.test_slackpipe_outliers import raw_profile

    original = Experiment._worker
    attempts = []

    def worker(self, stage, directory, **kwargs):
        if stage != "calibrate":
            return original(self, stage, directory, **kwargs)
        directory.mkdir(parents=True)
        attempt = kwargs["attempt"]
        attempts.append(attempt)
        spikes = [
            (4, "forward", 13, 356),
            (2, "forward", 50, 313),
            (5, "forward", 57, 306),
            (3, "forward", 98, 286),
        ]
        if systemic:
            spikes.append((4, "forward", 14, 356))
        thresholds = thresholds_from_args(self.args)
        if group_level:
            profile = group_profile(
                directory,
                [14, 16, 18, 20, 22] if systemic else [19, 19.2, 18.9, 19.1, 13.5],
                thresholds=thresholds,
            )
        else:
            profile, events = raw_profile(directory, spikes, thresholds=thresholds)
            write_json(directory / "raw.json", events)
        profile["model_config"] = dict(
            model_config_hash=fingerprint(self.model),
            dtype=self.args.precision,
            sequence_length=self.args.seq_length,
            micro_batch_size=self.args.micro_batch_size,
        )
        profile["parallel_config"] = {k: self.topology[k] for k in ("pp", "vpp", "tp", "dp", "cp")}
        if profile["schema_version"] == "slackpipe.cost_profile.v2":
            from megatron.core.pipeline_parallel.slackpipe.manifest import build_model_manifest
            from tools.slackpipe_eval_config import transformer_config

            profile["model_manifest_hash"] = build_model_manifest(
                transformer_config(self.model, self.topology, self.args.precision)
            )["manifest_hash"]
        publish_profile(
            directory / "cost_profile.json", profile, thresholds=thresholds, attempt=attempt
        )

    monkeypatch.setattr(Experiment, "_worker", worker)
    exp = Experiment(args_for(tmp_path, "benchmark"))
    if systemic:
        with pytest.warns(RuntimeWarning, match="Profiling measurements are inconsistent"):
            with pytest.raises(ProfilingQualityError, match="retry exhausted"):
                exp.execute()
        assert attempts == [0, 1]
        assert synthetic["solve"] == 0 and synthetic["slackpipe.benchmark"] == 0
    else:
        assert exp.execute()["status"] == "passed"
        assert attempts == [0]
        assert synthetic["solve"] == 1
        assert synthetic["slackpipe.benchmark"] == exp.args.repetitions
        profile = json.loads((tmp_path / exp.completed["calibrate"]["profile"]).read_text())
        if group_level:
            assert profile["quality"]["rejected_group_count"] == 1
        else:
            assert profile["quality"]["samples_discarded"] == 4


@pytest.mark.parametrize("schedule", ["1f1b", "interleaved"])
def test_native_benchmark_ignores_calibration_solver_trace(tmp_path, synthetic, schedule):
    first = execute(tmp_path, "benchmark", schedule=schedule)
    before = synthetic.copy()
    second = execute(
        tmp_path,
        "benchmark",
        schedule=schedule,
        calibration_iterations=20,
        calibration_warmups=12,
        solver_seconds=600,
        profiler_active=4,
    )
    assert directories(first) == directories(second)
    assert before == synthetic
    assert "solve" not in second.completed


@pytest.mark.parametrize(
    "field,value",
    [
        ("precision", "fp32"),
        ("seq_length", 512),
        ("pp", 2),
        ("microbatches", 16),
        ("logical_stages", 12),
        ("micro_batch_size", 2),
        ("transport", "nccl-rma"),
    ],
)
def test_real_inputs_remain_bound(tmp_path, synthetic, field, value):
    first = Experiment(args_for(tmp_path))._context("env", {})
    second = Experiment(args_for(tmp_path, **{field: value}))._context("env", {})
    assert fingerprint(first) != fingerprint(second)


def test_schedule_namespaces_and_shared_native_calibration(tmp_path, synthetic):
    inter = execute(tmp_path, "solve", schedule="interleaved")
    before = synthetic.copy()
    slack = execute(tmp_path, "solve", schedule="slackpipe")
    assert directories(inter) == directories(slack)
    assert before == synthetic
    for schedule in ("1f1b", "interleaved", "slackpipe"):
        execute(tmp_path, "benchmark", schedule=schedule, run_index=0)
        assert (tmp_path / f"receipts/{schedule}.benchmark.run000.json").exists()


def downgrade_to_v1(exp):
    """Reproduce the original global context and full-receipt parent hashes exactly."""
    a = exp.args
    old = {}
    for stage, receipt in exp.completed.items():
        context = dict(
            exp._base_context(),
            schedule=a.schedule,
            seed=a.seed,
            learning_rate=a.learning_rate,
            policy={
                k: getattr(a, k)
                for k in (
                    "warmups",
                    "iterations",
                    "calibration_iterations",
                    "repetitions",
                    "profiler_wait",
                    "profiler_warmup",
                    "profiler_active",
                    "profiler_repeat",
                )
            },
            parents={p: fingerprint(old[p]) for p in exp.dependencies(stage)},
            solver_sha256=digest(a.solver) if stage == "solve" and a.solver.is_file() else None,
            solver_seconds=a.solver_seconds if stage == "solve" else None,
            run_index=a.run_index if stage in ("benchmark", "trace") else None,
        )
        result = dict(receipt, schema_version="slackpipe.eval_receipt.v1", context=context)
        result.pop("context_hash")
        old[stage] = result
        write_json(exp.receipt_path(stage), result)
    return old


def test_legacy_expensive_artifacts_migrate_without_execution(tmp_path, synthetic):
    first = execute(tmp_path, "solve")
    old = downgrade_to_v1(first)
    before = synthetic.copy()
    hashes = {p: digest(tmp_path / p) for r in old.values() for p in r["artifacts"]}
    second = execute(
        tmp_path, "solve", repetitions=1, warmups=10, iterations=100, profiler_active=4
    )
    assert before == synthetic
    assert directories(first) == directories(second)
    for stage, receipt in second.completed.items():
        assert receipt["schema_version"] == RECEIPT_SCHEMA
        assert receipt["migration"]["original_receipt_hash"] == fingerprint(old[stage])
    assert hashes == {p: digest(tmp_path / p) for p in hashes}
    assert len(list((tmp_path / "receipts/history").glob("*.json"))) == len(old)
    execute(tmp_path, "solve", repetitions=1, warmups=10, iterations=100)
    assert before == synthetic


@pytest.mark.parametrize(
    "damage", ["artifact", "parent", "source", "warmups", "schema", "failed", "context", "profile"]
)
def test_legacy_unsafe_reuse_rejected(tmp_path, synthetic, damage):
    exp = execute(tmp_path, "solve")
    old = downgrade_to_v1(exp)
    receipt = old["calibrate"]
    if damage == "artifact":
        (tmp_path / receipt["profile"]).write_text("corrupt")
    elif damage == "parent":
        receipt["context"]["parents"]["native-smoke"] = "wrong"
    elif damage == "source":
        receipt["context"]["source"]["diff"] = "dirty"
    elif damage == "warmups":
        receipt["context"]["policy"]["warmups"] = 20
    elif damage == "schema":
        receipt["schema_version"] = "unknown"
    elif damage == "failed":
        receipt["status"] = "failed"
    elif damage == "context":
        del receipt["context"]["policy"]
    else:
        profile_path = tmp_path / receipt["profile"]
        profile = json.loads(profile_path.read_text())
        profile["model_config"]["sequence_length"] = 99
        write_json(profile_path, profile)
        receipt["artifacts"][receipt["profile"]] = digest(profile_path)
    write_json(exp.receipt_path("calibrate"), receipt)
    before = synthetic.copy()
    with pytest.raises(RuntimeError, match="stale"):
        execute(tmp_path, "solve", repetitions=1)
    assert before == synthetic


def test_legacy_source_fix_boundary(monkeypatch):
    # Exercise the audited boundary independently of newer worker changes in this checkout.
    monkeypatch.setattr(
        "tools.slackpipe_eval_receipts.subprocess.check_output", lambda *a, **k: b""
    )
    monkeypatch.setattr("tools.slackpipe_eval_receipts.subprocess.check_call", lambda *a, **k: 0)
    assert legacy_source_compatible(dict(commit=LEGACY_COMMIT, diff=EMPTY_DIFF), dict(commit="new"))
    assert not legacy_source_compatible(
        dict(commit=LEGACY_COMMIT, diff="dirty"), dict(commit="new")
    )
    assert not legacy_source_compatible(dict(commit="unknown", diff=EMPTY_DIFF), dict(commit="new"))
    monkeypatch.setattr(
        "tools.slackpipe_eval_receipts.subprocess.check_output", lambda *a, **k: b"worker changed"
    )
    assert not legacy_source_compatible(
        dict(commit=LEGACY_COMMIT, diff=EMPTY_DIFF), dict(commit="new")
    )


def test_octopipe_distinct_solver_shared_calibration(tmp_path, synthetic, monkeypatch):
    commands = []
    original = Experiment._execute

    def record(self, command, directory, name):
        commands.append((self.args.schedule, name, command))
        return original(self, command, directory, name)

    monkeypatch.setattr(Experiment, "_execute", record)
    slack = execute(tmp_path, "solve", schedule="slackpipe").completed["solve"]
    octo = execute(tmp_path, "solve", schedule="octopipe").completed["solve"]
    assert octo["context"]["solver_algorithm"] == "octopipe-algorithm1-fixed-stage"
    assert octo["context"]["octopipe_fixed_placement"] is True
    assert slack["plan"] != octo["plan"]
    assert slack["context"]["parents"]["calibrate"] == octo["context"]["parents"]["calibrate"]
    assert synthetic["octopipe.calibrate"] == 0
    command = next(c for method, name, c in commands if method == "octopipe" and name == "solve")
    assert command[command.index("--algorithm") + 1] == "octopipe-algorithm1-fixed-stage"
    assert command[command.index("--octopipe-fixed-placement") + 1] == "true"
    assert Experiment(args_for(tmp_path, schedule="octopipe")).dependencies("smoke") == ("solve",)
    before = synthetic.copy()
    execute(tmp_path, "solve", schedule="octopipe")
    assert before == synthetic


def test_interleave_alias_keeps_receipt_namespace(tmp_path, synthetic):
    first = execute(tmp_path, "benchmark", schedule="interleaved")
    before = synthetic.copy()
    alias = execute(tmp_path, "benchmark", schedule="1f1b-interleave")
    assert alias.completed["benchmark"]["directory"] == first.completed["benchmark"]["directory"]
    assert before == synthetic


def test_refined_distinct_receipts_and_effective_n(tmp_path, synthetic, monkeypatch):
    commands = []
    original = Experiment._execute

    def record(self, command, directory, name):
        if name == "solve":
            commands.append(command)
        return original(self, command, directory, name)

    monkeypatch.setattr(Experiment, "_execute", record)
    base = execute(tmp_path, "benchmark")
    refined = execute(tmp_path, "benchmark", schedule="slackpipe-refined")
    assert base.topology["num_stages"] == 8
    assert refined.topology["num_stages"] == 12
    assert refined.args.logical_stages == 8
    assert base.completed["calibrate"]["directory"] != refined.completed["calibrate"]["directory"]
    assert base.completed["solve"]["plan"] != refined.completed["solve"]["plan"]
    for cmd, n in zip(commands, (8, 12)):
        assert cmd[cmd.index("--algorithm") + 1] == "joint-unrestricted-no-overlap"
        assert cmd[cmd.index("--N") + 1] == n
    for key in ("solver_seconds", "solver_workers", "seed", "ratio", "solver_algorithm"):
        assert base.completed["solve"]["context"][key] == refined.completed["solve"]["context"][key]
    from megatron.core.pipeline_parallel.slackpipe.plan import load_slackpipe_plan

    plan = load_slackpipe_plan(
        tmp_path / refined.completed["solve"]["plan"], pipeline_model_parallel_size=4
    )
    assert plan.num_stages == 12
    assert sum(plan.layer_split) == refined.model["num_layers"]
    assert plan.stage_to_worker == tuple(s % 4 for s in range(12))
    before = synthetic.copy()
    execute(tmp_path, "benchmark", schedule="slackpipe-refined")
    assert before == synthetic


@pytest.mark.parametrize("schedule", ["slackpipe", "octopipe", "slackpipe-refined"])
def test_failed_reprofiling_prevents_campaign_solver(tmp_path, synthetic, monkeypatch, schedule):
    from megatron.core.pipeline_parallel.slackpipe.profile_quality import (
        ProfilingQualityError,
        publish_profile,
    )
    from tests.unit_tests.pipeline_parallel.test_slackpipe_profile_quality import shifted_profile

    original = Experiment._worker
    attempts = []

    def worker(self, stage, directory, **kwargs):
        if stage != "calibrate":
            return original(self, stage, directory, **kwargs)
        attempts.append(kwargs["attempt"])
        publish_profile(
            directory / "cost_profile.json", shifted_profile(), attempt=kwargs["attempt"]
        )

    monkeypatch.setattr(Experiment, "_worker", worker)
    experiment = Experiment(args_for(tmp_path, "solve", schedule=schedule))
    with (
        pytest.warns(RuntimeWarning),
        pytest.raises(ProfilingQualityError, match="retry exhausted"),
    ):
        experiment.execute()
    assert attempts == [0, 1]
    assert synthetic["solve"] == 0
    receipt = json.loads(experiment.receipt_path("calibrate").read_text())
    assert receipt["status"] == "rerun_required"
    assert any("profiling_attempts.json" in path for path in receipt["artifacts"])


def test_inspect_is_read_only_and_explains(tmp_path, synthetic):
    exp = execute(tmp_path, "solve")
    downgrade_to_v1(exp)
    original = {p: digest(p) for p in tmp_path.rglob("*") if p.is_file()}
    before = synthetic.copy()
    report = Experiment(args_for(tmp_path, "inspect", repetitions=1)).execute()
    by_stage = {r["stage"]: r for r in report["inspection"]}
    assert by_stage["solve"]["compatibility"] == "legacy-compatible"
    assert by_stage["benchmark"]["status"] == "missing"
    assert original == {p: digest(p) for p in tmp_path.rglob("*") if p.is_file()}
    assert before == synthetic


def test_worker_uses_stage_warmups(tmp_path, monkeypatch):
    exp = Experiment(
        args_for(tmp_path, warmups=30, calibration_warmups=7, smoke_warmups=2, trace_warmups=9)
    )
    commands = []
    monkeypatch.setattr(exp, "_execute", lambda cmd, *args: commands.append(cmd))
    for stage in ("calibrate", "smoke", "benchmark", "trace"):
        exp._worker(stage, tmp_path / stage)
    assert [c[c.index("--warmups") + 1] for c in commands] == [7, 2, 30, 9]
    exp.args.profiler_active = 7
    exp.args.iterations = 100
    commands.clear()
    for stage in ("calibrate", "smoke", "benchmark", "trace"):
        exp._worker(stage, tmp_path / stage)
    assert [c[c.index("--profiler-active") + 1] for c in commands] == [3, 3, 3, 7]
    assert commands[-1][commands[-1].index("--iterations") + 1] == (2 + 2 + 7) * 3


def test_real_campaign_partial_resume_three_repetitions(tmp_path, synthetic, monkeypatch):
    root = tmp_path / "llama/8b"
    upstream = execute(root, "solve", repetitions=3)
    # Complete exactly one campaign repetition, then resume the real campaign entry point.
    for schedule in COMPARISON_SCHEDULES:
        execute(root, "benchmark", schedule=schedule, run_index=0)
    before = synthetic.copy()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "campaign",
            "--families",
            "llama",
            "--sizes",
            "8b",
            "--output",
            str(tmp_path),
            "--repetitions",
            "3",
            "--resume",
            "--calibration-estimator",
            "existing-stage-wall-time-v1",
        ],
    )
    campaign.main()
    assert synthetic["solve"] == before["solve"]
    assert synthetic["slackpipe.calibrate"] == before["slackpipe.calibrate"]
    for schedule in COMPARISON_SCHEDULES:
        assert synthetic[(schedule, "benchmark", 0)] == before[(schedule, "benchmark", 0)]
        assert synthetic[(schedule, "benchmark", 1)] == 1
        assert synthetic[(schedule, "benchmark", 2)] == 1
    summaries = json.loads((tmp_path / "experiment_summary.json").read_text())
    assert [s["num_repetitions"] for s in summaries if s["stage"] == "benchmark"] == [3] * 5
    before = synthetic.copy()
    campaign.main()
    assert synthetic == before
    assert (
        execute(root, "solve").completed["solve"]["directory"]
        == upstream.completed["solve"]["directory"]
    )


def test_indexed_repetition_count_does_not_change_identity(tmp_path, synthetic):
    first = execute(tmp_path, "benchmark", run_index=0, repetitions=3)
    before = synthetic.copy()
    second = execute(tmp_path, "benchmark", run_index=0, repetitions=6)
    assert directories(first) == directories(second)
    assert synthetic == before


def test_isolated_receipt_resume_and_policy_invalidation(tmp_path, synthetic):
    estimator = "isolated-layer-compute-v1"
    first = execute(tmp_path, "solve", calibration_estimator=estimator)
    before = synthetic.copy()
    second = execute(tmp_path, "solve", calibration_estimator=estimator)
    assert directories(first) == directories(second) and synthetic == before
    third = execute(
        tmp_path, "solve", calibration_estimator=estimator, isolated_profile_iterations=40
    )
    assert third.completed["calibrate"]["directory"] != first.completed["calibrate"]["directory"]
    assert synthetic["slackpipe.calibrate"] == before["slackpipe.calibrate"] + 1


def test_force_preserves_history_and_full_overrides_resume(tmp_path, synthetic):
    first = execute(tmp_path, "full")
    selective = execute(tmp_path, "benchmark", force=True)
    assert (
        selective.completed["benchmark"]["directory"] != first.completed["benchmark"]["directory"]
    )
    assert selective.completed["solve"]["directory"] == first.completed["solve"]["directory"]
    full = execute(tmp_path, "full", force=True, resume=True)
    assert all(full.completed[s]["directory"] != r["directory"] for s, r in first.completed.items())
    assert all((tmp_path / r["directory"]).exists() for r in first.completed.values())


def test_solver_binary_hash_invalidates_only_solve_downstream(tmp_path, synthetic):
    binary = tmp_path / "solver-binary"
    binary.write_text("build1")
    first = execute(tmp_path / "experiment", "full", solver=binary)
    binary.write_text("build2")
    second = execute(tmp_path / "experiment", "full", solver=binary)
    assert {
        s
        for s in first.completed
        if first.completed[s]["directory"] != second.completed[s]["directory"]
    } == {"solve", "smoke", "benchmark", "trace"}


def test_legacy_across_receipt_only_source_fix(tmp_path, synthetic, monkeypatch):
    monkeypatch.setattr(
        "tools.slackpipe_eval_receipts.subprocess.check_output", lambda *a, **k: b""
    )
    monkeypatch.setattr("tools.slackpipe_eval_receipts.subprocess.check_call", lambda *a, **k: 0)
    base = Experiment._base_context

    def identity(self):
        context = base(self)
        context["source"] = dict(commit=LEGACY_COMMIT, diff=EMPTY_DIFF)
        context["environment"]["source"] = dict(
            commit=LEGACY_COMMIT, diff_sha256=EMPTY_DIFF, untracked_source_sha256={}
        )
        return context

    monkeypatch.setattr(Experiment, "_base_context", identity)
    first = execute(tmp_path, "solve")
    old = downgrade_to_v1(first)

    def new_identity(self):
        context = identity(self)
        context["source"] = dict(commit="receipt-only-fix", diff=EMPTY_DIFF)
        context["environment"]["source"]["commit"] = "receipt-only-fix"
        return context

    monkeypatch.setattr(Experiment, "_base_context", new_identity)
    before = synthetic.copy()
    migrated = execute(tmp_path, "solve", repetitions=1)
    assert directories(first) == directories(migrated)
    assert before == synthetic
    unknown = dict(old["env"], context=dict(old["env"]["context"], unrecognized_setting=True))
    state, reason, _ = assess_receipt(unknown, migrated._context("env", {}), tmp_path, {})
    assert state == "legacy-incompatible" and "unknown/missing legacy context fields" in reason


def test_fresh_campaign_shares_only_compatible_upstream_work(tmp_path, synthetic, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "campaign",
            "--families",
            "llama",
            "--sizes",
            "8b",
            "--output",
            str(tmp_path),
            "--calibration-estimator",
            "existing-stage-wall-time-v1",
        ],
    )
    campaign.main()
    assert synthetic["correctness"] == 3  # Physical, base VPP, and refined VPP topologies.
    assert sum(synthetic[f"{s}.calibrate"] for s in COMPARISON_SCHEDULES) == 2
    assert sum(synthetic[f"{s}.smoke"] for s in COMPARISON_SCHEDULES) == 7
    assert synthetic["solve"] == 3  # Never share different algorithms or effective N.
    for schedule in COMPARISON_SCHEDULES:
        assert synthetic[f"{schedule}.benchmark"] == 3
        assert synthetic[f"{schedule}.trace"] == 1
    root = tmp_path / "llama/8b/receipts"
    calibration = {
        s: json.loads((root / f"{s}.calibrate.json").read_text())
        for s in ("octopipe", "slackpipe", "slackpipe-refined")
    }
    assert calibration["octopipe"]["profile"] == calibration["slackpipe"]["profile"]
    assert calibration["slackpipe-refined"]["profile"] != calibration["slackpipe"]["profile"]
    before = synthetic.copy()
    with pytest.raises(RuntimeError, match="Existing/stale receipt"):
        campaign.main()  # Fresh sharing is not implicit resume of a previous campaign.
    assert synthetic == before


def test_campaign_sharing_checks_artifact_integrity(tmp_path, synthetic):
    first = execute(tmp_path, "solve", schedule="octopipe")
    shared = [first.receipt_path(s) for s in first.completed]
    calibration = first.completed["calibrate"]
    (tmp_path / calibration["profile"]).write_text("corrupted")
    second = Experiment(args_for(tmp_path, resume=False), shared_receipts=shared)
    second.execute()
    assert synthetic["slackpipe.calibrate"] == 1
    assert second.completed["calibrate"]["directory"] != calibration["directory"]


def test_campaign_sharing_does_not_adopt_unlisted_receipts(tmp_path, synthetic):
    first = execute(tmp_path, "solve", schedule="octopipe")
    second = Experiment(
        args_for(tmp_path, resume=False), shared_receipts=[first.receipt_path("env")]
    )
    second.execute()
    assert synthetic["correctness"] == 2
    assert synthetic["slackpipe.calibrate"] == 1
    assert second.completed["env"]["directory"] == first.completed["env"]["directory"]


@pytest.mark.parametrize("duplicate", ["family", "size", "symlink", "model", "output", "alias"])
def test_campaign_rejects_duplicates_before_launch(tmp_path, synthetic, monkeypatch, duplicate):
    configs = tmp_path / "configs"
    configs.mkdir()
    model = json.loads(CONFIG.read_text())
    write_json(configs / "llama_8b.json", model)
    families, sizes, schedules = "llama", "8b", "slackpipe"
    if duplicate == "family":
        families = "llama, llama"
    elif duplicate == "size":
        sizes = "8b,8b"
    elif duplicate in ("symlink", "model", "output"):
        sizes = "8b,copy"
        other = configs / "llama_copy.json"
        if duplicate == "symlink":
            other.symlink_to(configs / "llama_8b.json")
        else:
            if duplicate == "output":
                model["name"] = "different-model-same-output"
            write_json(other, model)
    else:
        schedules = "interleaved,1f1b-interleave"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "campaign",
            "--configs",
            str(configs),
            "--output",
            str(tmp_path / "results"),
            "--families",
            families,
            "--sizes",
            sizes,
            "--schedules",
            schedules,
        ],
    )
    with pytest.raises(SystemExit) as exc:
        campaign.main()
    assert exc.value.code == 2
    assert not synthetic
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize(
    "schedule,n", [("slackpipe", 8), ("slackpipe-refined", 12), ("interleaved", 8), ("octopipe", 8)]
)
def test_worker_effective_n_is_applied_once(tmp_path, monkeypatch, schedule, n):
    exp = Experiment(args_for(tmp_path, schedule=schedule))
    commands = []
    monkeypatch.setattr(exp, "_execute", lambda cmd, *rest: commands.append(cmd))
    exp._worker("calibrate", tmp_path / "calibration", schedule="interleaved")
    exp._worker("smoke", tmp_path / "execution", schedule=schedule)
    assert commands[0][commands[0].index("--logical-stages") + 1] == n
    assert commands[1][commands[1].index("--logical-stages") + 1] == 8
    assert Experiment(exp.args).topology["num_stages"] == n


def test_v2_integrity_and_detailed_diagnostics(tmp_path, synthetic):
    exp = execute(tmp_path, "benchmark")
    receipt = exp.completed["benchmark"]
    new = Experiment(args_for(tmp_path, "benchmark", iterations=100))
    expected = new._context("benchmark", {"smoke": exp.completed["smoke"]})
    state, reason, _ = assess_receipt(receipt, expected, tmp_path, {})
    assert state == "stale" and "policy.iterations" in reason
    damaged = dict(receipt, context_hash="wrong")
    assert assess_receipt(damaged, expected, tmp_path, {})[0] == "invalid"
    damaged = dict(receipt, context=dict(receipt["context"], context_schema_version=99))
    damaged["context_hash"] = fingerprint(damaged["context"])
    assert assess_receipt(damaged, expected, tmp_path, {})[0] == "invalid"
