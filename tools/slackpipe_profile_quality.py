# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Calibration attempt orchestration and standalone solver quality gate."""

import argparse
import json
import shutil
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path

from megatron.core.pipeline_parallel.slackpipe.isolated_profile import (
    ESTIMATOR,
    failure_summary,
    quality_report,
)
from megatron.core.pipeline_parallel.slackpipe.isolated_profile import (
    require_quality as require_isolated_quality,
)
from megatron.core.pipeline_parallel.slackpipe.profile_quality import (
    MESSAGE,
    ProfilingQualityError,
    require_profile_quality,
)


def _isolated_attempt(candidate: dict, report: dict, configuration: dict, worker: Path):
    """Validate isolated data directly, not through the pipeline quality evaluator."""
    evaluated = quality_report(candidate, raw_data_path=str(worker / "isolated_profile_raw.json"))
    for key in ("quality_schema_version", "thresholds", "status", "issues"):
        if report.get(key) != evaluated[key]:
            raise ProfilingQualityError(f"Isolated quality receipt differs from raw data: {key}")
    policy = configuration.get("policy", {})
    context = candidate["isolated"]["context"]
    for key, value in (
        ("model_manifest_hash", candidate["model_manifest_hash"]),
        ("warmups", context["warmups"]),
        ("iterations", context["iterations"]),
        ("quality", context["policy"]),
        ("execution", context["execution"]),
        ("statistic", context.get("statistic", "median")),
        ("quality_schema", evaluated["quality_schema_version"]),
        ("class_signatures", [c["class_signature"] for c in candidate["isolated"]["classes"]]),
    ):
        if key in policy and policy[key] != value:
            raise ProfilingQualityError(f"Isolated receipt/collector {key} mismatch")
    identity = [ESTIMATOR, json.dumps(context, sort_keys=True)]
    return evaluated, identity


def collect_with_retry(directory: Path, collect, configuration: dict) -> Path:
    """Estimator-neutral retry runner; quality semantics are dispatched explicitly."""
    attempts = []
    reference = None
    summary = dict(
        status="collecting",
        configuration=configuration,
        attempts=attempts,
        selected_profile=None,
        automatic_retry_limit=1,
    )
    directory.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        worker = directory / f"attempt{attempt}" / "worker"
        worker.parent.mkdir(parents=True, exist_ok=False)
        entry = dict(
            attempt=attempt,
            started_at=datetime.now(timezone.utc).isoformat(),
            directory=str(worker),
        )
        attempts.append(entry)
        try:
            collect(worker, attempt)
        except Exception:
            receipt = worker / "cost_profile.quality.json"
            if (
                not receipt.is_file()
                or json.loads(receipt.read_text()).get("status") != "rerun_required"
            ):
                entry.update(
                    status="collection_failed", ended_at=datetime.now(timezone.utc).isoformat()
                )
                summary["status"] = "collection_failed"
                (directory / "profiling_attempts.json").write_text(
                    json.dumps(summary, indent=2) + "\n"
                )
                raise
        report = json.loads((worker / "cost_profile.quality.json").read_text())
        candidate = json.loads((worker / "cost_profile.candidate.json").read_text())
        expected_estimator = configuration.get("policy", {}).get("estimator")
        isolated = candidate.get("estimator") == ESTIMATOR
        if expected_estimator == ESTIMATOR and not isolated:
            raise ProfilingQualityError("Collector returned a different calibration estimator")
        if isolated:
            report, identity = _isolated_attempt(candidate, report, configuration, worker)
        else:
            identity = sorted(
                json.dumps(
                    {
                        k: row.get(k)
                        for k in (
                            "calibration_group_id",
                            "stage",
                            "worker",
                            "stage_layer_range",
                            "measurement_context",
                        )
                    },
                    sort_keys=True,
                )
                for row in candidate.get("observed_stages", [])
            )
        entry.update(
            ended_at=datetime.now(timezone.utc).isoformat(), quality=report, status=report["status"]
        )
        identity.append(json.dumps(report["thresholds"], sort_keys=True))
        if reference is None:
            reference = identity
        elif identity != reference:
            entry.update(
                status="configuration_changed", ended_at=datetime.now(timezone.utc).isoformat()
            )
            summary["status"] = "rerun_required"
            (directory / "profiling_attempts.json").write_text(json.dumps(summary, indent=2) + "\n")
            raise ProfilingQualityError(
                "Retry changed calibration partitions/configuration/hardware; no profile selected"
            )
        profile = worker / "cost_profile.json"
        if report["status"] == "passed":
            selected = json.loads(profile.read_text())
            if isolated:
                require_isolated_quality(selected)
                if selected["cost_profile_hash"] != candidate["cost_profile_hash"]:
                    raise ProfilingQualityError("Selected isolated profile differs from candidate")
            else:
                require_profile_quality(selected)
            summary.update(status="passed", selected_profile=str(profile))
        else:
            summary["status"] = "rerun_required"
        (directory / "profiling_attempts.json").write_text(json.dumps(summary, indent=2) + "\n")
        if summary["status"] == "passed":
            return profile
        if isolated:
            warnings.warn(f"{MESSAGE} attempt={attempt} {failure_summary(report)}", RuntimeWarning)
    details = " ".join(
        f"attempt={entry['attempt']} {failure_summary(entry['quality'])}"
        for entry in attempts
        if entry["quality"].get("estimator") == ESTIMATOR
    )
    raise ProfilingQualityError(
        f"{MESSAGE} Automatic retry exhausted. {details} See {directory / 'profiling_attempts.json'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--collect-output", type=Path)
    parser.add_argument("--selected-profile", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.collect_output:
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            parser.error("--collect-output requires a profiling command after --")

        def collect(worker, attempt):
            current = list(command)
            for flag in ("--output", "--output-dir"):
                if flag in current:
                    current[current.index(flag) + 1] = str(worker)
                    break
            else:
                raise ValueError("Profiling command needs --output or --output-dir")
            if "--emit-cost-profile" in current:
                current[current.index("--emit-cost-profile") + 1] = str(
                    worker / "cost_profile.json"
                )
            current += ["--profiling-attempt", str(attempt)]
            subprocess.run(current, check=True)

        selected = collect_with_retry(args.collect_output, collect, dict(command=command))
        for name in (
            "cost_profile.json",
            "calibration_events.json",
            "observations.json",
            "model_manifest.json",
            "model.json",
            "isolated_profile_raw.json",
            "isolated_profile_summary.json",
        ):
            source = selected.parent / name
            target = args.collect_output / name
            if source.is_file():
                if target.exists():
                    raise FileExistsError(target)
                shutil.copyfile(source, target)
        if args.selected_profile:
            if args.selected_profile.exists():
                raise FileExistsError(args.selected_profile)
            args.selected_profile.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(selected, args.selected_profile)
        return
    if not args.profile:
        parser.error("--profile or --collect-output is required")
    require_profile_quality(json.loads(args.profile.read_text()))
    print("Profiling quality: passed")


if __name__ == "__main__":
    main()
