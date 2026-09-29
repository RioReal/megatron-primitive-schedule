# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Empirical, fail-closed calibration checks; never filter measured samples."""

import hashlib
import json
import math
import os
import shlex
import statistics
import sys
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

MESSAGE = "Profiling measurements are inconsistent. Rerun profiling before using this cost model."


class ProfilingQualityError(RuntimeError):
    """A calibration is not authorized for automatic solver consumption."""


@dataclass(frozen=True)
class QualityThresholds:
    """Heuristic limits, retaining the existing 10% CV review convention."""

    cv: float = 0.10
    median_shift: float = 0.30
    relative_rmse: float = 0.15
    min_samples: int = 3

    def __post_init__(self):
        if any(
            not math.isfinite(v) or v <= 0 for v in (self.cv, self.median_shift, self.relative_rmse)
        ):
            raise ValueError("Quality thresholds must be positive finite numbers")
        if not isinstance(self.min_samples, int) or self.min_samples < 3:
            raise ValueError("Quality checks require at least three measured iterations")


def add_quality_arguments(parser) -> None:
    """Add shared calibration-quality CLI flags."""
    for name, value in asdict(QualityThresholds()).items():
        parser.add_argument(f"--quality-{name.replace('_', '-')}", type=type(value), default=value)
    parser.add_argument("--profiling-attempt", type=int, choices=(0, 1), default=0)


def thresholds_from_args(args) -> QualityThresholds:
    """Read common flags, including callers predating these arguments."""
    return QualityThresholds(
        **{k: getattr(args, f"quality_{k}", v) for k, v in asdict(QualityThresholds()).items()}
    )


def tag_calibration_events(
    events: list,
    *,
    group: str,
    ranges: list,
    worker: int,
    configuration: dict,
    raw_path: Path,
    attempt: int = 0,
) -> None:
    """Attach explicit collection provenance before gathering/persisting events."""
    context = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, default=str).encode()
    ).hexdigest()
    for event in events:
        stage = int(event["logical_stage"])
        event.update(
            calibration_group_id=str(group),
            attempt=attempt,
            worker=worker,
            stage_layer_range=list(ranges[stage]),
            stage_role=(
                "first" if stage == 0 else ("last" if stage == len(ranges) - 1 else "middle")
            ),
            measurement_context=context,
            raw_data_path=str(raw_path.resolve()),
        )


def assess_profile(profile: dict, thresholds: QualityThresholds = QualityThresholds()) -> dict:
    """Check iteration-mean CV, comparable group medians, and model residuals.

    CV is population stddev / mean of iteration compute totals (equivalently
    per-operation iteration means for fixed B). Cross-group shift is max median /
    min median - 1, requiring two groups with >= min_samples iterations each.
    Relative RMSE is sqrt(mean((prediction-observation)^2)) / mean(observation),
    using all stage/group rows and at least two rows per phase. Missing provenance,
    inadequate repeated samples and nonfinite/nonpositive timings fail closed.
    """
    rows = profile.get("observed_stages", [])
    metrics, issues = [], []

    def record(rule, value, threshold, related, phase, failed):
        item = dict(
            rule=rule,
            value=value,
            threshold=threshold,
            phase=phase,
            measurements=[
                {
                    k: row.get(k)
                    for k in (
                        "calibration_group_id",
                        "stage",
                        "stage_layer_range",
                        "stage_role",
                        "worker",
                        "measurement_context",
                        "raw_data_path",
                        "forward_ms_per_op",
                        "backward_ms_per_op",
                    )
                }
                for row in related
            ],
        )
        metrics.append(item)
        if failed:
            issues.append(item)

    if not rows:
        record("missing_measurements", 0, 1, [], "both", True)
    for phase in ("forward", "backward"):
        comparable = defaultdict(list)
        observed, predicted = [], []
        for row in rows:
            provenance = all(
                row.get(k) is not None
                for k in (
                    "calibration_group_id",
                    "worker",
                    "stage_layer_range",
                    "stage_role",
                    "measurement_context",
                    "raw_data_path",
                )
            )
            samples = row.get(f"{phase}_diagnostics", {}).get("samples_ms", [])
            valid = len(samples) >= thresholds.min_samples and all(
                isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in samples
            )
            if not provenance or not valid:
                record(
                    "missing_provenance_or_samples",
                    len(samples),
                    thresholds.min_samples,
                    [row],
                    phase,
                    True,
                )
                continue
            cv = statistics.pstdev(samples) / statistics.fmean(samples)
            record("within_group_cv", cv, thresholds.cv, [row], phase, cv > thresholds.cv)
            value = row.get(f"{phase}_ms_per_op")
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                record("invalid_operation_cost", None, None, [row], phase, True)
                continue
            key = (
                tuple(row["stage_layer_range"]),
                row["stage_role"],
                row["worker"],
                row["measurement_context"],
            )
            comparable[key].append(row)
            if profile.get("schema_version") == "slackpipe.cost_profile.v2":
                begin, end = row["stage_layer_range"]
                prefix = profile[f"prefix_{phase}_us"]
                prediction = (
                    prefix[end]
                    - prefix[begin]
                    + profile["stage_role_bias_us"][row["stage_role"]][phase]
                ) / 1000
            else:
                suffix = "fwd" if phase == "forward" else "bwd"
                prediction = (
                    profile[f"a_{suffix}"] * row["layer_count"]
                    + profile[f"bias_{suffix}"][row["stage"]]
                )
            observed.append(value)
            predicted.append(prediction)
        for related in comparable.values():
            groups = defaultdict(list)
            for row in related:
                groups[row["calibration_group_id"]].append(row[f"{phase}_ms_per_op"])
            if len(groups) < 2:
                continue
            medians = [statistics.median(values) for values in groups.values()]
            shift = max(medians) / min(medians) - 1
            record(
                "cross_group_median_shift",
                shift,
                thresholds.median_shift,
                related,
                phase,
                shift > thresholds.median_shift,
            )
        if len(observed) < 2:
            record("insufficient_fit_observations", len(observed), 2, rows, phase, True)
        else:
            rmse = math.sqrt(
                statistics.fmean((a - b) ** 2 for a, b in zip(observed, predicted))
            ) / statistics.fmean(observed)
            record(
                "fit_relative_rmse",
                rmse if math.isfinite(rmse) else None,
                thresholds.relative_rmse,
                rows,
                phase,
                not math.isfinite(rmse) or rmse > thresholds.relative_rmse,
            )
    return dict(
        schema_version="slackpipe.profile_quality.v1",
        status="rerun_required" if issues else "passed",
        thresholds=asdict(thresholds),
        metrics=metrics,
        issues=issues,
        samples_discarded=0,
        cross_group_status=(
            "assessed"
            if any(m["rule"] == "cross_group_median_shift" for m in metrics)
            else "not_assessed_no_comparable_groups"
        ),
        cross_group_comparisons=sum(m["rule"] == "cross_group_median_shift" for m in metrics),
    )


def rerun_command() -> str:
    """Reconstruct a fresh-output worker invocation, including torchrun ranks."""
    argv = list(sys.argv)
    suffix = ".rerun-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    for flag in ("--output", "--output-dir", "--emit-cost-profile"):
        if flag in argv:
            index = argv.index(flag) + 1
            argv[index] += suffix
    launch = [sys.executable]
    if "WORLD_SIZE" in os.environ:
        launch += [
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node",
            os.environ["WORLD_SIZE"],
        ]
    environment = [
        f"{key}={os.environ[key]}"
        for key in (
            "PYTHONPATH",
            "OMP_NUM_THREADS",
            "CUDA_VISIBLE_DEVICES",
            "CUDA_DEVICE_MAX_CONNECTIONS",
            "PYTORCH_CUDA_ALLOC_CONF",
            "PYTORCH_ALLOC_CONF",
            "PYTORCH_NO_CUDA_MEMORY_CACHING",
            "CUDA_MODULE_LOADING",
            "MAMBA_DETERMINISTIC",
            "TRITON_CACHE_AUTOTUNING",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
            "CUBLAS_WORKSPACE_CONFIG",
        )
        if key in os.environ
    ]
    return f"cd {shlex.quote(os.getcwd())} && " + shlex.join(["env", *environment, *launch, *argv])


def publish_profile(
    path: Path,
    profile: dict,
    *,
    thresholds: QualityThresholds = QualityThresholds(),
    attempt: int = 0,
    command: str | None = None,
) -> dict:
    """Keep the candidate/diagnostics; publish a default profile only on pass."""
    from .cost_profile import profile_fingerprint, write_cost_profile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if any(
        p.exists()
        for p in (path, path.with_suffix(".candidate.json"), path.with_suffix(".quality.json"))
    ):
        raise FileExistsError(f"Refusing to replace profile {path}; use a fresh output directory")
    report = assess_profile(profile, thresholds)
    report.update(
        attempt=attempt,
        timestamp=datetime.now(timezone.utc).isoformat(),
        rerun_command=command or rerun_command(),
        selected_profile=str(path.resolve()) if report["status"] == "passed" else None,
    )
    # The existing C++ reader searches for the first schema_version string.
    # Keep that key unique in a profile; the standalone receipt still uses it.
    profile["quality"] = {k: v for k, v in report.items() if k != "schema_version"}
    profile["quality"]["quality_schema_version"] = report["schema_version"]
    profile["cost_profile_hash"] = profile_fingerprint(profile)
    write_cost_profile(path.with_suffix(".candidate.json"), profile)
    path.with_suffix(".quality.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] == "passed":
        write_cost_profile(path, profile)
    else:
        for issue in report["issues"]:
            warnings.warn(
                f"{MESSAGE} {json.dumps(issue)} Rerun: {report['rerun_command']}", RuntimeWarning
            )
    return report


def require_profile_quality(profile: dict) -> None:
    """Reject missing/failed quality receipts, including reused legacy profiles."""
    quality = profile.get("quality", {})
    if quality.get("status") != "passed":
        raise ProfilingQualityError(
            f"{MESSAGE} Rerun: {quality.get('rerun_command', 'rerun the original calibration command with a fresh output directory (legacy profile has no command provenance)')}"
        )
    report = assess_profile(profile, QualityThresholds(**quality["thresholds"]))
    if report["status"] != "passed":
        raise ProfilingQualityError(f"{MESSAGE} {json.dumps(report['issues'])}")
