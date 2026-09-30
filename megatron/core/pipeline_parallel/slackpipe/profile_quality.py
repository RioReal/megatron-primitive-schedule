# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Empirical calibration checks with auditable raw-sample spike isolation."""

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
QUALITY_SCHEMA = "slackpipe.profile_quality.v2"
OUTLIER_POLICY_VERSION = 1


class ProfilingQualityError(RuntimeError):
    """A calibration is not authorized for automatic solver consumption."""


@dataclass(frozen=True)
class QualityThresholds:
    """Heuristic limits, retaining the existing 10% CV review convention."""

    cv: float = 0.10
    median_shift: float = 0.30
    relative_rmse: float = 0.15
    min_samples: int = 3
    outlier_mad_multiplier: float = 10.0
    outlier_median_multiplier: float = 3.0
    max_outlier_fraction: float = 0.02
    max_outlier_iteration_fraction: float = 0.20
    max_consecutive_outliers: int = 1
    min_outlier_samples: int = 20

    def __post_init__(self):
        if any(
            not math.isfinite(v) or v <= 0 for v in (self.cv, self.median_shift, self.relative_rmse)
        ):
            raise ValueError("Quality thresholds must be positive finite numbers")
        if not isinstance(self.min_samples, int) or self.min_samples < 3:
            raise ValueError("Quality checks require at least three measured iterations")
        if (
            not math.isfinite(self.outlier_mad_multiplier)
            or self.outlier_mad_multiplier <= 0
            or not math.isfinite(self.outlier_median_multiplier)
            or self.outlier_median_multiplier <= 1
        ):
            raise ValueError("Outlier multipliers must be finite, MAD > 0 and median > 1")
        for value in (self.max_outlier_fraction, self.max_outlier_iteration_fraction):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Outlier fractions must be finite and in [0, 1]")
        if type(self.max_consecutive_outliers) is not int or self.max_consecutive_outliers < 1:
            raise ValueError("max_consecutive_outliers must be a positive integer")
        if type(self.min_outlier_samples) is not int or self.min_outlier_samples < 3:
            raise ValueError("min_outlier_samples must be at least three")


def outlier_policy(thresholds: QualityThresholds) -> dict:
    """Only aggregation-affecting policy, separate from CV/fit acceptance limits."""
    return dict(
        policy_version=OUTLIER_POLICY_VERSION,
        **{k: v for k, v in asdict(thresholds).items() if "outlier" in k},
    )


def isolate_timing_spikes(events: list, thresholds: QualityThresholds) -> tuple:
    """Return accepted events and diagnostics without mutating any raw record.

    Caller supplies one group/context/stage/phase. Strict upper-tail MAD AND
    median-ratio tests identify candidates; frequency/run-length gates below
    decide whether excluding them is tolerable, not a license to clean a bad run.
    """
    ordered = sorted(events, key=lambda e: (int(e["iteration"]), int(e.get("microbatch", 0))))
    values = [float(e["elapsed_ms"]) for e in ordered]
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Calibration timings must be finite and positive")
    median = statistics.median(values)
    mad = statistics.median(abs(v - median) for v in values)
    sigma = 1.4826 * mad
    mad_threshold = median + thresholds.outlier_mad_multiplier * sigma
    ratio_threshold = thresholds.outlier_median_multiplier * median
    rejected, accepted, affected = [], [], set()
    run = longest = 0
    for event, value in zip(ordered, values):
        bad = (
            len(values) >= thresholds.min_outlier_samples
            and value > mad_threshold
            and value > ratio_threshold
        )
        if bad:
            run += 1
            longest = max(longest, run)
            affected.add(int(event["iteration"]))
            rejected.append(
                dict(
                    **{
                        k: event.get(k)
                        for k in (
                            "calibration_group_id",
                            "attempt",
                            "stage_layer_range",
                            "stage_role",
                            "measurement_context",
                            "iteration",
                            "microbatch",
                            "raw_data_path",
                        )
                    },
                    stage=int(event["logical_stage"]),
                    phase=event["phase"].removesuffix("_compute"),
                    worker=event.get("worker", event.get("pp_rank", event.get("rank"))),
                    rank=event.get("rank", event.get("worker")),
                    elapsed_ms=value,
                    median=median,
                    MAD=mad,
                    robust_sigma=sigma,
                    mad_threshold_ms=mad_threshold,
                    ratio_threshold_ms=ratio_threshold,
                    threshold_ms=max(mad_threshold, ratio_threshold),
                    rejection_reason="upper_tail_exceeds_both_MAD_and_median_ratio",
                    discarded=True,
                )
            )
        else:
            run = 0
            accepted.append(event)
    iterations = {int(e["iteration"]) for e in ordered}
    return accepted, dict(
        policy=outlier_policy(thresholds),
        detection_status=(
            "assessed"
            if len(values) >= thresholds.min_outlier_samples
            else "insufficient_raw_samples_no_filtering"
        ),
        median=median,
        MAD=mad,
        robust_sigma=sigma,
        raw_sample_count=len(values),
        accepted_sample_count=len(accepted),
        samples_discarded=len(rejected),
        discarded_fraction=len(rejected) / len(values),
        affected_iterations=sorted(affected),
        affected_iteration_fraction=len(affected) / len(iterations),
        max_consecutive_outliers=longest,
        rejected_samples=rejected,
    )


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
    metrics, issues, sample_groups = [], [], []

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
            diagnostics = row.get(f"{phase}_diagnostics", {})
            filtering = diagnostics.get("outlier_filter")
            if filtering is None and profile.get("measurement_definition", {}).get("filtering"):
                record(
                    "missing_outlier_analysis",
                    None,
                    "raw-sample analysis required",
                    [row],
                    phase,
                    True,
                )
            if filtering is not None:
                record(
                    "outlier_policy_match",
                    filtering.get("policy"),
                    outlier_policy(thresholds),
                    [row],
                    phase,
                    filtering.get("policy") != outlier_policy(thresholds),
                )
                for field, limit in (
                    ("discarded_fraction", thresholds.max_outlier_fraction),
                    ("affected_iteration_fraction", thresholds.max_outlier_iteration_fraction),
                    ("max_consecutive_outliers", thresholds.max_consecutive_outliers),
                ):
                    value = filtering.get(field)
                    record(
                        field,
                        value,
                        limit,
                        [row],
                        phase,
                        not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value > limit,
                    )
                sample_groups.append(
                    dict(
                        **{
                            k: row.get(k)
                            for k in (
                                "stage",
                                "calibration_group_id",
                                "worker",
                                "stage_layer_range",
                                "raw_data_path",
                            )
                        },
                        phase=phase,
                        **filtering,
                        raw_cv=diagnostics.get("raw_cv"),
                        cleaned_cv=diagnostics.get("cv"),
                    )
                )
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
        schema_version=QUALITY_SCHEMA,
        status="rerun_required" if issues else "passed",
        thresholds=asdict(thresholds),
        metrics=metrics,
        issues=issues,
        raw_sample_count=sum(g["raw_sample_count"] for g in sample_groups),
        accepted_sample_count=sum(g["accepted_sample_count"] for g in sample_groups),
        samples_discarded=sum(g["samples_discarded"] for g in sample_groups),
        discarded_fraction=(
            (
                sum(g["samples_discarded"] for g in sample_groups)
                / sum(g["raw_sample_count"] for g in sample_groups)
            )
            if sample_groups
            else 0.0
        ),
        sample_groups=sample_groups,
        rejected_samples=[s for g in sample_groups for s in g["rejected_samples"]],
        raw_sample_status=(
            "available"
            if len(sample_groups) == 2 * len(rows) and rows
            else "unavailable_or_partial_legacy_aggregation"
        ),
        cv_definition="population stddev/mean of iteration per-operation means (B-normalized totals); raw_cv before filtering, cleaned_cv after filtering",
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
    if quality.get("status") != "passed" or quality.get("quality_schema_version") not in (
        "slackpipe.profile_quality.v1",
        QUALITY_SCHEMA,
    ):
        raise ProfilingQualityError(
            f"{MESSAGE} Rerun: {quality.get('rerun_command', 'rerun the original calibration command with a fresh output directory (legacy profile has no command provenance)')}"
        )
    report = assess_profile(profile, QualityThresholds(**quality["thresholds"]))
    if report["status"] != "passed":
        raise ProfilingQualityError(f"{MESSAGE} {json.dumps(report['issues'])}")
