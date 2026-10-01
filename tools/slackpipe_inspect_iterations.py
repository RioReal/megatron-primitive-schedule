# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Read-only raw-event replay for failed within-group calibration CV checks.

This is evidence collection, not an iteration filter or a solver authorization.
Run with ``python -m tools.slackpipe_inspect_iterations --help``.
"""

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from megatron.core.pipeline_parallel.slackpipe.profile_quality import (
    QualityThresholds,
    assess_profile,
    isolate_timing_spikes,
)

IDENTITY_FIELDS = (
    "calibration_group_id",
    "attempt",
    "worker",
    "stage_layer_range",
    "stage_role",
    "measurement_context",
    "raw_data_path",
)


def _key(record: dict, *, event: bool = False) -> str:
    identity = {key: record.get(key) for key in IDENTITY_FIELDS}
    if any(value is None for value in identity.values()):
        raise ValueError(
            "Raw replay requires explicit group/attempt/range/worker/context provenance"
        )
    identity["stage"] = record["logical_stage" if event else "stage"]
    return json.dumps(identity, sort_keys=True)


def _cv(values: list) -> float | None:
    return (
        statistics.pstdev(values) / statistics.fmean(values) if values and min(values) > 0 else None
    )


def _check_values(actual: list, expected: list, name: str) -> None:
    if len(actual) != len(expected) or any(
        not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(actual, expected)
    ):
        raise ValueError(f"Raw replay disagrees with saved {name}")


def inspect_iterations(profile: dict, quality: dict, raw: list) -> dict:
    """Replay every failing CV from tagged events without changing any input.

    Accept either rank-local event lists or the collector's list of partition
    bundles. Group identity always comes from events, never bundle order. Require
    exact microbatch coverage and agreement with the saved sample-level analysis.
    """
    thresholds = QualityThresholds(**quality["thresholds"])
    replay = assess_profile(profile, thresholds=thresholds)
    if replay["issues"] != quality["issues"] or replay["status"] != quality["status"]:
        raise ValueError("Quality report does not match this profile under its recorded policy")
    events = []
    for item in raw:
        events.extend(item["events"] if "events" in item else [item])
    buckets = defaultdict(list)
    for event in events:
        if event.get("phase") in ("forward_compute", "backward_compute"):
            buckets[(_key(event, event=True), event["phase"].removesuffix("_compute"))].append(
                event
            )
    rows = defaultdict(list)
    for row in profile["observed_stages"]:
        rows[(row["calibration_group_id"], row["stage"])].append(row)

    failures = []
    for issue in quality["issues"]:
        if issue["rule"] != "within_group_cv":
            continue
        phase = issue["phase"]
        for measurement in issue["measurements"]:
            matches = [
                row
                for row in rows[(measurement["calibration_group_id"], measurement["stage"])]
                if all(row.get(k) == v for k, v in measurement.items())
            ]
            if len(matches) != 1:
                raise ValueError("CV issue must resolve to exactly one profile observation")
            row = matches[0]
            diagnostic = row[f"{phase}_diagnostics"]
            iterations = diagnostic["global_iterations"]
            if not iterations or iterations != sorted(set(iterations)):
                raise ValueError("Expected unique ordered global iteration IDs")
            raw_events = buckets[(_key(row), phase)]
            selected = [e for e in raw_events if e["iteration"] in iterations]
            if not selected:
                raise ValueError("Missing raw events for failed stage/phase/context")
            accepted, filtering = isolate_timing_spikes(selected, thresholds)
            if filtering != diagnostic["outlier_filter"]:
                raise ValueError("Raw replay disagrees with saved sample filter")
            count = diagnostic["raw_sample_count"]
            if count % len(iterations):
                raise ValueError("Raw count is not a complete microbatch grid")
            microbatches = count // len(iterations)
            by_iteration, clean = defaultdict(list), defaultdict(list)
            for event in selected:
                by_iteration[event["iteration"]].append(event)
            for event in accepted:
                clean[event["iteration"]].append(float(event["elapsed_ms"]))
            details = []
            for iteration in iterations:
                group = by_iteration[iteration]
                ids = [e.get("microbatch") for e in group]
                if len(ids) != microbatches or set(ids) != set(range(microbatches)):
                    raise ValueError("Missing or duplicate microbatch operation in raw events")
                values = clean[iteration]
                mean = statistics.fmean(values) if values else None
                details.append(
                    dict(
                        iteration=iteration,
                        ranks=sorted({e["rank"] for e in group if "rank" in e}),
                        raw_microbatch_count=len(group),
                        accepted_microbatch_count=len(values),
                        sample_rejections=len(group) - len(values),
                        raw_aggregate_ms=sum(float(e["elapsed_ms"]) for e in group),
                        aggregate_ms=(sum(values) * microbatches / len(values) if values else 0.0),
                        mean_ms=mean,
                        median_ms=statistics.median(values) if values else None,
                        min_ms=min(values) if values else None,
                        max_ms=max(values) if values else None,
                    )
                )
            _check_values(
                [d["raw_aggregate_ms"] for d in details], diagnostic["raw_samples_ms"], "raw totals"
            )
            _check_values([d["aggregate_ms"] for d in details], diagnostic["samples_ms"], "totals")
            if [d["accepted_microbatch_count"] for d in details] != diagnostic[
                "accepted_counts_by_iteration"
            ]:
                raise ValueError("Raw replay disagrees with saved accepted sample counts")
            values = [d["mean_ms"] for d in details]
            if any(v is None for v in values):
                raise ValueError("Cannot diagnose iteration distribution with no accepted samples")
            median = statistics.median(values)
            mad = statistics.median(abs(v - median) for v in values)
            for detail in details:
                delta = detail["mean_ms"] - median
                detail["relative_deviation"] = abs(delta) / median
                detail["signed_deviation_ms"] = delta
                detail["robust_z"] = delta / (1.4826 * mad) if mad else None
            _check_values([_cv(values)], [issue["value"]], "within_group_cv")
            ordered = sorted(values)
            failures.append(
                dict(
                    **{k: row[k] for k in IDENTITY_FIELDS},
                    stage=row["stage"],
                    phase=phase,
                    layer_composition=row.get("layer_composition"),
                    model_manifest_hash=row.get("model_manifest_hash"),
                    cv_threshold=issue["threshold"],
                    raw_iteration_cv=_cv([d["raw_aggregate_ms"] for d in details]),
                    sample_cleaned_iteration_cv=_cv(values),
                    median_iteration_value_ms=median,
                    MAD_ms=mad,
                    robust_sigma_ms=1.4826 * mad,
                    first_half_median_ms=statistics.median(values[: len(values) // 2]),
                    second_half_median_ms=statistics.median(values[len(values) // 2 :]),
                    adjacent_changes_ms=[b - a for a, b in zip(values, values[1:])],
                    largest_sorted_gap_ms=max(b - a for a, b in zip(ordered, ordered[1:])),
                    raw_iteration_count=len(iterations),
                    accepted_iteration_count=len(iterations),
                    rejected_iteration_count=0,
                    iteration_discarded_fraction=0.0,
                    rejected_samples=filtering["rejected_samples"],
                    ignored_events_outside_measurement_window=len(raw_events) - len(selected),
                    iterations=details,
                    cause="unclassified_requires_review_of_time_order_and_runtime_evidence",
                )
            )
    return dict(
        schema_version="slackpipe.iteration_inspection.v1",
        diagnostic_only=True,
        iteration_filter_enabled=False,
        profile_status=quality["status"],
        selected_profile=quality.get("selected_profile"),
        thresholds=quality["thresholds"],
        definitions=dict(
            aggregate_ms="B * mean(accepted raw microbatch times), same scale as saved samples_ms",
            mean_ms="per-operation iteration value after existing raw sample filter",
            cv="population standard deviation / mean across iteration values",
            robust_z="signed deviation / (1.4826 * MAD); null for zero MAD, not an anomaly decision",
            descriptors="half medians/adjacent changes/sorted gap are descriptive, not drift or cluster classifiers",
        ),
        failed_stage_phase_count=len(failures),
        failures=failures,
        rejected_samples=replay["rejected_samples"],
        rejected_groups=replay["rejected_groups"],
        boundary_attribution="unavailable from elapsed-only raw events; no clock/allocator/communication attribution inferred",
    )


def main() -> int:
    """Export a new diagnostic file; never overwrite profiles or original events."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        paths = dict(profile=args.profile, quality=args.quality_report, events=args.events)
        inputs = {name: path.read_bytes() for name, path in paths.items()}
        report = inspect_iterations(
            *(json.loads(inputs[k]) for k in ("profile", "quality", "events"))
        )
        report["inputs"] = {
            name: dict(path=str(path.resolve()), sha256=hashlib.sha256(inputs[name]).hexdigest())
            for name, path in paths.items()
        }
        serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
        with args.output.open("x") as stream:
            stream.write(serialized)
        print(
            f"Diagnostic only: {report['failed_stage_phase_count']} failing stage/phase groups; no iteration filtering"
        )
        for failure in report["failures"]:
            print(
                f"{failure['calibration_group_id']} s{failure['stage']} {failure['phase']} "
                f"CV={failure['sample_cleaned_iteration_cv']:.6f} threshold={failure['cv_threshold']}"
            )
            for item in failure["iterations"]:
                print(
                    f"  i={item['iteration']} aggregate={item['aggregate_ms']:.6f} "
                    f"mean={item['mean_ms']:.6f} median={item['median_ms']:.6f} "
                    f"min={item['min_ms']:.6f} max={item['max_ms']:.6f} "
                    f"accepted={item['accepted_microbatch_count']}"
                )
        print(f"Saved {args.output}; profile status unchanged: {report['profile_status']}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Iteration inspection failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
