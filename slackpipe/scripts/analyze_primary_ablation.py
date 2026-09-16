#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from statistics import mean, median
from typing import Any


PRIMARY_MODES = ("joint", "partition-only", "schedule-only-uniform")
BUDGET_POLICY_VERSION = 1
COMPARISONS = (
    ("joint_vs_partition_only", "partition-only"),
    ("joint_vs_schedule_only", "schedule-only-uniform"),
)
SUCCESS_STATUSES = {"OPTIMAL", "FEASIBLE"}
ACTIVATION_COMPATIBILITY_FIELDS = (
    "activation_model",
    "activation_units_per_layer",
    "explicit_stage_activation_units",
    "activation_cap_mode",
    "activation_cap_units_per_worker",
    "activation_cap_enforcement_requested",
    "activation_cap_enforcement_mode",
    "activation_cap_derivation_hash",
    "activation_cap_formulation_version",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_int(value: Any) -> int | None:
    if value in {"", None}:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def safe_float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def successful(row: dict[str, Any] | None) -> bool:
    if row is None:
        return False
    return safe_int(row.get("replayed_makespan")) is not None and str(row.get("solver_status")) in SUCCESS_STATUSES


def truthy(value: Any) -> bool:
    return value in {True, "True", "true", "1", 1}


def activation_key(row: dict[str, Any] | None) -> tuple[Any, ...]:
    if row is None:
        return tuple()
    return tuple(row.get(field, "") for field in ACTIVATION_COMPATIBILITY_FIELDS)


def equal_memory_required(rows: list[dict[str, Any] | None]) -> bool:
    for row in rows:
        if row is None:
            continue
        if str(row.get("activation_cap_mode", "none")) not in {"", "none", "None"}:
            return True
        if truthy(row.get("activation_cap_enforced")):
            return True
    return False


def solver_backed_row(row: dict[str, Any]) -> bool:
    method = str(row.get("canonical_method", row.get("requested_method", "")))
    mode = str(row.get("optimization_mode", ""))
    if method == "uniform-breadth-first" or mode == "uniform-breadth-first":
        return False
    return mode in PRIMARY_MODES or method != ""


def equal_memory_eligible(row: dict[str, Any]) -> bool:
    if str(row.get("activation_cap_mode", "none")) in {"", "none", "None"}:
        return False
    if not truthy(row.get("activation_cap_enforcement_requested")):
        return False
    if not truthy(row.get("activation_cap_satisfied")):
        return False
    if row.get("activation_cap_formulation_version", "") in {"", None}:
        return False
    if row.get("activation_model_validation_agreement") is False:
        return False
    mode = str(row.get("activation_cap_enforcement_mode", ""))
    if mode == "deterministic_postconstruction_check":
        return not solver_backed_row(row)
    if mode != "solver":
        return False
    if solver_backed_row(row):
        return truthy(row.get("activation_cap_enforced_in_solver")) and truthy(
            row.get("activation_cap_constraints_added")
        )
    return True


def activation_memory_compatible(rows: list[dict[str, Any] | None]) -> bool:
    present = [row for row in rows if row is not None]
    if not present:
        return False
    keys = {activation_key(row) for row in present}
    if len(keys) != 1:
        return False
    if not equal_memory_required(present):
        return True
    for row in present:
        if not equal_memory_eligible(row):
            return False
    return True


def latest_completed_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("completed") not in {True, "True", "true", "1", 1}:
            continue
        run_id = str(row.get("run_id", ""))
        if run_id:
            latest[run_id] = row
    return latest


def group_results_by_configuration(
    manifest_rows: list[dict[str, str]],
    result_rows: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any] | None]]:
    grouped: dict[str, dict[str, dict[str, Any] | None]] = defaultdict(dict)
    for manifest in manifest_rows:
        config_id = manifest["configuration_id"]
        mode = manifest["optimization_mode"]
        grouped[config_id][mode] = result_rows.get(manifest["run_id"])
        grouped[config_id]["__manifest__"] = manifest
    return grouped


def comparison_outcome(joint_time: int | None, baseline_time: int | None) -> str:
    if joint_time is None or baseline_time is None:
        return "missing"
    if joint_time < baseline_time:
        return "win"
    if joint_time == baseline_time:
        return "tie"
    return "loss"


def improvement_percent(joint_time: int | None, baseline_time: int | None) -> float | None:
    if joint_time is None or baseline_time is None or baseline_time <= 0:
        return None
    return 100.0 * float(baseline_time - joint_time) / float(baseline_time)


def paired_rows(
    manifest_rows: list[dict[str, str]],
    result_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped = group_results_by_configuration(manifest_rows, result_rows)
    paired: list[dict[str, Any]] = []
    for config_id in sorted(grouped):
        group = grouped[config_id]
        manifest = group["__manifest__"]
        row: dict[str, Any] = {
            "configuration_id": config_id,
            "depth_set": manifest["depth_set"],
            "B": manifest["B"],
            "N": manifest["N"],
            "W": manifest["W"],
            "L": manifest["L"],
            "microbatches_per_worker": manifest.get("microbatches_per_worker", manifest.get("B_over_W", "")),
            "stages_per_worker": manifest.get("stages_per_worker", manifest.get("N_over_W", "")),
            "layers_per_stage": manifest.get("layers_per_stage", manifest.get("L_over_N", "")),
            "B_over_W": manifest.get("B_over_W", manifest.get("microbatches_per_worker", "")),
            "N_over_W": manifest.get("N_over_W", manifest.get("stages_per_worker", "")),
            "L_over_N": manifest.get("L_over_N", manifest.get("layers_per_stage", "")),
        }
        times: dict[str, int | None] = {}
        for mode in PRIMARY_MODES:
            result = group.get(mode)
            times[mode] = safe_int(result.get("replayed_makespan")) if result else None
            row[f"{mode}_status"] = "" if result is None else result.get("solver_status", "")
            row[f"{mode}_proven_optimal"] = "" if result is None else result.get("proven_optimal", "")
            row[f"{mode}_gap"] = "" if result is None else result.get("relative_optimality_gap", "")
            row[f"{mode}_T"] = "" if times[mode] is None else times[mode]
            row[f"{mode}_utilization"] = "" if result is None else result.get("utilization", "")
            row[f"{mode}_final_split"] = "" if result is None else result.get("final_split", "")
            row[f"{mode}_final_worker_local_order"] = "" if result is None else result.get("final_worker_local_order", "")
            row[f"{mode}_activation_peak_units"] = "" if result is None else result.get("maximum_worker_peak_activation_units", "")
            row[f"{mode}_activation_ratio_to_uniform"] = "" if result is None else result.get("activation_peak_ratio_to_uniform", "")
            row[f"{mode}_activation_cap_mode"] = "" if result is None else result.get("activation_cap_mode", "")
            row[f"{mode}_activation_cap_enforcement_mode"] = "" if result is None else result.get("activation_cap_enforcement_mode", "")
            row[f"{mode}_activation_cap_enforced_in_solver"] = "" if result is None else result.get("activation_cap_enforced_in_solver", "")
            row[f"{mode}_activation_cap_constraints_added"] = "" if result is None else result.get("activation_cap_constraints_added", "")
            row[f"{mode}_activation_cap_satisfied"] = "" if result is None else result.get("activation_cap_satisfied", "")
            row[f"{mode}_activation_equal_memory_eligible"] = "" if result is None else equal_memory_eligible(result)
        primary_results = [group.get(mode) for mode in PRIMARY_MODES]
        row["activation_memory_comparison_required"] = equal_memory_required(primary_results)
        row["activation_memory_compatible"] = activation_memory_compatible(primary_results)
        joint_time = times["joint"]
        for comparison, baseline_mode in COMPARISONS:
            baseline_time = times[baseline_mode]
            percent = improvement_percent(joint_time, baseline_time)
            if joint_time is not None and baseline_time is not None and joint_time > 0:
                row[f"{comparison}_ratio"] = float(baseline_time) / float(joint_time)
            else:
                row[f"{comparison}_ratio"] = ""
            row[f"{comparison}_outcome"] = comparison_outcome(joint_time, baseline_time)
            row[f"{comparison}_delta_ticks"] = (
                "" if joint_time is None or baseline_time is None else baseline_time - joint_time
            )
            row[f"{comparison}_improvement_percent"] = "" if percent is None else percent
        paired.append(row)
    return paired


def validate_primary_dataset(
    manifest_rows: list[dict[str, str]],
    results_by_run_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    manifest_config_ids = {row["configuration_id"] for row in manifest_rows}
    mode_counts = Counter(row["optimization_mode"] for row in manifest_rows)
    config_mode_counts = Counter((row["configuration_id"], row["optimization_mode"]) for row in manifest_rows)
    run_count = len(manifest_rows)
    missing_runs = [row for row in manifest_rows if row["run_id"] not in results_by_run_id]
    validation_errors: list[str] = []
    if len(manifest_config_ids) != 520:
        validation_errors.append(f"expected 520 configurations, found {len(manifest_config_ids)}")
    if run_count != 1560:
        validation_errors.append(f"expected 1560 run rows, found {run_count}")
    if dict(mode_counts) != {mode: 520 for mode in PRIMARY_MODES}:
        validation_errors.append(f"mode counts mismatch: {dict(mode_counts)}")
    if any(row["optimization_mode"] == "schedule-only-best" for row in manifest_rows):
        validation_errors.append("primary manifest includes schedule-only-best")
    duplicate_config_modes = {
        f"{config_id}/{mode}": count
        for (config_id, mode), count in config_mode_counts.items()
        if count > 1
    }
    if duplicate_config_modes:
        validation_errors.append("duplicate configuration/mode keys found")
    for row in manifest_rows:
        for field in ("B", "N", "W", "L"):
            value = safe_int(row.get(field))
            if value is None or value <= 0:
                validation_errors.append(f"nonpositive or invalid {field} in manifest row {row.get('run_id')}")
        N = safe_int(row.get("N"))
        W = safe_int(row.get("W"))
        L = safe_int(row.get("L"))
        if N is not None and W is not None and N % W != 0:
            validation_errors.append(f"N % W violation in manifest row {row.get('run_id')}")
        if N is not None and L is not None and N > L:
            validation_errors.append(f"N > L violation in manifest row {row.get('run_id')}")

    useful_work_by_config: dict[str, set[str]] = defaultdict(set)
    protocol_violations: list[str] = []
    restriction_violations: list[str] = []
    for manifest in manifest_rows:
        result = results_by_run_id.get(manifest["run_id"])
        if result is None:
            continue
        config_id = manifest["configuration_id"]
        if result.get("useful_work") not in {"", None}:
            useful_work_by_config[config_id].add(str(result["useful_work"]))
        if int(result.get("W", manifest["W"])) != int(result.get("pipeline_workers", manifest["W"])):
            protocol_violations.append(f"{manifest['run_id']}: pipeline_workers != W")
        if int(result.get("solver_threads", 0)) != 16:
            protocol_violations.append(f"{manifest['run_id']}: solver_threads != 16")
        if float(result.get("time_limit_seconds", 0.0)) != 600.0:
            protocol_violations.append(f"{manifest['run_id']}: time_limit_seconds != 600")
        if int(result.get("budget_policy_version", 0) or 0) != BUDGET_POLICY_VERSION:
            protocol_violations.append(
                f"{manifest['run_id']}: budget_policy_version != {BUDGET_POLICY_VERSION}"
            )
        if int(result.get("random_seed", 0)) != 1:
            protocol_violations.append(f"{manifest['run_id']}: random_seed != 1")
        if result.get("sensitivity_axis") != "primary":
            protocol_violations.append(f"{manifest['run_id']}: sensitivity row in primary output")
        if successful(result):
            try:
                makespan = float(result.get("replayed_makespan"))
                useful = float(result.get("useful_work"))
                utilization = float(result.get("utilization"))
                expected_utilization = useful / (float(result.get("W", manifest["W"])) * makespan)
                if abs(utilization - expected_utilization) > 1e-6:
                    restriction_violations.append(f"{manifest['run_id']}: utilization mismatch")
                incumbent = result.get("incumbent_objective", "")
                if incumbent not in {"", None} and abs(float(incumbent) - makespan) > 1e-6:
                    restriction_violations.append(f"{manifest['run_id']}: incumbent objective mismatch")
            except Exception as error:
                restriction_violations.append(f"{manifest['run_id']}: metric validation error {error}")
        if manifest["optimization_mode"] == "schedule-only-uniform" and successful(result):
            if result.get("final_split") != result.get("uniform_split"):
                restriction_violations.append(f"{manifest['run_id']}: schedule-only changed uniform split")
        if manifest["optimization_mode"] == "partition-only" and successful(result):
            if result.get("partition_order_validation_passed") not in {True, "True", "true", "1", 1}:
                restriction_violations.append(f"{manifest['run_id']}: partition-only order was not validated")

    useful_work_mismatches = {
        config_id: sorted(values)
        for config_id, values in useful_work_by_config.items()
        if len(values) > 1
    }
    if useful_work_mismatches:
        validation_errors.append("useful work differs across modes")
    if protocol_violations:
        validation_errors.append("primary protocol violations found")
    if restriction_violations:
        validation_errors.append("mode restriction violations found")

    status_counts = Counter(str(row.get("solver_status", "UNKNOWN")) for row in results_by_run_id.values())
    return {
        "manifest_rows": run_count,
        "configuration_count": len(manifest_config_ids),
        "mode_counts": dict(mode_counts),
        "expected_rows": 1560,
        "completed_rows": len(results_by_run_id),
        "missing_rows": len(missing_runs),
        "missing_run_ids": [row["run_id"] for row in missing_runs[:100]],
        "status_counts": dict(status_counts),
        "useful_work_mismatches": useful_work_mismatches,
        "duplicate_config_mode_keys": duplicate_config_modes,
        "protocol_violations": protocol_violations[:100],
        "restriction_violations": restriction_violations[:100],
        "validation_errors": validation_errors,
        "validation_passed": not validation_errors and not missing_runs,
    }


def win_tie_loss_rows(paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comparison, _ in COMPARISONS:
        counts = Counter(str(row[f"{comparison}_outcome"]) for row in paired)
        rows.append(
            {
                "comparison": comparison,
                "joint_wins": counts["win"],
                "ties": counts["tie"],
                "joint_losses": counts["loss"],
                "missing": counts["missing"],
            }
        )
    return rows


def numeric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = safe_float(row.get(field))
        if value is not None:
            values.append(value)
    return values


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * p
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def bootstrap_mean_interval(values: list[float], iterations: int = 2000, seed: int = 1) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    samples = []
    for _ in range(iterations):
        samples.append(mean(values[rng.randrange(n)] for _ in range(n)))
    return percentile(samples, 0.025), percentile(samples, 0.975)


def geometric_mean(values: list[float]) -> float | None:
    positives = [value for value in values if value > 0]
    if not positives:
        return None
    return math.exp(mean(math.log(value) for value in positives))


def comparison_success_rows(paired: list[dict[str, Any]], comparison: str) -> list[dict[str, Any]]:
    return [
        row
        for row in paired
        if row.get(f"{comparison}_outcome") in {"win", "tie", "loss"}
        and (
            not truthy(row.get("activation_memory_comparison_required"))
            or truthy(row.get("activation_memory_compatible"))
        )
    ]


def aggregate_summary_rows(paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comparison, baseline_mode in COMPARISONS:
        subset = comparison_success_rows(paired, comparison)
        outcomes = Counter(str(row.get(f"{comparison}_outcome")) for row in subset)
        improvements = numeric_values(subset, f"{comparison}_improvement_percent")
        ratios = numeric_values(subset, f"{comparison}_ratio")
        ci_lo, ci_hi = bootstrap_mean_interval(improvements)
        rows.append(
            {
                "comparison": comparison,
                "baseline_mode": baseline_mode,
                "paired_configuration_count": len(subset),
                "excluded_configuration_count": len(paired) - len(subset),
                "joint_wins": outcomes["win"],
                "ties": outcomes["tie"],
                "joint_losses": outcomes["loss"],
                "win_rate": "" if not subset else outcomes["win"] / len(subset),
                "median_improvement_percent": "" if not improvements else median(improvements),
                "mean_improvement_percent": "" if not improvements else mean(improvements),
                "geomean_speedup": "" if geometric_mean(ratios) is None else geometric_mean(ratios),
                "p05_improvement_percent": "" if percentile(improvements, 0.05) is None else percentile(improvements, 0.05),
                "p25_improvement_percent": "" if percentile(improvements, 0.25) is None else percentile(improvements, 0.25),
                "p75_improvement_percent": "" if percentile(improvements, 0.75) is None else percentile(improvements, 0.75),
                "p95_improvement_percent": "" if percentile(improvements, 0.95) is None else percentile(improvements, 0.95),
                "minimum_improvement_percent": "" if not improvements else min(improvements),
                "maximum_improvement_percent": "" if not improvements else max(improvements),
                "bootstrap_mean_improvement_ci95_low": "" if ci_lo is None else ci_lo,
                "bootstrap_mean_improvement_ci95_high": "" if ci_hi is None else ci_hi,
            }
        )
    return rows


def breakdown_rows(paired: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        groups[str(row[field])].append(row)
    out: list[dict[str, Any]] = []
    for value in sorted(groups, key=lambda item: (safe_float(item) is None, safe_float(item) or 0.0, item)):
        group = groups[value]
        row: dict[str, Any] = {"breakdown": field, "value": value, "configurations": len(group)}
        for comparison, _ in COMPARISONS:
            counts = Counter(str(item[f"{comparison}_outcome"]) for item in group)
            improvements = numeric_values(group, f"{comparison}_improvement_percent")
            success_count = counts["win"] + counts["tie"] + counts["loss"]
            row[f"{comparison}_wins"] = counts["win"]
            row[f"{comparison}_ties"] = counts["tie"]
            row[f"{comparison}_losses"] = counts["loss"]
            row[f"{comparison}_missing"] = counts["missing"]
            row[f"{comparison}_paired_configuration_count"] = success_count
            row[f"{comparison}_win_rate"] = "" if success_count == 0 else counts["win"] / success_count
            row[f"{comparison}_mean_improvement_percent"] = "" if not improvements else mean(improvements)
            row[f"{comparison}_median_improvement_percent"] = "" if not improvements else median(improvements)
            row[f"{comparison}_max_improvement_percent"] = "" if not improvements else max(improvements)
        out.append(row)
    return out


def status_summary_rows(results_by_run_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results_by_run_id.values():
        groups[(str(row.get("optimization_mode", "")), str(row.get("solver_status", "")))].append(row)
    out: list[dict[str, Any]] = []
    for (mode, status), rows in sorted(groups.items()):
        gaps = numeric_values(rows, "relative_optimality_gap")
        out.append(
            {
                "optimization_mode": mode,
                "solver_status": status,
                "rows": len(rows),
                "mean_relative_optimality_gap": "" if not gaps else mean(gaps),
                "max_relative_optimality_gap": "" if not gaps else max(gaps),
            }
        )
    return out


def optimality_gap_rows(results_by_run_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results_by_run_id.values():
        groups[str(row.get("optimization_mode", ""))].append(row)
    out: list[dict[str, Any]] = []
    for mode, rows in sorted(groups.items()):
        gaps = numeric_values(rows, "relative_optimality_gap")
        proven = sum(1 for row in rows if row.get("proven_optimal") in {True, "True", "true", "1", 1})
        out.append(
            {
                "optimization_mode": mode,
                "rows": len(rows),
                "proven_optimal_rows": proven,
                "best_incumbent_rows": len(rows) - proven,
                "mean_relative_optimality_gap": "" if not gaps else mean(gaps),
                "median_relative_optimality_gap": "" if not gaps else median(gaps),
                "max_relative_optimality_gap": "" if not gaps else max(gaps),
            }
        )
    return out


def write_validation_report(output_dir: Path, validation: dict[str, Any]) -> None:
    lines = [
        "SlackPipe Primary Ablation Validation Report",
        f"validation_passed: {validation['validation_passed']}",
        f"manifest_rows: {validation['manifest_rows']}",
        f"configuration_count: {validation['configuration_count']}",
        f"completed_rows: {validation['completed_rows']}",
        f"missing_rows: {validation['missing_rows']}",
        f"status_counts: {json.dumps(validation['status_counts'], sort_keys=True)}",
        f"validation_errors: {json.dumps(validation['validation_errors'], sort_keys=True)}",
        f"protocol_violations: {json.dumps(validation['protocol_violations'], sort_keys=True)}",
        f"restriction_violations: {json.dumps(validation['restriction_violations'], sort_keys=True)}",
        f"useful_work_mismatches: {json.dumps(validation['useful_work_mismatches'], sort_keys=True)}",
        f"missing_run_ids_first_100: {json.dumps(validation['missing_run_ids'], sort_keys=True)}",
    ]
    (output_dir / "validation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def wall_clock_duration_seconds(results_by_run_id: dict[str, dict[str, Any]]) -> float | None:
    timestamps = []
    for row in results_by_run_id.values():
        for key in ("start_timestamp_utc", "end_timestamp_utc"):
            value = row.get(key)
            if isinstance(value, str) and value.endswith("Z"):
                try:
                    timestamps.append(datetime_from_utc_text(value))
                except ValueError:
                    pass
    if len(timestamps) < 2:
        return None
    return (max(timestamps) - min(timestamps)).total_seconds()


def datetime_from_utc_text(value: str):
    from datetime import datetime, timezone

    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def write_execution_summary(
    output_dir: Path,
    validation: dict[str, Any],
    aggregate: list[dict[str, Any]],
    results_by_run_id: dict[str, dict[str, Any]],
    manifest_csv: Path,
    results_jsonl: Path,
) -> None:
    duration = wall_clock_duration_seconds(results_by_run_id)
    lines = [
        "# SlackPipe Primary Ablation Execution Summary",
        "",
        f"- Intended rows: {validation['expected_rows']}",
        f"- Completed rows: {validation['completed_rows']}",
        f"- Successful rows: {sum(count for status, count in validation['status_counts'].items() if status in {'OPTIMAL', 'FEASIBLE'})}",
        f"- Failed rows: {validation['completed_rows'] - sum(count for status, count in validation['status_counts'].items() if status in {'OPTIMAL', 'FEASIBLE'})}",
        f"- Actual wall-clock duration seconds: {'' if duration is None else duration}",
        f"- Solver-status distribution: `{json.dumps(validation['status_counts'], sort_keys=True)}`",
        f"- Missing rows: {validation['missing_rows']}",
        f"- Invariant violations: `{json.dumps(validation['validation_errors'], sort_keys=True)}`",
        f"- Manifest: `{manifest_csv}`",
        f"- Results JSONL: `{results_jsonl}`",
        "",
        "Results marked FEASIBLE are the best solution found within the common ten-minute budget.",
        "",
        "## Paired Comparisons",
        "",
        "| Comparison | Paired configs | Joint wins | Ties | Joint losses | Mean improvement % | Geomean speedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['comparison']} | {row['paired_configuration_count']} | {row['joint_wins']} | {row['ties']} | {row['joint_losses']} | {row['mean_improvement_percent']} | {row['geomean_speedup']} |"
        )
    lines.extend(
        [
            "",
            "## Artifact Paths",
            "",
        ]
    )
    for name in [
        "environment_report.txt",
        "config_manifest.csv",
        "raw_results.csv",
        "completeness_matrix.csv",
        "paired_results.csv",
        "aggregate_summary.csv",
        "win_tie_loss.csv",
        "by_B.csv",
        "by_N.csv",
        "by_W.csv",
        "by_L.csv",
        "by_B_over_W.csv",
        "by_N_over_W.csv",
        "by_L_over_N.csv",
        "solver_status_summary.csv",
        "optimality_gap_summary.csv",
        "normalized_heatmaps.pdf",
        "sorted_improvements.pdf",
        "improvement_cdfs.pdf",
        "structural_breakdowns.pdf",
        "solver_quality.pdf",
        "table.tex",
        "validation_report.txt",
    ]:
        lines.append(f"- `{output_dir / name}`")
    (output_dir / "execution_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table_tex(output_dir: Path, aggregate: list[dict[str, Any]]) -> None:
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\hline",
        "Comparison & Paired & Wins & Ties & Losses & Mean \\% & Geo. speedup \\\\",
        "\\hline",
    ]
    for row in aggregate:
        lines.append(
            f"{row['comparison']} & {row['paired_configuration_count']} & {row['joint_wins']} & {row['ties']} & {row['joint_losses']} & {float(row['mean_improvement_percent'] or 0):.3f} & {float(row['geomean_speedup'] or 0):.4f} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}"])
    (output_dir / "table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def svg_escape(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def value_color(value: float | None, max_abs: float) -> str:
    if value is None:
        return "#f2f2f2"
    if max_abs <= 0:
        return "#f7f7f7"
    intensity = min(1.0, abs(value) / max_abs)
    if value > 0:
        r = int(238 - 160 * intensity)
        g = int(245 - 80 * intensity)
        b = int(233 - 160 * intensity)
    elif value < 0:
        r = int(252 - 80 * intensity)
        g = int(232 - 150 * intensity)
        b = int(232 - 150 * intensity)
    else:
        r = g = b = 245
    return f"#{r:02x}{g:02x}{b:02x}"


def write_heatmap_svg(paired: list[dict[str, Any]], comparison: str, path: Path) -> None:
    x_values = ["1/2", "1", "2", "4"]
    y_values = ["1", "2", "4"]
    w_values = sorted({str(row["W"]) for row in paired}, key=lambda item: int(item))
    l_values = sorted({str(row["L"]) for row in paired}, key=lambda item: int(item))
    lookup: dict[tuple[str, str, str, str], float] = {}
    for row in paired:
        value = safe_float(row.get(f"{comparison}_improvement_percent"))
        if value is not None:
            lookup[(str(row["W"]), str(row["L"]), str(row["microbatches_per_worker"]), str(row["stages_per_worker"]))] = value
    max_abs = max([abs(value) for value in lookup.values()] + [1.0])

    facet_w = 104
    facet_h = 84
    cell = 18
    left = 70
    top = 45
    width = left + facet_w * len(l_values) + 30
    height = top + facet_h * len(w_values) + 55
    lines = [
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">",
        "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>",
        f"<text x=\"{left}\" y=\"24\" font-family=\"sans-serif\" font-size=\"16\" font-weight=\"700\">{svg_escape(comparison)} improvement percent</text>",
        f"<text x=\"{left}\" y=\"40\" font-family=\"sans-serif\" font-size=\"11\">Cell axes: x=B/W, y=N/W; facets by W rows and L columns</text>",
    ]
    for col, L in enumerate(l_values):
        x = left + col * facet_w + 34
        lines.append(f"<text x=\"{x}\" y=\"62\" font-family=\"sans-serif\" font-size=\"11\" text-anchor=\"middle\">L={L}</text>")
    for row_index, W in enumerate(w_values):
        y = top + row_index * facet_h + 58
        lines.append(f"<text x=\"12\" y=\"{y}\" font-family=\"sans-serif\" font-size=\"11\">W={W}</text>")
        for col, L in enumerate(l_values):
            fx = left + col * facet_w
            fy = top + row_index * facet_h + 25
            lines.append(f"<rect x=\"{fx}\" y=\"{fy}\" width=\"{cell * len(x_values)}\" height=\"{cell * len(y_values)}\" fill=\"none\" stroke=\"#ddd\"/>")
            for yi, stages in enumerate(y_values):
                for xi, micro in enumerate(x_values):
                    value = lookup.get((W, L, micro, stages))
                    color = value_color(value, max_abs)
                    x = fx + xi * cell
                    y = fy + yi * cell
                    label = "" if value is None else f"{value:.0f}"
                    lines.append(f"<rect x=\"{x}\" y=\"{y}\" width=\"{cell}\" height=\"{cell}\" fill=\"{color}\" stroke=\"#fff\"/>")
                    lines.append(f"<text x=\"{x + cell / 2}\" y=\"{y + 12}\" font-family=\"sans-serif\" font-size=\"8\" text-anchor=\"middle\">{label}</text>")
    lines.append(f"<text x=\"{left}\" y=\"{height - 18}\" font-family=\"sans-serif\" font-size=\"11\">Positive values mean Joint has lower replayed makespan than the baseline. Ties are 0.</text>")
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sorted_improvement_svg(paired: list[dict[str, Any]], comparison: str, path: Path) -> None:
    values = sorted(numeric_values(paired, f"{comparison}_improvement_percent"))
    width = 900
    height = 320
    left = 55
    top = 30
    plot_w = width - left - 30
    plot_h = height - top - 55
    if not values:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"/>\n", encoding="utf-8")
        return
    lo = min(min(values), 0.0)
    hi = max(max(values), 0.0)
    span = hi - lo if hi != lo else 1.0
    points = []
    for index, value in enumerate(values):
        x = left + plot_w * (index / max(1, len(values) - 1))
        y = top + plot_h * (1.0 - (value - lo) / span)
        points.append(f"{x:.1f},{y:.1f}")
    zero_y = top + plot_h * (1.0 - (0.0 - lo) / span)
    lines = [
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">",
        "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>",
        f"<text x=\"{left}\" y=\"22\" font-family=\"sans-serif\" font-size=\"16\" font-weight=\"700\">Sorted {svg_escape(comparison)} improvement</text>",
        f"<line x1=\"{left}\" y1=\"{zero_y:.1f}\" x2=\"{left + plot_w}\" y2=\"{zero_y:.1f}\" stroke=\"#aaa\"/>",
        f"<polyline fill=\"none\" stroke=\"#4e79a7\" stroke-width=\"2\" points=\"{' '.join(points)}\"/>",
        f"<text x=\"{left}\" y=\"{height - 18}\" font-family=\"sans-serif\" font-size=\"11\">Configurations sorted by improvement percent; positive means Joint is faster.</text>",
        "</svg>",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cdf_svg(paired: list[dict[str, Any]], comparison: str, path: Path) -> None:
    values = sorted(numeric_values(paired, f"{comparison}_improvement_percent"))
    width = 700
    height = 320
    left = 55
    top = 30
    plot_w = width - left - 30
    plot_h = height - top - 55
    if not values:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"/>\n", encoding="utf-8")
        return
    lo = min(min(values), 0.0)
    hi = max(max(values), 0.0)
    span = hi - lo if hi != lo else 1.0
    points = []
    for index, value in enumerate(values):
        x = left + plot_w * (value - lo) / span
        y = top + plot_h * (1.0 - index / max(1, len(values) - 1))
        points.append(f"{x:.1f},{y:.1f}")
    zero_x = left + plot_w * (0.0 - lo) / span
    lines = [
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">",
        "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>",
        f"<text x=\"{left}\" y=\"22\" font-family=\"sans-serif\" font-size=\"16\" font-weight=\"700\">Empirical CDF: {svg_escape(comparison)}</text>",
        f"<line x1=\"{zero_x:.1f}\" y1=\"{top}\" x2=\"{zero_x:.1f}\" y2=\"{top + plot_h}\" stroke=\"#aaa\"/>",
        f"<polyline fill=\"none\" stroke=\"#59a14f\" stroke-width=\"2\" points=\"{' '.join(points)}\"/>",
        f"<text x=\"{left}\" y=\"{height - 18}\" font-family=\"sans-serif\" font-size=\"11\">x is improvement percent; y is empirical cumulative fraction.</text>",
        "</svg>",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pyplot_and_pdfpages():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    return plt, PdfPages


def comparison_caption(paired: list[dict[str, Any]], comparison: str) -> str:
    included = len(comparison_success_rows(paired, comparison))
    excluded = len(paired) - included
    return f"Included configurations: {included}. Excluded rows/configurations without paired successful data: {excluded}."


def write_normalized_heatmaps_pdf(paired: list[dict[str, Any]], output_dir: Path) -> None:
    plt, PdfPages = pyplot_and_pdfpages()
    with PdfPages(output_dir / "normalized_heatmaps.pdf") as pdf:
        for comparison, _ in COMPARISONS:
            subset = comparison_success_rows(paired, comparison)
            x_values = sorted({str(row["B_over_W"]) for row in subset}, key=lambda item: float(Fraction(item)))
            y_values = sorted({str(row["N_over_W"]) for row in subset}, key=lambda item: float(Fraction(item)))
            matrix = []
            for y in y_values:
                line = []
                for x in x_values:
                    ratios = [
                        safe_float(row.get(f"{comparison}_ratio"))
                        for row in subset
                        if str(row["B_over_W"]) == x and str(row["N_over_W"]) == y
                    ]
                    ratios = [value for value in ratios if value is not None]
                    line.append(mean(ratios) if ratios else math.nan)
                matrix.append(line)
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            image = ax.imshow(matrix, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(x_values)), x_values)
            ax.set_yticks(range(len(y_values)), y_values)
            ax.set_xlabel("B/W")
            ax.set_ylabel("N/W")
            ax.set_title(f"Normalized makespan ratio: {comparison}")
            for yi, line in enumerate(matrix):
                for xi, value in enumerate(line):
                    if not math.isnan(value):
                        ax.text(xi, yi, f"{value:.2f}", ha="center", va="center", color="white" if value < 1.2 else "black")
            fig.colorbar(image, ax=ax, label="baseline / joint")
            fig.text(0.02, 0.01, comparison_caption(paired, comparison), fontsize=8)
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            pdf.savefig(fig)
            plt.close(fig)


def write_sorted_improvements_pdf(paired: list[dict[str, Any]], output_dir: Path) -> None:
    plt, PdfPages = pyplot_and_pdfpages()
    with PdfPages(output_dir / "sorted_improvements.pdf") as pdf:
        for comparison, _ in COMPARISONS:
            values = sorted(numeric_values(comparison_success_rows(paired, comparison), f"{comparison}_improvement_percent"))
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(range(1, len(values) + 1), values, marker=".", linewidth=1)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(f"Sorted improvements: {comparison}")
            ax.set_xlabel("Paired configuration rank")
            ax.set_ylabel("Improvement percent")
            fig.text(0.02, 0.01, comparison_caption(paired, comparison), fontsize=8)
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            pdf.savefig(fig)
            plt.close(fig)


def write_improvement_cdfs_pdf(paired: list[dict[str, Any]], output_dir: Path) -> None:
    plt, PdfPages = pyplot_and_pdfpages()
    with PdfPages(output_dir / "improvement_cdfs.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        captions = []
        for comparison, _ in COMPARISONS:
            values = sorted(numeric_values(comparison_success_rows(paired, comparison), f"{comparison}_improvement_percent"))
            if not values:
                continue
            y = [(index + 1) / len(values) for index in range(len(values))]
            ax.plot(values, y, label=comparison)
            captions.append(comparison_caption(paired, comparison))
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Empirical CDFs of improvement")
        ax.set_xlabel("Improvement percent")
        ax.set_ylabel("Empirical cumulative fraction")
        ax.legend()
        fig.text(0.02, 0.01, " ".join(captions), fontsize=8)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        pdf.savefig(fig)
        plt.close(fig)


def write_structural_breakdowns_pdf(paired: list[dict[str, Any]], output_dir: Path) -> None:
    plt, PdfPages = pyplot_and_pdfpages()
    with PdfPages(output_dir / "structural_breakdowns.pdf") as pdf:
        for field in ("B", "N", "W", "L", "B_over_W", "N_over_W", "L_over_N"):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            labels = sorted({str(row[field]) for row in paired}, key=lambda item: float(Fraction(item)))
            offsets = [-0.18, 0.18]
            width = 0.34
            for offset, (comparison, _) in zip(offsets, COMPARISONS):
                values = []
                for label in labels:
                    subset = [row for row in comparison_success_rows(paired, comparison) if str(row[field]) == label]
                    improvements = numeric_values(subset, f"{comparison}_improvement_percent")
                    values.append(mean(improvements) if improvements else 0.0)
                ax.bar([index + offset for index in range(len(labels))], values, width=width, label=comparison)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(range(len(labels)), labels, rotation=45 if len(labels) > 8 else 0)
            ax.set_title(f"Mean improvement by {field}")
            ax.set_ylabel("Mean improvement percent")
            ax.legend()
            fig.text(0.02, 0.01, "All plotted bars use paired successful configurations in their bin; missing bins are zero-height.", fontsize=8)
            fig.tight_layout(rect=(0, 0.05, 1, 1))
            pdf.savefig(fig)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = range(len(COMPARISONS))
        wins = []
        ties = []
        losses = []
        labels = []
        for comparison, _ in COMPARISONS:
            subset = comparison_success_rows(paired, comparison)
            counts = Counter(str(row[f"{comparison}_outcome"]) for row in subset)
            labels.append(comparison)
            wins.append(counts["win"])
            ties.append(counts["tie"])
            losses.append(counts["loss"])
        ax.bar(x, wins, label="Joint wins")
        ax.bar(x, ties, bottom=wins, label="Ties")
        ax.bar(x, losses, bottom=[wins[i] + ties[i] for i in range(len(wins))], label="Joint losses")
        ax.set_xticks(list(x), labels, rotation=20, ha="right")
        ax.set_ylabel("Configurations")
        ax.set_title("Win/tie/loss counts")
        ax.legend()
        fig.text(0.02, 0.01, "Counts include every paired successful configuration for each comparison.", fontsize=8)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        pdf.savefig(fig)
        plt.close(fig)


def write_solver_quality_pdf(results_by_run_id: dict[str, dict[str, Any]], output_dir: Path) -> None:
    plt, PdfPages = pyplot_and_pdfpages()
    rows = list(results_by_run_id.values())
    with PdfPages(output_dir / "solver_quality.pdf") as pdf:
        status_counts = Counter(str(row.get("solver_status", "UNKNOWN")) for row in rows)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        labels = list(status_counts)
        ax.bar(labels, [status_counts[label] for label in labels])
        ax.set_title("Solver status distribution")
        ax.set_ylabel("Rows")
        ax.tick_params(axis="x", rotation=35)
        fig.text(0.02, 0.01, f"Included rows: {len(rows)}. Excluded rows: 0 from completed result set.", fontsize=8)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for mode in PRIMARY_MODES:
            gaps = numeric_values([row for row in rows if row.get("optimization_mode") == mode], "relative_optimality_gap")
            if gaps:
                ax.hist(gaps, bins=30, alpha=0.55, label=mode)
        ax.set_title("Relative optimality gap distribution")
        ax.set_xlabel("Relative gap")
        ax.set_ylabel("Rows")
        ax.legend()
        fig.text(0.02, 0.01, f"Included rows: {len(rows)}. Proven optima have gap 0 when the bound equals the incumbent.", fontsize=8)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        pdf.savefig(fig)
        plt.close(fig)


def write_summary_md(output_dir: Path, validation: dict[str, Any], wins: list[dict[str, Any]]) -> None:
    lines = [
        "# Primary Joint Versus One-Dimensional Ablation",
        "",
        f"- Manifest configurations: {validation['configuration_count']}",
        f"- Manifest run rows: {validation['manifest_rows']}",
        f"- Completed rows: {validation['completed_rows']}",
        f"- Missing rows: {validation['missing_rows']}",
        f"- Validation passed: {validation['validation_passed']}",
        "",
        "Nonoptimal rows are interpreted as the best solution found within the common ten-minute optimization budget.",
        "",
        "## Win/Tie/Loss",
        "",
        "| Comparison | Joint wins | Ties | Joint losses | Missing |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in wins:
        lines.append(
            f"| {row['comparison']} | {row['joint_wins']} | {row['ties']} | {row['joint_losses']} | {row['missing']} |"
        )
    if validation["validation_errors"]:
        lines.extend(["", "## Validation Issues", ""])
        for issue in validation["validation_errors"]:
            lines.append(f"- {issue}")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(results_jsonl: Path, manifest_csv: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = read_csv(manifest_csv)
    result_rows = latest_completed_rows(read_jsonl(results_jsonl))
    validation = validate_primary_dataset(manifest_rows, result_rows)
    paired = paired_rows(manifest_rows, result_rows)
    wins = win_tie_loss_rows(paired)
    aggregate = aggregate_summary_rows(paired)

    paired_fields = list(paired[0].keys()) if paired else []
    write_csv(output_dir / "paired_results.csv", paired, paired_fields)
    write_csv(output_dir / "win_tie_loss.csv", wins, ["comparison", "joint_wins", "ties", "joint_losses", "missing"])
    write_csv(
        output_dir / "aggregate_summary.csv",
        aggregate,
        [
            "comparison",
            "baseline_mode",
            "paired_configuration_count",
            "excluded_configuration_count",
            "joint_wins",
            "ties",
            "joint_losses",
            "win_rate",
            "median_improvement_percent",
            "mean_improvement_percent",
            "geomean_speedup",
            "p05_improvement_percent",
            "p25_improvement_percent",
            "p75_improvement_percent",
            "p95_improvement_percent",
            "minimum_improvement_percent",
            "maximum_improvement_percent",
            "bootstrap_mean_improvement_ci95_low",
            "bootstrap_mean_improvement_ci95_high",
        ],
    )

    breakdown_fields = [
        "B",
        "N",
        "W",
        "L",
        "B_over_W",
        "N_over_W",
        "L_over_N",
    ]
    breakdown_columns = [
        "breakdown",
        "value",
        "configurations",
        "joint_vs_partition_only_wins",
        "joint_vs_partition_only_ties",
        "joint_vs_partition_only_losses",
        "joint_vs_partition_only_missing",
        "joint_vs_partition_only_paired_configuration_count",
        "joint_vs_partition_only_win_rate",
        "joint_vs_partition_only_mean_improvement_percent",
        "joint_vs_partition_only_median_improvement_percent",
        "joint_vs_partition_only_max_improvement_percent",
        "joint_vs_schedule_only_wins",
        "joint_vs_schedule_only_ties",
        "joint_vs_schedule_only_losses",
        "joint_vs_schedule_only_missing",
        "joint_vs_schedule_only_paired_configuration_count",
        "joint_vs_schedule_only_win_rate",
        "joint_vs_schedule_only_mean_improvement_percent",
        "joint_vs_schedule_only_median_improvement_percent",
        "joint_vs_schedule_only_max_improvement_percent",
    ]
    for field in breakdown_fields:
        rows = breakdown_rows(paired, field)
        write_csv(output_dir / f"breakdown_by_{field}.csv", rows, breakdown_columns)
        write_csv(output_dir / f"by_{field}.csv", rows, breakdown_columns)

    write_csv(
        output_dir / "solver_status_and_gap_summary.csv",
        status_summary_rows(result_rows),
        [
            "optimization_mode",
            "solver_status",
            "rows",
            "mean_relative_optimality_gap",
            "max_relative_optimality_gap",
        ],
    )
    write_csv(
        output_dir / "solver_status_summary.csv",
        status_summary_rows(result_rows),
        [
            "optimization_mode",
            "solver_status",
            "rows",
            "mean_relative_optimality_gap",
            "max_relative_optimality_gap",
        ],
    )
    write_csv(
        output_dir / "optimality_gap_summary.csv",
        optimality_gap_rows(result_rows),
        [
            "optimization_mode",
            "rows",
            "proven_optimal_rows",
            "best_incumbent_rows",
            "mean_relative_optimality_gap",
            "median_relative_optimality_gap",
            "max_relative_optimality_gap",
        ],
    )
    failed_rows = [
        row
        for row in paired
        for mode in PRIMARY_MODES
        if row.get(f"{mode}_status") not in SUCCESS_STATUSES
    ]
    write_csv(output_dir / "missing_or_failed_configurations.csv", failed_rows, paired_fields)
    failed_result_rows = [
        row
        for row in result_rows.values()
        if str(row.get("solver_status", "UNKNOWN")) not in SUCCESS_STATUSES
    ]
    if failed_result_rows:
        write_csv(output_dir / "failed_unknown_rows.csv", failed_result_rows, sorted({key for row in failed_result_rows for key in row.keys()}))
    else:
        write_csv(output_dir / "failed_unknown_rows.csv", [], ["run_id", "solver_status", "failure"])

    for comparison, _ in COMPARISONS:
        write_heatmap_svg(paired, comparison, output_dir / f"heatmap_{comparison}.svg")
        write_sorted_improvement_svg(paired, comparison, output_dir / f"sorted_improvement_{comparison}.svg")
        write_cdf_svg(paired, comparison, output_dir / f"cdf_{comparison}.svg")
    write_normalized_heatmaps_pdf(paired, output_dir)
    write_sorted_improvements_pdf(paired, output_dir)
    write_improvement_cdfs_pdf(paired, output_dir)
    write_structural_breakdowns_pdf(paired, output_dir)
    write_solver_quality_pdf(result_rows, output_dir)

    (output_dir / "validation_summary.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    commands = [
        str(row.get("command", ""))
        for row in result_rows.values()
        if row.get("command")
    ]
    (output_dir / "commands.txt").write_text("\n".join(sorted(commands)) + "\n", encoding="utf-8")
    write_validation_report(output_dir, validation)
    write_table_tex(output_dir, aggregate)
    write_execution_summary(output_dir, validation, aggregate, result_rows, manifest_csv, results_jsonl)
    write_summary_md(output_dir, validation, wins)
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the strict primary SlackPipe structural ablation.")
    parser.add_argument("--results-jsonl", type=Path, default=Path("results/primary_ablation/results.jsonl"))
    parser.add_argument("--manifest-csv", type=Path, default=Path("results/primary_ablation/manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/primary_ablation_report"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation = analyze(args.results_jsonl, args.manifest_csv, args.output_dir)
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
