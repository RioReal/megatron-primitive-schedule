#!/usr/bin/env python3
"""Analyze manifest-backed SlackPipe CAL evaluation results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import cal_manifest


DEFAULT_NORMALIZATION_BASELINE = "uniform-breadth-first"

METHOD_ORDER = [
    "uniform-breadth-first",
    "uniform-interleaved-1f1b",
    "partition-only-fixed-order",
    "schedule-only-uniform",
    "sequential-partition-then-schedule",
    "alternating-partition-schedule",
    "joint-unrestricted-no-overlap",
]

CONVERGENCE_METHODS = [
    "joint-unrestricted-no-overlap",
    "alternating-partition-schedule",
]

METHOD_LABELS = {
    "uniform-breadth-first": "Uniform BF",
    "uniform-interleaved-1f1b": "Uniform 1F1B",
    "partition-only-fixed-order": "Partition-only",
    "schedule-only-uniform": "Schedule-only",
    "sequential-partition-then-schedule": "Sequential",
    "alternating-partition-schedule": "Alternating",
    "joint-unrestricted-no-overlap": "Joint",
}

CHECKPOINTS_SECONDS = [0.1, 1.0, 10.0, 30.0]

UTILIZATION_BACKING_METHOD_ORDER = [
    "uniform-breadth-first",
    "uniform-interleaved-1f1b",
    "partition-only-fixed-order",
    "schedule-only-uniform",
    "sequential-partition-then-schedule",
    "alternating-partition-schedule",
    "joint-unrestricted-no-overlap",
]

UTILIZATION_DEFAULT_PLOT_METHODS = [
    "uniform-breadth-first",
    "uniform-interleaved-1f1b",
    "partition-only-fixed-order",
    "schedule-only-uniform",
    "alternating-partition-schedule",
    "joint-unrestricted-no-overlap",
]

UTILIZATION_DISPLAY_LABELS = {
    "uniform-breadth-first": "Breadth-first",
    "uniform-interleaved-1f1b": "Interleaved 1F1B",
    "partition-only-fixed-order": "Partition-only",
    "schedule-only-uniform": "Schedule-only",
    "sequential-partition-then-schedule": "Sequential",
    "alternating-partition-schedule": "Alternating",
    "joint-unrestricted-no-overlap": "Joint",
}

UTILIZATION_DEFAULT_COMMUNICATION_PANELS = ["moderate", "heavy"]
UTILIZATION_DEFAULT_N_OVER_W = 2.0
UTILIZATION_ALL_N_OVER_W = "all"
UTILIZATION_ALL_N_OVER_W_VALUES = [1.0, 2.0, 4.0]
UTILIZATION_ALL_NW_REQUIRED_BATCH_SIZES = [0.5, 1.0, 2.0, 4.0, 8.0]
UTILIZATION_ALL_NW_METHOD_ORDER = [
    "uniform-breadth-first",
    "partition-only-fixed-order",
    "schedule-only-uniform",
    "joint-unrestricted-no-overlap",
]
UTILIZATION_ALL_NW_DISPLAY_LABELS = {
    "uniform-breadth-first": "BFS",
    "partition-only-fixed-order": "Partition-only",
    "schedule-only-uniform": "Schedule-only",
    "joint-unrestricted-no-overlap": "SlackPipe",
}
UTILIZATION_DEFAULT_STEM = "fig_slackpipe_utilization_vs_batch"
UTILIZATION_ALL_NW_STEM = "fig_slackpipe_utilization_vs_batch_all_nw"

SOLVER_MACHINERY_PRODUCTION_VARIANT = "production-slackpipe"
SOLVER_MACHINERY_CANONICAL_VARIANT = "canonical-incumbent"
SOLVER_MACHINERY_NO_HINTS_VARIANT = "no-incumbent-hints"
SOLVER_MACHINERY_NO_BOUND_VARIANT = "no-incumbent-bound"

BAD_CLASSES = {
    "missing_result",
    "invalid_result",
    "failed_process",
    "schema_mismatch",
    "manifest_mismatch",
    "method_contract_mismatch",
    "cap_ineligible",
    "validation_failed",
}


PROTOCOL_GUARD_FIELDS = [
    "manifest_hash",
    "git_commit_expected",
    "expected_budget_policy_version",
    "expected_evaluation_method_version",
    "expected_validation_version",
    "expected_activation_analysis_version",
    "expected_activation_cap_formulation_version",
]


@dataclass
class AnalysisOptions:
    manifest: Path
    results_root: Path
    output_dir: Path
    mode: str = "both"
    include_groups: set[str] | None = None
    strict: bool = True
    allow_unavailable: bool = False
    strict_convergence: bool = False
    formats: set[str] | None = None
    normalization_baseline: str = DEFAULT_NORMALIZATION_BASELINE
    emit_utilization_vs_batch: bool = False
    utilization_panel_n_over_w: float | str = UTILIZATION_DEFAULT_N_OVER_W
    utilization_communication_panels: list[str] | None = None
    utilization_include_sequential: bool = False
    utilization_normalize: str = "no"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def as_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    result = payload.get("canonical_result")
    return result if isinstance(result, Mapping) else {}


def result_path_for_row(row: Mapping[str, Any], results_root: Path) -> Path:
    return cal_manifest.resolve_output_path(results_root, str(row["output_json"]))


def selected_modes(mode: str) -> set[str]:
    if mode == "both":
        return {"uncapped", "equal_memory"}
    if mode == "uncapped":
        return {"uncapped"}
    if mode == "equal-memory":
        return {"equal_memory"}
    raise ValueError(f"unknown mode: {mode}")


def mode_for_row(row: Mapping[str, Any]) -> str:
    return "equal_memory" if row.get("activation_cap_mode") != "none" else "uncapped"


def stage_mapping_equal(row: Mapping[str, Any], c: Mapping[str, Any]) -> bool:
    return c.get("stage_to_worker_mapping") == row.get("stage_to_worker_mapping")


def equal_memory_eligible(row: Mapping[str, Any], c: Mapping[str, Any]) -> bool:
    if row.get("activation_cap_mode") == "none":
        return False
    if c.get("activation_cap_satisfied") is not True:
        return False
    if row.get("method") in cal_manifest.DETERMINISTIC_METHODS:
        return (
            c.get("activation_cap_enforcement_mode")
            == "deterministic_postconstruction_check"
        )
    if cal_manifest.oracle_exact_enumeration_cap_compatible(row, c):
        return c.get("feasible") is True and c.get("optimal") is True
    return (
        c.get("activation_cap_enforcement_mode") == "solver"
        and c.get("activation_cap_enforced_in_solver") is True
        and c.get("activation_cap_constraints_added") is True
    )


def classify_loaded_row(
    row: Mapping[str, Any], payload: Mapping[str, Any]
) -> dict[str, Any]:
    c = canonical(payload)
    classes: list[str] = []
    reasons: list[str] = []
    compat = cal_manifest.check_result_against_manifest(row, payload)

    if payload.get("runner_outcome") == "timeout_or_failed_process":
        classes.append("failed_process")
    if compat.outcome == "unavailable":
        classes.append("unavailable")
        reasons.extend(compat.reasons)
    elif not c:
        classes.append("invalid_result")
        reasons.append("canonical_result missing")
    elif compat.outcome == "schema_mismatch":
        classes.append("schema_mismatch")
        reasons.extend(compat.reasons)
    elif compat.outcome == "manifest_mismatch":
        classes.append("manifest_mismatch")
        reasons.extend(compat.reasons)
    elif not compat.passed:
        if any("method_contract_hash" in reason for reason in compat.reasons):
            classes.append("method_contract_mismatch")
        elif any("result_validation" in reason for reason in compat.reasons):
            classes.append("validation_failed")
        elif row.get("enforce_activation_cap") and any(
            token in reason
            for reason in compat.reasons
            for token in [
                "cap",
                "equal-memory",
                "posthoc_only",
                "enforced in solver",
            ]
        ):
            classes.append("cap_ineligible")
        else:
            classes.append("invalid_result")
        reasons.extend(compat.reasons)

    if c and not stage_mapping_equal(row, c):
        classes.append("manifest_mismatch")
        reasons.append("stage_to_worker_mapping mismatch")

    reported_status = c.get("reported_status")
    if reported_status == "INVALID_RESULT":
        classes.append("invalid_result")
    if c.get("result_validation_passed") is False:
        classes.append("validation_failed")
    if compat.outcome == "completed_no_solution":
        classes.append("no_solution")
    if c.get("fallback_used") is True:
        classes.append("fallback_used")
    if c.get("feasible") is True and c.get("optimal") is False:
        classes.append("not_optimal_but_feasible")
    if row.get("experiment_group") == "oracle" and c.get("optimal") is not True:
        classes.append("oracle_unproven")

    if compat.passed and c and compat.outcome == "completed_valid":
        if mode_for_row(row) == "uncapped":
            classes.append("valid_for_uncapped")
        else:
            if equal_memory_eligible(row, c):
                classes.append("valid_for_equal_memory")
            elif (
                cal_manifest.is_oracle_enumeration_row(row)
                and c.get("activation_cap_enforced_by_enumeration") is True
            ):
                classes.append("oracle_unproven")
            else:
                classes.append("cap_ineligible")
        if row.get("method") in cal_manifest.DETERMINISTIC_METHODS:
            classes.append("deterministic_baseline_valid")

    classes = sorted(set(classes))
    if not classes:
        classes = ["invalid_result"]
    primary = first_primary_class(classes)
    return {
        "row": row,
        "payload": payload,
        "canonical": c,
        "classes": classes,
        "primary_class": primary,
        "reasons": sorted(set(reasons)),
    }


def first_primary_class(classes: Sequence[str]) -> str:
    priority = [
        "missing_result",
        "failed_process",
        "schema_mismatch",
        "manifest_mismatch",
        "method_contract_mismatch",
        "validation_failed",
        "invalid_result",
        "unavailable",
        "cap_ineligible",
        "oracle_unproven",
        "no_solution",
        "valid_for_equal_memory",
        "valid_for_uncapped",
        "deterministic_baseline_valid",
        "fallback_used",
        "not_optimal_but_feasible",
    ]
    for item in priority:
        if item in classes:
            return item
    return classes[0]


def classify_manifest_rows(
    rows: Sequence[Mapping[str, Any]], results_root: Path
) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        path = result_path_for_row(row, results_root)
        if not path.exists():
            records.append(
                {
                    "row": row,
                    "payload": None,
                    "canonical": {},
                    "classes": ["missing_result"],
                    "primary_class": "missing_result",
                    "reasons": [f"missing result: {path}"],
                }
            )
            continue
        try:
            payload = load_json(path)
        except Exception as exc:
            records.append(
                {
                    "row": row,
                    "payload": None,
                    "canonical": {},
                    "classes": ["invalid_result"],
                    "primary_class": "invalid_result",
                    "reasons": [f"unreadable result: {exc}"],
                }
            )
            continue
        records.append(classify_loaded_row(row, payload))
    return records


def preflight_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        row = record["row"]
        c = record.get("canonical") or {}
        rows.append(
            {
                "run_id": row["run_id"],
                "experiment_group": row["experiment_group"],
                "configuration_id": row["configuration_id"],
                "method": row["method"],
                "analysis_mode": mode_for_row(row),
                "primary_class": record["primary_class"],
                "classes": ";".join(record["classes"]),
                "reported_status": c.get("reported_status"),
                "solver_status_raw": c.get("solver_status_raw"),
                "final_solution_available": c.get("final_solution_available"),
                "final_solution_source": c.get("final_solution_source"),
                "no_solution_reason": c.get("no_solution_reason"),
                "fallback_used": bool(c.get("fallback_used")),
                "optimal": c.get("optimal"),
                "makespan": c.get("makespan"),
                "reasons": " | ".join(record["reasons"]),
            }
        )
    return rows


def count_classes(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for cls in record["classes"]:
            counts[cls] = counts.get(cls, 0) + 1
    return counts


def protocol_guard_failures(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for field in PROTOCOL_GUARD_FIELDS:
        values = sorted({as_key(row.get(field)) for row in rows})
        if len(values) > 1:
            failures.append(
                {
                    "field": field,
                    "reason": f"mixed {field}",
                    "value_count": len(values),
                    "values": values,
                }
            )
    by_method: dict[str, set[str]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), set()).add(
            as_key(row.get("method_contract_hash_expected"))
        )
    for method, values in sorted(by_method.items()):
        if len(values) > 1:
            failures.append(
                {
                    "field": "method_contract_hash_expected",
                    "method": method,
                    "reason": "mixed method contract hash for method",
                    "value_count": len(values),
                    "values": sorted(values),
                }
            )
    return failures


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def geometric_mean(values: Iterable[float]) -> float:
    data = [float(value) for value in values if value is not None]
    if not data:
        raise ValueError("geometric mean requires at least one value")
    if any(value <= 0 or not math.isfinite(value) for value in data):
        raise ValueError("geometric mean requires positive finite values")
    return math.exp(sum(math.log(value) for value in data) / len(data))


def median(values: Sequence[float]) -> float:
    return float(statistics.median([float(value) for value in values]))


def workload_key(row: Mapping[str, Any]) -> str:
    payload = {
        "B": row["B"],
        "N": row["N"],
        "W": row["W"],
        "L": row["L"],
        "mapping_type": row["mapping_type"],
        "stage_to_worker_mapping": row["stage_to_worker_mapping"],
        "forward_cost_ratio_num": row["forward_cost_ratio_num"],
        "forward_cost_ratio_den": row["forward_cost_ratio_den"],
        "backward_cost_ratio_num": row["backward_cost_ratio_num"],
        "backward_cost_ratio_den": row["backward_cost_ratio_den"],
        "communication_profile": row["communication_profile"],
        "communication_ticks": row["communication_ticks"],
        "activation_model": row["activation_model"],
        "activation_cap_mode": row["activation_cap_mode"],
        "activation_cap_units_per_worker": row.get("activation_cap_units_per_worker"),
        "activation_cap_derivation_hash": row.get("activation_cap_derivation_hash"),
        "enforce_activation_cap": row.get("enforce_activation_cap"),
    }
    return as_key(payload)


def valid_records_for_mode(
    records: Sequence[Mapping[str, Any]], analysis_mode: str
) -> list[Mapping[str, Any]]:
    required_class = (
        "valid_for_equal_memory" if analysis_mode == "equal_memory" else "valid_for_uncapped"
    )
    return [record for record in records if required_class in record["classes"]]


def makespan_from_record(record: Mapping[str, Any]) -> float | None:
    value = (record.get("canonical") or {}).get("makespan")
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    return value if value > 0 else None


def numeric_field_from_row_or_result(
    row: Mapping[str, Any],
    c: Mapping[str, Any],
    row_field: str,
    result_field: str,
) -> float | None:
    for value in (row.get(row_field), c.get(result_field)):
        if value is None:
            continue
        try:
            parsed = float(value)
        except Exception:
            continue
        if math.isfinite(parsed) and parsed > 0:
            return parsed
    return None


def ratio_from_row(
    row: Mapping[str, Any], numerator_field: str, denominator_field: str
) -> float | None:
    try:
        numerator = float(row[numerator_field])
        denominator = float(row[denominator_field])
    except Exception:
        return None
    if denominator <= 0 or not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    return numerator / denominator


def utilization_is_all_nw(panel_n_over_w: float | str) -> bool:
    return str(panel_n_over_w).strip().lower() == UTILIZATION_ALL_N_OVER_W


def parse_utilization_n_over_w(panel_n_over_w: float | str) -> float | str:
    if utilization_is_all_nw(panel_n_over_w):
        return UTILIZATION_ALL_N_OVER_W
    try:
        value = float(panel_n_over_w)
    except Exception as exc:
        raise ValueError(
            "--utilization-panel-n-over-w must be a positive number or 'all'"
        ) from exc
    if value <= 0 or not math.isfinite(value):
        raise ValueError(
            "--utilization-panel-n-over-w must be a positive number or 'all'"
        )
    return value


def utilization_allowed_n_over_w_values(
    panel_n_over_w: float | str,
) -> list[float]:
    parsed = parse_utilization_n_over_w(panel_n_over_w)
    if parsed == UTILIZATION_ALL_N_OVER_W:
        return list(UTILIZATION_ALL_N_OVER_W_VALUES)
    return [float(parsed)]


def utilization_output_stem(panel_n_over_w: float | str) -> str:
    if utilization_is_all_nw(parse_utilization_n_over_w(panel_n_over_w)):
        return UTILIZATION_ALL_NW_STEM
    return UTILIZATION_DEFAULT_STEM


def utilization_method_order(panel_n_over_w: float | str) -> list[str]:
    if utilization_is_all_nw(parse_utilization_n_over_w(panel_n_over_w)):
        return list(UTILIZATION_ALL_NW_METHOD_ORDER)
    return list(UTILIZATION_BACKING_METHOD_ORDER)


def utilization_display_label(method: str, panel_n_over_w: float | str) -> str:
    if utilization_is_all_nw(parse_utilization_n_over_w(panel_n_over_w)):
        return UTILIZATION_ALL_NW_DISPLAY_LABELS.get(method, method)
    return UTILIZATION_DISPLAY_LABELS.get(method, method)


def utilization_plot_methods(
    include_sequential: bool, panel_n_over_w: float | str
) -> list[str]:
    if utilization_is_all_nw(parse_utilization_n_over_w(panel_n_over_w)):
        return list(UTILIZATION_ALL_NW_METHOD_ORDER)
    if include_sequential:
        return list(UTILIZATION_BACKING_METHOD_ORDER)
    return list(UTILIZATION_DEFAULT_PLOT_METHODS)


def utilization_workload_key(
    row: Mapping[str, Any],
    *,
    B: float,
    N: float,
    W: float,
    L: float,
) -> str:
    return as_key(
        {
            "B": B,
            "N": N,
            "W": W,
            "L": L,
            "mapping_type": row.get("mapping_type"),
            "stage_to_worker_mapping": row.get("stage_to_worker_mapping"),
            "forward_cost_ratio_num": row.get("forward_cost_ratio_num"),
            "forward_cost_ratio_den": row.get("forward_cost_ratio_den"),
            "backward_cost_ratio_num": row.get("backward_cost_ratio_num"),
            "backward_cost_ratio_den": row.get("backward_cost_ratio_den"),
            "communication_profile": row.get("communication_profile"),
            "communication_ticks": row.get("communication_ticks"),
            "activation_model": row.get("activation_model"),
            "activation_cap_mode": "none",
            "enforce_activation_cap": False,
        }
    )


def utilization_record_row(
    record: Mapping[str, Any],
    *,
    panel_n_over_w: float | str,
    allowed_n_over_w_values: Sequence[float],
    communication_panels: Sequence[str],
    method_order: Sequence[str],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    row = record["row"]
    c = record.get("canonical") or {}
    errors = []
    method = str(row.get("method"))
    if method not in set(method_order):
        return None, []

    makespan = makespan_from_record(record)
    if makespan is None:
        errors.append(
            {
                "run_id": row.get("run_id"),
                "method": method,
                "reason": "missing or non-positive makespan",
            }
        )

    B = numeric_field_from_row_or_result(row, c, "B", "micro_batches")
    N = numeric_field_from_row_or_result(row, c, "N", "logical_stages")
    W = numeric_field_from_row_or_result(row, c, "W", "physical_workers")
    L = numeric_field_from_row_or_result(row, c, "L", "total_layers")
    missing_fields = [
        name
        for name, value in [("B", B), ("N", N), ("W", W), ("L", L)]
        if value is None
    ]
    if missing_fields:
        errors.append(
            {
                "run_id": row.get("run_id"),
                "method": method,
                "reason": "missing or non-positive workload field",
                "fields": ",".join(missing_fields),
            }
        )
    forward_cost = ratio_from_row(row, "forward_cost_ratio_num", "forward_cost_ratio_den")
    backward_cost = ratio_from_row(
        row, "backward_cost_ratio_num", "backward_cost_ratio_den"
    )
    if forward_cost is None or backward_cost is None:
        errors.append(
            {
                "run_id": row.get("run_id"),
                "method": method,
                "reason": "missing or invalid compute cost ratio",
            }
        )
    if errors:
        return None, errors

    assert B is not None
    assert N is not None
    assert W is not None
    assert L is not None
    assert makespan is not None
    assert forward_cost is not None
    assert backward_cost is not None
    n_over_w = N / W
    communication_profile = str(row.get("communication_profile"))
    if not any(
        math.isclose(n_over_w, value, rel_tol=1e-9, abs_tol=1e-9)
        for value in allowed_n_over_w_values
    ):
        return None, []
    if communication_profile not in set(communication_panels):
        return None, []

    useful_compute = B * L * (forward_cost + backward_cost)
    utilization_percent = 100.0 * useful_compute / (W * makespan)
    return (
        {
            "analysis_mode": "uncapped",
            "run_id": row["run_id"],
            "workload_key": utilization_workload_key(
                row, B=B, N=N, W=W, L=L
            ),
            "configuration_id": row["configuration_id"],
            "communication_profile": communication_profile,
            "communication_ticks": row["communication_ticks"],
            "B": B,
            "N": N,
            "W": W,
            "L": L,
            "n_over_w": n_over_w,
            "batch_size_per_worker": B / W,
            "method": method,
            "method_label": utilization_display_label(method, panel_n_over_w),
            "random_seed": row["random_seed"],
            "makespan": makespan,
            "forward_cost_per_layer": forward_cost,
            "backward_cost_per_layer": backward_cost,
            "useful_compute": useful_compute,
            "utilization_percent": utilization_percent,
        },
        [],
    )


def aggregate_utilization_vs_batch(
    records: Sequence[Mapping[str, Any]],
    *,
    panel_n_over_w: float | str = UTILIZATION_DEFAULT_N_OVER_W,
    communication_panels: Sequence[str] | None = None,
    include_sequential: bool = False,
    strict: bool = False,
    require_complete_profile: bool = False,
) -> dict[str, Any]:
    parsed_n_over_w = parse_utilization_n_over_w(panel_n_over_w)
    all_nw = parsed_n_over_w == UTILIZATION_ALL_N_OVER_W
    allowed_n_over_w_values = utilization_allowed_n_over_w_values(parsed_n_over_w)
    method_order = utilization_method_order(parsed_n_over_w)
    panels = list(communication_panels or UTILIZATION_DEFAULT_COMMUNICATION_PANELS)
    run_rows = []
    errors = []
    for record in valid_records_for_mode(records, "uncapped"):
        if (record.get("payload") or {}).get("runner_outcome") != "completed_valid":
            continue
        row, row_errors = utilization_record_row(
            record,
            panel_n_over_w=parsed_n_over_w,
            allowed_n_over_w_values=allowed_n_over_w_values,
            communication_panels=panels,
            method_order=method_order,
        )
        errors.extend(row_errors)
        if row is not None:
            run_rows.append(row)
    if strict and errors:
        raise RuntimeError(
            "utilization-vs-batch preflight failed: "
            + json.dumps(errors, sort_keys=True)
        )

    by_workload_method: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in run_rows:
        by_workload_method.setdefault((row["workload_key"], row["method"]), []).append(row)

    workload_rows = []
    for (key, method), rows in sorted(by_workload_method.items()):
        values = [float(row["utilization_percent"]) for row in rows]
        row0 = rows[0]
        workload_rows.append(
            {
                "analysis_mode": "uncapped",
                "workload_key": key,
                "configuration_id": row0["configuration_id"],
                "communication_profile": row0["communication_profile"],
                "communication_ticks": row0["communication_ticks"],
                "B": row0["B"],
                "N": row0["N"],
                "W": row0["W"],
                "L": row0["L"],
                "n_over_w": row0["n_over_w"],
                "batch_size_per_worker": row0["batch_size_per_worker"],
                "method": method,
                "method_label": utilization_display_label(method, parsed_n_over_w),
                "seed_count": len(rows),
                "seed_aggregate": "median",
                "utilization_percent": median(values),
                "utilization_percent_best_seed": max(values),
                "utilization_percent_worst_seed": min(values),
            }
        )

    plot_methods = utilization_plot_methods(include_sequential, parsed_n_over_w)
    aggregate_grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = {}
    for row in workload_rows:
        aggregate_grouped.setdefault(
            (
                str(row["communication_profile"]),
                float(row["batch_size_per_worker"]),
                str(row["method"]),
            ),
            [],
        ).append(row)

    method_rank = {method: index for index, method in enumerate(method_order)}
    panel_rank = {panel: index for index, panel in enumerate(panels)}
    aggregate_rows = []
    for (panel, batch_size, method), rows in sorted(
        aggregate_grouped.items(),
        key=lambda item: (
            panel_rank.get(item[0][0], len(panel_rank)),
            item[0][1],
            method_rank.get(item[0][2], len(method_rank)),
        ),
    ):
        values = [float(row["utilization_percent"]) for row in rows]
        all_positive = all(value > 0 and math.isfinite(value) for value in values)
        aggregate_rule = "geometric_mean" if all_positive else "arithmetic_mean"
        aggregate_value = (
            geometric_mean(values) if all_positive else sum(values) / len(values)
        )
        aggregate_rows.append(
            {
                "analysis_mode": "uncapped",
                "communication_profile": panel,
                "batch_size_per_worker": batch_size,
                "method": method,
                "method_label": utilization_display_label(
                    method, parsed_n_over_w
                ),
                "plotted": method in plot_methods,
                "workload_count": len(rows),
                "workload_aggregate": aggregate_rule,
                "utilization_percent": aggregate_value,
                "utilization_percent_min_workload": min(values),
                "utilization_percent_max_workload": max(values),
                "n_over_w_filter": parsed_n_over_w,
                "n_over_w_values": allowed_n_over_w_values,
            }
        )

    strict_failures = []
    if require_complete_profile or (strict and all_nw):
        required = set(plot_methods)
        for panel in panels:
            panel_rows = [
                row
                for row in aggregate_rows
                if row["communication_profile"] == panel and row["plotted"]
            ]
            if not panel_rows:
                strict_failures.append(
                    {
                        "communication_profile": panel,
                        "reason": "missing required utilization panel",
                    }
                )
                continue
            present = {str(row["method"]) for row in panel_rows}
            missing = sorted(required - present)
            if missing:
                strict_failures.append(
                    {
                        "communication_profile": panel,
                        "reason": "missing required utilization methods",
                        "missing_methods": missing,
                    }
                )
            if all_nw:
                present_batches = [
                    float(row["batch_size_per_worker"])
                    for row in panel_rows
                ]
                missing_batches = [
                    value
                    for value in UTILIZATION_ALL_NW_REQUIRED_BATCH_SIZES
                    if not any(
                        math.isclose(
                            value, present, rel_tol=1e-9, abs_tol=1e-9
                        )
                        for present in present_batches
                    )
                ]
                if missing_batches:
                    strict_failures.append(
                        {
                            "communication_profile": panel,
                            "reason": "missing required utilization batch sizes",
                            "missing_batch_size_per_worker": missing_batches,
                        }
                    )
    if strict and strict_failures:
        raise RuntimeError(
            "utilization-vs-batch required data missing: "
            + json.dumps(strict_failures, sort_keys=True)
        )

    return {
        "utilization_figure_status": "available" if aggregate_rows else "no_data",
        "metric": "simulated_compute_utilization_percent",
        "formula": "100 * B * L * (forward_cost_per_layer + backward_cost_per_layer) / (W * makespan)",
        "normalization": "none",
        "panel_n_over_w": parsed_n_over_w,
        "included_n_over_w_values": allowed_n_over_w_values,
        "communication_panels": panels,
        "plot_methods": plot_methods,
        "backing_methods": list(method_order),
        "output_stem": utilization_output_stem(parsed_n_over_w),
        "strict_failures": strict_failures,
        "invalid_rows": errors,
        "run_rows": run_rows,
        "workload_rows": workload_rows,
        "aggregate_rows": aggregate_rows,
    }


def plot_utilization_vs_batch(
    payload: Mapping[str, Any], output_dir: Path, formats: set[str]
) -> None:
    if "pdf" not in formats:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = list(payload.get("communication_panels") or [])
    if not panels:
        panels = UTILIZATION_DEFAULT_COMMUNICATION_PANELS
    rows = list(payload.get("aggregate_rows") or [])
    plot_methods = list(payload.get("plot_methods") or UTILIZATION_DEFAULT_PLOT_METHODS)
    output_stem = str(payload.get("output_stem") or UTILIZATION_DEFAULT_STEM)
    all_nw = output_stem == UTILIZATION_ALL_NW_STEM
    markers = ["o", "s", "^", "D", "P", "v", "X"]
    linestyles = ["-", "--", "-.", ":", "-", "--", "-."]
    method_order = list(payload.get("backing_methods") or UTILIZATION_BACKING_METHOD_ORDER)
    style = {
        method: (markers[index % len(markers)], linestyles[index % len(linestyles)])
        for index, method in enumerate(method_order)
    }

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )
    fig, axes = plt.subplots(
        1, len(panels), figsize=(3.25 * len(panels), 2.45), squeeze=False
    )
    handles = []
    labels = []
    for panel_index, (ax, panel) in enumerate(zip(axes[0], panels)):
        panel_rows = [row for row in rows if row["communication_profile"] == panel]
        for method in plot_methods:
            data = [
                row
                for row in panel_rows
                if row["method"] == method and row.get("plotted") is True
            ]
            if not data:
                continue
            data.sort(key=lambda row: float(row["batch_size_per_worker"]))
            marker, linestyle = style.get(method, ("o", "-"))
            (line,) = ax.plot(
                [float(row["batch_size_per_worker"]) for row in data],
                [float(row["utilization_percent"]) for row in data],
                marker=marker,
                linestyle=linestyle,
                linewidth=1.2,
                markersize=3.5,
                label=data[0].get("method_label", method),
            )
            if method not in labels:
                handles.append(line)
                labels.append(method)
        if not panel_rows:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#666666",
            )
        ax.set_title(f"({chr(ord('a') + panel_index)}) {panel}")
        ax.set_xlabel("Batch size / worker")
        if panel_index == 0:
            ax.set_ylabel("Simulated compute utilization (%)")
        if all_nw:
            ax.set_xscale("log", base=2)
            ax.set_xticks(UTILIZATION_ALL_NW_REQUIRED_BATCH_SIZES)
            ax.set_xticklabels(["1/2", "1", "2", "4", "8"])
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.45)
    if handles:
        fig.legend(
            handles,
            [
                next(
                    (
                        row.get("method_label", method)
                        for row in rows
                        if row.get("method") == method
                    ),
                    method,
                )
                for method in labels
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.04),
            ncol=min(len(handles), 3),
            frameon=False,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.94 if handles else 1))
    fig.savefig(output_dir / f"{output_stem}.pdf")
    plt.close(fig)


def tex_number(value: float) -> str:
    if math.isclose(value, round(value), rel_tol=1e-9, abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:.6g}"


def tex_coordinates(rows: Sequence[Mapping[str, Any]]) -> str:
    return " ".join(
        f"({tex_number(float(row['batch_size_per_worker']))},"
        f"{tex_number(float(row['utilization_percent']))})"
        for row in rows
    )


def write_utilization_all_nw_tex(payload: Mapping[str, Any], path: Path) -> None:
    rows = list(payload.get("aggregate_rows") or [])
    panels = list(payload.get("communication_panels") or [])
    plot_methods = list(payload.get("plot_methods") or [])
    styles = ["mark=*", "mark=square*", "mark=triangle*", "mark=diamond*"]
    title_map = {
        "moderate": "(a) Moderate communication",
        "heavy": "(b) Heavy communication",
    }
    lines = [
        "% Generated by scripts/analyze_cal_results.py.",
        "% Simulated compute utilization, not hardware GPU utilization.",
        "% xticks: {1/2,1,2,4,8}",
        "\\begin{tikzpicture}",
        "\\begin{groupplot}[",
        "  group style={group size=2 by 1, horizontal sep=1.1cm},",
        "  width=0.48\\linewidth,",
        "  height=0.34\\linewidth,",
        "  xmode=log,",
        "  log basis x={2},",
        "  xmin=0.45, xmax=8.9,",
        "  xtick={0.5,1,2,4,8},",
        "  xticklabels={{1/2},{1},{2},{4},{8}},",
        "  xlabel={Batch size / worker},",
        "  ylabel={Simulated compute utilization (\\%)},",
        "  grid=both,",
        "  grid style={dotted,gray!40},",
        "  tick label style={font=\\scriptsize},",
        "  label style={font=\\scriptsize},",
        "  title style={font=\\scriptsize},",
        "  legend style={font=\\scriptsize,draw=none},",
        "]",
    ]
    for panel_index, panel in enumerate(panels):
        title = title_map.get(str(panel), f"({chr(ord('a') + panel_index)}) {panel}")
        legend_entries = []
        lines.append(f"\\nextgroupplot[title={{{title}}}]")
        for method_index, method in enumerate(plot_methods):
            data = [
                row
                for row in rows
                if row.get("communication_profile") == panel
                and row.get("method") == method
                and row.get("plotted") is True
            ]
            if not data:
                continue
            data.sort(key=lambda row: float(row["batch_size_per_worker"]))
            label = str(data[0].get("method_label") or method)
            style = styles[method_index % len(styles)]
            lines.append(f"\\addplot+[{style}] coordinates {{{tex_coordinates(data)}}};")
            legend_entries.append(label)
        if panel_index == 0 and legend_entries:
            joined = ",".join(legend_entries)
            lines.append(f"\\legend{{{joined}}}")
    lines.extend(["\\end{groupplot}", "\\end{tikzpicture}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_utilization_vs_batch_outputs(
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    formats: set[str],
    *,
    panel_n_over_w: float = 2.0,
    communication_panels: Sequence[str] | None = None,
    include_sequential: bool = False,
    strict: bool = False,
    require_complete_profile: bool = False,
) -> dict[str, Any]:
    payload = aggregate_utilization_vs_batch(
        records,
        panel_n_over_w=panel_n_over_w,
        communication_panels=communication_panels,
        include_sequential=include_sequential,
        strict=strict,
        require_complete_profile=require_complete_profile,
    )
    output_stem = str(payload.get("output_stem") or UTILIZATION_DEFAULT_STEM)
    if "csv" in formats:
        write_csv(output_dir / f"{output_stem}.csv", payload["aggregate_rows"])
    if "json" in formats:
        write_json(output_dir / f"{output_stem}.json", payload)
    if output_stem == UTILIZATION_ALL_NW_STEM and "tex" in formats:
        write_utilization_all_nw_tex(payload, output_dir / f"{output_stem}.tex")
    plot_utilization_vs_batch(payload, output_dir, formats)
    return payload


def ordered_methods_present(records: Sequence[Mapping[str, Any]]) -> list[str]:
    present = {str(record["row"]["method"]) for record in records}
    ordered = [method for method in METHOD_ORDER if method in present]
    extras = sorted(present - set(METHOD_ORDER))
    return ordered + extras


def summary_baseline_method(summary: Mapping[str, Any]) -> str:
    return str(summary.get("normalization_baseline_method") or DEFAULT_NORMALIZATION_BASELINE)


def baseline_comparison_summary(
    records: Sequence[Mapping[str, Any]], analysis_mode: str
) -> dict[str, Any]:
    valid = valid_records_for_mode(records, analysis_mode)
    by_workload: dict[str, list[Mapping[str, Any]]] = {}
    for record in valid:
        by_workload.setdefault(workload_key(record["row"]), []).append(record)
    ratios = []
    strict_1f1b = 0
    strict_breadth_first = 0
    tied = 0
    for key, workload_records in by_workload.items():
        del key
        by_method: dict[str, list[float]] = {}
        for record in workload_records:
            makespan = makespan_from_record(record)
            if makespan is None:
                continue
            by_method.setdefault(str(record["row"]["method"]), []).append(makespan)
        breadth_first = by_method.get("uniform-breadth-first")
        one_f_one_b = by_method.get("uniform-interleaved-1f1b")
        if not breadth_first or not one_f_one_b:
            continue
        bf_median = median(breadth_first)
        f1b_median = median(one_f_one_b)
        if f1b_median <= 0:
            continue
        ratio = bf_median / f1b_median
        ratios.append(ratio)
        if f1b_median + 1e-9 < bf_median:
            strict_1f1b += 1
        elif bf_median + 1e-9 < f1b_median:
            strict_breadth_first += 1
        else:
            tied += 1
    if not ratios:
        return {"status": "unavailable"}
    return {
        "status": "available",
        "workload_count": len(ratios),
        "geomean_breadth_first_makespan_over_1f1b": geometric_mean(ratios),
        "one_f_one_b_strictly_better_than_breadth_first_workloads": strict_1f1b,
        "breadth_first_strictly_better_than_1f1b_workloads": strict_breadth_first,
        "tied_workloads": tied,
    }


def aggregate_normalized_makespan(
    records: Sequence[Mapping[str, Any]],
    analysis_mode: str,
    normalization_baseline: str = DEFAULT_NORMALIZATION_BASELINE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid = valid_records_for_mode(records, analysis_mode)
    by_workload: dict[str, list[Mapping[str, Any]]] = {}
    for record in valid:
        by_workload.setdefault(workload_key(record["row"]), []).append(record)

    raw_rows: list[dict[str, Any]] = []
    methods_present = ordered_methods_present(valid)
    method_values: dict[str, list[float]] = {
        method: [] for method in methods_present if method != normalization_baseline
    }
    joint_losses: list[dict[str, Any]] = []
    max_joint_improvement: dict[str, Any] | None = None
    win_counts: dict[str, int] = {method: 0 for method in methods_present}
    complete_workloads = 0

    for key, workload_records in sorted(by_workload.items()):
        baseline_records = [
            record
            for record in workload_records
            if record["row"]["method"] == normalization_baseline
        ]
        baseline_makespans = [
            makespan
            for makespan in (makespan_from_record(record) for record in baseline_records)
            if makespan is not None
        ]
        if not baseline_makespans:
            continue
        baseline = median(baseline_makespans)
        method_to_seed_norms: dict[str, list[tuple[int, float, float]]] = {}
        workload_method_norms: dict[str, float] = {
            normalization_baseline: 1.0,
        }
        for record in workload_records:
            method = record["row"]["method"]
            if method == normalization_baseline:
                continue
            makespan = makespan_from_record(record)
            if makespan is None:
                continue
            method_to_seed_norms.setdefault(method, []).append(
                (int(record["row"]["random_seed"]), makespan / baseline, makespan)
            )
        if not method_to_seed_norms:
            continue
        row0 = workload_records[0]["row"]
        workload_methods_seen = 1
        for method in methods_present:
            if method == normalization_baseline:
                continue
            values = method_to_seed_norms.get(method)
            if not values:
                continue
            workload_methods_seen += 1
            norms = [item[1] for item in values]
            makespans = [item[2] for item in values]
            norm_median = median(norms)
            makespan_median = median(makespans)
            method_values[method].append(norm_median)
            raw = {
                "analysis_mode": analysis_mode,
                "workload_key": key,
                "configuration_id": row0["configuration_id"],
                "B": row0["B"],
                "N": row0["N"],
                "W": row0["W"],
                "L": row0["L"],
                "stage_multiplier": float(row0["N"]) / float(row0["W"]),
                "batch_ratio": float(row0["B"]) / float(row0["N"]),
                "communication_profile": row0["communication_profile"],
                "communication_ticks": row0["communication_ticks"],
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "seed_count": len(values),
                "seed_aggregate": "median",
                "normalization_baseline_method": normalization_baseline,
                "normalization_baseline_label": METHOD_LABELS.get(
                    normalization_baseline, normalization_baseline
                ),
                "baseline_makespan": baseline,
                "makespan_median": makespan_median,
                "normalized_makespan_median": norm_median,
                "normalized_makespan_best_seed": min(norms),
                "normalized_makespan_worst_seed": max(norms),
            }
            raw_rows.append(raw)
            workload_method_norms[method] = norm_median
        if workload_method_norms:
            workload_best_norm = min(workload_method_norms.values())
            winning_methods = [
                method
                for method, value in workload_method_norms.items()
                if abs(value - workload_best_norm) <= 1e-9
            ]
            if len(winning_methods) == 1:
                win_counts[winning_methods[0]] += 1
        else:
            workload_best_norm = None
            winning_methods = []
        joint = next(
            (
                row
                for row in raw_rows
                if row["workload_key"] == key
                and row["method"] == "joint-unrestricted-no-overlap"
            ),
            None,
        )
        if joint is not None:
            if joint["normalized_makespan_median"] > 0:
                improvement = 1.0 / joint["normalized_makespan_median"]
                if (
                    max_joint_improvement is None
                    or improvement
                    > max_joint_improvement["improvement_over_baseline"]
                ):
                    max_joint_improvement = {
                        "configuration_id": row0["configuration_id"],
                        "communication_profile": row0["communication_profile"],
                        "normalized_makespan": joint["normalized_makespan_median"],
                        "improvement_over_baseline": improvement,
                        "improvement_over_uniform": improvement
                        if normalization_baseline == "uniform-breadth-first"
                        else None,
                    }
            if (
                workload_best_norm is not None
                and joint["normalized_makespan_median"] > workload_best_norm + 1e-9
            ):
                best_method = min(
                    workload_method_norms,
                    key=lambda method: workload_method_norms[method],
                )
                joint_losses.append(
                    {
                        "configuration_id": row0["configuration_id"],
                        "best_method": best_method,
                        "best_normalized_makespan": workload_best_norm,
                        "joint_normalized_makespan": joint[
                            "normalized_makespan_median"
                        ],
                    }
                )
        if workload_methods_seen == len(methods_present):
            complete_workloads += 1

    method_summary = {}
    for method, values in method_values.items():
        if not values:
            method_summary[method] = {"status": "missing"}
            continue
        gmean = geometric_mean(values)
        sorted_values = sorted(values)
        q1 = sorted_values[len(sorted_values) // 4]
        q3 = sorted_values[(3 * len(sorted_values)) // 4]
        method_summary[method] = {
            "method_label": METHOD_LABELS.get(method, method),
            "workload_count": len(values),
            "geomean_normalized_makespan": gmean,
            "geomean_improvement_over_baseline": 1.0 / gmean,
            "geomean_improvement_over_uniform": (1.0 / gmean)
            if normalization_baseline == "uniform-breadth-first"
            else None,
            "min_normalized_makespan": min(values),
            "max_normalized_makespan": max(values),
            "iqr_low": q1,
            "iqr_high": q3,
        }

    joint_gmean = method_summary.get("joint-unrestricted-no-overlap", {}).get(
        "geomean_normalized_makespan"
    )
    joint_over_methods = {}
    if joint_gmean:
        joint_over_methods[normalization_baseline] = 1.0 / joint_gmean
        for method in methods_present:
            if method == "joint-unrestricted-no-overlap":
                continue
            other = method_summary.get(method, {}).get("geomean_normalized_makespan")
            if other:
                joint_over_methods[method] = other / joint_gmean

    summary = {
        "analysis_mode": analysis_mode,
        "normalization_baseline_method": normalization_baseline,
        "normalization_baseline_label": METHOD_LABELS.get(
            normalization_baseline, normalization_baseline
        ),
        "seed_aggregate": "median",
        "workload_count_with_any_method": len(by_workload),
        "complete_workload_count": complete_workloads,
        "method_summary": method_summary,
        "joint_geomean_improvement_over_methods": joint_over_methods,
        "workload_win_counts": win_counts,
        "joint_loss_workloads": joint_losses,
        "maximum_joint_improvement_over_baseline": max_joint_improvement,
        "maximum_joint_improvement_over_uniform": max_joint_improvement
        if normalization_baseline == "uniform-breadth-first"
        else None,
        "breadth_first_vs_interleaved_1f1b": baseline_comparison_summary(
            records, analysis_mode
        ),
    }
    return raw_rows, summary


def normalization_precondition_failures(
    records: Sequence[Mapping[str, Any]],
    modes: Sequence[str],
    normalization_baseline: str = DEFAULT_NORMALIZATION_BASELINE,
) -> list[dict[str, Any]]:
    failures = []
    for mode in modes:
        mode_records = [
            record for record in records if mode_for_row(record["row"]) == mode
        ]
        if not mode_records:
            continue
        selected_keys = {workload_key(record["row"]) for record in mode_records}
        valid = valid_records_for_mode(records, mode)
        baseline_keys = {
            workload_key(record["row"])
            for record in valid
            if record["row"]["method"] == normalization_baseline
            and makespan_from_record(record) is not None
        }
        missing = sorted(selected_keys - baseline_keys)
        if missing:
            failures.append(
                {
                    "analysis_mode": mode,
                    "reason": f"missing valid {normalization_baseline} baseline",
                    "normalization_baseline_method": normalization_baseline,
                    "missing_workload_count": len(missing),
                    "missing_workload_keys": missing,
                }
            )
    return failures


def is_valid_solution_record(record: Mapping[str, Any]) -> bool:
    return (
        "valid_for_uncapped" in record["classes"]
        or "valid_for_equal_memory" in record["classes"]
    )


def is_no_solution_record(record: Mapping[str, Any]) -> bool:
    return "no_solution" in record["classes"]


def solver_machinery_variant_id(record: Mapping[str, Any]) -> str | None:
    value = record["row"].get("solver_machinery_variant")
    return str(value) if value else None


def solver_machinery_variant_label(variant_id: str) -> str:
    for variant in cal_manifest.SOLVER_MACHINERY_ABLATION_VARIANTS:
        if variant.variant_id == variant_id:
            return variant.label
    return variant_id


def solver_machinery_variant_order(records: Sequence[Mapping[str, Any]]) -> list[str]:
    present = {
        str(record["row"].get("solver_machinery_variant"))
        for record in records
        if record["row"].get("solver_machinery_variant")
    }
    ordered = [
        variant.variant_id
        for variant in cal_manifest.SOLVER_MACHINERY_ABLATION_VARIANTS
        if variant.variant_id in present
    ]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def solver_machinery_pair_key(row: Mapping[str, Any]) -> str:
    return as_key(
        {
            "W": row.get("W"),
            "N": row.get("N"),
            "B": row.get("B"),
            "L": row.get("L"),
            "communication_profile": row.get("communication_profile"),
            "activation_cap_mode": row.get("activation_cap_mode"),
            "random_seed": row.get("random_seed"),
        }
    )


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if math.isfinite(parsed) else None


def finite_positive_float(value: Any) -> float | None:
    parsed = finite_float(value)
    if parsed is None or parsed <= 0:
        return None
    return parsed


def numeric_from_record(
    record: Mapping[str, Any], fields: Sequence[str], *, positive: bool = False
) -> float | None:
    c = record.get("canonical") or {}
    payload = record.get("payload") or {}
    parser = finite_positive_float if positive else finite_float
    for source in (c, payload):
        if not isinstance(source, Mapping):
            continue
        for field in fields:
            value = parser(source.get(field))
            if value is not None:
                return value
    return None


def bool_from_record(record: Mapping[str, Any], fields: Sequence[str]) -> bool | None:
    c = record.get("canonical") or {}
    payload = record.get("payload") or {}
    for source in (c, payload):
        if not isinstance(source, Mapping):
            continue
        for field in fields:
            value = source.get(field)
            if isinstance(value, bool):
                return value
    return None


def solver_gap_from_record(record: Mapping[str, Any]) -> float | None:
    explicit = numeric_from_record(
        record, ["relative_optimality_gap", "optimality_gap"]
    )
    if explicit is not None:
        return explicit
    objective = numeric_from_record(
        record, ["best_objective", "solver_objective_ticks"], positive=True
    )
    bound = numeric_from_record(
        record, ["best_bound", "best_bound_ticks", "best_objective_bound"],
        positive=True,
    )
    if objective is None or bound is None or objective <= 0:
        return None
    return max(0.0, (objective - bound) / objective)


def solver_machinery_record_state(record: Mapping[str, Any]) -> str:
    if is_valid_solution_record(record):
        return "completed_valid"
    if is_no_solution_record(record):
        return "completed_no_solution"
    return str(record["primary_class"])


def solver_machinery_records_by_pair(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    by_pair: dict[str, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        variant = solver_machinery_variant_id(record)
        if not variant:
            continue
        by_pair.setdefault(solver_machinery_pair_key(record["row"]), {})[
            variant
        ] = record
    return by_pair


def compare_solver_machinery_variant(
    by_pair: Mapping[str, Mapping[str, Mapping[str, Any]]],
    variant: str,
) -> list[dict[str, Any]]:
    rows = []
    for key, records_by_variant in sorted(by_pair.items()):
        production = records_by_variant.get(SOLVER_MACHINERY_PRODUCTION_VARIANT)
        compared = (
            production
            if variant == SOLVER_MACHINERY_PRODUCTION_VARIANT
            else records_by_variant.get(variant)
        )
        if production is None or compared is None:
            row0 = (production or compared or {}).get("row", {})
            rows.append(
                {
                    "pair_key": key,
                    "variant": variant,
                    "variant_label": solver_machinery_variant_label(variant),
                    "configuration_id": row0.get("configuration_id"),
                    "activation_cap_mode": row0.get("activation_cap_mode"),
                    "communication_profile": row0.get("communication_profile"),
                    "random_seed": row0.get("random_seed"),
                    "classification": "unpaired",
                    "production_outcome": "missing"
                    if production is None
                    else solver_machinery_record_state(production),
                    "variant_outcome": "missing"
                    if compared is None
                    else solver_machinery_record_state(compared),
                    "production_makespan": None,
                    "variant_makespan": None,
                    "final_makespan_over_production": None,
                    "production_incumbent_makespan": None,
                    "variant_incumbent_makespan": None,
                    "incumbent_makespan_over_production_incumbent": None,
                    "production_incumbent_hints_effective": None,
                    "production_incumbent_bound_effective": None,
                }
            )
            continue

        row0 = production["row"]
        production_valid = is_valid_solution_record(production)
        compared_valid = is_valid_solution_record(compared)
        production_no_solution = is_no_solution_record(production)
        compared_no_solution = is_no_solution_record(compared)
        production_makespan = makespan_from_record(production)
        compared_makespan = makespan_from_record(compared)

        if production_valid and compared_valid and production_makespan and compared_makespan:
            if compared_makespan + 1e-9 < production_makespan:
                classification = "win"
            elif abs(compared_makespan - production_makespan) <= 1e-9:
                classification = "tie"
            else:
                classification = "loss"
            final_ratio = compared_makespan / production_makespan
        elif production_valid and compared_no_solution:
            classification = "no_solution"
            final_ratio = None
        elif production_no_solution and compared_valid:
            classification = "production_no_solution_variant_valid"
            final_ratio = None
        elif production_no_solution and compared_no_solution:
            classification = "both_no_solution"
            final_ratio = None
        else:
            classification = "invalid_or_incomplete"
            final_ratio = None

        production_inc = numeric_from_record(
            production, ["incumbent_makespan", "bfs_incumbent_makespan_ticks"],
            positive=True,
        )
        compared_inc = numeric_from_record(
            compared, ["incumbent_makespan", "bfs_incumbent_makespan_ticks"],
            positive=True,
        )
        incumbent_ratio = (
            compared_inc / production_inc
            if production_inc is not None and compared_inc is not None
            else None
        )
        rows.append(
            {
                "pair_key": key,
                "variant": variant,
                "variant_label": solver_machinery_variant_label(variant),
                "configuration_id": row0.get("configuration_id"),
                "activation_cap_mode": row0.get("activation_cap_mode"),
                "communication_profile": row0.get("communication_profile"),
                "random_seed": row0.get("random_seed"),
                "classification": classification,
                "production_outcome": solver_machinery_record_state(production),
                "variant_outcome": solver_machinery_record_state(compared),
                "production_makespan": production_makespan,
                "variant_makespan": compared_makespan,
                "final_makespan_over_production": final_ratio,
                "production_incumbent_makespan": production_inc,
                "variant_incumbent_makespan": compared_inc,
                "incumbent_makespan_over_production_incumbent": incumbent_ratio,
                "production_incumbent_hints_effective": bool_from_record(
                    production, ["incumbent_hints_effective"]
                ),
                "production_incumbent_bound_effective": bool_from_record(
                    production, ["incumbent_bound_effective"]
                ),
            }
        )
    return rows


def summarize_pair_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    attempted = [row for row in rows if row["classification"] != "unpaired"]
    successful = [
        row
        for row in attempted
        if row["classification"] in {"win", "tie", "loss"}
        and row["final_makespan_over_production"] is not None
    ]
    final_ratios = [
        float(row["final_makespan_over_production"]) for row in successful
    ]
    incumbent_ratios = [
        float(row["incumbent_makespan_over_production_incumbent"])
        for row in attempted
        if row.get("incumbent_makespan_over_production_incumbent") is not None
        and float(row["incumbent_makespan_over_production_incumbent"]) > 0
    ]
    counts = {
        name: sum(1 for row in attempted if row["classification"] == name)
        for name in [
            "win",
            "tie",
            "loss",
            "no_solution",
            "production_no_solution_variant_valid",
            "both_no_solution",
            "invalid_or_incomplete",
        ]
    }
    return {
        "paired_attempt_rows": len(attempted),
        "paired_successful_rows": len(successful),
        "final_makespan_over_production_geomean": geometric_mean(final_ratios)
        if final_ratios
        else None,
        "incumbent_makespan_over_production_incumbent_geomean": (
            geometric_mean(incumbent_ratios) if incumbent_ratios else None
        ),
        **counts,
    }


def summarize_mechanism_subset(
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    subset_name: str,
    predicate: Callable[[Mapping[str, Any]], bool],
) -> dict[str, Any]:
    subset = [
        row for row in pair_rows if row["variant"] == variant and predicate(row)
    ]
    summary = summarize_pair_rows(subset)
    summary.update(
        {
            "variant": variant,
            "subset": subset_name,
            "status": "available" if subset else "no_data",
        }
    )
    return summary


def slack_incumbent_comparison(
    pair_rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    canonical_pairs = [
        row
        for row in pair_rows
        if row["variant"] == SOLVER_MACHINERY_CANONICAL_VARIANT
        and row.get("incumbent_makespan_over_production_incumbent") is not None
    ]
    ratios = [
        float(row["incumbent_makespan_over_production_incumbent"])
        for row in canonical_pairs
        if float(row["incumbent_makespan_over_production_incumbent"]) > 0
    ]
    production_records = [
        record
        for record in records
        if solver_machinery_variant_id(record)
        == SOLVER_MACHINERY_PRODUCTION_VARIANT
    ]
    canonical_records = [
        record
        for record in records
        if solver_machinery_variant_id(record)
        == SOLVER_MACHINERY_CANONICAL_VARIANT
    ]

    def incumbent_available(record: Mapping[str, Any]) -> bool:
        return (
            bool_from_record(record, ["incumbent_valid"]) is True
            and numeric_from_record(record, ["incumbent_makespan"], positive=True)
            is not None
        )

    return {
        "status": "available" if ratios else "no_data",
        "paired_incumbents": len(ratios),
        "slack_better": sum(1 for ratio in ratios if ratio > 1.0 + 1e-9),
        "tie": sum(1 for ratio in ratios if abs(ratio - 1.0) <= 1e-9),
        "canonical_better": sum(1 for ratio in ratios if ratio + 1e-9 < 1.0),
        "geomean_canonical_over_slack_incumbent_makespan": (
            geometric_mean(ratios) if ratios else None
        ),
        "production_incumbent_availability_rate": (
            sum(1 for record in production_records if incumbent_available(record))
            / len(production_records)
            if production_records
            else None
        ),
        "canonical_incumbent_availability_rate": (
            sum(1 for record in canonical_records if incumbent_available(record))
            / len(canonical_records)
            if canonical_records
            else None
        ),
    }


def aggregate_solver_machinery_ablation(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    machinery_records = [
        record for record in records if solver_machinery_variant_id(record)
    ]
    if not machinery_records:
        return {"status": "not_applicable"}

    by_pair = solver_machinery_records_by_pair(machinery_records)
    variant_order = solver_machinery_variant_order(machinery_records)
    pair_rows: list[dict[str, Any]] = []
    for variant in variant_order:
        pair_rows.extend(compare_solver_machinery_variant(by_pair, variant))

    variant_summary = []
    for variant in variant_order:
        variant_records = [
            record
            for record in machinery_records
            if solver_machinery_variant_id(record) == variant
        ]
        valid_runs = [record for record in variant_records if is_valid_solution_record(record)]
        no_solution_runs = [
            record for record in variant_records if is_no_solution_record(record)
        ]
        denominator = len(valid_runs) + len(no_solution_runs)
        paired = [
            row for row in pair_rows if row["variant"] == variant
        ]
        paired_summary = summarize_pair_rows(paired)
        final_from_incumbent = [
            record
            for record in valid_runs
            if bool_from_record(record, ["fallback_used"]) is True
            or (record.get("canonical") or {}).get("final_solution_source")
            == "external_incumbent"
        ]
        gap_values = [
            value
            for value in (solver_gap_from_record(record) for record in variant_records)
            if value is not None
        ]
        first_feasible_values = [
            value
            for value in (
                numeric_from_record(
                    record,
                    [
                        "time_to_first_cpsat_feasible_seconds",
                        "time_to_first_cpsat_feasible_solution",
                    ],
                    positive=True,
                )
                for record in variant_records
            )
            if value is not None
        ]
        best_bounds = [
            value
            for value in (
                numeric_from_record(
                    record,
                    ["best_bound", "best_bound_ticks", "best_objective_bound"],
                    positive=True,
                )
                for record in variant_records
            )
            if value is not None
        ]
        branches = [
            value
            for value in (
                numeric_from_record(record, ["num_branches", "branches"])
                for record in variant_records
            )
            if value is not None
        ]
        conflicts = [
            value
            for value in (
                numeric_from_record(record, ["num_conflicts", "conflicts"])
                for record in variant_records
            )
            if value is not None
        ]
        variant_summary.append(
            {
                "variant": variant,
                "variant_label": solver_machinery_variant_label(variant),
                "total_runs": len(variant_records),
                "valid_solution_runs": len(valid_runs),
                "no_solution_runs": len(no_solution_runs),
                "valid_solution_rate": (
                    len(valid_runs) / denominator if denominator else None
                ),
                "paired_attempt_rows": paired_summary["paired_attempt_rows"],
                "paired_successful_rows": paired_summary[
                    "paired_successful_rows"
                ],
                "final_makespan_over_production_geomean": paired_summary[
                    "final_makespan_over_production_geomean"
                ],
                "incumbent_makespan_over_production_incumbent_geomean": (
                    paired_summary[
                        "incumbent_makespan_over_production_incumbent_geomean"
                    ]
                ),
                "final_from_incumbent_rate": (
                    len(final_from_incumbent) / len(valid_runs)
                    if valid_runs
                    else None
                ),
                "optimal_proven_rate": (
                    sum(
                        1
                        for record in valid_runs
                        if (record.get("canonical") or {}).get("optimal") is True
                    )
                    / len(valid_runs)
                    if valid_runs
                    else None
                ),
                "median_gap": median(gap_values) if gap_values else None,
                "median_first_cpsat_feasible_time_s": (
                    median(first_feasible_values)
                    if first_feasible_values
                    else None
                ),
                "median_best_bound": median(best_bounds) if best_bounds else None,
                "median_branches": median(branches) if branches else None,
                "median_conflicts": median(conflicts) if conflicts else None,
                "win": paired_summary["win"],
                "tie": paired_summary["tie"],
                "loss": paired_summary["loss"],
                "no_solution_vs_valid_production": paired_summary["no_solution"],
                "production_no_solution_variant_valid": paired_summary[
                    "production_no_solution_variant_valid"
                ],
                "both_no_solution": paired_summary["both_no_solution"],
                "invalid_or_incomplete_pairs": paired_summary[
                    "invalid_or_incomplete"
                ],
            }
        )

    mechanism_subsets = {
        "hint_ablation": {
            "all_pairs": summarize_mechanism_subset(
                pair_rows,
                variant=SOLVER_MACHINERY_NO_HINTS_VARIANT,
                subset_name="all_paired_attempts",
                predicate=lambda row: True,
            ),
            "production_hint_effective_subset": summarize_mechanism_subset(
                pair_rows,
                variant=SOLVER_MACHINERY_NO_HINTS_VARIANT,
                subset_name="production_incumbent_hints_effective",
                predicate=lambda row: row.get(
                    "production_incumbent_hints_effective"
                )
                is True,
            ),
        },
        "bound_ablation": {
            "all_pairs": summarize_mechanism_subset(
                pair_rows,
                variant=SOLVER_MACHINERY_NO_BOUND_VARIANT,
                subset_name="all_paired_attempts",
                predicate=lambda row: True,
            ),
            "production_bound_effective_subset": summarize_mechanism_subset(
                pair_rows,
                variant=SOLVER_MACHINERY_NO_BOUND_VARIANT,
                subset_name="production_incumbent_bound_effective",
                predicate=lambda row: row.get(
                    "production_incumbent_bound_effective"
                )
                is True,
            ),
        },
    }

    return {
        "status": "available",
        "production_variant": SOLVER_MACHINERY_PRODUCTION_VARIANT,
        "variant_order": variant_order,
        "pairing_key": [
            "W",
            "N",
            "B",
            "L",
            "communication_profile",
            "activation_cap_mode",
            "random_seed",
        ],
        "summary_policy": (
            "No-solution rows contribute to valid-solution rates, but final "
            "makespan ratios use only paired rows where both sides returned "
            "valid final schedules."
        ),
        "variant_summary": variant_summary,
        "pair_rows": pair_rows,
        "slack_incumbent_comparison": slack_incumbent_comparison(
            pair_rows, machinery_records
        ),
        "mechanism_eligible_subsets": mechanism_subsets,
    }


def write_solver_machinery_outputs(
    payload: Mapping[str, Any], output_dir: Path, formats: set[str]
) -> None:
    if payload.get("status") != "available":
        return
    if "json" in formats:
        write_json(output_dir / "solver_machinery_ablation_summary.json", payload)
    if "csv" in formats:
        write_csv(
            output_dir / "solver_machinery_ablation_variant_summary.csv",
            payload.get("variant_summary") or [],
        )
        write_csv(
            output_dir / "solver_machinery_ablation_pair_rows.csv",
            payload.get("pair_rows") or [],
        )


def plot_normalized_makespan(
    summaries: Mapping[str, Mapping[str, Any]], output_dir: Path, formats: set[str]
) -> None:
    if not ({"pdf", "png"} & formats):
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = [mode for mode in ["uncapped", "equal_memory"] if mode in summaries]
    if not modes:
        return
    fig, axes = plt.subplots(1, len(modes), figsize=(3.4 * len(modes), 2.4), squeeze=False)
    for ax, mode in zip(axes[0], modes):
        method_summary = summaries[mode]["method_summary"]
        values = []
        err_low = []
        err_high = []
        labels = []
        for method in METHOD_ORDER:
            if method == summary_baseline_method(summaries[mode]):
                continue
            item = method_summary.get(method, {})
            if not item or item.get("status") == "missing":
                continue
            value = item["geomean_normalized_makespan"]
            labels.append(METHOD_LABELS.get(method, method))
            values.append(value)
            err_low.append(value - item["min_normalized_makespan"])
            err_high.append(item["max_normalized_makespan"] - value)
        x = list(range(len(values)))
        ax.bar(x, values, color="#d9d9d9", edgecolor="#222222", linewidth=0.8)
        ax.errorbar(x, values, yerr=[err_low, err_high], fmt="none", ecolor="#222222", linewidth=0.8, capsize=2)
        ax.axhline(1.0, color="#444444", linestyle="--", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Normalized makespan")
        ax.set_title("Equal memory" if mode == "equal_memory" else "Uncapped")
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    if "pdf" in formats:
        fig.savefig(output_dir / "fig_cal_normalized_makespan.pdf")
    if "png" in formats:
        fig.savefig(output_dir / "fig_cal_normalized_makespan.png", dpi=200)
    plt.close(fig)


def write_normalized_outputs(
    records: Sequence[Mapping[str, Any]],
    modes: Sequence[str],
    output_dir: Path,
    formats: set[str],
    normalization_baseline: str = DEFAULT_NORMALIZATION_BASELINE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for mode in modes:
        rows, summary = aggregate_normalized_makespan(
            records, mode, normalization_baseline
        )
        all_rows.extend(rows)
        summaries[mode] = summary
    if "csv" in formats:
        write_csv(output_dir / "fig_cal_normalized_makespan.csv", all_rows)
    if "json" in formats:
        write_json(output_dir / "fig_cal_normalized_makespan.json", summaries)
    plot_normalized_makespan(summaries, output_dir, formats)
    return all_rows, summaries


def trace_points(c: Mapping[str, Any]) -> list[tuple[float, float]]:
    raw = c.get("incumbent_trace")
    points: list[tuple[float, float]] = []
    if not isinstance(raw, list):
        return points
    for entry in raw:
        if isinstance(entry, Mapping):
            t = entry.get("time_seconds") or entry.get("elapsed_seconds") or entry.get("time")
            v = entry.get("objective") or entry.get("makespan") or entry.get("value")
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            t, v = entry[0], entry[1]
        else:
            continue
        try:
            points.append((float(t), float(v)))
        except Exception:
            continue
    points.sort()
    best = math.inf
    best_points = []
    for t, value in points:
        if value < best:
            best = value
        best_points.append((t, best))
    return best_points


def best_so_far_at(points: Sequence[tuple[float, float]], checkpoint: float) -> float | None:
    best = None
    for time_value, objective in points:
        if time_value <= checkpoint:
            best = objective
        else:
            break
    return best


def convergence_data(
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    formats: set[str],
    strict: bool,
    normalization_baseline: str = DEFAULT_NORMALIZATION_BASELINE,
) -> dict[str, Any]:
    baseline_by_workload: dict[str, float] = {}
    for mode in ["uncapped", "equal_memory"]:
        for record in valid_records_for_mode(records, mode):
            if record["row"]["method"] == normalization_baseline:
                makespan = makespan_from_record(record)
                if makespan:
                    baseline_by_workload[workload_key(record["row"])] = makespan
    run_rows = []
    for record in records:
        method = record["row"]["method"]
        if method not in CONVERGENCE_METHODS:
            continue
        if "valid_for_uncapped" not in record["classes"] and "valid_for_equal_memory" not in record["classes"]:
            continue
        points = trace_points(record["canonical"])
        if not points:
            continue
        base = baseline_by_workload.get(workload_key(record["row"]))
        if not base:
            continue
        time_limit = float(record["row"]["time_limit_seconds"])
        for checkpoint in CHECKPOINTS_SECONDS:
            if checkpoint > time_limit:
                continue
            best = best_so_far_at(points, checkpoint)
            run_rows.append(
                {
                    "analysis_mode": mode_for_row(record["row"]),
                    "workload_key": workload_key(record["row"]),
                    "configuration_id": record["row"]["configuration_id"],
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "normalization_baseline_method": normalization_baseline,
                    "random_seed": record["row"]["random_seed"],
                    "checkpoint_seconds": checkpoint,
                    "best_makespan": best,
                    "feasible_by_checkpoint": best is not None,
                    "normalized_best_makespan": None if best is None else best / base,
                    "final_normalized_makespan": makespan_from_record(record) / base
                    if makespan_from_record(record)
                    else None,
                }
            )
    if not run_rows:
        status = {
            "convergence_figure_status": "unavailable_no_trace_data",
            "message": "No incumbent trace data was found in compatible result rows.",
        }
        if "json" in formats:
            write_json(output_dir / "fig_cal_search_convergence.json", status)
        if "csv" in formats:
            write_csv(output_dir / "fig_cal_search_convergence.csv", [status])
        if strict:
            raise RuntimeError(status["message"])
        return status

    grouped: dict[tuple[str, str, str, float], list[dict[str, Any]]] = {}
    for row in run_rows:
        grouped.setdefault(
            (
                row["analysis_mode"],
                row["workload_key"],
                row["method"],
                float(row["checkpoint_seconds"]),
            ),
            [],
        ).append(row)
    workload_rows = []
    for (mode, key, method, checkpoint), rows in sorted(grouped.items()):
        values = [
            row["normalized_best_makespan"]
            for row in rows
            if row["normalized_best_makespan"] is not None
        ]
        workload_rows.append(
            {
                "analysis_mode": mode,
                "workload_key": key,
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "normalization_baseline_method": normalization_baseline,
                "checkpoint_seconds": checkpoint,
                "seed_count": len(rows),
                "feasible_fraction": len(values) / len(rows),
                "median_normalized_best_makespan": median(values) if values else None,
                "final_normalized_makespan_median": median(
                    [
                        row["final_normalized_makespan"]
                        for row in rows
                        if row["final_normalized_makespan"] is not None
                    ]
                ),
            }
        )
    aggregate_grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for row in workload_rows:
        aggregate_grouped.setdefault(
            (row["analysis_mode"], row["method"], row["checkpoint_seconds"]), []
        ).append(row)
    aggregate_rows = []
    for (mode, method, checkpoint), rows in sorted(aggregate_grouped.items()):
        values = [
            row["median_normalized_best_makespan"]
            for row in rows
            if row["median_normalized_best_makespan"] is not None
        ]
        aggregate_rows.append(
            {
                "analysis_mode": mode,
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "checkpoint_seconds": checkpoint,
                "workload_count": len(rows),
                "feasible_fraction": sum(row["feasible_fraction"] for row in rows)
                / len(rows),
                "geomean_normalized_best_makespan": geometric_mean(values)
                if values
                else None,
            }
        )
    if "csv" in formats:
        write_csv(output_dir / "fig_cal_search_convergence.csv", aggregate_rows)
    payload = {
        "convergence_figure_status": "available",
        "run_rows": run_rows,
        "workload_rows": workload_rows,
        "aggregate_rows": aggregate_rows,
    }
    if "json" in formats:
        write_json(output_dir / "fig_cal_search_convergence.json", payload)
    plot_convergence(aggregate_rows, output_dir, formats)
    return payload


def plot_convergence(rows: Sequence[Mapping[str, Any]], output_dir: Path, formats: set[str]) -> None:
    if not rows or not ({"pdf", "png"} & formats):
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = sorted({row["analysis_mode"] for row in rows})
    fig, axes = plt.subplots(1, len(modes), figsize=(3.4 * len(modes), 2.4), squeeze=False)
    for ax, mode in zip(axes[0], modes):
        for method in CONVERGENCE_METHODS:
            data = [
                row
                for row in rows
                if row["analysis_mode"] == mode
                and row["method"] == method
                and row["geomean_normalized_best_makespan"] is not None
            ]
            if not data:
                continue
            data.sort(key=lambda row: row["checkpoint_seconds"])
            ax.plot(
                [row["checkpoint_seconds"] for row in data],
                [row["geomean_normalized_best_makespan"] for row in data],
                marker="o",
                label=METHOD_LABELS.get(method, method),
                linewidth=1.0,
            )
        ax.set_xscale("log")
        ax.axhline(1.0, color="#444444", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Seconds")
        ax.set_ylabel("Best-so-far / baseline")
        ax.set_title("Equal memory" if mode == "equal_memory" else "Uncapped")
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    if "pdf" in formats:
        fig.savefig(output_dir / "fig_cal_search_convergence.pdf")
    if "png" in formats:
        fig.savefig(output_dir / "fig_cal_search_convergence.png", dpi=200)
    plt.close(fig)


def activation_peak_ratio(c: Mapping[str, Any]) -> float | None:
    ratio = c.get("activation_peak_ratio_to_uniform")
    if isinstance(ratio, Mapping):
        value = ratio.get("maximum_worker_peak_units_ratio")
        if value is not None:
            return float(value)
    metrics = c.get("activation_memory_metrics")
    if isinstance(metrics, Mapping):
        value = metrics.get("maximum_worker_peak_units_ratio")
        if value is not None:
            return float(value)
    return None


def oracle_quality_summary(
    records: Sequence[Mapping[str, Any]], classes: Mapping[str, int]
) -> dict[str, Any]:
    oracle_records = [
        record for record in records if record["row"]["experiment_group"] == "oracle"
    ]
    proven_oracle = [
        record
        for record in oracle_records
        if record["canonical"].get("optimal") is True
        and record["canonical"].get("enumeration_proved_optimal") is True
        and record["canonical"].get("optimality_proof_source")
        == "exhaustive_enumeration"
        and makespan_from_record(record) is not None
    ]
    if not oracle_records:
        return {"oracle_status": "unavailable"}

    valid_records = [
        record
        for record in records
        if "valid_for_uncapped" in record["classes"]
        or "valid_for_equal_memory" in record["classes"]
    ]
    by_key_method: dict[tuple[str, str], list[float]] = {}
    for record in valid_records:
        makespan = makespan_from_record(record)
        if makespan is None:
            continue
        by_key_method.setdefault(
            (workload_key(record["row"]), record["row"]["method"]), []
        ).append(makespan)

    optimum_by_key: dict[str, float] = {}
    for record in proven_oracle:
        makespan = makespan_from_record(record)
        if makespan is None:
            continue
        key = workload_key(record["row"])
        current = optimum_by_key.get(key)
        if current is None or makespan < current:
            optimum_by_key[key] = makespan
    proven_configuration_ids = {
        record["row"]["configuration_id"] for record in proven_oracle
    }

    def method_reaches_percent(method: str) -> tuple[float | None, int]:
        comparable = 0
        reaches = 0
        for key, optimum in optimum_by_key.items():
            values = by_key_method.get((key, method))
            if not values:
                continue
            comparable += 1
            if abs(median(values) - optimum) <= 1e-9:
                reaches += 1
        if comparable == 0:
            return None, 0
        return 100.0 * reaches / comparable, comparable

    joint_gaps = []
    for key, optimum in optimum_by_key.items():
        values = by_key_method.get((key, "joint-unrestricted-no-overlap"))
        if not values or optimum <= 0:
            continue
        joint_gaps.append((median(values) - optimum) / optimum)

    joint_percent, joint_comparable = method_reaches_percent(
        "joint-unrestricted-no-overlap"
    )
    partition_percent, partition_comparable = method_reaches_percent(
        "partition-only-fixed-order"
    )
    alternating_percent, alternating_comparable = method_reaches_percent(
        "alternating-partition-schedule"
    )
    return {
        "oracle_status": "available",
        "oracle_configuration_count": len(
            {record["row"]["configuration_id"] for record in oracle_records}
        ),
        "proven_optimum_count": len(proven_configuration_ids),
        "proven_optimum_workload_count": len(optimum_by_key),
        "joint_reaches_proven_optimum_percent": joint_percent,
        "joint_comparable_count": joint_comparable,
        "median_optimality_gap_to_proven_optimum": median(joint_gaps)
        if joint_gaps
        else None,
        "maximum_optimality_gap_to_proven_optimum": max(joint_gaps)
        if joint_gaps
        else None,
        "partition_only_reaches_proven_optimum_percent": partition_percent,
        "partition_only_comparable_count": partition_comparable,
        "alternating_reaches_proven_optimum_percent": alternating_percent,
        "alternating_comparable_count": alternating_comparable,
        "oracle_unproven_count": classes.get("oracle_unproven", 0),
    }


def table_summary(
    records: Sequence[Mapping[str, Any]],
    normalized_rows: Sequence[Mapping[str, Any]],
    normalized_summary: Mapping[str, Any],
) -> dict[str, Any]:
    total = len(records)
    classes = count_classes(records)
    method_counts: dict[str, dict[str, int]] = {}
    for record in records:
        method = record["row"]["method"]
        method_counts.setdefault(method, {"rows": 0, "fallback": 0})
        method_counts[method]["rows"] += 1
        if "fallback_used" in record["classes"]:
            method_counts[method]["fallback"] += 1
    fallback_rate_by_method = {
        method: counts["fallback"] / counts["rows"]
        for method, counts in sorted(method_counts.items())
        if counts["rows"]
    }

    oracle_summary = oracle_quality_summary(records, classes)

    uncapped_joint_ratio_by_workload: dict[str, list[float]] = {}
    for record in records:
        if (
            record["row"]["method"] != "joint-unrestricted-no-overlap"
            or "valid_for_uncapped" not in record["classes"]
        ):
            continue
        value = activation_peak_ratio(record["canonical"])
        if value is not None:
            uncapped_joint_ratio_by_workload.setdefault(
                workload_key(record["row"]), []
            ).append(value)
    uncapped_joint_ratios = [
        median(values) for values in uncapped_joint_ratio_by_workload.values()
    ]
    activation_uncapped = {
        "max_peak_activation_ratio_to_uniform": max(uncapped_joint_ratios)
        if uncapped_joint_ratios
        else None,
        "median_peak_activation_ratio_to_uniform": median(uncapped_joint_ratios)
        if uncapped_joint_ratios
        else None,
        "joint_more_activation_than_uniform_workloads": sum(
            1 for value in uncapped_joint_ratios if value > 1.0
        ),
        "joint_less_or_equal_activation_than_uniform_workloads": sum(
            1 for value in uncapped_joint_ratios if value <= 1.0
        ),
    }
    activation_equal_memory = {
        "eligible_rows": classes.get("valid_for_equal_memory", 0),
        "cap_violation_count": sum(
            1
            for record in records
            if record["canonical"].get("activation_cap_satisfied") is False
        ),
        "posthoc_only_excluded_count": sum(
            1
            for record in records
            if record["row"].get("activation_cap_mode") != "none"
            and record["canonical"].get("activation_cap_enforcement_mode")
            == "posthoc_only"
        ),
        "solver_enforced_capped_rows": sum(
            1
            for record in records
            if record["canonical"].get("activation_cap_enforced_in_solver") is True
        ),
        "deterministic_baseline_cap_check_rows": sum(
            1
            for record in records
            if record["canonical"].get("activation_cap_enforcement_mode")
            == "deterministic_postconstruction_check"
        ),
    }
    communication = communication_summary(normalized_rows)
    structural = structural_summary(normalized_rows)
    return {
        "oracle": oracle_summary,
        "validation_and_fallback": {
            "total_rows_analyzed": total,
            "validation_failure_count": classes.get("validation_failed", 0),
            "invalid_result_count": classes.get("invalid_result", 0),
            "no_solution_count": classes.get("no_solution", 0),
            "fallback_count": classes.get("fallback_used", 0),
            "fallback_rate_by_method": fallback_rate_by_method,
            "unknown_fallback_count": sum(
                1
                for record in records
                if "fallback_used" in record["classes"]
                and record["canonical"].get("solver_status_raw") == "UNKNOWN"
            ),
            "not_run_fallback_count": sum(
                1
                for record in records
                if "fallback_used" in record["classes"]
                and record["canonical"].get("solver_status_raw") == "NOT_RUN"
            ),
            "unavailable_count": classes.get("unavailable", 0),
        },
        "activation_uncapped": activation_uncapped,
        "activation_equal_memory": activation_equal_memory,
        "communication_sensitivity": communication,
        "structural_sensitivity": structural,
    }


def communication_summary(normalized_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in normalized_rows:
        grouped.setdefault((row["analysis_mode"], row["communication_profile"]), []).append(row)
    out = []
    for (mode, profile), rows in sorted(grouped.items()):
        joint = [
            row["normalized_makespan_median"]
            for row in rows
            if row["method"] == "joint-unrestricted-no-overlap"
        ]
        if not joint:
            continue
        baseline_values = []
        for method in METHOD_ORDER:
            if method == "joint-unrestricted-no-overlap":
                continue
            values = [
                row["normalized_makespan_median"]
                for row in rows
                if row["method"] == method
            ]
            if values:
                baseline_values.append(geometric_mean(values))
        best_baseline = min(baseline_values) if baseline_values else None
        joint_gmean = geometric_mean(joint)
        out.append(
            {
                "analysis_mode": mode,
                "communication_profile": profile,
                "joint_geomean_normalized_makespan": joint_gmean,
                "best_separate_baseline_geomean_normalized_makespan": best_baseline,
                "joint_improvement_over_best_separate_baseline": (
                    best_baseline / joint_gmean if best_baseline else None
                ),
            }
        )
    return out


def structural_summary(normalized_rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "stage_multiplier": grouped_joint_summary(normalized_rows, "stage_multiplier"),
        "batch_ratio": grouped_joint_summary(normalized_rows, "batch_ratio"),
    }


def grouped_joint_summary(
    normalized_rows: Sequence[Mapping[str, Any]], field: str
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[Mapping[str, Any]]] = {}
    for row in normalized_rows:
        grouped.setdefault((row["analysis_mode"], float(row[field])), []).append(row)
    out = []
    for (mode, value), rows in sorted(grouped.items()):
        joint = [
            row["normalized_makespan_median"]
            for row in rows
            if row["method"] == "joint-unrestricted-no-overlap"
        ]
        if not joint:
            continue
        baseline_values = []
        for method in METHOD_ORDER:
            if method == "joint-unrestricted-no-overlap":
                continue
            values = [
                row["normalized_makespan_median"]
                for row in rows
                if row["method"] == method
            ]
            if values:
                baseline_values.append(geometric_mean(values))
        joint_gmean = geometric_mean(joint)
        best_baseline = min(baseline_values) if baseline_values else None
        out.append(
            {
                "analysis_mode": mode,
                field: value,
                "joint_geomean_normalized_makespan": joint_gmean,
                "joint_improvement_over_best_separate_baseline": (
                    best_baseline / joint_gmean if best_baseline else None
                ),
            }
        )
    return out


def table_rows_from_summary(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    validation = summary["validation_and_fallback"]
    activation = summary["activation_equal_memory"]
    oracle = summary["oracle"]
    rows = [
        {"section": "Validation", "metric": "Rows analyzed", "value": validation["total_rows_analyzed"]},
        {"section": "Validation", "metric": "Invalid results", "value": validation["invalid_result_count"]},
        {"section": "Validation", "metric": "No valid solution", "value": validation.get("no_solution_count", 0)},
        {"section": "Validation", "metric": "Fallbacks", "value": validation["fallback_count"]},
        {"section": "Validation", "metric": "Unavailable", "value": validation["unavailable_count"]},
        {"section": "Activation", "metric": "Equal-memory eligible rows", "value": activation["eligible_rows"]},
        {"section": "Activation", "metric": "Cap violations", "value": activation["cap_violation_count"]},
        {"section": "Activation", "metric": "Solver-enforced cap rows", "value": activation["solver_enforced_capped_rows"]},
        {"section": "Oracle", "metric": "Oracle status", "value": oracle["oracle_status"]},
    ]
    if oracle["oracle_status"] != "unavailable":
        rows.append({"section": "Oracle", "metric": "Proven optima", "value": oracle["proven_optimum_count"]})
    return rows


def write_latex_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "% CAL summary table fragment generated by analyze_cal_results.py",
        "% gmean = geometric mean over workload-level seed medians.",
        "% Equal-memory eligibility requires cap satisfaction and solver-side enforcement for solver-backed methods.",
        "% Oracle optimum claims require a proven optimal oracle row.",
        "\\begin{tabular}{lll}",
        "\\toprule",
        "Section & Metric & Value \\\\",
        "\\midrule",
    ]
    for row in rows:
        value = row["value"]
        if isinstance(value, float):
            value = f"{value:.2f}"
        lines.append(f"{row['section']} & {row['metric']} & {value} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_table_outputs(summary: Mapping[str, Any], output_dir: Path, formats: set[str]) -> None:
    rows = table_rows_from_summary(summary)
    if "csv" in formats:
        write_csv(output_dir / "table_cal_summary.csv", rows)
    if "json" in formats:
        write_json(output_dir / "table_cal_summary.json", summary)
    if "tex" in formats:
        write_latex_table(output_dir / "table_cal_summary.tex", rows)


def profile_phrase(groups: Sequence[str]) -> str:
    group_set = set(groups)
    if group_set == {"smoke"}:
        return "In the analyzed smoke profile"
    if group_set == {"main"}:
        return "In the main profile"
    return "In the analyzed profiles"


def write_text_summary(
    *,
    output_dir: Path,
    metadata: Mapping[str, Any],
    preflight_summary: Mapping[str, Any],
    normalized_summary: Mapping[str, Any],
    table: Mapping[str, Any],
    groups: Sequence[str],
    normalization_baseline: str = DEFAULT_NORMALIZATION_BASELINE,
    solver_machinery: Mapping[str, Any] | None = None,
) -> None:
    phrase = profile_phrase(groups)
    baseline_label = METHOD_LABELS.get(normalization_baseline, normalization_baseline)
    lines = [
        "# CAL Evaluation Summary",
        "",
        f"Analysis timestamp: `{metadata['analysis_timestamp_utc']}`.",
        f"Manifest id/hash: `{metadata['manifest_id']}` / `{metadata['manifest_hash']}`.",
        f"Git commit policy source: `{metadata.get('git_commit_expected')}`.",
        f"Aggregation rule: normalized makespan uses workload-level `{normalization_baseline}` ({baseline_label}); seeds are aggregated by median before geometric means across workloads.",
        f"Rows analyzed: `{preflight_summary['total_rows']}`.",
        "",
    ]
    for mode in ["uncapped", "equal_memory"]:
        if mode not in normalized_summary:
            continue
        summary = normalized_summary[mode]
        joint = summary["method_summary"].get("joint-unrestricted-no-overlap", {})
        if not joint or joint.get("status") == "missing":
            lines.append(f"No complete joint data was available for `{mode}`.")
            continue
        mode_label = "equal-memory" if mode == "equal_memory" else "uncapped"
        lines.append(
            f"{phrase} ({mode_label}), joint geometric mean normalized makespan was "
            f"`{joint['geomean_normalized_makespan']:.3f}`, an improvement of "
            f"`{joint['geomean_improvement_over_baseline']:.2f}x` over "
            f"`{normalization_baseline}` "
            f"across `{joint['workload_count']}` workloads."
        )
        for method, value in summary["joint_geomean_improvement_over_methods"].items():
            lines.append(
                f"Joint improvement over {METHOD_LABELS.get(method, method)} was `{value:.2f}x` "
                f"by geometric mean in {mode_label} mode."
            )
        lines.append(
            f"Joint won `{summary['workload_win_counts'].get('joint-unrestricted-no-overlap', 0)}` "
            f"workloads; joint loss cases: `{len(summary['joint_loss_workloads'])}`."
        )
        max_case = summary.get("maximum_joint_improvement_over_uniform")
        if max_case is None:
            max_case = summary.get("maximum_joint_improvement_over_baseline")
        if max_case:
            lines.append(
                f"Maximum joint improvement over `{normalization_baseline}` was "
                f"`{max_case['improvement_over_baseline']:.2f}x` on "
                f"`{max_case['configuration_id']}`."
            )
        comparison = summary.get("breadth_first_vs_interleaved_1f1b") or {}
        if comparison.get("status") == "available":
            lines.append(
                "Uniform breadth-first vs interleaved 1F1B: "
                f"1F1B was strictly better in "
                f"`{comparison['one_f_one_b_strictly_better_than_breadth_first_workloads']}` "
                "workloads, breadth-first was strictly better in "
                f"`{comparison['breadth_first_strictly_better_than_1f1b_workloads']}`, "
                f"and `{comparison['tied_workloads']}` tied."
            )
        lines.append("")
    activation = table["activation_uncapped"]
    if activation["median_peak_activation_ratio_to_uniform"] is not None:
        lines.append(
            "Uncapped activation tradeoff: median joint peak activation ratio to "
            f"uniform was `{activation['median_peak_activation_ratio_to_uniform']:.2f}`; "
            f"joint used more activation in `{activation['joint_more_activation_than_uniform_workloads']}` workloads."
        )
    equal_memory = table["activation_equal_memory"]
    lines.append(
        "Among rows eligible for equal-memory aggregation, "
        f"`{equal_memory['eligible_rows']}` rows passed; "
        f"`{equal_memory['posthoc_only_excluded_count']}` posthoc-only rows were excluded."
    )
    if table["oracle"]["oracle_status"] == "unavailable":
        lines.append("Oracle quality is unavailable because no oracle rows were analyzed.")
    else:
        lines.append(
            f"Oracle quality used `{table['oracle']['proven_optimum_count']}` proven optima; "
            "unproven oracle rows were not used for optimum-gap claims."
        )
    validation = table["validation_and_fallback"]
    lines.append(
        f"Validation caveats: invalid=`{validation['invalid_result_count']}`, "
        f"no-solution=`{validation.get('no_solution_count', 0)}`, "
        f"fallback=`{validation['fallback_count']}`, unavailable=`{validation['unavailable_count']}`."
    )
    if solver_machinery and solver_machinery.get("status") == "available":
        lines.append("")
        lines.append(
            "Solver-machinery ablation results are serialized in "
            "`solver_machinery_ablation_summary.json` and exclude "
            "no-solution rows from final-makespan geometric means."
        )
    lines.append("")
    lines.append("Communication sensitivity and structural trends are serialized in `table_cal_summary.json`.")
    (output_dir / "cal_eval_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        output_dir / "cal_eval_summary.json",
        {
            "metadata": metadata,
            "preflight_summary": preflight_summary,
            "normalized_summary": normalized_summary,
            "table_summary": table,
            "solver_machinery_ablation": solver_machinery,
        },
    )


def analyze(options: AnalysisOptions) -> dict[str, Any]:
    if options.utilization_normalize != "no":
        raise ValueError("utilization-vs-batch is absolute; normalization must be 'no'")
    formats = options.formats or {"pdf", "png", "csv", "json", "tex"}
    options.output_dir.mkdir(parents=True, exist_ok=True)
    rows = cal_manifest.read_manifest(options.manifest)
    manifest_metadata = {}
    metadata_path = options.manifest.with_suffix(options.manifest.suffix + ".metadata.json")
    if metadata_path.exists():
        try:
            manifest_metadata = load_json(metadata_path)
        except Exception:
            manifest_metadata = {}
    groups = options.include_groups or {str(row["experiment_group"]) for row in rows}
    modes = selected_modes(options.mode)
    selected = [
        row
        for row in rows
        if row["experiment_group"] in groups and mode_for_row(row) in modes
    ]
    protocol_failures = protocol_guard_failures(selected)
    records = classify_manifest_rows(selected, options.results_root)
    oracle_only = bool(selected) and all(
        row["experiment_group"] == "oracle" for row in selected
    )
    solver_machinery_only = bool(selected) and all(
        bool(row.get("solver_machinery_variant")) for row in selected
    )
    modes_to_analyze = (
        []
        if oracle_only or solver_machinery_only
        else (["uncapped", "equal_memory"] if options.mode == "both" else list(modes))
    )
    normalization_failures = normalization_precondition_failures(
        records, modes_to_analyze, options.normalization_baseline
    )
    preflight = preflight_rows(records)
    preflight_summary = {
        "total_rows": len(records),
        "class_counts": count_classes(records),
        "protocol_guard_failures": protocol_failures,
        "normalization_precondition_failures": normalization_failures,
    }
    write_csv(options.output_dir / "preflight_cal_rows.csv", preflight)
    write_json(options.output_dir / "preflight_cal_rows.json", preflight)
    write_json(options.output_dir / "preflight_cal_summary.json", preflight_summary)

    bad_counts = {
        cls: preflight_summary["class_counts"].get(cls, 0)
        for cls in BAD_CLASSES
        if preflight_summary["class_counts"].get(cls, 0)
    }
    unavailable_count = preflight_summary["class_counts"].get("unavailable", 0)
    if options.strict and (bad_counts or (unavailable_count and not options.allow_unavailable)):
        raise RuntimeError(
            "preflight failed: "
            + json.dumps(
                {"bad": bad_counts, "unavailable": unavailable_count},
                sort_keys=True,
            )
        )
    if options.strict and protocol_failures:
        raise RuntimeError(
            "protocol guard failed: " + json.dumps(protocol_failures, sort_keys=True)
        )

    if options.strict and normalization_failures:
        raise RuntimeError(
            "normalization preflight failed: "
            + json.dumps(normalization_failures, sort_keys=True)
        )

    solver_machinery = aggregate_solver_machinery_ablation(records)
    write_solver_machinery_outputs(solver_machinery, options.output_dir, formats)
    normalized_rows, normalized_summary = write_normalized_outputs(
        records,
        modes_to_analyze,
        options.output_dir,
        formats,
        options.normalization_baseline,
    )
    convergence = convergence_data(
        records,
        options.output_dir,
        formats,
        options.strict_convergence,
        options.normalization_baseline,
    )
    utilization = {"utilization_figure_status": "not_requested"}
    emit_utilization = should_emit_utilization_vs_batch(
        options, selected, manifest_metadata
    )
    require_complete_utilization_profile = any(
        "big_5min_1f1b_uncapped" in str(row.get("experiment_group") or "")
        for row in selected
    ) or "big_5min_1f1b_uncapped" in str(
        manifest_metadata.get("manifest_name") or manifest_metadata.get("manifest_id") or ""
    )
    if emit_utilization:
        utilization = write_utilization_vs_batch_outputs(
            records,
            options.output_dir,
            formats,
            panel_n_over_w=options.utilization_panel_n_over_w,
            communication_panels=options.utilization_communication_panels
            or UTILIZATION_DEFAULT_COMMUNICATION_PANELS,
            include_sequential=options.utilization_include_sequential,
            strict=options.strict,
            require_complete_profile=require_complete_utilization_profile,
        )
    table = table_summary(records, normalized_rows, normalized_summary)
    write_table_outputs(table, options.output_dir, formats)
    metadata = {
        "analysis_timestamp_utc": utc_now(),
        "manifest_id": selected[0]["manifest_id"] if selected else None,
        "manifest_hash": selected[0]["manifest_hash"] if selected else None,
        "git_commit_expected": selected[0].get("git_commit_expected") if selected else None,
        "manifest_metadata": manifest_metadata,
        "aggregation_rule": "median over seeds per workload/method, geometric mean across workloads",
        "normalization_baseline_method": options.normalization_baseline,
        "seed_aggregation_rule": "median",
        "groups": sorted(groups),
        "modes": sorted(modes),
        "convergence": convergence.get("convergence_figure_status"),
        "utilization_vs_batch": utilization.get("utilization_figure_status"),
        "solver_machinery_ablation": solver_machinery.get("status"),
    }
    write_text_summary(
        output_dir=options.output_dir,
        metadata=metadata,
        preflight_summary=preflight_summary,
        normalized_summary=normalized_summary,
        table=table,
        solver_machinery=solver_machinery,
        groups=sorted(groups),
        normalization_baseline=options.normalization_baseline,
    )
    return {
        "metadata": metadata,
        "preflight_summary": preflight_summary,
        "normalized_rows": normalized_rows,
        "normalized_summary": normalized_summary,
        "convergence": convergence,
        "utilization_vs_batch": utilization,
        "table_summary": table,
        "solver_machinery_ablation": solver_machinery,
    }


def parse_csv_set(value: str | None, default: set[str] | None = None) -> set[str] | None:
    if value is None:
        return default
    return {token.strip() for token in value.split(",") if token.strip()}


def parse_csv_list(value: str | None, default: Sequence[str]) -> list[str]:
    if value is None:
        return list(default)
    return [token.strip() for token in value.split(",") if token.strip()]


def parse_bool_text(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {value!r}")


def should_emit_utilization_vs_batch(
    options: AnalysisOptions,
    rows: Sequence[Mapping[str, Any]],
    manifest_metadata: Mapping[str, Any],
) -> bool:
    if options.emit_utilization_vs_batch:
        return True
    tokens = [
        str(manifest_metadata.get("manifest_name") or ""),
        str(manifest_metadata.get("manifest_id") or ""),
    ]
    tokens.extend(str(row.get("experiment_group") or "") for row in rows)
    return any("big_5min_1f1b_uncapped" in token for token in tokens)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=["uncapped", "equal-memory", "both"], default="both")
    parser.add_argument("--include-groups", default=None)
    parser.add_argument("--strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.add_argument("--allow-unavailable", action="store_true")
    parser.add_argument("--strict-convergence", action="store_true")
    parser.add_argument("--format", default="pdf,png,csv,json,tex")
    parser.add_argument(
        "--normalization-baseline",
        choices=cal_manifest.CONTRACTED_METHODS,
        default=DEFAULT_NORMALIZATION_BASELINE,
    )
    parser.add_argument("--emit-utilization-vs-batch", action="store_true")
    parser.add_argument(
        "--utilization-panel-n-over-w",
        default=str(UTILIZATION_DEFAULT_N_OVER_W),
        help="N/W filter for utilization-vs-batch, or 'all' for {1,2,4}",
    )
    parser.add_argument(
        "--utilization-aggregate-over-n-over-w",
        action="store_true",
        help="alias for --utilization-panel-n-over-w all",
    )
    parser.add_argument(
        "--utilization-communication-panels",
        default=",".join(UTILIZATION_DEFAULT_COMMUNICATION_PANELS),
    )
    parser.add_argument(
        "--utilization-include-sequential",
        type=parse_bool_text,
        default=False,
    )
    parser.add_argument(
        "--utilization-normalize",
        choices=["no"],
        default="no",
        help="utilization-vs-batch is absolute simulated utilization, not normalized",
    )
    args = parser.parse_args(argv)
    utilization_panel_n_over_w = (
        UTILIZATION_ALL_N_OVER_W
        if args.utilization_aggregate_over_n_over_w
        else args.utilization_panel_n_over_w
    )
    try:
        result = analyze(
            AnalysisOptions(
                manifest=args.manifest,
                results_root=args.results_root,
                output_dir=args.output_dir,
                mode=args.mode,
                include_groups=parse_csv_set(args.include_groups),
                strict=args.strict,
                allow_unavailable=args.allow_unavailable,
                strict_convergence=args.strict_convergence,
                formats=parse_csv_set(args.format, {"pdf", "png", "csv", "json", "tex"}),
                normalization_baseline=args.normalization_baseline,
                emit_utilization_vs_batch=args.emit_utilization_vs_batch,
                utilization_panel_n_over_w=utilization_panel_n_over_w,
                utilization_communication_panels=parse_csv_list(
                    args.utilization_communication_panels,
                    UTILIZATION_DEFAULT_COMMUNICATION_PANELS,
                ),
                utilization_include_sequential=args.utilization_include_sequential,
                utilization_normalize=args.utilization_normalize,
            )
        )
    except Exception as exc:
        print(f"analysis failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "rows": result["preflight_summary"]["total_rows"],
                "preflight": result["preflight_summary"]["class_counts"],
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
