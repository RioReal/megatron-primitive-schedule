#!/usr/bin/env python3
"""Print a compact activation-memory diagnostic table from result JSON/JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable


def result_objects(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            for line in handle:
                line = line.strip()
                if line:
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        yield payload
        else:
            payload = json.load(handle)
            if isinstance(payload, dict):
                yield payload


def compact_list(value: Any) -> str:
    if value is None or value == "":
        return ""
    if not isinstance(value, list):
        return str(value)
    return ";".join(str(item) for item in value)


def compact_bool(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "" if value is None else str(value)


def value_from(canonical: dict[str, Any], payload: dict[str, Any], key: str) -> Any:
    if key in canonical:
        return canonical[key]
    return payload.get(key, "")


def truthy(value: Any) -> bool:
    return value in {True, "True", "true", "1", 1}


def equal_memory_eligible(canonical: dict[str, Any], payload: dict[str, Any]) -> bool:
    cap_mode = str(value_from(canonical, payload, "activation_cap_mode") or "none")
    if cap_mode in {"", "none", "None"}:
        return False
    if not truthy(value_from(canonical, payload, "activation_cap_enforcement_requested")):
        return False
    if not truthy(value_from(canonical, payload, "activation_cap_satisfied")):
        return False
    if value_from(canonical, payload, "activation_cap_formulation_version") in {"", None}:
        return False
    mode = str(value_from(canonical, payload, "activation_cap_enforcement_mode") or "")
    if mode == "deterministic_postconstruction_check":
        return True
    if mode != "solver":
        return False
    return truthy(value_from(canonical, payload, "activation_cap_enforced_in_solver")) and truthy(
        value_from(canonical, payload, "activation_cap_constraints_added")
    )


def diagnostic_row(payload: dict[str, Any]) -> dict[str, Any]:
    canonical = payload.get("canonical_result", {})
    if not isinstance(canonical, dict):
        canonical = {}
    metrics = canonical.get("activation_memory_metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    global_metrics = metrics.get("global", {})
    if not isinstance(global_metrics, dict):
        global_metrics = {}
    ratio = canonical.get("activation_peak_ratio_to_uniform", {})
    if not isinstance(ratio, dict):
        ratio = {}
    peak_activation_units = global_metrics.get(
        "maximum_worker_peak_activation_units",
        payload.get("maximum_worker_peak_activation_units", ""),
    )
    global_simultaneous_peak_units = global_metrics.get(
        "peak_simultaneous_activation_units_across_workers",
        payload.get("global_simultaneous_peak_activation_units", ""),
    )
    ratio_to_uniform = ratio.get(
        "maximum_worker_peak_units_ratio",
        payload.get("activation_peak_ratio_to_uniform", ""),
    )
    return {
        "method": payload.get("canonical_method", canonical.get("canonical_method", "")),
        "status": payload.get("status", canonical.get("reported_status", "")),
        "makespan": payload.get(
            "makespan_ticks",
            payload.get("replayed_makespan", payload.get("makespan", "")),
        ),
        "peak_activation_units": peak_activation_units,
        "global_simultaneous_peak_units": global_simultaneous_peak_units,
        "ratio_to_uniform": ratio_to_uniform,
        "cap": compact_list(value_from(canonical, payload, "activation_cap_units_per_worker")),
        "cap_satisfied": compact_bool(value_from(canonical, payload, "activation_cap_satisfied")),
        "cap_enforced": compact_bool(value_from(canonical, payload, "activation_cap_enforced")),
        "cap_enforced_in_solver": compact_bool(
            value_from(canonical, payload, "activation_cap_enforced_in_solver")
        ),
        "enforcement_mode": value_from(canonical, payload, "activation_cap_enforcement_mode"),
        "solver_support": value_from(canonical, payload, "activation_cap_solver_support_level"),
        "constraints_added": compact_bool(
            value_from(canonical, payload, "activation_cap_constraints_added")
        ),
        "retained_intervals": value_from(canonical, payload, "activation_retained_interval_count"),
        "cumulative_constraints": canonical.get(
            "activation_cumulative_constraint_count",
            payload.get("activation_cumulative_constraint_count", ""),
        ),
        "variable_demands": value_from(canonical, payload, "activation_variable_demand_count"),
        "fixed_demands": value_from(canonical, payload, "activation_fixed_demand_count"),
        "incumbent_rejected": compact_bool(
            value_from(canonical, payload, "incumbent_rejected_for_activation_cap")
        ),
        "equal_memory_eligible": compact_bool(equal_memory_eligible(canonical, payload)),
        "activation_model": value_from(canonical, payload, "activation_model"),
        "cap_derivation_hash": value_from(canonical, payload, "activation_cap_derivation_hash"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    fields = [
        "method",
        "status",
        "makespan",
        "peak_activation_units",
        "global_simultaneous_peak_units",
        "ratio_to_uniform",
        "cap",
        "cap_satisfied",
        "cap_enforced",
        "cap_enforced_in_solver",
        "enforcement_mode",
        "solver_support",
        "constraints_added",
        "retained_intervals",
        "cumulative_constraints",
        "variable_demands",
        "fixed_demands",
        "incumbent_rejected",
        "equal_memory_eligible",
        "activation_model",
        "cap_derivation_hash",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fields)
    writer.writeheader()
    for path in args.paths:
        for payload in result_objects(path):
            writer.writerow(diagnostic_row(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
