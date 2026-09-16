import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_cal_results  # noqa: E402
import cal_manifest  # noqa: E402


FIXED_TIME = "2026-08-07T00:00:00Z"


_MANIFEST_CACHE = {}


def make_rows(profile="smoke"):
    if profile not in _MANIFEST_CACHE:
        _MANIFEST_CACHE[profile] = cal_manifest.generate_manifest(
            profile=profile,
            binary=None,
            time_limit_seconds=1.0,
            solver_threads=1,
            generated_at_utc=FIXED_TIME,
        )
    rows, meta = _MANIFEST_CACHE[profile]
    return copy.deepcopy(rows), copy.deepcopy(meta)


def rows_for_config(profile="main", activation_cap_mode="none"):
    rows, _ = make_rows(profile)
    config_id = rows[0]["configuration_id"]
    return [
        row
        for row in rows
        if row["configuration_id"] == config_id
        and row["activation_cap_mode"] == activation_cap_mode
    ]


def rows_for_utilization_panel(
    *,
    profile="main_big_5min_1f1b_uncapped",
    communication_profile="moderate",
    n_over_w=2.0,
    batch_size_per_worker=None,
):
    rows, _ = make_rows(profile)
    selected = [
        row
        for row in rows
        if row["activation_cap_mode"] == "none"
        and row["communication_profile"] == communication_profile
        and math.isclose(row["N"] / row["W"], n_over_w)
    ]
    if batch_size_per_worker is not None:
        selected = [
            row
            for row in selected
            if math.isclose(row["B"] / row["W"], batch_size_per_worker)
        ]
    if not selected:
        raise AssertionError("missing utilization panel rows")
    return selected


def valid_payload(
    row,
    *,
    makespan=100,
    optimal=True,
    feasible=True,
    validation_passed=True,
    fallback_used=False,
    trace=None,
    activation_ratio=None,
    incumbent_makespan=None,
    incumbent_valid=None,
    incumbent_hints_effective=None,
    incumbent_bound_effective=None,
):
    canonical = {
        "schema_version": row["expected_schema_version"],
        "budget_policy_version": row["expected_budget_policy_version"],
        "evaluation_method_version": row["expected_evaluation_method_version"],
        "activation_analysis_version": row["expected_activation_analysis_version"],
        "activation_cap_formulation_version": row[
            "expected_activation_cap_formulation_version"
        ],
        "canonical_method": row["canonical_method_expected"],
        "method_contract_hash": row["method_contract_hash_expected"],
        "micro_batches": row["B"],
        "logical_stages": row["N"],
        "physical_workers": row["W"],
        "total_layers": row["L"],
        "mapping_type": row["mapping_type"],
        "stage_to_worker_mapping": row["stage_to_worker_mapping"],
        "forward_cost_ratio_numerator": row["forward_cost_ratio_num"],
        "forward_cost_ratio_denominator": row["forward_cost_ratio_den"],
        "backward_cost_ratio_numerator": row["backward_cost_ratio_num"],
        "backward_cost_ratio_denominator": row["backward_cost_ratio_den"],
        "communication_ticks": row["communication_ticks"],
        "activation_model": row["activation_model"],
        "activation_units_per_layer": row["activation_units_per_layer"],
        "activation_cap_mode": row["activation_cap_mode"],
        "activation_cap_units_per_worker": row["activation_cap_units_per_worker"],
        "activation_cap_source": row["activation_cap_source"],
        "activation_cap_derivation_hash": row["activation_cap_derivation_hash"],
        "requested_time_limit_seconds": row["time_limit_seconds"],
        "random_seed": row["random_seed"],
        "solver_threads": row["solver_threads"],
        "git_commit": row["git_commit_expected"],
        "reported_status": "OPTIMAL" if optimal else "FEASIBLE",
        "solver_status_raw": "OPTIMAL" if optimal else "FEASIBLE",
        "feasible": feasible,
        "optimal": optimal,
        "solver_solution_available": feasible,
        "final_solution_available": feasible,
        "final_solution_source": "cpsat" if feasible else "none",
        "no_solution_reason": None,
        "result_validation_passed": validation_passed,
        "result_validation": {
            "validation_version": row["expected_validation_version"],
            "passed": validation_passed,
        },
        "fallback_used": fallback_used,
        "makespan": makespan,
        "activation_cap_satisfied": True if row["enforce_activation_cap"] else None,
        "activation_cap_enforcement_mode": "none",
        "activation_cap_enforced_in_solver": False,
        "activation_cap_enforced_by_enumeration": False,
        "activation_cap_enforcement_requested": bool(row["enforce_activation_cap"]),
        "activation_cap_constraints_added": False,
    }
    if row["method"] in cal_manifest.DETERMINISTIC_METHODS and row["enforce_activation_cap"]:
        canonical["activation_cap_enforcement_mode"] = (
            "deterministic_postconstruction_check"
        )
    elif row["enforce_activation_cap"]:
        canonical["activation_cap_enforcement_mode"] = "solver"
        canonical["activation_cap_enforced_in_solver"] = True
        canonical["activation_cap_constraints_added"] = True
    if row["fixed_order_partition_backend"] is not None:
        canonical["fixed_order_partition_backend_requested"] = row[
            "fixed_order_partition_backend"
        ]
        canonical["fixed_order_partition_backend_effective"] = row[
            "fixed_order_partition_backend"
        ]
        canonical["cp_sat_launched"] = row["fixed_order_partition_backend"] == "cpsat"
        canonical["cp_sat_models_solved"] = (
            1 if row["fixed_order_partition_backend"] == "cpsat" else 0
        )
    if row["experiment_group"] == "oracle" and optimal:
        canonical["best_objective_bound"] = makespan
        canonical["relative_optimality_gap"] = 0.0
        canonical["enumeration_proved_optimal"] = True
        canonical["optimality_proof_source"] = "exhaustive_enumeration"
    if row.get("solver_machinery_variant"):
        incumbent_method = row["incumbent_method_requested"]
        has_incumbent = incumbent_method != "none"
        effective_bound = (
            row["incumbent_bound_requested"] and has_incumbent
            if incumbent_bound_effective is None
            else incumbent_bound_effective
        )
        effective_hints = (
            row["incumbent_hints_requested"] and has_incumbent
            if incumbent_hints_effective is None
            else incumbent_hints_effective
        )
        effective_incumbent_valid = (
            has_incumbent if incumbent_valid is None else incumbent_valid
        )
        canonical.update(
            {
                "worker_balance_pruning_requested": row[
                    "worker_balance_pruning_requested"
                ],
                "worker_balance_pruning_effective": row[
                    "worker_balance_pruning_requested"
                ],
                "incumbent_method_requested": incumbent_method,
                "incumbent_method_effective": incumbent_method
                if has_incumbent
                else "none",
                "incumbent_bound_requested": row["incumbent_bound_requested"],
                "incumbent_bound_effective": effective_bound,
                "incumbent_hints_requested": row["incumbent_hints_requested"],
                "incumbent_hints_effective": effective_hints,
                "incumbent_hint_count": 1 if effective_hints else 0,
                "incumbent_found": has_incumbent,
                "incumbent_valid": effective_incumbent_valid,
                "incumbent_makespan": incumbent_makespan
                if incumbent_makespan is not None
                else (makespan + 10 if has_incumbent else None),
                "fallback_enabled": row["incumbent_fallback_requested"],
                "external_incumbent_available": effective_incumbent_valid,
                "external_incumbent_used_as_fallback": fallback_used,
                "cp_sat_models_solved": 1,
                "time_to_first_cpsat_feasible_seconds": 0.25,
                "best_objective_bound": makespan - 1,
                "relative_optimality_gap": 0.01,
                "num_branches": 1000,
                "num_conflicts": 100,
            }
        )
    if trace is not None:
        canonical["incumbent_trace"] = trace
    if activation_ratio is not None:
        canonical["activation_peak_ratio_to_uniform"] = {
            "maximum_worker_peak_units_ratio": activation_ratio
        }
    payload = {"canonical_result": canonical}
    return cal_manifest.stamp_result(
        row=row,
        payload=payload,
        command=["fake-slackpipe"],
        start_utc=FIXED_TIME,
        end_utc=FIXED_TIME,
        wall_time_seconds=0.0,
        exit_code=0,
        output_root=Path("/tmp/out"),
        outcome="completed_valid",
    )


def no_solution_payload(row):
    payload = valid_payload(row, makespan=100, optimal=False)
    canonical = payload["canonical_result"]
    canonical.update(
        {
            "reported_status": "NO_VALID_SOLUTION",
            "solver_status_raw": "FEASIBLE",
            "feasible": False,
            "optimal": False,
            "solver_solution_available": True,
            "final_solution_available": False,
            "final_solution_source": "none",
            "no_solution_reason": (
                "activation_cap_replay_rejected_until_deadline"
            ),
            "result_validation_passed": True,
            "makespan": None,
            "fallback_used": False,
            "activation_cap_satisfied": None,
            "activation_cap_enforcement_mode": "solver",
            "activation_cap_enforced_in_solver": True,
            "activation_cap_constraints_added": True,
            "cp_sat_models_solved": 1,
            "incumbent_method_effective": "none",
            "incumbent_bound_effective": False,
            "incumbent_hints_effective": False,
            "incumbent_hint_count": 0,
            "incumbent_found": False,
            "incumbent_valid": False,
            "fallback_enabled": False,
            "external_incumbent_available": False,
            "external_incumbent_used_as_fallback": False,
            "best_objective": 283,
            "best_bound": 276,
            "num_branches": 300000,
            "num_conflicts": 70000,
            "time_to_first_cpsat_feasible_seconds": 0.25,
        }
    )
    payload["runner_outcome"] = "completed_no_solution"
    return payload


def payload_for_target_utilization(row, utilization_percent):
    forward = row["forward_cost_ratio_num"] / row["forward_cost_ratio_den"]
    backward = row["backward_cost_ratio_num"] / row["backward_cost_ratio_den"]
    makespan = (
        100.0
        * row["B"]
        * row["L"]
        * (forward + backward)
        / (row["W"] * utilization_percent)
    )
    return valid_payload(row, makespan=makespan)


def write_result(root, row, payload):
    path = cal_manifest.resolve_output_path(root, row["output_json"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def write_manifest(path, rows, meta=None):
    if meta is None:
        meta = {
            "manifest_id": rows[0]["manifest_id"] if rows else "cal-test",
            "manifest_hash": rows[0]["manifest_hash"] if rows else "hash",
        }
    cal_manifest.write_manifest(rows, meta, path)


def row_by_method(rows, method, seed=None):
    for row in rows:
        if row["method"] != method:
            continue
        if seed is not None and row["random_seed"] != seed:
            continue
        return row
    raise AssertionError(f"missing method {method} seed={seed}")


def clone_as_method(row, method, group="main"):
    cloned = copy.deepcopy(row)
    cloned["experiment_group"] = group
    cloned["method"] = method
    cloned["canonical_method_expected"] = method
    cloned["method_contract_hash_expected"] = cal_manifest.FALLBACK_METHOD_CONTRACT_HASHES[
        method
    ]
    cloned["run_id"] = f"{row['run_id']}__as__{method}"
    cloned["output_json"] = f"{cloned['run_id']}/result.json"
    cloned["stdout_log"] = f"{cloned['run_id']}/stdout.log"
    cloned["stderr_log"] = f"{cloned['run_id']}/stderr.log"
    cloned["fixed_order_partition_backend"] = (
        "cpsat"
        if method
        in {
            "partition-only-fixed-order",
            "sequential-partition-then-schedule",
            "alternating-partition-schedule",
        }
        else None
    )
    return cloned


def solver_machinery_rows_for_config(
    *,
    activation_cap_mode="uniform_baseline",
    variants=None,
):
    rows, _ = make_rows("solver_machinery_ablation")
    config_id = next(
        row["configuration_id"]
        for row in rows
        if row["activation_cap_mode"] == activation_cap_mode
    )
    selected = [
        row
        for row in rows
        if row["configuration_id"] == config_id
        and row["activation_cap_mode"] == activation_cap_mode
    ]
    if variants is not None:
        selected = [
            row
            for row in selected
            if row["solver_machinery_variant"] in set(variants)
        ]
    return selected


class CalAnalysisTests(unittest.TestCase):
    def test_selected_modes_expands_both(self):
        self.assertEqual(
            analyze_cal_results.selected_modes("both"), {"uncapped", "equal_memory"}
        )

    def test_selected_modes_accepts_equal_memory_cli_spelling(self):
        self.assertEqual(
            analyze_cal_results.selected_modes("equal-memory"), {"equal_memory"}
        )

    def test_mode_for_row_distinguishes_uncapped_and_equal_memory(self):
        rows, _ = make_rows("smoke")
        modes = {
            row["activation_cap_mode"]: analyze_cal_results.mode_for_row(row)
            for row in rows
        }
        self.assertEqual(modes["none"], "uncapped")
        self.assertEqual(modes["uniform_baseline"], "equal_memory")

    def test_workload_key_separates_equal_memory_caps(self):
        rows, _ = make_rows("smoke")
        uncapped = next(row for row in rows if row["activation_cap_mode"] == "none")
        capped = next(
            row for row in rows if row["activation_cap_mode"] == "uniform_baseline"
        )
        self.assertNotEqual(
            analyze_cal_results.workload_key(uncapped),
            analyze_cal_results.workload_key(capped),
        )

    def test_valid_uncapped_row_is_classified_for_uncapped(self):
        rows, _ = make_rows("smoke")
        row = row_by_method(
            [row for row in rows if row["activation_cap_mode"] == "none"],
            "partition-only-fixed-order",
        )
        record = analyze_cal_results.classify_loaded_row(row, valid_payload(row))
        self.assertIn("valid_for_uncapped", record["classes"])

    def test_uniform_equal_memory_row_is_eligible_with_deterministic_check(self):
        rows, _ = make_rows("smoke")
        row = row_by_method(
            [row for row in rows if row["activation_cap_mode"] != "none"],
            "uniform-breadth-first",
        )
        record = analyze_cal_results.classify_loaded_row(row, valid_payload(row))
        self.assertIn("valid_for_equal_memory", record["classes"])
        self.assertIn("deterministic_baseline_valid", record["classes"])

    def test_solver_equal_memory_row_is_eligible_only_with_solver_enforcement(self):
        rows, _ = make_rows("smoke")
        row = row_by_method(
            [row for row in rows if row["activation_cap_mode"] != "none"],
            "joint-unrestricted-no-overlap",
        )
        record = analyze_cal_results.classify_loaded_row(row, valid_payload(row))
        self.assertIn("valid_for_equal_memory", record["classes"])

    def test_posthoc_equal_memory_row_is_cap_ineligible(self):
        rows, _ = make_rows("smoke")
        row = row_by_method(
            [row for row in rows if row["activation_cap_mode"] != "none"],
            "joint-unrestricted-no-overlap",
        )
        payload = valid_payload(row)
        payload["canonical_result"]["activation_cap_enforcement_mode"] = "posthoc_only"
        payload["canonical_result"]["activation_cap_enforced_in_solver"] = False
        payload["canonical_result"]["activation_cap_constraints_added"] = False
        record = analyze_cal_results.classify_loaded_row(row, payload)
        self.assertIn("cap_ineligible", record["classes"])
        self.assertNotIn("valid_for_equal_memory", record["classes"])

    def test_runner_unavailable_row_is_not_invalid_when_allowed(self):
        rows, _ = make_rows("smoke")
        row = row_by_method(rows, "schedule-only-uniform")
        payload = cal_manifest.stamp_result(
            row=row,
            payload=cal_manifest.synthetic_unavailable_payload("no OR-Tools"),
            command=["fake-slackpipe"],
            start_utc=FIXED_TIME,
            end_utc=FIXED_TIME,
            wall_time_seconds=0.0,
            exit_code=0,
            output_root=Path("/tmp/out"),
            outcome="unavailable",
        )
        record = analyze_cal_results.classify_loaded_row(row, payload)
        self.assertEqual(record["primary_class"], "unavailable")
        self.assertNotIn("invalid_result", record["classes"])

    def test_missing_result_row_is_classified(self):
        rows, _ = make_rows("smoke")
        with tempfile.TemporaryDirectory() as tmp:
            records = analyze_cal_results.classify_manifest_rows([rows[0]], Path(tmp))
        self.assertEqual(records[0]["primary_class"], "missing_result")

    def test_schema_mismatch_row_is_classified(self):
        rows, _ = make_rows("smoke")
        row = rows[0]
        payload = valid_payload(row)
        payload["canonical_result"]["schema_version"] = 0
        record = analyze_cal_results.classify_loaded_row(row, payload)
        self.assertIn("schema_mismatch", record["classes"])

    def test_validation_version_mismatch_row_is_classified(self):
        rows, _ = make_rows("smoke")
        row = rows[0]
        payload = valid_payload(row)
        payload["canonical_result"]["result_validation"]["validation_version"] = 0
        record = analyze_cal_results.classify_loaded_row(row, payload)
        self.assertIn("schema_mismatch", record["classes"])

    def test_manifest_mismatch_row_is_classified(self):
        rows, _ = make_rows("smoke")
        row = rows[0]
        payload = valid_payload(row)
        payload["manifest_hash"] = "wrong"
        record = analyze_cal_results.classify_loaded_row(row, payload)
        self.assertIn("manifest_mismatch", record["classes"])

    def test_method_contract_mismatch_row_is_classified(self):
        rows, _ = make_rows("smoke")
        row = rows[0]
        payload = valid_payload(row)
        payload["canonical_result"]["method_contract_hash"] = "wrong"
        record = analyze_cal_results.classify_loaded_row(row, payload)
        self.assertIn("method_contract_mismatch", record["classes"])

    def test_validation_failed_row_is_classified(self):
        rows, _ = make_rows("smoke")
        row = rows[0]
        payload = valid_payload(row, validation_passed=False)
        record = analyze_cal_results.classify_loaded_row(row, payload)
        self.assertIn("validation_failed", record["classes"])

    def test_fallback_row_keeps_valid_class_and_records_fallback(self):
        rows, _ = make_rows("smoke")
        row = rows[0]
        payload = valid_payload(row, fallback_used=True)
        record = analyze_cal_results.classify_loaded_row(row, payload)
        self.assertIn("valid_for_uncapped", record["classes"])
        self.assertIn("fallback_used", record["classes"])

    def test_not_optimal_but_feasible_row_is_recorded(self):
        rows, _ = make_rows("smoke")
        row = rows[0]
        record = analyze_cal_results.classify_loaded_row(
            row, valid_payload(row, optimal=False)
        )
        self.assertIn("not_optimal_but_feasible", record["classes"])

    def test_no_solution_row_is_classified_without_valid_makespan_class(self):
        row = next(
            row
            for row in solver_machinery_rows_for_config(
                variants={"bare-joint-cpsat"}
            )
            if row["random_seed"] == 0
        )
        record = analyze_cal_results.classify_loaded_row(
            row, no_solution_payload(row)
        )
        self.assertEqual(record["primary_class"], "no_solution")
        self.assertIn("no_solution", record["classes"])
        self.assertNotIn("valid_for_equal_memory", record["classes"])
        self.assertNotIn("valid_for_uncapped", record["classes"])

    def test_solver_machinery_summary_counts_no_solution_and_excludes_makespan(self):
        rows = solver_machinery_rows_for_config(
            variants={"production-slackpipe", "bare-joint-cpsat"}
        )
        records = []
        for row in rows:
            if row["random_seed"] == 0:
                if row["solver_machinery_variant"] == "production-slackpipe":
                    records.append(
                        analyze_cal_results.classify_loaded_row(
                            row, valid_payload(row, makespan=100)
                        )
                    )
                else:
                    records.append(
                        analyze_cal_results.classify_loaded_row(
                            row, no_solution_payload(row)
                        )
                    )
        summary = analyze_cal_results.aggregate_solver_machinery_ablation(records)
        bare = next(
            row
            for row in summary["variant_summary"]
            if row["variant"] == "bare-joint-cpsat"
        )
        self.assertEqual(bare["total_runs"], 1)
        self.assertEqual(bare["valid_solution_runs"], 0)
        self.assertEqual(bare["no_solution_runs"], 1)
        self.assertEqual(bare["valid_solution_rate"], 0.0)
        self.assertEqual(bare["paired_attempt_rows"], 1)
        self.assertEqual(bare["paired_successful_rows"], 0)
        self.assertIsNone(bare["final_makespan_over_production_geomean"])
        self.assertEqual(bare["no_solution_vs_valid_production"], 1)

    def test_solver_machinery_pair_counts_use_jointly_successful_rows(self):
        rows = solver_machinery_rows_for_config(
            variants={"production-slackpipe", "bare-joint-cpsat"}
        )
        records = []
        for row in rows:
            seed = row["random_seed"]
            variant = row["solver_machinery_variant"]
            if variant == "production-slackpipe":
                records.append(
                    analyze_cal_results.classify_loaded_row(
                        row, valid_payload(row, makespan=100)
                    )
                )
            elif seed in {0, 1}:
                records.append(
                    analyze_cal_results.classify_loaded_row(
                        row,
                        valid_payload(
                            row,
                            makespan={0: 200, 1: 800}[seed],
                        ),
                    )
                )
            else:
                records.append(
                    analyze_cal_results.classify_loaded_row(
                        row, no_solution_payload(row)
                    )
                )
        summary = analyze_cal_results.aggregate_solver_machinery_ablation(records)
        bare = next(
            row
            for row in summary["variant_summary"]
            if row["variant"] == "bare-joint-cpsat"
        )
        self.assertEqual(bare["paired_attempt_rows"], 3)
        self.assertEqual(bare["paired_successful_rows"], 2)
        self.assertEqual(bare["valid_solution_runs"], 2)
        self.assertEqual(bare["no_solution_runs"], 1)
        self.assertAlmostEqual(bare["valid_solution_rate"], 2 / 3)
        self.assertAlmostEqual(
            bare["final_makespan_over_production_geomean"], 4.0
        )

    def test_solver_machinery_hint_subset_uses_effective_hint_flag(self):
        rows = solver_machinery_rows_for_config(
            variants={"production-slackpipe", "no-incumbent-hints"}
        )
        records = []
        for row in rows:
            seed = row["random_seed"]
            variant = row["solver_machinery_variant"]
            if variant == "production-slackpipe":
                records.append(
                    analyze_cal_results.classify_loaded_row(
                        row,
                        valid_payload(
                            row,
                            makespan=100,
                            incumbent_hints_effective=seed == 0,
                        ),
                    )
                )
            else:
                records.append(
                    analyze_cal_results.classify_loaded_row(
                        row,
                        valid_payload(
                            row,
                            makespan={0: 200, 1: 300, 2: 400}[seed],
                        ),
                    )
                )
        summary = analyze_cal_results.aggregate_solver_machinery_ablation(records)
        subsets = summary["mechanism_eligible_subsets"]["hint_ablation"]
        self.assertEqual(subsets["all_pairs"]["paired_successful_rows"], 3)
        self.assertAlmostEqual(
            subsets["all_pairs"]["final_makespan_over_production_geomean"],
            analyze_cal_results.geometric_mean([2.0, 3.0, 4.0]),
        )
        self.assertEqual(
            subsets["production_hint_effective_subset"][
                "paired_successful_rows"
            ],
            1,
        )
        self.assertEqual(
            subsets["production_hint_effective_subset"][
                "final_makespan_over_production_geomean"
            ],
            2.0,
        )

    def test_solver_machinery_only_strict_analysis_skips_uniform_normalization(self):
        rows = solver_machinery_rows_for_config(
            variants={"production-slackpipe", "bare-joint-cpsat"}
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.jsonl"
            results = root / "results"
            output = root / "analysis"
            write_manifest(manifest, rows)
            for row in rows:
                payload = (
                    no_solution_payload(row)
                    if row["solver_machinery_variant"] == "bare-joint-cpsat"
                    else valid_payload(row)
                )
                write_result(results, row, payload)
            result = analyze_cal_results.analyze(
                analyze_cal_results.AnalysisOptions(
                    manifest=manifest,
                    results_root=results,
                    output_dir=output,
                    mode="equal-memory",
                    include_groups={"solver_machinery_ablation"},
                    formats={"csv", "json"},
                )
            )
            self.assertEqual(
                result["preflight_summary"][
                    "normalization_precondition_failures"
                ],
                [],
            )
            self.assertEqual(
                result["solver_machinery_ablation"]["status"], "available"
            )
            self.assertTrue(
                (output / "solver_machinery_ablation_summary.json").exists()
            )

    def test_oracle_unproven_row_is_recorded_without_optimum_claim(self):
        rows, _ = make_rows("oracle")
        row = rows[0]
        record = analyze_cal_results.classify_loaded_row(
            row, valid_payload(row, optimal=False)
        )
        self.assertIn("oracle_unproven", record["classes"])

    def test_oracle_exact_enumeration_equal_memory_is_eligible(self):
        rows, _ = make_rows("oracle")
        row = next(row for row in rows if row["enforce_activation_cap"])
        payload = valid_payload(row, makespan=12, optimal=True)
        payload["canonical_result"].update(
            {
                "activation_cap_enforcement_mode": "exact_enumeration",
                "activation_cap_enforced_by_enumeration": True,
                "activation_cap_enforced_in_solver": False,
                "activation_cap_constraints_added": False,
                "cp_sat_launched": False,
                "cp_sat_models_solved": 0,
                "enumeration_candidates_total": 3,
                "enumeration_candidates_valid_schedule": 3,
                "enumeration_candidates_cap_feasible": 2,
                "enumeration_candidates_cap_rejected": 1,
            }
        )
        record = analyze_cal_results.classify_loaded_row(row, payload)
        self.assertIn("valid_for_equal_memory", record["classes"])
        self.assertNotIn("cap_ineligible", record["classes"])

    def test_oracle_partial_enumeration_is_unproven_not_cap_ineligible(self):
        rows, _ = make_rows("oracle")
        row = next(row for row in rows if row["enforce_activation_cap"])
        payload = valid_payload(row, makespan=12, optimal=False)
        payload["canonical_result"].update(
            {
                "activation_cap_enforcement_mode": "partial_enumeration",
                "activation_cap_enforced_by_enumeration": True,
                "activation_cap_enforced_in_solver": False,
                "activation_cap_constraints_added": False,
                "cp_sat_launched": False,
                "cp_sat_models_solved": 0,
                "enumeration_proved_optimal": False,
                "optimality_proof_source": "none",
            }
        )
        record = analyze_cal_results.classify_loaded_row(row, payload)
        self.assertIn("oracle_unproven", record["classes"])
        self.assertNotIn("cap_ineligible", record["classes"])

    def test_stage_mapping_mismatch_is_classified(self):
        rows, _ = make_rows("smoke")
        row = rows[0]
        payload = valid_payload(row)
        payload["canonical_result"]["stage_to_worker_mapping"] = []
        record = analyze_cal_results.classify_loaded_row(row, payload)
        self.assertIn("manifest_mismatch", record["classes"])

    def test_geometric_mean_uses_log_space(self):
        self.assertAlmostEqual(analyze_cal_results.geometric_mean([1, 4]), 2.0)

    def test_geometric_mean_rejects_zero(self):
        with self.assertRaises(ValueError):
            analyze_cal_results.geometric_mean([1, 0])

    def test_trace_points_are_sorted_and_best_so_far(self):
        points = analyze_cal_results.trace_points(
            {"incumbent_trace": [[2.0, 12], [0.1, 20], [1.0, 15]]}
        )
        self.assertEqual(points, [(0.1, 20.0), (1.0, 15.0), (2.0, 12.0)])
        self.assertEqual(analyze_cal_results.best_so_far_at(points, 1.5), 15.0)

    def test_normalized_makespan_uses_uniform_baseline(self):
        rows = rows_for_config("main", "none")
        records = []
        for row in rows:
            makespan = 100
            if row["method"] == "joint-unrestricted-no-overlap":
                makespan = 50
            records.append(
                analyze_cal_results.classify_loaded_row(row, valid_payload(row, makespan=makespan))
            )
        raw, summary = analyze_cal_results.aggregate_normalized_makespan(
            records, "uncapped"
        )
        joint = next(
            row for row in raw if row["method"] == "joint-unrestricted-no-overlap"
        )
        self.assertEqual(joint["normalized_makespan_median"], 0.5)
        self.assertEqual(
            summary["method_summary"]["joint-unrestricted-no-overlap"][
                "geomean_improvement_over_uniform"
            ],
            2.0,
        )

    def test_normalized_makespan_can_use_one_f_one_b_baseline(self):
        rows = rows_for_config("smoke_big_1f1b", "none")
        records = []
        for row in rows:
            makespan = 100
            if row["method"] == "uniform-interleaved-1f1b":
                makespan = 80
            if row["method"] == "joint-unrestricted-no-overlap":
                makespan = 40
            records.append(
                analyze_cal_results.classify_loaded_row(
                    row, valid_payload(row, makespan=makespan)
                )
            )
        raw, summary = analyze_cal_results.aggregate_normalized_makespan(
            records, "uncapped", "uniform-interleaved-1f1b"
        )
        joint = next(
            row for row in raw if row["method"] == "joint-unrestricted-no-overlap"
        )
        breadth_first = next(
            row for row in raw if row["method"] == "uniform-breadth-first"
        )
        self.assertEqual(joint["normalization_baseline_method"], "uniform-interleaved-1f1b")
        self.assertEqual(joint["normalized_makespan_median"], 0.5)
        self.assertEqual(breadth_first["normalized_makespan_median"], 1.25)
        self.assertEqual(
            summary["joint_geomean_improvement_over_methods"][
                "uniform-interleaved-1f1b"
            ],
            2.0,
        )
        comparison = summary["breadth_first_vs_interleaved_1f1b"]
        self.assertEqual(
            comparison[
                "one_f_one_b_strictly_better_than_breadth_first_workloads"
            ],
            1,
        )

    def test_seed_median_is_used_before_workload_geomean(self):
        rows = rows_for_config("main", "none")
        records = []
        for row in rows:
            makespan = 100
            if row["method"] == "joint-unrestricted-no-overlap":
                makespan = {0: 50, 1: 70, 2: 90}[row["random_seed"]]
            records.append(
                analyze_cal_results.classify_loaded_row(row, valid_payload(row, makespan=makespan))
            )
        raw, _ = analyze_cal_results.aggregate_normalized_makespan(records, "uncapped")
        joint = next(
            row for row in raw if row["method"] == "joint-unrestricted-no-overlap"
        )
        self.assertEqual(joint["seed_count"], 3)
        self.assertEqual(joint["makespan_median"], 70.0)

    def test_utilization_vs_batch_computes_formula_and_batch_axis(self):
        row = row_by_method(
            rows_for_utilization_panel(batch_size_per_worker=1.0),
            "uniform-breadth-first",
        )
        records = [
            analyze_cal_results.classify_loaded_row(
                row, valid_payload(row, makespan=96)
            )
        ]
        payload = analyze_cal_results.aggregate_utilization_vs_batch(
            records,
            panel_n_over_w=2.0,
            communication_panels=["moderate"],
            strict=True,
        )
        run_row = payload["run_rows"][0]
        expected = 100.0 * 3.0 * row["B"] * row["L"] / (row["W"] * 96.0)
        self.assertAlmostEqual(run_row["utilization_percent"], expected)
        self.assertEqual(run_row["batch_size_per_worker"], row["B"] / row["W"])

    def test_utilization_vs_batch_takes_seed_median_before_aggregation(self):
        rows = [
            row
            for row in rows_for_utilization_panel(batch_size_per_worker=1.0)
            if row["method"] == "joint-unrestricted-no-overlap"
        ]
        records = []
        for row in rows:
            makespan = {0: 96, 1: 48, 2: 24}[row["random_seed"]]
            records.append(
                analyze_cal_results.classify_loaded_row(
                    row, valid_payload(row, makespan=makespan)
                )
            )
        payload = analyze_cal_results.aggregate_utilization_vs_batch(
            records,
            panel_n_over_w=2.0,
            communication_panels=["moderate"],
            strict=True,
        )
        workload_row = payload["workload_rows"][0]
        expected = 100.0 * 3.0 * rows[0]["B"] * rows[0]["L"] / (
            rows[0]["W"] * 48.0
        )
        self.assertEqual(workload_row["seed_count"], 3)
        self.assertAlmostEqual(workload_row["utilization_percent"], expected)

    def test_utilization_vs_batch_filters_to_requested_n_over_w(self):
        rows, _ = make_rows("main_big_5min_1f1b_uncapped")
        candidate_rows = [
            row
            for row in rows
            if row["activation_cap_mode"] == "none"
            and row["communication_profile"] == "moderate"
            and row["method"] == "uniform-breadth-first"
            and row["B"] / row["W"] == 1.0
            and row["N"] / row["W"] in {1.0, 2.0}
        ]
        records = [
            analyze_cal_results.classify_loaded_row(row, valid_payload(row))
            for row in candidate_rows
        ]
        payload = analyze_cal_results.aggregate_utilization_vs_batch(
            records,
            panel_n_over_w=2.0,
            communication_panels=["moderate"],
            strict=True,
        )
        self.assertTrue(payload["run_rows"])
        self.assertEqual({row["n_over_w"] for row in payload["run_rows"]}, {2.0})

    def test_utilization_vs_batch_splits_communication_panels(self):
        rows = rows_for_utilization_panel(
            communication_profile="moderate", batch_size_per_worker=1.0
        ) + rows_for_utilization_panel(
            communication_profile="heavy", batch_size_per_worker=1.0
        )
        rows = [row for row in rows if row["method"] == "uniform-breadth-first"]
        records = [
            analyze_cal_results.classify_loaded_row(row, valid_payload(row))
            for row in rows
        ]
        payload = analyze_cal_results.aggregate_utilization_vs_batch(
            records,
            panel_n_over_w=2.0,
            communication_panels=["moderate", "heavy"],
            strict=True,
        )
        self.assertEqual(
            {row["communication_profile"] for row in payload["aggregate_rows"]},
            {"moderate", "heavy"},
        )

    def test_utilization_vs_batch_includes_interleaved_1f1b(self):
        rows = rows_for_utilization_panel(batch_size_per_worker=1.0)
        records = [
            analyze_cal_results.classify_loaded_row(row, valid_payload(row))
            for row in rows
        ]
        payload = analyze_cal_results.aggregate_utilization_vs_batch(
            records,
            panel_n_over_w=2.0,
            communication_panels=["moderate"],
            strict=True,
        )
        self.assertIn(
            "uniform-interleaved-1f1b",
            {row["method"] for row in payload["aggregate_rows"]},
        )

    def test_utilization_vs_batch_omits_sequential_from_plot_by_default(self):
        rows = rows_for_utilization_panel(batch_size_per_worker=1.0)
        records = [
            analyze_cal_results.classify_loaded_row(row, valid_payload(row))
            for row in rows
        ]
        payload = analyze_cal_results.aggregate_utilization_vs_batch(
            records,
            panel_n_over_w=2.0,
            communication_panels=["moderate"],
            strict=True,
        )
        sequential = next(
            row
            for row in payload["aggregate_rows"]
            if row["method"] == "sequential-partition-then-schedule"
        )
        self.assertFalse(sequential["plotted"])
        self.assertIn(
            "sequential-partition-then-schedule", payload["backing_methods"]
        )
        self.assertNotIn(
            "sequential-partition-then-schedule", payload["plot_methods"]
        )

    def test_utilization_vs_batch_strict_fails_on_missing_required_method(self):
        rows = [
            row
            for row in rows_for_utilization_panel(batch_size_per_worker=1.0)
            if row["method"] != "joint-unrestricted-no-overlap"
        ]
        records = [
            analyze_cal_results.classify_loaded_row(row, valid_payload(row))
            for row in rows
        ]
        with self.assertRaisesRegex(RuntimeError, "required data missing"):
            analyze_cal_results.aggregate_utilization_vs_batch(
                records,
                panel_n_over_w=2.0,
                communication_panels=["moderate"],
                strict=True,
                require_complete_profile=True,
            )

    def test_utilization_vs_batch_strict_fails_on_invalid_makespan(self):
        row = row_by_method(
            rows_for_utilization_panel(batch_size_per_worker=1.0),
            "uniform-breadth-first",
        )
        records = [
            analyze_cal_results.classify_loaded_row(
                row, valid_payload(row, makespan=0)
            )
        ]
        with self.assertRaisesRegex(RuntimeError, "missing or non-positive makespan"):
            analyze_cal_results.aggregate_utilization_vs_batch(
                records,
                panel_n_over_w=2.0,
                communication_panels=["moderate"],
                strict=True,
            )

    def test_smoke_big_profile_can_emit_utilization_figure(self):
        rows, meta = make_rows("smoke_big_1f1b")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.jsonl"
            results = root / "results"
            output = root / "analysis"
            write_manifest(manifest, rows, meta)
            for row in rows:
                write_result(results, row, valid_payload(row))
            result = analyze_cal_results.analyze(
                analyze_cal_results.AnalysisOptions(
                    manifest=manifest,
                    results_root=results,
                    output_dir=output,
                    mode="uncapped",
                    include_groups={"smoke_big_1f1b"},
                    formats={"pdf", "csv", "json"},
                    emit_utilization_vs_batch=True,
                    utilization_communication_panels=["none"],
                )
            )
            self.assertEqual(
                result["utilization_vs_batch"]["utilization_figure_status"],
                "available",
            )
            self.assertTrue(
                (output / "fig_slackpipe_utilization_vs_batch.pdf").exists()
            )
            self.assertTrue(
                (output / "fig_slackpipe_utilization_vs_batch.csv").exists()
            )
            self.assertTrue(
                (output / "fig_slackpipe_utilization_vs_batch.json").exists()
            )

    def test_all_nw_utilization_produces_requested_batch_axis(self):
        rows, _ = make_rows("main_big_5min_1f1b_uncapped")
        records = [
            analyze_cal_results.classify_loaded_row(row, valid_payload(row))
            for row in rows
        ]
        payload = analyze_cal_results.aggregate_utilization_vs_batch(
            records,
            panel_n_over_w="all",
            communication_panels=["moderate", "heavy"],
            strict=True,
        )
        self.assertEqual(
            {row["batch_size_per_worker"] for row in payload["aggregate_rows"]},
            {0.5, 1.0, 2.0, 4.0, 8.0},
        )
        self.assertEqual(payload["included_n_over_w_values"], [1.0, 2.0, 4.0])
        self.assertEqual(payload["output_stem"], "fig_slackpipe_utilization_vs_batch_all_nw")

    def test_all_nw_utilization_includes_only_paper_methods(self):
        rows, _ = make_rows("main_big_5min_1f1b_uncapped")
        records = [
            analyze_cal_results.classify_loaded_row(row, valid_payload(row))
            for row in rows
        ]
        payload = analyze_cal_results.aggregate_utilization_vs_batch(
            records,
            panel_n_over_w="all",
            communication_panels=["moderate", "heavy"],
            strict=True,
        )
        expected = {
            "uniform-breadth-first",
            "partition-only-fixed-order",
            "schedule-only-uniform",
            "joint-unrestricted-no-overlap",
        }
        self.assertEqual({row["method"] for row in payload["aggregate_rows"]}, expected)
        self.assertEqual(
            payload["plot_methods"],
            [
                "uniform-breadth-first",
                "partition-only-fixed-order",
                "schedule-only-uniform",
                "joint-unrestricted-no-overlap",
            ],
        )
        labels = {row["method"]: row["method_label"] for row in payload["aggregate_rows"]}
        self.assertEqual(labels["uniform-breadth-first"], "BFS")
        self.assertEqual(labels["joint-unrestricted-no-overlap"], "SlackPipe")

    def test_all_nw_utilization_aggregates_across_n_over_w(self):
        rows, _ = make_rows("main_big_5min_1f1b_uncapped")
        bf_rows = [
            row
            for row in rows
            if row["method"] == "uniform-breadth-first"
            and row["communication_profile"] == "moderate"
            and math.isclose(row["B"] / row["W"], 1.0)
            and row["N"] / row["W"] in {1.0, 2.0}
        ]
        by_nw = {}
        for row in bf_rows:
            by_nw.setdefault(row["N"] / row["W"], row)
        selected = [by_nw[1.0], by_nw[2.0]]
        records = [
            analyze_cal_results.classify_loaded_row(
                selected[0], payload_for_target_utilization(selected[0], 10.0)
            ),
            analyze_cal_results.classify_loaded_row(
                selected[1], payload_for_target_utilization(selected[1], 40.0)
            ),
        ]
        payload = analyze_cal_results.aggregate_utilization_vs_batch(
            records,
            panel_n_over_w="all",
            communication_panels=["moderate"],
            strict=False,
        )
        aggregate = payload["aggregate_rows"][0]
        self.assertEqual(aggregate["workload_count"], 2)
        self.assertAlmostEqual(aggregate["utilization_percent"], 20.0)

    def test_all_nw_utilization_takes_seed_median_before_geomean(self):
        rows = [
            row
            for row in rows_for_utilization_panel(
                batch_size_per_worker=1.0, n_over_w=1.0
            )
            if row["method"] == "joint-unrestricted-no-overlap"
        ]
        records = []
        for row in rows:
            target = {0: 10.0, 1: 20.0, 2: 80.0}[row["random_seed"]]
            records.append(
                analyze_cal_results.classify_loaded_row(
                    row, payload_for_target_utilization(row, target)
                )
            )
        payload = analyze_cal_results.aggregate_utilization_vs_batch(
            records,
            panel_n_over_w="all",
            communication_panels=["moderate"],
            strict=False,
        )
        self.assertAlmostEqual(
            payload["workload_rows"][0]["utilization_percent"], 20.0
        )
        self.assertAlmostEqual(
            payload["aggregate_rows"][0]["utilization_percent"], 20.0
        )

    def test_all_nw_utilization_strict_fails_when_endpoint_batch_missing(self):
        rows, _ = make_rows("main_big_5min_1f1b_uncapped")
        rows = [
            row
            for row in rows
            if row["communication_profile"] in {"moderate", "heavy"}
            and row["method"]
            in {
                "uniform-breadth-first",
                "partition-only-fixed-order",
                "schedule-only-uniform",
                "joint-unrestricted-no-overlap",
            }
            and row["B"] / row["W"] not in {0.5, 8.0}
        ]
        records = [
            analyze_cal_results.classify_loaded_row(row, valid_payload(row))
            for row in rows
        ]
        with self.assertRaisesRegex(RuntimeError, "missing required utilization batch sizes"):
            analyze_cal_results.aggregate_utilization_vs_batch(
                records,
                panel_n_over_w="all",
                communication_panels=["moderate", "heavy"],
                strict=True,
            )

    def test_all_nw_utilization_strict_fails_when_required_method_missing(self):
        rows, _ = make_rows("main_big_5min_1f1b_uncapped")
        rows = [
            row
            for row in rows
            if row["communication_profile"] in {"moderate", "heavy"}
            and row["method"] != "joint-unrestricted-no-overlap"
        ]
        records = [
            analyze_cal_results.classify_loaded_row(row, valid_payload(row))
            for row in rows
        ]
        with self.assertRaisesRegex(RuntimeError, "missing required utilization methods"):
            analyze_cal_results.aggregate_utilization_vs_batch(
                records,
                panel_n_over_w="all",
                communication_panels=["moderate", "heavy"],
                strict=True,
            )

    def test_all_nw_utilization_writes_pgfplots_snippet(self):
        rows, _ = make_rows("main_big_5min_1f1b_uncapped")
        records = [
            analyze_cal_results.classify_loaded_row(row, valid_payload(row))
            for row in rows
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            analyze_cal_results.write_utilization_vs_batch_outputs(
                records,
                output,
                {"tex", "csv", "json"},
                panel_n_over_w="all",
                communication_panels=["moderate", "heavy"],
                strict=True,
            )
            tex = (
                output / "fig_slackpipe_utilization_vs_batch_all_nw.tex"
            ).read_text(encoding="utf-8")
        self.assertIn("xticks: {1/2,1,2,4,8}", tex)
        self.assertIn("xticklabels={{1/2},{1},{2},{4},{8}}", tex)
        self.assertIn("\\addplot+", tex)

    def test_utilization_default_n_over_w_two_behavior_is_unchanged(self):
        rows, _ = make_rows("main_big_5min_1f1b_uncapped")
        records = [
            analyze_cal_results.classify_loaded_row(row, valid_payload(row))
            for row in rows
        ]
        payload = analyze_cal_results.aggregate_utilization_vs_batch(
            records,
            communication_panels=["moderate"],
            strict=True,
        )
        self.assertEqual(
            {row["batch_size_per_worker"] for row in payload["aggregate_rows"]},
            {1.0, 2.0, 4.0},
        )
        self.assertEqual(payload["panel_n_over_w"], 2.0)
        self.assertEqual(payload["output_stem"], "fig_slackpipe_utilization_vs_batch")
        self.assertIn(
            "uniform-interleaved-1f1b",
            {row["method"] for row in payload["aggregate_rows"]},
        )

    def test_complete_workload_requires_all_primary_methods(self):
        rows = rows_for_config("main", "none")
        records = [
            analyze_cal_results.classify_loaded_row(row, valid_payload(row))
            for row in rows
        ]
        _, summary = analyze_cal_results.aggregate_normalized_makespan(
            records, "uncapped"
        )
        self.assertEqual(summary["complete_workload_count"], 1)

    def test_missing_uniform_baseline_prevents_normalization(self):
        rows = [
            row
            for row in rows_for_config("smoke", "none")
            if row["method"] != "uniform-breadth-first"
        ]
        records = [
            analyze_cal_results.classify_loaded_row(row, valid_payload(row))
            for row in rows
        ]
        raw, _ = analyze_cal_results.aggregate_normalized_makespan(
            records, "uncapped"
        )
        failures = analyze_cal_results.normalization_precondition_failures(
            records, ["uncapped"]
        )
        self.assertEqual(raw, [])
        self.assertEqual(failures[0]["reason"], "missing valid uniform-breadth-first baseline")

    def test_joint_loss_cases_are_reported(self):
        rows = rows_for_config("main", "none")
        records = []
        for row in rows:
            makespan = 100
            if row["method"] == "partition-only-fixed-order":
                makespan = 40
            if row["method"] == "joint-unrestricted-no-overlap":
                makespan = 60
            records.append(
                analyze_cal_results.classify_loaded_row(row, valid_payload(row, makespan=makespan))
            )
        _, summary = analyze_cal_results.aggregate_normalized_makespan(
            records, "uncapped"
        )
        self.assertEqual(len(summary["joint_loss_workloads"]), 1)

    def test_ties_are_not_reported_as_joint_losses(self):
        rows = rows_for_config("smoke", "none")
        records = [
            analyze_cal_results.classify_loaded_row(row, valid_payload(row))
            for row in rows
        ]
        _, summary = analyze_cal_results.aggregate_normalized_makespan(
            records, "uncapped"
        )
        self.assertEqual(len(summary["joint_loss_workloads"]), 0)
        self.assertEqual(
            sum(summary["workload_win_counts"].values()),
            0,
        )

    def test_equal_memory_cap_ineligible_rows_are_excluded_from_aggregation(self):
        rows = rows_for_config("smoke", "uniform_baseline")
        records = []
        for row in rows:
            payload = valid_payload(row)
            if row["method"] == "joint-unrestricted-no-overlap":
                payload["canonical_result"]["activation_cap_enforcement_mode"] = (
                    "posthoc_only"
                )
                payload["canonical_result"]["activation_cap_enforced_in_solver"] = False
                payload["canonical_result"]["activation_cap_constraints_added"] = False
            records.append(analyze_cal_results.classify_loaded_row(row, payload))
        raw, _ = analyze_cal_results.aggregate_normalized_makespan(
            records, "equal_memory"
        )
        self.assertFalse(
            any(row["method"] == "joint-unrestricted-no-overlap" for row in raw)
        )

    def test_convergence_without_trace_writes_unavailable_status(self):
        rows = rows_for_config("smoke", "none")
        records = [
            analyze_cal_results.classify_loaded_row(row, valid_payload(row))
            for row in rows
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            payload = analyze_cal_results.convergence_data(
                records, output, {"csv", "json"}, strict=False
            )
            status = json.loads(
                (output / "fig_cal_search_convergence.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(payload["convergence_figure_status"], "unavailable_no_trace_data")
        self.assertEqual(status["convergence_figure_status"], "unavailable_no_trace_data")

    def test_convergence_with_trace_interpolates_checkpoints(self):
        rows = rows_for_config("smoke", "none")
        records = []
        for row in rows:
            trace = None
            makespan = 100
            if row["method"] == "joint-unrestricted-no-overlap":
                trace = [[0.05, 90], [0.5, 80], [2.0, 70]]
                makespan = 70
            records.append(
                analyze_cal_results.classify_loaded_row(
                    row, valid_payload(row, makespan=makespan, trace=trace)
                )
            )
        with tempfile.TemporaryDirectory() as tmp:
            payload = analyze_cal_results.convergence_data(
                records, Path(tmp), {"csv", "json"}, strict=False
            )
        self.assertEqual(payload["convergence_figure_status"], "available")
        aggregate = payload["aggregate_rows"]
        self.assertTrue(any(row["checkpoint_seconds"] == 0.1 for row in aggregate))
        at_one = next(row for row in aggregate if row["checkpoint_seconds"] == 1.0)
        self.assertAlmostEqual(at_one["geomean_normalized_best_makespan"], 0.8)

    def test_convergence_marks_no_feasible_solution_before_checkpoint(self):
        rows = rows_for_config("smoke", "none")
        records = []
        for row in rows:
            trace = None
            if row["method"] == "joint-unrestricted-no-overlap":
                trace = [[0.5, 80]]
            records.append(
                analyze_cal_results.classify_loaded_row(
                    row, valid_payload(row, makespan=100, trace=trace)
                )
            )
        with tempfile.TemporaryDirectory() as tmp:
            payload = analyze_cal_results.convergence_data(
                records, Path(tmp), {"json"}, strict=False
            )
        at_point_one = next(
            row
            for row in payload["aggregate_rows"]
            if row["checkpoint_seconds"] == 0.1
        )
        self.assertEqual(at_point_one["feasible_fraction"], 0.0)
        self.assertIsNone(at_point_one["geomean_normalized_best_makespan"])

    def test_convergence_omits_checkpoints_past_row_limit(self):
        rows = rows_for_config("smoke", "none")
        records = []
        for row in rows:
            trace = [[0.05, 90], [0.5, 80]]
            records.append(
                analyze_cal_results.classify_loaded_row(
                    row, valid_payload(row, trace=trace)
                )
            )
        with tempfile.TemporaryDirectory() as tmp:
            payload = analyze_cal_results.convergence_data(
                records, Path(tmp), {"json"}, strict=False
            )
        checkpoints = {
            row["checkpoint_seconds"] for row in payload["aggregate_rows"]
        }
        self.assertEqual(checkpoints, {0.1, 1.0})

    def test_communication_profile_grouping_is_correct(self):
        rows = [
            {
                "analysis_mode": "uncapped",
                "communication_profile": "none",
                "method": "joint-unrestricted-no-overlap",
                "normalized_makespan_median": 0.5,
            },
            {
                "analysis_mode": "uncapped",
                "communication_profile": "none",
                "method": "partition-only-fixed-order",
                "normalized_makespan_median": 0.75,
            },
            {
                "analysis_mode": "uncapped",
                "communication_profile": "heavy",
                "method": "joint-unrestricted-no-overlap",
                "normalized_makespan_median": 0.8,
            },
            {
                "analysis_mode": "uncapped",
                "communication_profile": "heavy",
                "method": "schedule-only-uniform",
                "normalized_makespan_median": 0.9,
            },
        ]
        grouped = analyze_cal_results.communication_summary(rows)
        by_profile = {row["communication_profile"]: row for row in grouped}
        self.assertEqual(by_profile["none"]["joint_improvement_over_best_separate_baseline"], 1.5)
        self.assertAlmostEqual(
            by_profile["heavy"]["joint_improvement_over_best_separate_baseline"],
            1.125,
        )

    def test_stage_multiplier_grouping_is_correct(self):
        rows = [
            {
                "analysis_mode": "uncapped",
                "stage_multiplier": 1.0,
                "batch_ratio": 1.0,
                "method": "joint-unrestricted-no-overlap",
                "normalized_makespan_median": 0.5,
            },
            {
                "analysis_mode": "uncapped",
                "stage_multiplier": 1.0,
                "batch_ratio": 1.0,
                "method": "partition-only-fixed-order",
                "normalized_makespan_median": 0.75,
            },
        ]
        grouped = analyze_cal_results.structural_summary(rows)
        self.assertEqual(grouped["stage_multiplier"][0]["stage_multiplier"], 1.0)
        self.assertEqual(
            grouped["stage_multiplier"][0][
                "joint_improvement_over_best_separate_baseline"
            ],
            1.5,
        )

    def test_batch_ratio_grouping_is_correct(self):
        rows = [
            {
                "analysis_mode": "uncapped",
                "stage_multiplier": 1.0,
                "batch_ratio": 2.0,
                "method": "joint-unrestricted-no-overlap",
                "normalized_makespan_median": 0.6,
            },
            {
                "analysis_mode": "uncapped",
                "stage_multiplier": 1.0,
                "batch_ratio": 2.0,
                "method": "schedule-only-uniform",
                "normalized_makespan_median": 0.9,
            },
        ]
        grouped = analyze_cal_results.structural_summary(rows)
        self.assertEqual(grouped["batch_ratio"][0]["batch_ratio"], 2.0)
        self.assertEqual(
            grouped["batch_ratio"][0][
                "joint_improvement_over_best_separate_baseline"
            ],
            1.5,
        )

    def test_oracle_proven_optimum_comparisons_require_proven_oracle(self):
        oracle_rows, _ = make_rows("oracle")
        oracle = row_by_method(
            [row for row in oracle_rows if row["activation_cap_mode"] == "none"],
            "partition-only-fixed-order",
        )
        joint = clone_as_method(oracle, "joint-unrestricted-no-overlap")
        alternating = clone_as_method(oracle, "alternating-partition-schedule")
        records = [
            analyze_cal_results.classify_loaded_row(
                oracle, valid_payload(oracle, makespan=100, optimal=True)
            ),
            analyze_cal_results.classify_loaded_row(
                joint, valid_payload(joint, makespan=110, optimal=False)
            ),
            analyze_cal_results.classify_loaded_row(
                alternating, valid_payload(alternating, makespan=100, optimal=False)
            ),
        ]
        summary = analyze_cal_results.table_summary(records, [], {})
        oracle_summary = summary["oracle"]
        self.assertEqual(oracle_summary["proven_optimum_count"], 1)
        self.assertEqual(oracle_summary["joint_comparable_count"], 1)
        self.assertEqual(oracle_summary["joint_reaches_proven_optimum_percent"], 0.0)
        self.assertAlmostEqual(
            oracle_summary["median_optimality_gap_to_proven_optimum"], 0.1
        )
        self.assertEqual(
            oracle_summary["partition_only_reaches_proven_optimum_percent"], 100.0
        )
        self.assertEqual(
            oracle_summary["alternating_reaches_proven_optimum_percent"], 100.0
        )

    def test_unproven_oracle_does_not_produce_gap_claims(self):
        oracle_rows, _ = make_rows("oracle")
        oracle = row_by_method(
            [row for row in oracle_rows if row["activation_cap_mode"] == "none"],
            "partition-only-fixed-order",
        )
        records = [
            analyze_cal_results.classify_loaded_row(
                oracle, valid_payload(oracle, makespan=100, optimal=False)
            )
        ]
        summary = analyze_cal_results.table_summary(records, [], {})
        oracle_summary = summary["oracle"]
        self.assertEqual(oracle_summary["proven_optimum_count"], 0)
        self.assertIsNone(oracle_summary["median_optimality_gap_to_proven_optimum"])

    def test_table_summary_counts_fallback_and_activation_rows(self):
        rows = rows_for_config("smoke", "none") + rows_for_config(
            "smoke", "uniform_baseline"
        )
        records = []
        for row in rows:
            records.append(
                analyze_cal_results.classify_loaded_row(
                    row,
                    valid_payload(
                        row,
                        fallback_used=row["method"] == "joint-unrestricted-no-overlap",
                        activation_ratio=1.25
                        if row["method"] == "joint-unrestricted-no-overlap"
                        else None,
                    ),
                )
            )
        normalized_rows, normalized_summary = analyze_cal_results.write_normalized_outputs(
            records, ["uncapped", "equal_memory"], Path(tempfile.mkdtemp()), {"json"}
        )
        summary = analyze_cal_results.table_summary(
            records, normalized_rows, normalized_summary
        )
        self.assertEqual(summary["validation_and_fallback"]["fallback_count"], 2)
        self.assertEqual(summary["activation_equal_memory"]["eligible_rows"], 6)
        self.assertEqual(
            summary["activation_uncapped"]["joint_more_activation_than_uniform_workloads"],
            1,
        )

    def test_table_outputs_include_csv_json_and_tex(self):
        summary = {
            "oracle": {"oracle_status": "unavailable"},
            "validation_and_fallback": {
                "total_rows_analyzed": 1,
                "invalid_result_count": 0,
                "fallback_count": 0,
                "unavailable_count": 0,
            },
            "activation_equal_memory": {
                "eligible_rows": 0,
                "cap_violation_count": 0,
                "solver_enforced_capped_rows": 0,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            analyze_cal_results.write_table_outputs(
                summary, output, {"csv", "json", "tex"}
            )
            self.assertTrue((output / "table_cal_summary.csv").exists())
            self.assertTrue((output / "table_cal_summary.json").exists())
            self.assertIn("\\begin{tabular}", (output / "table_cal_summary.tex").read_text())

    def test_full_analyze_writes_expected_artifacts(self):
        rows, meta = make_rows("smoke")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.jsonl"
            results = root / "results"
            output = root / "analysis"
            write_manifest(manifest, rows, meta)
            for row in rows:
                write_result(results, row, valid_payload(row))
            result = analyze_cal_results.analyze(
                analyze_cal_results.AnalysisOptions(
                    manifest=manifest,
                    results_root=results,
                    output_dir=output,
                    mode="both",
                    include_groups={"smoke"},
                    formats={"csv", "json", "tex"},
                )
                )
            self.assertEqual(result["preflight_summary"]["total_rows"], 12)
            for name in [
                "preflight_cal_rows.csv",
                "fig_cal_normalized_makespan.csv",
                "fig_cal_search_convergence.json",
                "table_cal_summary.tex",
                "cal_eval_summary.md",
                "cal_eval_summary.json",
            ]:
                self.assertTrue((output / name).exists(), name)

    def test_strict_analyze_rejects_mixed_manifest_hashes(self):
        rows, meta = make_rows("smoke")
        rows = rows[:2]
        rows[1]["manifest_hash"] = "different"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.jsonl"
            results = root / "results"
            output = root / "analysis"
            write_manifest(manifest, rows, meta)
            for row in rows:
                write_result(results, row, valid_payload(row))
            with self.assertRaisesRegex(RuntimeError, "protocol guard failed"):
                analyze_cal_results.analyze(
                    analyze_cal_results.AnalysisOptions(
                        manifest=manifest,
                        results_root=results,
                        output_dir=output,
                        mode="uncapped",
                        include_groups={"smoke"},
                        formats={"json"},
                    )
                )

    def test_figure_csv_contains_values_from_json_summary(self):
        rows, meta = make_rows("smoke")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.jsonl"
            results = root / "results"
            output = root / "analysis"
            write_manifest(manifest, rows, meta)
            for row in rows:
                makespan = 100
                if row["method"] == "joint-unrestricted-no-overlap":
                    makespan = 50
                write_result(results, row, valid_payload(row, makespan=makespan))
            analyze_cal_results.analyze(
                analyze_cal_results.AnalysisOptions(
                    manifest=manifest,
                    results_root=results,
                    output_dir=output,
                    mode="uncapped",
                    include_groups={"smoke"},
                    formats={"csv", "json"},
                )
            )
            summary = json.loads(
                (output / "fig_cal_normalized_makespan.json").read_text(
                    encoding="utf-8"
                )
            )
            csv_rows = list(
                __import__("csv").DictReader(
                    (output / "fig_cal_normalized_makespan.csv").open(
                        encoding="utf-8"
                    )
                )
            )
        joint_csv = next(
            row for row in csv_rows if row["method"] == "joint-unrestricted-no-overlap"
        )
        joint_json = summary["uncapped"]["method_summary"][
            "joint-unrestricted-no-overlap"
        ]["geomean_normalized_makespan"]
        self.assertEqual(float(joint_csv["normalized_makespan_median"]), joint_json)

    def test_analyze_filters_include_groups(self):
        rows, meta = make_rows("all")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.jsonl"
            results = root / "results"
            output = root / "analysis"
            smoke_rows = [row for row in rows if row["experiment_group"] == "smoke"]
            write_manifest(manifest, rows, meta)
            for row in smoke_rows:
                write_result(results, row, valid_payload(row))
            result = analyze_cal_results.analyze(
                analyze_cal_results.AnalysisOptions(
                    manifest=manifest,
                    results_root=results,
                    output_dir=output,
                    include_groups={"smoke"},
                    formats={"json"},
                )
            )
        self.assertEqual(result["preflight_summary"]["total_rows"], 12)

    def test_strict_analyze_fails_without_uniform_baseline(self):
        rows, meta = make_rows("smoke")
        rows = [row for row in rows if row["method"] != "uniform-breadth-first"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.jsonl"
            results = root / "results"
            write_manifest(manifest, rows, meta)
            for row in rows:
                write_result(results, row, valid_payload(row))
            with self.assertRaisesRegex(RuntimeError, "normalization preflight"):
                analyze_cal_results.analyze(
                    analyze_cal_results.AnalysisOptions(
                        manifest=manifest,
                        results_root=results,
                        output_dir=root / "analysis",
                        formats={"json"},
                    )
                )

    def test_strict_oracle_only_analysis_skips_uniform_normalization(self):
        rows, meta = make_rows("oracle")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.jsonl"
            results = root / "results"
            write_manifest(manifest, rows, meta)
            for row in rows:
                payload = valid_payload(row, makespan=12, optimal=True)
                if row["enforce_activation_cap"]:
                    payload["canonical_result"].update(
                        {
                            "activation_cap_enforcement_mode": "exact_enumeration",
                            "activation_cap_enforced_by_enumeration": True,
                            "activation_cap_enforced_in_solver": False,
                            "activation_cap_constraints_added": False,
                            "cp_sat_launched": False,
                            "cp_sat_models_solved": 0,
                            "enumeration_candidates_total": 3,
                            "enumeration_candidates_valid_schedule": 3,
                            "enumeration_candidates_cap_feasible": 2,
                            "enumeration_candidates_cap_rejected": 1,
                        }
                    )
                write_result(results, row, payload)
            result = analyze_cal_results.analyze(
                analyze_cal_results.AnalysisOptions(
                    manifest=manifest,
                    results_root=results,
                    output_dir=root / "analysis",
                    include_groups={"oracle"},
                    formats={"json"},
                )
            )
        self.assertEqual(result["preflight_summary"]["total_rows"], 4)
        self.assertEqual(
            result["preflight_summary"]["normalization_precondition_failures"],
            [],
        )
        self.assertEqual(
            result["table_summary"]["oracle"]["proven_optimum_count"], 2
        )

    def test_strict_preflight_fails_on_missing_result(self):
        rows, meta = make_rows("smoke")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.jsonl"
            write_manifest(manifest, rows, meta)
            with self.assertRaises(RuntimeError):
                analyze_cal_results.analyze(
                    analyze_cal_results.AnalysisOptions(
                        manifest=manifest,
                        results_root=root / "results",
                        output_dir=root / "analysis",
                        formats={"json"},
                    )
                )

    def test_allow_unavailable_passes_strict_preflight(self):
        rows, meta = make_rows("smoke")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.jsonl"
            results = root / "results"
            write_manifest(manifest, rows, meta)
            for row in rows:
                payload = valid_payload(row)
                if row["method"] == "schedule-only-uniform":
                    payload = cal_manifest.stamp_result(
                        row=row,
                        payload=cal_manifest.synthetic_unavailable_payload("no OR"),
                        command=["fake-slackpipe"],
                        start_utc=FIXED_TIME,
                        end_utc=FIXED_TIME,
                        wall_time_seconds=0.0,
                        exit_code=0,
                        output_root=results,
                        outcome="unavailable",
                    )
                write_result(results, row, payload)
            result = analyze_cal_results.analyze(
                analyze_cal_results.AnalysisOptions(
                    manifest=manifest,
                    results_root=results,
                    output_dir=root / "analysis",
                    allow_unavailable=True,
                    formats={"json"},
                )
            )
        self.assertEqual(result["preflight_summary"]["class_counts"]["unavailable"], 2)

    def test_cli_returns_nonzero_for_strict_missing_result(self):
        rows, meta = make_rows("smoke")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.jsonl"
            write_manifest(manifest, rows, meta)
            code = analyze_cal_results.main(
                [
                    "--manifest",
                    str(manifest),
                    "--results-root",
                    str(root / "results"),
                    "--output-dir",
                    str(root / "analysis"),
                    "--format",
                    "json",
                ]
            )
        self.assertEqual(code, 1)

    def test_cli_accepts_no_strict_for_missing_result(self):
        rows, meta = make_rows("smoke")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.jsonl"
            write_manifest(manifest, rows, meta)
            code = analyze_cal_results.main(
                [
                    "--manifest",
                    str(manifest),
                    "--results-root",
                    str(root / "results"),
                    "--output-dir",
                    str(root / "analysis"),
                    "--no-strict",
                    "--format",
                    "json",
                ]
            )
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
