import copy
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cal_manifest  # noqa: E402
import generate_cal_manifest  # noqa: E402


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
    return copy.deepcopy(_MANIFEST_CACHE[profile])


def uncached_rows(profile="smoke"):
    return cal_manifest.generate_manifest(
        profile=profile,
        binary=None,
        time_limit_seconds=1.0,
        solver_threads=1,
        generated_at_utc=FIXED_TIME,
    )


def valid_payload(row):
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
        "reported_status": "FEASIBLE",
        "solver_status_raw": "FEASIBLE",
        "feasible": True,
        "optimal": False,
        "solver_solution_available": True,
        "final_solution_available": True,
        "final_solution_source": "cpsat",
        "no_solution_reason": None,
        "result_validation_passed": True,
        "result_validation": {
            "validation_version": row["expected_validation_version"],
            "passed": True,
        },
        "fallback_used": False,
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
    if row.get("solver_machinery_variant"):
        incumbent_method = row["incumbent_method_requested"]
        has_incumbent = incumbent_method != "none"
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
                "incumbent_bound_effective": row["incumbent_bound_requested"]
                and has_incumbent,
                "incumbent_hints_requested": row["incumbent_hints_requested"],
                "incumbent_hints_effective": row["incumbent_hints_requested"]
                and has_incumbent,
                "incumbent_hint_count": 1
                if row["incumbent_hints_requested"] and has_incumbent
                else 0,
                "incumbent_found": has_incumbent,
                "incumbent_valid": has_incumbent,
                "fallback_enabled": row["incumbent_fallback_requested"],
                "external_incumbent_available": has_incumbent,
                "external_incumbent_used_as_fallback": False,
            }
        )
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
    payload = valid_payload(row)
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
            "fallback_used": False,
            "activation_cap_satisfied": None,
            "activation_cap_enforcement_mode": "solver",
            "activation_cap_enforced_in_solver": True,
            "activation_cap_constraints_added": True,
            "cp_sat_models_solved": 1,
            "best_objective_bound": 276,
        }
    )
    payload["runner_outcome"] = "completed_no_solution"
    return payload


class CalManifestTests(unittest.TestCase):
    def test_manifest_hash_is_stable(self):
        rows_a, meta_a = uncached_rows()
        rows_b, meta_b = uncached_rows()
        self.assertEqual(meta_a["manifest_hash"], meta_b["manifest_hash"])
        self.assertEqual(rows_a[0]["manifest_hash"], rows_b[0]["manifest_hash"])

    def test_manifest_hash_changes_when_controlled_field_changes(self):
        rows, meta = make_rows()
        original = meta["manifest_hash"]
        rows[0]["B"] += 1
        meta["total_run_count"] = len(rows)
        changed = cal_manifest.assign_manifest_hash(rows, meta)
        self.assertNotEqual(original, changed)

    def test_run_ids_are_unique(self):
        rows, _ = make_rows("main")
        self.assertEqual(len(rows), len({row["run_id"] for row in rows}))

    def test_smoke_profile_row_count_at_most_12(self):
        rows, meta = make_rows("smoke")
        self.assertLessEqual(len(rows), 12)
        self.assertEqual(len(rows), 12)
        self.assertEqual(meta["total_run_count"], 12)

    def test_main_profile_expected_row_count_is_documented(self):
        rows, meta = make_rows("main")
        self.assertEqual(len(rows), 2592)
        self.assertIn("2592", meta["matrix_definition"]["row_count_formula"])

    def test_big_one_f_one_b_profile_expected_row_count_is_documented(self):
        rows, meta = cal_manifest.generate_manifest(
            profile="main_big_5min_1f1b",
            binary=None,
            generated_at_utc=FIXED_TIME,
        )
        self.assertEqual(len(rows), 2754)
        self.assertIn("2754", meta["matrix_definition"]["row_count_formula"])
        self.assertIn("uniform-interleaved-1f1b", meta["methods"])

    def test_big_one_f_one_b_uncapped_profile_expected_row_count_is_documented(self):
        rows, meta = cal_manifest.generate_manifest(
            profile="main_big_5min_1f1b_uncapped",
            binary=None,
            generated_at_utc=FIXED_TIME,
        )
        self.assertEqual(len(rows), 1377)
        self.assertIn("1377", meta["matrix_definition"]["row_count_formula"])
        self.assertEqual(meta["matrix_definition"]["activation_modes"], ["uncapped"])
        self.assertIn("uniform-breadth-first", meta["methods"])
        self.assertIn("uniform-interleaved-1f1b", meta["methods"])

    def test_solver_machinery_smoke_profile_expected_row_count_is_documented(self):
        rows, meta = cal_manifest.generate_manifest(
            profile="solver_machinery_ablation_smoke",
            binary=None,
            generated_at_utc=FIXED_TIME,
        )
        self.assertEqual(len(rows), 18)
        self.assertIn("18", meta["matrix_definition"]["row_count_formula"])
        self.assertEqual({row["activation_cap_mode"] for row in rows}, {"none"})
        self.assertEqual(
            {row["solver_machinery_variant"] for row in rows},
            {
                "production-slackpipe",
                "canonical-incumbent",
                "no-incumbent-bound",
                "no-incumbent-hints",
                "no-bound-or-hints",
                "bare-joint-cpsat",
            },
        )

    def test_solver_machinery_full_profile_expected_row_count_is_documented(self):
        rows, meta = cal_manifest.generate_manifest(
            profile="solver_machinery_ablation",
            binary=None,
            generated_at_utc=FIXED_TIME,
        )
        self.assertEqual(len(rows), 2916)
        self.assertIn("2916", meta["matrix_definition"]["row_count_formula"])
        self.assertEqual({row["method"] for row in rows}, {"joint-unrestricted-no-overlap"})

    def test_balance_pruning_profile_requires_explicit_tolerance(self):
        with self.assertRaises(ValueError):
            cal_manifest.generate_manifest(
                profile="solver_machinery_balance_pruning_smoke",
                binary=None,
                generated_at_utc=FIXED_TIME,
            )

    def test_balance_pruning_smoke_profile_records_tolerance(self):
        rows, meta = cal_manifest.generate_manifest(
            profile="solver_machinery_balance_pruning_smoke",
            binary=None,
            generated_at_utc=FIXED_TIME,
            worker_balance_tolerance_layers=0,
        )
        self.assertEqual(len(rows), 6)
        self.assertIn("6", meta["matrix_definition"]["row_count_formula"])
        self.assertEqual(
            {row["solver_machinery_variant"] for row in rows},
            {
                "production-plus-balance-pruning",
                "bare-joint-cpsat-plus-balance-pruning",
            },
        )
        self.assertEqual({row["worker_balance_pruning_requested"] for row in rows}, {True})
        self.assertEqual({row["worker_balance_tolerance_layers"] for row in rows}, {0})

    def test_big_one_f_one_b_profile_records_five_minutes_and_eight_threads(self):
        rows, _ = cal_manifest.generate_manifest(
            profile="main_big_5min_1f1b",
            binary=None,
            generated_at_utc=FIXED_TIME,
        )
        self.assertEqual({row["time_limit_seconds"] for row in rows}, {300.0})
        self.assertEqual({row["solver_threads"] for row in rows}, {8})
        partition_rows = [
            row
            for row in rows
            if row["method"] == "partition-only-fixed-order"
        ]
        self.assertTrue(partition_rows)
        self.assertEqual(
            {row["fixed_order_partition_backend"] for row in partition_rows},
            {"cpsat"},
        )

    def test_big_one_f_one_b_uncapped_profile_records_uncapped_big_settings(self):
        rows, _ = cal_manifest.generate_manifest(
            profile="main_big_5min_1f1b_uncapped",
            binary=None,
            generated_at_utc=FIXED_TIME,
        )
        self.assertEqual({row["time_limit_seconds"] for row in rows}, {300.0})
        self.assertEqual({row["solver_threads"] for row in rows}, {8})
        self.assertEqual({row["activation_cap_mode"] for row in rows}, {"none"})
        self.assertEqual({row["enforce_activation_cap"] for row in rows}, {False})
        self.assertEqual(
            {row["activation_cap_units_per_worker"] for row in rows},
            {None},
        )
        self.assertEqual(
            {row["activation_cap_derivation_hash"] for row in rows},
            {None},
        )
        self.assertEqual(
            {row["activation_cap_baseline_method"] for row in rows},
            {None},
        )
        partition_rows = [
            row
            for row in rows
            if row["method"] == "partition-only-fixed-order"
        ]
        self.assertTrue(partition_rows)
        self.assertEqual(
            {row["fixed_order_partition_backend"] for row in partition_rows},
            {"cpsat"},
        )

    def test_solver_machinery_commands_pass_explicit_controls(self):
        rows, _ = make_rows("solver_machinery_ablation_smoke")
        by_variant = {row["solver_machinery_variant"]: row for row in rows}
        with tempfile.TemporaryDirectory() as tmp:
            production = cal_manifest.command_for_row(
                by_variant["production-slackpipe"],
                Path("/bin/slackpipe_cli"),
                Path(tmp),
            )
            bare = cal_manifest.command_for_row(
                by_variant["bare-joint-cpsat"],
                Path("/bin/slackpipe_cli"),
                Path(tmp),
            )
        self.assertIn("--worker-balance-pruning", production)
        self.assertIn("--incumbent-method", production)
        self.assertIn("--incumbent-bound", production)
        self.assertIn("--incumbent-hints", production)
        self.assertIn("slack", production)
        self.assertIn("none", bare)
        self.assertEqual(
            bare[bare.index("--incumbent-bound") + 1],
            "off",
        )
        self.assertEqual(
            bare[bare.index("--incumbent-hints") + 1],
            "off",
        )

    def test_solver_machinery_contract_rejects_bare_joint_with_slack_incumbent(self):
        rows, _ = make_rows("solver_machinery_ablation_smoke")
        bare = next(
            row
            for row in rows
            if row["solver_machinery_variant"] == "bare-joint-cpsat"
        )
        payload = valid_payload(bare)
        payload["canonical_result"]["incumbent_method_effective"] = "slack"
        result = cal_manifest.check_result_against_manifest(bare, payload)
        self.assertFalse(result.passed)
        self.assertIn("incumbent_method_effective", "\n".join(result.reasons))

    def test_solver_machinery_contract_rejects_production_balance_pruning(self):
        rows, _ = make_rows("solver_machinery_ablation_smoke")
        production = next(
            row
            for row in rows
            if row["solver_machinery_variant"] == "production-slackpipe"
        )
        payload = valid_payload(production)
        payload["canonical_result"]["worker_balance_pruning_effective"] = True
        result = cal_manifest.check_result_against_manifest(production, payload)
        self.assertFalse(result.passed)
        self.assertIn("worker_balance_pruning_effective", "\n".join(result.reasons))

    def test_smoke_big_profile_includes_one_f_one_b(self):
        rows, meta = make_rows("smoke_big_1f1b")
        self.assertEqual(len(rows), 14)
        self.assertIn("uniform-interleaved-1f1b", meta["methods"])

    def test_deterministic_methods_are_not_duplicated_by_seed(self):
        rows, _ = make_rows("main")
        uniform = [
            row
            for row in rows
            if row["method"] == "uniform-breadth-first"
            and row["configuration_id"] == rows[0]["configuration_id"]
            and row["activation_cap_mode"] == "none"
        ]
        self.assertEqual(len(uniform), 1)

        big_rows, _ = make_rows("smoke_big_1f1b")
        one_f_one_b = [
            row
            for row in big_rows
            if row["method"] == "uniform-interleaved-1f1b"
            and row["activation_cap_mode"] == "none"
        ]
        self.assertEqual(len(one_f_one_b), 1)

    def test_solver_backed_methods_are_duplicated_by_seed(self):
        rows, _ = make_rows("main")
        config_id = rows[0]["configuration_id"]
        joint = [
            row
            for row in rows
            if row["method"] == "joint-unrestricted-no-overlap"
            and row["configuration_id"] == config_id
            and row["activation_cap_mode"] == "none"
        ]
        self.assertEqual(sorted(row["random_seed"] for row in joint), [0, 1, 2])

    def test_equal_memory_rows_share_one_cap_vector_per_configuration(self):
        rows, _ = make_rows("main")
        by_config = {}
        for row in rows:
            if row["activation_cap_mode"] != "uniform_baseline":
                continue
            key = row["configuration_id"]
            value = (
                tuple(row["activation_cap_units_per_worker"]),
                row["activation_cap_derivation_hash"],
            )
            by_config.setdefault(key, value)
            self.assertEqual(by_config[key], value)
            self.assertEqual(
                row["activation_cap_baseline_method"], "uniform-breadth-first"
            )
            self.assertEqual(
                row["activation_cap_baseline_method_contract_hash"],
                row["method_contract_hash_expected_for_uniform"],
            )

    def test_uncapped_and_equal_memory_rows_are_not_accidentally_aggregated(self):
        rows, _ = make_rows("smoke")
        modes = {row["activation_cap_mode"] for row in rows}
        groups = {row["experiment_group"] for row in rows}
        self.assertEqual(modes, {"none", "uniform_baseline"})
        self.assertEqual(groups, {"smoke"})

    def test_communication_profiles_map_to_exact_scalar_ticks(self):
        self.assertEqual(cal_manifest.COMMUNICATION_TICKS["none"], 0)
        self.assertEqual(cal_manifest.COMMUNICATION_TICKS["moderate"], 2)
        self.assertEqual(cal_manifest.COMMUNICATION_TICKS["heavy"], 8)

    def test_manifest_command_generation_includes_all_controlled_fields(self):
        rows, _ = make_rows("smoke")
        row = next(
            row
            for row in rows
            if row["method"] == "alternating-partition-schedule"
            and row["activation_cap_mode"] == "uniform_baseline"
        )
        command = cal_manifest.command_for_row(row, Path("slackpipe_cli"), Path("out"))
        for flag in [
            "--B",
            "--N",
            "--J",
            "--L",
            "--communication",
            "--ratio-num",
            "--ratio-den",
            "--algorithm",
            "--time-limit-seconds",
            "--num-workers",
            "--random-seed",
            "--activation-model",
            "--activation-cap-mode",
            "--activation-cap-units",
            "--activation-cap-derivation-hash",
            "--activation-cap-enforcement",
            "--fixed-order-partition-backend",
            "--alternating-max-rounds",
            "--output-prefix",
        ]:
            self.assertIn(flag, command)

    def test_oracle_equal_memory_command_uses_exact_enumeration(self):
        rows, _ = make_rows("oracle")
        row = next(row for row in rows if row["enforce_activation_cap"])
        command = cal_manifest.command_for_row(row, Path("slackpipe_cli"), Path("out"))
        self.assertIn("--activation-cap-enforcement", command)
        self.assertEqual(
            command[command.index("--activation-cap-enforcement") + 1],
            "exact-enumeration",
        )

    def test_completed_valid_row_is_skipped(self):
        rows, _ = make_rows("smoke")
        row = rows[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = cal_manifest.resolve_output_path(root, row["output_json"])
            out.parent.mkdir(parents=True)
            out.write_text(json.dumps(valid_payload(row)), encoding="utf-8")
            should_run, outcome, reasons = cal_manifest.should_run_row(row, root)
            self.assertFalse(should_run)
            self.assertEqual(outcome, "skipped_valid")
            self.assertEqual(reasons, [])

    def test_invalid_row_is_rerun(self):
        rows, _ = make_rows("smoke")
        row = rows[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = cal_manifest.resolve_output_path(root, row["output_json"])
            out.parent.mkdir(parents=True)
            out.write_text('{"canonical_result": null}', encoding="utf-8")
            should_run, outcome, _ = cal_manifest.should_run_row(row, root)
            self.assertTrue(should_run)
            self.assertEqual(outcome, "rerun_invalid")

    def test_unavailable_row_is_classified_separately(self):
        rows, _ = make_rows("smoke")
        row = rows[1]
        payload = valid_payload(row)
        payload["canonical_result"]["reported_status"] = "UNAVAILABLE"
        result = cal_manifest.check_result_against_manifest(row, payload)
        self.assertFalse(result.passed)
        self.assertEqual(result.outcome, "unavailable")

    def test_valid_no_solution_schema_is_accepted(self):
        rows, _ = make_rows("solver_machinery_ablation")
        row = next(
            row
            for row in rows
            if row["solver_machinery_variant"] == "bare-joint-cpsat"
            and row["activation_cap_mode"] != "none"
        )
        result = cal_manifest.check_result_against_manifest(
            row, no_solution_payload(row)
        )
        self.assertTrue(result.passed, result.reasons)
        self.assertEqual(result.outcome, "completed_no_solution")

    def test_contradictory_no_solution_schema_is_invalid(self):
        rows, _ = make_rows("solver_machinery_ablation")
        row = next(
            row
            for row in rows
            if row["solver_machinery_variant"] == "bare-joint-cpsat"
            and row["activation_cap_mode"] != "none"
        )
        payload = no_solution_payload(row)
        payload["canonical_result"]["final_solution_available"] = True
        payload["canonical_result"]["makespan"] = 123
        result = cal_manifest.check_result_against_manifest(row, payload)
        self.assertFalse(result.passed)
        self.assertEqual(result.outcome, "completed_invalid")
        self.assertIn("no-solution", "\n".join(result.reasons))

    def test_completed_no_solution_row_is_skipped_with_own_outcome(self):
        rows, _ = make_rows("solver_machinery_ablation")
        row = next(
            row
            for row in rows
            if row["solver_machinery_variant"] == "bare-joint-cpsat"
            and row["activation_cap_mode"] != "none"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = cal_manifest.resolve_output_path(root, row["output_json"])
            out.parent.mkdir(parents=True)
            out.write_text(json.dumps(no_solution_payload(row)), encoding="utf-8")
            should_run, outcome, reasons = cal_manifest.should_run_row(row, root)
            self.assertFalse(should_run)
            self.assertEqual(outcome, "completed_no_solution")
            self.assertEqual(reasons, [])

    def test_fail_fast_continues_on_no_solution(self):
        rows, _ = make_rows("solver_machinery_ablation")
        no_solution_row = next(
            row
            for row in rows
            if row["solver_machinery_variant"] == "bare-joint-cpsat"
            and row["activation_cap_mode"] != "none"
        )
        valid_row = next(
            row
            for row in rows
            if row["solver_machinery_variant"] == "production-slackpipe"
            and row["activation_cap_mode"] != "none"
            and row["configuration_id"] == no_solution_row["configuration_id"]
            and row["random_seed"] == no_solution_row["random_seed"]
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for row, payload in [
                (no_solution_row, no_solution_payload(no_solution_row)),
                (valid_row, valid_payload(valid_row)),
            ]:
                out = cal_manifest.resolve_output_path(root, row["output_json"])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(payload), encoding="utf-8")
            results = cal_manifest.run_manifest(
                rows=[no_solution_row, valid_row],
                binary=Path("/does/not/exist"),
                output_root=root,
                fail_fast=True,
            )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["outcome"], "completed_no_solution")
        self.assertEqual(results[1]["outcome"], "skipped_valid")

    def test_fail_fast_stops_on_invalid_result(self):
        rows, _ = make_rows("smoke")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = cal_manifest.resolve_output_path(root, rows[0]["output_json"])
            out.parent.mkdir(parents=True)
            out.write_text('{"canonical_result": null}', encoding="utf-8")
            results = cal_manifest.run_manifest(
                rows=rows[:2],
                binary=Path("/does/not/exist"),
                output_root=root,
                fail_fast=True,
                no_rerun_invalid=True,
            )
        self.assertEqual(len(results), 1)
        self.assertIn(
            results[0]["outcome"],
            {"completed_invalid", "schema_mismatch", "manifest_mismatch"},
        )

    def test_git_mismatch_causes_failure_unless_policy_allows(self):
        rows, _ = make_rows("smoke")
        row = rows[0]
        row["git_commit_expected"] = "1" * 40
        payload = valid_payload(row)
        payload["canonical_result"]["git_commit"] = "wrong"
        strict = dict(row)
        strict["git_dirty_policy"] = "require_clean"
        self.assertFalse(cal_manifest.check_result_against_manifest(strict, payload).passed)
        loose = dict(row)
        loose["git_dirty_policy"] = "allow_dirty_build"
        self.assertTrue(cal_manifest.check_result_against_manifest(loose, payload).passed)

    def test_method_contract_hash_mismatch_is_detected(self):
        rows, _ = make_rows("smoke")
        row = rows[0]
        payload = valid_payload(row)
        payload["canonical_result"]["method_contract_hash"] = "bad"
        result = cal_manifest.check_result_against_manifest(row, payload)
        self.assertFalse(result.passed)
        self.assertIn("method_contract_hash", "\n".join(result.reasons))

    def test_cap_ineligible_equal_memory_result_is_rejected(self):
        rows, _ = make_rows("smoke")
        row = next(
            row
            for row in rows
            if row["method"] == "joint-unrestricted-no-overlap"
            and row["activation_cap_mode"] == "uniform_baseline"
        )
        payload = valid_payload(row)
        payload["canonical_result"]["activation_cap_enforced_in_solver"] = False
        payload["canonical_result"]["activation_cap_constraints_added"] = False
        payload["canonical_result"]["activation_cap_enforcement_mode"] = "posthoc_only"
        result = cal_manifest.check_result_against_manifest(row, payload)
        self.assertFalse(result.passed)

    def test_oracle_exact_enumeration_equal_memory_result_is_accepted(self):
        rows, _ = make_rows("oracle")
        row = next(row for row in rows if row["enforce_activation_cap"])
        payload = valid_payload(row)
        payload["canonical_result"].update(
            {
                "reported_status": "OPTIMAL",
                "solver_status_raw": "OPTIMAL",
                "feasible": True,
                "optimal": True,
                "makespan": 12,
                "best_objective_bound": 12,
                "relative_optimality_gap": 0.0,
                "activation_cap_enforcement_mode": "exact_enumeration",
                "activation_cap_enforced_by_enumeration": True,
                "activation_cap_enforced_in_solver": False,
                "activation_cap_constraints_added": False,
                "cp_sat_launched": False,
                "cp_sat_models_solved": 0,
                "enumeration_proved_optimal": True,
                "optimality_proof_source": "exhaustive_enumeration",
                "enumeration_candidates_total": 3,
                "enumeration_candidates_valid_schedule": 3,
                "enumeration_candidates_cap_feasible": 2,
                "enumeration_candidates_cap_rejected": 1,
            }
        )
        result = cal_manifest.check_result_against_manifest(row, payload)
        self.assertTrue(result.passed, result.reasons)

    def test_oracle_exact_enumeration_requires_completion_proof(self):
        rows, _ = make_rows("oracle")
        row = next(row for row in rows if row["enforce_activation_cap"])
        payload = valid_payload(row)
        payload["canonical_result"].update(
            {
                "reported_status": "OPTIMAL",
                "solver_status_raw": "OPTIMAL",
                "feasible": True,
                "optimal": True,
                "makespan": 12,
                "best_objective_bound": 12,
                "relative_optimality_gap": 0.0,
                "activation_cap_enforcement_mode": "exact_enumeration",
                "activation_cap_enforced_by_enumeration": True,
                "cp_sat_launched": False,
                "cp_sat_models_solved": 0,
                "enumeration_proved_optimal": False,
                "optimality_proof_source": "none",
            }
        )
        result = cal_manifest.check_result_against_manifest(row, payload)
        self.assertFalse(result.passed)
        self.assertIn("proof", "\n".join(result.reasons))

    def test_main_equal_memory_rejects_fabricated_exact_enumeration(self):
        rows, _ = make_rows("smoke")
        row = next(
            row
            for row in rows
            if row["method"] == "partition-only-fixed-order"
            and row["activation_cap_mode"] == "uniform_baseline"
        )
        payload = valid_payload(row)
        payload["canonical_result"].update(
            {
                "activation_cap_enforcement_mode": "exact_enumeration",
                "activation_cap_enforced_by_enumeration": True,
                "activation_cap_enforced_in_solver": True,
                "activation_cap_constraints_added": True,
                "cp_sat_launched": True,
                "cp_sat_models_solved": 1,
                "enumeration_proved_optimal": True,
                "optimality_proof_source": "exhaustive_enumeration",
            }
        )
        result = cal_manifest.check_result_against_manifest(row, payload)
        self.assertFalse(result.passed)
        self.assertIn("solver activation cap", "\n".join(result.reasons))

    def test_oracle_enumeration_guards_reject_invalid_claims(self):
        rows, _ = make_rows("oracle")
        row = next(row for row in rows if row["enforce_activation_cap"])
        base = valid_payload(row)
        base["canonical_result"].update(
            {
                "reported_status": "OPTIMAL",
                "solver_status_raw": "OPTIMAL",
                "feasible": True,
                "optimal": True,
                "makespan": 12,
                "best_objective_bound": 12,
                "relative_optimality_gap": 0.0,
                "activation_cap_enforcement_mode": "exact_enumeration",
                "activation_cap_enforced_by_enumeration": True,
                "activation_cap_enforced_in_solver": False,
                "activation_cap_constraints_added": False,
                "cp_sat_launched": False,
                "cp_sat_models_solved": 0,
                "enumeration_proved_optimal": True,
                "optimality_proof_source": "exhaustive_enumeration",
            }
        )

        cap_violation = copy.deepcopy(base)
        cap_violation["canonical_result"]["activation_cap_satisfied"] = False
        self.assertFalse(
            cal_manifest.check_result_against_manifest(row, cap_violation).passed
        )

        partial_labeled_optimal = copy.deepcopy(base)
        partial_labeled_optimal["canonical_result"].update(
            {
                "activation_cap_enforcement_mode": "partial_enumeration",
                "enumeration_proved_optimal": False,
                "optimality_proof_source": "none",
            }
        )
        self.assertFalse(
            cal_manifest.check_result_against_manifest(
                row, partial_labeled_optimal
            ).passed
        )

        posthoc = copy.deepcopy(base)
        posthoc["canonical_result"].update(
            {
                "activation_cap_enforcement_mode": "posthoc_only",
                "activation_cap_enforced_by_enumeration": False,
            }
        )
        self.assertFalse(cal_manifest.check_result_against_manifest(row, posthoc).passed)

    def test_dry_run_executes_nothing(self):
        rows, _ = make_rows("smoke")
        with tempfile.TemporaryDirectory() as tmp:
            results = cal_manifest.run_manifest(
                rows=rows[:1],
                binary=Path("/does/not/exist"),
                output_root=Path(tmp),
                dry_run=True,
            )
            self.assertEqual(results[0]["outcome"], "dry_run")
            self.assertFalse(
                cal_manifest.resolve_output_path(Path(tmp), rows[0]["output_json"]).exists()
            )

    def test_filters_select_the_intended_subset(self):
        rows, _ = make_rows("smoke")
        selected = cal_manifest.filter_rows(
            rows,
            group="smoke",
            method="joint-unrestricted-no-overlap",
            configuration_id="smoke_W2_N2_B2_L4_comm-none",
        )
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(row["method"] == "joint-unrestricted-no-overlap" for row in selected))

    def test_runner_does_not_overwrite_valid_output_without_force(self):
        rows, _ = make_rows("smoke")
        row = rows[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = cal_manifest.resolve_output_path(root, row["output_json"])
            out.parent.mkdir(parents=True)
            original = json.dumps(valid_payload(row), sort_keys=True)
            out.write_text(original, encoding="utf-8")
            results = cal_manifest.run_manifest(
                rows=[row],
                binary=Path("/does/not/exist"),
                output_root=root,
                dry_run=False,
            )
            self.assertEqual(results[0]["outcome"], "skipped_valid")
            self.assertEqual(out.read_text(encoding="utf-8"), original)

    def test_runner_refuses_output_root_with_different_manifest_hash(self):
        rows, _ = make_rows("smoke")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / cal_manifest.RUNNER_MANIFEST_MARKER
            marker.write_text(
                json.dumps({"manifest_hash": "different"}, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "refusing to append"):
                cal_manifest.run_manifest(
                    rows=[rows[0]],
                    binary=Path("/does/not/exist"),
                    output_root=root,
                    dry_run=True,
                )

    def test_runner_force_new_manifest_allows_different_manifest_hash(self):
        rows, _ = make_rows("smoke")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / cal_manifest.RUNNER_MANIFEST_MARKER
            marker.write_text(
                json.dumps({"manifest_hash": "different"}, sort_keys=True),
                encoding="utf-8",
            )
            results = cal_manifest.run_manifest(
                rows=[rows[0]],
                binary=Path("/does/not/exist"),
                output_root=root,
                dry_run=True,
                force_new_manifest=True,
            )
            self.assertEqual(results[0]["outcome"], "dry_run")

    def test_runner_warns_on_jobs_times_solver_threads_oversubscription(self):
        rows, _ = cal_manifest.generate_manifest(
            profile="main_big_5min_1f1b",
            binary=None,
            generated_at_utc=FIXED_TIME,
        )
        jobs = (cal_manifest.os.cpu_count() or 1) + 1
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(stderr):
            results = cal_manifest.run_manifest(
                rows=rows[:1],
                binary=Path("/does/not/exist"),
                output_root=Path(tmp),
                jobs=jobs,
                dry_run=True,
            )
        self.assertEqual(results[0]["outcome"], "dry_run")
        self.assertIn("jobs * solver_threads", stderr.getvalue())

    def test_manifest_metadata_total_count_matches_rows(self):
        rows, meta = make_rows("ablation")
        self.assertEqual(meta["total_run_count"], len(rows))
        self.assertEqual(len(rows), 96)

    def test_generate_manifest_accepts_seed_list_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "cal_smoke.jsonl"
            code = generate_cal_manifest.main(
                [
                    "--profile",
                    "smoke",
                    "--output",
                    str(output),
                    "--time-limit-seconds",
                    "1",
                    "--seed-list",
                    "0",
                ]
            )
            rows = cal_manifest.read_manifest(output)
        self.assertEqual(code, 0)
        self.assertEqual(len(rows), 12)

    def test_oracle_rows_are_separate_from_baseline_rows(self):
        rows, _ = make_rows("oracle")
        self.assertTrue(rows)
        self.assertTrue(all(row["experiment_group"] == "oracle" for row in rows))
        self.assertTrue(
            all(row["fixed_order_partition_backend"] == "enumerate" for row in rows)
        )

    def test_python_compatibility_checker_rejects_schema_version_mismatch(self):
        rows, _ = make_rows("smoke")
        row = rows[0]
        payload = valid_payload(row)
        payload["canonical_result"]["schema_version"] = 0
        result = cal_manifest.check_result_against_manifest(row, payload)
        self.assertFalse(result.passed)
        self.assertEqual(result.outcome, "schema_mismatch")

    def test_result_stamping_contains_manifest_hash_and_run_id(self):
        rows, _ = make_rows("smoke")
        row = rows[0]
        stamped = valid_payload(row)
        self.assertEqual(stamped["manifest_hash"], row["manifest_hash"])
        self.assertEqual(stamped["manifest_run_id"], row["run_id"])
        self.assertIn("runner_command", stamped)


if __name__ == "__main__":
    unittest.main()
