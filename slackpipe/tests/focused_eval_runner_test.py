import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import focused_eval  # noqa: E402
import analyze_primary_ablation  # noqa: E402


class FocusedEvalValidationTests(unittest.TestCase):
    def test_equal_memory_guard_rejects_posthoc_only_rows(self):
        base = {
            "optimization_mode": "joint",
            "canonical_method": "joint-unrestricted-no-overlap",
            "activation_model": "linear_in_stage_layers",
            "activation_units_per_layer": 1,
            "explicit_stage_activation_units": "",
            "activation_cap_mode": "explicit",
            "activation_cap_units_per_worker": "4;4",
            "activation_cap_derivation_hash": "cap",
            "activation_cap_formulation_version": 1,
            "activation_cap_satisfied": True,
        }
        solver_row = {
            **base,
            "activation_cap_enforcement_requested": True,
            "activation_cap_enforcement_mode": "solver",
            "activation_cap_enforced_in_solver": True,
            "activation_cap_constraints_added": True,
        }
        posthoc_row = {
            **base,
            "activation_cap_enforcement_requested": False,
            "activation_cap_enforcement_mode": "posthoc_only",
            "activation_cap_enforced_in_solver": False,
            "activation_cap_constraints_added": False,
        }

        self.assertTrue(analyze_primary_ablation.equal_memory_eligible(solver_row))
        self.assertFalse(analyze_primary_ablation.equal_memory_eligible(posthoc_row))
        self.assertFalse(
            analyze_primary_ablation.activation_memory_compatible(
                [solver_row, posthoc_row]
            )
        )

    def test_candidate_pool_shape(self):
        instances = focused_eval.candidate_instances()
        self.assertEqual(len(instances), 32)
        self.assertEqual(sorted({spec.N for spec in instances}), [32, 48, 64, 80])
        self.assertEqual(sorted({spec.B for spec in instances}), [24, 32, 48, 64])
        self.assertEqual(sorted({spec.layers_per_stage for spec in instances}), [2, 4])
        for spec in instances:
            self.assertEqual(spec.J, 16)
            self.assertEqual(spec.N % spec.J, 0)
            self.assertGreater(spec.N, spec.J)
            self.assertGreater(spec.B, spec.J)

    def test_focused_algorithm_set(self):
        self.assertEqual(
            [spec.label for spec in focused_eval.focused_algorithms()],
            [
                "optimize-joint",
                "slackpipe-worker-local-b1",
                "slackpipe-worker-local-b2",
            ],
        )

    def test_main_cli_command_uses_fixed_workers_and_limit(self):
        instance = focused_eval.candidate_instances()[0]
        for algo in focused_eval.focused_algorithms():
            command = focused_eval.make_cli_command(
                cli_path=Path("build/release/slackpipe_cli"),
                instance=instance,
                algo=algo,
                seed=1,
                num_workers=focused_eval.DEFAULT_NUM_WORKERS,
                time_limit_seconds=focused_eval.DEFAULT_TIME_LIMIT,
                output_prefix=Path("/tmp/focused"),
            )
            self.assertIn("--num-workers", command)
            self.assertEqual(command[command.index("--num-workers") + 1], "16")
            if algo.algorithm == "optimize-joint":
                self.assertIn("--time-limit-seconds", command)
                self.assertEqual(
                    command[command.index("--time-limit-seconds") + 1], "300.0"
                )
            else:
                self.assertIn("--time-limit-seconds", command)
                self.assertEqual(
                    command[command.index("--time-limit-seconds") + 1],
                    "300.0",
                )


    def test_cpp_scheduler_terms_are_removed(self):
        roots = ["include", "src", "apps"]
        files = [ROOT / "README.md", ROOT / "CMakeLists.txt"]
        for root in roots:
            files.extend((ROOT / root).rglob("*"))
        files.extend([ROOT / "tests" / "core_tests.cc", ROOT / "tests" / "benchmark_tests.cc"])
        terms = [
            "max_" + "dep" + "th",
            "selected_" + "dep" + "th",
            "SlackPipe" + "De" + "pth" + "Result",
            "dep" + "th-indexed",
            "dep" + "th-limited",
            "search " + "dep" + "th",
            "DFS " + "dep" + "th",
        ]
        matches = []
        for path in files:
            if not path.is_file():
                continue
            if any(part in {"build", "node_modules", "results", ".venv"} for part in path.parts):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for term in terms:
                if term in text:
                    matches.append(f"{path.relative_to(ROOT)}:{term}")
        self.assertEqual(matches, [])

    def test_pressure_pruning_terms_are_owned_by_explicit_feature(self):
        roots = ["include", "src", "apps"]
        files = [ROOT / "README.md", ROOT / "CMakeLists.txt"]
        for root in roots:
            files.extend((ROOT / root).rglob("*"))
        files.extend([ROOT / "tests" / "core_tests.cc", ROOT / "tests" / "benchmark_tests.cc"])
        expected_paths = {
            "CMakeLists.txt",
            "apps/slackpipe_cli.cc",
            "include/slackpipe/joint_solver.h",
            "include/slackpipe/pressure_pruning.h",
            "include/slackpipe/slackpipe_solver.h",
            "src/io.cc",
            "src/joint_solver_cpsat.cc",
            "src/pressure_pruning.cc",
            "src/slackpipe_solver.cc",
            "src/slackpipe_solver_cpsat.cc",
            "tests/core_tests.cc",
        }
        matches = []
        for path in files:
            if not path.is_file():
                continue
            if any(part in {"build", "node_modules", "results", ".venv"} for part in path.parts):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            if "pre" + "ssure" in text:
                matches.append(str(path.relative_to(ROOT)))
        self.assertEqual(sorted(set(matches)), sorted(expected_paths))

    def test_resume_requires_matching_json_contents(self):
        instance = focused_eval.candidate_instances()[0]
        algo = focused_eval.focused_algorithms()[0]
        path = Path("/tmp/focused-resume-wrapper.json")
        path.write_text(
            '{"run_id":"wrong","B":24,"N":32,"J":16,"L":64,'
            '"seed":1,"random_seed":1,"algorithm":"optimize-joint",'
            '"time_limit_seconds":300,"num_workers":16,'
            '"budget_policy_version":1,"status":"OPTIMAL"}\n',
            encoding="utf-8",
        )
        self.assertFalse(
            focused_eval.is_complete_run(
                path,
                instance,
                algo,
                seed=1,
                time_limit_seconds=300.0,
                num_workers=16,
            )
        )
        path.unlink()

    def test_resume_rejects_legacy_budget_policy(self):
        instance = focused_eval.candidate_instances()[0]
        algo = focused_eval.focused_algorithms()[0]
        path = Path("/tmp/focused-legacy-budget-wrapper.json")
        path.write_text(
            json.dumps(
                {
                    "run_id": focused_eval.run_id_for(instance, algo, 1),
                    "B": instance.B,
                    "N": instance.N,
                    "J": instance.J,
                    "L": instance.L,
                    "seed": 1,
                    "random_seed": 1,
                    "algorithm": algo.algorithm,
                    "time_limit_seconds": focused_eval.DEFAULT_TIME_LIMIT,
                    "num_workers": focused_eval.DEFAULT_NUM_WORKERS,
                    "status": "OPTIMAL",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.assertFalse(
            focused_eval.is_complete_run(
                path,
                instance,
                algo,
                seed=1,
                time_limit_seconds=focused_eval.DEFAULT_TIME_LIMIT,
                num_workers=focused_eval.DEFAULT_NUM_WORKERS,
            )
        )
        path.unlink()

    def test_unknown_without_incumbent_keeps_empty_trace(self):
        instance = focused_eval.candidate_instances()[0]
        algo = focused_eval.focused_algorithms()[0]
        row = focused_eval.normalize_run(
            instance=instance,
            algo=algo,
            seed=1,
            time_limit_seconds=300.0,
            num_workers=16,
            build_meta={
                "git_commit": "unknown",
                "build_type": "Release",
                "hostname": "test",
                "cpu_information": "test",
            },
            cli_json={"algorithm": "optimize-joint", "status": "UNKNOWN"},
            elapsed_seconds=1.0,
            peak_rss_kb=None,
        )
        self.assertEqual(row["final_objective"], 0)
        self.assertEqual(row["incumbent_trace"], [])

    def test_nonzero_pilot_seed_is_rejected(self):
        with self.assertRaises(ValueError):
            focused_eval.main(
                [
                    "--cli",
                    "build/release/slackpipe_cli",
                    "--phase",
                    "validate",
                    "--pilot-seed",
                    "7",
                ]
            )

    def test_rejects_wrong_worker_count(self):
        spec = focused_eval.InstanceSpec(
            instance_id="bad-j",
            B=24,
            N=32,
            J=8,
            L=64,
            layers_per_stage=2,
        )
        with self.assertRaises(ValueError):
            focused_eval.validate_instance(spec)

    def test_rejects_nondivisible_stage_count(self):
        spec = focused_eval.InstanceSpec(
            instance_id="bad-n-mod",
            B=24,
            N=34,
            J=16,
            L=68,
            layers_per_stage=2,
        )
        with self.assertRaises(ValueError):
            focused_eval.validate_instance(spec)

    def test_rejects_n_leq_j(self):
        spec = focused_eval.InstanceSpec(
            instance_id="bad-n-leq-j",
            B=24,
            N=16,
            J=16,
            L=32,
            layers_per_stage=2,
        )
        with self.assertRaises(ValueError):
            focused_eval.validate_instance(spec)

    def test_rejects_b_leq_j(self):
        spec = focused_eval.InstanceSpec(
            instance_id="bad-b-leq-j",
            B=16,
            N=32,
            J=16,
            L=64,
            layers_per_stage=2,
        )
        with self.assertRaises(ValueError):
            focused_eval.validate_instance(spec)

    def test_rejects_unknown_algorithm(self):
        spec = focused_eval.AlgorithmSpec("optimize-bfs")
        with self.assertRaises(ValueError):
            focused_eval.validate_algorithm_spec(spec)

    def test_rejects_bad_worker_local_budget(self):
        spec = focused_eval.AlgorithmSpec(
            "slackpipe", split_mode="worker-local", worker_move_budget=3
        )
        with self.assertRaises(ValueError):
            focused_eval.validate_algorithm_spec(spec)

    def test_rejects_per_worker_delta(self):
        spec = focused_eval.AlgorithmSpec(
            "slackpipe",
            split_mode="worker-local",
            worker_move_budget=1,
            per_worker_delta=1,
        )
        with self.assertRaises(ValueError):
            focused_eval.validate_algorithm_spec(spec)


if __name__ == "__main__":
    unittest.main()
