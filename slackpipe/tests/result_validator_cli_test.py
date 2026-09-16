#!/usr/bin/env python3
import json
import pathlib
import subprocess
import sys
import tempfile


def run(args, check=True):
    return subprocess.run(
        args, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: result_validator_cli_test.py /path/to/slackpipe_cli")
    cli = pathlib.Path(sys.argv[1])
    if not cli.exists():
        raise SystemExit(f"slackpipe_cli not found: {cli}")

    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = pathlib.Path(tmpdir) / "result"
        run(
            [
                str(cli),
                "--algorithm",
                "eval-bfs",
                "--B",
                "1",
                "--N",
                "2",
                "--J",
                "2",
                "--L",
                "4",
                "--ratio-num",
                "2",
                "--split",
                "2,2",
                "--output-prefix",
                str(prefix),
            ]
        )

        valid = run([str(cli), "validate-result", "--input", str(prefix) + ".json"])
        valid_json = json.loads(valid.stdout)
        assert valid_json["passed"] is True
        assert valid_json["reconstructed_makespan"] == 12

        activation = run(
            [
                str(cli),
                "validate-result",
                "--input",
                str(prefix) + ".json",
                "--activation-summary",
            ]
        )
        activation_json = json.loads(activation.stdout)
        activation_canonical = activation_json["canonical_result"]
        assert activation_canonical["activation_analysis_version"] == 1
        assert activation_canonical["activation_model"] == "linear_in_stage_layers"
        assert activation_canonical["activation_memory_metrics"]["global"][
            "maximum_worker_peak_activation_units"
        ] == 2

        described = run([str(cli), "describe-method", "alternating-partition-schedule"])
        described_json = json.loads(described.stdout)
        assert described_json["canonical_name"] == "alternating-partition-schedule"
        assert described_json["evaluation_method_version"] == 1
        assert described_json["partition_optimized"] is True
        assert described_json["schedule_optimized"] is True

        build_info = run([str(cli), "build-info"])
        build_info_json = json.loads(build_info.stdout)
        assert build_info_json["schema_version"] == 1
        assert build_info_json["evaluation_result_schema_version"] == 1
        assert build_info_json["evaluation_method_version"] == 1
        assert build_info_json["budget_policy_version"] == 1
        assert build_info_json["validation_version"] == 1
        assert build_info_json["activation_analysis_version"] == 1
        assert isinstance(build_info_json["ortools_compiled"], bool)
        assert isinstance(build_info_json["ortools_enabled"], bool)
        assert "ortools_version" in build_info_json
        assert "cumulative_constraint_supported" in build_info_json
        assert "variable_cumulative_demand_supported" in build_info_json
        assert build_info_json["activation_cap_solver_support"] in {
            "none",
            "fixed_demands_only",
            "variable_demands",
        }

        deterministic_prefix = pathlib.Path(tmpdir) / "deterministic"
        run(
            [
                str(cli),
                "--algorithm",
                "eval-bfs",
                "--B",
                "2",
                "--N",
                "1",
                "--J",
                "1",
                "--L",
                "1",
                "--ratio-num",
                "2",
                "--activation-cap-mode",
                "explicit",
                "--activation-cap-units",
                "10",
                "--enforce-activation-cap",
                "--output-prefix",
                str(deterministic_prefix),
            ]
        )
        deterministic_json = json.loads((deterministic_prefix.with_suffix(".json")).read_text())
        deterministic_canonical = deterministic_json["canonical_result"]
        assert deterministic_canonical["activation_cap_enforced_in_solver"] is False
        assert (
            deterministic_canonical["activation_cap_enforcement_mode"]
            == "deterministic_postconstruction_check"
        )

        posthoc_prefix = pathlib.Path(tmpdir) / "posthoc"
        run(
            [
                str(cli),
                "--algorithm",
                "eval-bfs",
                "--B",
                "2",
                "--N",
                "1",
                "--J",
                "1",
                "--L",
                "1",
                "--ratio-num",
                "2",
                "--activation-cap-mode",
                "explicit",
                "--activation-cap-units",
                "10",
                "--output-prefix",
                str(posthoc_prefix),
            ]
        )
        posthoc_json = json.loads((posthoc_prefix.with_suffix(".json")).read_text())
        posthoc_canonical = posthoc_json["canonical_result"]
        assert posthoc_canonical["activation_cap_enforcement_requested"] is False
        assert posthoc_canonical["activation_cap_enforcement_mode"] == "posthoc_only"
        assert posthoc_canonical["activation_cap_enforced_in_solver"] is False

        if not build_info_json["ortools_enabled"]:
            unavailable_prefix = pathlib.Path(tmpdir) / "unavailable"
            unavailable = run(
                [
                    str(cli),
                    "--algorithm",
                    "partition-only-fixed-order",
                    "--fixed-order-partition-backend",
                    "cpsat",
                    "--B",
                    "2",
                    "--N",
                    "2",
                    "--J",
                    "2",
                    "--L",
                    "4",
                    "--ratio-num",
                    "2",
                    "--activation-cap-mode",
                    "explicit",
                    "--activation-cap-units",
                    "10",
                    "--enforce-activation-cap",
                    "--output-prefix",
                    str(unavailable_prefix),
                ],
                check=False,
            )
            assert unavailable.returncode == 3
            unavailable_json = json.loads(
                (unavailable_prefix.with_suffix(".json")).read_text()
            )
            unavailable_canonical = unavailable_json["canonical_result"]
            assert unavailable_canonical["reported_status"] == "UNAVAILABLE"
            assert unavailable_canonical["activation_cap_enforcement_requested"] is True
            assert unavailable_canonical["activation_cap_solver_support_level"] == "none"
            assert unavailable_canonical["activation_cap_solver_supported"] is False
            assert unavailable_canonical["activation_cap_constraints_added"] is False
            assert unavailable_canonical["activation_cap_enforced_in_solver"] is False

        bad_path = pathlib.Path(tmpdir) / "bad.json"
        bad_path.write_text(
            (prefix.with_suffix(".json")).read_text().replace(
                '"makespan": 12', '"makespan": 13', 1
            )
        )
        invalid = run(
            [str(cli), "validate-result", "--input", str(bad_path)], check=False
        )
        assert invalid.returncode != 0
        invalid_json = json.loads(invalid.stdout)
        assert invalid_json["passed"] is False
        assert invalid_json["error_code"] == "reported_makespan_mismatch"


if __name__ == "__main__":
    main()
