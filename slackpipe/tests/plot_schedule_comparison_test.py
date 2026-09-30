"""Offline checks for plotting native evaluator intervals."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from plot_schedule_comparison import load_schedule  # noqa: E402


class SchedulePlotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.prefix = Path(self.directory.name) / "tiny"
        self.rows = [
            dict(kind="F", microbatch=0, stage=0, worker=0, start=0, end=1, duration=1),
            dict(kind="F", microbatch=0, stage=1, worker=1, start=1, end=3, duration=2),
            dict(kind="B", microbatch=0, stage=1, worker=1, start=3, end=5, duration=2),
            dict(kind="B", microbatch=0, stage=0, worker=0, start=5, end=6, duration=1),
        ]
        self.plan = dict(
            num_microbatches=1,
            num_stages=2,
            num_workers=2,
            num_layers=3,
            stage_to_worker=[0, 1],
            layer_split=[1, 2],
            predicted_makespan=6,
            operations=[
                [
                    {k: op[k] for k in ("kind", "microbatch", "stage")}
                    for op in self.rows
                    if op["worker"] == rank
                ]
                for rank in range(2)
            ],
        )

    def write(self) -> None:
        Path(f"{self.prefix}.plan.json").write_text(json.dumps(self.plan))
        with Path(f"{self.prefix}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_exact_bubbles(self) -> None:
        self.write()
        metrics = load_schedule(self.prefix)["metrics"]
        self.assertEqual(metrics["boundary"], [0, 2])
        self.assertEqual(metrics["residual"], [4, 0])
        self.assertEqual(metrics["delta_b"], 2)
        self.assertEqual(metrics["unique_operations"], 4)

    def test_duplicate_operation(self) -> None:
        self.rows[-1] = self.rows[0].copy()
        self.write()
        with self.assertRaisesRegex(ValueError, "Missing, duplicate"):
            load_schedule(self.prefix)

    def test_overlapping_intervals(self) -> None:
        self.rows[2].update(start=2, end=4)
        self.write()
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            load_schedule(self.prefix)

    def test_plan_order_mismatch(self) -> None:
        self.plan["operations"][0].reverse()
        self.write()
        with self.assertRaisesRegex(ValueError, "worker order differs"):
            load_schedule(self.prefix)

    def test_makespan_mismatch(self) -> None:
        self.plan["predicted_makespan"] = 7
        self.write()
        with self.assertRaisesRegex(ValueError, "makespans disagree"):
            load_schedule(self.prefix)


if __name__ == "__main__":
    unittest.main()
