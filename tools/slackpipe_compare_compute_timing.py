# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compare a fresh legacy calibration and a separate non-serializing probe.

No filtering, profile publication or solver authorization. Both runs must use
identical calibration configuration, hardware/software and partitions.
"""

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def _summary(values: list) -> dict:
    mean = statistics.fmean(values)
    return dict(
        median_ms=statistics.median(values),
        cv=statistics.pstdev(values) / mean,
        iteration_means_ms=values,
    )


def load_run(root: Path, *, diagnostic: bool) -> dict:
    """Read recorded data only; validate complete rank/iteration/operation grids."""
    hashes = {}

    def read(path):
        data = path.read_bytes()
        hashes[str(path.resolve())] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    manifest = read(root / "model_manifest.json")
    source = read(root / ("timing_events.json" if diagnostic else "calibration_events.json"))
    events = [e for item in source for e in (item["events"] if "events" in item else [item])]
    metadata = {}
    for path in sorted(root.glob("partition*/result.rank*.json")):
        item = read(path)
        key = (path.parent.name, item["rank"])
        if key in metadata:
            raise ValueError("Duplicate rank/partition metadata")
        if bool(item.get("timing_diagnostic")) != diagnostic:
            raise ValueError("Mislabeled timing mode")
        if "calibration_configuration" not in item:
            raise ValueError("Missing comparable configuration: collect a fresh legacy reference")
        metadata[key] = item
    if not metadata:
        raise ValueError("Missing partition/rank metadata")
    seen = set()
    expected = set()
    worker_configurations = {}
    for (group, worker), item in metadata.items():
        c = item["calibration_configuration"]
        if worker in worker_configurations and worker_configurations[worker] != c:
            raise ValueError("Configuration changed between partitions on the same worker")
        worker_configurations[worker] = c
        if c["manifest"] != manifest["manifest_hash"]:
            raise ValueError("Manifest/configuration mismatch")
        if sorted(r for g, r in metadata if g == group) != list(range(c["pp"])):
            raise ValueError("Missing rank metadata")
        for s in range(worker, c["stages"], c["pp"]):
            for phase in ("forward_compute", "backward_compute"):
                for i in item["measurement_iterations"]:
                    for b in range(c["microbatches"]):
                        expected.add((group, worker, s, phase, i, b))
    for event in events:
        group, worker = event["calibration_group_id"], event["worker"]
        s = event["logical_stage"]
        key = (group, worker, s, event["phase"], event["iteration"], event["microbatch"])
        if key in seen or key not in expected:
            raise ValueError("Duplicate/unexpected timing operation")
        seen.add(key)
        item = metadata[(group, worker)]
        timing = (
            "compute-stream-events-diagnostic-v1"
            if diagnostic
            else "synchronized-stage-wall-time-v1"
        )
        context = hashlib.sha256(
            json.dumps(
                dict(**item["calibration_configuration"], timing=timing),
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
        if event.get("measurement_context") != context or event.get("rank") != worker:
            raise ValueError("Event timing context/rank does not match metadata")
        begin, end = item["stage_layer_ranges"][s]
        composition = [layer["config_class"] for layer in manifest["layers"][begin:end]]
        if event["stage_layer_range"] != [begin, end] or event["layer_composition"] != composition:
            raise ValueError("Layer range/composition mismatch")
        for field in ("compute_gpu_ms", "cpu_wall_ms") if diagnostic else ("elapsed_ms",):
            if not math.isfinite(event[field]) or event[field] <= 0:
                raise ValueError("Timing must be positive and finite")
    if seen != expected:
        raise ValueError("Missing timing operations")
    return dict(events=events, metadata=metadata, hashes=hashes, manifest=manifest["manifest_hash"])


def compare_runs(legacy: dict, diagnostic: dict) -> dict:
    """Use all raw iteration means; no deletion, rescaling or kernel-time claim."""
    if (
        legacy["manifest"] != diagnostic["manifest"]
        or legacy["metadata"].keys() != diagnostic["metadata"].keys()
    ):
        raise ValueError("Model/partition/rank mismatch")
    for key, item in legacy["metadata"].items():
        other = diagnostic["metadata"][key]
        for field in (
            "calibration_configuration",
            "measurement_iterations",
            "warmup_iterations",
            "stage_layer_ranges",
        ):
            if item[field] != other[field]:
                raise ValueError(f"Incomparable {key}: {field} differs")
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    identities = {}
    for run, fields in (
        (legacy, {"elapsed_ms": "existing_stage_wall_ms"}),
        (diagnostic, {"compute_gpu_ms": "compute_gpu_ms", "cpu_wall_ms": "cpu_wall_ms"}),
    ):
        for event in run["events"]:
            identity = {
                k: event[k]
                for k in (
                    "logical_stage",
                    "stage_layer_range",
                    "layer_composition",
                    "stage_role",
                    "worker",
                    "phase",
                )
            }
            key = json.dumps(identity, sort_keys=True)
            identities[key] = identity
            group = event["calibration_group_id"]
            for source, metric in fields.items():
                groups[(key, group)][metric][event["iteration"]].append(event[source])
    rows, comparable = [], defaultdict(list)
    for (key, group), fields in sorted(groups.items()):
        summaries = {}
        for metric, by_iteration in fields.items():
            summaries[metric] = dict(
                **_summary(
                    [statistics.fmean(values) for _, values in sorted(by_iteration.items())]
                ),
                global_iterations=sorted(by_iteration),
            )
        row = dict(**identities[key], calibration_group_id=group, metrics=summaries)
        rows.append(row)
        comparable[key].append(row)
    comparisons = []
    for key, members in comparable.items():
        shifts = {}
        for metric in ("existing_stage_wall_ms", "compute_gpu_ms", "cpu_wall_ms"):
            medians = [m["metrics"][metric]["median_ms"] for m in members]
            shifts[metric] = max(medians) / min(medians) - 1 if len(medians) > 1 else None
        comparisons.append(
            dict(
                **identities[key],
                groups=[m["calibration_group_id"] for m in members],
                cross_partition_median_shift=shifts,
            )
        )
    return dict(
        schema_version="slackpipe.compute_timing_comparison.v1",
        diagnostic_only=True,
        estimator_validated=False,
        samples_discarded=0,
        definitions=dict(
            cv="population stddev/mean of raw iteration operation means",
            cross_partition_median_shift="max(group medians)/min(group medians)-1; null with <2 groups",
            comparison="separate matched runs, not simultaneous measurements; events are stream envelopes, not intrinsic kernel sums",
        ),
        inputs={**legacy["hashes"], **diagnostic["hashes"]},
        groups=rows,
        comparisons=comparisons,
        diagnostic_iterations=[
            dict(group=g, worker=w, **item["timing_diagnostic"])
            for (g, w), item in diagnostic["metadata"].items()
        ],
        raw_legacy_operations=legacy["events"],
        raw_diagnostic_operations=diagnostic["events"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_runs(
        load_run(args.legacy, diagnostic=False), load_run(args.diagnostic, diagnostic=True)
    )
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with args.output.open("x") as stream:
        stream.write(serialized)
    print(
        f"Saved {len(report['groups'])} stage/phase/group comparisons; estimator remains unvalidated"
    )


if __name__ == "__main__":
    main()
