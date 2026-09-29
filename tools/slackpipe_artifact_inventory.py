# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fingerprint existing CAL evidence without rewriting it or certifying new runs."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a fresh output; evidence inventories are immutable")
    root = Path(__file__).resolve().parents[1]
    run = args.run.resolve()
    if not run.is_dir() or not run.is_relative_to(root):
        parser.error("--run must be an existing run directory inside the checkout")
    records = []
    sources, statuses, methods, contracts = Counter(), Counter(), Counter(), Counter()
    for path in sorted(run.rglob("*")):
        if not path.is_file() or path.suffix not in (".json", ".jsonl", ".csv", ".log", ".md"):
            continue
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        records.append(
            dict(path=str(path.relative_to(run)), sha256=checksum, bytes=path.stat().st_size)
        )
        if path.name == "result.json":
            data = json.loads(path.read_text())
            canonical = data.get("canonical_result", data.get("canonical", data))
            sources[str(canonical.get("git_commit"))] += 1
            statuses[str(canonical.get("solver_status_raw"))] += 1
            methods[str(canonical.get("canonical_method"))] += 1
            contracts[str(canonical.get("method_contract_hash"))] += 1
    source_objects = {}
    if not methods:
        parser.error("no native result.json records found; cannot inventory missing evidence")
    for commit in sources:
        source_objects[commit] = (
            subprocess.run(
                ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
                cwd=root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
    tracked = subprocess.check_output(
        ["git", "ls-files", "--", str(run)], cwd=root, text=True
    ).splitlines()
    payload = dict(
        schema_version="slackpipe.artifact_inventory.v1",
        kind="historical_CAL_simulation",
        run=str(run.relative_to(root)),
        current_execution=False,
        gpu_measurement=False,
        result_count=sum(methods.values()),
        sources=sources,
        source_commit_objects_available=source_objects,
        statuses=statuses,
        methods=methods,
        method_contracts=contracts,
        tracked_files=len(tracked),
        records=records,
        records_sha256=hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
        scope="Byte inventory only; use analyze_cal_results.py --strict for semantic validation",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
