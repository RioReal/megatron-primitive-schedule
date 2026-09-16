#!/usr/bin/env python3
"""Run or resume SlackPipe CAL evaluation rows from a JSONL manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cal_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--binary", type=Path, default=Path("build/ortools/slackpipe_cli"))
    parser.add_argument("--group")
    parser.add_argument("--method")
    parser.add_argument("--configuration-id")
    parser.add_argument("--run-id")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-new-manifest", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-rerun-invalid", action="store_true")
    args = parser.parse_args(argv)

    rows = cal_manifest.read_manifest(args.manifest)
    rows = cal_manifest.filter_rows(
        rows,
        group=args.group,
        method=args.method,
        configuration_id=args.configuration_id,
        run_id=args.run_id,
    )
    results = cal_manifest.run_manifest(
        rows=rows,
        binary=args.binary,
        output_root=args.output_root,
        jobs=args.jobs,
        dry_run=args.dry_run,
        force=args.force,
        force_new_manifest=args.force_new_manifest,
        no_rerun_invalid=args.no_rerun_invalid,
        fail_fast=args.fail_fast,
    )
    summary = cal_manifest.summarize_outcomes(results)
    print(json.dumps({"rows": len(rows), "outcomes": summary}, sort_keys=True))
    bad = {
        "completed_invalid",
        "timeout_or_failed_process",
        "schema_mismatch",
        "manifest_mismatch",
    }
    return 1 if any(summary.get(outcome, 0) for outcome in bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
