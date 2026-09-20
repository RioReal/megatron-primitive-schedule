"""Re-correlate saved captures into a NEW directory; never alter raw experiments."""

import argparse
import hashlib
import json
from pathlib import Path

from megatron.core.pipeline_parallel.slackpipe.figure_trace import (
    compact_profiler_trace,
    write_compact_trace,
)


def reprocess(source: Path, output: Path) -> dict:
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("Output must be outside the original capture directory")
    inputs = sorted(source.rglob("*.samples.json")) + sorted(source.rglob("rank*_capture.json"))
    if not inputs:
        raise ValueError("No collection samples or legacy capture metadata found")
    output.mkdir(parents=True, exist_ok=False)
    report = []
    for path in inputs:
        metadata = json.loads(path.read_text())
        legacy = "measured_steps" in metadata
        rank = metadata["rank"]
        method = metadata["mode"] if legacy else metadata["method"]
        captures = (
            [dict(cycle=0, raw_path=str(path.parent / "torch_profiler" / f"rank{rank}_trace.json"))]
            if legacy
            else metadata["captures"]
        )
        measured = (
            metadata["measured_steps"]
            if legacy
            else {str(s["iteration"]): s for s in metadata["samples"]}
        )
        for capture in captures:
            raw_path = Path(capture["raw_path"])
            if not raw_path.exists() and not legacy:
                raw_path = path.with_name(
                    path.name.removesuffix(".samples.json")
                    + f".cycle{capture['cycle']:03d}.torch.json"
                )
            raw_bytes = raw_path.read_bytes()
            digest = hashlib.sha256(raw_bytes).hexdigest()
            trace = compact_profiler_trace(
                json.loads(raw_bytes),
                rank=rank,
                mode=method,
                transport=metadata.get("transport", "see collection config"),
                measured_steps=measured,
                config=metadata["config"],
            )
            if not legacy:
                trace["collection"] = dict(
                    run_id=metadata["run_id"],
                    capture_id=capture["capture_id"],
                    cycle=capture["cycle"],
                    profiler=metadata["profiler"],
                    active_iterations=capture["active_iterations"],
                    environment=metadata["environment"],
                    timing_definitions=metadata["timing_definitions"],
                )
                if [s["iteration"] for s in trace["steps"]] != capture["active_iterations"]:
                    raise ValueError("Raw capture iteration IDs differ from collection metadata")
            target = (
                output
                / method
                / f"{metadata.get('run_id', 'legacy')}.rank{rank}.cycle{capture['cycle']}.compact.json"
            )
            if target.exists():
                raise ValueError(f"Duplicate capture identity: {target}")
            write_compact_trace(target, trace)
            if hashlib.sha256(raw_path.read_bytes()).hexdigest() != digest:
                raise RuntimeError("Input changed during offline processing")
            report.append(
                dict(
                    source=str(raw_path),
                    source_sha256=digest,
                    output=str(target),
                    rank=rank,
                    method=method,
                    cycle=capture["cycle"],
                    steps=trace["steps"],
                    unassigned_gpu_activity_count=trace["unassigned_gpu_activity_count"],
                )
            )
    result = dict(
        schema_version="slackpipe.trace_audit.v1",
        captures=report,
        raw_inputs_unchanged=True,
        warning="Offline correlation validates recorded evidence, not new GPU performance",
    )
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = reprocess(args.input, args.output)
    print(f"Validated {len(report['captures'])} captures; original raw files unchanged")


if __name__ == "__main__":
    main()
