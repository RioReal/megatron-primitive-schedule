# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Stage identity, read-only compatibility checks and auditable v1 migration."""

import copy
import hashlib
import subprocess
from pathlib import Path

from tools.run_slackpipe_nemotron_h8b_pp4 import ROOT, digest
from tools.slackpipe_eval_config import fingerprint
from tools.slackpipe_hybrid import write_json

RECEIPT_SCHEMA = "slackpipe.eval_receipt.v2"
EMPTY_DIFF = hashlib.sha256(b"").hexdigest()
# This migration is for the audited generic-driver v1 implementation, not arbitrary old code.
LEGACY_COMMIT = "5f000dcc5da4b5ece14a617456a4eca3acd3bbf7"
RECEIPT_ONLY_PATHS = (
    "tools/run_slackpipe_eval.py",
    "tools/run_slackpipe_real_system_campaign.py",
    "tools/slackpipe_eval_receipts.py",
    "tools/slackpipe_eval_tables.py",
    "tests/unit_tests/pipeline_parallel/test_slackpipe_eval.py",
    "tests/unit_tests/pipeline_parallel/test_slackpipe_eval_receipts.py",
    "README.md",
    "docs/slackpipe_real_system_eval.md",
)


def artifact_error(receipt: dict, root: Path) -> str | None:
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        return "missing artifact checksums"
    for name, checksum in artifacts.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            return f"artifact outside experiment: {name}"
        if not path.is_file() or digest(path) != checksum:
            return f"missing/changed artifact: {name}"
    for key in ("profile", "plan", "summary"):
        if key in receipt and receipt[key] not in artifacts:
            return f"unbound {key} artifact"
    return None


def receipt_valid(receipt: dict, context: dict, root: Path) -> bool:
    """Exact context/byte check, also used by the offline table exporter."""
    return (
        fingerprint(receipt.get("context")) == fingerprint(context)
        and receipt.get("context_hash", fingerprint(context)) == fingerprint(context)
        and artifact_error(receipt, root) is None
    )


def parent_identity(receipt: dict) -> str:
    """Migration notes, aliases and timestamps must not invalidate scientific outputs."""
    return fingerprint({k: receipt[k] for k in ("stage", "context", "artifacts")})


def archive_receipt(path: Path, receipt: dict) -> None:
    archive = path.parent / "history" / f"{path.stem}.{fingerprint(receipt)}.json"
    if not archive.exists():
        write_json(archive, receipt)


def context_differences(old: dict, new: dict, prefix: str = "") -> list[str]:
    changed = []
    for key in sorted(set(old) | set(new)):
        label = f"{prefix}{key}"
        if key not in old or key not in new:
            changed.append(label)
        elif isinstance(old[key], dict) and isinstance(new[key], dict):
            changed.extend(context_differences(old[key], new[key], label + "."))
        elif fingerprint(old[key]) != fingerprint(new[key]):
            changed.append(label)
    return changed


def legacy_source_compatible(old: dict, current: dict) -> bool:
    if old == current:
        return True
    if old != dict(commit=LEGACY_COMMIT, diff=EMPTY_DIFF):
        return False
    # Permit this orchestration-only fix, but never waive worker/model/solver changes.
    try:
        diff = subprocess.check_output(
            [
                "git",
                "diff",
                LEGACY_COMMIT,
                "--",
                ".",
                *(f":(exclude){p}" for p in RECEIPT_ONLY_PATHS),
            ],
            cwd=ROOT,
        )
        subprocess.check_call(
            ["git", "merge-base", "--is-ancestor", LEGACY_COMMIT, "HEAD"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return not diff
    except (OSError, subprocess.CalledProcessError):
        return False


def legacy_context(receipt: dict, expected: dict, parents: dict) -> dict:
    """Project the audited v1 invocation onto v2, retaining every real old input."""
    old = receipt["context"]
    stage = expected["stage"]
    if set(old) != {
        "model_config_hash",
        "topology",
        "schedule",
        "precision",
        "seq_length",
        "micro_batch_size",
        "transport",
        "seed",
        "learning_rate",
        "policy",
        "source",
        "environment",
        "parents",
        "solver_sha256",
        "solver_seconds",
        "run_index",
    } or set(old["policy"]) != {
        "warmups",
        "iterations",
        "calibration_iterations",
        "repetitions",
        "profiler_wait",
        "profiler_warmup",
        "profiler_active",
        "profiler_repeat",
    }:
        raise ValueError("unknown/missing legacy context fields; cannot certify invocation")
    if stage == "trace":
        raise ValueError(
            "v1 trace has a different dependency contract; use --force for a new trace"
        )
    projected = copy.deepcopy(expected)
    for key in (
        "model_config_hash",
        "topology",
        "precision",
        "seq_length",
        "micro_batch_size",
        "transport",
    ):
        projected[key] = old[key]
    for key in ("seed", "learning_rate"):
        if key in expected:
            projected[key] = old[key]
    if "schedule" in expected:
        projected["schedule"] = (
            ("1f1b" if old["topology"]["num_stages"] == old["topology"]["pp"] else "interleaved")
            if stage in ("native-smoke", "calibrate")
            else old["schedule"]
        )
    if not legacy_source_compatible(old["source"], expected["source"]):
        raise ValueError(
            "legacy source differs beyond the audited receipt-only fix, or was dirty/unknown"
        )
    environment = copy.deepcopy(old["environment"])
    before = environment.pop("source", {})
    after = copy.deepcopy(expected["environment"])
    current_source = after.pop("source", {})
    if before != current_source:
        if before.get("diff_sha256") != EMPTY_DIFF:
            raise ValueError("legacy environment source was dirty/unknown")
        if before.get("commit") != old["source"]["commit"]:
            raise ValueError("inconsistent legacy source identities")
        if before.get("untracked_source_sha256", {}) != current_source.get(
            "untracked_source_sha256", {}
        ):
            raise ValueError("untracked source differs; cannot prove legacy compatibility")
    if fingerprint(environment) != fingerprint(after):
        raise ValueError("legacy hardware/dependency/allocator environment differs")
    if set(old["parents"]) != set(parents):
        raise ValueError("legacy parent stage set differs")
    for name, parent in parents.items():
        old_parent_hash = parent.get("migration", {}).get(
            "original_receipt_hash", fingerprint(parent)
        )
        if old["parents"][name] != old_parent_hash:
            raise ValueError(f"legacy parent hash mismatch: {name}")
    policy = old["policy"]
    if stage in ("smoke", "native-smoke", "calibrate", "benchmark"):
        projected["policy"]["warmups"] = policy["warmups"]
        projected["policy"]["iterations"] = (
            policy["calibration_iterations"]
            if stage == "calibrate"
            else (2 if stage in ("smoke", "native-smoke") else policy["iterations"])
        )
    if stage == "solve":
        projected.update(solver_sha256=old["solver_sha256"], solver_seconds=old["solver_seconds"])
    if stage == "benchmark":
        projected.update(run_index=old["run_index"], repetitions=policy["repetitions"])
    return projected


def assess_receipt(receipt: dict, context: dict, root: Path, parents: dict) -> tuple:
    """Return (classification, explanation, accepted receipt), without filesystem writes."""
    if receipt.get("status") != "passed":
        return "invalid", f"receipt status is {receipt.get('status')}", None
    if error := artifact_error(receipt, root):
        return "invalid", error, None
    schema = receipt.get("schema_version")
    if schema == RECEIPT_SCHEMA:
        if (
            not isinstance(receipt.get("context"), dict)
            or receipt["context"].get("context_schema_version") != 2
        ):
            return "invalid", "unknown/missing context schema", None
        if receipt.get("context_hash") != fingerprint(receipt.get("context")):
            return "invalid", "context hash mismatch", None
        if receipt.get("stage") != context["stage"]:
            return "invalid", "receipt stage mismatch", None
        changed = context_differences(receipt["context"], context)
        if changed:
            return "stale", "changed stage inputs: " + ", ".join(changed), None
        return "compatible", "stage inputs and artifact checksums match", receipt
    if schema != "slackpipe.eval_receipt.v1":
        return "invalid", f"unknown receipt schema: {schema}", None
    try:
        if receipt["stage"] != context["stage"]:
            raise ValueError("legacy stage mismatch")
        projected = legacy_context(receipt, context, parents)
        if fingerprint(projected) != fingerprint(context):
            changed = context_differences(projected, context)
            raise ValueError("legacy relevant inputs differ: " + ", ".join(changed))
        if receipt["stage"] in ("calibrate", "solve"):
            import json

            from tools.slackpipe_eval_config import load_model
            from tools.slackpipe_eval_worker import validate_plan_provenance, validate_profile

            model = load_model(root / "config.json")
            if fingerprint(model) != context["model_config_hash"]:
                raise ValueError("experiment config hash mismatch")
            calibration = receipt if receipt["stage"] == "calibrate" else parents["calibrate"]
            profile_path = root / calibration["profile"]
            profile = json.loads(profile_path.read_text())
            validate_profile(
                profile,
                model,
                context["topology"],
                context["precision"],
                context["seq_length"],
                context["micro_batch_size"],
            )
            if receipt["stage"] == "solve":
                validate_plan_provenance(root / receipt["plan"], profile_path, profile)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return "legacy-incompatible", str(exc), None
    migrated = dict(
        receipt,
        schema_version=RECEIPT_SCHEMA,
        context=context,
        context_hash=fingerprint(context),
        migration=dict(
            original_receipt_hash=fingerprint(receipt),
            original_context_hash=fingerprint(receipt["context"]),
            original_source=receipt["context"]["source"],
            reason="v1 projected onto actual stage inputs; original parents/artifacts verified; no execution",
        ),
    )
    return "legacy-compatible", migrated["migration"]["reason"], migrated
