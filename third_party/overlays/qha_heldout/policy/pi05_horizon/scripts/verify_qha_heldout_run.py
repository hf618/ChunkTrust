#!/usr/bin/env python3
"""Verify a registered six-train/two-held-out QHA training run before release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


QHA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = QHA_ROOT.parents[1]
if str(QHA_ROOT) not in sys.path:
    sys.path.insert(0, str(QHA_ROOT))

from heldout_protocol import (
    CANDIDATE_HORIZONS,
    SCHEMA_VERSION,
    TRAIN_REPO_IDS,
    ProtocolError,
    load_training_protocol,
    sha256_file,
    sha256_tree,
    validate_heldout_runtime_source_manifest,
    validate_teacher_manifest,
)


def add_error(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def load_json(path: Path, errors: list[str], label: str) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read {label}: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{label} is not a JSON object")
        return None
    return value


def verify_lineage(path: Path, *, samples_per_task: int, errors: list[str]) -> int:
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError as exc:
        errors.append(f"cannot read teacher lineage: {exc}")
        return 0
    expected_set = set(TRAIN_REPO_IDS)
    for line_index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"teacher lineage line {line_index} is invalid JSON: {exc}")
            continue
        groups = record.get("groups")
        if not isinstance(groups, list) or len(groups) != len(TRAIN_REPO_IDS):
            errors.append(f"teacher lineage line {line_index} does not contain six task groups")
            continue
        repo_ids = {str(group.get("repo_id")) for group in groups if isinstance(group, dict)}
        if repo_ids != expected_set:
            errors.append(f"teacher lineage line {line_index} task membership is not the registered six-task set")
        for group in groups:
            if not isinstance(group, dict):
                errors.append(f"teacher lineage line {line_index} has a non-object group")
                continue
            if int(group.get("sample_count", -1)) != samples_per_task:
                errors.append(f"teacher lineage line {line_index} has an unexpected per-task sample count")
            if tuple(group.get("candidate_horizons", ())) != CANDIDATE_HORIZONS:
                errors.append(f"teacher lineage line {line_index} has a non-dense candidate grid")
            posterior_hash = str(group.get("teacher_posterior_sha256", ""))
            if len(posterior_hash) != 64 or any(char not in "0123456789abcdef" for char in posterior_hash):
                errors.append(f"teacher lineage line {line_index} has an invalid posterior hash")
    return len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--step", type=int, help="Require and verify this finalized QHA-only checkpoint")
    parser.add_argument("--require-final", action="store_true", help="Require step to equal registered num_train_steps")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    if args.require_final and args.step is None:
        parser.error("--require-final requires --step")
    if args.step is not None and args.step <= 0:
        parser.error("--step must be positive")

    run_dir = args.run_dir.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    audit_dir = run_dir / "audit"
    errors: list[str] = []
    audit = load_json(audit_dir / "training_audit.json", errors, "training audit")
    source_manifest = load_json(audit_dir / "training_runtime_source_manifest.json", errors, "runtime source manifest")
    if audit is None or source_manifest is None:
        print(json.dumps({"status": "FAIL", "errors": errors}, indent=2))
        raise SystemExit(1)

    add_error(errors, audit.get("schema_version") == SCHEMA_VERSION, "training audit schema is invalid")
    add_error(errors, audit.get("kind") == "qha_training_audit", "training audit kind is invalid")
    protocol_path = repo_root / "policy/pi05_horizon/configs/qha_heldout_protocol.json"
    teacher_path = repo_root / "policy/pi05_horizon/configs/qha_heldout6_teacher_manifest.json"
    try:
        protocol = load_training_protocol(protocol_path)
        add_error(errors, audit.get("protocol") == protocol, "embedded protocol differs from registered protocol")
        add_error(errors, audit.get("protocol_sha256") == sha256_file(protocol_path), "protocol hash differs from registered protocol")
    except ProtocolError as exc:
        errors.append(str(exc))
    try:
        teacher_manifest = validate_teacher_manifest(teacher_path)
        add_error(errors, audit.get("teacher_manifest") == teacher_manifest, "embedded teacher manifest differs from registered manifest")
        add_error(errors, audit.get("teacher_manifest_sha256") == sha256_file(teacher_path), "teacher manifest hash differs")
    except ProtocolError as exc:
        errors.append(str(exc))
    add_error(
        errors,
        audit.get("runtime_source_manifest_sha256") == sha256_file(audit_dir / "training_runtime_source_manifest.json"),
        "runtime source manifest hash is inconsistent with the training audit",
    )
    add_error(errors, audit.get("runtime_source_manifest") == source_manifest, "embedded runtime source manifest differs")
    errors.extend(validate_heldout_runtime_source_manifest(source_manifest, repo_root))

    config = audit.get("resolved_train_config")
    if not isinstance(config, dict):
        errors.append("training audit has no resolved train config")
        config = {}
    add_error(errors, config.get("train_repo_ids") == list(TRAIN_REPO_IDS), "training repo IDs differ from registered six tasks")
    add_error(errors, config.get("qha_train_mode") == "qha_only", "training is not QHA-only")
    add_error(errors, config.get("qha_candidate_mode") == "dense_full", "training does not use dense candidates")
    add_error(errors, config.get("teacher_execution_backend") == "gpu_workers", "training did not use dedicated GPU teachers")
    student_devices = set(config.get("student_cuda_visible_devices", []))
    teacher_devices = set(config.get("teacher_worker_devices", []))
    add_error(errors, not (student_devices & teacher_devices), "student and teacher devices overlap")
    add_error(errors, len(teacher_devices) == len(TRAIN_REPO_IDS), "training does not have one teacher worker per task")
    batch_size = int(config.get("batch_size", 0))
    add_error(errors, batch_size > 0 and batch_size % len(TRAIN_REPO_IDS) == 0, "batch cannot be evenly task-balanced")
    samples_per_task = batch_size // len(TRAIN_REPO_IDS) if batch_size else -1
    lineage_count = verify_lineage(audit_dir / "teacher_lineage.jsonl", samples_per_task=samples_per_task, errors=errors)

    checkpoint_summary: dict[str, Any] | None = None
    if args.step is not None:
        step_dir = run_dir / str(args.step)
        add_error(errors, (step_dir / "assets/_QHA_ONLY").is_file(), "checkpoint is not marked QHA-only")
        records_path = audit_dir / "checkpoints.jsonl"
        try:
            records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read checkpoint audit: {exc}")
            records = []
        matching = [
            record
            for record in records
            if record.get("step") == args.step
            and record.get("checkpoint_relpath") == str(args.step)
        ]
        add_error(errors, len(matching) == 1, "final checkpoint does not have exactly one matching audit record")
        if matching and step_dir.exists():
            actual_tree_hash = sha256_tree(step_dir)
            add_error(errors, matching[0].get("checkpoint_tree_sha256") == actual_tree_hash, "checkpoint tree hash mismatch")
            checkpoint_summary = {"step": args.step, "tree_sha256": actual_tree_hash}
        if args.require_final:
            add_error(errors, args.step == int(config.get("num_train_steps", -1)), "step is not the registered final step")
            add_error(errors, lineage_count >= args.step, "teacher lineage has fewer records than training steps")

    summary = {
        "status": "PASS" if not errors else "FAIL",
        "run_dir": str(run_dir),
        "lineage_records": lineage_count,
        "samples_per_task": samples_per_task,
        "checkpoint": checkpoint_summary,
        "errors": errors,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
