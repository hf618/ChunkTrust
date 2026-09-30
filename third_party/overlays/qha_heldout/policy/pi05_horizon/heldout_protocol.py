"""Immutable protocol helpers for the QHA six-train/two-held-out experiment.

This module deliberately uses only the standard library so it can be used by
the training process, the evaluator, and a clean Ubuntu delivery environment.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable


SCHEMA_VERSION = "qha-heldout-v1"
HELDOUT_TASKS = ("blocks_ranking_rgb", "place_bread_basket")
TRAIN_TASKS = (
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "place_a2b_left",
    "place_bread_skillet",
    "place_can_basket",
)
TRAIN_REPO_IDS = tuple(f"{task}_aloha-agilex_clean_50_repo" for task in TRAIN_TASKS)
CANDIDATE_HORIZONS = tuple(range(1, 51))

# These are the executable files needed to reproduce the registered QHA
# training and held-out runtime. The training audit records their file hashes
# individually, avoiding an ambiguous whole-worktree hash after helper tools or
# logs are added later.
HELDOUT_RUNTIME_SOURCE_FILES = (
    "policy/pi05_horizon/configs/qha_heldout_protocol.json",
    "policy/pi05_horizon/configs/qha_heldout6_teacher_manifest.json",
    "policy/pi05_horizon/configs/qha_heldout_manifest_registry.json",
    "policy/pi05_horizon/heldout_protocol.py",
    "policy/pi05_horizon/pi_model.py",
    "policy/pi05_horizon/deploy_policy.py",
    "policy/pi05_horizon/deploy_policy.yml",
    "policy/pi05_horizon/scripts/eval_qha_heldout_manifest.sh",
    "policy/pi05_horizon/scripts/train.py",
    "policy/pi05_horizon/scripts/train_qha_heldout_8gpu.sh",
    "policy/pi05_horizon/scripts/verify_qha_heldout_run.py",
    "policy/pi05_horizon/src/openpi/models/pi0.py",
    "policy/pi05_horizon/src/openpi/models/pi0_config.py",
    "policy/pi05_horizon/src/openpi/models/qha.py",
    "policy/pi05_horizon/src/openpi/training/checkpoints.py",
    "policy/pi05_horizon/src/openpi/training/config.py",
    "policy/pi05_horizon/src/openpi/training/data_loader.py",
    "policy/pi05_horizon/src/openpi/training/online_teachers.py",
    "policy/pi05_horizon/src/openpi/training/online_teacher_workers.py",
    "script/build_qha_heldout_manifest.py",
    "script/eval_policy.py",
)


class ProtocolError(ValueError):
    """Raised when a run cannot satisfy the frozen held-out protocol."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: str | Path, *, ignored_dir_names: Iterable[str] = ()) -> str:
    """Hash regular files together with their relative names, deterministically."""
    root = Path(path).resolve()
    ignored = set(ignored_dir_names)
    digest = hashlib.sha256()
    for directory, dir_names, file_names in os.walk(root):
        dir_names[:] = sorted(name for name in dir_names if name not in ignored)
        for file_name in sorted(file_names):
            file_path = Path(directory) / file_name
            if not file_path.is_file():
                continue
            relative = file_path.relative_to(root).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            with file_path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def heldout_runtime_source_manifest(repo_root: str | Path) -> dict[str, Any]:
    """Return hashes for the versioned runtime surface of this experiment."""
    root = Path(repo_root).expanduser().resolve()
    files: dict[str, dict[str, Any]] = {}
    for relative in HELDOUT_RUNTIME_SOURCE_FILES:
        file_path = root / relative
        if not file_path.is_file():
            raise ProtocolError(f"Required held-out runtime source is missing: {file_path}")
        files[relative] = {
            "sha256": sha256_file(file_path),
            "bytes": file_path.stat().st_size,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "qha_heldout_runtime_source_manifest",
        "repo_root": str(root),
        "files": files,
    }


def validate_heldout_runtime_source_manifest(payload: dict[str, Any], repo_root: str | Path) -> list[str]:
    """Return mismatches between a frozen runtime manifest and a source tree."""
    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("source manifest schema_version is not qha-heldout-v1")
    if payload.get("kind") != "qha_heldout_runtime_source_manifest":
        errors.append("source manifest kind is invalid")
    raw_files = payload.get("files")
    if not isinstance(raw_files, dict):
        return errors + ["source manifest files is not an object"]
    if tuple(raw_files) != HELDOUT_RUNTIME_SOURCE_FILES:
        errors.append("source manifest file set or ordering does not match the registered runtime surface")
    root = Path(repo_root).expanduser().resolve()
    for relative, expected in raw_files.items():
        file_path = root / relative
        if not file_path.is_file():
            errors.append(f"missing runtime source: {relative}")
            continue
        if not isinstance(expected, dict):
            errors.append(f"invalid source manifest entry: {relative}")
            continue
        actual_hash = sha256_file(file_path)
        if actual_hash != expected.get("sha256"):
            errors.append(f"runtime source hash mismatch: {relative}")
        if file_path.stat().st_size != expected.get("bytes"):
            errors.append(f"runtime source byte count mismatch: {relative}")
    return errors


def load_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProtocolError(f"Required protocol file does not exist: {resolved}") from exc
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Invalid JSON in {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"Expected JSON object in {resolved}, got {type(value).__name__}")
    return value


def _require_exact_list(value: Any, expected: tuple[str, ...], field: str, path: Path) -> None:
    if not isinstance(value, list) or tuple(value) != expected:
        raise ProtocolError(
            f"{path}:{field} must exactly equal {list(expected)!r}; got {value!r}. "
            "Task order is part of the audit contract."
        )


def load_training_protocol(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    protocol = load_json(resolved)
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ProtocolError(f"{resolved}: unsupported schema_version={protocol.get('schema_version')!r}")
    _require_exact_list(protocol.get("heldout_tasks"), HELDOUT_TASKS, "heldout_tasks", resolved)
    _require_exact_list(protocol.get("train_tasks"), TRAIN_TASKS, "train_tasks", resolved)
    _require_exact_list(protocol.get("train_repo_ids"), TRAIN_REPO_IDS, "train_repo_ids", resolved)
    _require_exact_list(protocol.get("candidate_horizons"), CANDIDATE_HORIZONS, "candidate_horizons", resolved)
    if protocol.get("selector_mode") != "expected_round":
        raise ProtocolError(f"{resolved}: selector_mode must be 'expected_round'")
    if float(protocol.get("temperature", -1)) != 1.0:
        raise ProtocolError(f"{resolved}: temperature must be 1.0")
    return protocol


def validate_teacher_manifest(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    manifest = load_json(resolved)
    entries = manifest.get("teachers")
    if not isinstance(entries, list):
        raise ProtocolError(f"{resolved}: teachers must be a list")
    repo_ids = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ProtocolError(f"{resolved}: teacher entry {index} must be an object")
        repo_id = str(entry.get("repo_id", ""))
        if not repo_id:
            raise ProtocolError(f"{resolved}: teacher entry {index} has no repo_id")
        repo_ids.append(repo_id)
        for key in ("train_config_name", "checkpoint_root_tag", "model_name", "checkpoint_id"):
            if not str(entry.get(key, "")):
                raise ProtocolError(f"{resolved}: teacher entry {index} missing {key}")
    if tuple(repo_ids) != TRAIN_REPO_IDS:
        raise ProtocolError(
            f"{resolved}: teacher repo_ids must exactly equal the six training repo IDs; got {repo_ids!r}"
        )
    return manifest


def load_episode_manifest(path: str | Path, *, task_name: str, task_config: str) -> tuple[dict[str, Any], str]:
    """Load literal seed+instruction episodes without any seed substitution."""
    resolved = Path(path).expanduser().resolve()
    manifest = load_json(resolved)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ProtocolError(f"{resolved}: unsupported schema_version={manifest.get('schema_version')!r}")
    if manifest.get("kind") != "immutable_eval_manifest":
        raise ProtocolError(f"{resolved}: kind must be immutable_eval_manifest")
    if manifest.get("task_name") != task_name or manifest.get("task_config") != task_config:
        raise ProtocolError(
            f"{resolved}: manifest is for {manifest.get('task_name')}/{manifest.get('task_config')}, "
            f"not {task_name}/{task_config}"
        )
    if task_name not in HELDOUT_TASKS:
        raise ProtocolError(f"{resolved}: immutable held-out manifests are only permitted for {list(HELDOUT_TASKS)}")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ProtocolError(f"{resolved}: episodes must be a non-empty list")

    seen_ids: set[str] = set()
    seen_seeds: set[int] = set()
    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise ProtocolError(f"{resolved}: episode {index} must be an object")
        episode_id = str(episode.get("episode_id", "")).strip()
        instruction = episode.get("instruction")
        try:
            seed = int(episode.get("seed"))
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"{resolved}: episode {index} has an invalid seed") from exc
        if not episode_id or not isinstance(instruction, str) or not instruction.strip():
            raise ProtocolError(f"{resolved}: episode {index} requires non-empty episode_id and instruction")
        if episode_id in seen_ids or seed in seen_seeds:
            raise ProtocolError(f"{resolved}: duplicate episode_id or seed at episode {index}")
        seen_ids.add(episode_id)
        seen_seeds.add(seed)
    return manifest, sha256_file(resolved)


def write_immutable_episode_manifest(
    path: str | Path,
    *,
    task_name: str,
    task_config: str,
    episodes: list[dict[str, Any]],
    generator_metadata: dict[str, Any],
) -> str:
    if task_name not in HELDOUT_TASKS:
        raise ProtocolError(f"Only held-out tasks may receive an immutable eval manifest: {task_name}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "immutable_eval_manifest",
        "task_name": task_name,
        "task_config": task_config,
        "episodes": episodes,
        "generator_metadata": generator_metadata,
    }
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
    destination.write_text(encoded, encoding="utf-8")
    load_episode_manifest(destination, task_name=task_name, task_config=task_config)
    return sha256_file(destination)


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    def _run(*args: str) -> str | None:
        try:
            return subprocess.check_output(args, cwd=repo_root, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = _run("git", "status", "--porcelain=v1", "--untracked-files=all")
    return {
        "git_head": _run("git", "rev-parse", "HEAD"),
        "git_status_sha256": sha256_bytes((status or "").encode("utf-8")),
        "git_worktree_dirty": bool(status),
    }


@dataclasses.dataclass
class TrainingAudit:
    """Append-only training audit writer for a registered held-out run."""

    audit_dir: Path
    lineage_path: Path
    checkpoint_records_path: Path

    @classmethod
    def create(cls, config: Any) -> "TrainingAudit | None":
        protocol_path = str(getattr(config, "heldout_protocol_path", "")).strip()
        if not protocol_path:
            return None
        protocol = load_training_protocol(protocol_path)
        data = getattr(config, "data", None)
        if data is None or not hasattr(data, "normalized_repo_ids"):
            raise ProtocolError("Held-out QHA training requires MultiLeRobotAlohaDataConfig.")
        repo_ids = tuple(data.normalized_repo_ids())
        if repo_ids != TRAIN_REPO_IDS:
            raise ProtocolError(f"Training repo IDs must exactly equal {list(TRAIN_REPO_IDS)!r}; got {list(repo_ids)!r}")
        if any(task in " ".join(repo_ids) for task in HELDOUT_TASKS):
            raise ProtocolError("A held-out task appeared in the training repo IDs.")
        model = getattr(config, "model", None)
        if not bool(getattr(model, "use_qha", False)):
            raise ProtocolError("Held-out protocol requires use_qha=True.")
        if str(getattr(model, "qha_train_mode", "")).lower() != "qha_only":
            raise ProtocolError("Held-out protocol requires qha_train_mode=qha_only.")
        if str(getattr(model, "qha_candidate_mode", "")).lower() != "dense_full":
            raise ProtocolError("Held-out protocol requires qha_candidate_mode=dense_full.")
        worker_devices = [
            value.strip()
            for value in str(getattr(config, "teacher_worker_devices", "")).split(",")
            if value.strip()
        ]
        if str(getattr(config, "teacher_execution_backend", "")).lower() != "gpu_workers":
            raise ProtocolError("Held-out protocol requires teacher_execution_backend=gpu_workers.")
        if not worker_devices:
            raise ProtocolError(
                "Held-out protocol requires explicit teacher_worker_devices; falling back to student GPUs is forbidden."
            )
        student_devices = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
        overlap = sorted(set(student_devices) & set(worker_devices))
        if overlap:
            raise ProtocolError(
                f"Teacher workers overlap student CUDA_VISIBLE_DEVICES={student_devices}: {overlap}."
            )
        manifest_path = str(getattr(config, "teacher_checkpoint_manifest", "")).strip()
        teacher_manifest = validate_teacher_manifest(manifest_path)

        audit_dir = Path(config.checkpoint_dir) / "audit"
        audit_dir.mkdir(parents=True, exist_ok=False)
        root = Path(__file__).resolve().parents[2]
        qha_root = Path(__file__).resolve().parent
        runtime_source_manifest = heldout_runtime_source_manifest(root)
        runtime_source_manifest_path = audit_dir / "training_runtime_source_manifest.json"
        runtime_source_manifest_path.write_text(
            json.dumps(runtime_source_manifest, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        source_tree_hash = sha256_tree(
            qha_root,
            ignored_dir_names={
                ".venv",
                "assets",
                "checkpoints",
                "eval_result",
                "log",
                "logs",
                "wandb",
                "__pycache__",
            },
        )
        resolved = {
            "name": config.name,
            "exp_name": config.exp_name,
            "checkpoint_dir": str(config.checkpoint_dir),
            "checkpoint_root_tag": config.checkpoint_root_tag,
            "seed": int(config.seed),
            "batch_size": int(config.batch_size),
            "num_train_steps": int(config.num_train_steps),
            "save_interval": int(config.save_interval),
            "fsdp_devices": int(config.fsdp_devices),
            "use_preprocessed_cache": bool(config.use_preprocessed_cache),
            "preprocessed_cache_root": str(config.preprocessed_cache_root),
            "preprocessed_cache_compat_config_name": str(config.preprocessed_cache_compat_config_name),
            "preprocessed_cache_compat_data_factory_type": str(
                config.preprocessed_cache_compat_data_factory_type
            ),
            "teacher_runtime_mode": str(config.teacher_runtime_mode),
            "teacher_execution_backend": str(config.teacher_execution_backend),
            "teacher_worker_devices": worker_devices,
            "student_cuda_visible_devices": student_devices,
            "train_repo_ids": list(repo_ids),
            "qha_train_mode": str(model.qha_train_mode),
            "qha_candidate_mode": str(model.qha_candidate_mode),
        }
        payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": "qha_training_audit",
            "created_unix_s": time.time(),
            "protocol_path": str(Path(protocol_path).resolve()),
            "protocol_sha256": sha256_file(protocol_path),
            "protocol": protocol,
            "teacher_manifest_path": str(Path(manifest_path).resolve()),
            "teacher_manifest_sha256": sha256_file(manifest_path),
            "teacher_manifest": teacher_manifest,
            "runtime_source_manifest_path": str(runtime_source_manifest_path.resolve()),
            "runtime_source_manifest_sha256": sha256_file(runtime_source_manifest_path),
            "runtime_source_manifest": runtime_source_manifest,
            "resolved_train_config": resolved,
            "source": {
                "repo_root": str(root),
                "pi05_horizon_root": str(qha_root),
                "pi05_horizon_tree_sha256": source_tree_hash,
                **_git_metadata(root),
            },
        }
        (audit_dir / "training_audit.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        return cls(
            audit_dir=audit_dir,
            lineage_path=audit_dir / "teacher_lineage.jsonl",
            checkpoint_records_path=audit_dir / "checkpoints.jsonl",
        )

    def record_teacher_batch(self, *, batch_index: int, groups: list[dict[str, Any]]) -> None:
        repo_ids = [str(group["repo_id"]) for group in groups]
        if any(repo_id not in TRAIN_REPO_IDS for repo_id in repo_ids):
            raise ProtocolError(f"Teacher lineage attempted to write non-training repo IDs: {repo_ids!r}")
        payload = {
            "batch_index": int(batch_index),
            "repo_ids": repo_ids,
            "groups": groups,
            "recorded_unix_s": time.time(),
        }
        with self.lineage_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")

    def record_checkpoint(self, checkpoint_dir: str | Path, *, step: int) -> None:
        resolved = Path(checkpoint_dir).resolve()
        if not resolved.exists():
            raise ProtocolError(f"Cannot audit missing checkpoint: {resolved}")
        run_dir = self.audit_dir.parent.resolve()
        try:
            checkpoint_relpath = resolved.relative_to(run_dir).as_posix()
        except ValueError as exc:
            raise ProtocolError(
                f"Checkpoint must be inside the audited run directory {run_dir}: {resolved}"
            ) from exc
        if checkpoint_relpath != str(int(step)):
            raise ProtocolError(
                f"Checkpoint relative path must be its numeric step {step}, got {checkpoint_relpath!r}"
            )
        payload = {
            "step": int(step),
            # The delivery archive is intentionally relocatable. The checkpoint
            # content hash and this run-relative location identify it without
            # binding the audit to the training server's absolute path.
            "checkpoint_relpath": checkpoint_relpath,
            "checkpoint_tree_sha256": sha256_tree(resolved),
            "recorded_unix_s": time.time(),
        }
        with self.checkpoint_records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")
