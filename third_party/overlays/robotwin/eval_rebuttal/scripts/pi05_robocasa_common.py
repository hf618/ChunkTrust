#!/usr/bin/env python3
"""Shared immutable protocol helpers for the pi0.5 RoboCasa rebuttal run."""

from __future__ import annotations
from chunktrust.paths import resolve_legacy_path as _ct_path

import hashlib
import json
import os
from pathlib import Path
import random
import re
import tempfile
from typing import Any, Iterable


REPO_ROOT = Path(_ct_path('ROBOTWIN_ROOT', ''))
EXPERIMENT_ROOT = REPO_ROOT / "eval_rebuttal/09_pi05_robocasa_gr1_tabletop24_base_vs_ahs_20260801"
PACKAGE_ROOT = Path(_ct_path('CHUNKTRUST_MODELS_ROOT', 'pi05_robocasa_gr1_tabletop24_step30000'))
PACKAGE_MANIFEST = PACKAGE_ROOT / "configs/robocasa_gr1_tabletop24_manifest.json"
CHECKPOINT = PACKAGE_ROOT / "checkpoint"
ROBOCASA_ROOT = Path(_ct_path('CHUNKTRUST_WORKSPACE_ROOT', 'robocasa-gr1-tabletop-tasks'))
STARVLA_ROOT = Path(_ct_path('CHUNKTRUST_WORKSPACE_ROOT', 'starVLA'))
HORIZON_SELECTOR = STARVLA_ROOT / "examples/Robocasa_tabletop/eval_files/horizon_selector.py"

MODEL_ID = "NewGain/pi05_robocasa_gr1_tabletop24"
MODEL_REVISION = "master"
OPTIMIZER_STEP = 30_000
NORM_ASSET_ID = "robocasa_gr1_tabletop24_global"
NORM_STATS_SHA256 = "55fef4aefd96f317daff2a06510dfba70c34de9f6c8e561536f819c56a02d331"
SOURCE_ARCHIVE_SHA256 = "6d91c7531891854ce788c3bbea950541e02a5e069265cb7863a510b9e5f139ce"
PACKAGE_TREE_SHA256 = "a9cd68a081b692e0755994f59ad753ab214bf3dcfa6301698c19b0265b3bb5fe"
PROTOCOL_ID = "pi05_robocasa_gr1_tabletop24_base_ahs_formal50_seeded_reset_v3_20260802"

# These are the values stated for RoboCasa in the submitted paper's Appendix A.2.
AHS_CONFIG: dict[str, Any] = {
    "horizon_candidates": [4, 8, 12, 16],
    "exec_mode": "expected_round",
    "expected_temp": 0.8,
    "intra_alpha": 4.0,
    "intra_cut_t": 0.25,
    "ts_epsilon": 0.05,
    "ts_seed": 0,
    "ts_update_mode": "kernel_forget",
    "ts_kernel_bandwidth": 10.0,
    "ts_forget_rho": 0.99,
    "ts_update_eta": 1.0,
    "z_mode": "window_rms",
    "tau_window": 3,
    "mix_mode": "both",
    "use_speed_uniformity": True,
    "speed_window": 20,
    "speed_weight": 0.5,
    "speed_beta": 4.0,
    "speed_fallback": "ehh_only",
}

BASE_CONFIG: dict[str, Any] = {
    "fixed_k": 16,
    "use_qha": False,
}

SAMPLING_CONFIG: dict[str, Any] = {
    "image_key": "video.ego_view_bg_crop_pad_res256_freq20",
    "image_source": "frozen NVIDIA source observation.images.ego_view pixel-equivalent",
    "generated_action_horizon": 16,
    "model_action_dim": 32,
    "active_action_dim": 29,
    "denoise_steps": 10,
    "noise_rule": "sha256(protocol_id,task,seed,replan_index)->PCG64 float32 standard_normal",
    "max_environment_steps": 720,
    "success_rule": "any official info['success'] during the episode",
    "reset_rule": "python_random+numpy_global+underlying_robosuite_rng seeded from literal episode seed before reset",
}


def reset_with_literal_seed(env: Any, seed: int) -> tuple[Any, Any]:
    """Apply Gym's literal seed to every RNG used by the RoboCasa wrapper stack."""
    seed = int(seed)
    random.seed(seed)
    np = __import__("numpy")
    np.random.seed(seed)

    unwrapped = env.unwrapped
    robosuite_env = getattr(unwrapped, "env", None)
    if robosuite_env is None or not hasattr(robosuite_env, "rng"):
        raise RuntimeError("Could not locate the underlying RoboSuite RNG")
    robosuite_env.rng = np.random.default_rng(seed)
    return env.reset(seed=seed)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(value)
        temporary = Path(stream.name)
    temporary.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def load_package_manifest() -> dict[str, Any]:
    return json.loads(PACKAGE_MANIFEST.read_text(encoding="utf-8"))


def repo_id_to_env_name(repo_id: str) -> str:
    match = re.fullmatch(r"gr1_unified\.(.+)_GR1ArmsAndWaistFourierHands_1000", repo_id)
    if match is None:
        raise ValueError(f"Unsupported package task repo_id: {repo_id}")
    return f"gr1_unified/{match.group(1)}_GR1ArmsAndWaistFourierHands_Env"


def task_name_from_env(env_name: str) -> str:
    name = env_name.rsplit("/", 1)[-1]
    name = name.removesuffix("_GR1ArmsAndWaistFourierHands_Env")
    name = re.sub(r"^PosttrainPnPNovel", "posttrain_pnp_novel_", name)
    name = re.sub(r"^PnP", "pnp_", name)
    name = name.replace("From", "from_").replace("To", "_to_").replace("SplitA", "")
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return re.sub(r"__+", "_", name).strip("_").lower()


def task_records() -> list[dict[str, str]]:
    manifest = load_package_manifest()
    records = []
    for item in manifest["tasks"]:
        env_name = repo_id_to_env_name(item["repo_id"])
        records.append(
            {
                "repo_id": item["repo_id"],
                "env_name": env_name,
                "task_name": task_name_from_env(env_name),
            }
        )
    if len(records) != 24 or len({item["env_name"] for item in records}) != 24:
        raise RuntimeError(f"Expected 24 unique package tasks, got {len(records)}")
    return records


def manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    data = "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows)
    atomic_write_text(path, data)


def instruction_text(value: Any) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid literal instruction: {value!r}")
    return value.strip()


def result_key(row: dict[str, Any], method: str, manifest_sha256: str) -> str:
    identity = {
        "task_name": row["task_name"],
        "env_name": row["env_name"],
        "seed": int(row["seed"]),
        "instruction": row["instruction"],
        "method": method,
        "manifest_sha256": manifest_sha256,
    }
    return canonical_json_sha256(identity)


def deterministic_noise(task_name: str, seed: int, replan_index: int) -> Any:
    import numpy as np

    material = f"{PROTOCOL_ID}|{task_name}|{seed}|{replan_index}".encode("ascii")
    rng_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=False)
    rng = np.random.default_rng(rng_seed)
    return rng.standard_normal((16, 32), dtype=np.float32)


def source_hashes(extra_paths: Iterable[Path] = ()) -> dict[str, str]:
    paths = [Path(__file__).resolve(), HORIZON_SELECTOR, *extra_paths]
    return {str(path.resolve()): file_sha256(path.resolve()) for path in paths}


def environment_snapshot() -> dict[str, Any]:
    return {
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "pyopengl_platform": os.environ.get("PYOPENGL_PLATFORM"),
    }
