#!/usr/bin/env python3
"""Run one resumable pi0.5 RoboCasa task/method cell from an immutable manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from pi05_robocasa_common import (  # noqa: E402
    AHS_CONFIG,
    CHECKPOINT,
    EXPERIMENT_ROOT,
    HORIZON_SELECTOR,
    MODEL_ID,
    NORM_ASSET_ID,
    NORM_STATS_SHA256,
    OPTIMIZER_STEP,
    PACKAGE_ROOT,
    PACKAGE_TREE_SHA256,
    PROTOCOL_ID,
    ROBOCASA_ROOT,
    SAMPLING_CONFIG,
    SOURCE_ARCHIVE_SHA256,
    atomic_write_json,
    canonical_json_sha256,
    deterministic_noise,
    file_sha256,
    instruction_text,
    manifest_rows,
    reset_with_literal_seed,
    result_key,
)

CLIENT_ROOT = PACKAGE_ROOT / "openpi_runtime/packages/openpi-client/src"
PACKAGE_ADAPTER = PACKAGE_ROOT / "openpi_runtime/src/openpi/policies/gr1_tabletop_policy.py"
SERVER_SCRIPT = SCRIPT_DIR / "run_pi05_robocasa_server.py"
COMMON_SCRIPT = SCRIPT_DIR / "pi05_robocasa_common.py"

# Exact canonical active-order mapping frozen by the delivered package adapter.
ACTIVE_PARTS = ("left_arm", "right_arm", "left_hand", "right_hand", "waist")
PART_DIMS = {"left_arm": 7, "right_arm": 7, "left_hand": 6, "right_hand": 6, "waist": 3}
MODEL_IMAGE_KEY = "video.ego_view_bg_crop_pad_res256_freq20"
RAW_SLICES = {
    "left_arm": slice(0, 7),
    "left_hand": slice(7, 13),
    "right_arm": slice(22, 29),
    "right_hand": slice(29, 35),
    "waist": slice(41, 44),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--method", choices=("base", "ahs"), required=True)
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5920)
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--save-video", type=int, choices=(0, 1), default=0)
    parser.add_argument("--max-environment-steps", type=int, default=720)
    parser.add_argument("--freeze-runtime", action="store_true")
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def current_runtime_sources() -> dict[str, str]:
    paths = (Path(__file__).resolve(), COMMON_SCRIPT, SERVER_SCRIPT, HORIZON_SELECTOR, PACKAGE_ADAPTER)
    return {str(path.resolve()): file_sha256(path.resolve()) for path in paths}


def verify_runtime_freeze(output_root: Path, freeze: bool) -> tuple[dict[str, str], str]:
    path = output_root / "provenance/RUNTIME_SOURCE_HASHES.json"
    sources = current_runtime_sources()
    digest = canonical_json_sha256(sources)
    payload = {
        "schema_version": 1,
        "status": "FROZEN",
        "runtime_sha256": digest,
        "sources": sources,
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("runtime_sha256") != digest or existing.get("sources") != sources:
            raise RuntimeError(f"Runtime source drift detected against {path}")
    elif freeze:
        atomic_write_json(path, payload)
    else:
        raise RuntimeError(f"Runtime is not frozen; run smoke with --freeze-runtime first: {path}")
    return sources, digest


def active_state(observation: dict[str, Any]) -> np.ndarray:
    values = []
    for part in ACTIVE_PARTS:
        value = np.asarray(observation[f"state.{part}"], dtype=np.float32).reshape(-1)
        if value.shape != (PART_DIMS[part],):
            raise ValueError(f"state.{part}: expected {(PART_DIMS[part],)}, got {value.shape}")
        values.append(value)
    return np.concatenate(values, axis=0)


def expand_active(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (29,):
        raise ValueError(f"Expected active state shape (29,), got {value.shape}")
    output = np.zeros((44,), dtype=np.float32)
    offset = 0
    for part in ACTIVE_PARTS:
        width = PART_DIMS[part]
        output[RAW_SLICES[part]] = value[offset : offset + width]
        offset += width
    return output


def active_to_action_dict(value: np.ndarray) -> dict[str, np.ndarray]:
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (29,):
        raise ValueError(f"Expected active action shape (29,), got {value.shape}")
    result = {}
    offset = 0
    for part in ACTIVE_PARTS:
        width = PART_DIMS[part]
        result[f"action.{part}"] = value[offset : offset + width]
        offset += width
    return result


def official_success(info: dict[str, Any]) -> bool:
    value = np.asarray(info.get("success", False)).reshape(-1)
    return bool(value[0]) if value.size else False


def extract_policy_output(output: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float, float]:
    actions = np.asarray(output["actions"], dtype=np.float32)
    if actions.shape != (16, 29) or not np.all(np.isfinite(actions)):
        raise ValueError(f"Expected finite actions shape (16, 29), got {actions.shape}")
    trace = output.get("denoise_trace")
    if not isinstance(trace, dict) or "v" not in trace:
        raise ValueError("Policy response is missing denoise_trace.v")
    v_trace = np.asarray(trace["v"], dtype=np.float32)
    if v_trace.ndim != 3 or v_trace.shape[1] != 16 or v_trace.shape[2] < 29:
        raise ValueError(f"Unexpected denoise trace shape: {v_trace.shape}")
    v_trace = v_trace[:, :, :29]
    if not np.all(np.isfinite(v_trace)):
        raise ValueError("Denoise trace contains non-finite values")
    policy_ms = float(output.get("policy_timing", {}).get("infer_ms", np.nan))
    if not np.isfinite(policy_ms) or policy_ms < 0:
        raise ValueError(f"Invalid policy inference timing: {policy_ms}")
    server_ms = float(output.get("server_timing", {}).get("infer_ms", np.nan))
    if not np.isfinite(server_ms) or server_ms < policy_ms:
        raise ValueError(f"Invalid server inference timing: {server_ms}")
    return actions, v_trace, policy_ms / 1000.0, server_ms / 1000.0


def make_video_writer(path: Path, frame: np.ndarray):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open smoke video writer: {path}")
    return writer


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(jsonable(payload), sort_keys=True, ensure_ascii=True) + "\n")
        stream.flush()


def existing_result(
    path: Path,
    *,
    expected_key: str,
    manifest_sha: str,
    protocol_sha: str,
    runtime_sha: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Existing result is unreadable and will not be overwritten: {path}") from exc
    expected = {
        "status": "COMPLETED",
        "result_key": expected_key,
        "manifest_sha256": manifest_sha,
        "frozen_protocol_sha256": protocol_sha,
        "runtime_sha256": runtime_sha,
    }
    drift = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
    if drift:
        raise RuntimeError(f"Existing result provenance drift at {path}: {drift}")
    return payload


def run_episode(
    *,
    row: dict[str, Any],
    method: str,
    client: Any,
    env: Any,
    manifest_sha: str,
    protocol_sha: str,
    runtime_sha: str,
    output_root: Path,
    save_video: bool,
    max_steps: int,
) -> dict[str, Any]:
    seed = int(row["seed"])
    key = result_key(row, method, manifest_sha)
    task = row["task_name"]
    result_path = output_root / f"raw_results/{row['phase']}/{method}/{task}/seed{seed:05d}__{key}.json"
    completed = existing_result(
        result_path,
        expected_key=key,
        manifest_sha=manifest_sha,
        protocol_sha=protocol_sha,
        runtime_sha=runtime_sha,
    )
    if completed is not None:
        return completed

    trace_path = output_root / f"traces/{row['phase']}/{method}/{task}/seed{seed:05d}__{key}.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    video_path = output_root / f"smoke_videos/{task}__seed{seed:05d}__{method}.mp4"
    writer = None
    episode_start = time.perf_counter()
    observation, reset_info = reset_with_literal_seed(env, seed)
    literal_instruction = instruction_text(observation["annotation.human.coarse_action"])
    if literal_instruction != row["instruction"]:
        raise RuntimeError(
            f"Manifest instruction drift for {task}/seed{seed}: "
            f"{literal_instruction!r} != {row['instruction']!r}"
        )

    if save_video:
        first_frame = np.asarray(observation[MODEL_IMAGE_KEY])
        writer = make_video_writer(video_path, first_frame)
        import cv2

        writer.write(cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))

    history_actions: list[np.ndarray] = []
    selected_k: list[int] = []
    policy_inference_seconds = 0.0
    server_inference_seconds = 0.0
    policy_roundtrip_seconds = 0.0
    selector_seconds = 0.0
    environment_step_seconds = 0.0
    success = official_success(reset_info)
    first_success_step = 0 if success else None
    terminated = False
    truncated = False
    env_steps = 0
    replan_index = 0
    ts_state: dict[str, Any] = {}
    ts_rng = np.random.default_rng(int(AHS_CONFIG["ts_seed"]))

    try:
        while env_steps < max_steps and not (terminated or truncated):
            request = {
                "observation/image": np.asarray(observation[MODEL_IMAGE_KEY]),
                "observation/state": expand_active(active_state(observation)),
                "prompt": literal_instruction,
                "policy_noise": deterministic_noise(task, seed, replan_index),
            }
            infer_start = time.perf_counter()
            output = client.infer(request)
            roundtrip_seconds = time.perf_counter() - infer_start
            actions, v_trace, model_seconds, server_seconds = extract_policy_output(output)
            policy_roundtrip_seconds += roundtrip_seconds
            policy_inference_seconds += model_seconds
            server_inference_seconds += server_seconds

            horizon_info: dict[str, Any]
            if method == "base":
                k_exec = 16
                horizon_info = {
                    "method": "fixed_k",
                    "chosen_horizon": 16,
                    "candidate_horizons": [16],
                }
            else:
                sys.path.insert(0, str(HORIZON_SELECTOR.parent))
                from horizon_selector import select_exec_k_horizon

                selector_start = time.perf_counter()
                k_exec, horizon_info = select_exec_k_horizon(
                    v_trace,
                    16,
                    horizon_candidates=AHS_CONFIG["horizon_candidates"],
                    exec_mode=AHS_CONFIG["exec_mode"],
                    expected_temp=AHS_CONFIG["expected_temp"],
                    intra_alpha=AHS_CONFIG["intra_alpha"],
                    intra_cut_t=AHS_CONFIG["intra_cut_t"],
                    ts_epsilon=AHS_CONFIG["ts_epsilon"],
                    ts_update_mode=AHS_CONFIG["ts_update_mode"],
                    ts_kernel_bandwidth=AHS_CONFIG["ts_kernel_bandwidth"],
                    ts_forget_rho=AHS_CONFIG["ts_forget_rho"],
                    ts_update_eta=AHS_CONFIG["ts_update_eta"],
                    z_mode=AHS_CONFIG["z_mode"],
                    tau_window=AHS_CONFIG["tau_window"],
                    xt_step=actions,
                    history_actions=history_actions,
                    mix_mode=AHS_CONFIG["mix_mode"],
                    use_speed_uniformity=AHS_CONFIG["use_speed_uniformity"],
                    speed_window=AHS_CONFIG["speed_window"],
                    speed_weight=AHS_CONFIG["speed_weight"],
                    speed_beta=AHS_CONFIG["speed_beta"],
                    speed_fallback=AHS_CONFIG["speed_fallback"],
                    ts_state=ts_state,
                    ts_rng=ts_rng,
                )
                selector_elapsed = time.perf_counter() - selector_start
                selector_seconds += selector_elapsed
                k_exec = int(k_exec)
                if not 4 <= k_exec <= 16:
                    raise RuntimeError(f"AHS expected-round selected invalid K={k_exec}")

            selected_k.append(k_exec)
            actual_executed = 0
            step_start_index = env_steps
            for action in actions[:k_exec]:
                step_start = time.perf_counter()
                observation, _reward, terminated, truncated, info = env.step(active_to_action_dict(action))
                environment_step_seconds += time.perf_counter() - step_start
                env_steps += 1
                actual_executed += 1
                history_actions.append(np.asarray(action, dtype=np.float32).copy())
                step_success = official_success(info)
                if step_success and first_success_step is None:
                    first_success_step = env_steps
                success = success or step_success
                if writer is not None and env_steps % 2 == 0:
                    import cv2

                    frame = np.asarray(observation[MODEL_IMAGE_KEY])
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if terminated or truncated or env_steps >= max_steps:
                    break

            append_jsonl(
                trace_path,
                {
                    "schema_version": 1,
                    "result_key": key,
                    "task_name": task,
                    "seed": seed,
                    "method": method,
                    "replan_index": replan_index,
                    "start_environment_step": step_start_index,
                    "selected_k": k_exec,
                    "actual_executed": actual_executed,
                    "policy_model_seconds": model_seconds,
                    "server_inference_seconds": server_seconds,
                    "policy_roundtrip_seconds": roundtrip_seconds,
                    "selector_seconds": selector_elapsed if method == "ahs" else 0.0,
                    "horizon_info": horizon_info,
                    "success_observed": success,
                },
            )
            replan_index += 1
    finally:
        if writer is not None:
            writer.release()

    result = {
        "schema_version": 1,
        "status": "COMPLETED",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "result_key": key,
        "phase": row["phase"],
        "task_name": task,
        "env_name": row["env_name"],
        "repo_id": row["repo_id"],
        "episode_id": row["episode_id"],
        "episode_index": int(row["episode_index"]),
        "seed": seed,
        "instruction": literal_instruction,
        "method": method,
        "outcome": bool(success),
        "official_success_predicate": "any info['success'] during the episode",
        "first_success_environment_step": first_success_step,
        "environment_steps": env_steps,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "max_step_timeout": bool(env_steps >= max_steps and not (terminated or truncated)),
        "policy_calls": replan_index,
        "selected_k": selected_k,
        "policy_inference_seconds": policy_inference_seconds,
        "server_inference_seconds": server_inference_seconds,
        "policy_roundtrip_seconds": policy_roundtrip_seconds,
        "ahs_overhead_seconds": selector_seconds,
        "environment_step_seconds": environment_step_seconds,
        "wall_seconds": time.perf_counter() - episode_start,
        "trace_path": str(trace_path.resolve()),
        "video_path": str(video_path.resolve()) if save_video else None,
        "model": {
            "id": MODEL_ID,
            "optimizer_step": OPTIMIZER_STEP,
            "checkpoint": str(CHECKPOINT.resolve()),
            "package_tree_sha256": PACKAGE_TREE_SHA256,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "norm_asset_id": NORM_ASSET_ID,
            "norm_stats_sha256": NORM_STATS_SHA256,
        },
        "manifest_sha256": manifest_sha,
        "frozen_protocol_sha256": protocol_sha,
        "runtime_sha256": runtime_sha,
        "ahs_config": AHS_CONFIG if method == "ahs" else None,
        "sampling_config": SAMPLING_CONFIG,
    }
    atomic_write_json(result_path, result)
    return result


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    manifest = args.manifest.resolve()
    rows = [row for row in manifest_rows(manifest) if row["task_name"] == args.task_name]
    if not rows:
        raise ValueError(f"No rows for task {args.task_name!r} in {manifest}")
    if any(row.get("phase") != args.phase for row in rows):
        raise ValueError(f"Manifest phase mismatch for --phase {args.phase}")

    manifest_sha = file_sha256(manifest)
    protocol_path = output_root / "FROZEN_EVAL_PROTOCOL.json"
    if not protocol_path.exists():
        raise FileNotFoundError(protocol_path)
    protocol_sha = file_sha256(protocol_path)
    _, runtime_sha = verify_runtime_freeze(output_root, args.freeze_runtime)

    sys.path.insert(0, str(CLIENT_ROOT))
    sys.path.insert(0, str(ROBOCASA_ROOT))
    import gymnasium as gym
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from robocasa.utils.gym_utils import gymnasium_groot  # noqa: F401

    client = WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = client.get_server_metadata()
    expected_metadata = {
        "protocol_id": PROTOCOL_ID,
        "model_id": MODEL_ID,
        "optimizer_step": OPTIMIZER_STEP,
        "norm_asset_id": NORM_ASSET_ID,
        "norm_stats_sha256": NORM_STATS_SHA256,
        "package_tree_sha256": PACKAGE_TREE_SHA256,
        "use_qha": False,
        "trace_num_steps_static_bound": SAMPLING_CONFIG["denoise_steps"],
        "trace_wire_fields": ["v"],
        "trace_wire_dtype": "float32",
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise RuntimeError(f"Server metadata mismatch for {key}: {metadata.get(key)!r} != {expected!r}")

    env_names = {row["env_name"] for row in rows}
    if len(env_names) != 1:
        raise ValueError(f"Task rows map to multiple environments: {env_names}")
    env = gym.make(next(iter(env_names)), enable_render=True)
    completed = 0
    try:
        for row in rows:
            key = result_key(row, args.method, manifest_sha)
            attempt_log = output_root / f"logs/attempts/{args.phase}/{args.method}/{args.task_name}.jsonl"
            try:
                result = run_episode(
                    row=row,
                    method=args.method,
                    client=client,
                    env=env,
                    manifest_sha=manifest_sha,
                    protocol_sha=protocol_sha,
                    runtime_sha=runtime_sha,
                    output_root=output_root,
                    save_video=bool(args.save_video),
                    max_steps=args.max_environment_steps,
                )
                completed += 1
                append_jsonl(
                    attempt_log,
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "result_key": key,
                        "seed": int(row["seed"]),
                        "status": "COMPLETED",
                        "outcome": result["outcome"],
                        "result_path": str(
                            output_root
                            / f"raw_results/{args.phase}/{args.method}/{args.task_name}/"
                            f"seed{int(row['seed']):05d}__{key}.json"
                        ),
                    },
                )
            except (Exception, SystemExit) as exc:
                append_jsonl(
                    attempt_log,
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "result_key": key,
                        "seed": int(row["seed"]),
                        "status": "INFRASTRUCTURE_ERROR",
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                raise
    finally:
        env.close()

    print(
        json.dumps(
            {
                "status": "PASS",
                "phase": args.phase,
                "task_name": args.task_name,
                "method": args.method,
                "completed": completed,
                "manifest_sha256": manifest_sha,
                "runtime_sha256": runtime_sha,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
