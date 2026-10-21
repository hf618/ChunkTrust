#!/usr/bin/env python3
"""Lightweight per-episode NPZ logger for real-robot rollout/video alignment."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


SCHEMA_VERSION = "robodriver_eval_episode_v1"
CAMERA_NAMES = ("top", "right", "left", "overview")


class EvalEpisodeNpzLogger:
    def __init__(self, *, recorder: Any, rollout_kind: str, args: Any) -> None:
        self.recorder = recorder
        self.rollout_kind = rollout_kind
        self.args = args
        self._steps: list[dict[str, Any]] = []
        self._replans: list[dict[str, Any]] = []
        self._async_events: list[dict[str, Any]] = []

    def log_async_event(
        self,
        *,
        requested_at_step: int,
        ready_at_step: int,
        hold_steps: int,
        infer_latency_sec: float,
    ) -> None:
        self._async_events.append(
            {
                "requested_at_step": int(requested_at_step),
                "ready_at_step": int(ready_at_step),
                "hold_steps": int(hold_steps),
                "infer_latency_sec": float(infer_latency_sec),
            }
        )

    def log_replan(
        self,
        *,
        replan_idx: int,
        env_step_idx: int,
        selected_k: int,
        monotonic_timestamp_ns: int,
        ahs: dict[str, Any] | None = None,
    ) -> None:
        env_step_idx = int(env_step_idx)
        selected_k = int(selected_k)
        record = {
            "replan_idx": int(replan_idx),
            "env_step_idx": env_step_idx,
            "replan_env_step_idx": env_step_idx,
            "mp4_frame_idx": env_step_idx,
            "monotonic_timestamp_ns": int(monotonic_timestamp_ns),
            "selected_K": selected_k,
            "K_exec": selected_k,
            "executed_action_start": env_step_idx,
            "executed_action_end": env_step_idx + selected_k,
            "executed_action_range": np.asarray([env_step_idx, env_step_idx + selected_k], dtype=np.int32),
        }
        if ahs is not None:
            record["ahs"] = dict(ahs)
        self._replans.append(record)

    def log_step(
        self,
        *,
        step_index: int,
        replan_index: int,
        chunk_index: int,
        step_start_time_ns: int,
        step_start_monotonic_ns: int,
        state_raw26_before: np.ndarray,
        policy_action_lr14: np.ndarray,
        executed_action_lr14: np.ndarray,
        execute_enabled: bool,
        ahs: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "step_index": int(step_index),
            "replan_index": int(replan_index),
            "chunk_index": int(chunk_index),
            "step_start_time_ns": int(step_start_time_ns),
            "step_start_monotonic_ns": int(step_start_monotonic_ns),
            "state_raw26_before": np.asarray(state_raw26_before, dtype=np.float32).reshape(26).copy(),
            "policy_action_lr14": np.asarray(policy_action_lr14, dtype=np.float32).reshape(-1)[:14].copy(),
            "executed_action_lr14": np.asarray(executed_action_lr14, dtype=np.float32).reshape(-1)[:14].copy(),
            "execute_enabled": bool(execute_enabled),
        }
        if ahs is not None:
            record["ahs"] = dict(ahs)
        self._steps.append(record)

    def save(self, *, status: str) -> Path | None:
        episode_dir = getattr(self.recorder, "episode_dir", None)
        if episode_dir is None:
            return None
        episode_dir = Path(episode_dir)
        episode_dir.mkdir(parents=True, exist_ok=True)
        out_path = episode_dir / "episode_data.npz"

        step_count = len(self._steps)
        replan_count = len(self._replans)
        step_time_ns = np.asarray([r["step_start_time_ns"] for r in self._steps], dtype=np.int64)
        step_monotonic_ns = np.asarray([r["step_start_monotonic_ns"] for r in self._steps], dtype=np.int64)
        replan_monotonic_ns = np.asarray([r["monotonic_timestamp_ns"] for r in self._replans], dtype=np.int64)

        payload: dict[str, np.ndarray] = {
            "schema_version": np.asarray(SCHEMA_VERSION),
            "status": np.asarray(str(status)),
            "task_name": np.asarray(str(getattr(self.recorder, "task_name", ""))),
            "rollout_kind": np.asarray(str(self.rollout_kind)),
            "prompt": np.asarray(str(getattr(self.args, "prompt", ""))),
            "checkpoint_dir": np.asarray(str(getattr(self.args, "record_checkpoint_dir", "") or "")),
            "episode_dir": np.asarray(str(episode_dir)),
            "camera_names": np.asarray(CAMERA_NAMES),
            "video_paths": np.asarray([str(episode_dir / f"{name}.mp4") for name in CAMERA_NAMES]),
            "step_index": np.asarray([r["step_index"] for r in self._steps], dtype=np.int32),
            "replan_index": np.asarray([r["replan_index"] for r in self._steps], dtype=np.int32),
            "chunk_index": np.asarray([r["chunk_index"] for r in self._steps], dtype=np.int32),
            "step_start_time_ns": step_time_ns,
            "step_start_monotonic_ns": step_monotonic_ns,
            "state_raw26_before": _stack_or_empty([r["state_raw26_before"] for r in self._steps], (0, 26)),
            "policy_action_lr14": _stack_or_empty([r["policy_action_lr14"] for r in self._steps], (0, 14)),
            "executed_action_lr14": _stack_or_empty([r["executed_action_lr14"] for r in self._steps], (0, 14)),
            "execute_enabled": np.asarray([r["execute_enabled"] for r in self._steps], dtype=np.bool_),
            "async_prefetch_steps": np.asarray(_to_int(getattr(self.args, "async_prefetch_steps", -1), default=-1), dtype=np.int32),
            "async_hold_on_empty": np.asarray(_to_int(getattr(self.args, "async_hold_on_empty", -1), default=-1), dtype=np.int32),
            "async_replan_requested_at_step": np.asarray(
                [r["requested_at_step"] for r in self._async_events],
                dtype=np.int32,
            ),
            "async_replan_ready_at_step": np.asarray(
                [r["ready_at_step"] for r in self._async_events],
                dtype=np.int32,
            ),
            "async_hold_steps": np.asarray([r["hold_steps"] for r in self._async_events], dtype=np.int32),
            "async_infer_latency_sec": np.asarray(
                [r["infer_latency_sec"] for r in self._async_events],
                dtype=np.float32,
            ),
            "replan_idx": np.asarray([r["replan_idx"] for r in self._replans], dtype=np.int32),
            "env_step_idx": np.asarray([r["env_step_idx"] for r in self._replans], dtype=np.int32),
            "replan_env_step_idx": np.asarray([r["replan_env_step_idx"] for r in self._replans], dtype=np.int32),
            "mp4_frame_idx": np.asarray([r["mp4_frame_idx"] for r in self._replans], dtype=np.int32),
            "monotonic_timestamp_ns": replan_monotonic_ns,
            "selected_K": np.asarray([r["selected_K"] for r in self._replans], dtype=np.int32),
            "K_exec": np.asarray([r["K_exec"] for r in self._replans], dtype=np.int32),
            "executed_action_start": np.asarray([r["executed_action_start"] for r in self._replans], dtype=np.int32),
            "executed_action_end": np.asarray([r["executed_action_end"] for r in self._replans], dtype=np.int32),
            "executed_action_range": _stack_or_empty(
                [r["executed_action_range"] for r in self._replans],
                (0, 2),
            ).astype(np.int32, copy=False),
        }

        for camera_name in CAMERA_NAMES:
            frame_times = _load_camera_monotonic_times(episode_dir, camera_name)
            frame_index, frame_delta = _nearest_frame_indices(step_monotonic_ns, frame_times)
            payload[f"{camera_name}_frame_index"] = frame_index
            payload[f"{camera_name}_frame_time_delta_ns"] = frame_delta
            payload[f"{camera_name}_frame_monotonic_ns"] = frame_times
            replan_frame_index, replan_frame_delta = _nearest_frame_indices(replan_monotonic_ns, frame_times)
            payload[f"replan_{camera_name}_frame_index"] = replan_frame_index
            payload[f"replan_{camera_name}_frame_time_delta_ns"] = replan_frame_delta

        if self.rollout_kind == "ahs":
            self._add_ahs_payload(payload, step_count)
        self._add_ahs_replan_payload(payload, replan_count)

        np.savez(out_path, **payload)
        return out_path

    def _add_ahs_payload(self, payload: dict[str, np.ndarray], step_count: int) -> None:
        ahs_records = [r.get("ahs", {}) for r in self._steps]
        payload["ahs_exec_length"] = np.asarray(
            [_to_int(record.get("exec_length"), default=-1) for record in ahs_records],
            dtype=np.int32,
        )
        payload["ahs_expected_horizon"] = np.asarray(
            [_to_float(record.get("expected_horizon")) for record in ahs_records],
            dtype=np.float32,
        )
        payload["ahs_feedback"] = np.asarray(
            [_to_float(record.get("feedback")) for record in ahs_records],
            dtype=np.float32,
        )
        payload["ahs_source"] = np.asarray([str(record.get("source", "")) for record in ahs_records])

        for key in ["candidates", "scores", "q_mix", "z_norm", "p_inter"]:
            rows = [_as_float_array(record.get(key)) for record in ahs_records]
            payload[f"ahs_{key}"] = _pad_rows(rows, step_count)

    def _add_ahs_replan_payload(self, payload: dict[str, np.ndarray], replan_count: int) -> None:
        ahs_records = [r.get("ahs", {}) for r in self._replans]
        payload["ahs_replan_source"] = np.asarray([str(record.get("source", "")) for record in ahs_records])
        payload["ahs_replan_expected_horizon"] = np.asarray(
            [_to_float(record.get("expected_horizon")) for record in ahs_records],
            dtype=np.float32,
        )
        payload["ahs_replan_feedback"] = np.asarray(
            [_to_float(record.get("feedback")) for record in ahs_records],
            dtype=np.float32,
        )
        payload["ahs_replan_candidates"] = _pad_int_rows(
            [_as_int_array(record.get("candidates")) for record in ahs_records],
            replan_count,
        )
        for key in ["scores", "q_mix", "z_norm", "p_inter"]:
            rows = [_as_float_array(record.get(key)) for record in ahs_records]
            payload[f"ahs_replan_{key}"] = _pad_rows(rows, replan_count)
        payload["ahs_beta_candidates"] = _pad_int_rows(
            [_as_int_array(record.get("beta_candidates")) for record in ahs_records],
            replan_count,
        )
        payload["ahs_beta_ab_before"] = _pad_matrices(
            [_as_float_matrix(record.get("beta_ab_before")) for record in ahs_records],
            replan_count,
            width_hint=payload["ahs_beta_candidates"].shape[1] if "ahs_beta_candidates" in payload else 0,
            depth=2,
        )
        payload["ahs_beta_ab_after"] = _pad_matrices(
            [_as_float_matrix(record.get("beta_ab_after")) for record in ahs_records],
            replan_count,
            width_hint=payload["ahs_beta_candidates"].shape[1] if "ahs_beta_candidates" in payload else 0,
            depth=2,
        )


def _stack_or_empty(rows: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    if not rows:
        return np.empty(shape, dtype=np.float32)
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def _to_int(value: Any, *, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_float_array(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _as_int_array(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.int32)
    return np.asarray(value, dtype=np.int32).reshape(-1)


def _as_float_matrix(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0, 0), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        return np.empty((0, 0), dtype=np.float32)
    return arr


def _pad_rows(rows: list[np.ndarray], row_count: int) -> np.ndarray:
    width = max((row.size for row in rows), default=0)
    if width == 0:
        return np.full((row_count, 0), np.nan, dtype=np.float32)
    out = np.full((row_count, width), np.nan, dtype=np.float32)
    for index, row in enumerate(rows):
        if row.size:
            out[index, : row.size] = row
    return out


def _pad_int_rows(rows: list[np.ndarray], row_count: int) -> np.ndarray:
    width = max((row.size for row in rows), default=0)
    if width == 0:
        return np.full((row_count, 0), -1, dtype=np.int32)
    out = np.full((row_count, width), -1, dtype=np.int32)
    for index, row in enumerate(rows):
        if row.size:
            out[index, : row.size] = row
    return out


def _pad_matrices(rows: list[np.ndarray], row_count: int, *, width_hint: int = 0, depth: int = 2) -> np.ndarray:
    width = max([width_hint, *[int(row.shape[0]) for row in rows if row.ndim == 2]], default=0)
    if width == 0:
        return np.full((row_count, 0, depth), np.nan, dtype=np.float32)
    out = np.full((row_count, width, depth), np.nan, dtype=np.float32)
    for index, row in enumerate(rows):
        if row.ndim != 2 or row.size == 0:
            continue
        row_width = min(int(row.shape[0]), width)
        row_depth = min(int(row.shape[1]), depth)
        out[index, :row_width, :row_depth] = row[:row_width, :row_depth]
    return out


def _nearest_frame_indices(step_monotonic_ns: np.ndarray, frame_monotonic_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if step_monotonic_ns.size == 0:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int64)
    if frame_monotonic_ns.size == 0:
        return (
            np.full((step_monotonic_ns.size,), -1, dtype=np.int32),
            np.full((step_monotonic_ns.size,), np.iinfo(np.int64).max, dtype=np.int64),
        )

    positions = np.searchsorted(frame_monotonic_ns, step_monotonic_ns, side="left")
    right = np.clip(positions, 0, frame_monotonic_ns.size - 1)
    left = np.clip(positions - 1, 0, frame_monotonic_ns.size - 1)
    left_delta = np.abs(step_monotonic_ns - frame_monotonic_ns[left])
    right_delta = np.abs(frame_monotonic_ns[right] - step_monotonic_ns)
    choose_right = right_delta < left_delta
    indices = np.where(choose_right, right, left).astype(np.int32)
    deltas = (step_monotonic_ns - frame_monotonic_ns[indices]).astype(np.int64)
    return indices, deltas


def _load_camera_monotonic_times(episode_dir: Path, camera_name: str) -> np.ndarray:
    if camera_name == "overview":
        return _overview_monotonic_times(episode_dir)

    path = episode_dir / f"{camera_name}_timestamps.csv"
    if not path.exists():
        return np.empty((0,), dtype=np.int64)
    values: list[int] = []
    with path.open("r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                values.append(int(row["monotonic_ns"]))
            except (KeyError, TypeError, ValueError):
                continue
    return np.asarray(values, dtype=np.int64)


def _overview_monotonic_times(episode_dir: Path) -> np.ndarray:
    start_path = episode_dir / "overview_start.json"
    if not start_path.exists():
        return np.empty((0,), dtype=np.int64)
    try:
        start = json.loads(start_path.read_text())
        start_monotonic_ns = int(start["start_monotonic_ns"])
        fps = float(start.get("fps") or 30.0)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return np.empty((0,), dtype=np.int64)

    frame_count = _probe_video_frame_count(episode_dir / "overview.mp4")
    if frame_count <= 0:
        frame_count = _estimate_overview_frame_count(episode_dir, fps)
    if frame_count <= 0 or fps <= 0:
        return np.empty((0,), dtype=np.int64)
    frame_period_ns = int(round(1_000_000_000.0 / fps))
    return start_monotonic_ns + np.arange(frame_count, dtype=np.int64) * frame_period_ns


def _probe_video_frame_count(path: Path) -> int:
    if not path.exists():
        return 0
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames,nb_frames,duration,avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=10.0)
    except (OSError, subprocess.TimeoutExpired):
        return 0
    if result.returncode != 0:
        return 0
    try:
        streams = json.loads(result.stdout).get("streams", [])
    except json.JSONDecodeError:
        return 0
    if not streams:
        return 0
    stream = streams[0]
    for key in ["nb_read_frames", "nb_frames"]:
        value = stream.get(key)
        try:
            if value not in (None, "N/A"):
                count = int(value)
                if count > 0:
                    return count
        except ValueError:
            pass
    return _frame_count_from_duration(stream)


def _frame_count_from_duration(stream: dict[str, Any]) -> int:
    try:
        duration = float(stream.get("duration"))
    except (TypeError, ValueError):
        return 0
    rate = stream.get("avg_frame_rate") or ""
    try:
        if "/" in rate:
            numerator, denominator = rate.split("/", 1)
            fps = float(numerator) / max(float(denominator), 1e-9)
        else:
            fps = float(rate)
    except (TypeError, ValueError):
        fps = 0.0
    if duration <= 0 or fps <= 0:
        return 0
    return max(0, int(round(duration * fps)))


def _estimate_overview_frame_count(episode_dir: Path, fps: float) -> int:
    metadata_path = episode_dir / "episode_metadata.json"
    if not metadata_path.exists():
        return 0
    try:
        metadata = json.loads(metadata_path.read_text())
        start = int(metadata["start_monotonic_ns"])
        end = int(metadata["end_monotonic_ns"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 0
    duration_sec = max(0.0, (end - start) / 1_000_000_000.0)
    return int(round(duration_sec * float(fps)))
