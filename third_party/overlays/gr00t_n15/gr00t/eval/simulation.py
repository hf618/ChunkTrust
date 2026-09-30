# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from tqdm import tqdm

# Required for robocasa environments
import robocasa  # noqa: F401
import robosuite  # noqa: F401
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401

from gr00t.data.dataset import ModalityConfig
from gr00t.eval.denoise_logger import DenoiseLogger
from gr00t.eval.service import BaseInferenceClient
from gr00t.eval.stride_selector import select_exec_k_stride
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
from gr00t.eval.wrappers.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)
from gr00t.model.policy import BasePolicy


@dataclass
class VideoConfig:
    """Configuration for video recording settings."""

    video_dir: Optional[str] = None
    steps_per_render: int = 2
    max_episode_steps: int = 720
    fps: int = 10
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 1
    file_name_mode: str = "episode_success"


@dataclass
class MultiStepConfig:
    """Configuration for multi-step environment settings."""

    video_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    state_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    n_action_steps: int = 16
    max_episode_steps: int = 1440


@dataclass
class StrideEvalConfig:
    eval_type: str = "default"
    dump_denoise: bool = False
    horizon_candidates: str = "4,8,12,16"
    horizon_exec_mode: str = "expected_round"
    horizon_expected_temp: float = 1.0
    horizon_intra_alpha: float = 4.0
    horizon_intra_cut_t: float = 0.25
    horizon_ts_epsilon: float = 0.05
    horizon_ts_seed: int = 0
    horizon_ts_update_mode: str = "kernel_forget"
    horizon_ts_kernel_bandwidth: float = 10.0
    horizon_ts_forget_rho: float = 0.99
    horizon_ts_update_eta: float = 1.0
    horizon_intra_z_mode: str = "window_rms"
    horizon_intra_tau_window: int = 1
    horizon_mix_mode: str = "both"
    horizon_inter_use_speed_uniformity: bool = True
    horizon_inter_window: int = 20
    horizon_inter_weight: float = 0.5
    horizon_inter_beta: float = 4.0
    horizon_inter_fallback: str = "ehh_only"
    replan_log_path: str = ""
    run_summary_path: str = ""

    @property
    def stride_candidates(self) -> str:
        return self.horizon_candidates

    @property
    def stride_exec_mode(self) -> str:
        return self.horizon_exec_mode

    @property
    def stride_expected_temp(self) -> float:
        return self.horizon_expected_temp

    @property
    def stride_ts_alpha(self) -> float:
        return self.horizon_intra_alpha

    @property
    def stride_ts_epsilon(self) -> float:
        return self.horizon_ts_epsilon

    @property
    def stride_ts_seed(self) -> int:
        return self.horizon_ts_seed

    @property
    def stride_ts_update_mode(self) -> str:
        return self.horizon_ts_update_mode

    @property
    def stride_ts_kernel_bandwidth(self) -> float:
        return self.horizon_ts_kernel_bandwidth

    @property
    def stride_ts_forget_rho(self) -> float:
        return self.horizon_ts_forget_rho

    @property
    def stride_ts_update_eta(self) -> float:
        return self.horizon_ts_update_eta

    @property
    def stride_z_mode(self) -> str:
        return self.horizon_intra_z_mode

    @property
    def stride_tau_window(self) -> int:
        return self.horizon_intra_tau_window

    @property
    def stride_signal_mode(self) -> str:
        return self.horizon_mix_mode

    @property
    def stride_use_speed_uniformity(self) -> bool:
        return self.horizon_inter_use_speed_uniformity

    @property
    def stride_speed_window(self) -> int:
        return self.horizon_inter_window

    @property
    def stride_speed_weight(self) -> float:
        return self.horizon_inter_weight

    @property
    def stride_speed_beta(self) -> float:
        return self.horizon_inter_beta

    @property
    def stride_speed_fallback(self) -> str:
        return self.horizon_inter_fallback


@dataclass
class SimulationConfig:
    """Main configuration for simulation environment."""

    env_name: str
    n_episodes: int = 2
    n_envs: int = 1
    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)
    eval: StrideEvalConfig = field(default_factory=StrideEvalConfig)


def _video_dir_for_env(video_dir: Path, env_idx: int, total_n_envs: int) -> Path:
    if total_n_envs <= 1:
        return video_dir
    return video_dir / f"env{env_idx:02d}"


def _normalize_task_descriptions(
    observations: dict[str, Any],
    batch_size: int,
) -> list[str | None]:
    raw_task_description = None
    for key in (
        "annotation.human.action.task_description",
        "annotation.human.coarse_action",
        "task",
    ):
        if key in observations:
            raw_task_description = observations[key]
            break

    if raw_task_description is None:
        return [None] * batch_size

    if isinstance(raw_task_description, np.ndarray):
        raw_task_description = raw_task_description.tolist()

    if isinstance(raw_task_description, (list, tuple)):
        task_descriptions = list(raw_task_description)
    else:
        task_descriptions = [raw_task_description]

    normalized = []
    for item in task_descriptions:
        if isinstance(item, np.ndarray):
            item = item.tolist()
        if isinstance(item, (list, tuple)) and len(item) == 1:
            item = item[0]
        normalized.append(None if item is None else str(item))

    if not normalized:
        normalized = [None] * batch_size
    elif len(normalized) == 1 and batch_size > 1:
        normalized = normalized * batch_size
    elif len(normalized) != batch_size:
        raise RuntimeError(
            "Task description batch size mismatch: "
            f"got {len(normalized)} descriptions for batch size {batch_size}"
        )
    return normalized


def _flatten_action_chunks(actions: dict[str, Any], action_modality_keys: list[str]) -> np.ndarray:
    flat_chunks = []
    horizon = None
    for key in action_modality_keys:
        parsed_key = key if key.startswith("action.") else f"action.{key}"
        if parsed_key not in actions:
            raise KeyError(f"Missing action key {parsed_key} in policy output")
        arr = np.asarray(actions[parsed_key], dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Unexpected action chunk shape for {parsed_key}: {arr.shape}")
        if horizon is None:
            horizon = arr.shape[1]
        else:
            horizon = min(horizon, arr.shape[1])
        flat_chunks.append(arr)

    if not flat_chunks or horizon is None:
        raise ValueError("No action chunks found to flatten")

    return np.concatenate([arr[:, :horizon, :] for arr in flat_chunks], axis=-1)


class EvalTraceManager:
    def __init__(
        self,
        *,
        stride_config: StrideEvalConfig,
        n_envs: int,
        n_action_steps: int,
    ):
        self.stride_config = stride_config
        self.n_envs = int(n_envs)
        self.n_action_steps = int(n_action_steps)
        self.enabled = bool(
            self.stride_config.dump_denoise
            or self.stride_config.eval_type == "horizon"
            or self.stride_config.replan_log_path
        )
        self.replan_log_path = (
            Path(self.stride_config.replan_log_path)
            if self.stride_config.replan_log_path
            else None
        )
        if self.replan_log_path is not None:
            self.replan_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._denoise_loggers = {
            env_idx: DenoiseLogger(enabled=self.stride_config.dump_denoise)
            for env_idx in range(self.n_envs)
        }
        self._denoise_overhead_seconds = {env_idx: 0.0 for env_idx in range(self.n_envs)}
        self._episode_paths: dict[int, Path | None] = {
            env_idx: None for env_idx in range(self.n_envs)
        }
        self._episode_indices = {env_idx: 0 for env_idx in range(self.n_envs)}
        self._replan_indices = {env_idx: 0 for env_idx in range(self.n_envs)}
        self.executed_action_history = [
            np.zeros((0, 0), dtype=np.float32) for _ in range(self.n_envs)
        ]
        self.stride_ts_states = [dict() for _ in range(self.n_envs)]
        self.stride_ts_rngs = [
            np.random.default_rng(int(self.stride_config.stride_ts_seed) + env_idx)
            for env_idx in range(self.n_envs)
        ]
        self._reset_run_score_state()

    @staticmethod
    def _mean_finite(values) -> float | None:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size <= 0:
            return None
        finite = arr[np.isfinite(arr)]
        if finite.size <= 0:
            return None
        return float(np.mean(finite))

    def _reset_run_score_state(self) -> None:
        self._run_score_state = {
            "avg_q_intra": {"sum": 0.0, "count": 0},
            "avg_p_inter": {"sum": 0.0, "count": 0},
            "avg_q_mix": {"sum": 0.0, "count": 0},
            "avg_q_intra_contrib": {"sum": 0.0, "count": 0},
            "avg_p_inter_contrib": {"sum": 0.0, "count": 0},
            "avg_z_intra_raw": {"sum": 0.0, "count": 0},
            "avg_z_intra_norm": {"sum": 0.0, "count": 0},
            "avg_u_inter_raw": {"sum": 0.0, "count": 0},
            "avg_u_inter_norm": {"sum": 0.0, "count": 0},
        }

    def _record_run_score_mean(self, key: str, value: float | None) -> None:
        if value is None or not np.isfinite(value):
            return
        bucket = self._run_score_state[key]
        bucket["sum"] += float(value)
        bucket["count"] += 1

    def _compute_effective_inter_array(self, p_inter: np.ndarray, q_intra: np.ndarray) -> np.ndarray:
        p_arr = np.asarray(p_inter, dtype=np.float64)
        q_arr = np.asarray(q_intra, dtype=np.float64)
        effective = np.array(p_arr, copy=True)
        finite_mask = np.isfinite(effective)
        if np.any(finite_mask):
            effective[finite_mask] = np.clip(effective[finite_mask], 0.0, 1.0)
        invalid_mask = ~finite_mask
        if np.any(invalid_mask):
            if self.stride_config.horizon_inter_fallback == "neutral":
                effective[invalid_mask] = 0.5
            else:
                effective[invalid_mask] = q_arr[invalid_mask]
        return effective

    def update_run_score_stats(self, horizon_info: dict[str, Any] | None) -> None:
        if self.stride_config.eval_type != "horizon" or not horizon_info:
            return
        q_intra = np.asarray(horizon_info.get("q_intra"), dtype=np.float64)
        p_inter = np.asarray(horizon_info.get("p_inter"), dtype=np.float64)
        q_mix = np.asarray(horizon_info.get("q_mix"), dtype=np.float64)
        z_intra_raw = np.asarray(horizon_info.get("z_intra"), dtype=np.float64)
        z_intra_norm = np.asarray(horizon_info.get("z_intra_norm"), dtype=np.float64)
        u_inter_raw = np.asarray(horizon_info.get("u_inter_raw"), dtype=np.float64)
        u_inter_norm = np.asarray(horizon_info.get("u_inter_norm"), dtype=np.float64)
        if q_intra.size <= 0 or q_mix.size <= 0:
            return

        p_effective = self._compute_effective_inter_array(p_inter, q_intra)
        finite_p = np.isfinite(p_inter)
        invalid_p = ~finite_p
        q_intra_contrib = np.zeros_like(q_intra, dtype=np.float64)
        p_inter_contrib = np.zeros_like(q_intra, dtype=np.float64)

        if self.stride_config.horizon_mix_mode == "intra":
            q_intra_contrib = q_mix
        elif self.stride_config.horizon_mix_mode == "inter":
            if np.any(finite_p):
                p_inter_contrib[finite_p] = p_effective[finite_p]
            if np.any(invalid_p):
                if self.stride_config.horizon_inter_fallback == "neutral":
                    p_inter_contrib[invalid_p] = 0.5
                else:
                    q_intra_contrib[invalid_p] = q_intra[invalid_p]
        else:
            if np.any(finite_p):
                w = float(self.stride_config.horizon_inter_weight)
                q_intra_contrib[finite_p] = (1.0 - w) * q_intra[finite_p]
                p_inter_contrib[finite_p] = w * p_effective[finite_p]
            if np.any(invalid_p):
                if self.stride_config.horizon_inter_fallback == "neutral":
                    w = float(self.stride_config.horizon_inter_weight)
                    q_intra_contrib[invalid_p] = (1.0 - w) * q_intra[invalid_p]
                    p_inter_contrib[invalid_p] = w * 0.5
                else:
                    q_intra_contrib[invalid_p] = q_intra[invalid_p]

        self._record_run_score_mean("avg_q_intra", self._mean_finite(q_intra))
        self._record_run_score_mean("avg_p_inter", self._mean_finite(p_inter))
        self._record_run_score_mean("avg_q_mix", self._mean_finite(q_mix))
        self._record_run_score_mean("avg_q_intra_contrib", self._mean_finite(q_intra_contrib))
        self._record_run_score_mean("avg_p_inter_contrib", self._mean_finite(p_inter_contrib))
        self._record_run_score_mean("avg_z_intra_raw", self._mean_finite(z_intra_raw))
        self._record_run_score_mean("avg_z_intra_norm", self._mean_finite(z_intra_norm))
        self._record_run_score_mean("avg_u_inter_raw", self._mean_finite(u_inter_raw))
        self._record_run_score_mean("avg_u_inter_norm", self._mean_finite(u_inter_norm))

    def _format_horizon_policy_name(self) -> str:
        parts = ["horizon", "ehh1dpad", str(self.stride_config.horizon_intra_z_mode)]
        if self.stride_config.horizon_inter_use_speed_uniformity and self.stride_config.horizon_mix_mode in ("inter", "both"):
            parts.append("speedu")
        parts.append(str(self.stride_config.horizon_mix_mode))
        parts.append(str(self.stride_config.horizon_exec_mode))
        if self.stride_config.horizon_ts_update_mode != "independent":
            parts.append(str(self.stride_config.horizon_ts_update_mode))
        if self.stride_config.horizon_exec_mode == "expected_round":
            parts.append(f"temp{float(self.stride_config.horizon_expected_temp):g}".replace(".", "p"))
        return "_".join(parts)

    def get_run_score_averages(self) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for key, bucket in self._run_score_state.items():
            count = int(bucket["count"])
            out[key] = None if count <= 0 else float(bucket["sum"] / float(count))
        return out

    def get_horizon_run_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "horizon_policy": self._format_horizon_policy_name(),
            "horizon_candidates": str(self.stride_config.horizon_candidates),
            "horizon_exec_mode": str(self.stride_config.horizon_exec_mode),
            "horizon_expected_temp": float(self.stride_config.horizon_expected_temp),
            "horizon_intra_alpha": float(self.stride_config.horizon_intra_alpha),
            "horizon_intra_z_mode": str(self.stride_config.horizon_intra_z_mode),
            "horizon_intra_tau_window": int(self.stride_config.horizon_intra_tau_window),
            "horizon_inter_weight": float(self.stride_config.horizon_inter_weight),
            "horizon_inter_beta": float(self.stride_config.horizon_inter_beta),
            "horizon_inter_window": int(self.stride_config.horizon_inter_window),
            "horizon_ts_update_mode": str(self.stride_config.horizon_ts_update_mode),
            "horizon_ts_kernel_bandwidth": float(self.stride_config.horizon_ts_kernel_bandwidth),
            "horizon_ts_forget_rho": float(self.stride_config.horizon_ts_forget_rho),
            "horizon_ts_update_eta": float(self.stride_config.horizon_ts_update_eta),
        }
        summary.update(self.get_run_score_averages())
        return summary

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return [EvalTraceManager._json_safe(v) for v in value]
        if isinstance(value, list):
            return [EvalTraceManager._json_safe(v) for v in value]
        if isinstance(value, dict):
            return {str(k): EvalTraceManager._json_safe(v) for k, v in value.items()}
        return value

    @staticmethod
    def _float_list_or_nan(values) -> np.ndarray:
        return np.asarray(
            [np.nan if value is None else float(value) for value in values],
            dtype=np.float32,
        )

    def _stride_info_to_npz_extras(
        self, stride_info: dict[str, Any] | None
    ) -> dict[str, np.ndarray]:
        if not stride_info:
            return {}

        return {
            "stride_values": np.asarray(stride_info.get("strides", []), dtype=np.int32),
            "stride_z_ehh_rms_from_tau0": np.asarray(
                stride_info.get("z_ehh_rms_from_tau0", []),
                dtype=np.float32,
            ),
            "stride_q_proxy": np.asarray(stride_info.get("q_proxy", []), dtype=np.float32),
            "stride_q_mix": np.asarray(stride_info.get("q_mix", []), dtype=np.float32),
            "stride_score": np.asarray(stride_info.get("score", []), dtype=np.float32),
            "stride_speed_uniformity_cv": self._float_list_or_nan(
                stride_info.get("speed_uniformity_cv", [])
            ),
            "stride_speed_uniformity_proxy": self._float_list_or_nan(
                stride_info.get("speed_uniformity_proxy", [])
            ),
            "stride_score_prob": np.asarray(
                stride_info.get("score_prob", []),
                dtype=np.float32,
            ),
            "stride_expected_prob": np.asarray(
                stride_info.get("expected_prob", []),
                dtype=np.float32,
            ),
            "stride_theta_sample": np.asarray(
                stride_info.get("theta_sample", []),
                dtype=np.float32,
            ),
            "stride_posterior_alpha_before": np.asarray(
                stride_info.get("posterior_alpha_before", []),
                dtype=np.float32,
            ),
            "stride_posterior_beta_before": np.asarray(
                stride_info.get("posterior_beta_before", []),
                dtype=np.float32,
            ),
            "stride_posterior_alpha_after": np.asarray(
                stride_info.get("posterior_alpha_after", []),
                dtype=np.float32,
            ),
            "stride_posterior_beta_after": np.asarray(
                stride_info.get("posterior_beta_after", []),
                dtype=np.float32,
            ),
            "stride_expected_stride": np.asarray(
                float(stride_info.get("expected_stride", np.nan)),
                dtype=np.float32,
            ),
            "stride_chosen_stride": np.asarray(
                int(stride_info.get("chosen_stride", 0)),
                dtype=np.int32,
            ),
            "stride_update_center": np.asarray(
                float(stride_info.get("update_center", np.nan)),
                dtype=np.float32,
            ),
            "stride_update_feedback": np.asarray(
                float(stride_info.get("update_feedback", np.nan)),
                dtype=np.float32,
            ),
            "stride_ehh_curve_by_stride": np.asarray(
                stride_info.get("ehh_curve_by_stride", []),
                dtype=np.float32,
            ),
            "stride_selector_mode": np.asarray(
                str(stride_info.get("selector_mode", "")),
            ),
            "stride_choose_mode": np.asarray(
                str(stride_info.get("choose_mode", "")),
            ),
            "stride_info_json": np.asarray(
                json.dumps(self._json_safe(stride_info), ensure_ascii=False)
            ),
        }

    def _resolve_episode_path(self, video_dir: str | None, env_idx: int) -> Path | None:
        if video_dir:
            video_dir_path = Path(video_dir)
            if video_dir_path.parent.name.lower() == "videos":
                episode_dir = video_dir_path.parent.parent / "episodes" / video_dir_path.name
            elif video_dir_path.name.lower() == "videos":
                episode_dir = video_dir_path.parent / "episodes"
            else:
                episode_dir = video_dir_path.parent / "episodes"
        elif self.replan_log_path is not None:
            if self.n_envs > 1:
                episode_dir = self.replan_log_path.parent / "episodes" / f"env{env_idx:02d}"
            else:
                episode_dir = self.replan_log_path.parent / "episodes"
        else:
            return None

        episode_dir.mkdir(parents=True, exist_ok=True)
        episode_idx = int(self._episode_indices[env_idx])
        return episode_dir / f"episode{episode_idx}.npz"

    def begin_episode(self, env_idx: int, video_dir: str | None = None) -> None:
        if not self.enabled:
            return
        env_idx = int(env_idx)
        self._episode_indices[env_idx] += 1
        self._replan_indices[env_idx] = 0
        self._denoise_overhead_seconds[env_idx] = 0.0
        self.executed_action_history[env_idx] = np.zeros((0, 0), dtype=np.float32)
        self.stride_ts_states[env_idx] = {}
        self.stride_ts_rngs[env_idx] = np.random.default_rng(
            int(self.stride_config.stride_ts_seed) + env_idx
        )
        logger = self._denoise_loggers[env_idx]
        logger.reset_episode()
        episode_path = self._resolve_episode_path(video_dir=video_dir, env_idx=env_idx)
        self._episode_paths[env_idx] = episode_path
        if episode_path is not None:
            logger.start_episode(episode_path)

    def ensure_action_history(self, action_dim: int) -> None:
        for env_idx in range(self.n_envs):
            history = self.executed_action_history[env_idx]
            if history.shape[1] != int(action_dim):
                self.executed_action_history[env_idx] = np.zeros(
                    (0, int(action_dim)),
                    dtype=np.float32,
                )

    def append_executed_actions(
        self, flat_actions: np.ndarray, exec_horizon: np.ndarray
    ) -> None:
        if not self.enabled:
            return
        for env_idx, k_exec in enumerate(np.asarray(exec_horizon).reshape(-1).tolist()):
            executed = np.asarray(flat_actions[env_idx, : int(k_exec), :], dtype=np.float32)
            if executed.size <= 0:
                continue
            history = self.executed_action_history[env_idx]
            if history.size == 0 or history.shape[1] != executed.shape[1]:
                self.executed_action_history[env_idx] = executed.copy()
            else:
                self.executed_action_history[env_idx] = np.concatenate(
                    [history, executed],
                    axis=0,
                )

    def log_replan(
        self,
        *,
        task_descriptions: list[str | None],
        exec_horizon: np.ndarray,
        stride_infos: list[dict[str, Any] | None] | None,
    ) -> None:
        if self.replan_log_path is None:
            return

        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        records = []
        for env_idx, k_exec in enumerate(np.asarray(exec_horizon).reshape(-1)):
            stride_info = None
            if stride_infos is not None and env_idx < len(stride_infos):
                stride_info = stride_infos[env_idx]
            records.append(
                {
                    "timestamp": timestamp,
                    "episode_index": int(self._episode_indices[env_idx]),
                    "replan_index": int(self._replan_indices[env_idx]),
                    "batch_index": int(env_idx),
                    "task_description": self._json_safe(task_descriptions[env_idx]),
                    "eval_type": self.stride_config.eval_type,
                    "exec_horizon": int(k_exec),
                    "horizon_info": self._json_safe(stride_info),
                }
            )

        with self.replan_log_path.open("a", encoding="utf-8") as fp:
            for record in records:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_denoise_step(
        self,
        *,
        denoise_trace: dict[str, Any] | None,
        flat_actions: np.ndarray,
        exec_horizon: np.ndarray,
        stride_infos: list[dict[str, Any] | None] | None,
    ) -> None:
        if not self.stride_config.dump_denoise or denoise_trace is None:
            return

        trace_v = np.asarray(denoise_trace["v"], dtype=np.float32)
        used_h = min(int(self.n_action_steps), int(flat_actions.shape[1]))
        trace_ds = denoise_trace.get("ds")
        trace_x0 = denoise_trace.get("x0")
        trace_xt_internal = denoise_trace.get("xt_internal")

        for env_idx in range(flat_actions.shape[0]):
            episode_path = self._episode_paths.get(int(env_idx))
            if episode_path is None:
                continue

            k_exec = int(np.asarray(exec_horizon).reshape(-1)[env_idx])
            xt_log = np.array(flat_actions[env_idx, :used_h, :], dtype=np.float32, copy=True)
            if k_exec < xt_log.shape[0]:
                xt_log[k_exec:, :] = 0.0

            xt_internal_log = None
            if trace_xt_internal is not None:
                xt_internal_log = np.asarray(
                    trace_xt_internal[env_idx, :used_h, :],
                    dtype=np.float32,
                ).copy()
                if k_exec < xt_internal_log.shape[0]:
                    xt_internal_log[k_exec:, :] = 0.0

            x0 = None
            if trace_x0 is not None:
                x0 = np.asarray(trace_x0[env_idx, :used_h, :], dtype=np.float32)

            ds = None
            if trace_ds is not None:
                ds_arr = np.asarray(trace_ds)
                if ds_arr.ndim == 2:
                    ds = np.asarray(ds_arr[env_idx], dtype=np.float32)
                else:
                    ds = np.asarray(ds_arr, dtype=np.float32)

            extras = {}
            if self.stride_config.eval_type == "horizon" and stride_infos is not None:
                extras = self._stride_info_to_npz_extras(stride_infos[env_idx])

            t0 = time.time()
            self._denoise_loggers[env_idx].log_step(
                np.asarray(trace_v[env_idx, :, :used_h, :], dtype=np.float32),
                ds=ds,
                x0=x0,
                xt=xt_log,
                xt_internal=xt_internal_log,
                k_exec=k_exec if self.stride_config.eval_type == "horizon" else None,
                extras=extras,
            )
            self._denoise_overhead_seconds[env_idx] += float(time.time() - t0)

    def finalize_episode(
        self,
        *,
        env_idx: int,
        success: bool,
        success_step: int | None,
        episode_steps: int,
        run_time_seconds: float,
    ) -> None:
        if not self.enabled:
            return
        logger = self._denoise_loggers[int(env_idx)]
        logger.set_episode_info(
            success=success,
            success_step=success_step,
            episode_steps=episode_steps,
            run_time_seconds=run_time_seconds,
            denoise_overhead_seconds=self._denoise_overhead_seconds[int(env_idx)],
        )
        logger.flush_episode()

    def advance_replan_indices(self) -> None:
        if not self.enabled:
            return
        for env_idx in range(self.n_envs):
            self._replan_indices[env_idx] += 1


def _success_from_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    if isinstance(value, list):
        return bool(np.any(np.asarray(value)))
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return bool(value)


class SimulationInferenceClient(BaseInferenceClient, BasePolicy):
    """Client for running simulations and communicating with the inference server."""

    def __init__(self, host: str = "localhost", port: int = 5555):
        """Initialize the simulation client with server connection details."""
        super().__init__(host=host, port=port)
        self.env = None

    def get_action(
        self,
        observations: Dict[str, Any],
        options: Dict[str, Any] | None = None,
    ) -> Any:
        """Get action from the inference server based on observations."""
        obs_copy = observations.copy()
        if "video.ego_view_bg_crop_pad_res256_freq20" in obs_copy:
            obs_copy["video.ego_view"] = obs_copy.pop(
                "video.ego_view_bg_crop_pad_res256_freq20"
            )
        payload = obs_copy if options is None else {"observation": obs_copy, "options": options}
        return self.call_endpoint("get_action", payload)

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """Get modality configuration from the inference server."""
        return self.call_endpoint("get_modality_config", requires_input=False)

    def setup_environment(self, config: SimulationConfig) -> gym.vector.VectorEnv:
        """Set up the simulation environment based on the provided configuration."""
        env_fns = [
            partial(_create_single_env, config=config, idx=i) for i in range(config.n_envs)
        ]
        if config.n_envs == 1:
            return gym.vector.SyncVectorEnv(env_fns)
        return gym.vector.AsyncVectorEnv(
            env_fns,
            shared_memory=False,
            context="spawn",
        )

    def run_simulation(self, config: SimulationConfig) -> Tuple[str, List[bool]]:
        """Run the simulation for the specified number of episodes."""
        start_time = time.time()
        print(
            f"Running {config.n_episodes} episodes for {config.env_name} with {config.n_envs} environments"
        )

        self.env = self.setup_environment(config)
        action_modality_keys = self.get_modality_config()["action"].modality_keys
        trace_manager = EvalTraceManager(
            stride_config=config.eval,
            n_envs=config.n_envs,
            n_action_steps=config.multistep.n_action_steps,
        )

        current_rewards = [0] * config.n_envs
        current_lengths = [0] * config.n_envs
        current_successes = [False] * config.n_envs
        first_success_steps: list[int | None] = [None] * config.n_envs
        episode_start_times = [time.time()] * config.n_envs
        episode_successes = []
        completed_episodes = 0

        obs, _ = self.env.reset()
        for env_idx in range(config.n_envs):
            env_video_dir = None
            if config.video.video_dir is not None:
                env_video_dir = str(
                    _video_dir_for_env(Path(config.video.video_dir), env_idx, config.n_envs)
                )
            trace_manager.begin_episode(env_idx=env_idx, video_dir=env_video_dir)

        pbar = tqdm(total=config.n_episodes, desc="Episodes")
        stop_collection = False
        while completed_episodes < config.n_episodes:
            batch_size = int(
                len(
                    next(
                        value
                        for key, value in obs.items()
                        if key.startswith("video.") or key.startswith("state.")
                    )
                )
            )
            task_descriptions = _normalize_task_descriptions(obs, batch_size)
            policy_options = None
            if config.eval.eval_type == "horizon" or config.eval.dump_denoise:
                policy_options = {"return_trace": True}

            actions, policy_info = self._get_actions_from_server(obs, options=policy_options)
            exec_horizon = np.full(
                (batch_size,),
                config.multistep.n_action_steps,
                dtype=np.int32,
            )
            stride_infos: list[dict[str, Any] | None] | None = None
            flat_actions = None
            if (
                config.eval.eval_type == "horizon"
                or config.eval.dump_denoise
                or trace_manager.replan_log_path is not None
            ):
                flat_actions = _flatten_action_chunks(actions, action_modality_keys)
                trace_manager.ensure_action_history(flat_actions.shape[-1])
                if config.eval.eval_type == "horizon":
                    denoise_trace = policy_info.get("denoise_trace")
                    if denoise_trace is None:
                        raise RuntimeError(
                            "eval_type='horizon' requires denoise_trace from the policy output."
                        )
                    v_trace = np.asarray(denoise_trace["v"], dtype=np.float32)
                    if v_trace.ndim != 4:
                        raise RuntimeError(
                            f"Unexpected denoise_trace['v'] shape: {v_trace.shape}"
                        )
                    stride_infos = []
                    used_h = min(
                        int(config.multistep.n_action_steps),
                        int(flat_actions.shape[1]),
                    )
                    for env_idx in range(batch_size):
                        k_exec, stride_info = select_exec_k_stride(
                            v_trace[env_idx, :, :used_h, :],
                            used_h,
                            stride_candidates=config.eval.stride_candidates,
                            exec_mode=config.eval.stride_exec_mode,
                            expected_temp=config.eval.stride_expected_temp,
                        stride_alpha=config.eval.stride_ts_alpha,
                        intra_cut_t=config.eval.horizon_intra_cut_t,
                        ts_epsilon=config.eval.stride_ts_epsilon,
                            ts_update_mode=config.eval.stride_ts_update_mode,
                            ts_kernel_bandwidth=config.eval.stride_ts_kernel_bandwidth,
                            ts_forget_rho=config.eval.stride_ts_forget_rho,
                            ts_update_eta=config.eval.stride_ts_update_eta,
                            z_mode=config.eval.stride_z_mode,
                            tau_window=config.eval.stride_tau_window,
                            xt_step=np.asarray(
                                flat_actions[env_idx, :used_h, :],
                                dtype=np.float32,
                            ),
                            history_actions=trace_manager.executed_action_history[env_idx],
                            signal_mode=config.eval.stride_signal_mode,
                            use_speed_uniformity=config.eval.stride_use_speed_uniformity,
                            speed_window=config.eval.stride_speed_window,
                            speed_weight=config.eval.stride_speed_weight,
                            speed_beta=config.eval.stride_speed_beta,
                            speed_fallback=config.eval.stride_speed_fallback,
                            ts_state=trace_manager.stride_ts_states[env_idx],
                            ts_rng=trace_manager.stride_ts_rngs[env_idx],
                        )
                        exec_horizon[env_idx] = int(k_exec)
                        stride_infos.append(stride_info)
                        trace_manager.update_run_score_stats(stride_info)
                    trace_manager.append_executed_actions(flat_actions, exec_horizon)

                actions["exec_horizon"] = exec_horizon[:, None]
                trace_manager.log_replan(
                    task_descriptions=task_descriptions,
                    exec_horizon=exec_horizon,
                    stride_infos=stride_infos,
                )
                trace_manager.log_denoise_step(
                    denoise_trace=policy_info.get("denoise_trace"),
                    flat_actions=flat_actions,
                    exec_horizon=exec_horizon,
                    stride_infos=stride_infos,
                )
                trace_manager.advance_replan_indices()

            next_obs, rewards, terminations, truncations, env_infos = self.env.step(actions)
            for env_idx in range(config.n_envs):
                if "success" in env_infos:
                    current_successes[env_idx] |= _success_from_value(
                        env_infos["success"][env_idx]
                    )
                if (
                    "final_info" in env_infos
                    and env_infos["final_info"][env_idx] is not None
                    and "success" in env_infos["final_info"][env_idx]
                ):
                    current_successes[env_idx] |= _success_from_value(
                        env_infos["final_info"][env_idx]["success"]
                    )
                if current_successes[env_idx] and first_success_steps[env_idx] is None:
                    first_success_steps[env_idx] = current_lengths[env_idx] + 1

                current_rewards[env_idx] += rewards[env_idx]
                current_lengths[env_idx] += 1

                if terminations[env_idx] or truncations[env_idx]:
                    episode_successes.append(bool(current_successes[env_idx]))
                    trace_manager.finalize_episode(
                        env_idx=env_idx,
                        success=bool(current_successes[env_idx]),
                        success_step=first_success_steps[env_idx],
                        episode_steps=current_lengths[env_idx],
                        run_time_seconds=float(time.time() - episode_start_times[env_idx]),
                    )
                    current_successes[env_idx] = False
                    first_success_steps[env_idx] = None
                    current_rewards[env_idx] = 0
                    current_lengths[env_idx] = 0
                    episode_start_times[env_idx] = time.time()
                    completed_episodes += 1
                    pbar.update(1)

                    if completed_episodes >= config.n_episodes:
                        stop_collection = True
                        break

                    env_video_dir = None
                    if config.video.video_dir is not None:
                        env_video_dir = str(
                            _video_dir_for_env(
                                Path(config.video.video_dir),
                                env_idx,
                                config.n_envs,
                            )
                        )
                    trace_manager.begin_episode(env_idx=env_idx, video_dir=env_video_dir)

            if stop_collection:
                break
            obs = next_obs

        pbar.close()
        self.env.reset()
        self.env.close()
        self.env = None
        print(
            f"Collecting {config.n_episodes} episodes took {time.time() - start_time:.2f} seconds"
        )
        if config.eval.run_summary_path:
            summary_path = Path(config.eval.run_summary_path)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary = trace_manager.get_horizon_run_summary()
            summary["env_name"] = config.env_name
            summary["n_episodes"] = int(config.n_episodes)
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        assert len(episode_successes) == config.n_episodes, (
            f"Expected exactly {config.n_episodes} episodes, got {len(episode_successes)}"
        )
        return config.env_name, episode_successes

    def _get_actions_from_server(
        self,
        observations: Dict[str, Any],
        options: Dict[str, Any] | None = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Process observations and get actions from the inference server."""
        response = self.get_action(observations, options=options)
        info: Dict[str, Any] = {}
        action_dict: Dict[str, Any]

        if isinstance(response, (list, tuple)) and len(response) == 2:
            action_dict = response[0]
            info = response[1] if isinstance(response[1], dict) else {}
        else:
            action_dict = response

        if isinstance(action_dict, dict) and "actions" in action_dict:
            info = action_dict.get("info", info if isinstance(info, dict) else {})
            actions = action_dict["actions"]
        else:
            actions = action_dict
        return actions, info if isinstance(info, dict) else {}


def _create_single_env(config: SimulationConfig, idx: int) -> gym.Env:
    """Create a single environment with appropriate wrappers."""
    env = gym.make(config.env_name, enable_render=True)
    if config.video.video_dir is not None:
        base_video_dir = Path(config.video.video_dir)
        per_env_video_dir = _video_dir_for_env(base_video_dir, idx, config.n_envs)
        video_recorder = VideoRecorder.create_h264(
            fps=config.video.fps,
            codec=config.video.codec,
            input_pix_fmt=config.video.input_pix_fmt,
            crf=config.video.crf,
            thread_type=config.video.thread_type,
            thread_count=config.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=per_env_video_dir,
            steps_per_render=config.video.steps_per_render,
            max_episode_steps=config.video.max_episode_steps,
            file_name_mode=config.video.file_name_mode,
        )
    env = MultiStepWrapper(
        env,
        video_delta_indices=config.multistep.video_delta_indices,
        state_delta_indices=config.multistep.state_delta_indices,
        n_action_steps=config.multistep.n_action_steps,
        max_episode_steps=config.multistep.max_episode_steps,
    )
    return env


def run_evaluation(
    env_name: str,
    host: str = "localhost",
    port: int = 5555,
    video_dir: Optional[str] = None,
    n_episodes: int = 2,
    n_envs: int = 1,
    n_action_steps: int = 2,
    max_episode_steps: int = 100,
    eval_config: StrideEvalConfig | None = None,
) -> Tuple[str, List[bool]]:
    """
    Simple entry point to run a simulation evaluation.
    """
    config = SimulationConfig(
        env_name=env_name,
        n_episodes=n_episodes,
        n_envs=n_envs,
        video=VideoConfig(
            video_dir=video_dir,
            max_episode_steps=max_episode_steps,
            file_name_mode="episode_success",
        ),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps, max_episode_steps=max_episode_steps
        ),
        eval=eval_config or StrideEvalConfig(),
    )
    client = SimulationInferenceClient(host=host, port=port)
    results = client.run_simulation(config)
    print(f"Results for {env_name}:")
    print(f"Success rate: {np.mean(results[1]):.2f}")
    return results


if __name__ == "__main__":
    run_evaluation(
        env_name="robocasa_gr1_arms_only_fourier_hands/TwoArmPnPCarPartBrakepedal_GR1ArmsOnlyFourierHands_Env",
        host="localhost",
        port=5555,
        video_dir="./videos",
    )
