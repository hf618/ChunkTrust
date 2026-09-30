import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
import json
from pathlib import Path
import time
from typing import Any
import uuid

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.denoise_logger import DenoiseLogger
from gr00t.eval.sim.env_utils import get_embodiment_tag_from_env_name
from gr00t.eval.horizon_selector import select_exec_k_horizon
from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
from gr00t.policy import BasePolicy
import gymnasium as gym
import numpy as np
from tqdm import tqdm


@dataclass
class VideoConfig:
    """Configuration for video recording settings.

    Attributes:
        video_dir: Directory to save videos (if None, no videos are saved)
        steps_per_render: Number of steps between each call to env.render() while recording
            during rollout
        fps: Frames per second for the output video
        codec: Video codec to use for compression
        input_pix_fmt: Input pixel format
        crf: Constant Rate Factor for video compression (lower = better quality)
        thread_type: Threading strategy for video encoding
        thread_count: Number of threads to use for encoding
    """

    video_dir: str | None = None
    steps_per_render: int = 2
    max_episode_steps: int = 720
    fps: int = 20
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 1
    overlay_text: bool = True
    n_action_steps: int = 8
    file_name_mode: str = "legacy"


@dataclass
class MultiStepConfig:
    """Configuration for multi-step environment settings.

    Attributes:
        video_delta_indices: Indices of video observations to stack
        state_delta_indices: Indices of state observations to stack
        n_action_steps: Number of action steps to execute
        max_episode_steps: Maximum number of steps per episode
    """

    video_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    state_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    n_action_steps: int = 16
    max_episode_steps: int = 720
    terminate_on_success: bool = False


@dataclass
class WrapperConfigs:
    """Container for various environment wrapper configurations.

    Attributes:
        video: Configuration for video recording
        multistep: Configuration for multi-step processing
    """

    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)


@dataclass
class HorizoEvalConfig:
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
    def horizon_ts_alpha(self) -> float:
        return self.horizon_intra_alpha

    @property
    def horizon_z_mode(self) -> str:
        return self.horizon_intra_z_mode

    @property
    def horizon_tau_window(self) -> int:
        return self.horizon_intra_tau_window

    @property
    def horizon_signal_mode(self) -> str:
        return self.horizon_mix_mode

    @property
    def horizon_use_speed_uniformity(self) -> bool:
        return self.horizon_inter_use_speed_uniformity

    @property
    def horizon_speed_window(self) -> int:
        return self.horizon_inter_window

    @property
    def horizon_speed_weight(self) -> float:
        return self.horizon_inter_weight

    @property
    def horizon_speed_beta(self) -> float:
        return self.horizon_inter_beta

    @property
    def horizon_speed_fallback(self) -> str:
        return self.horizon_inter_fallback


def get_robocasa_env_fn(
    env_name: str,
):
    def env_fn():
        import os

        import robocasa  # noqa: F401
        from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401
        import robosuite  # noqa: F401

        os.environ["MUJOCO_GL"] = "egl"
        return gym.make(env_name, enable_render=True)

    return env_fn


def get_groot_locomanip_env_fn(
    env_name: str,
):
    def env_fn():
        from gr00t_wbc.control.envs.robocasa.sync_env import SyncEnv  # noqa: F401
        from gr00t_wbc.control.main.teleop.configs.configs import BaseConfig
        from gr00t_wbc.control.utils.n1_utils import WholeBodyControlWrapper
        import robocasa  # noqa: F401

        gym_env = gym.make(
            env_name,
            onscreen=False,
            offscreen=True,
            enable_waist=True,
            randomize_cameras=False,
            camera_names=[
                "robot0_oak_egoview",
                "robot0_rs_tppview",
            ],
        )
        wbc_config = BaseConfig(wbc_version="gear_wbc", enable_waist=True).to_dict()
        gym_env = WholeBodyControlWrapper(gym_env, wbc_config)
        return gym_env

    return env_fn


def get_simpler_env_fn(
    env_name: str,
):
    def env_fn():
        from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs

        register_simpler_envs()
        return gym.make(env_name)

    return env_fn


def get_libero_env_fn(
    env_name: str,
):
    def env_fn():
        from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs

        register_libero_envs()
        return gym.make(env_name)

    return env_fn


def get_behavior_env_fn(
    env_name: str,
    env_idx: int,
    total_n_envs: int,
):
    def env_fn():
        from gr00t.eval.sim.BEHAVIOR.behavior_env import register_behavior_envs

        register_behavior_envs()
        return gym.make(env_name, env_idx=env_idx, total_n_envs=total_n_envs)

    return env_fn


def get_gym_env(env_name: str, env_idx: int, total_n_envs: int):
    """Create Ray environment factory function without wrappers."""

    env_embodiment = get_embodiment_tag_from_env_name(env_name)

    if env_embodiment in (
        EmbodimentTag.GR1,
        EmbodimentTag.ROBOCASA_PANDA_OMRON,
    ):
        env_fn = get_robocasa_env_fn(env_name)

    elif env_embodiment in (EmbodimentTag.UNITREE_G1,):
        env_fn = get_groot_locomanip_env_fn(env_name)

    elif env_embodiment in (EmbodimentTag.OXE_GOOGLE, EmbodimentTag.OXE_WIDOWX):
        env_fn = get_simpler_env_fn(env_name)

    elif env_embodiment in (EmbodimentTag.LIBERO_PANDA,):
        env_fn = get_libero_env_fn(env_name)

    elif env_embodiment in (EmbodimentTag.BEHAVIOR_R1_PRO,):
        env_fn = get_behavior_env_fn(env_name, env_idx, total_n_envs)
    else:
        raise ValueError(f"Invalid environment name: {env_name}")

    return env_fn()


def create_eval_env(
    env_name: str, env_idx: int, total_n_envs: int, wrapper_configs: WrapperConfigs
) -> gym.Env:
    """Create a single evaluation environment with wrappers.

    Args:
        env_name: Name of the gymnasium environment to use
        idx: Environment index (used to determine video recording)
        wrapper_configs: Configuration for environment wrappers
    Returns:
        Wrapped gymnasium environment
    """

    env = get_gym_env(env_name, env_idx, total_n_envs)
    if wrapper_configs.video.video_dir is not None:
        from gr00t.eval.sim.wrapper.video_recording_wrapper import (
            VideoRecorder,
            VideoRecordingWrapper,
        )

        base_video_dir = Path(wrapper_configs.video.video_dir)
        per_env_video_dir = _video_dir_for_env(base_video_dir, env_idx, total_n_envs)
        video_recorder = VideoRecorder.create_h264(
            fps=wrapper_configs.video.fps,
            codec=wrapper_configs.video.codec,
            input_pix_fmt=wrapper_configs.video.input_pix_fmt,
            crf=wrapper_configs.video.crf,
            thread_type=wrapper_configs.video.thread_type,
            thread_count=wrapper_configs.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=per_env_video_dir,
            steps_per_render=wrapper_configs.video.steps_per_render,
            max_episode_steps=wrapper_configs.video.max_episode_steps,
            overlay_text=wrapper_configs.video.overlay_text,
            file_name_mode=wrapper_configs.video.file_name_mode,
        )

    env = MultiStepWrapper(
        env,
        video_delta_indices=wrapper_configs.multistep.video_delta_indices,
        state_delta_indices=wrapper_configs.multistep.state_delta_indices,
        n_action_steps=wrapper_configs.multistep.n_action_steps,
        max_episode_steps=wrapper_configs.multistep.max_episode_steps,
        terminate_on_success=wrapper_configs.multistep.terminate_on_success,
    )
    return env


def _video_dir_for_env(video_dir: Path, env_idx: int, total_n_envs: int) -> Path:
    if total_n_envs <= 1:
        return video_dir
    return video_dir / f"env{env_idx:02d}"


def _normalize_task_descriptions(observations: dict[str, Any], batch_size: int) -> list[str | None]:
    raw_task_description = None
    for key in ("annotation.human.coarse_action", "task"):
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
        horizon_config: HorizoEvalConfig,
        n_envs: int,
        n_action_steps: int,
    ):
        self.horizon_config = horizon_config
        self.n_envs = int(n_envs)
        self.n_action_steps = int(n_action_steps)
        self.enabled = bool(
            self.horizon_config.dump_denoise
            or self.horizon_config.eval_type == "horizon"
            or self.horizon_config.replan_log_path
        )
        self.replan_log_path = (
            Path(self.horizon_config.replan_log_path)
            if self.horizon_config.replan_log_path
            else None
        )
        if self.replan_log_path is not None:
            self.replan_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._denoise_loggers = {
            env_idx: DenoiseLogger(enabled=self.horizon_config.dump_denoise)
            for env_idx in range(self.n_envs)
        }
        self._denoise_overhead_seconds = {env_idx: 0.0 for env_idx in range(self.n_envs)}
        self._episode_paths: dict[int, Path | None] = {env_idx: None for env_idx in range(self.n_envs)}
        self._episode_indices = {env_idx: 0 for env_idx in range(self.n_envs)}
        self._replan_indices = {env_idx: 0 for env_idx in range(self.n_envs)}
        self.executed_action_history = [np.zeros((0, 0), dtype=np.float32) for _ in range(self.n_envs)]
        self.horizon_ts_states = [dict() for _ in range(self.n_envs)]
        self.horizon_ts_rngs = [
            np.random.default_rng(int(self.horizon_config.horizon_ts_seed) + env_idx)
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
            if self.horizon_config.horizon_inter_fallback == "neutral":
                effective[invalid_mask] = 0.5
            else:
                effective[invalid_mask] = q_arr[invalid_mask]
        return effective

    def update_run_score_stats(self, horizon_info: dict[str, Any] | None) -> None:
        if self.horizon_config.eval_type != "horizon" or not horizon_info:
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

        if self.horizon_config.horizon_mix_mode == "intra":
            q_intra_contrib = q_mix
        elif self.horizon_config.horizon_mix_mode == "inter":
            if np.any(finite_p):
                p_inter_contrib[finite_p] = p_effective[finite_p]
            if np.any(invalid_p):
                if self.horizon_config.horizon_inter_fallback == "neutral":
                    p_inter_contrib[invalid_p] = 0.5
                else:
                    q_intra_contrib[invalid_p] = q_intra[invalid_p]
        else:
            if np.any(finite_p):
                w = float(self.horizon_config.horizon_inter_weight)
                q_intra_contrib[finite_p] = (1.0 - w) * q_intra[finite_p]
                p_inter_contrib[finite_p] = w * p_effective[finite_p]
            if np.any(invalid_p):
                if self.horizon_config.horizon_inter_fallback == "neutral":
                    w = float(self.horizon_config.horizon_inter_weight)
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
        parts = ["horizon", "ehh1dpad", str(self.horizon_config.horizon_intra_z_mode)]
        if self.horizon_config.horizon_inter_use_speed_uniformity and self.horizon_config.horizon_mix_mode in ("inter", "both"):
            parts.append("speedu")
        parts.append(str(self.horizon_config.horizon_mix_mode))
        parts.append(str(self.horizon_config.horizon_exec_mode))
        if self.horizon_config.horizon_ts_update_mode != "independent":
            parts.append(str(self.horizon_config.horizon_ts_update_mode))
        if self.horizon_config.horizon_exec_mode == "expected_round":
            parts.append(f"temp{float(self.horizon_config.horizon_expected_temp):g}".replace(".", "p"))
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
            "horizon_candidates": str(self.horizon_config.horizon_candidates),
            "horizon_exec_mode": str(self.horizon_config.horizon_exec_mode),
            "horizon_expected_temp": float(self.horizon_config.horizon_expected_temp),
            "horizon_intra_alpha": float(self.horizon_config.horizon_intra_alpha),
            "horizon_intra_z_mode": str(self.horizon_config.horizon_intra_z_mode),
            "horizon_intra_tau_window": int(self.horizon_config.horizon_intra_tau_window),
            "horizon_inter_weight": float(self.horizon_config.horizon_inter_weight),
            "horizon_inter_beta": float(self.horizon_config.horizon_inter_beta),
            "horizon_inter_window": int(self.horizon_config.horizon_inter_window),
            "horizon_ts_update_mode": str(self.horizon_config.horizon_ts_update_mode),
            "horizon_ts_kernel_bandwidth": float(self.horizon_config.horizon_ts_kernel_bandwidth),
            "horizon_ts_forget_rho": float(self.horizon_config.horizon_ts_forget_rho),
            "horizon_ts_update_eta": float(self.horizon_config.horizon_ts_update_eta),
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

    def _horizon_info_to_npz_extras(self, horizon_info: dict[str, Any] | None) -> dict[str, np.ndarray]:
        if not horizon_info:
            return {}

        return {
            "horizon_values": np.asarray(horizon_info.get("horizons", []), dtype=np.int32),
            "horizon_z_ehh_rms_from_tau0": np.asarray(
                horizon_info.get("z_ehh_rms_from_tau0", []),
                dtype=np.float32,
            ),
            "horizon_q_proxy": np.asarray(horizon_info.get("q_proxy", []), dtype=np.float32),
            "horizon_q_mix": np.asarray(horizon_info.get("q_mix", []), dtype=np.float32),
            "horizon_score": np.asarray(horizon_info.get("score", []), dtype=np.float32),
            "horizon_speed_uniformity_cv": self._float_list_or_nan(
                horizon_info.get("speed_uniformity_cv", [])
            ),
            "horizon_speed_uniformity_proxy": self._float_list_or_nan(
                horizon_info.get("speed_uniformity_proxy", [])
            ),
            "horizon_score_prob": np.asarray(horizon_info.get("score_prob", []), dtype=np.float32),
            "horizon_expected_prob": np.asarray(
                horizon_info.get("expected_prob", []),
                dtype=np.float32,
            ),
            "horizon_theta_sample": np.asarray(
                horizon_info.get("theta_sample", []),
                dtype=np.float32,
            ),
            "horizon_posterior_alpha_before": np.asarray(
                horizon_info.get("posterior_alpha_before", []),
                dtype=np.float32,
            ),
            "horizon_posterior_beta_before": np.asarray(
                horizon_info.get("posterior_beta_before", []),
                dtype=np.float32,
            ),
            "horizon_posterior_alpha_after": np.asarray(
                horizon_info.get("posterior_alpha_after", []),
                dtype=np.float32,
            ),
            "horizon_posterior_beta_after": np.asarray(
                horizon_info.get("posterior_beta_after", []),
                dtype=np.float32,
            ),
            "horizon_expected_horizon": np.asarray(
                float(horizon_info.get("expected_horizon", np.nan)),
                dtype=np.float32,
            ),
            "horizon_chosen_horizon": np.asarray(
                int(horizon_info.get("chosen_horizon", 0)),
                dtype=np.int32,
            ),
            "horizon_update_center": np.asarray(
                float(horizon_info.get("update_center", np.nan)),
                dtype=np.float32,
            ),
            "horizon_update_feedback": np.asarray(
                float(horizon_info.get("update_feedback", np.nan)),
                dtype=np.float32,
            ),
            "horizon_ehh_curve_by_horizon": np.asarray(
                horizon_info.get("ehh_curve_by_horizon", []),
                dtype=np.float32,
            ),
            "horizon_selector_mode": np.asarray(
                str(horizon_info.get("selector_mode", "")),
            ),
            "horizon_choose_mode": np.asarray(
                str(horizon_info.get("choose_mode", "")),
            ),
            "horizon_info_json": np.asarray(
                json.dumps(self._json_safe(horizon_info), ensure_ascii=False)
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
        self.horizon_ts_states[env_idx] = {}
        self.horizon_ts_rngs[env_idx] = np.random.default_rng(
            int(self.horizon_config.horizon_ts_seed) + env_idx
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
                    (0, int(action_dim)), dtype=np.float32
                )

    def append_executed_actions(self, flat_actions: np.ndarray, exec_horizon: np.ndarray) -> None:
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
                self.executed_action_history[env_idx] = np.concatenate([history, executed], axis=0)

    def log_replan(
        self,
        *,
        task_descriptions: list[str | None],
        exec_horizon: np.ndarray,
        horizon_infos: list[dict[str, Any] | None] | None,
    ) -> None:
        if self.replan_log_path is None:
            return

        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        records = []
        for env_idx, k_exec in enumerate(np.asarray(exec_horizon).reshape(-1)):
            horizon_info = None
            if horizon_infos is not None and env_idx < len(horizon_infos):
                horizon_info = horizon_infos[env_idx]
            records.append(
                {
                    "timestamp": timestamp,
                    "episode_index": int(self._episode_indices[env_idx]),
                    "replan_index": int(self._replan_indices[env_idx]),
                    "batch_index": int(env_idx),
                    "task_description": self._json_safe(task_descriptions[env_idx]),
                    "eval_type": self.horizon_config.eval_type,
                    "exec_horizon": int(k_exec),
                    "horizon_info": self._json_safe(horizon_info),
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
        horizon_infos: list[dict[str, Any] | None] | None,
    ) -> None:
        if not self.horizon_config.dump_denoise or denoise_trace is None:
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
            if self.horizon_config.eval_type == "horizon" and horizon_infos is not None:
                extras = self._horizon_info_to_npz_extras(horizon_infos[env_idx])

            t0 = time.time()
            self._denoise_loggers[env_idx].log_step(
                np.asarray(trace_v[env_idx, :, :used_h, :], dtype=np.float32),
                ds=ds,
                x0=x0,
                xt=xt_log,
                xt_internal=xt_internal_log,
                k_exec=k_exec if self.horizon_config.eval_type == "horizon" else None,
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


def run_rollout_gymnasium_policy(
    env_name: str,
    policy: BasePolicy,
    wrapper_configs: WrapperConfigs,
    horizon_config: HorizoEvalConfig,
    n_episodes: int = 10,
    n_envs: int = 1,
) -> Any:
    """Run policy rollouts in parallel environments.

    Args:
        env_name: Name of the gymnasium environment to use
        policy: Policy instance
        n_episodes: Number of episodes to run
        n_envs: Number of parallel environments
        wrapper_configs: Configuration for environment wrappers
        ray_env: Whether to use ray gym env to create each env.
    Returns:
        Collection results from running the episodes
    """
    start_time = time.time()
    print(f"Running collecting {n_episodes} episodes for {env_name} with {n_envs} vec envs")

    env_fns = [
        partial(
            create_eval_env,
            env_idx=idx,
            env_name=env_name,
            total_n_envs=n_envs,
            wrapper_configs=wrapper_configs,
        )
        for idx in range(n_envs)
    ]

    if n_envs == 1:
        env = gym.vector.SyncVectorEnv(env_fns)
    else:
        env = gym.vector.AsyncVectorEnv(
            env_fns,
            shared_memory=False,
            context="spawn",
        )

    # Storage for results
    episode_lengths = []
    current_rewards = [0] * n_envs
    current_lengths = [0] * n_envs
    episode_start_times = [0.0] * n_envs
    first_success_steps: list[int | None] = [None] * n_envs
    completed_episodes = 0
    current_successes = [False] * n_envs
    episode_successes = []
    episode_infos = defaultdict(list)
    trace_manager = EvalTraceManager(
        horizon_config=horizon_config,
        n_envs=n_envs,
        n_action_steps=wrapper_configs.multistep.n_action_steps,
    )
    action_modality_keys = policy.get_modality_config()["action"].modality_keys

    # Initial reset
    observations, _ = env.reset()
    policy.reset()
    episode_start_times = [time.time()] * n_envs
    for env_idx in range(n_envs):
        env_video_dir = None
        if wrapper_configs.video.video_dir is not None:
            env_video_dir = str(
                _video_dir_for_env(
                    Path(wrapper_configs.video.video_dir),
                    env_idx,
                    n_envs,
                )
            )
        trace_manager.begin_episode(env_idx=env_idx, video_dir=env_video_dir)

    pbar = tqdm(total=n_episodes, desc="Episodes")
    stop_collection = False
    while completed_episodes < n_episodes:
        batch_size = int(
            len(
                next(
                    value
                    for key, value in observations.items()
                    if key.startswith("video.") or key.startswith("state.")
                )
            )
        )
        task_descriptions = _normalize_task_descriptions(observations, batch_size)
        policy_options = {
            "return_trace": bool(
                horizon_config.eval_type == "horizon" or horizon_config.dump_denoise
            )
        }
        actions, policy_info = policy.get_action(observations, policy_options)
        exec_horizon = np.full(
            (batch_size,), wrapper_configs.multistep.n_action_steps, dtype=np.int32
        )
        horizon_infos: list[dict[str, Any] | None] | None = None
        flat_actions = None
        if (
            horizon_config.eval_type == "horizon"
            or horizon_config.dump_denoise
            or trace_manager.replan_log_path is not None
        ):
            flat_actions = _flatten_action_chunks(actions, action_modality_keys)
            trace_manager.ensure_action_history(flat_actions.shape[-1])
            if horizon_config.eval_type == "horizon":
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
                horizon_infos = []
                used_h = min(int(wrapper_configs.multistep.n_action_steps), int(flat_actions.shape[1]))
                for env_idx in range(batch_size):
                    k_exec, horizon_info = select_exec_k_horizon(
                        v_trace[env_idx, :, :used_h, :],
                        used_h,
                        horizon_candidates=horizon_config.horizon_candidates,
                        exec_mode=horizon_config.horizon_exec_mode,
                        expected_temp=horizon_config.horizon_expected_temp,
                        horizon_alpha=horizon_config.horizon_ts_alpha,
                        intra_cut_t=horizon_config.horizon_intra_cut_t,
                        ts_epsilon=horizon_config.horizon_ts_epsilon,
                        ts_update_mode=horizon_config.horizon_ts_update_mode,
                        ts_kernel_bandwidth=horizon_config.horizon_ts_kernel_bandwidth,
                        ts_forget_rho=horizon_config.horizon_ts_forget_rho,
                        ts_update_eta=horizon_config.horizon_ts_update_eta,
                        z_mode=horizon_config.horizon_z_mode,
                        tau_window=horizon_config.horizon_tau_window,
                        xt_step=np.asarray(flat_actions[env_idx, :used_h, :], dtype=np.float32),
                        history_actions=trace_manager.executed_action_history[env_idx],
                        signal_mode=horizon_config.horizon_signal_mode,
                        use_speed_uniformity=horizon_config.horizon_use_speed_uniformity,
                        speed_window=horizon_config.horizon_speed_window,
                        speed_weight=horizon_config.horizon_speed_weight,
                        speed_beta=horizon_config.horizon_speed_beta,
                        speed_fallback=horizon_config.horizon_speed_fallback,
                        ts_state=trace_manager.horizon_ts_states[env_idx],
                        ts_rng=trace_manager.horizon_ts_rngs[env_idx],
                    )
                    exec_horizon[env_idx] = int(k_exec)
                    horizon_infos.append(horizon_info)
                    trace_manager.update_run_score_stats(horizon_info)
                trace_manager.append_executed_actions(flat_actions, exec_horizon)
            actions["exec_horizon"] = exec_horizon[:, None]
            trace_manager.log_replan(
                task_descriptions=task_descriptions,
                exec_horizon=exec_horizon,
                horizon_infos=horizon_infos,
            )
            trace_manager.log_denoise_step(
                denoise_trace=policy_info.get("denoise_trace"),
                flat_actions=flat_actions,
                exec_horizon=exec_horizon,
                horizon_infos=horizon_infos,
            )
            trace_manager.advance_replan_indices()

        next_obs, rewards, terminations, truncations, env_infos = env.step(actions)
        # Update episode tracking
        for env_idx in range(n_envs):
            if "success" in env_infos:
                env_success = env_infos["success"][env_idx]
                if isinstance(env_success, list):
                    env_success = np.any(env_success)
                elif isinstance(env_success, np.ndarray):
                    env_success = np.any(env_success)
                elif isinstance(env_success, bool):
                    env_success = env_success
                elif isinstance(env_success, int):
                    env_success = bool(env_success)
                else:
                    raise ValueError(f"Unknown success dtype: {type(env_success)}")
                current_successes[env_idx] |= bool(env_success)
            else:
                current_successes[env_idx] = False
            if current_successes[env_idx] and first_success_steps[env_idx] is None:
                first_success_steps[env_idx] = current_lengths[env_idx] + 1

            if "final_info" in env_infos and env_infos["final_info"][env_idx] is not None:
                env_success = env_infos["final_info"][env_idx]["success"]
                if isinstance(env_success, list):
                    env_success = any(env_success)
                elif isinstance(env_success, np.ndarray):
                    env_success = np.any(env_success)
                elif isinstance(env_success, bool):
                    env_success = env_success
                elif isinstance(env_success, int):
                    env_success = bool(env_success)
                else:
                    raise ValueError(f"Unknown success dtype: {type(env_success)}")
                current_successes[env_idx] |= bool(env_success)
            current_rewards[env_idx] += rewards[env_idx]
            current_lengths[env_idx] += 1

            # If episode ended, store results
            if terminations[env_idx] or truncations[env_idx]:
                if "final_info" in env_infos:
                    current_successes[env_idx] |= any(env_infos["final_info"][env_idx]["success"])
                if "task_progress" in env_infos:
                    episode_infos["task_progress"].append(env_infos["task_progress"][env_idx][-1])
                if "q_score" in env_infos:
                    episode_infos["q_score"].append(np.max(env_infos["q_score"][env_idx]))
                episode_valid = True
                if "valid" in env_infos:
                    episode_valid = all(env_infos["valid"][env_idx])
                    episode_infos["valid"].append(episode_valid)
                # Accumulate results
                episode_lengths.append(current_lengths[env_idx])
                episode_successes.append(current_successes[env_idx])
                trace_manager.finalize_episode(
                    env_idx=env_idx,
                    success=bool(current_successes[env_idx]),
                    success_step=first_success_steps[env_idx],
                    episode_steps=current_lengths[env_idx],
                    run_time_seconds=float(time.time() - episode_start_times[env_idx]),
                )
                # Reset trackers for this environment.
                current_successes[env_idx] = False
                first_success_steps[env_idx] = None
                if episode_valid:
                    completed_episodes += 1
                    pbar.update(1)
                current_rewards[env_idx] = 0
                current_lengths[env_idx] = 0
                episode_start_times[env_idx] = time.time()
                if completed_episodes >= n_episodes:
                    stop_collection = True
                    break
                env_video_dir = None
                if wrapper_configs.video.video_dir is not None:
                    env_video_dir = str(
                        _video_dir_for_env(
                            Path(wrapper_configs.video.video_dir),
                            env_idx,
                            n_envs,
                        )
                    )
                trace_manager.begin_episode(env_idx=env_idx, video_dir=env_video_dir)
        if stop_collection:
            break
        observations = next_obs
    pbar.close()

    env.reset()
    env.close()
    print(f"Collecting {n_episodes} episodes took {time.time() - start_time} seconds")

    episode_infos = dict(episode_infos)  # Convert defaultdict to dict
    for key, value in episode_infos.items():
        assert len(value) == len(episode_successes), (
            f"Length of {key} is not equal to the number of episodes"
        )

    # process valid results
    if "valid" in episode_infos:
        valids = episode_infos["valid"]
        valid_idxs = np.where(valids)[0]
        episode_successes = [episode_successes[i] for i in valid_idxs]
        episode_infos = {k: [v[i] for i in valid_idxs] for k, v in episode_infos.items()}

    if horizon_config.run_summary_path:
        summary_path = Path(horizon_config.run_summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = trace_manager.get_horizon_run_summary()
        summary["env_name"] = env_name
        summary["n_episodes"] = int(n_episodes)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    assert len(episode_successes) == n_episodes, (
        f"Expected exactly {n_episodes} episodes after filtering, got {len(episode_successes)}"
    )
    for key, value in episode_infos.items():
        assert len(value) == n_episodes, (
            f"Expected exactly {n_episodes} entries for {key}, got {len(value)}"
        )

    return env_name, episode_successes, episode_infos


def create_gr00t_sim_policy(
    model_path: str,
    embodiment_tag: EmbodimentTag,
    policy_client_host: str = "",
    policy_client_port: int | None = None,
) -> BasePolicy:
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    if policy_client_host and policy_client_port:
        from gr00t.policy.server_client import PolicyClient

        policy = PolicyClient(host=policy_client_host, port=policy_client_port)
    else:
        policy = Gr00tSimPolicyWrapper(
            Gr00tPolicy(
                embodiment_tag=embodiment_tag,
                model_path=model_path,
                device=0,
            )
        )
    return policy


def run_gr00t_sim_policy(
    env_name: str,
    n_episodes: int,
    max_episode_steps: int,
    model_path: str = "",
    policy_client_host: str = "",
    policy_client_port: int | None = None,
    n_envs: int = 8,
    n_action_steps: int = 8,
    video_dir: str = "",
    horizon_config: HorizoEvalConfig | None = None,
):
    if horizon_config is None:
        horizon_config = HorizoEvalConfig()
    embodiment_tag = get_embodiment_tag_from_env_name(env_name)

    if video_dir:
        resolved_video_dir = video_dir
    elif model_path:
        resolved_video_dir = (
            f"/tmp/sim_eval_videos_{model_path.split('/')[-3]}_ac{n_action_steps}_{uuid.uuid4()}"
        )
    else:
        resolved_video_dir = f"/tmp/sim_eval_videos_{env_name}_ac{n_action_steps}_{uuid.uuid4()}"
    wrapper_configs = WrapperConfigs(
        video=VideoConfig(
            video_dir=resolved_video_dir,
            max_episode_steps=max_episode_steps,
            file_name_mode="episode_success",
        ),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
            terminate_on_success=True,
        ),
    )

    policy = create_gr00t_sim_policy(
        model_path, embodiment_tag, policy_client_host, policy_client_port
    )

    results = run_rollout_gymnasium_policy(
        env_name=env_name,
        policy=policy,
        wrapper_configs=wrapper_configs,
        horizon_config=horizon_config,
        n_episodes=n_episodes,
        n_envs=n_envs,
    )
    print("Video saved to: ", wrapper_configs.video.video_dir)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_episode_steps", type=int, default=504)
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
    )
    parser.add_argument("--policy_client_host", type=str, default="")
    parser.add_argument("--policy_client_port", type=int, default=None)
    parser.add_argument(
        "--env_name",
        type=str,
        default="gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
    )
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument("--video_dir", type=str, default="")
    parser.add_argument("--eval_type", type=str, default="default")
    parser.add_argument("--dump_denoise", type=int, default=0)
    parser.add_argument("--horizon_candidates", type=str, default="4,8,12,16")
    parser.add_argument("--horizon_exec_mode", type=str, default="expected_round")
    parser.add_argument("--horizon_expected_temp", type=float, default=1.0)
    parser.add_argument("--horizon_intra_alpha", type=float, default=4.0)
    parser.add_argument("--horizon_intra_cut_t", type=float, default=0.25)
    parser.add_argument("--horizon_ts_epsilon", type=float, default=0.05)
    parser.add_argument("--horizon_ts_seed", type=int, default=0)
    parser.add_argument("--horizon_ts_update_mode", type=str, default="kernel_forget")
    parser.add_argument("--horizon_ts_kernel_bandwidth", type=float, default=10.0)
    parser.add_argument("--horizon_ts_forget_rho", type=float, default=0.99)
    parser.add_argument("--horizon_ts_update_eta", type=float, default=1.0)
    parser.add_argument("--horizon_intra_z_mode", type=str, default="window_rms")
    parser.add_argument("--horizon_intra_tau_window", type=int, default=1)
    parser.add_argument("--horizon_mix_mode", type=str, default="both")
    parser.add_argument("--horizon_inter_use_speed_uniformity", type=int, default=1)
    parser.add_argument("--horizon_inter_window", type=int, default=20)
    parser.add_argument("--horizon_inter_weight", type=float, default=0.5)
    parser.add_argument("--horizon_inter_beta", type=float, default=4.0)
    parser.add_argument("--horizon_inter_fallback", type=str, default="ehh_only")
    parser.add_argument("--replan_log_path", type=str, default="")
    parser.add_argument("--run_summary_path", type=str, default="")

    args = parser.parse_args()

    # validate policy configuration
    assert (args.model_path and not (args.policy_client_host or args.policy_client_port)) or (
        not args.model_path and args.policy_client_host and args.policy_client_port is not None
    ), (
        "Invalid policy configuration: You must provide EITHER model_path OR (policy_client_host & policy_client_port), not both.\n"
        "If all 3 arguments are provided, explicitly choose one:\n"
        '  - To use policy client: set --policy_client_host and --policy_client_port, and set --model_path ""\n'
        '  - To use model path: set --model_path, and set --policy_client_host "" (and leave --policy_client_port unset)'
    )

    results = run_gr00t_sim_policy(
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        max_episode_steps=args.max_episode_steps,
        model_path=args.model_path,
        policy_client_host=args.policy_client_host,
        policy_client_port=args.policy_client_port,
        n_envs=args.n_envs,
        n_action_steps=args.n_action_steps,
        video_dir=args.video_dir,
        horizon_config=HorizoEvalConfig(
            eval_type=str(args.eval_type).strip().lower(),
            dump_denoise=bool(args.dump_denoise),
            horizon_candidates=args.horizon_candidates,
            horizon_exec_mode=str(args.horizon_exec_mode).strip().lower(),
            horizon_expected_temp=args.horizon_expected_temp,
            horizon_intra_alpha=args.horizon_intra_alpha,
            horizon_intra_cut_t=args.horizon_intra_cut_t,
            horizon_ts_epsilon=args.horizon_ts_epsilon,
            horizon_ts_seed=args.horizon_ts_seed,
            horizon_ts_update_mode=str(args.horizon_ts_update_mode).strip().lower(),
            horizon_ts_kernel_bandwidth=args.horizon_ts_kernel_bandwidth,
            horizon_ts_forget_rho=args.horizon_ts_forget_rho,
            horizon_ts_update_eta=args.horizon_ts_update_eta,
            horizon_intra_z_mode=str(args.horizon_intra_z_mode).strip().lower(),
            horizon_intra_tau_window=args.horizon_intra_tau_window,
            horizon_mix_mode=str(args.horizon_mix_mode).strip().lower(),
            horizon_inter_use_speed_uniformity=bool(args.horizon_inter_use_speed_uniformity),
            horizon_inter_window=args.horizon_inter_window,
            horizon_inter_weight=args.horizon_inter_weight,
            horizon_inter_beta=args.horizon_inter_beta,
            horizon_inter_fallback=str(args.horizon_inter_fallback).strip().lower(),
            replan_log_path=args.replan_log_path,
            run_summary_path=args.run_summary_path,
        ),
    )
    print("results: ", results)
    print("success rate: ", np.mean(results[1]))
