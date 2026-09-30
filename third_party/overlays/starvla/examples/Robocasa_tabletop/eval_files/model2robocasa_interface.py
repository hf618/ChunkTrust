from collections import deque
import json
from datetime import datetime
import os
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

from examples.Robocasa_tabletop.eval_files.adaptive_ensemble import AdaptiveEnsembler
from examples.Robocasa_tabletop.eval_files.denoise_logger import DenoiseLogger
from examples.Robocasa_tabletop.eval_files.horizon_selector import (
    parse_horizon_candidates,
    select_exec_k_horizon,
)

from starVLA.model.framework.share_tools import read_mode_config



class PolicyWarper:
    @staticmethod
    def _normalize_eval_type(eval_type: str) -> str:
        normalized = str(eval_type or "default").strip().lower()
        if normalized == "stride":
            raise ValueError("eval_type='stride' has been removed; use eval_type='horizon'.")
        if normalized not in {"default", "horizon"}:
            raise ValueError(f"Unsupported eval_type={eval_type!r}; expected one of default/horizon.")
        return normalized

    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble = False, # @Jinhui
        action_ensemble_horizon: Optional[int] = 3, # different cross sim
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha = 0.1,
        host="0.0.0.0",
        port=10095,
        n_action_steps=2,
        eval_type: str = "default",
        dump_denoise: bool = False,
        horizon_candidates: str = "4,8,12,16",
        horizon_exec_mode: str = "expected_round",
        horizon_expected_temp: float = 1.0,
        horizon_intra_alpha: float = 4.0,
        horizon_intra_cut_t: float = 0.25,
        horizon_ts_epsilon: float = 0.05,
        horizon_ts_seed: int = 0,
        horizon_ts_update_mode: str = "kernel_forget",
        horizon_ts_kernel_bandwidth: float = 10.0,
        horizon_ts_forget_rho: float = 0.99,
        horizon_ts_update_eta: float = 1.0,
        horizon_intra_z_mode: str = "window_rms",
        horizon_intra_tau_window: int = 1,
        horizon_mix_mode: str = "both",
        horizon_inter_use_speed_uniformity: bool = True,
        horizon_inter_window: int = 20,
        horizon_inter_weight: float = 0.5,
        horizon_inter_beta: float = 4.0,
        horizon_inter_fallback: str = "ehh_only",
        replan_log_path: Optional[str] = None,
    ) -> None:
        self.eval_type = self._normalize_eval_type(eval_type)

        # build client to connect server policy
        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon #0
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.n_action_steps = n_action_steps
        self.dump_denoise = bool(dump_denoise)
        self.horizon_candidates = parse_horizon_candidates(horizon_candidates, default_max=n_action_steps)
        self.horizon_exec_mode = str(horizon_exec_mode).strip().lower()
        self.horizon_expected_temp = float(horizon_expected_temp)
        self.horizon_intra_alpha = float(horizon_intra_alpha)
        self.horizon_intra_cut_t = float(horizon_intra_cut_t)
        self.horizon_ts_epsilon = float(horizon_ts_epsilon)
        self.horizon_ts_seed = int(horizon_ts_seed)
        self.horizon_ts_update_mode = str(horizon_ts_update_mode).strip().lower()
        self.horizon_ts_kernel_bandwidth = float(horizon_ts_kernel_bandwidth)
        self.horizon_ts_forget_rho = float(horizon_ts_forget_rho)
        self.horizon_ts_update_eta = float(horizon_ts_update_eta)
        self.horizon_intra_z_mode = str(horizon_intra_z_mode).strip().lower()
        self.horizon_intra_tau_window = int(horizon_intra_tau_window)
        self.horizon_mix_mode = str(horizon_mix_mode).strip().lower()
        if self.horizon_mix_mode not in ("intra", "inter", "both"):
            raise ValueError(
                f"Unsupported horizon_mix_mode={self.horizon_mix_mode!r}; expected one of intra/inter/both."
            )
        self.horizon_inter_use_speed_uniformity = bool(horizon_inter_use_speed_uniformity)
        self.horizon_inter_window = int(horizon_inter_window)
        self.horizon_inter_weight = float(horizon_inter_weight)
        self.horizon_inter_beta = float(horizon_inter_beta)
        self.horizon_inter_fallback = str(horizon_inter_fallback).strip().lower()
        if self.horizon_inter_fallback not in ("ehh_only", "neutral"):
            raise ValueError(
                f"Unsupported horizon_inter_fallback={self.horizon_inter_fallback!r}; expected ehh_only/neutral."
            )
        self.last_denoise_trace = None
        self.last_horizon_info = None
        self.replan_log_path = Path(replan_log_path) if replan_log_path else None
        if self.replan_log_path is not None:
            self.replan_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._denoise_loggers: dict[int, DenoiseLogger] = {}
        self._episode_npz_paths: dict[int, Path] = {}
        self._denoise_overhead_seconds: dict[int, float] = {}
        self._env_episode_indices: dict[int, int] = {}
        self._horizon_ts_states: dict[int, dict[int, list[float]]] = {}
        self._horizon_ts_rngs: dict[int, np.random.Generator] = {}
        self._replan_stdout = str(os.getenv("STARVLA_REPLAN_STDOUT", "")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.episode_index = 0
        self.replan_index = 0
        self.executed_action_history = []
        self._reset_run_score_state()

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str or tuple) -> None:
       
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def _get_denoise_logger(self, env_idx: int) -> DenoiseLogger:
        env_idx = int(env_idx)
        logger = self._denoise_loggers.get(env_idx)
        if logger is None:
            logger = DenoiseLogger(enabled=self.dump_denoise)
            self._denoise_loggers[env_idx] = logger
        return logger

    def begin_episode(self, video_dir: Optional[str] = None, env_idx: int = 0) -> None:
        env_idx = int(env_idx)
        self._env_episode_indices[env_idx] = int(self._env_episode_indices.get(env_idx, 0)) + 1
        self.episode_index = max(self._env_episode_indices.values(), default=0)
        self.replan_index = 0
        self.last_denoise_trace = None
        self.last_horizon_info = None
        self._episode_npz_paths.pop(env_idx, None)
        self._denoise_overhead_seconds[env_idx] = 0.0
        self._horizon_ts_states[env_idx] = {}
        self._horizon_ts_rngs[env_idx] = np.random.default_rng(self.horizon_ts_seed + env_idx)
        if env_idx < len(self.executed_action_history):
            history = self.executed_action_history[env_idx]
            action_dim = int(history.shape[1]) if getattr(history, "ndim", 0) == 2 else 0
            self.executed_action_history[env_idx] = np.zeros((0, action_dim), dtype=np.float32)
        self._get_denoise_logger(env_idx).reset_episode()
        self.prepare_episode_logging(video_dir=video_dir, env_idx=env_idx)

    @staticmethod
    def _json_safe(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return [PolicyWarper._json_safe(v) for v in value]
        if isinstance(value, list):
            return [PolicyWarper._json_safe(v) for v in value]
        if isinstance(value, dict):
            return {str(k): PolicyWarper._json_safe(v) for k, v in value.items()}
        return value

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
            if self.horizon_inter_fallback == "neutral":
                effective[invalid_mask] = 0.5
            else:
                effective[invalid_mask] = q_arr[invalid_mask]
        return effective

    def _update_run_score_stats(self, horizon_info: Optional[dict]) -> None:
        if self.eval_type != "horizon" or not horizon_info:
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

        if self.horizon_mix_mode == "intra":
            q_intra_contrib = q_mix
        elif self.horizon_mix_mode == "inter":
            if np.any(finite_p):
                p_inter_contrib[finite_p] = p_effective[finite_p]
            if np.any(invalid_p):
                if self.horizon_inter_fallback == "neutral":
                    p_inter_contrib[invalid_p] = 0.5
                else:
                    q_intra_contrib[invalid_p] = q_intra[invalid_p]
        else:
            if np.any(finite_p):
                q_intra_contrib[finite_p] = (1.0 - self.horizon_inter_weight) * q_intra[finite_p]
                p_inter_contrib[finite_p] = self.horizon_inter_weight * p_effective[finite_p]
            if np.any(invalid_p):
                if self.horizon_inter_fallback == "neutral":
                    q_intra_contrib[invalid_p] = (1.0 - self.horizon_inter_weight) * q_intra[invalid_p]
                    p_inter_contrib[invalid_p] = self.horizon_inter_weight * 0.5
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
        parts = ["horizon", "ehh1dpad", str(self.horizon_intra_z_mode)]
        if self.horizon_inter_use_speed_uniformity and self.horizon_mix_mode in ("inter", "both"):
            parts.append("speedu")
        parts.append(self.horizon_mix_mode)
        parts.append(self.horizon_exec_mode)
        if self.horizon_ts_update_mode != "independent":
            parts.append(self.horizon_ts_update_mode)
        if self.horizon_exec_mode == "expected_round":
            parts.append(f"temp{self.horizon_expected_temp:g}".replace(".", "p"))
        return "_".join(parts)

    def get_run_score_averages(self) -> Dict[str, float | None]:
        out: Dict[str, float | None] = {}
        for key, bucket in self._run_score_state.items():
            count = int(bucket["count"])
            out[key] = None if count <= 0 else float(bucket["sum"] / float(count))
        return out

    def get_horizon_run_summary(self) -> Dict[str, object]:
        summary: Dict[str, object] = {
            "horizon_policy": self._format_horizon_policy_name(),
            "horizon_candidates": ",".join(str(int(v)) for v in self.horizon_candidates),
            "horizon_exec_mode": self.horizon_exec_mode,
            "horizon_expected_temp": float(self.horizon_expected_temp),
            "horizon_intra_alpha": float(self.horizon_intra_alpha),
            "horizon_intra_cut_t": float(self.horizon_intra_cut_t),
            "horizon_intra_z_mode": str(self.horizon_intra_z_mode),
            "horizon_intra_tau_window": int(self.horizon_intra_tau_window),
            "horizon_inter_use_speed_uniformity": bool(self.horizon_inter_use_speed_uniformity),
            "horizon_inter_weight": float(self.horizon_inter_weight),
            "horizon_inter_beta": float(self.horizon_inter_beta),
            "horizon_inter_window": int(self.horizon_inter_window),
            "horizon_inter_fallback": str(self.horizon_inter_fallback),
            "horizon_ts_update_mode": str(self.horizon_ts_update_mode),
            "horizon_ts_kernel_bandwidth": float(self.horizon_ts_kernel_bandwidth),
            "horizon_ts_forget_rho": float(self.horizon_ts_forget_rho),
            "horizon_ts_update_eta": float(self.horizon_ts_update_eta),
        }
        summary.update(self.get_run_score_averages())
        return summary

    def _log_replan(self, exec_horizon: np.ndarray) -> None:
        if self.replan_log_path is None:
            return

        horizon_info_batch = self.last_horizon_info
        timestamp = datetime.now().isoformat(timespec="seconds")
        records = []
        for batch_idx, k_exec in enumerate(np.asarray(exec_horizon).reshape(-1)):
            horizon_info = None
            if horizon_info_batch is not None and batch_idx < len(horizon_info_batch):
                horizon_info = horizon_info_batch[batch_idx]
            record = {
                "timestamp": timestamp,
                "episode_index": int(self._env_episode_indices.get(batch_idx, self.episode_index)),
                "replan_index": int(self.replan_index),
                "batch_index": int(batch_idx),
                "task_description": self._json_safe(self.task_description),
                "eval_type": self.eval_type,
                "exec_horizon": int(k_exec),
                "horizon_info": self._json_safe(horizon_info),
            }
            records.append(record)

        with self.replan_log_path.open("a", encoding="utf-8") as fp:
            for record in records:
                line = json.dumps(record, ensure_ascii=False)
                fp.write(line + "\n")
                if self._replan_stdout:
                    horizon_info = record.get("horizon_info") or {}
                    expected_horizon = horizon_info.get("expected_horizon")
                    summary = (
                        "REPLAN_LOG "
                        f"episode={record['episode_index']} "
                        f"replan={record['replan_index']} "
                        f"batch={record['batch_index']} "
                        f"k={record['exec_horizon']}"
                    )
                    if expected_horizon is not None:
                        summary += f" expected={float(expected_horizon):.2f}"
                    print(summary, flush=True)

    def _resolve_episode_log_dir(self, video_dir: Optional[str] = None, env_idx: int = 0) -> Optional[Path]:
        if video_dir:
            video_dir_path = Path(video_dir)
            if video_dir_path.parent.name.lower() == "videos":
                return video_dir_path.parent.parent / "episodes" / video_dir_path.name
            if video_dir_path.name.lower() == "videos":
                return video_dir_path.parent / "episodes"
        if self.replan_log_path is not None:
            return self.replan_log_path.parent / "episodes"
        return None

    def prepare_episode_logging(self, video_dir: Optional[str] = None, env_idx: int = 0) -> Optional[Path]:
        episode_dir = self._resolve_episode_log_dir(video_dir=video_dir, env_idx=env_idx)
        if episode_dir is None:
            self._episode_npz_paths.pop(int(env_idx), None)
            return None

        episode_dir.mkdir(parents=True, exist_ok=True)
        env_idx = int(env_idx)
        episode_idx = int(self._env_episode_indices.get(env_idx, self.episode_index))
        episode_path = episode_dir / f"episode{episode_idx}.npz"
        self._episode_npz_paths[env_idx] = episode_path
        if self.dump_denoise:
            self._get_denoise_logger(env_idx).start_episode(episode_path)
        return episode_path

    def finalize_episode_logging(
        self,
        *,
        env_idx: int = 0,
        success: Optional[bool] = None,
        success_step: Optional[int] = None,
        episode_steps: Optional[int] = None,
        run_time_seconds: Optional[float] = None,
    ) -> Optional[Path]:
        env_idx = int(env_idx)
        episode_npz_path = self._episode_npz_paths.get(env_idx)
        if episode_npz_path is None:
            return None

        if not self.dump_denoise:
            out = {
                "success": np.asarray(bool(success), dtype=np.bool_) if success is not None else np.asarray(False, dtype=np.bool_),
                "run_time_seconds": np.asarray(
                    [float(run_time_seconds or 0.0), 0.0],
                    dtype=np.float32,
                ),
            }
            if success_step is not None:
                out["success_step"] = np.asarray(int(success_step), dtype=np.int32)
            if episode_steps is not None:
                out["episode_steps"] = np.asarray(int(episode_steps), dtype=np.int32)
            np.savez(episode_npz_path, **out)
            return episode_npz_path

        self._get_denoise_logger(env_idx).set_episode_info(
            success=success,
            success_step=success_step,
            episode_steps=episode_steps,
            run_time_seconds=run_time_seconds,
            denoise_overhead_seconds=self._denoise_overhead_seconds.get(env_idx, 0.0),
        )
        return self._get_denoise_logger(env_idx).flush_episode()

    @staticmethod
    def _float_list_or_nan(values) -> np.ndarray:
        return np.asarray(
            [np.nan if value is None else float(value) for value in values],
            dtype=np.float32,
        )

    def _horizon_info_to_npz_extras(self, horizon_info: Optional[dict]) -> dict[str, np.ndarray]:
        if not horizon_info:
            return {}

        return {
            "horizon_values": np.asarray(horizon_info.get("horizons", []), dtype=np.int32),
            "horizon_z_ehh_rms_from_tau0": np.asarray(
                horizon_info.get("z_ehh_rms_from_tau0", []),
                dtype=np.float32,
            ),
            "horizon_q_intra": np.asarray(horizon_info.get("q_intra", []), dtype=np.float32),
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

    def _maybe_log_denoise_step(self, raw_actions: np.ndarray, exec_horizon: np.ndarray) -> None:
        if not self.dump_denoise:
            return
        if self.last_denoise_trace is None:
            return

        used_h = min(int(self.n_action_steps), int(raw_actions.shape[1]))
        trace = self.last_denoise_trace
        for batch_idx in range(raw_actions.shape[0]):
            if int(batch_idx) not in self._episode_npz_paths:
                continue

            k_exec = int(np.asarray(exec_horizon).reshape(-1)[batch_idx])

            xt_log = np.array(raw_actions[batch_idx, :used_h, :], dtype=np.float32, copy=True)
            if k_exec < xt_log.shape[0]:
                xt_log[k_exec:, :] = 0.0

            xt_internal_log = None
            xt_internal = trace.get("xt_internal")
            if xt_internal is not None:
                xt_internal_log = np.array(
                    xt_internal[batch_idx, :used_h, :],
                    dtype=np.float32,
                    copy=True,
                )
                if k_exec < xt_internal_log.shape[0]:
                    xt_internal_log[k_exec:, :] = 0.0

            x0 = None
            if trace.get("x0") is not None:
                x0 = np.asarray(trace["x0"][batch_idx, :used_h, :], dtype=np.float32)

            ds = None
            if trace.get("ds") is not None:
                ds = np.asarray(trace["ds"][batch_idx], dtype=np.float32)

            extras = {}
            if self.eval_type == "horizon" and self.last_horizon_info:
                extras = self._horizon_info_to_npz_extras(self.last_horizon_info[batch_idx])

            t0 = time.time()
            self._get_denoise_logger(batch_idx).log_step(
                np.asarray(trace["v"][batch_idx, :, :used_h, :], dtype=np.float32),
                ds=ds,
                x0=x0,
                xt=xt_log,
                xt_internal=xt_internal_log,
                k_exec=k_exec if self.eval_type == "horizon" else None,
                extras=extras,
            )
            self._denoise_overhead_seconds[batch_idx] = (
                self._denoise_overhead_seconds.get(batch_idx, 0.0) + float(time.time() - t0)
            )

    def _select_exec_horizons(self, raw_actions: np.ndarray) -> np.ndarray:
        exec_horizon = np.full((raw_actions.shape[0],), self.n_action_steps, dtype=np.int32)
        if len(self.executed_action_history) != raw_actions.shape[0]:
            self.executed_action_history = [
                np.zeros((0, raw_actions.shape[-1]), dtype=np.float32)
                for _ in range(raw_actions.shape[0])
            ]
        for batch_idx in range(raw_actions.shape[0]):
            self._horizon_ts_states.setdefault(batch_idx, {})
            self._horizon_ts_rngs.setdefault(
                batch_idx,
                np.random.default_rng(self.horizon_ts_seed + batch_idx),
            )

        if self.eval_type != "horizon":
            self.last_horizon_info = None
            return exec_horizon

        if self.last_denoise_trace is None:
            raise RuntimeError("eval_type='horizon' requires denoise_trace from the policy server.")
        v_trace = np.asarray(self.last_denoise_trace["v"], dtype=np.float32)
        if v_trace.ndim != 4:
            raise RuntimeError(f"Unexpected denoise_trace['v'] shape: {v_trace.shape}")

        used_h = min(int(self.n_action_steps), int(raw_actions.shape[1]))
        horizon_infos = []
        for batch_idx in range(raw_actions.shape[0]):
            k_exec, horizon_info = select_exec_k_horizon(
                v_trace[batch_idx, :, :used_h, :],
                used_h,
                horizon_candidates=self.horizon_candidates,
                exec_mode=self.horizon_exec_mode,
                expected_temp=self.horizon_expected_temp,
                intra_alpha=self.horizon_intra_alpha,
                intra_cut_t=self.horizon_intra_cut_t,
                ts_epsilon=self.horizon_ts_epsilon,
                ts_update_mode=self.horizon_ts_update_mode,
                ts_kernel_bandwidth=self.horizon_ts_kernel_bandwidth,
                ts_forget_rho=self.horizon_ts_forget_rho,
                ts_update_eta=self.horizon_ts_update_eta,
                z_mode=self.horizon_intra_z_mode,
                tau_window=self.horizon_intra_tau_window,
                xt_step=np.asarray(raw_actions[batch_idx, :used_h, :], dtype=np.float32),
                history_actions=self.executed_action_history[batch_idx],
                mix_mode=self.horizon_mix_mode,
                use_speed_uniformity=self.horizon_inter_use_speed_uniformity,
                speed_window=self.horizon_inter_window,
                speed_weight=self.horizon_inter_weight,
                speed_beta=self.horizon_inter_beta,
                speed_fallback=self.horizon_inter_fallback,
                ts_state=self._horizon_ts_states[batch_idx],
                ts_rng=self._horizon_ts_rngs[batch_idx],
            )
            exec_horizon[batch_idx] = int(k_exec)
            horizon_infos.append(horizon_info)
            self._update_run_score_stats(horizon_info)
        self.last_horizon_info = horizon_infos

        for batch_idx, k_exec in enumerate(exec_horizon.tolist()):
            executed = np.asarray(raw_actions[batch_idx, : int(k_exec), :], dtype=np.float32)
            if executed.size <= 0:
                continue
            history = self.executed_action_history[batch_idx]
            if history.size == 0:
                self.executed_action_history[batch_idx] = executed.copy()
            elif history.shape[1] != executed.shape[1]:
                self.executed_action_history[batch_idx] = executed.copy()
            else:
                self.executed_action_history[batch_idx] = np.concatenate(
                    [history, executed], axis=0
                )
        return exec_horizon

    def step(
        self, 
        observations,
        **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        执行一步推理
        :param image: 输入图像 (H, W, 3) uint8格式
        :param task_description: 任务描述文本
        :return: (原始动作, 处理后的动作)
        """

        ego_view = observations['video.ego_view']  # (N, 1, H, W, 3)
        images = ego_view
        state = {}
        state['left_arm'] = observations['state.left_arm']     
        state['right_arm'] = observations['state.right_arm']             # (N, 1, 7)
        state['left_hand'] = observations['state.left_hand']              # (N, 1, 6)
        state['right_hand'] = observations['state.right_hand']            # (N, 1, 6)
        state['waist'] = observations['state.waist']                      # (N, 1, 3)
        

        state = self.normalize_state(state)
        input_state = []
        for key in state.keys():
            input_state.append(state[key])
        input_state = np.concatenate(input_state, axis=-1)

        batch_size = int(len(images))
        raw_task_description = observations['annotation.human.coarse_action']
        if isinstance(raw_task_description, np.ndarray):
            raw_task_description = raw_task_description.tolist()

        if isinstance(raw_task_description, (list, tuple)):
            task_descriptions = list(raw_task_description)
        else:
            task_descriptions = [raw_task_description]

        normalized_task_descriptions = []
        for item in task_descriptions:
            if isinstance(item, np.ndarray):
                item = item.tolist()
            if isinstance(item, (list, tuple)) and len(item) == 1:
                item = item[0]
            normalized_task_descriptions.append(item)

        if not normalized_task_descriptions:
            normalized_task_descriptions = [None] * batch_size
        elif len(normalized_task_descriptions) == 1 and batch_size > 1:
            normalized_task_descriptions = normalized_task_descriptions * batch_size
        elif len(normalized_task_descriptions) != batch_size:
            raise RuntimeError(
                "annotation.human.coarse_action batch size mismatch: "
                f"got {len(normalized_task_descriptions)} descriptions for batch size {batch_size}"
            )

        task_description = normalized_task_descriptions[0]

        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        # image: Image.Image = Image.fromarray(image)

        images = [[self._resize_image(img) for img in sample] for sample in images] # (B, N_view, H, W, 3)
        input_state = [input_s for input_s in input_state] # B, state_dim*(sin, cos)

        # prepare vla input
        examples = []
        instructions = normalized_task_descriptions
        for b in range(batch_size):
            example = {
                "image": images[b],  # A list of multi-view images for a single sample
                "lang": instructions[b],
                "state": input_state[b],  # N_history, 58 #Hack BUG
            }
            examples.append(example)
        
        vla_input = {
            "examples": examples,
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
            "dump_denoise": self.dump_denoise,
            "eval_type": self.eval_type,
        }
        
        response = self.client.predict_action(vla_input)
        
        
        # unnormalize the action
        normalized_actions = response["data"]["normalized_actions"] # B, chunk, D        
        self.last_denoise_trace = response["data"].get("denoise_trace")
        
        # unnormalize actions in batch form
        raw_actions = self.unnormalize_actions(normalized_actions=normalized_actions, action_norm_stats=self.action_norm_stats)
        exec_horizon = self._select_exec_horizons(raw_actions)

        self.replan_index += 1
        self._log_replan(exec_horizon)
        self._maybe_log_denoise_step(raw_actions, exec_horizon)
        
        # raw_actions shape: (B, chunk, D)
        if self.action_ensemble:
            # 对batch中的每个样本进行ensemble
            batch_size = raw_actions.shape[0]
            ensembled_actions = []
            for b in range(batch_size):
                ensembled = self.action_ensembler.ensemble_action(raw_actions[b])[None]  # (1, D)
                ensembled_actions.append(ensembled)
            raw_actions = np.stack(ensembled_actions, axis=0)  # (B, 1, D)

        raw_action = {
            "action.left_arm": raw_actions[:, :self.n_action_steps, :7],      # (B, n_action_steps, 7)
            "action.right_arm": raw_actions[:, :self.n_action_steps, 7:14],   # (B, n_action_steps, 7)
            "action.left_hand": raw_actions[:, :self.n_action_steps, 14:20],  # (B, n_action_steps, 6)
            "action.right_hand": raw_actions[:, :self.n_action_steps, 20:26], # (B, n_action_steps, 6)
            "action.waist": raw_actions[:, :self.n_action_steps, 26:29],      # (B, n_action_steps, 3)
        }
        raw_action["exec_horizon"] = exec_horizon[:, None]

        return {"actions": raw_action}

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Args:
            normalized_actions: shape (B, chunk, D) (chunk, D)
            action_norm_stats:
        Returns:
            actions
        """
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        
        normalized_actions = np.clip(normalized_actions, -1, 1)
        
        actions = np.where(
            mask,
            (normalized_actions + 1) / 2 * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        return actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        """
        Duplicate stats accessor (retained for backward compatibility).
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)  # read config and norm_stats

        unnorm_key = PolicyWarper._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]



    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
    
    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Duplicate helper (retained for backward compatibility).
        See primary _check_unnorm_key above.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key
    
    def normalize_state(self, state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Normalize the state
        """
        for key in state.keys():
            sin_state = np.sin(state[key])
            cos_state = np.cos(state[key])
            state[key] = np.concatenate([sin_state, cos_state], axis=-1)
        return state
    
    
