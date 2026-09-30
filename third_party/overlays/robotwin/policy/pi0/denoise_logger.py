from pathlib import Path
from typing import Any

import numpy as np


class DenoiseLogger:
    """Collects per-step denoising traces and dumps one npz per rollout."""

    _INT32_TRACE_KEYS = {
        "K_exec",
        "replan_env_step_idx",
        "replan_idx",
        "env_step_idx",
        "mp4_frame_idx",
        "selected_K",
        "executed_action_start",
        "executed_action_end",
        "executed_action_range",
        "action_only_tie_count",
    }
    _INT64_TRACE_KEYS = {"monotonic_timestamp_ns"}

    def __init__(self, enabled: bool = False, compress: bool = False):
        self.enabled = bool(enabled)
        self.compress = bool(compress)
        self.reset_episode()

    def _save_npz(self, path: Path, **kwargs: np.ndarray) -> None:
        if self.compress:
            np.savez_compressed(path, **kwargs)
        else:
            np.savez(path, **kwargs)

    def reset_episode(self) -> None:
        self._episode_path: Path | None = None
        self._step_traces: list[dict[str, np.ndarray]] = []
        self._episode_info: dict[str, Any] = {}

    def start_episode(self, episode_path: str | Path, meta: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        self._episode_path = Path(episode_path)
        self._episode_path.parent.mkdir(parents=True, exist_ok=True)
        self._step_traces = []
        self._episode_info = {}
        if meta:
            for key, value in meta.items():
                self._episode_info[key] = value

    def set_episode_info(
        self,
        *,
        success: bool | None = None,
        success_step: int | None = None,
        episode_steps: int | None = None,
        run_time_seconds: float | None = None,
        denoise_overhead_seconds: float | None = None,
        policy_infer_seconds_total: float | None = None,
        horizon_metric_compute_seconds_total: float | None = None,
        horizon_selection_seconds_total: float | None = None,
    ) -> None:
        if success is not None:
            self._episode_info["success"] = bool(success)
        if success_step is not None:
            self._episode_info["success_step"] = int(success_step)
        if episode_steps is not None:
            self._episode_info["episode_steps"] = int(episode_steps)
        if run_time_seconds is not None:
            self._episode_info["run_time_seconds_total"] = float(run_time_seconds)
        if denoise_overhead_seconds is not None:
            self._episode_info["run_time_seconds_denoise_overhead"] = float(denoise_overhead_seconds)
        if policy_infer_seconds_total is not None:
            self._episode_info["policy_infer_seconds_total"] = float(policy_infer_seconds_total)
        if horizon_metric_compute_seconds_total is not None:
            self._episode_info["horizon_metric_compute_seconds_total"] = float(horizon_metric_compute_seconds_total)
        if horizon_selection_seconds_total is not None:
            self._episode_info["horizon_selection_seconds_total"] = float(horizon_selection_seconds_total)

    def log_step(
        self,
        v: np.ndarray,
        *,
        ds: np.ndarray | None = None,
        x0: np.ndarray | None = None,
        xt: np.ndarray | None = None,
        xt_internal: np.ndarray | None = None,
        k_exec: int | None = None,
        replan_idx: int | None = None,
        replan_env_step_idx: int | None = None,
        env_step_idx: int | None = None,
        mp4_frame_idx: int | None = None,
        monotonic_timestamp_ns: int | None = None,
        selected_k: int | None = None,
        executed_action_start: int | None = None,
        executed_action_end: int | None = None,
        executed_action_range: np.ndarray | None = None,
        a_exec: np.ndarray | None = None,
        a_exec_mask: np.ndarray | None = None,
        gripper_exec: np.ndarray | None = None,
        q_intra: np.ndarray | None = None,
        p_inter: np.ndarray | None = None,
        q_mix: np.ndarray | None = None,
        z_intra_raw: np.ndarray | None = None,
        z_intra_norm: np.ndarray | None = None,
        u_inter_raw: np.ndarray | None = None,
        u_inter_norm: np.ndarray | None = None,
        action_only_candidate_metrics: np.ndarray | None = None,
        action_only_tie_count: int | None = None,
        action_only_continuous_action_dimensions: np.ndarray | None = None,
    ) -> None:
        if not self.enabled:
            return
        record: dict[str, np.ndarray] = {
            "v": np.asarray(v),
            "replan_idx": np.asarray(len(self._step_traces) if replan_idx is None else int(replan_idx), dtype=np.int32),
        }
        if ds is not None:
            record["ds"] = np.asarray(ds)
        if x0 is not None:
            record["X0"] = np.asarray(x0)
        if xt is not None:
            record["XT"] = np.asarray(xt)
        if xt_internal is not None:
            record["XT_internal"] = np.asarray(xt_internal)
        if k_exec is not None:
            record["K_exec"] = np.asarray(int(k_exec), dtype=np.int32)
            record["selected_K"] = np.asarray(int(k_exec), dtype=np.int32)
        if selected_k is not None:
            record["selected_K"] = np.asarray(int(selected_k), dtype=np.int32)
        if replan_env_step_idx is not None:
            record["replan_env_step_idx"] = np.asarray(int(replan_env_step_idx), dtype=np.int32)
            record["env_step_idx"] = np.asarray(int(replan_env_step_idx), dtype=np.int32)
            record["mp4_frame_idx"] = np.asarray(int(replan_env_step_idx), dtype=np.int32)
        if env_step_idx is not None:
            record["env_step_idx"] = np.asarray(int(env_step_idx), dtype=np.int32)
            record["replan_env_step_idx"] = np.asarray(int(env_step_idx), dtype=np.int32)
            if mp4_frame_idx is None:
                record["mp4_frame_idx"] = np.asarray(int(env_step_idx), dtype=np.int32)
        if mp4_frame_idx is not None:
            record["mp4_frame_idx"] = np.asarray(int(mp4_frame_idx), dtype=np.int32)
        if monotonic_timestamp_ns is not None:
            record["monotonic_timestamp_ns"] = np.asarray(int(monotonic_timestamp_ns), dtype=np.int64)
        if executed_action_start is not None:
            record["executed_action_start"] = np.asarray(int(executed_action_start), dtype=np.int32)
        if executed_action_end is not None:
            record["executed_action_end"] = np.asarray(int(executed_action_end), dtype=np.int32)
        if executed_action_range is not None:
            record["executed_action_range"] = np.asarray(executed_action_range, dtype=np.int32)
        if a_exec is not None:
            record["A_exec"] = np.asarray(a_exec)
        if a_exec_mask is not None:
            record["A_exec_mask"] = np.asarray(a_exec_mask)
        if gripper_exec is not None:
            record["gripper_exec"] = np.asarray(gripper_exec)
        if q_intra is not None:
            record["q_intra"] = np.asarray(q_intra)
        if p_inter is not None:
            record["p_inter"] = np.asarray(p_inter)
        if q_mix is not None:
            record["q_mix"] = np.asarray(q_mix)
        if z_intra_raw is not None:
            record["z_intra_raw"] = np.asarray(z_intra_raw)
        if z_intra_norm is not None:
            record["z_intra_norm"] = np.asarray(z_intra_norm)
        if u_inter_raw is not None:
            record["u_inter_raw"] = np.asarray(u_inter_raw)
        if u_inter_norm is not None:
            record["u_inter_norm"] = np.asarray(u_inter_norm)
        if action_only_candidate_metrics is not None:
            record["action_only_candidate_metrics"] = np.asarray(action_only_candidate_metrics, dtype=np.float32)
        if action_only_tie_count is not None:
            record["action_only_tie_count"] = np.asarray(int(action_only_tie_count), dtype=np.int32)
        if action_only_continuous_action_dimensions is not None:
            record["action_only_continuous_action_dimensions"] = np.asarray(
                action_only_continuous_action_dimensions,
                dtype=np.int32,
            )
        self._step_traces.append(record)

    def update_last_step(self, **kwargs: np.ndarray | int | float | str | None) -> None:
        if not self.enabled or not self._step_traces:
            return
        record = self._step_traces[-1]
        for key, value in kwargs.items():
            if value is None:
                continue
            if key == "executed_action_range":
                record[key] = np.asarray(value, dtype=np.int32)
            elif key in self._INT32_TRACE_KEYS:
                record[key] = np.asarray(int(value), dtype=np.int32)
            elif key in self._INT64_TRACE_KEYS:
                record[key] = np.asarray(int(value), dtype=np.int64)
            else:
                record[key] = np.asarray(value)

    def update_last_step_alignment(
        self,
        *,
        env_step_idx: int,
        selected_k: int,
        mp4_frame_idx: int | None = None,
        monotonic_timestamp_ns: int | None = None,
    ) -> None:
        end = int(env_step_idx) + int(selected_k)
        self.update_last_step(
            replan_env_step_idx=int(env_step_idx),
            env_step_idx=int(env_step_idx),
            mp4_frame_idx=int(env_step_idx) if mp4_frame_idx is None else int(mp4_frame_idx),
            selected_K=int(selected_k),
            K_exec=int(selected_k),
            executed_action_start=int(env_step_idx),
            executed_action_end=end,
            executed_action_range=np.asarray([int(env_step_idx), end], dtype=np.int32),
            monotonic_timestamp_ns=monotonic_timestamp_ns,
        )

    def _stack_trace_key(self, key: str) -> np.ndarray:
        stacked = np.stack([trace[key] for trace in self._step_traces], axis=0)
        if key in self._INT64_TRACE_KEYS:
            return stacked.astype(np.int64, copy=False)
        if key in self._INT32_TRACE_KEYS:
            return stacked.astype(np.int32, copy=False)
        return stacked.astype(np.float32, copy=False)

    @staticmethod
    def _coerce_episode_value(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, (bool, np.bool_)):
            return np.asarray(value, dtype=np.bool_)
        if isinstance(value, (int, np.integer)):
            return np.asarray(int(value), dtype=np.int32)
        if isinstance(value, (float, np.floating)):
            return np.asarray(float(value), dtype=np.float32)
        if isinstance(value, (list, tuple)):
            return np.asarray(value)
        return np.asarray(str(value))

    def flush_episode(self) -> Path | None:
        if not self.enabled or self._episode_path is None:
            return None

        out: dict[str, np.ndarray] = {}
        if self._step_traces:
            keys = set(self._step_traces[0].keys())
            for trace in self._step_traces[1:]:
                keys.intersection_update(trace.keys())

            for key in sorted(keys):
                out[key] = self._stack_trace_key(key)

        if "success" in self._episode_info:
            out["success"] = np.asarray(self._episode_info["success"], dtype=np.bool_)
        if "success_step" in self._episode_info:
            out["success_step"] = np.asarray(self._episode_info["success_step"], dtype=np.int32)
        if "episode_steps" in self._episode_info:
            out["episode_steps"] = np.asarray(self._episode_info["episode_steps"], dtype=np.int32)
        if "run_time_seconds_total" in self._episode_info:
            total = float(self._episode_info["run_time_seconds_total"])
            denoise = float(self._episode_info.get("run_time_seconds_denoise_overhead", 0.0))
            out["run_time_seconds"] = np.asarray([total, denoise], dtype=np.float32)
        for key, value in self._episode_info.items():
            if key in {"success", "success_step", "episode_steps", "run_time_seconds_total", "run_time_seconds_denoise_overhead"}:
                continue
            out[key] = self._coerce_episode_value(value)
        self._save_npz(self._episode_path, **out)
        return self._episode_path
