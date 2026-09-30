from pathlib import Path
from typing import Any

import numpy as np


class DenoiseLogger:
    """Collects full denoising traces or lightweight replan metadata."""

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
        "qha_candidate_horizons",
        "qha_argmax_horizon",
        "qha_selector_candidate_horizons_sparse_runtime",
        "qha_runtime_candidate_horizons",
        "v3_k_minus",
        "v3_k",
        "v3_k_plus",
        "counterfactual_target_replan_idx",
        "counterfactual_natural_k",
        "counterfactual_live_natural_k",
        "counterfactual_executed_k",
        "counterfactual_role_code",
    }
    _INT64_TRACE_KEYS = {"monotonic_timestamp_ns"}
    _BOOL_TRACE_KEYS = {
        "action_plan_valid_mask",
        "same_observation_action_plan_valid",
        "qha_context_mask",
        "v3_lower_action_plan_valid_mask",
        "v3_lower_shadow_valid",
        "counterfactual_enabled",
        "counterfactual_fired",
        "counterfactual_prefix_replay",
    }

    def __init__(self, enabled: bool = False, compress: bool = False, include_arrays: bool = True):
        self.enabled = bool(enabled)
        self.compress = bool(compress)
        self.include_arrays = bool(include_arrays)
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
        self._episode_meta: dict[str, Any] = {}
        self._shift_trace_shapes: dict[str, tuple[int, ...]] = {}
        self._shift_trace_keys: frozenset[str] | None = None
        self._rsi_env_progress: list[tuple[int, float, int, int]] = []

    def start_episode(self, episode_path: str | Path, meta: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        self._episode_path = Path(episode_path)
        self._episode_path.parent.mkdir(parents=True, exist_ok=True)
        self._step_traces = []
        self._episode_info = {}
        self._episode_meta = dict(meta or {})
        self._shift_trace_shapes = {}
        self._shift_trace_keys = None
        self._rsi_env_progress = []

    def log_rsi_env_progress(self, env_step_idx: int, progress: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._rsi_env_progress.append(
            (
                int(env_step_idx),
                float(progress["progress"]),
                int(progress["placed_count"]),
                int(progress["bread_count"]),
            )
        )

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
        action_chunk_size: int | None = None,
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
        if action_chunk_size is not None:
            self._episode_info["action_chunk_size"] = int(action_chunk_size)

    def log_step(
        self,
        v: np.ndarray | None,
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
        policy_infer_seconds: float | None = None,
        horizon_metric_compute_seconds: float | None = None,
        horizon_selection_seconds: float | None = None,
        shift_trace: dict[str, np.ndarray | int | float | bool] | None = None,
    ) -> None:
        if not self.enabled:
            return
        record: dict[str, np.ndarray] = {
            "replan_idx": np.asarray(len(self._step_traces) if replan_idx is None else int(replan_idx), dtype=np.int32),
        }
        if self.include_arrays and v is not None:
            record["v"] = np.asarray(v)
        if self.include_arrays and ds is not None:
            record["ds"] = np.asarray(ds)
        if self.include_arrays and x0 is not None:
            record["X0"] = np.asarray(x0)
        if self.include_arrays and xt is not None:
            record["XT"] = np.asarray(xt)
        if self.include_arrays and xt_internal is not None:
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
        if policy_infer_seconds is not None:
            record["policy_infer_seconds"] = np.asarray(float(policy_infer_seconds), dtype=np.float32)
        if horizon_metric_compute_seconds is not None:
            record["horizon_metric_compute_seconds"] = np.asarray(float(horizon_metric_compute_seconds), dtype=np.float32)
        if horizon_selection_seconds is not None:
            record["horizon_selection_seconds"] = np.asarray(float(horizon_selection_seconds), dtype=np.float32)
        if shift_trace is not None:
            current_keys = frozenset(shift_trace)
            if self._shift_trace_keys is None:
                self._shift_trace_keys = current_keys
            elif current_keys != self._shift_trace_keys:
                missing = sorted(self._shift_trace_keys - current_keys)
                extra = sorted(current_keys - self._shift_trace_keys)
                raise ValueError(f"SHIFT trace keys changed within episode: missing={missing}, extra={extra}")
            for key, value in shift_trace.items():
                arr = np.asarray(value)
                expected_shape = self._shift_trace_shapes.setdefault(key, arr.shape)
                if arr.shape != expected_shape:
                    raise ValueError(
                        f"SHIFT trace field {key!r} changed shape within episode: "
                        f"expected {expected_shape}, got {arr.shape}"
                    )
                record[key] = arr
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
        if key in self._BOOL_TRACE_KEYS:
            return stacked.astype(np.bool_, copy=False)
        return stacked.astype(np.float32, copy=False)

    def flush_episode(self) -> Path | None:
        if not self.enabled or self._episode_path is None:
            return None

        out: dict[str, np.ndarray] = {}
        if self._rsi_env_progress:
            progress = np.asarray(self._rsi_env_progress, dtype=np.float64)
            out["rsi_progress_env_step_idx"] = progress[:, 0].astype(np.int32)
            out["rsi_progress_per_env_step"] = progress[:, 1].astype(np.float32)
            out["rsi_progress_placed_count_per_env_step"] = progress[:, 2].astype(np.int32)
            out["rsi_progress_total_count_per_env_step"] = progress[:, 3].astype(np.int32)
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
        if "policy_infer_seconds_total" in self._episode_info:
            out["policy_infer_seconds_total"] = np.asarray(
                float(self._episode_info["policy_infer_seconds_total"]),
                dtype=np.float32,
            )
        if "horizon_metric_compute_seconds_total" in self._episode_info:
            out["horizon_metric_compute_seconds_total"] = np.asarray(
                float(self._episode_info["horizon_metric_compute_seconds_total"]),
                dtype=np.float32,
            )
        if "horizon_selection_seconds_total" in self._episode_info:
            out["horizon_selection_seconds_total"] = np.asarray(
                float(self._episode_info["horizon_selection_seconds_total"]),
                dtype=np.float32,
            )
        action_chunk_size = self._episode_info.get(
            "action_chunk_size",
            self._episode_meta.get("action_chunk_size", self._episode_meta.get("pi0_step")),
        )
        if action_chunk_size is not None:
            out["action_chunk_size"] = np.asarray(int(action_chunk_size), dtype=np.int32)
        for key, value in self._episode_meta.items():
            if key in out or value is None:
                continue
            if isinstance(value, (bool, int, float, str)):
                out[key] = np.asarray(value)
        self._save_npz(self._episode_path, **out)
        return self._episode_path
