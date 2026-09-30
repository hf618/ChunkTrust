from pathlib import Path
from typing import Any

import numpy as np


class DenoiseLogger:
    """Collect per-replan traces and dump one episode-level npz."""

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

    def start_episode(self, episode_path: str | Path) -> None:
        self._episode_path = Path(episode_path)
        self._episode_path.parent.mkdir(parents=True, exist_ok=True)
        self._step_traces = []
        self._episode_info = {}

    def set_episode_info(
        self,
        *,
        success: bool | None = None,
        success_step: int | None = None,
        episode_steps: int | None = None,
        run_time_seconds: float | None = None,
        denoise_overhead_seconds: float | None = None,
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

    def log_step(
        self,
        v: np.ndarray,
        *,
        ds: np.ndarray | None = None,
        x0: np.ndarray | None = None,
        xt: np.ndarray | None = None,
        xt_internal: np.ndarray | None = None,
        k_exec: int | None = None,
        p_conf: float | None = None,
        features_t: np.ndarray | None = None,
        extras: dict[str, np.ndarray] | None = None,
    ) -> None:
        if not self.enabled:
            return

        record: dict[str, np.ndarray] = {"v": np.asarray(v, dtype=np.float32)}
        if ds is not None:
            record["ds"] = np.asarray(ds, dtype=np.float32)
        if x0 is not None:
            record["X0"] = np.asarray(x0, dtype=np.float32)
        if xt is not None:
            record["XT"] = np.asarray(xt, dtype=np.float32)
        if xt_internal is not None:
            record["XT_internal"] = np.asarray(xt_internal, dtype=np.float32)
        if k_exec is not None:
            record["K_exec"] = np.asarray(int(k_exec), dtype=np.int32)
        if p_conf is not None:
            record["p_conf"] = np.asarray(float(p_conf), dtype=np.float32)
        if features_t is not None:
            record["features_t"] = np.asarray(features_t, dtype=np.float32)
        if extras:
            for key, value in extras.items():
                if value is None:
                    continue
                record[str(key)] = np.asarray(value)
        self._step_traces.append(record)

    def flush_episode(self) -> Path | None:
        if self._episode_path is None:
            return None

        out: dict[str, np.ndarray] = {}
        if self._step_traces:
            keys = set(self._step_traces[0].keys())
            for trace in self._step_traces[1:]:
                keys.intersection_update(trace.keys())

            for key in sorted(keys):
                out[key] = np.stack([trace[key] for trace in self._step_traces], axis=0)

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

        self._save_npz(self._episode_path, **out)
        return self._episode_path
