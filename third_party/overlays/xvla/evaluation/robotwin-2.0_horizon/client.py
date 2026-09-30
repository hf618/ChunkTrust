from chunktrust.paths import resolve_legacy_path as _ct_path
import os
import sys
from pathlib import Path


def _resolve_robowin_root() -> Path:
    env_value = os.environ.get("ROBOTWIN_ROOT", "").strip()
    candidates = [env_value, _ct_path('ROBOTWIN_ROOT', '')]
    for item in candidates:
        if not item:
            continue
        path = Path(item).expanduser().resolve()
        if path.exists():
            return path
    raise FileNotFoundError(
        "RoboTwin root not found. Please set ROBOTWIN_ROOT, e.g. "
        "'export ROBOTWIN_ROOT=/path/to/RoboTwin'."
    )


robowin_root = _resolve_robowin_root()
if str(robowin_root) not in sys.path:
    sys.path.insert(0, str(robowin_root))

os.chdir(robowin_root)

import argparse
import csv
from datetime import datetime
import importlib
import json
import logging
import math
import subprocess
import time
import traceback

import cv2
import imageio
import json_numpy
import numpy as np
import requests
import torch
import yaml
from denoise_logger import DenoiseLogger
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.float32)

ALL_TASKS = [
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle_horizontally",
    "shake_bottle",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
]

print("Number of tasks evaluating:", len(ALL_TASKS))

BASE_MODEL_TYPE = "xvla"


def task_config_to_setting(task_config: str) -> str:
    if task_config == "demo_clean":
        return "clean"
    if task_config in {"demo_randomized", "demo_random"}:
        return "randomized"
    return task_config.replace("demo_", "")


def make_run_layout(eval_root: str | Path, task_name: str, task_config: str, ckpt_tag: str, eval_tag: str) -> dict[str, Path | str]:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    setting = task_config_to_setting(task_config)
    task_name_with_setting = f"{task_name}_{setting}"
    run_dir = Path(eval_root) / task_name_with_setting / BASE_MODEL_TYPE / ckpt_tag / eval_tag / timestamp
    info_dir = run_dir / "info"
    episodes_dir = run_dir / "episodes"
    summary_dir = Path(eval_root) / "_summary" / BASE_MODEL_TYPE / ckpt_tag / eval_tag
    for directory in (run_dir, info_dir, episodes_dir, summary_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "timestamp": timestamp,
        "task_name_with_setting": task_name_with_setting,
        "run_dir": run_dir,
        "info_dir": info_dir,
        "episodes_dir": episodes_dir,
        "summary_dir": summary_dir,
        "summary_csv": summary_dir / "summary.csv",
    }


def append_summary_row(summary_csv: Path, row: dict[str, object]) -> None:
    fieldnames = [
        "task_name",
        "task_config",
        "task_name_with_setting",
        "eval_type",
        "run_dir",
        "info_json",
        "video_dir",
        "success_rate",
        "success_count",
        "expected_rollouts",
        "timestamp",
        "avg_q_intra",
        "avg_p_inter",
        "avg_q_mix",
        "avg_q_intra_contrib",
        "avg_p_inter_contrib",
        "avg_z_intra_raw",
        "avg_z_intra_norm",
        "avg_u_inter_raw",
        "avg_u_inter_norm",
    ]
    existing_rows: list[dict[str, object]] = []
    write_header = True
    if summary_csv.exists():
        with summary_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing_header = list(reader.fieldnames or [])
            existing_rows = list(reader)
        if existing_header == fieldnames:
            write_header = False
        else:
            with summary_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for existing_row in existing_rows:
                    writer.writerow({key: existing_row.get(key, "") for key in fieldnames})
            write_header = False

    with summary_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_run_info(info_path: Path, info: dict[str, object]) -> None:
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=True)


def quat_to_rotate6D(q: np.ndarray) -> np.ndarray:
    return R.from_quat(q).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))


def rotate6D_to_quat(v6: np.ndarray) -> np.ndarray:
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError(f"Last dimension must be 6 (got {v6.shape[-1]})")
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)
    return R.from_matrix(rot_mats).as_quat()


class ClientModel:
    def __init__(
        self,
        host,
        port,
        *,
        seed=0,
        num_denoise_steps=10,
        eval_type="default",
        fixed_exec_k=0,
        dump_denoise=0,
        denoise_npz_compress=0,
        horizon_candidates="10,20,30",
        horizon_intra_alpha=4.0,
        horizon_ts_epsilon=0.05,
        horizon_ts_seed="seed",
        horizon_ts_update_mode="kernel_forget",
        horizon_ts_kernel_bandwidth=10.0,
        horizon_ts_forget_rho=0.99,
        horizon_ts_update_eta=1.0,
        horizon_exec_mode="expected_round",
        horizon_expected_temp=1.0,
        horizon_intra_cut_t=0.25,
        horizon_intra_z_mode="window_rms",
        horizon_intra_tau_window=3,
        horizon_mix_mode="both",
        horizon_inter_use_speed_uniformity=1,
        horizon_inter_window=40,
        horizon_inter_weight=0.5,
        horizon_inter_beta=4.0,
        horizon_inter_fallback="ehh_only",
    ):
        self.url = f"http://{host}:{port}/act"
        self.seed = int(seed)
        self.num_denoise_steps = int(num_denoise_steps)
        self.eval_type = str(eval_type).strip().lower()
        if self.eval_type not in {"default", "horizon"}:
            self.eval_type = "default"
        self.fixed_exec_k = int(fixed_exec_k)
        self.dump_denoise = bool(int(dump_denoise))
        self.denoise_npz_compress = bool(int(denoise_npz_compress))
        self.trace_enabled = self.dump_denoise or (self.eval_type == "horizon")

        self.horizon_candidates = self._parse_horizon_candidates(horizon_candidates)
        self.horizon_intra_alpha = float(horizon_intra_alpha)
        self.horizon_ts_epsilon = float(np.clip(float(horizon_ts_epsilon), 0.0, 1.0))
        self.horizon_ts_seed = self._parse_horizon_seed(horizon_ts_seed, seed=self.seed)
        self.horizon_ts_update_mode = self._parse_horizon_ts_update_mode(horizon_ts_update_mode)
        self.horizon_ts_kernel_bandwidth = float(max(1e-6, float(horizon_ts_kernel_bandwidth)))
        self.horizon_ts_forget_rho = float(np.clip(float(horizon_ts_forget_rho), 1e-6, 1.0))
        self.horizon_ts_update_eta = float(max(0.0, float(horizon_ts_update_eta)))
        self.horizon_exec_mode = self._parse_horizon_exec_mode(horizon_exec_mode)
        self.horizon_expected_temp = float(max(1e-6, float(horizon_expected_temp)))
        self.horizon_intra_cut_t = float(np.clip(float(horizon_intra_cut_t), 0.0, 1.0))
        self.horizon_intra_z_mode = self._parse_horizon_z_mode(horizon_intra_z_mode)
        self.horizon_intra_tau_window = max(1, int(horizon_intra_tau_window))
        self.horizon_mix_mode = self._parse_horizon_mix_mode(horizon_mix_mode)
        self.horizon_inter_use_speed_uniformity = bool(int(horizon_inter_use_speed_uniformity))
        self.horizon_inter_window = max(4, int(horizon_inter_window))
        self.horizon_inter_weight = float(np.clip(float(horizon_inter_weight), 0.0, 1.0))
        self.horizon_inter_beta = float(max(0.0, float(horizon_inter_beta)))
        self.horizon_inter_fallback = self._parse_horizon_inter_fallback(horizon_inter_fallback)

        self.denoise_logger = DenoiseLogger(enabled=self.dump_denoise, compress=self.denoise_npz_compress)
        self.instruction = None
        self._episode_path = None
        self._denoise_overhead_seconds = 0.0
        self.last_exec_k = None
        self._reset_run_score_state()
        self._reset_horizon_state()

    @staticmethod
    def _parse_horizon_seed(raw, *, seed: int) -> int:
        if raw is None:
            return int(seed)
        text = str(raw).strip().lower()
        if text == "seed":
            return int(seed)
        return int(raw)

    @staticmethod
    def _parse_horizon_candidates(raw) -> list[int]:
        if raw is None:
            return [10, 20, 30]
        vals = raw if isinstance(raw, (list, tuple)) else str(raw).split(",")
        out = []
        for item in vals:
            try:
                value = int(str(item).strip())
            except Exception:
                continue
            if value > 0:
                out.append(value)
        return sorted(set(out or [10, 20, 30]))

    @staticmethod
    def _parse_horizon_exec_mode(raw) -> str:
        mode = str(raw or "thompson").strip().lower()
        if mode not in {"thompson", "expected_round"}:
            mode = "thompson"
        return mode

    @staticmethod
    def _parse_horizon_ts_update_mode(raw) -> str:
        mode = str(raw or "independent").strip().lower()
        if mode not in {"independent", "kernel_forget"}:
            mode = "independent"
        return mode

    @staticmethod
    def _parse_horizon_z_mode(raw) -> str:
        mode = str(raw or "window_rms").strip().lower()
        if mode not in {"tau0_rms", "window_rms", "trend"}:
            mode = "window_rms"
        return mode

    @staticmethod
    def _parse_horizon_mix_mode(raw) -> str:
        mode = str(raw or "both").strip().lower()
        if mode not in {"intra", "inter", "both"}:
            mode = "both"
        return mode

    @staticmethod
    def _parse_horizon_inter_fallback(raw) -> str:
        mode = str(raw or "ehh_only").strip().lower()
        if mode not in {"ehh_only", "neutral"}:
            mode = "ehh_only"
        return mode

    def _reset_horizon_state(self) -> None:
        self._horizon_ts_rng = np.random.default_rng(self.horizon_ts_seed)
        self._horizon_ts_state = {int(s): [1.0, 1.0] for s in self.horizon_candidates}
        self._executed_action_history = np.zeros((0, 0), dtype=np.float32)

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

    def reset_run_score_state(self) -> None:
        self._reset_run_score_state()

    def reset_episode_state(self) -> None:
        self.last_exec_k = None
        self._episode_path = None
        self._denoise_overhead_seconds = 0.0
        self.denoise_logger.reset_episode()
        self._reset_horizon_state()

    @staticmethod
    def _mean_finite(values: np.ndarray | list[float]) -> float | None:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size <= 0:
            return None
        finite = arr[np.isfinite(arr)]
        if finite.size <= 0:
            return None
        return float(np.mean(finite))

    def _record_run_score_mean(self, key: str, value: float | None) -> None:
        if value is None or not np.isfinite(value):
            return
        bucket = self._run_score_state[key]
        bucket["sum"] += float(value)
        bucket["count"] += 1

    @staticmethod
    def _normalize_candidate_metric(values: np.ndarray | list[float], eps: float = 1e-12) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size <= 0:
            return np.zeros((0,), dtype=np.float64)
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        finite = np.isfinite(arr)
        if not np.any(finite):
            return out
        valid = arr[finite]
        if valid.size <= 1 or float(np.max(valid) - np.min(valid)) <= eps:
            out[finite] = 0.0
            return out
        out[finite] = (valid - float(np.min(valid))) / (float(np.max(valid) - np.min(valid)) + eps)
        return out

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

    def _update_run_score_stats(self, horizon_info: dict | None) -> None:
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

    def get_run_score_averages(self) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for key, bucket in self._run_score_state.items():
            count = int(bucket["count"])
            if count <= 0:
                out[key] = None
            else:
                out[key] = float(bucket["sum"] / float(count))
        return out

    def set_instruction(self, instruction):
        self.instruction = instruction

    def prepare_episode_logging(self, episode_dir: Path, episode_idx: int) -> None:
        if not self.trace_enabled:
            return
        self._episode_path = Path(episode_dir) / f"episode{episode_idx}.npz"
        meta = {
            "eval_type": self.eval_type,
            "horizon_policy": self._horizon_policy_name(),
            "horizon_candidates": np.asarray(self.horizon_candidates, dtype=np.int32),
            "horizon_intra_alpha": self.horizon_intra_alpha,
            "horizon_ts_epsilon": self.horizon_ts_epsilon,
            "horizon_ts_seed": self.horizon_ts_seed,
            "horizon_ts_update_mode": self.horizon_ts_update_mode,
            "horizon_ts_kernel_bandwidth": self.horizon_ts_kernel_bandwidth,
            "horizon_ts_forget_rho": self.horizon_ts_forget_rho,
            "horizon_ts_update_eta": self.horizon_ts_update_eta,
            "horizon_exec_mode": self.horizon_exec_mode,
            "horizon_expected_temp": self.horizon_expected_temp,
            "horizon_intra_cut_t": self.horizon_intra_cut_t,
            "horizon_intra_z_mode": self.horizon_intra_z_mode,
            "horizon_intra_tau_window": self.horizon_intra_tau_window,
            "horizon_mix_mode": self.horizon_mix_mode,
            "horizon_inter_use_speed_uniformity": int(self.horizon_inter_use_speed_uniformity),
            "horizon_inter_window": self.horizon_inter_window,
            "horizon_inter_weight": self.horizon_inter_weight,
            "horizon_inter_beta": self.horizon_inter_beta,
            "horizon_inter_fallback": self.horizon_inter_fallback,
            "fixed_exec_k": self.fixed_exec_k,
        }
        self.denoise_logger.start_episode(self._episode_path, meta=meta)

    def finalize_episode_logging(
        self,
        *,
        success=None,
        success_step=None,
        episode_steps=None,
        run_time_seconds=None,
    ):
        if not self.trace_enabled:
            return None
        self.denoise_logger.set_episode_info(
            success=success,
            success_step=success_step,
            episode_steps=episode_steps,
            run_time_seconds=run_time_seconds,
            denoise_overhead_seconds=self._denoise_overhead_seconds,
        )
        return self.denoise_logger.flush_episode()

    def append_executed_actions(self, actions: np.ndarray) -> None:
        if self.eval_type != "horizon":
            return
        actions = np.asarray(actions, dtype=np.float32)
        if actions.size == 0:
            return
        if self._executed_action_history.size == 0:
            self._executed_action_history = actions.copy()
            return
        if self._executed_action_history.shape[1] != actions.shape[1]:
            self._executed_action_history = actions.copy()
            return
        self._executed_action_history = np.concatenate([self._executed_action_history, actions], axis=0)

    def _horizon_policy_name(self) -> str:
        parts = ["xvla"]
        if self.eval_type == "horizon":
            parts.extend(["horizon", self.horizon_intra_z_mode])
            if self.horizon_inter_use_speed_uniformity and self.horizon_mix_mode in ("inter", "both"):
                parts.append("speedu")
            parts.append(self.horizon_mix_mode)
            parts.append(self.horizon_exec_mode)
            if self.horizon_ts_update_mode != "independent":
                parts.append(self.horizon_ts_update_mode)
            if self.horizon_exec_mode == "expected_round":
                temp_str = f"{float(self.horizon_expected_temp):g}".replace(".", "p")
                parts.append(f"temp{temp_str}")
        else:
            parts.append("default")
        return "_".join(parts)

    @staticmethod
    def _normalize_score_distribution(score_arr: np.ndarray, temp: float = 1.0) -> np.ndarray:
        arr = np.asarray(score_arr, dtype=np.float64)
        if arr.size <= 0:
            return np.zeros((0,), dtype=np.float64)
        arr = np.maximum(arr, 0.0)
        arr_sum = float(np.sum(arr))
        if arr_sum <= 1e-12 or not np.isfinite(arr_sum):
            return np.full((arr.size,), 1.0 / float(arr.size), dtype=np.float64)
        if abs(float(temp) - 1.0) <= 1e-12:
            return arr / arr_sum
        base = arr / arr_sum
        sharp = np.power(np.clip(base, 0.0, 1.0), 1.0 / float(max(1e-6, temp)))
        sharp_sum = float(np.sum(sharp))
        if sharp_sum <= 1e-12 or not np.isfinite(sharp_sum):
            return np.full((arr.size,), 1.0 / float(arr.size), dtype=np.float64)
        return sharp / sharp_sum

    def _compute_ehh_1d_padded(self, mat_h_da: np.ndarray, h_full: int, eps: float = 1e-12) -> float:
        hs = int(mat_h_da.shape[0])
        da = int(mat_h_da.shape[1])
        pad_h = max(int(h_full), hs)
        if pad_h == hs:
            mat_pad = np.asarray(mat_h_da, dtype=np.float64)
        else:
            mat_pad = np.zeros((pad_h, da), dtype=np.float64)
            mat_pad[:hs, :] = np.asarray(mat_h_da, dtype=np.float64)
        spec = np.fft.rfft(mat_pad, axis=0)
        pwr = (spec.real * spec.real) + (spec.imag * spec.imag)
        pwr_f = np.mean(pwr, axis=1)
        f_t = int(pwr_f.shape[0])
        ft_c = max(1, int(self.horizon_intra_cut_t * f_t))
        e_low = float(np.sum(pwr_f[:ft_c]))
        e_high = float(np.sum(pwr_f[ft_c:]))
        e_all = e_low + e_high + eps
        return float(e_high / e_all)

    def _ehh_curves_by_horizon_batched(
        self, v_step: np.ndarray, horizons: list[int], h_full: int | None = None, eps: float = 1e-12
    ) -> dict[int, tuple[np.ndarray, float]]:
        n_tau = int(v_step.shape[0])
        h = int(v_step.shape[1])
        d_a = int(v_step.shape[2])
        if n_tau <= 0:
            return {}
        if h_full is None:
            h_full = h
        h_full = max(int(h_full), h)
        horizons = [int(s) for s in horizons]
        if not horizons:
            return {}

        batch = np.zeros((len(horizons), n_tau, h_full, d_a), dtype=np.float64)
        v64 = np.asarray(v_step, dtype=np.float64)
        for i, s in enumerate(horizons):
            s_clip = max(0, min(int(s), h))
            if s_clip > 0:
                batch[i, :, :s_clip, :] = v64[:, :s_clip, :]

        spec = np.fft.rfft(batch, axis=2)
        pwr = (spec.real * spec.real) + (spec.imag * spec.imag)
        pwr_f = np.mean(pwr, axis=3)
        f_t = int(pwr_f.shape[2])
        ft_c = max(1, int(self.horizon_intra_cut_t * f_t))
        e_low = np.sum(pwr_f[:, :, :ft_c], axis=2)
        e_high = np.sum(pwr_f[:, :, ft_c:], axis=2)
        e_vals_all = e_high / (e_low + e_high + eps)

        out = {}
        for i, s in enumerate(horizons):
            e_vals = np.asarray(e_vals_all[i], dtype=np.float64)
            out[int(s)] = (e_vals, float(np.mean(e_vals)))
        return out

    @staticmethod
    def _z_trend_abs_slope(y: np.ndarray) -> float:
        n = int(y.size)
        if n <= 1:
            return 0.0
        x = np.arange(n, dtype=np.float64)
        x = x - np.mean(x)
        y0 = y - np.mean(y)
        denom = float(np.sum(x * x))
        if denom <= 1e-12:
            return 0.0
        slope = float(np.sum(x * y0) / denom)
        return abs(slope)

    def _compute_z_from_ehh(self, e_vals: np.ndarray, tau0_idx: int = 0) -> float:
        if e_vals.size <= 1:
            return 0.0
        tau0 = int(np.clip(tau0_idx, 0, max(0, int(e_vals.size) - 1)))
        tail = np.asarray(e_vals[tau0:], dtype=np.float64)
        if tail.size <= 1:
            return 0.0
        if self.horizon_intra_z_mode == "trend":
            return self._z_trend_abs_slope(tail)
        if self.horizon_intra_z_mode == "tau0_rms":
            baseline = float(e_vals[tau0])
        else:
            hi = min(int(e_vals.size), tau0 + self.horizon_intra_tau_window)
            baseline = float(np.mean(e_vals[tau0:hi]))
        dev = tail - baseline
        return float(math.sqrt(float(np.mean(dev * dev))))

    def _speed_uniformity_proxy_by_horizon(
        self, xt_step: np.ndarray, horizons: list[int], eps: float = 1e-12
    ) -> tuple[list[float], list[float], list[float]]:
        xt_step = np.asarray(xt_step, dtype=np.float32)
        if xt_step.ndim != 2 or xt_step.shape[1] <= 0:
            empty = [float("nan")] * len(horizons)
            return empty, empty, empty

        h = int(xt_step.shape[0])
        da = int(xt_step.shape[1])
        win = max(4, int(self.horizon_inter_window))
        history_actions = self._executed_action_history
        if history_actions.ndim != 2 or int(history_actions.shape[1]) != da:
            history_actions = np.zeros((0, da), dtype=np.float32)
            self._executed_action_history = history_actions
        history_len = int(history_actions.shape[0])
        max_horizon = max([int(s) for s in horizons], default=0)
        history_tail_len = min(history_len, win)

        tail = np.asarray(history_actions[-history_tail_len:, :], dtype=np.float32)
        xt_prefix = np.asarray(xt_step[:max_horizon, :], dtype=np.float32)
        seq = np.concatenate([tail, xt_prefix], axis=0)
        if seq.shape[0] >= 2:
            speed = np.linalg.norm(np.diff(seq, axis=0), axis=1).astype(np.float64, copy=False)
        else:
            speed = np.zeros((0,), dtype=np.float64)
        speed_prefix = np.concatenate([[0.0], np.cumsum(speed, dtype=np.float64)])
        speed_sq_prefix = np.concatenate([[0.0], np.cumsum(speed * speed, dtype=np.float64)])

        u_raw = []
        for s in horizons:
            s = int(s)
            if s < 1 or s > h:
                u_raw.append(float("nan"))
                continue
            pre = int(win - s)
            post = int(s)
            if pre < 1 or post < 1 or history_len < pre or h < post:
                u_raw.append(float("nan"))
                continue

            start = int(history_tail_len - pre)
            speed_start = start
            speed_end = speed_start + (win - 1)
            if speed_start < 0 or speed_end > int(speed.shape[0]):
                u_raw.append(float("nan"))
                continue

            count = float(win - 1)
            sum_speed = float(speed_prefix[speed_end] - speed_prefix[speed_start])
            sum_speed_sq = float(speed_sq_prefix[speed_end] - speed_sq_prefix[speed_start])
            mu = sum_speed / count
            if mu <= eps:
                u_raw.append(0.0)
                continue
            var = max(0.0, (sum_speed_sq / count) - (mu * mu))
            cv = float(math.sqrt(var) / (mu + eps))
            u_raw.append(cv if np.isfinite(cv) else float("nan"))

        u_norm_vals = self._normalize_candidate_metric(u_raw, eps=eps)
        p_proxy = []
        for u_norm in u_norm_vals:
            if not np.isfinite(u_norm):
                p_proxy.append(float("nan"))
                continue
            p = float(np.exp(-self.horizon_inter_beta * max(0.0, u_norm)))
            p_proxy.append(float(np.clip(p, 0.0, 1.0)))
        return u_raw, np.asarray(u_norm_vals, dtype=np.float64).tolist(), p_proxy

    @staticmethod
    def _gaussian_kernel_weights(
        horizons: list[int], center: float, bandwidth: float, eps: float = 1e-12
    ) -> np.ndarray:
        vals = np.asarray(horizons, dtype=np.float64)
        if vals.size <= 0:
            return np.zeros((0,), dtype=np.float64)
        if not np.isfinite(center):
            return np.full((vals.size,), 1.0 / float(vals.size), dtype=np.float64)
        bw = max(float(bandwidth), eps)
        weights = np.exp(-((vals - float(center)) ** 2) / (2.0 * bw * bw))
        weight_sum = float(np.sum(weights))
        if weight_sum <= eps or not np.isfinite(weight_sum):
            return np.full((vals.size,), 1.0 / float(vals.size), dtype=np.float64)
        return weights / weight_sum

    def _apply_horizon_ts_update(
        self,
        horizons: list[int],
        q_mix_vals: list[float],
        chosen_idx: int,
        choose_mode: str,
        k_exec: int,
        expected_prob: np.ndarray | None = None,
    ) -> tuple[float, float]:
        if self.horizon_ts_update_mode == "independent":
            if choose_mode == "expected_round" and expected_prob is not None:
                feedback = float(
                    np.dot(
                        np.asarray(expected_prob, dtype=np.float64),
                        np.asarray(q_mix_vals, dtype=np.float64),
                    )
                )
                for i, s in enumerate(horizons):
                    prob_i = float(expected_prob[i]) if i < int(expected_prob.size) else 0.0
                    if prob_i <= 0.0:
                        continue
                    qa_i = float(q_mix_vals[i])
                    a, b = self._horizon_ts_state.get(int(s), [1.0, 1.0])
                    self._horizon_ts_state[int(s)] = [a + (prob_i * qa_i), b + (prob_i * (1.0 - qa_i))]
                return float(k_exec), feedback

            qa = float(q_mix_vals[chosen_idx])
            a, b = self._horizon_ts_state.get(int(k_exec), [1.0, 1.0])
            self._horizon_ts_state[int(k_exec)] = [a + qa, b + (1.0 - qa)]
            return float(k_exec), qa

        if choose_mode == "expected_round" and expected_prob is not None:
            feedback = float(
                np.dot(
                    np.asarray(expected_prob, dtype=np.float64),
                    np.asarray(q_mix_vals, dtype=np.float64),
                )
            )
        else:
            feedback = float(q_mix_vals[chosen_idx])
        feedback = float(np.clip(feedback, 0.0, 1.0))
        update_center = float(k_exec)
        weights = self._gaussian_kernel_weights(
            horizons,
            center=update_center,
            bandwidth=self.horizon_ts_kernel_bandwidth,
        )
        rho = self.horizon_ts_forget_rho
        eta = self.horizon_ts_update_eta
        for i, s in enumerate(horizons):
            a, b = self._horizon_ts_state.get(int(s), [1.0, 1.0])
            w_i = float(weights[i]) if i < int(weights.size) else 0.0
            incr = eta * w_i
            self._horizon_ts_state[int(s)] = [
                rho * a + incr * feedback,
                rho * b + incr * (1.0 - feedback),
            ]
        return update_center, feedback

    def _select_exec_k_horizon(
        self,
        v_step: np.ndarray,
        used_h: int,
        xt_step: np.ndarray | None = None,
        forced_k: int | None = None,
    ) -> tuple[int, dict]:
        horizons = [s for s in self.horizon_candidates if s <= int(used_h)]
        if not horizons:
            horizons = [int(used_h)]

        use_inter_signal = self.horizon_inter_use_speed_uniformity and (self.horizon_mix_mode in ("inter", "both"))
        if use_inter_signal and xt_step is not None:
            u_vals, u_norm_vals, p_vals = self._speed_uniformity_proxy_by_horizon(
                np.asarray(xt_step, dtype=np.float32),
                horizons,
            )
        else:
            u_vals = [float("nan")] * len(horizons)
            u_norm_vals = [float("nan")] * len(horizons)
            p_vals = [float("nan")] * len(horizons)

        ehh_by_horizon = self._ehh_curves_by_horizon_batched(
            np.asarray(v_step[:, :used_h, :], dtype=np.float32),
            horizons,
            h_full=used_h,
        )

        z_vals = []
        for s in horizons:
            e_vals, _ = ehh_by_horizon[int(s)]
            z = self._compute_z_from_ehh(e_vals, tau0_idx=0)
            z_vals.append(float(z))

        z_norm_vals = self._normalize_candidate_metric(z_vals)
        q_intra_vals = []
        q_mix_vals = []
        score_vals = []
        for i, s in enumerate(horizons):
            z = z_vals[i]
            z_norm = float(z_norm_vals[i]) if i < int(z_norm_vals.size) else float("nan")
            if np.isfinite(z_norm):
                q_intra = float(np.exp(-self.horizon_intra_alpha * max(0.0, z_norm)))
            else:
                q_intra = float("nan")
            p_inter = p_vals[i]
            if self.horizon_mix_mode == "intra":
                q_mix = q_intra
            elif self.horizon_mix_mode == "inter":
                if np.isfinite(p_inter):
                    q_mix = float(np.clip(p_inter, 0.0, 1.0))
                else:
                    q_mix = 0.5 if self.horizon_inter_fallback == "neutral" else q_intra
            else:
                if np.isfinite(p_inter):
                    q_mix = float(
                        (1.0 - self.horizon_inter_weight) * q_intra
                        + self.horizon_inter_weight * float(np.clip(p_inter, 0.0, 1.0))
                    )
                else:
                    q_mix = (
                        float((1.0 - self.horizon_inter_weight) * q_intra + self.horizon_inter_weight * 0.5)
                        if self.horizon_inter_fallback == "neutral"
                        else q_intra
                    )
            a, b = self._horizon_ts_state.get(int(s), [1.0, 1.0])
            theta = float(self._horizon_ts_rng.beta(a, b))
            q_intra_vals.append(float(q_intra))
            q_mix_vals.append(float(q_mix))
            score_vals.append(float(theta * q_mix))

        score_arr = np.asarray(score_vals, dtype=np.float64)
        score_prob = self._normalize_score_distribution(score_arr, temp=1.0)
        expected_prob = score_prob
        if self.horizon_exec_mode == "expected_round":
            expected_prob = self._normalize_score_distribution(score_arr, temp=self.horizon_expected_temp)
        expected_horizon = float(np.dot(expected_prob, np.asarray(horizons, dtype=np.float64))) if expected_prob.size > 0 else None

        choose_mode = "thompson"
        update_center = None
        update_feedback = None
        if forced_k is not None:
            k_exec = int(np.clip(int(forced_k), 1, int(used_h)))
            choose_mode = "fixed_exec_k"
        elif float(self._horizon_ts_rng.random()) < self.horizon_ts_epsilon:
            chosen_idx = int(self._horizon_ts_rng.integers(0, len(horizons)))
            k_exec = int(horizons[chosen_idx])
            choose_mode = "epsilon_random"
            update_center, update_feedback = self._apply_horizon_ts_update(
                horizons=horizons,
                q_mix_vals=q_mix_vals,
                chosen_idx=chosen_idx,
                choose_mode=choose_mode,
                k_exec=k_exec,
            )
        elif self.horizon_exec_mode == "expected_round":
            if expected_horizon is None or not np.isfinite(expected_horizon):
                expected_horizon = float(np.mean(np.asarray(horizons, dtype=np.float64)))
            k_exec = int(np.clip(int(round(expected_horizon)), 1, int(used_h)))
            choose_mode = "expected_round"
            update_center, update_feedback = self._apply_horizon_ts_update(
                horizons=horizons,
                q_mix_vals=q_mix_vals,
                chosen_idx=-1,
                choose_mode=choose_mode,
                k_exec=k_exec,
                expected_prob=expected_prob,
            )
        else:
            chosen_idx = int(np.argmax(score_arr))
            k_exec = int(horizons[chosen_idx])
            choose_mode = "thompson"
            update_center, update_feedback = self._apply_horizon_ts_update(
                horizons=horizons,
                q_mix_vals=q_mix_vals,
                chosen_idx=chosen_idx,
                choose_mode=choose_mode,
                k_exec=k_exec,
            )

        return k_exec, {
            "horizons": np.asarray(horizons, dtype=np.int32),
            "z_intra": np.asarray(z_vals, dtype=np.float32),
            "z_intra_norm": np.asarray(z_norm_vals, dtype=np.float32),
            "u_inter_raw": np.asarray(u_vals, dtype=np.float32),
            "u_inter_norm": np.asarray(u_norm_vals, dtype=np.float32),
            "q_intra": np.asarray(q_intra_vals, dtype=np.float32),
            "p_inter": np.asarray(p_vals, dtype=np.float32),
            "q_mix": np.asarray(q_mix_vals, dtype=np.float32),
            "scores": np.asarray(score_vals, dtype=np.float32),
            "horizon_ts_update_mode": self.horizon_ts_update_mode,
            "horizon_ts_kernel_bandwidth": float(self.horizon_ts_kernel_bandwidth),
            "horizon_ts_forget_rho": float(self.horizon_ts_forget_rho),
            "horizon_ts_update_eta": float(self.horizon_ts_update_eta),
            "choose_mode": choose_mode,
            "chosen_horizon": int(k_exec),
            "expected_horizon": expected_horizon,
            "update_center": update_center,
            "update_feedback": update_feedback,
        }

    @staticmethod
    def _parse_trace_payload(trace_payload):
        if trace_payload is None:
            return None
        return {key: np.asarray(value, dtype=np.float32) for key, value in trace_payload.items()}

    def _build_query(self, obs):
        head_view = obs["observation"]["head_camera"]["rgb"]
        left_view = obs["observation"]["left_camera"]["rgb"]
        right_view = obs["observation"]["right_camera"]["rgb"]
        left_ee = np.expand_dims(np.array(obs["endpose"]["left_endpose"]), axis=0)
        right_ee = np.expand_dims(np.array(obs["endpose"]["right_endpose"]), axis=0)
        left_grip = np.expand_dims(np.array(obs["endpose"]["left_gripper"]), axis=0)
        right_grip = np.expand_dims(np.array(obs["endpose"]["right_gripper"]), axis=0)
        left_grip = 1 - left_grip * 2
        right_grip = 1 - right_grip * 2
        abs_eef = np.concatenate(
            [
                left_ee[:, :3],
                quat_to_rotate6D(left_ee[:, 3:]),
                left_grip[:, None],
                right_ee[:, :3],
                quat_to_rotate6D(right_ee[:, 3:]),
                right_grip[:, None],
            ],
            axis=-1,
        )

        query = {
            "domain_id": 6,
            "proprio": json_numpy.dumps(abs_eef.squeeze(0)),
            "language_instruction": self.instruction,
            "image0": json_numpy.dumps(head_view),
            "image1": json_numpy.dumps(left_view),
            "image2": json_numpy.dumps(right_view),
            "steps": self.num_denoise_steps,
        }
        if self.trace_enabled:
            query["return_denoise_trace"] = True
        return query

    def step(self, obs):
        query = self._build_query(obs)
        response = requests.post(self.url, json=query, timeout=180)
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(payload["error"])

        action = np.asarray(payload["action"], dtype=np.float32)
        if action.ndim != 2:
            raise RuntimeError(f"Expected action chunk with shape [H, D], got {action.shape}")

        trace = self._parse_trace_payload(payload.get("denoise_trace"))
        if self.trace_enabled and trace is None:
            raise RuntimeError("Server did not return denoise_trace. Restart deploy after updating server code.")

        used_h = int(action.shape[0])
        k_exec = used_h
        horizon_info = None
        t0 = time.time() if trace is not None else None
        if self.eval_type == "horizon":
            forced_k = self.fixed_exec_k if self.fixed_exec_k > 0 else None
            k_exec, horizon_info = self._select_exec_k_horizon(
                trace["v"],
                used_h,
                xt_step=action,
                forced_k=forced_k,
            )
        elif self.fixed_exec_k > 0:
            k_exec = int(np.clip(self.fixed_exec_k, 1, used_h))

        self.last_exec_k = int(k_exec)

        if self.dump_denoise and trace is not None:
            xt_log = np.array(action, dtype=np.float32, copy=True)
            xt_internal_log = np.array(action, dtype=np.float32, copy=True)
            if k_exec < xt_log.shape[0]:
                xt_log[k_exec:, :] = 0.0
                xt_internal_log[k_exec:, :] = 0.0
            self.denoise_logger.log_step(
                v=trace["v"],
                ds=trace.get("ds"),
                x0=trace.get("x0"),
                xt=xt_log,
                xt_internal=xt_internal_log,
                k_exec=k_exec if self.eval_type == "horizon" else None,
                chosen_horizons=None if horizon_info is None else horizon_info["horizons"],
                intra_scores=None if horizon_info is None else horizon_info["q_intra"],
                inter_scores=None if horizon_info is None else horizon_info["p_inter"],
                mixed_scores=None if horizon_info is None else horizon_info["q_mix"],
            )
        if t0 is not None:
            self._denoise_overhead_seconds += float(time.time() - t0)
        self._update_run_score_stats(horizon_info)

        return {
            "action": action,
            "k_exec": int(k_exec),
            "trace": trace,
            "horizon_info": horizon_info,
        }


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except Exception as exc:
        raise SystemExit("No such task") from exc
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def load_env(task_name, task_config):
    configs_path = "task_config"
    task = class_decorator(task_name)

    config_path = os.path.join(configs_path, f"{task_config}.yml")
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(configs_path, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_cfg):
        robot_file = embodiment_types[embodiment_cfg]["file_path"]
        if robot_file is None:
            raise ValueError("missing embodiment files")
        return robot_file

    with open(os.path.join(configs_path, "_camera_config.yml"), "r", encoding="utf-8") as f:
        camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_camera_type]["h"]
    args["head_camera_w"] = camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("number of embodiment config parameters should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["embodiment_name"] = "+".join(embodiment_type) if len(embodiment_type) > 1 else embodiment_type[0]
    args["task_config"] = task_config

    return task, args


def _start_continuous_video(env, output_path: Path, video_size: str, fps: int = 10):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            video_size,
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            str(output_path),
        ],
        stdin=subprocess.PIPE,
    )
    env._set_eval_video_ffmpeg(ffmpeg)


def _close_continuous_video(env):
    if getattr(env, "eval_video_ffmpeg", None) is None:
        return
    try:
        env._del_eval_video_ffmpeg()
    except Exception as exc:
        print(f"Warning: failed to close eval video cleanly: {exc}")


def _convert_action_chunk_to_rollout(actions: np.ndarray) -> np.ndarray:
    left_xyz = actions[:, :3]
    left_rotate6d = actions[:, 3:9]
    left_gripper = actions[:, 9:10]
    left_quat = rotate6D_to_quat(left_rotate6d)
    left_grip = 1 - 2 * (left_gripper > 0.7)
    left_new = np.concatenate([left_xyz, left_quat, left_grip], axis=1)

    right_xyz = actions[:, 10:13]
    right_rotate6d = actions[:, 13:19]
    right_quat = rotate6D_to_quat(right_rotate6d)
    right_gripper = actions[:, 19:20]
    right_grip = 1 - 2 * (right_gripper > 0.7)
    right_new = np.concatenate([right_xyz, right_quat, right_grip], axis=1)

    return np.concatenate([left_new, right_new], axis=1)


def _rollout(env, policy, video_mode="continuous"):
    success_flag = False
    error_flag = False
    images = [] if video_mode == "legacy" else None
    episode_steps = 0
    success_step = None
    start_time = time.time()
    step_limit = int(getattr(env, "step_lim", 0) or 1000)
    total_env_step_time = 0.0  # 累计环境交互时间

    env._update_render()
    if env.render_freq:
        env.viewer.render()
    env.actor_pose = True
    obs = env.get_obs()

    while int(getattr(env, "take_action_cnt", 0)) < step_limit:
        if getattr(env, "eval_success", False):
            success_flag = True
            success_step = int(getattr(env, "take_action_cnt", episode_steps))
            break
        infer_output = policy.step(obs)
        actions = np.asarray(infer_output["action"], dtype=np.float32)
        k_exec = int(np.clip(infer_output["k_exec"], 1, int(actions.shape[0])))
        remaining_budget = max(0, step_limit - int(getattr(env, "take_action_cnt", 0)))
        if remaining_budget <= 0:
            break
        exec_budget = min(k_exec, int(actions.shape[0]), remaining_budget)
        if exec_budget <= 0:
            break
        rollout_action = _convert_action_chunk_to_rollout(actions[:exec_budget])

        executed_count = 0
        for action in tqdm(rollout_action):
            step_start = time.time()  # 开始计时
            
            env.take_action(action, action_type="ee")
            executed_count += 1
            episode_steps += 1

            obs = env.get_obs()
            
            step_end = time.time()  # 结束计时
            total_env_step_time += (step_end - step_start)  # 累加环境交互时间
            obs["endpose"]["left_endpose"] = list(action[:7].reshape(7,))
            obs["endpose"]["right_endpose"] = list(action[8:-1].reshape(7,))

            if video_mode == "legacy":
                images.append(obs["observation"]["head_camera"]["rgb"])

            if env.check_success():
                success_flag = True
                success_step = int(getattr(env, "take_action_cnt", episode_steps))
                break
            if env.actor_pose is False:
                print("false actor_pose")
                error_flag = True
                break
            if getattr(env, "eval_success", False):
                success_flag = True
                success_step = int(getattr(env, "take_action_cnt", episode_steps))
                break

        policy.append_executed_actions(actions[:executed_count])

        if error_flag:
            print("\nfail due to false actor_pose!")
            return {
                "status": 0,
                "images": images,
                "episode_steps": episode_steps,
                "success_step": success_step,
                "run_time_seconds": time.time() - start_time,
                "env_step_time_seconds": total_env_step_time,
            }
        if success_flag:
            print("\nsuccess!")
            return {
                "status": 1,
                "images": images,
                "episode_steps": episode_steps,
                "success_step": success_step,
                "run_time_seconds": time.time() - start_time,
                "env_step_time_seconds": total_env_step_time,
            }
        if env.actor_pose is False:
            print("false actor_pose2")
            break
        if getattr(env, "eval_success", False):
            success_flag = True
            success_step = int(getattr(env, "take_action_cnt", episode_steps))
            break
        env._update_render()

    if success_flag:
        print("\nsuccess!")
        return {
            "status": 1,
            "images": images,
            "episode_steps": episode_steps,
            "success_step": success_step,
            "run_time_seconds": time.time() - start_time,
            "env_step_time_seconds": total_env_step_time,
        }

    print("\nfail!")
    return {
        "status": 0,
        "images": images,
        "episode_steps": episode_steps,
        "success_step": success_step,
        "run_time_seconds": time.time() - start_time,
        "env_step_time_seconds": total_env_step_time,
    }


def eval_episodes(
    task_name,
    task_config,
    policy,
    test_num=10,
    seed=0,
    eval_log_dir=None,
    instruction_type=None,
    video_mode="continuous",
    eval_tag="",
    ckpt_tag="X-VLA-RoboTwin2",
):
    eval_tag = str(eval_tag).strip() or str(task_config)
    ckpt_tag = str(ckpt_tag).strip() or "X-VLA-RoboTwin2"
    run_layout = make_run_layout(eval_log_dir, task_name, task_config, ckpt_tag, eval_tag)
    run_dir = run_layout["run_dir"]
    info_dir = run_layout["info_dir"]
    episodes_dir = run_layout["episodes_dir"]
    summary_csv = run_layout["summary_csv"]
    task_name_with_setting = run_layout["task_name_with_setting"]
    timestamp = run_layout["timestamp"]

    print("save to", str(run_dir))

    task_env, args = load_env(task_name, task_config)
    video_size = f"{args['head_camera_w']}x{args['head_camera_h']}"
    policy.reset_run_score_state()

    st_seed = 2000 * (1 + seed)
    expert_check = True
    task_env.suc = 0
    task_env.test_num = 0
    now_id = 0
    succ_seed = 0
    now_seed = st_seed
    clear_cache_freq = args["clear_cache_freq"]

    args["policy_name"] = "V4"
    args["eval_mode"] = True
    args["render_freq"] = 0
    args["ckpt_setting"] = "60k"

    point = 0.0
    total_run_time_seconds = 0.0
    total_episode_steps = 0
    total_env_step_time_seconds = 0.0
    while succ_seed < test_num:
        args.pop("eval_video_save_dir", None)
        render_freq = args["render_freq"]
        print("Running test", now_id)

        if expert_check:
            try:
                task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                task_env.play_once()
                task_env.close_env()
            except Exception as exc:
                print("Error: ", exc)
                task_env.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue

        if (not expert_check) or (task_env.plan_success and task_env.check_success()):
            succ_seed += 1
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq
        if video_mode == "continuous":
            args["eval_video_save_dir"] = episodes_dir

        policy.reset_episode_state()
        task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        instruction = args["task_name"].replace("_", " ")
        print("instruction:", instruction)
        policy.set_instruction(instruction=instruction)
        policy.prepare_episode_logging(episodes_dir, now_id)

        rollout_result = None
        temp_video_path = None
        try:
            if video_mode == "continuous":
                temp_video_path = episodes_dir / f"episode{now_id}.mp4"
                _start_continuous_video(task_env, temp_video_path, video_size=video_size, fps=10)
            rollout_result = _rollout(task_env, policy, video_mode=video_mode)
        except Exception as exc:
            print("Rollout error:", exc)
            traceback.print_exc()
        finally:
            if video_mode == "continuous":
                _close_continuous_video(task_env)

        status = None if rollout_result is None else rollout_result["status"]
        if video_mode == "continuous" and temp_video_path is not None and temp_video_path.exists():
            status_suffix = str(status) if status is not None else "error"
            final_video_path = episodes_dir / f"{now_id}_{status_suffix}.mp4"
            if final_video_path.exists():
                final_video_path.unlink()
            os.replace(temp_video_path, final_video_path)

        policy.finalize_episode_logging(
            success=(None if status is None else bool(status)),
            success_step=(None if rollout_result is None else rollout_result["success_step"]),
            episode_steps=(None if rollout_result is None else rollout_result["episode_steps"]),
            run_time_seconds=(None if rollout_result is None else rollout_result["run_time_seconds"]),
        )

        if status is None:
            task_env.close_env()
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        if status == 1:
            task_env.suc += 1
        total_run_time_seconds += float(rollout_result["run_time_seconds"])
        total_episode_steps += int(rollout_result["episode_steps"])
        total_env_step_time_seconds += float(rollout_result["env_step_time_seconds"])

        if video_mode == "legacy":
            save_path = episodes_dir / f"{now_id}_{status}.mp4"
            save_video(str(save_path), rollout_result["images"])

        metrics = {f"sim/{task_name}": status}
        save_path = info_dir / "results.json"
        _log_results(metrics, str(save_path))

        now_id += 1
        task_env.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))
        if task_env.render_freq:
            task_env.viewer.close()

        task_env.test_num += 1
        point = round(task_env.suc / task_env.test_num * 100, 1)
        print(f"Success rate: {point}%")
        now_seed += 1

    metrics = {f"sim/{task_name}": point}
    _log_results(metrics, str(info_dir / "perc.json"))

    total_success_count = int(task_env.suc)
    expected_rollouts = int(task_env.test_num)
    success_rate = float(total_success_count / expected_rollouts) if expected_rollouts > 0 else 0.0
    run_score_averages = policy.get_run_score_averages()

    # 计算平均时间
    mean_step_time_seconds = total_run_time_seconds / total_episode_steps if total_episode_steps > 0 else 0.0
    mean_env_step_time_seconds = total_env_step_time_seconds / total_episode_steps if total_episode_steps > 0 else 0.0

    info = {
        "timestamp": timestamp,
        "task_name": task_name,
        "task_config": task_config,
        "task_name_with_setting": task_name_with_setting,
        "base_model_type": BASE_MODEL_TYPE,
        "ckpt_tag": ckpt_tag,
        "eval_tag": eval_tag,
        "eval_type": policy.eval_type,
        "video_mode": video_mode,
        "seed": int(seed),
        "expected_rollouts": expected_rollouts,
        "success_count": total_success_count,
        "success_rate": success_rate,
        "total_episode_steps": total_episode_steps,
        "total_run_time_seconds": total_run_time_seconds,
        "mean_step_time_seconds": mean_step_time_seconds,
        "mean_env_step_time_seconds": mean_env_step_time_seconds,
        "model_path": ckpt_tag,
        "fixed_exec_k": int(policy.fixed_exec_k),
        "dump_denoise": int(policy.dump_denoise),
        "avg_q_intra": run_score_averages["avg_q_intra"],
        "avg_p_inter": run_score_averages["avg_p_inter"],
        "avg_q_mix": run_score_averages["avg_q_mix"],
        "avg_q_intra_contrib": run_score_averages["avg_q_intra_contrib"],
        "avg_p_inter_contrib": run_score_averages["avg_p_inter_contrib"],
        "avg_z_intra_raw": run_score_averages["avg_z_intra_raw"],
        "avg_z_intra_norm": run_score_averages["avg_z_intra_norm"],
        "avg_u_inter_raw": run_score_averages["avg_u_inter_raw"],
        "avg_u_inter_norm": run_score_averages["avg_u_inter_norm"],
    }
    if policy.eval_type == "horizon":
        info.update(
            {
                "horizon_policy": policy._horizon_policy_name(),
                "horizon_exec_mode": policy.horizon_exec_mode,
                "horizon_candidates": policy.horizon_candidates,
                "horizon_expected_temp": policy.horizon_expected_temp,
                "horizon_intra_alpha": policy.horizon_intra_alpha,
                "horizon_intra_z_mode": policy.horizon_intra_z_mode,
                "horizon_intra_tau_window": policy.horizon_intra_tau_window,
                "horizon_inter_weight": policy.horizon_inter_weight,
                "horizon_inter_beta": policy.horizon_inter_beta,
                "horizon_inter_window": policy.horizon_inter_window,
                "horizon_ts_update_mode": policy.horizon_ts_update_mode,
                "horizon_ts_kernel_bandwidth": policy.horizon_ts_kernel_bandwidth,
                "horizon_ts_forget_rho": policy.horizon_ts_forget_rho,
                "horizon_ts_update_eta": policy.horizon_ts_update_eta,
            }
        )
    write_run_info(info_dir / "info.json", info)
    append_summary_row(
        summary_csv,
        {
            "task_name": task_name,
            "task_config": task_config,
            "task_name_with_setting": task_name_with_setting,
            "eval_type": policy.eval_type,
            "run_dir": str(run_dir),
            "info_json": str(info_dir / "info.json"),
            "video_dir": str(episodes_dir),
            "success_rate": success_rate,
            "success_count": total_success_count,
            "expected_rollouts": expected_rollouts,
            "timestamp": timestamp,
            "avg_q_intra": run_score_averages["avg_q_intra"],
            "avg_p_inter": run_score_averages["avg_p_inter"],
            "avg_q_mix": run_score_averages["avg_q_mix"],
            "avg_q_intra_contrib": run_score_averages["avg_q_intra_contrib"],
            "avg_p_inter_contrib": run_score_averages["avg_p_inter_contrib"],
            "avg_z_intra_raw": run_score_averages["avg_z_intra_raw"],
            "avg_z_intra_norm": run_score_averages["avg_z_intra_norm"],
            "avg_u_inter_raw": run_score_averages["avg_u_inter_raw"],
            "avg_u_inter_norm": run_score_averages["avg_u_inter_norm"],
        },
    )

    return now_seed, task_env.suc


def save_video(output_path, frames, fps=30):
    print("saving video to", output_path)
    imageio.mimsave(output_path, frames, fps=fps)


def _log_results(metrics, log_path):
    with open(log_path, "a+", encoding="utf-8") as f:
        line = json.dumps(metrics)
        f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on multistep sequences with language goals."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Your client host ip")
    parser.add_argument("--port", default="8001", help="Your client port")
    parser.add_argument(
        "--eval_log_dir",
        default=_ct_path('CHUNKTRUST_RESULTS_ROOT', 'robotwin2.0/eval_results'),
        type=str,
        help="Where to log the evaluation results.",
    )
    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    parser.add_argument("--num_episodes", default=1000, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--task_name", type=str, required=True, help="Name of the task (envs.<task>)")
    parser.add_argument("--task_config", type=str, required=True, help="Task config name (without .yml)")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save the output video")
    parser.add_argument("--instruction_type", type=str, required=False, help="Instruction override")
    parser.add_argument(
        "--video_mode",
        type=str,
        default="continuous",
        choices=["continuous", "legacy"],
        help="Video recording mode: continuous uses RoboTwin ffmpeg recording, legacy saves sparse frames.",
    )
    parser.add_argument("--eval_type", type=str, default="default", choices=["default", "horizon"])
    parser.add_argument("--fixed_exec_k", type=int, default=0)
    parser.add_argument("--dump_denoise", type=int, default=0)
    parser.add_argument("--denoise_npz_compress", type=int, default=0)
    parser.add_argument("--eval_tag", type=str, default="")
    parser.add_argument("--ckpt_tag", type=str, default="X-VLA-RoboTwin2")
    parser.add_argument("--horizon_candidates", type=str, default="10,20,30")
    parser.add_argument("--horizon_intra_alpha", type=float, default=4.0)
    parser.add_argument("--horizon_ts_epsilon", type=float, default=0.05)
    parser.add_argument("--horizon_ts_seed", type=str, default="seed")
    parser.add_argument("--horizon_ts_update_mode", type=str, default="kernel_forget", choices=["independent", "kernel_forget"])
    parser.add_argument("--horizon_ts_kernel_bandwidth", type=float, default=10.0)
    parser.add_argument("--horizon_ts_forget_rho", type=float, default=0.99)
    parser.add_argument("--horizon_ts_update_eta", type=float, default=1.0)
    parser.add_argument("--horizon_exec_mode", type=str, default="expected_round", choices=["thompson", "expected_round"])
    parser.add_argument("--horizon_expected_temp", type=float, default=1.0)
    parser.add_argument("--horizon_intra_cut_t", type=float, default=0.25)
    parser.add_argument(
        "--horizon_intra_z_mode",
        type=str,
        default="window_rms",
        choices=["tau0_rms", "window_rms", "trend"],
    )
    parser.add_argument("--horizon_intra_tau_window", type=int, default=3)
    parser.add_argument("--horizon_mix_mode", type=str, default="both", choices=["intra", "inter", "both"])
    parser.add_argument("--horizon_inter_use_speed_uniformity", type=int, default=1)
    parser.add_argument("--horizon_inter_window", type=int, default=40)
    parser.add_argument("--horizon_inter_weight", type=float, default=0.5)
    parser.add_argument("--horizon_inter_beta", type=float, default=4.0)
    parser.add_argument("--horizon_inter_fallback", type=str, default="ehh_only", choices=["ehh_only", "neutral"])
    args = parser.parse_args()

    model = ClientModel(
        host=args.host,
        port=args.port,
        seed=args.seed,
        eval_type=args.eval_type,
        fixed_exec_k=args.fixed_exec_k,
        dump_denoise=args.dump_denoise,
        denoise_npz_compress=args.denoise_npz_compress,
        horizon_candidates=args.horizon_candidates,
        horizon_intra_alpha=args.horizon_intra_alpha,
        horizon_ts_epsilon=args.horizon_ts_epsilon,
        horizon_ts_seed=args.horizon_ts_seed,
        horizon_ts_update_mode=args.horizon_ts_update_mode,
        horizon_ts_kernel_bandwidth=args.horizon_ts_kernel_bandwidth,
        horizon_ts_forget_rho=args.horizon_ts_forget_rho,
        horizon_ts_update_eta=args.horizon_ts_update_eta,
        horizon_exec_mode=args.horizon_exec_mode,
        horizon_expected_temp=args.horizon_expected_temp,
        horizon_intra_cut_t=args.horizon_intra_cut_t,
        horizon_intra_z_mode=args.horizon_intra_z_mode,
        horizon_intra_tau_window=args.horizon_intra_tau_window,
        horizon_mix_mode=args.horizon_mix_mode,
        horizon_inter_use_speed_uniformity=args.horizon_inter_use_speed_uniformity,
        horizon_inter_window=args.horizon_inter_window,
        horizon_inter_weight=args.horizon_inter_weight,
        horizon_inter_beta=args.horizon_inter_beta,
        horizon_inter_fallback=args.horizon_inter_fallback,
    )

    if args.task_name == "all":
        for task in tqdm(ALL_TASKS):
            print(f"Evaluating task {task} for {args.num_episodes} episodes...")
            eval_episodes(
                task_name=task,
                task_config=args.task_config,
                policy=model,
                seed=args.seed,
                test_num=args.num_episodes,
                eval_log_dir=args.eval_log_dir,
                instruction_type=args.instruction_type,
                video_mode=args.video_mode,
                eval_tag=args.eval_tag,
                ckpt_tag=args.ckpt_tag,
            )
    else:
        print(f"Evaluating task {args.task_name} for {args.num_episodes} episodes...")
        eval_episodes(
            task_name=args.task_name,
            task_config=args.task_config,
            policy=model,
            seed=args.seed,
            test_num=args.num_episodes,
            eval_log_dir=args.eval_log_dir,
            instruction_type=args.instruction_type,
            video_mode=args.video_mode,
            eval_tag=args.eval_tag,
            ckpt_tag=args.ckpt_tag,
        )


if __name__ == "__main__":
    main()
