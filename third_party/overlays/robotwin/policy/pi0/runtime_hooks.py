from __future__ import annotations
from chunktrust.paths import resolve_legacy_path as _ct_path

import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np


_HORIZON_RENAMED_OVERRIDES = {
    "cut_t": "horizon_intra_cut_t",
    "stride_candidates": "horizon_candidates",
    "stride_ts_alpha": "horizon_intra_alpha",
    "stride_ts_epsilon": "horizon_ts_epsilon",
    "stride_ts_seed": "horizon_ts_seed",
    "stride_ts_update_mode": "horizon_ts_update_mode",
    "stride_ts_kernel_bandwidth": "horizon_ts_kernel_bandwidth",
    "stride_ts_forget_rho": "horizon_ts_forget_rho",
    "stride_ts_update_eta": "horizon_ts_update_eta",
    "stride_z_mode": "horizon_intra_z_mode",
    "stride_exec_mode": "horizon_exec_mode",
    "stride_expected_temp": "horizon_expected_temp",
    "stride_tau_window": "horizon_intra_tau_window",
    "stride_signal_mode": "horizon_mix_mode",
    "stride_use_speed_uniformity": "horizon_inter_use_speed_uniformity",
    "stride_speed_window": "horizon_inter_window",
    "stride_speed_weight": "horizon_inter_weight",
    "stride_speed_beta": "horizon_inter_beta",
    "stride_speed_fallback": "horizon_inter_fallback",
    "stride_rms_threshold": None,
    "stride_jump_threshold": None,
    "stride_summary_csv": None,
}


def normalize_usr_args(usr_args: dict) -> dict:
    out = dict(usr_args)
    if str(out.get("eval_type", "default")) == "logistics_prior":
        raise RuntimeError("eval_type=logistics_prior has been removed from the pi0 eval runtime path.")

    if "horizon_intra_alpha" not in out and "horizon_ts_alpha" in out:
        out["horizon_intra_alpha"] = out["horizon_ts_alpha"]
    out.setdefault("horizon_candidates", "10,20,30,40,50")
    out.setdefault("horizon_intra_alpha", 4.0)
    out.setdefault("horizon_ts_epsilon", 0.05)
    out.setdefault("horizon_ts_seed", int(out.get("seed", 0)))
    out.setdefault("horizon_ts_update_mode", "kernel_forget")
    out.setdefault("horizon_ts_kernel_bandwidth", 10.0)
    out.setdefault("horizon_ts_forget_rho", 0.99)
    out.setdefault("horizon_ts_update_eta", 1.0)
    out.setdefault("horizon_intra_z_mode", "window_rms")
    out.setdefault("horizon_exec_mode", "expected_round")
    out.setdefault("horizon_expected_temp", 1.0)
    out.setdefault("horizon_intra_tau_window", 3)
    out.setdefault("horizon_inter_use_speed_uniformity", 1)
    out.setdefault("horizon_mix_mode", "both")
    out.setdefault("horizon_inter_window", 60)
    out.setdefault("horizon_inter_weight", 0.5)
    out.setdefault("horizon_inter_beta", 4.0)
    out.setdefault("horizon_inter_fallback", "ehh_only")
    out.setdefault("horizon_intra_cut_t", 0.25)
    out.setdefault("fixed_exec_k", 0)
    out.setdefault("denoise_npz_compress", 0)
    return out


def validate_overrides(config: dict, overrides: dict) -> None:
    details = []
    renamed = sorted(set(overrides) & set(_HORIZON_RENAMED_OVERRIDES))
    for key in renamed:
        replacement = _HORIZON_RENAMED_OVERRIDES[key]
        if replacement:
            details.append(f"--{key}: removed for pi0; use --{replacement} instead.")
        else:
            details.append(f"--{key}: removed for pi0; no replacement.")
    if overrides.get("eval_type") == "stride":
        details.append("--eval_type stride: removed for pi0; use --eval_type horizon instead.")
    if details:
        raise SystemExit("Removed override(s):\n" + "\n".join(details))


def _sanitize_path_tag(raw, fallback: str) -> str:
    s = str(raw).strip() if raw is not None else ""
    if not s:
        s = str(fallback)
    s = s.replace(os.sep, "_").replace("/", "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._+=,@:-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._-")
    return s or str(fallback)


def _infer_base_model_type(train_config_name: str) -> str:
    if not train_config_name:
        return "unknown"
    parts = train_config_name.split("_")
    return parts[0] if parts else "unknown"


def _get_eval_setting_suffix(task_config: str) -> str:
    if task_config == "demo_clean":
        return "clean"
    if task_config == "demo_randomized":
        return "randomized"
    return task_config.replace("demo_", "")


def resolve_run_layout(
    usr_args: dict,
    task_name: str,
    task_config: str,
    ckpt_setting: str,
    now: datetime,
) -> dict:
    default_result_root = _ct_path('CHUNKTRUST_RESULTS_ROOT', 'robotwin2.0/eval_results')
    result_root = Path(str(os.environ.get("RESULT_ROOT", default_result_root))).expanduser()
    train_config_name = usr_args.get("train_config_name", usr_args.get("policy_name", "pi0"))
    base_model_type = _infer_base_model_type(train_config_name)
    eval_tag = _sanitize_path_tag(usr_args.get("eval_tag", task_config), task_config)
    ckpt_tag = _sanitize_path_tag(train_config_name, train_config_name)
    task_name_with_setting = f"{task_name}_{_get_eval_setting_suffix(task_config)}"
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = result_root / task_name_with_setting / base_model_type / ckpt_tag / eval_tag / timestamp
    return {
        "save_dir": save_dir,
        "info_dir": save_dir / "info",
        "episodes_dir": save_dir / "episodes",
        "base_model_type": base_model_type,
        "eval_tag": eval_tag,
        "ckpt_tag": ckpt_tag,
        "save_dir_layout": "task/base_model_type/ckpt_tag/eval_tag/timestamp",
    }


def _run_episodes_dir(run_dir: Path) -> Path:
    d = run_dir / "episodes"
    return d if d.exists() else run_dir


def collect_step_timing_stats(run_dir: Path) -> dict:
    npz_files = sorted(_run_episodes_dir(run_dir).glob("episode*.npz"))
    if not npz_files:
        return {
            "episodes_with_timing": 0,
            "episodes_with_step_count": 0,
            "total_episode_steps": 0,
            "mean_step_time_seconds_weighted": None,
            "mean_step_time_seconds_unweighted": None,
            "mean_step_denoise_overhead_seconds_weighted": None,
            "mean_step_denoise_overhead_seconds_unweighted": None,
            "mean_step_policy_infer_seconds_weighted": None,
            "mean_step_policy_infer_seconds_unweighted": None,
            "mean_step_horizon_metric_compute_seconds_weighted": None,
            "mean_step_horizon_metric_compute_seconds_unweighted": None,
            "mean_step_horizon_selection_seconds_weighted": None,
            "mean_step_horizon_selection_seconds_unweighted": None,
        }

    def _read_optional_total(data: np.lib.npyio.NpzFile, key: str) -> float | None:
        if key not in data.files:
            return None
        raw = np.asarray(data[key], dtype=np.float64).reshape(-1)
        if raw.size == 0:
            return None
        value = float(raw[0])
        return value if np.isfinite(value) else None

    total_time_sum = 0.0
    overhead_time_sum = 0.0
    policy_infer_time_sum = 0.0
    horizon_metric_time_sum = 0.0
    horizon_selection_time_sum = 0.0
    total_steps_sum = 0
    per_ep_step_time = []
    per_ep_overhead_step_time = []
    per_ep_policy_infer_step_time = []
    per_ep_horizon_metric_step_time = []
    per_ep_horizon_selection_step_time = []
    episodes_with_timing = 0
    episodes_with_step_count = 0
    has_policy_infer_timing = False
    has_horizon_metric_timing = False
    has_horizon_selection_timing = False

    for npz_path in npz_files:
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                if "run_time_seconds" not in data.files:
                    continue
                rt = np.asarray(data["run_time_seconds"], dtype=np.float64).reshape(-1)
                if rt.size == 0:
                    continue
                total_t = float(rt[0])
                overhead_t = float(rt[1]) if rt.size >= 2 else 0.0
                if not np.isfinite(total_t):
                    continue
                episodes_with_timing += 1
                if "episode_steps" not in data.files:
                    continue
                steps = int(np.asarray(data["episode_steps"]).item())
                if steps <= 0:
                    continue
                episodes_with_step_count += 1
                policy_infer_t = _read_optional_total(data, "policy_infer_seconds_total")
                horizon_metric_t = _read_optional_total(data, "horizon_metric_compute_seconds_total")
                horizon_selection_t = _read_optional_total(data, "horizon_selection_seconds_total")
                total_time_sum += total_t
                overhead_time_sum += overhead_t if np.isfinite(overhead_t) else 0.0
                if policy_infer_t is not None:
                    has_policy_infer_timing = True
                    policy_infer_time_sum += policy_infer_t
                if horizon_metric_t is not None:
                    has_horizon_metric_timing = True
                    horizon_metric_time_sum += horizon_metric_t
                if horizon_selection_t is not None:
                    has_horizon_selection_timing = True
                    horizon_selection_time_sum += horizon_selection_t
                total_steps_sum += steps
                per_ep_step_time.append(total_t / steps)
                per_ep_overhead_step_time.append((overhead_t / steps) if np.isfinite(overhead_t) else 0.0)
                if policy_infer_t is not None:
                    per_ep_policy_infer_step_time.append(policy_infer_t / steps)
                if horizon_metric_t is not None:
                    per_ep_horizon_metric_step_time.append(horizon_metric_t / steps)
                if horizon_selection_t is not None:
                    per_ep_horizon_selection_step_time.append(horizon_selection_t / steps)
        except Exception:
            continue

    weighted_step_time = None
    weighted_overhead_step_time = None
    weighted_policy_infer_step_time = None
    weighted_horizon_metric_step_time = None
    weighted_horizon_selection_step_time = None
    unweighted_step_time = None
    unweighted_overhead_step_time = None
    unweighted_policy_infer_step_time = None
    unweighted_horizon_metric_step_time = None
    unweighted_horizon_selection_step_time = None
    if total_steps_sum > 0:
        weighted_step_time = float(total_time_sum / total_steps_sum)
        weighted_overhead_step_time = float(overhead_time_sum / total_steps_sum)
        if has_policy_infer_timing:
            weighted_policy_infer_step_time = float(policy_infer_time_sum / total_steps_sum)
        if has_horizon_metric_timing:
            weighted_horizon_metric_step_time = float(horizon_metric_time_sum / total_steps_sum)
        if has_horizon_selection_timing:
            weighted_horizon_selection_step_time = float(horizon_selection_time_sum / total_steps_sum)
    if per_ep_step_time:
        unweighted_step_time = float(np.mean(np.asarray(per_ep_step_time, dtype=np.float64)))
    if per_ep_overhead_step_time:
        unweighted_overhead_step_time = float(np.mean(np.asarray(per_ep_overhead_step_time, dtype=np.float64)))
    if per_ep_policy_infer_step_time:
        unweighted_policy_infer_step_time = float(np.mean(np.asarray(per_ep_policy_infer_step_time, dtype=np.float64)))
    if per_ep_horizon_metric_step_time:
        unweighted_horizon_metric_step_time = float(np.mean(np.asarray(per_ep_horizon_metric_step_time, dtype=np.float64)))
    if per_ep_horizon_selection_step_time:
        unweighted_horizon_selection_step_time = float(
            np.mean(np.asarray(per_ep_horizon_selection_step_time, dtype=np.float64))
        )

    return {
        "episodes_with_timing": int(episodes_with_timing),
        "episodes_with_step_count": int(episodes_with_step_count),
        "total_episode_steps": int(total_steps_sum),
        "mean_step_time_seconds_weighted": weighted_step_time,
        "mean_step_time_seconds_unweighted": unweighted_step_time,
        "mean_step_denoise_overhead_seconds_weighted": weighted_overhead_step_time,
        "mean_step_denoise_overhead_seconds_unweighted": unweighted_overhead_step_time,
        "mean_step_policy_infer_seconds_weighted": weighted_policy_infer_step_time,
        "mean_step_policy_infer_seconds_unweighted": unweighted_policy_infer_step_time,
        "mean_step_horizon_metric_compute_seconds_weighted": weighted_horizon_metric_step_time,
        "mean_step_horizon_metric_compute_seconds_unweighted": unweighted_horizon_metric_step_time,
        "mean_step_horizon_selection_seconds_weighted": weighted_horizon_selection_step_time,
        "mean_step_horizon_selection_seconds_unweighted": unweighted_horizon_selection_step_time,
    }


def _format_horizon_policy_name(usr_args: dict) -> str:
    z_mode = str(usr_args.get("horizon_intra_z_mode", "window_rms"))
    signal_mode = str(usr_args.get("horizon_mix_mode", "both"))
    exec_mode = str(usr_args.get("horizon_exec_mode", "expected_round"))
    update_mode = str(usr_args.get("horizon_ts_update_mode", "kernel_forget"))
    use_speed = bool(int(usr_args.get("horizon_inter_use_speed_uniformity", 1)))
    parts = ["horizon", "ehh1dpad", z_mode]
    if use_speed and signal_mode in ("inter", "both"):
        parts.append("speedu")
    parts.append(signal_mode)
    parts.append(exec_mode)
    if update_mode != "independent":
        parts.append(update_mode)
    if exec_mode == "expected_round":
        temp_str = f"{float(usr_args.get('horizon_expected_temp', 1.0)):g}".replace(".", "p")
        parts.append(f"temp{temp_str}")
    return "_".join(parts)


def build_run_info(base_info: dict, usr_args: dict, run_dir: Path, model=None) -> dict:
    info = dict(base_info)
    if str(info.get("eval_type", "default")) == "horizon":
        info.update(
            {
                "horizon_policy": _format_horizon_policy_name(usr_args),
                "horizon_candidates": str(usr_args.get("horizon_candidates", "10,20,30,40,50")),
                "horizon_intra_alpha": float(usr_args.get("horizon_intra_alpha", usr_args.get("horizon_ts_alpha", 4.0))),
                "horizon_ts_epsilon": float(usr_args.get("horizon_ts_epsilon", 0.05)),
                "horizon_ts_seed": int(usr_args.get("horizon_ts_seed", usr_args.get("seed", 0))),
                "horizon_ts_update_mode": str(usr_args.get("horizon_ts_update_mode", "kernel_forget")),
                "horizon_ts_kernel_bandwidth": float(usr_args.get("horizon_ts_kernel_bandwidth", 10.0)),
                "horizon_ts_forget_rho": float(usr_args.get("horizon_ts_forget_rho", 0.99)),
                "horizon_ts_update_eta": float(usr_args.get("horizon_ts_update_eta", 1.0)),
                "horizon_intra_z_mode": str(usr_args.get("horizon_intra_z_mode", "window_rms")),
                "horizon_exec_mode": str(usr_args.get("horizon_exec_mode", "expected_round")),
                "horizon_expected_temp": float(usr_args.get("horizon_expected_temp", 1.0)),
                "horizon_intra_tau_window": int(usr_args.get("horizon_intra_tau_window", 3)),
                "horizon_inter_use_speed_uniformity": int(usr_args.get("horizon_inter_use_speed_uniformity", 1)),
                "horizon_mix_mode": str(usr_args.get("horizon_mix_mode", "both")),
                "horizon_inter_window": int(usr_args.get("horizon_inter_window", 60)),
                "horizon_inter_weight": float(usr_args.get("horizon_inter_weight", 0.5)),
                "horizon_inter_beta": float(usr_args.get("horizon_inter_beta", 4.0)),
                "horizon_inter_fallback": str(usr_args.get("horizon_inter_fallback", "ehh_only")),
            }
        )
        if model is not None and hasattr(model, "get_run_score_averages"):
            try:
                info.update(model.get_run_score_averages())
            except Exception:
                pass
    return info
