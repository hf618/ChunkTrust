import math
from typing import MutableMapping, Sequence

import numpy as np


HorizonPosteriorState = MutableMapping[int, list[float]]


def parse_horizon_candidates(value: Sequence[int] | str | None, default_max: int) -> list[int]:
    if value is None:
        value = "2,4,6,8,10,12"
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        parsed = [int(item) for item in items]
    else:
        parsed = [int(item) for item in value]
    parsed = sorted({candidate for candidate in parsed if candidate > 0})
    if parsed:
        return parsed
    return [max(1, int(default_max))]


def normalize_horizon_exec_mode(value: str | None) -> str:
    mode = str(value or "expected_round").strip().lower()
    if mode not in {"expected_round", "greedy", "thompson"}:
        raise ValueError(f"Unsupported exec_mode: {value}")
    return mode


def normalize_horizon_ts_update_mode(value: str | None) -> str:
    mode = str(value or "kernel_forget").strip().lower()
    if mode not in {"legacy", "independent", "kernel_forget"}:
        raise ValueError(f"Unsupported stride_ts_update_mode: {value}")
    return mode


def normalize_horizon_intra_cut_t(value: float | int | None, default: float = 0.25) -> float:
    try:
        cut_t = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(cut_t):
        return float(default)
    # Keep the split ratio inside a valid (0, 1] range.
    if cut_t <= 0.0:
        return float(default)
    if cut_t > 1.0:
        return 1.0
    return cut_t


def normalize_score_distribution(score_arr: np.ndarray, temp: float = 1.0) -> np.ndarray:
    arr = np.asarray(score_arr, dtype=np.float64)
    if arr.size <= 0:
        return np.zeros((0,), dtype=np.float64)
    arr = np.maximum(arr, 0.0)
    arr_sum = float(np.sum(arr))
    if arr_sum <= 1e-12 or (not np.isfinite(arr_sum)):
        return np.full((arr.size,), 1.0 / float(arr.size), dtype=np.float64)
    if abs(float(temp) - 1.0) <= 1e-12:
        return arr / arr_sum
    base = arr / arr_sum
    sharp = np.power(np.clip(base, 0.0, 1.0), 1.0 / float(max(1e-6, temp)))
    sharp_sum = float(np.sum(sharp))
    if sharp_sum <= 1e-12 or (not np.isfinite(sharp_sum)):
        return np.full((arr.size,), 1.0 / float(arr.size), dtype=np.float64)
    return sharp / sharp_sum


def normalize_candidate_metric(values: np.ndarray | list[float], eps: float = 1e-12) -> np.ndarray:
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


def compute_ehh_1d_padded(
    mat_h_da: np.ndarray, h_full: int, eps: float = 1e-12, intra_cut_t: float = 0.25
) -> float:
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
    cut_t = normalize_horizon_intra_cut_t(intra_cut_t)
    ft_c = max(1, int(cut_t * f_t))
    e_low = float(np.sum(pwr_f[:ft_c]))
    e_high = float(np.sum(pwr_f[ft_c:]))
    e_all = e_low + e_high + eps
    return float(e_high / e_all)


def ehh_curve_and_mean_over_tau(
    v_step_prefix: np.ndarray,
    h_full: int | None = None,
    eps: float = 1e-12,
    intra_cut_t: float = 0.25,
) -> tuple[np.ndarray, float]:
    n_tau = int(v_step_prefix.shape[0])
    if n_tau <= 0:
        return np.zeros((0,), dtype=np.float64), 0.0
    if h_full is None:
        h_full = int(v_step_prefix.shape[1])
    e_vals = np.zeros((n_tau,), dtype=np.float64)
    cut_t = normalize_horizon_intra_cut_t(intra_cut_t)
    for tau in range(n_tau):
        e_vals[tau] = compute_ehh_1d_padded(
            v_step_prefix[tau], h_full=int(h_full), eps=eps, intra_cut_t=cut_t
        )
    return e_vals, float(np.mean(e_vals))


def ehh_curves_by_stride_batched(
    v_step: np.ndarray,
    strides: list[int],
    h_full: int | None = None,
    eps: float = 1e-12,
    intra_cut_t: float = 0.25,
) -> dict[int, tuple[np.ndarray, float]]:
    n_tau = int(v_step.shape[0])
    h = int(v_step.shape[1])
    d_a = int(v_step.shape[2])
    if n_tau <= 0:
        return {}
    if h_full is None:
        h_full = h
    h_full = max(int(h_full), h)
    strides = [int(s) for s in strides]
    if not strides:
        return {}

    batch = np.zeros((len(strides), n_tau, h_full, d_a), dtype=np.float64)
    v64 = np.asarray(v_step, dtype=np.float64)
    for i, stride in enumerate(strides):
        s_clip = max(0, min(int(stride), h))
        if s_clip > 0:
            batch[i, :, :s_clip, :] = v64[:, :s_clip, :]

    spec = np.fft.rfft(batch, axis=2)
    pwr = (spec.real * spec.real) + (spec.imag * spec.imag)
    pwr_f = np.mean(pwr, axis=3)
    f_t = int(pwr_f.shape[2])
    cut_t = normalize_horizon_intra_cut_t(intra_cut_t)
    ft_c = max(1, int(cut_t * f_t))
    e_low = np.sum(pwr_f[:, :, :ft_c], axis=2)
    e_high = np.sum(pwr_f[:, :, ft_c:], axis=2)
    e_vals_all = e_high / (e_low + e_high + eps)

    out = {}
    for i, stride in enumerate(strides):
        e_vals = np.asarray(e_vals_all[i], dtype=np.float64)
        out[int(stride)] = (e_vals, float(np.mean(e_vals)))
    return out


def z_trend_abs_slope(y: np.ndarray) -> float:
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


def compute_z_from_ehh(
    e_vals: np.ndarray,
    tau0_idx: int = 0,
    mode: str = "window_rms",
    tau_window: int = 3,
) -> float:
    if e_vals.size <= 1:
        return 0.0
    tau0 = int(np.clip(tau0_idx, 0, max(0, int(e_vals.size) - 1)))
    tail = np.asarray(e_vals[tau0:], dtype=np.float64)
    if tail.size <= 1:
        return 0.0
    if mode == "trend":
        return z_trend_abs_slope(tail)
    if mode == "tau0_rms":
        baseline = float(e_vals[tau0])
    else:
        hi = min(int(e_vals.size), tau0 + max(1, int(tau_window)))
        baseline = float(np.mean(e_vals[tau0:hi]))
    dev = tail - baseline
    return float(math.sqrt(float(np.mean(dev * dev))))


def speed_uniformity_cv_from_history_window(
    history_actions: np.ndarray,
    xt_next: np.ndarray,
    stride: int,
    window: int,
    eps: float = 1e-12,
) -> float:
    h = int(xt_next.shape[0])
    s = int(stride)
    if s < 1 or s > h:
        return float("nan")
    win = max(4, int(window))
    pre = int(win - s)
    post = int(s)
    if pre < 1 or post < 1:
        return float("nan")
    if int(history_actions.shape[0]) < pre:
        return float("nan")
    if h < post:
        return float("nan")
    z = np.concatenate([history_actions[-pre:, :], xt_next[:post, :]], axis=0)
    if z.shape[0] < 3:
        return float("nan")
    spd = np.linalg.norm(np.diff(z, axis=0), axis=1)
    mu = float(np.mean(spd))
    if mu <= eps:
        return 0.0
    cv = float(np.std(spd, ddof=0) / (mu + eps))
    return cv if np.isfinite(cv) else float("nan")


def speed_uniformity_proxy_by_stride(
    xt_step: np.ndarray,
    strides: list[int],
    *,
    history_actions: np.ndarray,
    speed_window: int,
    speed_beta: float,
    eps: float = 1e-12,
) -> tuple[list[float], list[float], list[float]]:
    u_raw = []
    for stride in strides:
        u_val = speed_uniformity_cv_from_history_window(
            history_actions=history_actions,
            xt_next=xt_step,
            stride=int(stride),
            window=int(speed_window),
            eps=eps,
        )
        u_raw.append(float(u_val) if np.isfinite(u_val) else float("nan"))

    u_norm_vals = normalize_candidate_metric(u_raw, eps=eps)
    p_proxy = []
    for u_norm in u_norm_vals:
        if not np.isfinite(u_norm):
            p_proxy.append(float("nan"))
            continue
        p_val = float(np.exp(-float(speed_beta) * max(0.0, u_norm)))
        p_proxy.append(float(np.clip(p_val, 0.0, 1.0)))
    return u_raw, np.asarray(u_norm_vals, dtype=np.float64).tolist(), p_proxy


def gaussian_kernel_weights(
    strides: list[int],
    center: float,
    bandwidth: float,
    eps: float = 1e-12,
) -> np.ndarray:
    vals = np.asarray(strides, dtype=np.float64)
    if vals.size <= 0:
        return np.zeros((0,), dtype=np.float64)
    if not np.isfinite(center):
        return np.full((vals.size,), 1.0 / float(vals.size), dtype=np.float64)
    bw = max(float(bandwidth), eps)
    weights = np.exp(-((vals - float(center)) ** 2) / (2.0 * bw * bw))
    weight_sum = float(np.sum(weights))
    if weight_sum <= eps or (not np.isfinite(weight_sum)):
        return np.full((vals.size,), 1.0 / float(vals.size), dtype=np.float64)
    return weights / weight_sum


def initialize_stride_ts_state(
    strides: Sequence[int],
    ts_state: HorizonPosteriorState | None = None,
) -> HorizonPosteriorState:
    if ts_state is None:
        ts_state = {}
    for stride in strides:
        key = int(stride)
        state = ts_state.get(key)
        if not isinstance(state, (list, tuple)) or len(state) != 2:
            ts_state[key] = [1.0, 1.0]
            continue
        alpha = max(1e-6, float(state[0]))
        beta = max(1e-6, float(state[1]))
        ts_state[key] = [alpha, beta]
    return ts_state


def apply_stride_ts_update(
    *,
    strides: list[int],
    q_mix_vals: list[float],
    chosen_idx: int,
    choose_mode: str,
    k_exec: int,
    ts_state: HorizonPosteriorState,
    ts_update_mode: str,
    ts_kernel_bandwidth: float,
    ts_forget_rho: float,
    ts_update_eta: float,
    expected_prob: np.ndarray | None = None,
) -> tuple[float, float]:
    if ts_update_mode == "independent":
        if choose_mode == "expected_round" and expected_prob is not None:
            for i, stride in enumerate(strides):
                prob_i = float(expected_prob[i]) if i < int(expected_prob.size) else 0.0
                if prob_i <= 0.0:
                    continue
                qa_i = float(q_mix_vals[i])
                alpha, beta = ts_state.get(int(stride), [1.0, 1.0])
                ts_state[int(stride)] = [
                    max(1e-6, alpha + (prob_i * qa_i)),
                    max(1e-6, beta + (prob_i * (1.0 - qa_i))),
                ]
            feedback = float(
                np.dot(np.asarray(expected_prob, dtype=np.float64), np.asarray(q_mix_vals, dtype=np.float64))
            )
            return float(k_exec), feedback

        qa = float(q_mix_vals[chosen_idx])
        alpha, beta = ts_state.get(int(k_exec), [1.0, 1.0])
        ts_state[int(k_exec)] = [
            max(1e-6, alpha + qa),
            max(1e-6, beta + (1.0 - qa)),
        ]
        return float(k_exec), qa

    if choose_mode == "expected_round" and expected_prob is not None:
        feedback = float(
            np.dot(np.asarray(expected_prob, dtype=np.float64), np.asarray(q_mix_vals, dtype=np.float64))
        )
    else:
        feedback = float(q_mix_vals[chosen_idx])
    feedback = float(np.clip(feedback, 0.0, 1.0))

    update_center = float(k_exec)
    weights = gaussian_kernel_weights(
        strides,
        center=update_center,
        bandwidth=ts_kernel_bandwidth,
    )
    rho = float(np.clip(float(ts_forget_rho), 1e-6, 1.0))
    eta = float(max(0.0, float(ts_update_eta)))
    for i, stride in enumerate(strides):
        alpha, beta = ts_state.get(int(stride), [1.0, 1.0])
        w_i = float(weights[i]) if i < int(weights.size) else 0.0
        incr = eta * w_i
        ts_state[int(stride)] = [
            max(1e-6, rho * alpha + incr * feedback),
            max(1e-6, rho * beta + incr * (1.0 - feedback)),
        ]
    return update_center, feedback


def select_exec_k_horizon(
    v_step: np.ndarray,
    used_h: int,
    *,
    horizon_candidates: Sequence[int] | str,
    exec_mode: str = "expected_round",
    expected_temp: float = 1.0,
    horizon_alpha: float = 4.0,
    intra_cut_t: float = 0.25,
    z_mode: str = "window_rms",
    tau_window: int = 3,
    xt_step: np.ndarray | None = None,
    history_actions: np.ndarray | None = None,
    signal_mode: str = "intra",
    use_speed_uniformity: bool = True,
    speed_window: int = 60,
    speed_weight: float = 0.5,
    speed_beta: float = 4.0,
    speed_fallback: str = "ehh_only",
    ts_state: HorizonPosteriorState | None = None,
    ts_rng: np.random.Generator | None = None,
    ts_epsilon: float = 0.05,
    ts_update_mode: str = "kernel_forget",
    ts_kernel_bandwidth: float = 10.0,
    ts_forget_rho: float = 0.99,
    ts_update_eta: float = 1.0,
) -> tuple[int, dict]:
    used_h = int(used_h)
    exec_mode = normalize_horizon_exec_mode(exec_mode)
    requested_ts_update_mode = normalize_horizon_ts_update_mode(ts_update_mode)
    effective_ts_update_mode = requested_ts_update_mode
    if exec_mode == "thompson" and effective_ts_update_mode == "legacy":
        effective_ts_update_mode = "independent"

    candidates = parse_horizon_candidates(horizon_candidates, default_max=used_h)
    strides = [stride for stride in candidates if stride <= used_h]
    if not strides:
        strides = [used_h]

    signal_mode = str(signal_mode).strip().lower()
    if signal_mode not in ("intra", "inter", "both"):
        raise ValueError(f"Unsupported signal_mode: {signal_mode}")
    speed_fallback = str(speed_fallback).strip().lower()
    if speed_fallback not in ("ehh_only", "neutral"):
        raise ValueError(f"Unsupported speed_fallback: {speed_fallback}")

    intra_cut_t = normalize_horizon_intra_cut_t(intra_cut_t)
    z_vals = []
    q_vals = []
    z_norm_vals = []
    q_mix_vals = []
    u_vals = []
    u_norm_vals = []
    p_vals = []
    ehh_curves = []

    use_inter_signal = bool(use_speed_uniformity) and signal_mode in ("inter", "both")
    if use_inter_signal and xt_step is not None and history_actions is not None:
        u_vals, u_norm_vals, p_vals = speed_uniformity_proxy_by_stride(
            np.asarray(xt_step, dtype=np.float32),
            strides,
            history_actions=np.asarray(history_actions, dtype=np.float32),
            speed_window=int(speed_window),
            speed_beta=float(speed_beta),
        )
    else:
        u_vals = [float("nan")] * len(strides)
        u_norm_vals = [float("nan")] * len(strides)
        p_vals = [float("nan")] * len(strides)

    ehh_by_stride = ehh_curves_by_stride_batched(
        np.asarray(v_step, dtype=np.float32), strides, h_full=used_h, intra_cut_t=intra_cut_t
    )
    for stride in strides:
        e_vals, _ = ehh_by_stride[int(stride)]
        z_val = compute_z_from_ehh(e_vals, tau0_idx=0, mode=z_mode, tau_window=tau_window)
        ehh_curves.append(np.asarray(e_vals, dtype=np.float32))
        z_vals.append(float(z_val))

    z_norm_arr = normalize_candidate_metric(z_vals)
    for i in range(len(strides)):
        z_norm = float(z_norm_arr[i]) if i < int(z_norm_arr.size) else float("nan")
        z_norm_vals.append(z_norm)
        if np.isfinite(z_norm):
            q_val = float(np.exp(-float(horizon_alpha) * max(0.0, z_norm)))
        else:
            q_val = float("nan")
        p_val = p_vals[i]
        if signal_mode == "intra":
            q_mix = q_val
        elif signal_mode == "inter":
            if np.isfinite(p_val):
                q_mix = float(np.clip(p_val, 0.0, 1.0))
            elif speed_fallback == "neutral":
                q_mix = 0.5
            else:
                q_mix = q_val
        else:
            if np.isfinite(p_val):
                q_mix = float(
                    (1.0 - float(speed_weight)) * q_val
                    + float(speed_weight) * float(np.clip(p_val, 0.0, 1.0))
                )
            elif speed_fallback == "neutral":
                q_mix = float((1.0 - float(speed_weight)) * q_val + float(speed_weight) * 0.5)
            else:
                q_mix = q_val
        q_vals.append(q_val)
        q_mix_vals.append(float(q_mix))

    legacy_selector = effective_ts_update_mode == "legacy" and exec_mode != "thompson"
    if legacy_selector:
        score_arr = np.asarray(q_mix_vals, dtype=np.float64)
        score_prob = normalize_score_distribution(score_arr, temp=1.0)
        if exec_mode == "expected_round":
            expected_prob = normalize_score_distribution(score_arr, temp=expected_temp)
            expected_stride = float(np.dot(expected_prob, np.asarray(strides, dtype=np.float64)))
            k_exec = int(np.clip(int(round(expected_stride)), 1, used_h))
            choose_mode = "expected_round"
        else:
            expected_prob = score_prob
            expected_stride = float(strides[int(np.argmax(score_arr))])
            k_exec = int(np.clip(int(expected_stride), 1, used_h))
            choose_mode = "greedy"

        return k_exec, {
            "selector_mode": "legacy",
            "strides": [int(stride) for stride in strides],
            "z_ehh_rms_from_tau0": z_vals,
            "z_intra": z_vals,
            "z_intra_norm": z_norm_vals,
            "q_proxy": q_vals,
            "q_intra": q_vals,
            "q_mix": q_mix_vals,
            "u_inter_raw": [float(value) if np.isfinite(value) else None for value in u_vals],
            "u_inter_norm": [float(value) if np.isfinite(value) else None for value in u_norm_vals],
            "p_inter": [float(value) if np.isfinite(value) else None for value in p_vals],
            "speed_uniformity_cv": [float(value) if np.isfinite(value) else None for value in u_vals],
            "speed_uniformity_proxy": [
                float(value) if np.isfinite(value) else None for value in p_vals
            ],
            "score_prob": [float(prob) for prob in score_prob.tolist()],
            "expected_prob": [float(prob) for prob in expected_prob.tolist()],
            "expected_stride": expected_stride,
            "horizon_exec_mode": exec_mode,
            "horizon_expected_temp": float(expected_temp),
            "horizon_intra_alpha": float(horizon_alpha),
            "horizon_intra_cut_t": float(intra_cut_t),
            "horizon_intra_z_mode": z_mode,
            "horizon_intra_tau_window": int(tau_window),
            "horizon_mix_mode": signal_mode,
            "horizon_inter_use_speed_uniformity": bool(use_speed_uniformity),
            "horizon_inter_window": int(speed_window),
            "horizon_inter_weight": float(speed_weight),
            "horizon_inter_beta": float(speed_beta),
            "horizon_inter_fallback": speed_fallback,
            "horizon_ts_epsilon": float(ts_epsilon),
            "horizon_ts_update_mode_requested": requested_ts_update_mode,
            "horizon_ts_update_mode": effective_ts_update_mode,
            "horizon_ts_kernel_bandwidth": float(ts_kernel_bandwidth),
            "horizon_ts_forget_rho": float(ts_forget_rho),
            "horizon_ts_update_eta": float(ts_update_eta),
            "choose_mode": choose_mode,
            "ehh_curve_by_stride": [curve.tolist() for curve in ehh_curves],
            "chosen_stride": int(k_exec),
        }

    ts_state = initialize_stride_ts_state(strides, ts_state=ts_state)
    if ts_rng is None:
        ts_rng = np.random.default_rng(0)

    theta_vals = []
    score_vals = []
    alpha_before = []
    beta_before = []
    for i, stride in enumerate(strides):
        alpha, beta = ts_state.get(int(stride), [1.0, 1.0])
        alpha = max(1e-6, float(alpha))
        beta = max(1e-6, float(beta))
        theta = float(ts_rng.beta(alpha, beta))
        score = theta * float(q_mix_vals[i])
        alpha_before.append(alpha)
        beta_before.append(beta)
        theta_vals.append(theta)
        score_vals.append(float(score))

    score_arr = np.asarray(score_vals, dtype=np.float64)
    score_prob = normalize_score_distribution(score_arr, temp=1.0)
    expected_prob = score_prob
    expected_stride = None
    if exec_mode == "expected_round":
        expected_prob = normalize_score_distribution(score_arr, temp=expected_temp)
        if expected_prob.size > 0:
            expected_stride = float(np.dot(expected_prob, np.asarray(strides, dtype=np.float64)))

    eps_rand = max(0.0, min(1.0, float(ts_epsilon)))
    if float(ts_rng.random()) < eps_rand:
        chosen_idx = int(ts_rng.integers(0, len(strides)))
        choose_mode = "epsilon_random"
        k_exec = int(strides[chosen_idx])
        update_center, update_feedback = apply_stride_ts_update(
            strides=strides,
            q_mix_vals=q_mix_vals,
            chosen_idx=chosen_idx,
            choose_mode=choose_mode,
            k_exec=k_exec,
            ts_state=ts_state,
            ts_update_mode=effective_ts_update_mode,
            ts_kernel_bandwidth=ts_kernel_bandwidth,
            ts_forget_rho=ts_forget_rho,
            ts_update_eta=ts_update_eta,
        )
    elif exec_mode == "expected_round":
        if expected_stride is None or (not np.isfinite(expected_stride)):
            expected_stride = float(np.mean(np.asarray(strides, dtype=np.float64)))
        k_exec = int(np.clip(int(round(expected_stride)), 1, used_h))
        choose_mode = "expected_round"
        update_center, update_feedback = apply_stride_ts_update(
            strides=strides,
            q_mix_vals=q_mix_vals,
            chosen_idx=-1,
            choose_mode=choose_mode,
            k_exec=k_exec,
            expected_prob=expected_prob,
            ts_state=ts_state,
            ts_update_mode=effective_ts_update_mode,
            ts_kernel_bandwidth=ts_kernel_bandwidth,
            ts_forget_rho=ts_forget_rho,
            ts_update_eta=ts_update_eta,
        )
    else:
        chosen_idx = int(np.argmax(score_arr))
        choose_mode = "thompson" if exec_mode == "thompson" else "greedy"
        k_exec = int(strides[chosen_idx])
        update_center, update_feedback = apply_stride_ts_update(
            strides=strides,
            q_mix_vals=q_mix_vals,
            chosen_idx=chosen_idx,
            choose_mode=choose_mode,
            k_exec=k_exec,
            ts_state=ts_state,
            ts_update_mode=effective_ts_update_mode,
            ts_kernel_bandwidth=ts_kernel_bandwidth,
            ts_forget_rho=ts_forget_rho,
            ts_update_eta=ts_update_eta,
        )

    alpha_after = [float(ts_state[int(stride)][0]) for stride in strides]
    beta_after = [float(ts_state[int(stride)][1]) for stride in strides]
    return k_exec, {
        "selector_mode": "bandit",
        "strides": [int(stride) for stride in strides],
        "z_ehh_rms_from_tau0": z_vals,
        "z_intra": z_vals,
        "z_intra_norm": z_norm_vals,
        "q_proxy": q_vals,
        "q_intra": q_vals,
        "q_mix": q_mix_vals,
        "u_inter_raw": [float(value) if np.isfinite(value) else None for value in u_vals],
        "u_inter_norm": [float(value) if np.isfinite(value) else None for value in u_norm_vals],
        "p_inter": [float(value) if np.isfinite(value) else None for value in p_vals],
        "speed_uniformity_cv": [float(value) if np.isfinite(value) else None for value in u_vals],
        "speed_uniformity_proxy": [
            float(value) if np.isfinite(value) else None for value in p_vals
        ],
        "score": [float(score) for score in score_vals],
        "score_prob": [float(prob) for prob in score_prob.tolist()],
        "expected_prob": [float(prob) for prob in expected_prob.tolist()],
        "expected_stride": expected_stride,
        "theta_sample": [float(theta) for theta in theta_vals],
        "posterior_alpha_before": alpha_before,
        "posterior_beta_before": beta_before,
        "posterior_alpha_after": alpha_after,
        "posterior_beta_after": beta_after,
        "horizon_exec_mode": exec_mode,
        "horizon_expected_temp": float(expected_temp),
        "horizon_intra_alpha": float(horizon_alpha),
        "horizon_intra_cut_t": float(intra_cut_t),
        "horizon_intra_z_mode": z_mode,
        "horizon_intra_tau_window": int(tau_window),
        "horizon_mix_mode": signal_mode,
        "horizon_inter_use_speed_uniformity": bool(use_speed_uniformity),
        "horizon_inter_window": int(speed_window),
        "horizon_inter_weight": float(speed_weight),
        "horizon_inter_beta": float(speed_beta),
        "horizon_inter_fallback": speed_fallback,
        "horizon_ts_epsilon": float(ts_epsilon),
        "horizon_ts_update_mode_requested": requested_ts_update_mode,
        "horizon_ts_update_mode": effective_ts_update_mode,
        "horizon_ts_kernel_bandwidth": float(ts_kernel_bandwidth),
        "horizon_ts_forget_rho": float(ts_forget_rho),
        "horizon_ts_update_eta": float(ts_update_eta),
        "choose_mode": choose_mode,
        "update_center": float(update_center),
        "update_feedback": float(update_feedback),
        "ehh_curve_by_stride": [curve.tolist() for curve in ehh_curves],
        "chosen_stride": int(k_exec),
    }


