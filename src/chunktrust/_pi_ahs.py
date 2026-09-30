# Frozen from pi05_horizon/pi_model.py; numerical methods copied by AST, no model imports.
from __future__ import annotations
import math
import numpy as np
from typing import Any

class Selector:
    def __init__(self, candidates=(10,20,30,40), seed=0):
        self.fft_horizon = 50
        self.horizon_intra_cut_t = 0.25
        self.horizon_intra_alpha = 4.0
        self.horizon_ts_epsilon = 0.05
        self.horizon_ts_update_mode = 'kernel_forget'
        self.horizon_ts_kernel_bandwidth = 10.0
        self.horizon_ts_forget_rho = 0.99
        self.horizon_ts_update_eta = 1.0
        self.horizon_intra_z_mode = 'window_rms'
        self.horizon_exec_mode = 'expected_round'
        self.horizon_expected_temp = 0.8
        self.horizon_intra_tau_window = 3
        self.horizon_inter_use_speed_uniformity = True
        self.horizon_mix_mode = 'both'
        self.horizon_inter_window = 60
        self.horizon_inter_weight = 0.5
        self.horizon_inter_beta = 4.0
        self.horizon_inter_fallback = 'ehh_only'
        self.horizon_candidates = list(candidates)
        self._horizon_ts_state = {s: [1., 1.] for s in candidates}
        self._horizon_ts_rng = np.random.default_rng(seed)
        self._executed_action_history = np.zeros((0,14), dtype=np.float32)

    def select(self, velocity, actions, history):
        if len(actions) < max(self.horizon_candidates):
            raise ValueError("available suffix cannot support frozen candidate set")
        if velocity.shape[1] != len(actions) or len(actions) > self.fft_horizon:
            raise ValueError('velocity/action suffix alignment or H=50 contract violated')
        self._executed_action_history = np.asarray(history, dtype=np.float32).reshape(-1,14)[-60:]
        # The actual suffix supplies candidate prefixes; zero padding and the
        # frequency grid retain the original prediction horizon. Feasibility is
        # checked against the available actions above, not the FFT length.
        info = self._compute_horizon_info(velocity[..., :14], self.fft_horizon, xt_step=actions[:, :14])
        info['fft_horizon'] = self.fft_horizon
        info['available_horizon'] = len(actions)
        if not np.all(np.isfinite(info['q_mix'])):
            raise FloatingPointError('nonfinite AHS score')
        return self._select_exec_k_horizon_from_info(info, len(actions))

    @staticmethod
    def _parse_horizon_z_mode(raw) -> str:
        mode = str(raw or 'window_rms').strip().lower()
        if mode not in {'tau0_rms', 'window_rms', 'trend'}:
            mode = 'window_rms'
        return mode

    def _uses_horizon_posterior(self) -> bool:
        return self.horizon_exec_mode in {'thompson', 'expected_round'}

    @staticmethod
    def _normalize_candidate_metric(values: np.ndarray | list[float], eps: float=1e-12) -> np.ndarray:
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

    @staticmethod
    def _normalize_score_distribution(score_arr: np.ndarray, temp: float=1.0) -> np.ndarray:
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
        sharp = np.power(np.clip(base, 0.0, 1.0), 1.0 / float(max(1e-06, temp)))
        sharp_sum = float(np.sum(sharp))
        if sharp_sum <= 1e-12 or not np.isfinite(sharp_sum):
            return np.full((arr.size,), 1.0 / float(arr.size), dtype=np.float64)
        return sharp / sharp_sum

    def _ehh_curves_by_horizon_batched(self, v_step: np.ndarray, horizons: list[int], h_full: int | None=None, eps: float=1e-12) -> dict[int, tuple[np.ndarray, float]]:
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
        pwr = spec.real * spec.real + spec.imag * spec.imag
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

    def _compute_z_from_ehh(self, e_vals: np.ndarray, tau0_idx: int=0, mode_override: str | None=None, tau_window_override: int | None=None) -> float:
        if e_vals.size <= 1:
            return 0.0
        mode = self._parse_horizon_z_mode(mode_override if mode_override is not None else self.horizon_intra_z_mode)
        tau0 = int(np.clip(tau0_idx, 0, max(0, int(e_vals.size) - 1)))
        tail = np.asarray(e_vals[tau0:], dtype=np.float64)
        if tail.size <= 1:
            return 0.0
        if mode == 'trend':
            return self._z_trend_abs_slope(tail)
        if mode == 'tau0_rms':
            baseline = float(e_vals[tau0])
        else:
            w = max(1, int(tau_window_override if tau_window_override is not None else self.horizon_intra_tau_window))
            hi = min(int(e_vals.size), tau0 + w)
            baseline = float(np.mean(e_vals[tau0:hi]))
        dev = tail - baseline
        return float(math.sqrt(float(np.mean(dev * dev))))

    def _speed_uniformity_proxy_by_horizon(self, xt_step: np.ndarray, horizons: list[int], eps: float=1e-12) -> tuple[list[float], list[float], list[float]]:
        xt_step = np.asarray(xt_step, dtype=np.float32)
        if xt_step.ndim != 2 or xt_step.shape[1] <= 0:
            nan_list = [float('nan')] * len(horizons)
            return (nan_list, nan_list, nan_list)
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
        speed = np.linalg.norm(np.diff(seq, axis=0), axis=1).astype(np.float64, copy=False) if seq.shape[0] >= 2 else np.zeros((0,), dtype=np.float64)
        speed_prefix = np.concatenate([[0.0], np.cumsum(speed, dtype=np.float64)])
        speed_sq_prefix = np.concatenate([[0.0], np.cumsum(speed * speed, dtype=np.float64)])
        u_raw = []
        for s in horizons:
            s = int(s)
            if s < 1 or s > h:
                u_raw.append(float('nan'))
                continue
            pre = int(win - s)
            post = int(s)
            if pre < 1 or post < 1 or history_len < pre or (h < post):
                u_raw.append(float('nan'))
                continue
            start = int(history_tail_len - pre)
            speed_start = start
            speed_end = speed_start + (win - 1)
            if speed_start < 0 or speed_end > int(speed.shape[0]):
                u_raw.append(float('nan'))
                continue
            count = float(win - 1)
            sum_speed = float(speed_prefix[speed_end] - speed_prefix[speed_start])
            sum_speed_sq = float(speed_sq_prefix[speed_end] - speed_sq_prefix[speed_start])
            mu = sum_speed / count
            if mu <= eps:
                u_raw.append(0.0)
                continue
            var = max(0.0, sum_speed_sq / count - mu * mu)
            cv = float(math.sqrt(var) / (mu + eps))
            u_raw.append(cv if np.isfinite(cv) else float('nan'))
        u_norm_vals = self._normalize_candidate_metric(u_raw, eps=eps)
        p_proxy = []
        for u_norm in u_norm_vals:
            if not np.isfinite(u_norm):
                p_proxy.append(float('nan'))
                continue
            p = float(np.exp(-self.horizon_inter_beta * max(0.0, u_norm)))
            p_proxy.append(float(np.clip(p, 0.0, 1.0)))
        return (u_raw, np.asarray(u_norm_vals, dtype=np.float64).tolist(), p_proxy)

    @staticmethod
    def _gaussian_kernel_weights(horizons: list[int], center: float, bandwidth: float, eps: float=1e-12) -> np.ndarray:
        vals = np.asarray(horizons, dtype=np.float64)
        if vals.size <= 0:
            return np.zeros((0,), dtype=np.float64)
        if not np.isfinite(center):
            return np.full((vals.size,), 1.0 / float(vals.size), dtype=np.float64)
        bw = max(float(bandwidth), eps)
        weights = np.exp(-(vals - float(center)) ** 2 / (2.0 * bw * bw))
        weight_sum = float(np.sum(weights))
        if weight_sum <= eps or not np.isfinite(weight_sum):
            return np.full((vals.size,), 1.0 / float(vals.size), dtype=np.float64)
        return weights / weight_sum

    def _apply_horizon_ts_update(self, horizons: list[int], q_mix_vals: list[float], chosen_idx: int, choose_mode: str, k_exec: int, expected_prob: np.ndarray | None=None) -> tuple[float, float]:
        if self.horizon_ts_update_mode == 'independent':
            if choose_mode == 'expected_round' and expected_prob is not None:
                for i, s in enumerate(horizons):
                    prob_i = float(expected_prob[i]) if i < int(expected_prob.size) else 0.0
                    if prob_i <= 0.0:
                        continue
                    qa_i = float(q_mix_vals[i])
                    a, b = self._horizon_ts_state.get(int(s), [1.0, 1.0])
                    self._horizon_ts_state[int(s)] = [a + prob_i * qa_i, b + prob_i * (1.0 - qa_i)]
                feedback = float(np.dot(np.asarray(expected_prob, dtype=np.float64), np.asarray(q_mix_vals, dtype=np.float64)))
                return (float(k_exec), feedback)
            qa = float(q_mix_vals[chosen_idx])
            a, b = self._horizon_ts_state.get(int(k_exec), [1.0, 1.0])
            self._horizon_ts_state[int(k_exec)] = [a + qa, b + (1.0 - qa)]
            return (float(k_exec), qa)
        if choose_mode == 'expected_round' and expected_prob is not None:
            feedback = float(np.dot(np.asarray(expected_prob, dtype=np.float64), np.asarray(q_mix_vals, dtype=np.float64)))
        else:
            feedback = float(q_mix_vals[chosen_idx])
        feedback = float(np.clip(feedback, 0.0, 1.0))
        update_center = float(k_exec)
        weights = self._gaussian_kernel_weights(horizons, center=update_center, bandwidth=self.horizon_ts_kernel_bandwidth)
        rho = self.horizon_ts_forget_rho
        eta = self.horizon_ts_update_eta
        for i, s in enumerate(horizons):
            a, b = self._horizon_ts_state.get(int(s), [1.0, 1.0])
            w_i = float(weights[i]) if i < int(weights.size) else 0.0
            incr = eta * w_i
            self._horizon_ts_state[int(s)] = [rho * a + incr * feedback, rho * b + incr * (1.0 - feedback)]
        return (update_center, feedback)

    def _compute_horizon_info(self, v_step: np.ndarray, used_h: int, xt_step: np.ndarray | None=None) -> dict[str, Any]:
        horizons = [s for s in self.horizon_candidates if s <= int(used_h)]
        if not horizons:
            horizons = [int(used_h)]
        use_inter_signal = self.horizon_inter_use_speed_uniformity and self.horizon_mix_mode in ('inter', 'both')
        if use_inter_signal and xt_step is not None:
            u_vals, u_norm_vals, p_vals = self._speed_uniformity_proxy_by_horizon(np.asarray(xt_step, dtype=np.float32), horizons)
        else:
            u_vals = [float('nan')] * len(horizons)
            u_norm_vals = [float('nan')] * len(horizons)
            p_vals = [float('nan')] * len(horizons)
        ehh_by_horizon = self._ehh_curves_by_horizon_batched(np.asarray(v_step[:, :used_h, :], dtype=np.float32), horizons, h_full=used_h)
        z_vals = []
        for s in horizons:
            e_vals, _ = ehh_by_horizon[int(s)]
            z = self._compute_z_from_ehh(e_vals, tau0_idx=0)
            z_vals.append(float(z))
        z_norm_vals = self._normalize_candidate_metric(z_vals)
        q_intra_vals = []
        q_mix_vals = []
        for i, s in enumerate(horizons):
            z_norm = float(z_norm_vals[i]) if i < int(z_norm_vals.size) else float('nan')
            if np.isfinite(z_norm):
                q_intra = float(np.exp(-self.horizon_intra_alpha * max(0.0, z_norm)))
            else:
                q_intra = float('nan')
            p_inter = p_vals[i]
            if self.horizon_mix_mode == 'intra':
                q_mix = q_intra
            elif self.horizon_mix_mode == 'inter':
                if np.isfinite(p_inter):
                    q_mix = float(np.clip(p_inter, 0.0, 1.0))
                else:
                    q_mix = 0.5 if self.horizon_inter_fallback == 'neutral' else q_intra
            elif np.isfinite(p_inter):
                q_mix = float((1.0 - self.horizon_inter_weight) * q_intra + self.horizon_inter_weight * float(np.clip(p_inter, 0.0, 1.0)))
            else:
                q_mix = float((1.0 - self.horizon_inter_weight) * q_intra + self.horizon_inter_weight * 0.5) if self.horizon_inter_fallback == 'neutral' else q_intra
            q_intra_vals.append(float(q_intra))
            q_mix_vals.append(float(q_mix))
        return {'horizons': np.asarray(horizons, dtype=np.int32), 'z_intra': np.asarray(z_vals, dtype=np.float32), 'z_intra_norm': np.asarray(z_norm_vals, dtype=np.float32), 'u_inter_raw': np.asarray(u_vals, dtype=np.float32), 'u_inter_norm': np.asarray(u_norm_vals, dtype=np.float32), 'q_intra': np.asarray(q_intra_vals, dtype=np.float32), 'p_inter': np.asarray(p_vals, dtype=np.float32), 'q_mix': np.asarray(q_mix_vals, dtype=np.float32), 'z_ehh_rms_from_tau0': np.asarray(z_vals, dtype=np.float32), 'z_mode': self.horizon_intra_z_mode, 'horizon_exec_mode': self.horizon_exec_mode, 'horizon_ts_update_mode': self.horizon_ts_update_mode, 'horizon_ts_kernel_bandwidth': float(self.horizon_ts_kernel_bandwidth), 'horizon_ts_forget_rho': float(self.horizon_ts_forget_rho), 'horizon_ts_update_eta': float(self.horizon_ts_update_eta), 'horizon_expected_temp': float(self.horizon_expected_temp), 'tau_window': int(self.horizon_intra_tau_window)}

    def _select_exec_k_horizon_from_info(self, horizon_info: dict[str, Any], used_h: int) -> tuple[int, dict[str, Any]]:
        horizons = np.asarray(horizon_info.get('horizons', []), dtype=np.int32).tolist()
        if not horizons:
            horizons = [int(used_h)]
        q_mix_vals = np.asarray(horizon_info.get('q_mix', []), dtype=np.float64).tolist()
        if not q_mix_vals:
            q_mix_vals = [1.0 for _ in horizons]
        score_vals = []
        for s, q_mix in zip(horizons, q_mix_vals, strict=True):
            if self._uses_horizon_posterior():
                a, b = self._horizon_ts_state.get(int(s), [1.0, 1.0])
                theta = float(self._horizon_ts_rng.beta(a, b))
                score_vals.append(float(theta * q_mix))
            else:
                score_vals.append(float(q_mix))
        score_arr = np.asarray(score_vals, dtype=np.float64)
        score_prob = self._normalize_score_distribution(score_arr, temp=1.0)
        expected_prob = score_prob
        expected_horizon = None
        if self.horizon_exec_mode in ('expected_round', 'qmix_expected_round'):
            expected_prob = self._normalize_score_distribution(score_arr, temp=self.horizon_expected_temp)
        if expected_prob.size > 0:
            expected_horizon = float(np.dot(expected_prob, np.asarray(horizons, dtype=np.float64)))
        if self.horizon_exec_mode == 'qmix_expected_round':
            if expected_horizon is None or not np.isfinite(expected_horizon):
                expected_horizon = float(np.mean(np.asarray(horizons, dtype=np.float64)))
            k_exec = int(np.clip(int(round(expected_horizon)), 1, int(used_h)))
            choose_mode = 'qmix_expected_round'
            update_center = float('nan')
            update_feedback = float('nan')
        else:
            eps_rand = max(0.0, min(1.0, self.horizon_ts_epsilon))
            if float(self._horizon_ts_rng.random()) < eps_rand:
                chosen_idx = int(self._horizon_ts_rng.integers(0, len(horizons)))
                choose_mode = 'epsilon_random'
                k_exec = int(horizons[chosen_idx])
                update_center, update_feedback = self._apply_horizon_ts_update(horizons=horizons, q_mix_vals=q_mix_vals, chosen_idx=chosen_idx, choose_mode=choose_mode, k_exec=k_exec)
            elif self.horizon_exec_mode == 'expected_round':
                if expected_horizon is None or not np.isfinite(expected_horizon):
                    expected_horizon = float(np.mean(np.asarray(horizons, dtype=np.float64)))
                k_exec = int(np.clip(int(round(expected_horizon)), 1, int(used_h)))
                choose_mode = 'expected_round'
                update_center, update_feedback = self._apply_horizon_ts_update(horizons=horizons, q_mix_vals=q_mix_vals, chosen_idx=-1, choose_mode=choose_mode, k_exec=k_exec, expected_prob=expected_prob)
            else:
                chosen_idx = int(np.argmax(score_arr))
                choose_mode = 'thompson'
                k_exec = int(horizons[chosen_idx])
                update_center, update_feedback = self._apply_horizon_ts_update(horizons=horizons, q_mix_vals=q_mix_vals, chosen_idx=chosen_idx, choose_mode=choose_mode, k_exec=k_exec)
        out = dict(horizon_info)
        out.update({'horizons': np.asarray(horizons, dtype=np.int32), 'q_mix': np.asarray(q_mix_vals, dtype=np.float32), 'scores': np.asarray(score_vals, dtype=np.float32), 'expected_horizon': expected_horizon, 'choose_mode': choose_mode, 'chosen_horizon': int(k_exec), 'update_center': float(update_center), 'update_feedback': float(update_feedback)})
        return (k_exec, out)
