#!/usr/bin/env python3
"""Horizon-aware safe rollout bridge for OpenPI pi05 and AgileX ALOHA Dora."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from openpi_bridge_common import (  # noqa: E402
    PROMPT as DEFAULT_PROMPT,
    action_dict_from_left_right,
    get_complete_observation,
    make_robot,
    observation_to_policy_input,
    observation_to_raw26,
    print_delta,
    raw26_to_left_right,
)
from eval_video_recorder import add_recording_args, create_eval_video_recorder  # noqa: E402
from eval_episode_npz_logger import EvalEpisodeNpzLogger  # noqa: E402


@dataclass
class AHSChoice:
    exec_length: int
    source: str
    mode: str
    candidates: np.ndarray | None = None
    posterior: np.ndarray | None = None
    q_mix: np.ndarray | None = None
    expected_horizon: float | None = None
    feedback: float | None = None
    inactive_reason: str | None = None
    details: dict[str, Any] | None = None


class AHSRuntimeSelector:
    """Small client-side port of the reference HybridQHARuntimeSelector."""

    def __init__(
        self,
        *,
        candidate_horizons: list[int],
        exec_mode: str,
        expected_temp: float,
        ts_epsilon: float,
        ts_seed: int,
        ts_update_mode: str,
        ts_kernel_bandwidth: float,
        ts_forget_rho: float,
        ts_update_eta: float,
        prior_gamma: float,
    ) -> None:
        self.candidate_horizons = tuple(int(v) for v in candidate_horizons)
        self.exec_mode = exec_mode
        self.expected_temp = float(max(expected_temp, 1e-6))
        self.ts_epsilon = float(np.clip(ts_epsilon, 0.0, 1.0))
        self.ts_update_mode = ts_update_mode
        self.ts_kernel_bandwidth = float(max(ts_kernel_bandwidth, 1e-6))
        self.ts_forget_rho = float(np.clip(ts_forget_rho, 1e-6, 1.0))
        self.ts_update_eta = float(max(ts_update_eta, 0.0))
        self.prior_gamma = float(max(prior_gamma, 0.0))
        self._rng = np.random.default_rng(int(ts_seed))
        self._state = {int(s): [1.0, 1.0] for s in self.candidate_horizons}

    def ensure_candidates(self, candidates: list[int]) -> None:
        new_candidates = tuple(int(v) for v in candidates)
        if new_candidates == self.candidate_horizons:
            return
        self.candidate_horizons = new_candidates
        self._state = {int(s): [1.0, 1.0] for s in self.candidate_horizons}

    def select(self, posterior: np.ndarray, q_mix: np.ndarray, *, max_exec_length: int) -> dict[str, Any]:
        posterior = np.asarray(posterior, dtype=np.float64).reshape(-1)
        q_mix = np.asarray(q_mix, dtype=np.float64).reshape(-1)
        if posterior.shape != q_mix.shape:
            raise ValueError(f"posterior/q_mix shape mismatch: {posterior.shape} vs {q_mix.shape}")
        if posterior.shape[0] != len(self.candidate_horizons):
            raise ValueError(
                "posterior length must match candidate_horizons: "
                f"{posterior.shape[0]} vs {len(self.candidate_horizons)}"
            )

        prior = np.power(np.clip(posterior, 1e-12, 1.0), self.prior_gamma)
        effective = np.clip(q_mix, 0.0, 1.0) * prior
        theta = np.asarray([self._rng.beta(*self._state[int(s)]) for s in self.candidate_horizons], dtype=np.float64)
        runtime_scores = theta * effective
        expected_prob = None
        expected_horizon = None

        if self.exec_mode == "qmix_expected_round":
            expected_prob = normalize_scores(effective, temp=self.expected_temp)
            expected_horizon = float(np.dot(expected_prob, np.asarray(self.candidate_horizons, dtype=np.float64)))
            k_exec = int(np.clip(int(round(expected_horizon)), 1, int(max_exec_length)))
            choose_mode = "qmix_expected_round"
            feedback = float(np.dot(expected_prob, effective))
        elif float(self._rng.random()) < self.ts_epsilon:
            chosen_idx = int(self._rng.integers(0, len(self.candidate_horizons)))
            k_exec = int(self.candidate_horizons[chosen_idx])
            choose_mode = "epsilon_random"
            feedback = self._apply_update(effective, chosen_idx=chosen_idx, k_exec=k_exec)
        elif self.exec_mode == "expected_round":
            expected_prob = normalize_scores(runtime_scores, temp=self.expected_temp)
            expected_horizon = float(np.dot(expected_prob, np.asarray(self.candidate_horizons, dtype=np.float64)))
            k_exec = int(np.clip(int(round(expected_horizon)), 1, int(max_exec_length)))
            choose_mode = "expected_round"
            feedback = self._apply_update(effective, chosen_idx=-1, k_exec=k_exec, expected_prob=expected_prob)
        else:
            chosen_idx = int(np.argmax(runtime_scores))
            k_exec = int(self.candidate_horizons[chosen_idx])
            choose_mode = "thompson"
            feedback = self._apply_update(effective, chosen_idx=chosen_idx, k_exec=k_exec)

        return {
            "exec_length": int(np.clip(k_exec, 1, int(max_exec_length))),
            "choose_mode": choose_mode,
            "runtime_scores": runtime_scores.astype(np.float32),
            "runtime_effective_scores": effective.astype(np.float32),
            "runtime_expected_prob": None if expected_prob is None else expected_prob.astype(np.float32),
            "expected_horizon": expected_horizon,
            "feedback": float(feedback),
        }

    def _gaussian_kernel_weights(self, center: float) -> np.ndarray:
        vals = np.asarray(self.candidate_horizons, dtype=np.float64)
        weights = np.exp(-((vals - float(center)) ** 2) / (2.0 * self.ts_kernel_bandwidth * self.ts_kernel_bandwidth))
        total = float(np.sum(weights))
        if total <= 1e-12 or not np.isfinite(total):
            return np.full((vals.size,), 1.0 / float(max(vals.size, 1)), dtype=np.float64)
        return weights / total

    def _apply_update(
        self,
        effective_scores: np.ndarray,
        *,
        chosen_idx: int,
        k_exec: int,
        expected_prob: np.ndarray | None = None,
    ) -> float:
        if self.ts_update_mode == "independent":
            if expected_prob is not None:
                for index, horizon in enumerate(self.candidate_horizons):
                    prob_i = float(expected_prob[index]) if index < expected_prob.size else 0.0
                    if prob_i <= 0.0:
                        continue
                    score_i = float(effective_scores[index])
                    a, b = self._state[int(horizon)]
                    self._state[int(horizon)] = [a + prob_i * score_i, b + prob_i * (1.0 - score_i)]
                return float(np.dot(expected_prob, effective_scores))
            score = float(effective_scores[chosen_idx])
            a, b = self._state[int(k_exec)]
            self._state[int(k_exec)] = [a + score, b + (1.0 - score)]
            return score

        feedback = float(np.dot(expected_prob, effective_scores)) if expected_prob is not None else float(effective_scores[chosen_idx])
        feedback = float(np.clip(feedback, 0.0, 1.0))
        weights = self._gaussian_kernel_weights(float(k_exec))
        for index, horizon in enumerate(self.candidate_horizons):
            a, b = self._state[int(horizon)]
            incr = self.ts_update_eta * float(weights[index])
            self._state[int(horizon)] = [
                self.ts_forget_rho * a + incr * feedback,
                self.ts_forget_rho * b + incr * (1.0 - feedback),
            ]
        return feedback


class TestTimeHorizonSelector:
    """Client-side port of the reference test-time horizon selector."""

    def __init__(
        self,
        *,
        candidate_horizons: list[int],
        exec_mode: str,
        expected_temp: float,
        ts_epsilon: float,
        ts_seed: int,
        ts_update_mode: str,
        ts_kernel_bandwidth: float,
        ts_forget_rho: float,
        ts_update_eta: float,
        intra_cut_t: float,
        intra_z_mode: str,
        intra_tau_window: int,
        intra_alpha: float,
        inter_use_speed_uniformity: bool,
        mix_mode: str,
        inter_window: int,
        inter_weight: float,
        inter_beta: float,
        inter_fallback: str,
    ) -> None:
        self.candidate_horizons = tuple(int(v) for v in candidate_horizons)
        self.exec_mode = exec_mode
        self.expected_temp = float(max(expected_temp, 1e-6))
        self.ts_epsilon = float(np.clip(ts_epsilon, 0.0, 1.0))
        self.ts_update_mode = ts_update_mode
        self.ts_kernel_bandwidth = float(max(ts_kernel_bandwidth, 1e-6))
        self.ts_forget_rho = float(np.clip(ts_forget_rho, 1e-6, 1.0))
        self.ts_update_eta = float(max(ts_update_eta, 0.0))
        self.intra_cut_t = float(np.clip(intra_cut_t, 0.0, 1.0))
        self.intra_z_mode = intra_z_mode
        self.intra_tau_window = max(1, int(intra_tau_window))
        self.intra_alpha = float(max(intra_alpha, 0.0))
        self.inter_use_speed_uniformity = bool(inter_use_speed_uniformity)
        self.mix_mode = mix_mode
        self.inter_window = max(4, int(inter_window))
        self.inter_weight = float(np.clip(inter_weight, 0.0, 1.0))
        self.inter_beta = float(max(inter_beta, 0.0))
        self.inter_fallback = inter_fallback
        self._rng = np.random.default_rng(int(ts_seed))
        self._state = {int(s): [1.0, 1.0] for s in self.candidate_horizons}
        self._executed_action_history = np.zeros((0, 14), dtype=np.float32)

    def ensure_candidates(self, candidates: list[int]) -> None:
        new_candidates = tuple(int(v) for v in candidates)
        if new_candidates == self.candidate_horizons:
            return
        self.candidate_horizons = new_candidates
        self._state = {int(s): [1.0, 1.0] for s in self.candidate_horizons}

    def record_executed_actions(self, actions: np.ndarray) -> None:
        arr = np.asarray(actions, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] <= 0:
            return
        arr = arr[:, :14]
        self._executed_action_history = np.concatenate([self._executed_action_history, arr], axis=0)
        max_len = max(self.inter_window * 4, max(self.candidate_horizons, default=50) * 4)
        if self._executed_action_history.shape[0] > max_len:
            self._executed_action_history = self._executed_action_history[-max_len:, :]

    def select(self, trace: dict[str, Any], actions: np.ndarray, *, max_exec_length: int) -> dict[str, Any]:
        if "v" not in trace:
            raise ValueError("denoise_trace is missing velocity trace key 'v'")
        v_step = np.asarray(trace["v"], dtype=np.float32)
        if v_step.ndim == 4 and v_step.shape[0] == 1:
            v_step = v_step[0]
        if v_step.ndim != 3:
            raise ValueError(f"Expected denoise_trace['v'] shape [tau,horizon,dim], got {v_step.shape}")

        used_h = max(1, min(int(max_exec_length), int(v_step.shape[1]), int(actions.shape[0])))
        xt_step = np.asarray(actions[:used_h, :14], dtype=np.float32)
        horizon_info = self._compute_horizon_info(v_step[:, :used_h, :14], used_h, xt_step=xt_step)
        k_exec, selected = self._select_from_info(horizon_info, used_h)
        selected["exec_length"] = int(np.clip(k_exec, 1, max_exec_length))
        return selected

    def _ehh_curves_by_horizon_batched(
        self,
        v_step: np.ndarray,
        horizons: list[int],
        *,
        h_full: int,
        eps: float = 1e-12,
    ) -> dict[int, np.ndarray]:
        n_tau, h, action_dim = int(v_step.shape[0]), int(v_step.shape[1]), int(v_step.shape[2])
        h_full = max(int(h_full), h)
        batch = np.zeros((len(horizons), n_tau, h_full, action_dim), dtype=np.float64)
        v64 = np.asarray(v_step, dtype=np.float64)
        for index, horizon in enumerate(horizons):
            s_clip = max(0, min(int(horizon), h))
            if s_clip > 0:
                batch[index, :, :s_clip, :] = v64[:, :s_clip, :]

        spec = np.fft.rfft(batch, axis=2)
        pwr = (spec.real * spec.real) + (spec.imag * spec.imag)
        pwr_f = np.mean(pwr, axis=3)
        freq_count = int(pwr_f.shape[2])
        low_cut = max(1, int(self.intra_cut_t * freq_count))
        e_low = np.sum(pwr_f[:, :, :low_cut], axis=2)
        e_high = np.sum(pwr_f[:, :, low_cut:], axis=2)
        e_vals_all = e_high / (e_low + e_high + eps)
        return {int(horizon): np.asarray(e_vals_all[index], dtype=np.float64) for index, horizon in enumerate(horizons)}

    @staticmethod
    def _z_trend_abs_slope(values: np.ndarray) -> float:
        n = int(values.size)
        if n <= 1:
            return 0.0
        x = np.arange(n, dtype=np.float64)
        x = x - np.mean(x)
        y = values - np.mean(values)
        denom = float(np.sum(x * x))
        if denom <= 1e-12:
            return 0.0
        return abs(float(np.sum(x * y) / denom))

    def _compute_z_from_ehh(self, e_vals: np.ndarray) -> float:
        if e_vals.size <= 1:
            return 0.0
        tail = np.asarray(e_vals, dtype=np.float64)
        if self.intra_z_mode == "trend":
            return self._z_trend_abs_slope(tail)
        if self.intra_z_mode == "tau0_rms":
            baseline = float(tail[0])
        else:
            baseline = float(np.mean(tail[: min(tail.size, self.intra_tau_window)]))
        dev = tail - baseline
        return float(math.sqrt(float(np.mean(dev * dev))))

    @staticmethod
    def _normalize_candidate_metric(values: np.ndarray | list[float], eps: float = 1e-12) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
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

    def _speed_uniformity_proxy_by_horizon(
        self,
        xt_step: np.ndarray,
        horizons: list[int],
        eps: float = 1e-12,
    ) -> tuple[list[float], list[float], list[float]]:
        xt_step = np.asarray(xt_step, dtype=np.float32)
        if xt_step.ndim != 2 or xt_step.shape[1] <= 0:
            nan_list = [float("nan")] * len(horizons)
            return nan_list, nan_list, nan_list

        h = int(xt_step.shape[0])
        action_dim = int(xt_step.shape[1])
        history = self._executed_action_history
        if history.ndim != 2 or int(history.shape[1]) != action_dim:
            history = np.zeros((0, action_dim), dtype=np.float32)
            self._executed_action_history = history
        history_len = int(history.shape[0])
        max_horizon = max([int(s) for s in horizons], default=0)
        history_tail_len = min(history_len, self.inter_window)

        tail = np.asarray(history[-history_tail_len:, :], dtype=np.float32)
        prefix = np.asarray(xt_step[:max_horizon, :], dtype=np.float32)
        seq = np.concatenate([tail, prefix], axis=0)
        if seq.shape[0] >= 2:
            speed = np.linalg.norm(np.diff(seq, axis=0), axis=1).astype(np.float64, copy=False)
        else:
            speed = np.zeros((0,), dtype=np.float64)
        speed_prefix = np.concatenate([[0.0], np.cumsum(speed, dtype=np.float64)])
        speed_sq_prefix = np.concatenate([[0.0], np.cumsum(speed * speed, dtype=np.float64)])

        u_raw: list[float] = []
        for horizon in horizons:
            s = int(horizon)
            pre = int(self.inter_window - s)
            post = int(s)
            if s < 1 or s > h or pre < 1 or post < 1 or history_len < pre or h < post:
                u_raw.append(float("nan"))
                continue
            start = int(history_tail_len - pre)
            speed_start = start
            speed_end = speed_start + (self.inter_window - 1)
            if speed_start < 0 or speed_end > int(speed.shape[0]):
                u_raw.append(float("nan"))
                continue
            count = float(self.inter_window - 1)
            sum_speed = float(speed_prefix[speed_end] - speed_prefix[speed_start])
            sum_speed_sq = float(speed_sq_prefix[speed_end] - speed_sq_prefix[speed_start])
            mu = sum_speed / count
            if mu <= eps:
                u_raw.append(0.0)
                continue
            var = max(0.0, (sum_speed_sq / count) - (mu * mu))
            cv = float(math.sqrt(var) / (mu + eps))
            u_raw.append(cv if np.isfinite(cv) else float("nan"))

        u_norm = self._normalize_candidate_metric(u_raw, eps=eps)
        p_proxy = []
        for value in u_norm:
            if not np.isfinite(value):
                p_proxy.append(float("nan"))
            else:
                p_proxy.append(float(np.clip(np.exp(-self.inter_beta * max(0.0, float(value))), 0.0, 1.0)))
        return u_raw, np.asarray(u_norm, dtype=np.float64).tolist(), p_proxy

    def _compute_horizon_info(self, v_step: np.ndarray, used_h: int, xt_step: np.ndarray | None) -> dict[str, Any]:
        horizons = [s for s in self.candidate_horizons if int(s) <= int(used_h)]
        if not horizons:
            horizons = [int(used_h)]
        self.ensure_candidates(horizons)

        use_inter = self.inter_use_speed_uniformity and self.mix_mode in ("inter", "both") and xt_step is not None
        if use_inter:
            u_vals, u_norm_vals, p_vals = self._speed_uniformity_proxy_by_horizon(xt_step, horizons)
        else:
            u_vals = [float("nan")] * len(horizons)
            u_norm_vals = [float("nan")] * len(horizons)
            p_vals = [float("nan")] * len(horizons)

        ehh = self._ehh_curves_by_horizon_batched(v_step, horizons, h_full=used_h)
        z_vals = [self._compute_z_from_ehh(ehh[int(s)]) for s in horizons]
        z_norm_vals = self._normalize_candidate_metric(z_vals)

        q_intra_vals: list[float] = []
        q_mix_vals: list[float] = []
        for index, _horizon in enumerate(horizons):
            z_norm = float(z_norm_vals[index]) if index < int(z_norm_vals.size) else float("nan")
            q_intra = float(np.exp(-self.intra_alpha * max(0.0, z_norm))) if np.isfinite(z_norm) else float("nan")
            p_inter = float(p_vals[index])
            if self.mix_mode == "intra":
                q_mix = q_intra
            elif self.mix_mode == "inter":
                if np.isfinite(p_inter):
                    q_mix = float(np.clip(p_inter, 0.0, 1.0))
                else:
                    q_mix = 0.5 if self.inter_fallback == "neutral" else q_intra
            else:
                if np.isfinite(p_inter):
                    q_mix = (1.0 - self.inter_weight) * q_intra + self.inter_weight * float(np.clip(p_inter, 0.0, 1.0))
                else:
                    q_mix = (
                        (1.0 - self.inter_weight) * q_intra + self.inter_weight * 0.5
                        if self.inter_fallback == "neutral"
                        else q_intra
                    )
            q_intra_vals.append(float(np.clip(q_intra, 0.0, 1.0)) if np.isfinite(q_intra) else 0.0)
            q_mix_vals.append(float(np.clip(q_mix, 0.0, 1.0)) if np.isfinite(q_mix) else 0.0)

        return {
            "horizons": np.asarray(horizons, dtype=np.int32),
            "z_intra": np.asarray(z_vals, dtype=np.float32),
            "z_intra_norm": np.asarray(z_norm_vals, dtype=np.float32),
            "u_inter_raw": np.asarray(u_vals, dtype=np.float32),
            "u_inter_norm": np.asarray(u_norm_vals, dtype=np.float32),
            "q_intra": np.asarray(q_intra_vals, dtype=np.float32),
            "p_inter": np.asarray(p_vals, dtype=np.float32),
            "q_mix": np.asarray(q_mix_vals, dtype=np.float32),
        }

    def _gaussian_kernel_weights(self, center: float) -> np.ndarray:
        vals = np.asarray(self.candidate_horizons, dtype=np.float64)
        if vals.size <= 0:
            return np.zeros((0,), dtype=np.float64)
        weights = np.exp(-((vals - float(center)) ** 2) / (2.0 * self.ts_kernel_bandwidth * self.ts_kernel_bandwidth))
        total = float(np.sum(weights))
        if total <= 1e-12 or not np.isfinite(total):
            return np.full((vals.size,), 1.0 / float(vals.size), dtype=np.float64)
        return weights / total

    def _apply_update(
        self,
        horizons: list[int],
        q_mix: np.ndarray,
        *,
        chosen_idx: int,
        choose_mode: str,
        k_exec: int,
        expected_prob: np.ndarray | None = None,
    ) -> float:
        if self.ts_update_mode == "independent":
            if choose_mode == "expected_round" and expected_prob is not None:
                for index, horizon in enumerate(horizons):
                    prob_i = float(expected_prob[index]) if index < int(expected_prob.size) else 0.0
                    if prob_i <= 0.0:
                        continue
                    score_i = float(q_mix[index])
                    a, b = self._state[int(horizon)]
                    self._state[int(horizon)] = [a + prob_i * score_i, b + prob_i * (1.0 - score_i)]
                return float(np.dot(expected_prob, q_mix))
            score = float(q_mix[chosen_idx])
            a, b = self._state[int(k_exec)]
            self._state[int(k_exec)] = [a + score, b + (1.0 - score)]
            return score

        feedback = float(np.dot(expected_prob, q_mix)) if expected_prob is not None else float(q_mix[chosen_idx])
        feedback = float(np.clip(feedback, 0.0, 1.0))
        weights = self._gaussian_kernel_weights(float(k_exec))
        for index, horizon in enumerate(self.candidate_horizons):
            a, b = self._state[int(horizon)]
            incr = self.ts_update_eta * float(weights[index])
            self._state[int(horizon)] = [
                self.ts_forget_rho * a + incr * feedback,
                self.ts_forget_rho * b + incr * (1.0 - feedback),
            ]
        return feedback

    def _select_from_info(self, horizon_info: dict[str, Any], used_h: int) -> tuple[int, dict[str, Any]]:
        horizons = [int(v) for v in np.asarray(horizon_info.get("horizons", []), dtype=np.int32).reshape(-1).tolist()]
        if not horizons:
            horizons = [int(used_h)]
        q_mix = np.asarray(horizon_info.get("q_mix", []), dtype=np.float64).reshape(-1)
        if q_mix.size != len(horizons):
            q_mix = np.ones((len(horizons),), dtype=np.float64)
        self.ensure_candidates(horizons)

        if self.exec_mode == "qmix_expected_round":
            scores = np.clip(q_mix, 0.0, 1.0)
        else:
            theta = np.asarray([self._rng.beta(*self._state[int(s)]) for s in horizons], dtype=np.float64)
            scores = theta * np.clip(q_mix, 0.0, 1.0)
        expected_prob = normalize_scores(scores, temp=self.expected_temp)
        expected_horizon = float(np.dot(expected_prob, np.asarray(horizons, dtype=np.float64)))

        if self.exec_mode == "qmix_expected_round":
            k_exec = int(np.clip(int(round(expected_horizon)), 1, int(used_h)))
            choose_mode = "qmix_expected_round"
            feedback = float("nan")
        elif float(self._rng.random()) < self.ts_epsilon:
            chosen_idx = int(self._rng.integers(0, len(horizons)))
            k_exec = int(horizons[chosen_idx])
            choose_mode = "epsilon_random"
            feedback = self._apply_update(horizons, q_mix, chosen_idx=chosen_idx, choose_mode=choose_mode, k_exec=k_exec)
        elif self.exec_mode == "expected_round":
            k_exec = int(np.clip(int(round(expected_horizon)), 1, int(used_h)))
            choose_mode = "expected_round"
            feedback = self._apply_update(
                horizons,
                q_mix,
                chosen_idx=-1,
                choose_mode=choose_mode,
                k_exec=k_exec,
                expected_prob=expected_prob,
            )
        else:
            chosen_idx = int(np.argmax(scores))
            k_exec = int(horizons[chosen_idx])
            choose_mode = "thompson"
            feedback = self._apply_update(horizons, q_mix, chosen_idx=chosen_idx, choose_mode=choose_mode, k_exec=k_exec)

        out = dict(horizon_info)
        out.update(
            {
                "scores": np.asarray(scores, dtype=np.float32),
                "expected_prob": np.asarray(expected_prob, dtype=np.float32),
                "expected_horizon": expected_horizon,
                "choose_mode": choose_mode,
                "chosen_horizon": int(k_exec),
                "feedback": feedback,
            }
        )
        return k_exec, out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--state-format", choices=["raw26"], default="raw26")
    parser.add_argument("--rollout-mode", choices=["chunk"], default="chunk")
    parser.add_argument("--chunk-prefix-steps", type=int, default=5)
    parser.add_argument("--episode-max-action-steps", type=int, default=25)
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--policy-alpha", type=float, default=1.0)
    parser.add_argument("--action-ema", type=float, default=0.0)
    parser.add_argument("--max-joint-delta-rad", type=float, default=0.03)
    parser.add_argument("--max-gripper-delta", type=float, default=0.005)
    parser.add_argument("--allow-dual-arm", action="store_true")
    parser.add_argument("--allow-gripper", action="store_true")
    parser.add_argument("--single-arm-only", choices=["left", "right"], default=None)
    parser.add_argument("--post-check-sec", type=float, default=0.0)
    parser.add_argument("--ctrl-c-soft-reset", type=int, choices=[0, 1], default=1)
    parser.add_argument("--soft-reset-repeat", type=int, default=10)
    parser.add_argument("--soft-reset-sleep-sec", type=float, default=0.05)
    parser.add_argument("--progress-style", choices=["compact", "verbose"], default=None)
    parser.add_argument("--motion-watchdog", type=int, choices=[0, 1], default=1)
    parser.add_argument("--motion-watchdog-steps", type=int, default=20)
    parser.add_argument("--motion-command-threshold-rad", type=float, default=0.01)
    parser.add_argument("--motion-observed-threshold-rad", type=float, default=0.003)
    parser.add_argument("--ahs-candidates", default="10,20,30,40,50")
    parser.add_argument(
        "--ahs-exec-mode",
        choices=["expected_round", "thompson", "qmix_expected_round"],
        default="expected_round",
    )
    parser.add_argument("--ahs-expected-temp", type=float, default=1.0)
    parser.add_argument("--ahs-ts-epsilon", type=float, default=0.05)
    parser.add_argument("--ahs-ts-seed", type=int, default=0)
    parser.add_argument("--ahs-ts-update-mode", choices=["independent", "kernel_forget"], default="kernel_forget")
    parser.add_argument("--ahs-ts-kernel-bandwidth", type=float, default=10.0)
    parser.add_argument("--ahs-ts-forget-rho", type=float, default=0.99)
    parser.add_argument("--ahs-ts-update-eta", type=float, default=1.0)
    parser.add_argument("--ahs-prior-gamma", type=float, default=1.0)
    parser.add_argument("--ahs-metadata-policy", choices=["fallback", "require"], default="fallback")
    parser.add_argument("--horizon-intra-cut-t", type=float, default=0.25)
    parser.add_argument("--horizon-intra-z-mode", choices=["tau0_rms", "window_rms", "trend"], default="window_rms")
    parser.add_argument("--horizon-intra-tau-window", type=int, default=3)
    parser.add_argument("--horizon-intra-alpha", type=float, default=4.0)
    parser.add_argument("--horizon-inter-use-speed-uniformity", type=int, choices=[0, 1], default=1)
    parser.add_argument("--horizon-mix-mode", choices=["intra", "inter", "both"], default="both")
    parser.add_argument("--horizon-inter-window", type=int, default=60)
    parser.add_argument("--horizon-inter-weight", type=float, default=0.5)
    parser.add_argument("--horizon-inter-beta", type=float, default=4.0)
    parser.add_argument("--horizon-inter-fallback", choices=["ehh_only", "neutral"], default="ehh_only")
    add_recording_args(parser)
    parser.add_argument("--execute", type=int, choices=[0, 1], default=0)
    return parser.parse_args(argv)


def parse_candidates(raw: str) -> list[int]:
    candidates: list[int] = []
    raw = str(raw).replace("：", ":")
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            fields = [field.strip() for field in part.split(":")]
            if len(fields) not in (2, 3):
                continue
            try:
                start = int(fields[0])
                stop = int(fields[1])
                step = int(fields[2]) if len(fields) == 3 else 1
            except ValueError:
                continue
            if step == 0:
                continue
            if start <= stop and step > 0:
                candidates.extend(range(start, stop + 1, step))
            elif start >= stop and step < 0:
                candidates.extend(range(start, stop - 1, step))
            continue
        try:
            value = int(part)
        except ValueError:
            continue
        if value > 0:
            candidates.append(value)
    return sorted(set(candidates)) or [10, 20, 30, 40, 50]


def normalize_probabilities(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = np.clip(arr, 0.0, None)
    total = float(np.sum(arr))
    if total <= 1e-12 or not np.isfinite(total):
        return np.full((arr.size,), 1.0 / float(max(arr.size, 1)), dtype=np.float32)
    return (arr / total).astype(np.float32)


def normalize_scores(score_arr: np.ndarray, *, temp: float = 1.0) -> np.ndarray:
    arr = np.asarray(score_arr, dtype=np.float64).reshape(-1)
    if arr.size <= 0:
        return np.zeros((0,), dtype=np.float64)
    arr = np.maximum(arr, 0.0)
    total = float(np.sum(arr))
    if total <= 1e-12 or not np.isfinite(total):
        return np.full((arr.size,), 1.0 / float(arr.size), dtype=np.float64)
    base = arr / total
    if abs(float(temp) - 1.0) <= 1e-12:
        return base
    sharpened = np.power(np.clip(base, 0.0, 1.0), 1.0 / float(max(1e-6, temp)))
    sharpened_total = float(np.sum(sharpened))
    if sharpened_total <= 1e-12 or not np.isfinite(sharpened_total):
        return np.full((arr.size,), 1.0 / float(arr.size), dtype=np.float64)
    return sharpened / sharpened_total


def compress_posterior_to_basis(source_horizons: np.ndarray, posterior: np.ndarray, target_horizons: np.ndarray) -> np.ndarray:
    source = np.asarray(source_horizons, dtype=np.float64).reshape(-1)
    post = np.asarray(posterior, dtype=np.float64).reshape(-1)
    target = np.asarray(target_horizons, dtype=np.float64).reshape(-1)
    if source.shape != post.shape:
        raise ValueError(f"source/posterior mismatch: {source.shape} vs {post.shape}")
    bucket_idx = np.argmin(np.abs(source[:, None] - target[None, :]), axis=1)
    out = np.zeros((target.size,), dtype=np.float64)
    np.add.at(out, bucket_idx, post)
    return normalize_probabilities(out)


def clip_step(current: np.ndarray, desired: np.ndarray, *, max_joint_delta: float, max_gripper_delta: float) -> np.ndarray:
    delta = desired - current
    delta[:6] = np.clip(delta[:6], -max_joint_delta, max_joint_delta)
    delta[6] = np.clip(delta[6], -max_gripper_delta, max_gripper_delta)
    return current + delta


def target_from_policy_step(
    current_left: np.ndarray,
    current_right: np.ndarray,
    policy_step: np.ndarray,
    prev_left: np.ndarray | None,
    prev_right: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    desired_left = policy_step[0:7].astype(np.float32)
    desired_right = policy_step[7:14].astype(np.float32)

    desired_left = current_left + args.policy_alpha * (desired_left - current_left)
    desired_right = current_right + args.policy_alpha * (desired_right - current_right)

    if args.action_ema > 0 and prev_left is not None and prev_right is not None:
        ema = args.action_ema
        desired_left = ema * prev_left + (1.0 - ema) * desired_left
        desired_right = ema * prev_right + (1.0 - ema) * desired_right

    target_left = clip_step(
        current_left,
        desired_left,
        max_joint_delta=args.max_joint_delta_rad,
        max_gripper_delta=args.max_gripper_delta,
    )
    target_right = clip_step(
        current_right,
        desired_right,
        max_joint_delta=args.max_joint_delta_rad,
        max_gripper_delta=args.max_gripper_delta,
    )

    if args.single_arm_only == "left":
        target_right = current_right.copy()
    elif args.single_arm_only == "right":
        target_left = current_left.copy()
    elif not args.allow_dual_arm:
        target_left = current_left.copy()

    if not args.allow_gripper:
        target_left[6] = current_left[6]
        target_right[6] = current_right[6]

    return target_left, target_right


def choose_ahs_exec_length(
    result: dict[str, Any],
    *,
    actions: np.ndarray,
    actions_shape: tuple[int, ...],
    remaining_steps: int,
    selector: AHSRuntimeSelector,
    horizon_selector: TestTimeHorizonSelector,
    args: argparse.Namespace,
) -> AHSChoice:
    max_exec_length = max(1, min(int(actions_shape[0]), int(remaining_steps)))
    fallback_length = max(1, min(int(args.chunk_prefix_steps), max_exec_length))

    if "exec_length" in result:
        exec_length = int(np.clip(int(np.asarray(result["exec_length"]).item()), 1, max_exec_length))
        return AHSChoice(
            exec_length=exec_length,
            source="server_exec_length",
            mode=str(result.get("qha_exec_mode", "server")),
            expected_horizon=scalar_or_none(result.get("qha_expected_horizon")),
        )

    if "denoise_trace" in result:
        try:
            selected = horizon_selector.select(
                result["denoise_trace"],
                actions,
                max_exec_length=max_exec_length,
            )
        except (TypeError, ValueError) as exc:
            if args.ahs_metadata_policy == "require":
                raise RuntimeError(f"test-time horizon metadata unusable: {exc}") from exc
            return AHSChoice(
                exec_length=fallback_length,
                source="fallback",
                mode="fixed_prefix",
                inactive_reason=f"unusable denoise_trace: {exc}",
            )
        candidates = np.asarray(selected.get("horizons", []), dtype=np.int32).reshape(-1)
        q_mix = np.asarray(selected.get("q_mix", []), dtype=np.float32).reshape(-1)
        return AHSChoice(
            exec_length=int(selected["exec_length"]),
            source="denoise_trace",
            mode=str(selected.get("choose_mode", args.ahs_exec_mode)),
            candidates=candidates if candidates.size else None,
            q_mix=q_mix if q_mix.size else None,
            expected_horizon=scalar_or_none(selected.get("expected_horizon")),
            feedback=scalar_or_none(selected.get("feedback")),
            details=selected,
        )

    if "qha_posterior" not in result or "qha_candidate_horizons" not in result:
        reason = "missing denoise_trace/qha_posterior/qha_candidate_horizons"
        if args.ahs_metadata_policy == "require":
            raise RuntimeError(f"AHS metadata required but unavailable: {reason}")
        return AHSChoice(
            exec_length=fallback_length,
            source="fallback",
            mode="fixed_prefix",
            inactive_reason=reason,
        )

    candidates = np.asarray(result["qha_candidate_horizons"], dtype=np.int32).reshape(-1)
    posterior = normalize_probabilities(np.asarray(result["qha_posterior"], dtype=np.float32).reshape(-1))
    if candidates.shape != posterior.shape:
        raise RuntimeError(f"qha_candidate_horizons/qha_posterior shape mismatch: {candidates.shape} vs {posterior.shape}")

    runtime_candidates = candidates
    runtime_posterior = posterior
    q_mix = None

    if "qha_selector_q_mix" in result:
        q_mix_candidate = np.asarray(result["qha_selector_q_mix"], dtype=np.float32).reshape(-1)
        if q_mix_candidate.shape == posterior.shape:
            q_mix = normalize_probabilities(q_mix_candidate)
    if q_mix is None and "qha_selector_q_mix_sparse_runtime" in result:
        q_mix_sparse = np.asarray(result["qha_selector_q_mix_sparse_runtime"], dtype=np.float32).reshape(-1)
        sparse_candidates = np.asarray(
            result.get("qha_selector_candidate_horizons_sparse_runtime", []),
            dtype=np.int32,
        ).reshape(-1)
        if sparse_candidates.shape == q_mix_sparse.shape and sparse_candidates.size > 0:
            runtime_candidates = sparse_candidates
            runtime_posterior = compress_posterior_to_basis(candidates, posterior, runtime_candidates)
            q_mix = normalize_probabilities(q_mix_sparse)
    if q_mix is None:
        q_mix = posterior.copy()

    candidate_list = [int(v) for v in runtime_candidates.tolist() if int(v) > 0]
    if not candidate_list:
        raise RuntimeError("AHS candidate list is empty after metadata parsing.")
    selector.ensure_candidates(candidate_list)
    selected = selector.select(runtime_posterior, q_mix, max_exec_length=max_exec_length)
    expected = selected.get("expected_horizon")
    if expected is None and selected.get("runtime_expected_prob") is not None:
        expected = float(
            np.dot(
                np.asarray(selected["runtime_expected_prob"], dtype=np.float64),
                np.asarray(candidate_list, dtype=np.float64),
            )
        )

    return AHSChoice(
        exec_length=int(selected["exec_length"]),
        source="qha",
        mode=str(selected["choose_mode"]),
        candidates=np.asarray(candidate_list, dtype=np.int32),
        posterior=np.asarray(runtime_posterior, dtype=np.float32),
        q_mix=np.asarray(q_mix, dtype=np.float32),
        expected_horizon=None if expected is None else float(expected),
        feedback=selected.get("feedback"),
    )


def scalar_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(np.asarray(value).item())
    except (TypeError, ValueError):
        return None


def progress_style(args: argparse.Namespace) -> str:
    if args.progress_style is not None:
        return args.progress_style
    return "compact" if args.execute else "verbose"


def max_joint_delta(current: np.ndarray, target: np.ndarray) -> float:
    return float(np.max(np.abs(target[:6] - current[:6])))


def gripper_delta(current: np.ndarray, target: np.ndarray) -> float:
    return float(target[6] - current[6])


def total_replans(args: argparse.Namespace, ahs_candidates: list[int] | None = None) -> int:
    if ahs_candidates:
        chunk = max(1, min(int(v) for v in ahs_candidates if int(v) > 0))
    else:
        chunk = max(1, args.chunk_prefix_steps)
    return (args.episode_max_action_steps + chunk - 1) // chunk


def print_replan_header(
    *,
    replan_index: int,
    replan_total: int,
    actions_shape: tuple[int, ...],
    prefix_steps: int,
    executed_steps: int,
    choice: AHSChoice,
    args: argparse.Namespace,
    style: str,
) -> None:
    if style == "compact":
        print(
            f"\n[replan {replan_index:02d}/{replan_total:02d}] "
            f"policy_chunk={actions_shape}, taking={prefix_steps}, "
            f"total_step={executed_steps:03d}/{args.episode_max_action_steps}"
        )
    else:
        print(
            f"chunk actions shape: {actions_shape}; taking {prefix_steps}; "
            f"source={choice.source}; mode={choice.mode}; execute: {args.execute}"
        )
    if style == "compact":
        print(f"horizon: source={choice.source}, mode={choice.mode}, ahs_exec={choice.exec_length}")
    print_ahs_details(choice)


def print_ahs_details(choice: AHSChoice) -> None:
    if choice.inactive_reason:
        print(f"AHS inactive: {choice.inactive_reason}; using fixed-prefix fallback")
        return
    if choice.source == "denoise_trace" and choice.details is not None:
        details = choice.details
        candidates = np.asarray(details.get("horizons", []), dtype=np.int32).reshape(-1)
        q_mix = np.asarray(details.get("q_mix", []), dtype=np.float32).reshape(-1)
        z_norm = np.asarray(details.get("z_intra_norm", []), dtype=np.float32).reshape(-1)
        p_inter = np.asarray(details.get("p_inter", []), dtype=np.float32).reshape(-1)
        scores = np.asarray(details.get("scores", []), dtype=np.float32).reshape(-1)
        msg = f"horizon_candidates={np.array2string(candidates, separator=' ')}"
        if q_mix.size:
            msg += f", q_mix={np.array2string(q_mix, precision=4, suppress_small=False)}"
        if z_norm.size:
            msg += f", z_norm={np.array2string(z_norm, precision=4, suppress_small=False)}"
        if p_inter.size:
            msg += f", p_inter={np.array2string(p_inter, precision=4, suppress_small=False)}"
        if scores.size:
            msg += f", scores={np.array2string(scores, precision=4, suppress_small=False)}"
        if choice.expected_horizon is not None:
            msg += f", expected={choice.expected_horizon:.3f}"
        if choice.feedback is not None and np.isfinite(choice.feedback):
            msg += f", feedback={choice.feedback:.4f}"
        print(msg)
        return
    if choice.candidates is None or choice.posterior is None:
        if choice.expected_horizon is not None:
            print(f"ahs_expected_horizon={choice.expected_horizon:.3f}")
        return
    posterior = np.array2string(choice.posterior, precision=4, suppress_small=False)
    candidates = np.array2string(choice.candidates, separator=" ")
    msg = f"qha_candidates={candidates}, posterior={posterior}"
    if choice.q_mix is not None:
        msg += f", q_mix={np.array2string(choice.q_mix, precision=4, suppress_small=False)}"
    if choice.expected_horizon is not None:
        msg += f", expected={choice.expected_horizon:.3f}"
    if choice.feedback is not None:
        msg += f", feedback={choice.feedback:.4f}"
    print(msg)


def ahs_log_fields(choice: AHSChoice, *, executed_length: int) -> dict[str, Any]:
    details = choice.details or {}
    candidates = details.get("horizons", choice.candidates)
    q_mix = details.get("q_mix", choice.q_mix)
    return {
        "exec_length": int(executed_length),
        "source": choice.source,
        "expected_horizon": choice.expected_horizon,
        "feedback": choice.feedback,
        "candidates": candidates,
        "scores": details.get("scores", choice.posterior),
        "q_mix": q_mix,
        "z_norm": details.get("z_intra_norm"),
        "p_inter": details.get("p_inter"),
    }


def beta_state_snapshot(selector: Any) -> dict[int, tuple[float, float]]:
    state = getattr(selector, "_state", {})
    snapshot: dict[int, tuple[float, float]] = {}
    for key, value in state.items():
        try:
            a, b = value
            snapshot[int(key)] = (float(a), float(b))
        except (TypeError, ValueError):
            continue
    return snapshot


def beta_candidates_for_choice(choice: AHSChoice, fallback_candidates: list[int]) -> np.ndarray:
    details = choice.details or {}
    candidates = details.get("horizons", choice.candidates)
    if candidates is None:
        candidates = fallback_candidates
    arr = np.asarray(candidates, dtype=np.int32).reshape(-1)
    if arr.size == 0:
        arr = np.asarray(fallback_candidates, dtype=np.int32).reshape(-1)
    return arr


def beta_ab_from_snapshot(snapshot: dict[int, tuple[float, float]], candidates: np.ndarray) -> np.ndarray:
    out = np.full((int(candidates.size), 2), np.nan, dtype=np.float32)
    for index, candidate in enumerate(candidates.astype(np.int32).tolist()):
        if int(candidate) in snapshot:
            out[index, :] = np.asarray(snapshot[int(candidate)], dtype=np.float32)
    return out


def print_step_progress(
    *,
    replan_index: int,
    replan_total: int,
    chunk_index: int,
    prefix_steps: int,
    executed_steps: int,
    current_left: np.ndarray,
    current_right: np.ndarray,
    target_left: np.ndarray,
    target_right: np.ndarray,
    args: argparse.Namespace,
    style: str,
) -> None:
    if style == "compact":
        line = (
            f"\rreplan {replan_index:02d}/{replan_total:02d} | "
            f"chunk {chunk_index + 1:02d}/{prefix_steps:02d} | "
            f"step {executed_steps + 1:04d}/{args.episode_max_action_steps:04d} | "
            f"dL={max_joint_delta(current_left, target_left):.4f} "
            f"dR={max_joint_delta(current_right, target_right):.4f} | "
            f"gL={gripper_delta(current_left, target_left):+.4f} "
            f"gR={gripper_delta(current_right, target_right):+.4f} | "
            f"{args.hz:g}Hz"
        )
        print(line, end="", flush=True)
        return

    print(f"step {executed_steps + 1}/{args.episode_max_action_steps} chunk_index={chunk_index}")
    print_delta("left", current_left, target_left)
    print_delta("right", current_right, target_right)


def finish_compact_line(style: str) -> None:
    if style == "compact":
        print()


def observed_joint_motion(start_left: np.ndarray, start_right: np.ndarray, now_left: np.ndarray, now_right: np.ndarray) -> float:
    left_motion = float(np.max(np.abs(now_left[:6] - start_left[:6])))
    right_motion = float(np.max(np.abs(now_right[:6] - start_right[:6])))
    return max(left_motion, right_motion)


def make_ahs_selectors(args: argparse.Namespace, ahs_candidates: list[int]) -> tuple[AHSRuntimeSelector, TestTimeHorizonSelector]:
    """Build both selectors with the original synchronous rollout parameters."""
    ahs_selector = AHSRuntimeSelector(
        candidate_horizons=ahs_candidates,
        exec_mode=args.ahs_exec_mode,
        expected_temp=args.ahs_expected_temp,
        ts_epsilon=args.ahs_ts_epsilon,
        ts_seed=args.ahs_ts_seed,
        ts_update_mode=args.ahs_ts_update_mode,
        ts_kernel_bandwidth=args.ahs_ts_kernel_bandwidth,
        ts_forget_rho=args.ahs_ts_forget_rho,
        ts_update_eta=args.ahs_ts_update_eta,
        prior_gamma=args.ahs_prior_gamma,
    )
    horizon_selector = TestTimeHorizonSelector(
        candidate_horizons=ahs_candidates,
        exec_mode=args.ahs_exec_mode,
        expected_temp=args.ahs_expected_temp,
        ts_epsilon=args.ahs_ts_epsilon,
        ts_seed=args.ahs_ts_seed,
        ts_update_mode=args.ahs_ts_update_mode,
        ts_kernel_bandwidth=args.ahs_ts_kernel_bandwidth,
        ts_forget_rho=args.ahs_ts_forget_rho,
        ts_update_eta=args.ahs_ts_update_eta,
        intra_cut_t=args.horizon_intra_cut_t,
        intra_z_mode=args.horizon_intra_z_mode,
        intra_tau_window=args.horizon_intra_tau_window,
        intra_alpha=args.horizon_intra_alpha,
        inter_use_speed_uniformity=bool(args.horizon_inter_use_speed_uniformity),
        mix_mode=args.horizon_mix_mode,
        inter_window=args.horizon_inter_window,
        inter_weight=args.horizon_inter_weight,
        inter_beta=args.horizon_inter_beta,
        inter_fallback=args.horizon_inter_fallback,
    )
    return ahs_selector, horizon_selector


def main() -> None:
    args = parse_args()
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    ahs_candidates = parse_candidates(args.ahs_candidates)
    ahs_selector, horizon_selector = make_ahs_selectors(args, ahs_candidates)
    robot = make_robot()
    recorder = create_eval_video_recorder(args, robot, rollout_kind="ahs")
    if recorder is not None:
        recorder.start()
    npz_logger = EvalEpisodeNpzLogger(recorder=recorder, rollout_kind="ahs", args=args) if recorder is not None else None
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    period = 1.0 / args.hz if args.hz > 0 else 0.0
    executed_steps = 0
    replan_index = 0
    replan_total = total_replans(args, ahs_candidates)
    style = progress_style(args)
    prev_left: np.ndarray | None = None
    prev_right: np.ndarray | None = None
    watchdog_start_left: np.ndarray | None = None
    watchdog_start_right: np.ndarray | None = None
    watchdog_commanded = False
    recorder_status = "error"

    try:
        print(f"task prompt: {args.prompt}")
        print(
            "AHS config: "
            f"candidates={ahs_candidates}, mode={args.ahs_exec_mode}, "
            f"metadata_policy={args.ahs_metadata_policy}, "
            f"horizon_mix={args.horizon_mix_mode}, z_mode={args.horizon_intra_z_mode}"
        )
        while executed_steps < args.episode_max_action_steps:
            observation = get_complete_observation(robot)
            raw26 = observation_to_raw26(observation)
            policy_input = observation_to_policy_input(observation, raw26)
            policy_input["prompt"] = args.prompt
            result = client.infer(policy_input)
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] < 14:
                raise RuntimeError(f"Expected policy actions [horizon, >=14], got {actions.shape}")

            beta_before_horizon = beta_state_snapshot(horizon_selector)
            beta_before_qha = beta_state_snapshot(ahs_selector)
            choice = choose_ahs_exec_length(
                result,
                actions=actions,
                actions_shape=actions.shape,
                remaining_steps=args.episode_max_action_steps - executed_steps,
                selector=ahs_selector,
                horizon_selector=horizon_selector,
                args=args,
            )
            beta_after_horizon = beta_state_snapshot(horizon_selector)
            beta_after_qha = beta_state_snapshot(ahs_selector)
            prefix_steps = min(choice.exec_length, actions.shape[0], args.episode_max_action_steps - executed_steps)
            replan_index += 1
            print_replan_header(
                replan_index=replan_index,
                replan_total=replan_total,
                actions_shape=actions.shape,
                prefix_steps=prefix_steps,
                executed_steps=executed_steps,
                choice=choice,
                args=args,
                style=style,
            )
            ahs_step_log = ahs_log_fields(choice, executed_length=prefix_steps)
            beta_candidates = beta_candidates_for_choice(choice, ahs_candidates)
            if choice.source == "denoise_trace":
                beta_before = beta_ab_from_snapshot(beta_before_horizon, beta_candidates)
                beta_after = beta_ab_from_snapshot(beta_after_horizon, beta_candidates)
            elif choice.source == "qha":
                beta_before = beta_ab_from_snapshot(beta_before_qha, beta_candidates)
                beta_after = beta_ab_from_snapshot(beta_after_qha, beta_candidates)
            else:
                beta_before = np.full((int(beta_candidates.size), 2), np.nan, dtype=np.float32)
                beta_after = np.full((int(beta_candidates.size), 2), np.nan, dtype=np.float32)
            ahs_replan_log = dict(ahs_step_log)
            ahs_replan_log.update(
                {
                    "beta_candidates": beta_candidates,
                    "beta_ab_before": beta_before,
                    "beta_ab_after": beta_after,
                }
            )
            if npz_logger is not None:
                npz_logger.log_replan(
                    replan_idx=replan_index - 1,
                    env_step_idx=executed_steps,
                    selected_k=prefix_steps,
                    monotonic_timestamp_ns=time.monotonic_ns(),
                    ahs=ahs_replan_log,
                )

            for chunk_index in range(prefix_steps):
                loop_start = time.perf_counter()
                step_start_time_ns = time.time_ns()
                step_start_monotonic_ns = time.monotonic_ns()
                current_obs = get_complete_observation(robot)
                current_raw26 = observation_to_raw26(current_obs)
                current_left, current_right = raw26_to_left_right(current_raw26)
                if watchdog_start_left is None or watchdog_start_right is None:
                    watchdog_start_left = current_left.copy()
                    watchdog_start_right = current_right.copy()
                target_left, target_right = target_from_policy_step(
                    current_left,
                    current_right,
                    actions[chunk_index],
                    prev_left,
                    prev_right,
                    args,
                )
                print_step_progress(
                    replan_index=replan_index,
                    replan_total=replan_total,
                    chunk_index=chunk_index,
                    prefix_steps=prefix_steps,
                    executed_steps=executed_steps,
                    current_left=current_left,
                    current_right=current_right,
                    target_left=target_left,
                    target_right=target_right,
                    args=args,
                    style=style,
                )

                if args.execute:
                    robot.send_action(action_dict_from_left_right(target_left, target_right))

                if npz_logger is not None:
                    npz_logger.log_step(
                        step_index=executed_steps,
                        replan_index=replan_index,
                        chunk_index=chunk_index,
                        step_start_time_ns=step_start_time_ns,
                        step_start_monotonic_ns=step_start_monotonic_ns,
                        state_raw26_before=current_raw26,
                        policy_action_lr14=actions[chunk_index, :14],
                        executed_action_lr14=np.concatenate([target_left, target_right]),
                        execute_enabled=bool(args.execute),
                        ahs=ahs_step_log,
                    )

                commanded_motion = max(max_joint_delta(current_left, target_left), max_joint_delta(current_right, target_right))
                watchdog_commanded = watchdog_commanded or commanded_motion >= args.motion_command_threshold_rad
                prev_left = target_left
                prev_right = target_right
                executed_steps += 1

                sleep_time = period - (time.perf_counter() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                if (
                    args.execute
                    and args.motion_watchdog
                    and watchdog_commanded
                    and executed_steps >= args.motion_watchdog_steps
                    and watchdog_start_left is not None
                    and watchdog_start_right is not None
                ):
                    check_obs = get_complete_observation(robot)
                    check_left, check_right = raw26_to_left_right(observation_to_raw26(check_obs))
                    observed = observed_joint_motion(watchdog_start_left, watchdog_start_right, check_left, check_right)
                    if observed < args.motion_observed_threshold_rad:
                        finish_compact_line(style)
                        raise RuntimeError(
                            "Motion watchdog stopped rollout: commands were sent, but observed joint motion was "
                            f"{observed:.5f} rad after {executed_steps} steps. Check Piper enable/control mode, "
                            "Dora piper node action input, and CAN wiring before running policy again."
                        )
                    watchdog_start_left = check_left.copy()
                    watchdog_start_right = check_right.copy()
                    watchdog_commanded = False
                if executed_steps >= args.episode_max_action_steps:
                    break
            horizon_selector.record_executed_actions(actions[:prefix_steps, :14])
            finish_compact_line(style)

        if args.post_check_sec > 0:
            time.sleep(args.post_check_sec)
        final_obs = get_complete_observation(robot)
        final_left, final_right = raw26_to_left_right(observation_to_raw26(final_obs))
        print("final state sample:")
        print(f"left={np.array2string(final_left, precision=5)}")
        print(f"right={np.array2string(final_right, precision=5)}")
        recorder_status = "completed"
    except KeyboardInterrupt:
        recorder_status = "interrupted"
        finish_compact_line(style)
        print("KeyboardInterrupt received.")
        if args.ctrl_c_soft_reset and args.execute:
            soft_reset_hold_current(robot, args)
        elif args.ctrl_c_soft_reset and not args.execute:
            print("ctrl-c soft reset skipped because execute=0")
        else:
            print("ctrl-c soft reset disabled")
    finally:
        if recorder is not None:
            recorder.stop(status=recorder_status)
        if npz_logger is not None:
            try:
                npz_path = npz_logger.save(status=recorder_status)
                if npz_path is not None:
                    print(f"episode npz saved: {npz_path}")
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: failed to save episode_data.npz: {exc}")
        if robot.is_connected:
            robot.disconnect()


def soft_reset_hold_current(robot, args: argparse.Namespace) -> None:
    print(
        f"ctrl-c soft reset: holding current pose "
        f"repeat={args.soft_reset_repeat}, sleep={args.soft_reset_sleep_sec}"
    )
    for index in range(max(1, args.soft_reset_repeat)):
        observation = get_complete_observation(robot)
        current_left, current_right = raw26_to_left_right(observation_to_raw26(observation))
        robot.send_action(action_dict_from_left_right(current_left, current_right))
        print(f"ctrl-c soft reset sent {index + 1}/{args.soft_reset_repeat}")
        if args.soft_reset_sleep_sec > 0 and index + 1 < args.soft_reset_repeat:
            time.sleep(args.soft_reset_sleep_sec)


if __name__ == "__main__":
    main()
