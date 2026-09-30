"""NumPy QHA prior fusion, extracted without numerical changes from OpenPI integration."""
from __future__ import annotations
from typing import Any, Sequence
import numpy as np

def _normalize_score_distribution_np(score_arr: np.ndarray, temp: float = 1.0) -> np.ndarray:
    arr = np.asarray(score_arr, dtype=np.float64)
    arr = np.maximum(arr, 0.0)
    arr_sum = float(np.sum(arr))
    if arr_sum <= 1e-12 or (not np.isfinite(arr_sum)):
        return np.full((arr.size,), 1.0 / float(max(arr.size, 1)), dtype=np.float64)
    base = arr / arr_sum
    if abs(float(temp) - 1.0) <= 1e-12:
        return base
    sharp = np.power(np.clip(base, 0.0, 1.0), 1.0 / float(max(1e-6, temp)))
    sharp_sum = float(np.sum(sharp))
    if sharp_sum <= 1e-12 or (not np.isfinite(sharp_sum)):
        return np.full((arr.size,), 1.0 / float(max(arr.size, 1)), dtype=np.float64)
    return sharp / sharp_sum

class HybridQHARuntimeSelector:

    def __init__(
        self,
        *,
        candidate_horizons: Sequence[int],
        exec_mode: str = "expected_round",
        expected_temp: float = 1.0,
        ts_epsilon: float = 0.05,
        ts_seed: int = 0,
        ts_update_mode: str = "kernel_forget",
        ts_kernel_bandwidth: float = 10.0,
        ts_forget_rho: float = 0.99,
        ts_update_eta: float = 1.0,
        prior_gamma: float = 1.0,
    ):
        self.candidate_horizons = tuple(int(v) for v in candidate_horizons)
        self.exec_mode = str(exec_mode).strip().lower()
        self.expected_temp = float(max(expected_temp, 1e-6))
        self.ts_epsilon = float(np.clip(ts_epsilon, 0.0, 1.0))
        self.ts_update_mode = str(ts_update_mode).strip().lower()
        self.ts_kernel_bandwidth = float(max(ts_kernel_bandwidth, 1e-6))
        self.ts_forget_rho = float(np.clip(ts_forget_rho, 1e-6, 1.0))
        self.ts_update_eta = float(max(ts_update_eta, 0.0))
        self.prior_gamma = float(max(prior_gamma, 0.0))
        self._seed = int(ts_seed)
        self._rng = np.random.default_rng(self._seed)
        self._state = {int(s): [1.0, 1.0] for s in self.candidate_horizons}

    def reset(self, seed: int | None = None) -> None:
        for key in self._state:
            self._state[key] = [1.0, 1.0]
        episode_seed = self._seed if seed is None else int(seed)
        self._rng = np.random.default_rng(np.random.SeedSequence([self._seed, episode_seed]))

    def _gaussian_kernel_weights(self, center: float) -> np.ndarray:
        vals = np.asarray(self.candidate_horizons, dtype=np.float64)
        weights = np.exp(-((vals - float(center)) ** 2) / (2.0 * self.ts_kernel_bandwidth * self.ts_kernel_bandwidth))
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-12 or (not np.isfinite(weight_sum)):
            return np.full((vals.size,), 1.0 / float(max(vals.size, 1)), dtype=np.float64)
        return weights / weight_sum

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
                for i, s in enumerate(self.candidate_horizons):
                    prob_i = float(expected_prob[i]) if i < expected_prob.size else 0.0
                    if prob_i <= 0.0:
                        continue
                    qa_i = float(effective_scores[i])
                    a, b = self._state[int(s)]
                    self._state[int(s)] = [a + (prob_i * qa_i), b + (prob_i * (1.0 - qa_i))]
                return float(np.dot(expected_prob, effective_scores))
            qa = float(effective_scores[chosen_idx])
            a, b = self._state[int(k_exec)]
            self._state[int(k_exec)] = [a + qa, b + (1.0 - qa)]
            return qa

        feedback = float(np.dot(expected_prob, effective_scores)) if expected_prob is not None else float(effective_scores[chosen_idx])
        feedback = float(np.clip(feedback, 0.0, 1.0))
        weights = self._gaussian_kernel_weights(float(k_exec))
        for i, s in enumerate(self.candidate_horizons):
            a, b = self._state[int(s)]
            incr = self.ts_update_eta * float(weights[i])
            self._state[int(s)] = [
                self.ts_forget_rho * a + incr * feedback,
                self.ts_forget_rho * b + incr * (1.0 - feedback),
            ]
        return feedback

    def select(self, posterior: np.ndarray, q_mix: np.ndarray, *, max_exec_length: int) -> dict[str, Any]:
        posterior = np.asarray(posterior, dtype=np.float64).reshape(-1)
        q_mix = np.asarray(q_mix, dtype=np.float64).reshape(-1)
        if posterior.shape != q_mix.shape:
            raise ValueError(f"posterior/q_mix shape mismatch: {posterior.shape} vs {q_mix.shape}")
        if posterior.shape[0] != len(self.candidate_horizons):
            raise ValueError(
                f"posterior length must match candidate_horizons. got {posterior.shape[0]} vs {len(self.candidate_horizons)}"
            )

        prior = np.power(np.clip(posterior, 1e-12, 1.0), self.prior_gamma)
        effective = np.clip(q_mix, 0.0, 1.0) * prior
        theta = np.asarray(
            [self._rng.beta(*self._state[int(s)]) for s in self.candidate_horizons],
            dtype=np.float64,
        )
        runtime_scores = theta * effective
        expected_prob = None

        if float(self._rng.random()) < self.ts_epsilon:
            chosen_idx = int(self._rng.integers(0, len(self.candidate_horizons)))
            k_exec = int(self.candidate_horizons[chosen_idx])
            choose_mode = "epsilon_random"
            feedback = self._apply_update(effective, chosen_idx=chosen_idx, k_exec=k_exec)
        elif self.exec_mode == "expected_round":
            expected_prob = _normalize_score_distribution_np(runtime_scores, temp=self.expected_temp)
            expected_horizon = float(np.dot(expected_prob, np.asarray(self.candidate_horizons, dtype=np.float64)))
            k_exec = int(np.clip(int(round(expected_horizon)), 1, int(max_exec_length)))
            choose_mode = "expected_round"
            chosen_idx = -1
            feedback = self._apply_update(effective, chosen_idx=chosen_idx, k_exec=k_exec, expected_prob=expected_prob)
        else:
            chosen_idx = int(np.argmax(runtime_scores))
            k_exec = int(self.candidate_horizons[chosen_idx])
            choose_mode = "thompson"
            feedback = self._apply_update(effective, chosen_idx=chosen_idx, k_exec=k_exec)

        return {
            "exec_length": int(np.clip(k_exec, 1, int(max_exec_length))),
            "choose_mode": choose_mode,
            "runtime_theta": theta.astype(np.float32),
            "runtime_scores": runtime_scores.astype(np.float32),
            "runtime_effective_scores": effective.astype(np.float32),
            "runtime_expected_prob": None if expected_prob is None else expected_prob.astype(np.float32),
            "runtime_feedback": float(feedback),
        }
