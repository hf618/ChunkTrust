import unittest

import numpy as np

from examples.Robocasa_tabletop.eval_files.horizon_selector import (
    normalize_horizon_exec_mode,
    normalize_horizon_ts_update_mode,
    select_exec_k_horizon,
)


def _build_synthetic_v_step(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tau = 4
    horizon = 16
    action_dim = 3
    t = np.arange(horizon, dtype=np.float32)
    v_step = np.zeros((tau, horizon, action_dim), dtype=np.float32)
    for tau_idx in range(tau):
        for dim_idx in range(action_dim):
            low = np.sin(2.0 * np.pi * (t + tau_idx) / (horizon / (1 + dim_idx)))
            high = 0.6 * np.sin(2.0 * np.pi * (t * (3 + dim_idx) + tau_idx) / horizon)
            trend = (0.03 * tau_idx) * t
            noise = 0.02 * rng.standard_normal(horizon)
            v_step[tau_idx, :, dim_idx] = low + high + trend + noise
    return v_step


class HorizonSelectorTests(unittest.TestCase):
    def test_intra_cut_t_changes_ehh_signal_and_choice(self) -> None:
        v_step = _build_synthetic_v_step(seed=0)
        raw_actions = np.cumsum(np.ones((16, 3), dtype=np.float32), axis=0)

        common_kwargs = dict(
            horizon_candidates="4,8,12,16",
            exec_mode="expected_round",
            expected_temp=1.0,
            intra_alpha=4.0,
            z_mode="window_rms",
            tau_window=1,
            xt_step=raw_actions,
            history_actions=np.zeros((40, 3), dtype=np.float32),
            mix_mode="intra",
            use_speed_uniformity=False,
            ts_epsilon=0.0,
            ts_update_mode="kernel_forget",
        )

        small_choice, small_info = select_exec_k_horizon(
            v_step,
            16,
            intra_cut_t=0.25,
            ts_rng=np.random.default_rng(0),
            ts_state={},
            **common_kwargs,
        )
        large_choice, large_info = select_exec_k_horizon(
            v_step,
            16,
            intra_cut_t=0.8,
            ts_rng=np.random.default_rng(0),
            ts_state={},
            **common_kwargs,
        )

        self.assertNotEqual(small_info["q_intra"], large_info["q_intra"])
        self.assertNotEqual(small_info["ehh_curve_by_horizon"], large_info["ehh_curve_by_horizon"])
        self.assertLess(int(small_choice), int(large_choice))
        self.assertEqual(int(small_choice), int(small_info["chosen_horizon"]))
        self.assertEqual(int(large_choice), int(large_info["chosen_horizon"]))
        self.assertAlmostEqual(float(small_info["horizon_intra_cut_t"]), 0.25)
        self.assertAlmostEqual(float(large_info["horizon_intra_cut_t"]), 0.8)

    def test_invalid_public_modes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_horizon_exec_mode("greedy")
        with self.assertRaises(ValueError):
            normalize_horizon_ts_update_mode("legacy")


if __name__ == "__main__":
    unittest.main()
