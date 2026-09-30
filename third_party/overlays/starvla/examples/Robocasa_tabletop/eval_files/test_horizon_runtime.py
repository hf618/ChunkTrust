import unittest
from types import SimpleNamespace

import numpy as np
import torch

from examples.Robocasa_tabletop.eval_files.model2robocasa_interface import PolicyWarper
from starVLA.model.framework.QwenGR00T import Qwen_GR00T


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


class _DummyVLM:
    def build_qwenvl_inputs(self, images, instructions):
        return {}

    def __call__(self, **kwargs):
        return SimpleNamespace(hidden_states=[torch.zeros((1, 1, 4), dtype=torch.float32)])


class _DummyActionModel:
    def predict_action(self, last_hidden, state, return_trace=False):
        out = torch.zeros((1, 2, 3), dtype=torch.float32)
        if return_trace:
            return out, {"v": torch.zeros((1, 4, 2, 3), dtype=torch.float32)}
        return out


class _DummyQwenModel:
    _should_enable_trace = staticmethod(Qwen_GR00T._should_enable_trace)

    def __init__(self):
        self.config = SimpleNamespace(
            datasets=SimpleNamespace(
                vla_data=SimpleNamespace(image_size=None),
            )
        )
        self.qwen_vl_interface = _DummyVLM()
        self.action_model = _DummyActionModel()


class HorizonRuntimeTests(unittest.TestCase):
    def test_eval_type_normalization_rejects_stride(self) -> None:
        self.assertEqual(PolicyWarper._normalize_eval_type("default"), "default")
        self.assertEqual(PolicyWarper._normalize_eval_type("HORIZON"), "horizon")
        with self.assertRaises(ValueError):
            PolicyWarper._normalize_eval_type("stride")
        with self.assertRaises(ValueError):
            PolicyWarper._normalize_eval_type("weird")

    def test_qwen_predict_action_enables_trace_for_horizon(self) -> None:
        dummy_model = _DummyQwenModel()
        examples = [{"image": [np.zeros((4, 4, 3), dtype=np.uint8)], "lang": "test"}]

        horizon_out = Qwen_GR00T.predict_action(dummy_model, examples, eval_type="horizon")
        default_out = Qwen_GR00T.predict_action(dummy_model, examples, eval_type="default")

        self.assertIn("denoise_trace", horizon_out)
        self.assertNotIn("denoise_trace", default_out)
        with self.assertRaises(ValueError):
            Qwen_GR00T.predict_action(dummy_model, examples, eval_type="stride")

    def test_horizon_runtime_path_selects_dynamic_exec_horizon(self) -> None:
        policy = PolicyWarper.__new__(PolicyWarper)
        policy.eval_type = "horizon"
        policy.n_action_steps = 16
        policy.horizon_candidates = [4, 8, 12, 16]
        policy.horizon_exec_mode = "expected_round"
        policy.horizon_expected_temp = 1.0
        policy.horizon_intra_alpha = 4.0
        policy.horizon_intra_cut_t = 0.25
        policy.horizon_ts_epsilon = 0.0
        policy.horizon_ts_seed = 0
        policy.horizon_ts_update_mode = "kernel_forget"
        policy.horizon_ts_kernel_bandwidth = 10.0
        policy.horizon_ts_forget_rho = 0.99
        policy.horizon_ts_update_eta = 1.0
        policy.horizon_intra_z_mode = "window_rms"
        policy.horizon_intra_tau_window = 1
        policy.horizon_mix_mode = "intra"
        policy.horizon_inter_use_speed_uniformity = False
        policy.horizon_inter_window = 20
        policy.horizon_inter_weight = 0.5
        policy.horizon_inter_beta = 4.0
        policy.horizon_inter_fallback = "ehh_only"
        policy.last_horizon_info = None
        policy.executed_action_history = []
        policy._horizon_ts_states = {}
        policy._horizon_ts_rngs = {}
        PolicyWarper._reset_run_score_state(policy)

        v_step = _build_synthetic_v_step(seed=0)
        policy.last_denoise_trace = {"v": v_step[None, ...]}
        raw_actions = np.cumsum(np.ones((1, 16, 3), dtype=np.float32), axis=1)

        exec_horizon = PolicyWarper._select_exec_horizons(policy, raw_actions)

        self.assertEqual(exec_horizon.shape, (1,))
        self.assertIsNotNone(policy.last_horizon_info)
        self.assertEqual(len(policy.last_horizon_info), 1)
        self.assertEqual(int(exec_horizon[0]), int(policy.last_horizon_info[0]["chosen_horizon"]))
        self.assertLess(int(exec_horizon[0]), int(policy.n_action_steps))
        self.assertGreater(float(policy.last_horizon_info[0]["expected_horizon"]), 0.0)


if __name__ == "__main__":
    unittest.main()
