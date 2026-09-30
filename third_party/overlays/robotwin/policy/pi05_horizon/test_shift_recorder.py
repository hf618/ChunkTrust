from pathlib import Path
import types

import jax
import numpy as np

from denoise_logger import DenoiseLogger
from pi_model import PI0
from openpi.policies.policy import Policy


def _qha_outputs(action_offset: float = 0.0) -> dict[str, np.ndarray]:
    actions = np.arange(5 * 16, dtype=np.float32).reshape(5, 16) + action_offset
    return {
        "actions": actions,
        "qha_candidate_horizons": np.arange(1, 6, dtype=np.int32),
        "qha_logits": np.linspace(-1.0, 1.0, 5, dtype=np.float32),
        "qha_posterior": np.full((5,), 0.2, dtype=np.float32),
        "qha_expected_horizon": np.asarray(3.0, dtype=np.float32),
        "qha_argmax_horizon": np.asarray(3, dtype=np.int32),
        "qha_selector_q_intra": np.linspace(0.1, 0.5, 5, dtype=np.float32),
        "qha_selector_p_inter": np.linspace(0.5, 0.9, 5, dtype=np.float32),
        "qha_selector_q_mix": np.linspace(0.2, 0.6, 5, dtype=np.float32),
        "qha_selector_candidate_horizons_sparse_runtime": np.asarray([1, 3, 5], dtype=np.int32),
        "qha_selector_q_mix_sparse_runtime": np.asarray([0.2, 0.4, 0.6], dtype=np.float32),
        "qha_context_tokens": np.arange(12, dtype=np.float32).reshape(3, 4),
        "qha_context_mask": np.asarray([True, True, False]),
        "qha_action_latents": np.arange(20, dtype=np.float32).reshape(5, 4),
    }


class _PolicyStub:
    def __init__(self, outputs: dict[str, np.ndarray]):
        self.outputs = outputs

    def infer(self, observation):
        del observation
        return {key: np.array(value, copy=True) for key, value in self.outputs.items()}

    def infer_with_forked_rng(self, observation, *, fold_in: int):
        del observation, fold_in
        out = {key: np.array(value, copy=True) for key, value in self.outputs.items()}
        out["actions"] = out["actions"] + 1000.0
        return out


def _stub_pi0(tmp_path: Path, *, mode: str, return_features: bool) -> tuple[PI0, Path]:
    model = PI0.__new__(PI0)
    model.observation_window = {"state": np.arange(14, dtype=np.float32)}
    model.policy = _PolicyStub(_qha_outputs())
    model.pi0_step = 5
    model.use_qha = True
    model.shift_trace_mode = mode
    model.return_qha_features = return_features
    model.shift_same_observation_max_replans = 0
    model.shift_v3_anchor_grid = [1, 2, 3, 4, 5]
    model._shift_replan_count = 0
    model._episode_replan_count = 0
    model._counterfactual = None
    model._v3_pending_shadow = None
    model._policy_infer_seconds = 0.0
    model._horizon_metric_compute_seconds = 0.0
    model._horizon_selection_seconds = 0.0
    model._denoise_overhead_seconds = 0.0
    model.last_exec_k = None
    model._select_exec_length = lambda outputs, *, max_exec_length: (
        2,
        {
            "qha_exec_mode": "expected_round",
            "qha_runtime_theta": np.linspace(0.1, 0.5, 5, dtype=np.float32),
            "qha_runtime_scores": np.linspace(0.2, 0.6, 5, dtype=np.float32),
            "qha_runtime_effective_scores": np.linspace(0.3, 0.7, 5, dtype=np.float32),
            "qha_runtime_expected_prob": np.full((5,), 0.2, dtype=np.float32),
            "qha_runtime_feedback": np.asarray(0.4, dtype=np.float32),
            "qha_runtime_candidate_horizons": np.arange(1, 6, dtype=np.int32),
        },
    )
    path = tmp_path / f"{mode}.npz"
    model.denoise_logger = DenoiseLogger(enabled=True, compress=False, include_arrays=False)
    model.denoise_logger.start_episode(path)
    return model, path


def test_qha_boundary_trace_stacks_and_preserves_unexecuted_tail(tmp_path: Path):
    model, path = _stub_pi0(tmp_path, mode="boundary", return_features=False)

    first = model.get_action_plan()
    second = model.get_action_plan()
    model.denoise_logger.flush_episode()

    assert np.array_equal(first["actions"], second["actions"])
    with np.load(path, allow_pickle=False) as data:
        assert data["action_plan"].shape == (2, 5, 14)
        assert data["action_plan_valid_mask"].shape == (2, 5)
        assert data["replan_state"].shape == (2, 14)
        assert data["K_exec"].tolist() == [2, 2]
        assert data["qha_logits"].shape == (2, 5)
        assert data["qha_posterior"].shape == (2, 5)
        assert data["qha_runtime_scores"].shape == (2, 5)
        assert np.any(data["action_plan"][:, 2:, :] != 0.0)
        assert "qha_context_tokens" not in data.files
        assert "qha_action_latents" not in data.files


def test_qha_execution_actions_match_lossless_replay_trace(tmp_path: Path):
    model, path = _stub_pi0(tmp_path, mode="boundary", return_features=False)
    outputs = _qha_outputs()
    outputs["actions"] = outputs["actions"].astype(np.float64)
    outputs["actions"][:, :14] += np.float64(3.141592653589793e-8)
    model.policy = _PolicyStub(outputs)

    plan = model.get_action_plan()
    model.denoise_logger.flush_episode()

    with np.load(path, allow_pickle=False) as data:
        assert plan["actions"].dtype == np.float32
        np.testing.assert_array_equal(plan["actions"][:, :14], data["action_plan"][0])


def test_runtime_expected_prob_keeps_schema_during_epsilon_random_step(tmp_path: Path):
    model, path = _stub_pi0(tmp_path, mode="boundary", return_features=False)
    calls = 0

    def select(outputs, *, max_exec_length):
        nonlocal calls
        del outputs, max_exec_length
        calls += 1
        return 2, {
            "qha_exec_mode": "epsilon_random" if calls == 2 else "expected_round",
            "qha_runtime_theta": np.linspace(0.1, 0.5, 5, dtype=np.float32),
            "qha_runtime_scores": np.linspace(0.2, 0.6, 5, dtype=np.float32),
            "qha_runtime_effective_scores": np.linspace(0.3, 0.7, 5, dtype=np.float32),
            "qha_runtime_expected_prob": None if calls == 2 else np.full((5,), 0.2, dtype=np.float32),
            "qha_runtime_feedback": np.asarray(0.4, dtype=np.float32),
            "qha_runtime_candidate_horizons": np.arange(1, 6, dtype=np.int32),
        }

    model._select_exec_length = select
    model.get_action_plan()
    model.get_action_plan()
    model.denoise_logger.flush_episode()

    with np.load(path, allow_pickle=False) as data:
        assert data["qha_runtime_expected_prob"].shape == (2, 5)
        assert np.allclose(data["qha_runtime_expected_prob"][0], 0.2)
        assert np.isnan(data["qha_runtime_expected_prob"][1]).all()


def test_trainable_trace_includes_features_and_off_records_nothing(tmp_path: Path):
    trainable, trainable_path = _stub_pi0(tmp_path, mode="trainable", return_features=True)
    trainable.get_action_plan()
    trainable.denoise_logger.flush_episode()
    with np.load(trainable_path, allow_pickle=False) as data:
        assert data["qha_context_tokens"].shape == (1, 3, 4)
        assert data["qha_context_mask"].shape == (1, 3)
        assert data["qha_context_mask"].dtype == np.bool_
        assert data["qha_action_latents"].shape == (1, 5, 4)

    off, _ = _stub_pi0(tmp_path, mode="off", return_features=False)
    expected = _qha_outputs()["actions"]
    actual = off.get_action_plan()["actions"]
    assert np.array_equal(actual, expected)
    assert off.denoise_logger._step_traces == []


def test_episode_trace_records_exact_per_environment_step_progress(tmp_path: Path):
    logger = DenoiseLogger(enabled=True, compress=False, include_arrays=False)
    path = tmp_path / "progress_steps.npz"
    logger.start_episode(path)
    logger.log_rsi_env_progress(11, {"progress": 0.0, "placed_count": 0, "bread_count": 2})
    logger.log_rsi_env_progress(12, {"progress": 0.5, "placed_count": 1, "bread_count": 2})
    logger.flush_episode()

    with np.load(path, allow_pickle=False) as data:
        assert data["rsi_progress_env_step_idx"].tolist() == [11, 12]
        assert data["rsi_progress_per_env_step"].tolist() == [0.0, 0.5]
        assert data["rsi_progress_placed_count_per_env_step"].tolist() == [0, 1]
        assert data["rsi_progress_total_count_per_env_step"].tolist() == [2, 2]


def test_v3_trainable_trace_records_lower_shadow_without_changing_action(tmp_path: Path):
    model, path = _stub_pi0(tmp_path, mode="v3_trainable", return_features=True)

    plan = model.get_action_plan()
    recorded = model.maybe_record_v3_lower_shadow(1)
    model.denoise_logger.flush_episode()

    assert recorded is True
    assert np.array_equal(plan["actions"], _qha_outputs()["actions"])
    with np.load(path, allow_pickle=False) as data:
        assert data["v3_k_minus"].tolist() == [1]
        assert data["v3_k"].tolist() == [2]
        assert data["v3_k_plus"].tolist() == [3]
        assert data["v3_lower_shadow_valid"].tolist() == [True]
        assert data["v3_lower_action_plan"].shape == (1, 5, 14)
        assert data["v3_lower_action_plan_valid_mask"].shape == (1, 5)
        assert np.allclose(data["v3_lower_action_plan"][0, 0, :14], _qha_outputs()["actions"][0, :14] + 1000.0)
        assert data["qha_context_tokens"].shape == (1, 3, 4)


def test_infer_with_forked_rng_restores_deployment_rng():
    policy = Policy.__new__(Policy)
    policy._is_pytorch_model = False
    policy._rng = jax.random.key(17)

    def fake_infer(self, obs, *, noise=None):
        del obs, noise
        self._rng, sample_rng = jax.random.split(self._rng)
        return {"sample_rng": sample_rng}

    policy.infer = types.MethodType(fake_infer, policy)
    saved_rng = np.asarray(jax.random.key_data(policy._rng))

    forked = policy.infer_with_forked_rng({}, fold_in=9)

    assert not np.array_equal(np.asarray(jax.random.key_data(forked["sample_rng"])), saved_rng)
    assert np.array_equal(np.asarray(jax.random.key_data(policy._rng)), saved_rng)

    control_next_rng, control_sample_rng = jax.random.split(jax.random.wrap_key_data(saved_rng))
    deployed = policy.infer({})
    assert np.array_equal(
        np.asarray(jax.random.key_data(deployed["sample_rng"])),
        np.asarray(jax.random.key_data(control_sample_rng)),
    )
    assert np.array_equal(
        np.asarray(jax.random.key_data(policy._rng)),
        np.asarray(jax.random.key_data(control_next_rng)),
    )
