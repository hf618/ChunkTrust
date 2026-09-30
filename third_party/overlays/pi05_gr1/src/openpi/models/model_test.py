import os

from flax import nnx
import jax
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils

RUN_HEAVY_MODEL_TESTS = os.getenv("OPENPI_RUN_HEAVY_MODEL_TESTS") == "1"


@pytest.mark.skipif(not RUN_HEAVY_MODEL_TESTS, reason="heavy model init test")
def test_pi0_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi05_qha_config_defaults():
    cfg = pi0_config.Pi0Config(pi05=True, use_qha=True)
    assert cfg.model_type == _model.ModelType.PI05
    assert cfg.discrete_state_input is True
    assert cfg.qha_train_mode == "end_to_end"
    assert cfg.qha_candidate_mode == "sparse"
    assert cfg.qha_infer_mode == "hybrid"
    assert cfg.qha_candidates == "10,20,30,40,50"
    assert cfg.qha_loss_type == "KL"
    assert cfg.qha_history_max_length == 64
    assert cfg.horizon_selector_exec_mode == "expected_round"


def test_pi05_qha_config_normalizes_dense_candidates():
    cfg = pi0_config.Pi0Config(
        pi05=True,
        use_qha=True,
        action_horizon=8,
        max_token_len=16,
        qha_candidate_mode="dense_full",
        qha_candidates="8,4,4,2,10",
        qha_loss_type="kl",
    )
    assert cfg.qha_candidates == "1,2,3,4,5,6,7,8"
    assert cfg.qha_loss_type == "KL"


def test_pi05_qha_config_rejects_inconsistent_values():
    with pytest.raises(ValueError):
        pi0_config.Pi0Config(pi05=True, use_qha=True, qha_train_mode="bad_mode")
    with pytest.raises(ValueError):
        pi0_config.Pi0Config(pi05=True, use_qha=True, qha_candidate_mode="bad_mode")
    with pytest.raises(ValueError):
        pi0_config.Pi0Config(pi05=True, use_qha=True, qha_num_queries=0)
    with pytest.raises(ValueError):
        pi0_config.Pi0Config(pi05=True, use_qha=True, qha_d_model=130, qha_num_heads=8)
    with pytest.raises(ValueError):
        pi0_config.Pi0Config(pi05=True, use_qha=True, qha_history_max_length=0)


@pytest.mark.skipif(not RUN_HEAVY_MODEL_TESTS, reason="heavy model init test")
def test_pi05_qha_model_loss_and_teacher_shapes():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=8,
        action_horizon=8,
        max_token_len=16,
        use_qha=True,
        qha_train_mode="qha_only",
        qha_candidates="2,4,6,8",
        qha_num_queries=4,
        qha_d_model=32,
        qha_num_heads=4,
        qha_num_bridge_layers=1,
        qha_history_max_length=8,
    )
    model = config.create(key)
    obs, act = config.fake_obs(batch_size=2), config.fake_act(batch_size=2)

    loss, metrics = model.compute_loss_and_metrics(key, obs, act, train=False)
    teacher = model.predict_qha_teacher_outputs(key, obs)
    rollout = model.sample_actions_and_qha_outputs(key, obs)

    assert loss.shape == (2, config.action_horizon)
    assert metrics["qha_loss"].shape == (2,)
    assert metrics["total_loss"].shape == (2,)
    assert teacher is not None
    assert teacher["qha_teacher_posterior"].shape == (2, 4)
    assert teacher["qha_teacher_expected_horizon"].shape == (2,)
    assert teacher["qha_teacher_argmax_horizon"].shape == (2,)
    assert rollout["actions"].shape == (2, config.action_horizon, config.action_dim)
    assert rollout["qha_posterior"].shape == (2, 4)
    assert rollout["qha_candidate_horizons"].shape == (4,)


@pytest.mark.skipif(not RUN_HEAVY_MODEL_TESTS, reason="heavy model init test")
def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


@pytest.mark.skipif(not RUN_HEAVY_MODEL_TESTS, reason="heavy model init test")
def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


@pytest.mark.skipif(not RUN_HEAVY_MODEL_TESTS, reason="heavy model init test")
def test_pi0_fast_lora_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


@pytest.mark.manual
@pytest.mark.skipif(os.getenv("OPENPI_RUN_NETWORK_TESTS") != "1", reason="network-dependent restore test")
def test_model_restore():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
