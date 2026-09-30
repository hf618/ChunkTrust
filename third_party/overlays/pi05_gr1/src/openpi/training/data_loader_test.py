import dataclasses

import jax
import pytest

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def _make_fake_train_config(*, use_qha: bool = False) -> _config.TrainConfig:
    model = pi0_config.Pi0Config(
        pi05=True,
        action_dim=24,
        action_horizon=8,
        max_token_len=16,
        use_qha=use_qha,
        qha_history_max_length=6 if use_qha else 64,
    )
    return _config.TrainConfig(
        name="unit_test_fake",
        exp_name="unit_test_fake_exp",
        model=model,
        data=_config.FakeDataConfig(),
        batch_size=4,
        num_workers=0,
    )


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_fake_dataset_qha_observation_fields():
    config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=8,
        action_horizon=8,
        max_token_len=16,
        use_qha=True,
        qha_history_max_length=6,
    )
    dataset = _data_loader.FakeDataset(config, 2)
    item = dataset[0]
    assert item["hist_actions"].shape == (6, 8)
    assert item["hist_actions_mask"].shape == (6,)
    assert item["actions"].shape == (8, 8)


def test_preprocessed_cache_signature_changes_with_qha_settings():
    base = dataclasses.replace(
        _make_fake_train_config(use_qha=False),
        exp_name="sig_a",
        data=dataclasses.replace(_make_fake_train_config(use_qha=False).data, repo_id="fake_signature_repo"),
    )
    qha = dataclasses.replace(
        base,
        exp_name="sig_b",
        model=dataclasses.replace(base.model, use_qha=True, qha_history_max_length=12),
    )

    base_info = _data_loader.resolve_preprocessed_cache_info(base, base.data.create(base.assets_dirs, base.model))
    qha_info = _data_loader.resolve_preprocessed_cache_info(qha, qha.data.create(qha.assets_dirs, qha.model))

    assert base_info.signature != qha_info.signature
    assert "fake_signature_repo" in str(qha_info.cache_dir)


def test_missing_preprocessed_cache_error_includes_build_command():
    config = dataclasses.replace(_make_fake_train_config(use_qha=True), exp_name="cache_miss", use_preprocessed_cache=True)

    with pytest.raises(FileNotFoundError, match=r"scripts/build_preprocessed_cache\.py"):
        _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=1)


def test_with_fake_dataset():
    config = _make_fake_train_config(use_qha=False)

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch, _ in batches:
        for x in jax.tree.leaves(batch):
            if hasattr(x, "shape") and len(x.shape) > 0:
                assert x.shape[0] == config.batch_size

    for (_, actions), _ in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


@pytest.mark.skip(reason="requires local real LeRobot dataset availability")
def test_with_real_dataset():
    config = dataclasses.replace(_config.get_config("pi05_base_aloha_robotwin_lora"), batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
