from __future__ import annotations
from chunktrust.paths import resolve_legacy_path as _ct_path

import dataclasses
import importlib.util
import json
import os
import pathlib
import sys

import numpy as np

from openpi.policies import policy_config
from openpi.training import checkpoints
from openpi.training import config
from openpi.training import data_loader
import openpi.transforms as transforms


ROOT = pathlib.Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "configs" / "robotwin_clean50_multitask50_manifest.json"


def _load_norm_module():
    path = ROOT / "scripts" / "compute_norm_stats.py"
    spec = importlib.util.spec_from_file_location("_multitask50_compute_norm_stats", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_cache_rematerialization_module():
    path = ROOT / "scripts" / "rematerialize_multitask50_cache_norm.py"
    spec = importlib.util.spec_from_file_location("_multitask50_cache_rematerialization", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    env_names = (
        "JAX_PLATFORMS",
        "CUDA_VISIBLE_DEVICES",
        "HF_HUB_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "LEROBOT_SKIP_TIMESTAMP_CHECK",
    )
    previous_env = {name: os.environ.get(name) for name in env_names}
    try:
        spec.loader.exec_module(module)
    finally:
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return module


def test_formal_config_matches_frozen_manifest_and_recipe() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected_repo_ids = tuple(item["repo_id"] for item in manifest["tasks"])
    assert expected_repo_ids == config.ROBOTWIN_CLEAN50_MULTITASK50_REPO_IDS

    train_config = config.get_config("pi05_base_aloha_robotwin_full_multitask50")
    assert train_config.batch_size == 256
    assert train_config.num_train_steps == 30_000
    assert train_config.seed == 42
    assert train_config.fsdp_devices == 4
    assert train_config.multi_task_batch_mode == "mixed"
    assert train_config.model.pi05
    assert not train_config.model.use_qha
    assert train_config.save_interval == 5_000
    assert train_config.keep_period == 5_000
    assert train_config.num_workers == 32
    assert train_config.dataloader_prefetch_factor == 2
    assert train_config.host_prefetch == 32
    assert train_config.device_prefetch == 8
    assert train_config.lr_schedule.warmup_steps == 2_000
    assert train_config.lr_schedule.peak_lr == 5e-5
    assert train_config.lr_schedule.decay_steps == 30_000
    assert train_config.lr_schedule.decay_lr == 5e-6
    assert not train_config.eager_compile_step_fns


def test_all_subconfigs_bind_same_global_norm_asset() -> None:
    train_config = config.get_config("pi05_base_aloha_robotwin_full_multitask50")
    created = train_config.data.create(train_config.assets_dirs, train_config.model)
    assert isinstance(created, config.MultiDataConfig)
    assert len(created.sub_configs) == 50
    assert {item.asset_id for item in created.sub_configs} == {
        "robotwin_aloha_agilex_clean50_multitask50_global"
    }
    assert {item.lerobot_root for item in created.sub_configs} == {
        _ct_path('CHUNKTRUST_CACHE_ROOT', 'huggingface/lerobot_robotwin_clean50_multitask50_20260728')
    }


def test_policy_selects_task_but_preserves_shared_asset() -> None:
    train_config = config.get_config("pi05_base_aloha_robotwin_full_multitask50")
    created = train_config.data.create(train_config.assets_dirs, train_config.model)
    selected = policy_config._resolve_policy_data_config(created, "place_bread_basket")
    assert selected.repo_id == "place_bread_basket_aloha-agilex_clean_50_repo"
    assert selected.asset_id == "robotwin_aloha_agilex_clean50_multitask50_global"


def test_fast_norm_transform_matches_full_horizon_delta_actions() -> None:
    module = _load_norm_module()
    state = np.arange(28, dtype=np.float32).reshape(2, 14)
    actions = state + 3
    action_sequences = module._build_action_sequences(actions, action_horizon=3)
    transformed_state, transformed_actions = module._apply_aloha_numeric_transforms(
        state,
        action_sequences,
        use_delta_joint_actions=True,
        adapt_to_pi=False,
    )
    mask = np.asarray(module.transforms.make_bool_mask(6, -1, 6, -1), dtype=np.bool_)
    assert transformed_state.shape == (2, 14)
    assert transformed_actions.shape == (2, 3, 14)
    np.testing.assert_array_equal(transformed_state, state)
    np.testing.assert_array_equal(
        transformed_actions[1][:, mask],
        np.full((3, 12), 3, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        transformed_actions[..., ~mask],
        action_sequences[..., ~mask],
    )


def test_quantile_cache_rematerialization_matches_direct_target_norm() -> None:
    module = _load_cache_rematerialization_module()
    raw = np.asarray(
        [[-2.0, 0.25, 4.0], [0.0, 1.5, 8.0]],
        dtype=np.float64,
    )
    old_q01 = np.asarray([-1.0, 0.0, 2.0])
    old_q99 = np.asarray([1.0, 2.0, 10.0])
    new_q01 = np.asarray([-3.0, -1.0, 0.0])
    new_q99 = np.asarray([3.0, 4.0, 12.0])
    old_normalized = (raw - old_q01) / (old_q99 - old_q01 + 1e-6) * 2.0 - 1.0
    expected = (raw - new_q01) / (new_q99 - new_q01 + 1e-6) * 2.0 - 1.0

    actual = module.remap_quantile_normalized(
        old_normalized,
        old_q01,
        old_q99,
        new_q01,
        new_q99,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_batch_multitask_metrics_counts_every_task() -> None:
    metrics = data_loader._batch_multitask_metrics(
        np.asarray([0, 2, 2, 49], dtype=np.int32),
        num_tasks=50,
    )

    assert metrics["batch_task_unique_count"] == 3
    assert metrics["batch_task_count_00"] == 1
    assert metrics["batch_task_count_01"] == 0
    assert metrics["batch_task_count_02"] == 2
    assert metrics["batch_task_count_49"] == 1


def test_batched_cache_resize_is_exactly_equal_to_individual_resize() -> None:
    rng = np.random.default_rng(42)
    items = [
        {
            "image": {
                "base_0_rgb": rng.integers(0, 256, size=(3, 480, 640), dtype=np.uint8),
                "left_wrist_0_rgb": rng.integers(0, 256, size=(3, 480, 640), dtype=np.uint8),
            },
            "value": np.asarray(index),
        }
        for index in range(3)
    ]
    pipeline = (transforms.ResizeImages(224, 224),)
    expected = [
        transforms.compose(pipeline)(
            {
                "image": {key: value.copy() for key, value in item["image"].items()},
                "value": item["value"],
            }
        )
        for item in items
    ]
    actual = data_loader._transform_preprocessed_cache_chunk(
        items,
        pipeline,
        resize_batch_size=8,
    )

    for expected_item, actual_item in zip(expected, actual, strict=True):
        assert expected_item["value"] == actual_item["value"]
        for key in expected_item["image"]:
            np.testing.assert_array_equal(expected_item["image"][key], actual_item["image"][key])


@dataclasses.dataclass
class _CheckpointConfig:
    name: str
    checkpoint_metadata_files: tuple[str, ...]


def test_checkpoint_snapshot_copies_immutable_metadata(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "manifest.json"
    source.write_text('{"tasks": 50}\n', encoding="utf-8")
    assets = tmp_path / "assets"
    assets.mkdir()
    train_config = _CheckpointConfig(name="formal", checkpoint_metadata_files=(str(source),))

    checkpoints.save_train_config_snapshot(assets, train_config)
    checkpoints.save_checkpoint_metadata_files(assets, train_config)

    assert (assets / checkpoints.TRAIN_CONFIG_SNAPSHOT_NAME).is_file()
    assert (assets / "_audit" / "manifest.json").read_bytes() == source.read_bytes()
    hash_line = (assets / "_audit" / "metadata.sha256").read_text(encoding="utf-8")
    assert hash_line.endswith("  manifest.json\n")
