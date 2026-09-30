import importlib.util
import json
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openpi import transforms
from openpi.policies import gr1_tabletop_policy
from openpi.training import config
from openpi.training import data_loader


ROOT = pathlib.Path(__file__).parents[3]
MANIFEST = ROOT / "configs" / "robocasa_gr1_tabletop24_manifest.json"
CONFIG_NAME = "pi05_base_gr1_tabletop24_full"


def load_compute_norm_stats():
    path = ROOT / "scripts" / "compute_norm_stats.py"
    spec = importlib.util.spec_from_file_location("_gr1_compute_norm_stats", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_and_registered_membership_are_identical() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = [item["repo_id"] for item in manifest["tasks"]]
    assert expected == list(config.ROBOCASA_GR1_TABLETOP24_REPO_IDS)
    assert len(expected) == len(set(expected)) == 24
    assert sum(item["episodes"] for item in manifest["tasks"]) == 24_000
    assert sum(item["frames"] for item in manifest["tasks"]) == 6_020_058


def test_formal_recipe_is_frozen() -> None:
    train_config = config.get_config(CONFIG_NAME)
    assert train_config.model.pi05 is True
    assert train_config.model.use_qha is False
    assert train_config.model.action_dim == 32
    assert train_config.model.action_horizon == 16
    assert train_config.batch_size == 256
    assert train_config.seed == 42
    assert train_config.num_train_steps == 30_000
    assert train_config.lr_schedule.warmup_steps == 2_000
    assert train_config.lr_schedule.peak_lr == 5e-5
    assert train_config.lr_schedule.decay_steps == 30_000
    assert train_config.lr_schedule.decay_lr == 5e-6
    assert train_config.optimizer.b1 == 0.9
    assert train_config.optimizer.b2 == 0.95
    assert train_config.optimizer.eps == 1e-8
    assert train_config.optimizer.weight_decay == 1e-10
    assert train_config.optimizer.clip_gradient_norm == 1.0
    assert train_config.ema_decay == 0.99
    assert train_config.save_interval == 10_000
    assert train_config.keep_period == 10_000
    assert train_config.multi_task_batch_mode == "mixed"
    assert train_config.num_workers == 32
    assert train_config.dataloader_multiprocessing_context == "forkserver"
    assert train_config.use_preprocessed_cache is False


def test_every_task_binds_one_global_norm_and_no_delta_action_transform() -> None:
    train_config = config.get_config(CONFIG_NAME)
    created = train_config.data.create(train_config.assets_dirs, train_config.model)
    assert isinstance(created, config.MultiDataConfig)
    assert len(created.sub_configs) == 24
    assert {item.asset_id for item in created.sub_configs} == {
        "robocasa_gr1_tabletop24_global"
    }
    for item in created.sub_configs:
        assert item.prompt_from_task is True
        assert item.prompt_index_key == "annotation.human.coarse_action"
        assert item.local_files_only is True
        assert item.action_sequence_keys == ("action",)
        assert not any(
            isinstance(transform, transforms.ResizeImages)
            for transform in item.model_transforms.inputs
        )
        assert not any(
            isinstance(transform, transforms.DeltaActions)
            for transform in item.data_transforms.inputs
        )


def test_cache_signature_binds_coarse_instruction_source() -> None:
    train_config = config.get_config(CONFIG_NAME)
    created = train_config.data.create(train_config.assets_dirs, train_config.model)
    data_config = created.sub_configs[0]

    payload = data_loader._preprocessed_cache_signature_payload(train_config, data_config)

    assert payload["prompt_from_task"] is True
    assert payload["prompt_index_key"] == "annotation.human.coarse_action"


def test_padded_model_actions_unnormalize_before_gr1_slice() -> None:
    stats = transforms.NormStats(
        mean=np.zeros(29),
        std=np.ones(29),
        q01=np.full(29, -2.0),
        q99=np.full(29, 2.0),
    )
    padded = np.zeros((16, 32), dtype=np.float32)
    padded[..., :29] = 0.5

    unnormalized = transforms.Unnormalize(
        {"actions": stats},
        use_quantiles=True,
    )({"actions": padded})
    active = gr1_tabletop_policy.Gr1TabletopOutputs()(unnormalized)["actions"]

    assert active.shape == (16, 29)
    np.testing.assert_allclose(active, 1.0, atol=1e-6)


def test_masked_camera_cache_fields_are_verified_and_synthetic(tmp_path: pathlib.Path) -> None:
    writers: dict[str, np.memmap] = {}
    specs: dict[str, dict] = {}
    item = {
        "image": {
            "base_0_rgb": np.ones((4, 4, 3), dtype=np.uint8),
            "left_wrist_0_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
            "right_wrist_0_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
        }
    }
    data_loader._write_preprocessed_cache_item(
        item=item,
        row_index=0,
        num_samples=2,
        tmp_dir=tmp_path,
        writers=writers,
        field_specs=specs,
        reference_keys=None,
        synthetic_zero_fields=frozenset(
            {"image/left_wrist_0_rgb", "image/right_wrist_0_rgb"}
        ),
    )
    assert specs["image/left_wrist_0_rgb"]["synthetic"] == "zeros"
    assert specs["image/right_wrist_0_rgb"]["synthetic"] == "zeros"
    assert "filename" not in specs["image/left_wrist_0_rgb"]
    assert "filename" not in specs["image/right_wrist_0_rgb"]
    assert "image/base_0_rgb" in writers


def test_global_norm_fast_path_counts_each_gr1_action_frame_once(
    tmp_path: pathlib.Path,
) -> None:
    repo_id = "one_gr1_task"
    repo_root = tmp_path / repo_id
    parquet_dir = repo_root / "data" / "chunk-000"
    parquet_dir.mkdir(parents=True)
    (repo_root / "meta").mkdir()
    (repo_root / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 1}),
        encoding="utf-8",
    )
    raw_state = np.arange(3 * 44, dtype=np.float64).reshape(3, 44)
    raw_actions = raw_state + 1000
    pq.write_table(
        pa.table(
            {
                "observation.state": raw_state.tolist(),
                "action": raw_actions.tolist(),
            }
        ),
        parquet_dir / "episode_000000.parquet",
    )
    data_config = config.DataConfig(
        repo_id=repo_id,
        lerobot_root=str(tmp_path),
    )
    result = load_compute_norm_stats()._fast_gr1_parquet_batches(
        data_config,
        max_frames=None,
    )
    assert result is not None
    batches, num_batches, num_frames = result
    batch = next(iter(batches))
    assert num_batches == 1
    assert num_frames == 3
    assert batch["state"].shape == (3, 29)
    assert batch["actions"].shape == (3, 29)
    np.testing.assert_array_equal(
        batch["state"],
        gr1_tabletop_policy.extract_active(raw_state).astype(np.float32),
    )
    np.testing.assert_array_equal(
        batch["actions"],
        gr1_tabletop_policy.extract_active(raw_actions).astype(np.float32),
    )
