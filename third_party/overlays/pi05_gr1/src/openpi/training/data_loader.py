import bisect
import collections
from collections.abc import Iterator, Sequence
import dataclasses
import hashlib
import io
import json
import logging
import multiprocessing
import os
import pathlib
import shutil
import typing
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
from lerobot.common.datasets.utils import hf_transform_to_torch
import numpy as np
import PIL.Image
import pyarrow.parquet as pq
import torch
import tqdm_loggable.auto as tqdm

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.image_tools as _image_tools
import openpi.training.config as _config
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)
_SAMPLE_WEIGHT_KEY = "__sample_weight__"
_PREPROCESSED_CACHE_VERSION = "openpi_preprocessed_v2_image_layout"
_PREPROCESSED_CACHE_MANIFEST = "manifest.json"
_PREPROCESSED_CACHE_LOG_INTERVAL = 1000
_CACHE_RESIZE_ACCELERATOR_LOGGED = False


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def data_configs(self) -> tuple[_config.DataConfig, ...]:
        """Get every data config represented by this loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_configs().")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")

    def sampling_metrics(self) -> dict[str, float]:
        """Optional static sampler diagnostics."""
        raise NotImplementedError("Subclasses of DataLoader should implement sampling_metrics().")


class TransformedDataset(Dataset[T_co]):

    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class InstrumentedDataset(Dataset[T_co]):

    def __init__(self, dataset: Dataset, sample_weights: np.ndarray):
        self._dataset = dataset
        self._sample_weights = np.asarray(sample_weights, dtype=np.float32)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        idx = index.__index__()
        item = self._dataset[idx]
        if not isinstance(item, dict):
            raise TypeError(f"InstrumentedDataset expects dict items, got {type(item)}")
        out = dict(item)
        out[_SAMPLE_WEIGHT_KEY] = np.asarray(self._sample_weights[idx], dtype=np.float32)
        return typing.cast(T_co, out)

    def __len__(self) -> int:
        return len(self._dataset)


class TaskHomogeneousBatchSampler:
    """Yield batches containing samples from a single task."""

    def __init__(
        self,
        *,
        task_lengths: Sequence[int],
        batch_size: int,
        seed: int,
        num_samples: int = 0,
        batches_per_task_block: int = 1,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if not task_lengths:
            raise ValueError("task_lengths must be non-empty")
        if any(int(length) <= 0 for length in task_lengths):
            raise ValueError(f"All task lengths must be positive, got {task_lengths}")

        self._task_lengths = [int(length) for length in task_lengths]
        self._batch_size = int(batch_size)
        self._seed = int(seed)
        self._batches_per_task_block = max(1, int(batches_per_task_block))
        total_len = sum(self._task_lengths)
        samples_per_epoch = int(num_samples) if int(num_samples) > 0 else total_len
        self._num_batches = max(1, samples_per_epoch // self._batch_size)
        self._offsets = np.cumsum([0, *self._task_lengths[:-1]], dtype=np.int64).tolist()

    def __iter__(self):
        rng = np.random.default_rng(self._seed)
        task_order = np.arange(len(self._task_lengths), dtype=np.int64)
        cursors = [0 for _ in self._task_lengths]
        permutations = [rng.permutation(length) for length in self._task_lengths]

        for batch_id in range(self._num_batches):
            block_id = batch_id // self._batches_per_task_block
            if block_id % len(task_order) == 0 and batch_id % self._batches_per_task_block == 0:
                rng.shuffle(task_order)
            task_index = int(task_order[block_id % len(task_order)])
            length = self._task_lengths[task_index]
            cursor = cursors[task_index]
            if cursor + self._batch_size > length:
                permutations[task_index] = rng.permutation(length)
                cursor = 0
            local_indices = permutations[task_index][cursor:cursor + self._batch_size]
            if local_indices.shape[0] < self._batch_size:
                extra = rng.choice(length, size=self._batch_size - local_indices.shape[0], replace=True)
                local_indices = np.concatenate([local_indices, extra])
            cursors[task_index] = cursor + self._batch_size
            offset = self._offsets[task_index]
            yield [int(offset + idx) for idx in local_indices]

    def __len__(self) -> int:
        return self._num_batches


class TaskBalancedMixedBatchSampler:
    """Yield mixed-task batches with an approximately equal number of samples per task."""

    def __init__(
        self,
        *,
        task_lengths: Sequence[int],
        batch_size: int,
        seed: int,
        num_samples: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if not task_lengths:
            raise ValueError("task_lengths must be non-empty")
        if any(int(length) <= 0 for length in task_lengths):
            raise ValueError(f"All task lengths must be positive, got {task_lengths}")

        self._task_lengths = [int(length) for length in task_lengths]
        self._batch_size = int(batch_size)
        self._seed = int(seed)
        total_len = sum(self._task_lengths)
        samples_per_epoch = int(num_samples) if int(num_samples) > 0 else total_len
        self._num_batches = max(1, samples_per_epoch // self._batch_size)
        self._offsets = np.cumsum([0, *self._task_lengths[:-1]], dtype=np.int64).tolist()
        self._epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self._seed + self._epoch)
        self._epoch += 1
        num_tasks = len(self._task_lengths)
        base_count = self._batch_size // num_tasks
        remainder = self._batch_size % num_tasks
        remainder_order = rng.permutation(num_tasks)
        cursors = [0 for _ in self._task_lengths]
        permutations = [rng.permutation(length) for length in self._task_lengths]

        for batch_id in range(self._num_batches):
            counts = np.full(num_tasks, base_count, dtype=np.int64)
            if remainder > 0:
                if batch_id % num_tasks == 0:
                    remainder_order = rng.permutation(num_tasks)
                rolled = np.roll(remainder_order, -(batch_id % num_tasks))
                counts[rolled[:remainder]] += 1

            batch_indices: list[int] = []
            for task_index, count in enumerate(counts.tolist()):
                if count <= 0:
                    continue
                length = self._task_lengths[task_index]
                cursor = cursors[task_index]
                if cursor + count > length:
                    permutations[task_index] = rng.permutation(length)
                    cursor = 0
                local_indices = permutations[task_index][cursor:cursor + count]
                if local_indices.shape[0] < count:
                    extra = rng.choice(length, size=count - local_indices.shape[0], replace=True)
                    local_indices = np.concatenate([local_indices, extra])
                cursors[task_index] = cursor + count
                offset = self._offsets[task_index]
                batch_indices.extend(int(offset + idx) for idx in local_indices)

            rng.shuffle(batch_indices)
            yield batch_indices

    def __len__(self) -> int:
        return self._num_batches


class TaskBalancedCoverageBatchSampler:
    """Balanced batches which cover every task's dataset at least once per epoch.

    An ordinary balanced sampler allocates one-sixth of a fixed global sample
    budget to every task. That can silently omit examples from the longer
    RoboTwin tasks. A coverage epoch instead has enough batches for the
    longest task at the minimum per-task quota. Shorter tasks wrap only after
    all of their examples have appeared once.
    """

    def __init__(self, *, task_lengths: Sequence[int], batch_size: int, seed: int, num_samples: int = 0):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if not task_lengths or any(int(length) <= 0 for length in task_lengths):
            raise ValueError(f"All task lengths must be positive, got {task_lengths}")
        if int(num_samples) > 0:
            raise ValueError(
                "task_balanced_coverage does not accept horizon_resample_num_samples; "
                "a partial sample budget would invalidate its full-coverage guarantee."
            )
        self._task_lengths = [int(length) for length in task_lengths]
        self._batch_size = int(batch_size)
        self._seed = int(seed)
        self._epoch = 0
        self._offsets = np.cumsum([0, *self._task_lengths[:-1]], dtype=np.int64).tolist()
        min_per_task = self._batch_size // len(self._task_lengths)
        if min_per_task <= 0:
            raise ValueError("batch_size must be at least the number of tasks for balanced coverage.")
        self._num_batches = max(
            1,
            max(int(np.ceil(length / float(min_per_task))) for length in self._task_lengths),
        )

    def __iter__(self):
        rng = np.random.default_rng(self._seed + self._epoch)
        self._epoch += 1
        num_tasks = len(self._task_lengths)
        base_count = self._batch_size // num_tasks
        remainder = self._batch_size % num_tasks
        remainder_order = rng.permutation(num_tasks)
        cursors = [0 for _ in self._task_lengths]
        permutations = [rng.permutation(length) for length in self._task_lengths]

        for batch_id in range(self._num_batches):
            counts = np.full(num_tasks, base_count, dtype=np.int64)
            if remainder:
                if batch_id % num_tasks == 0:
                    remainder_order = rng.permutation(num_tasks)
                counts[np.roll(remainder_order, -(batch_id % num_tasks))[:remainder]] += 1

            batch_indices: list[int] = []
            for task_index, count in enumerate(counts.tolist()):
                length = self._task_lengths[task_index]
                remaining = int(count)
                selected: list[np.ndarray] = []
                while remaining > 0:
                    cursor = cursors[task_index]
                    take = min(remaining, length - cursor)
                    selected.append(permutations[task_index][cursor:cursor + take])
                    cursor += take
                    remaining -= take
                    if cursor == length:
                        permutations[task_index] = rng.permutation(length)
                        cursor = 0
                    cursors[task_index] = cursor
                local_indices = np.concatenate(selected)
                offset = self._offsets[task_index]
                batch_indices.extend(int(offset + idx) for idx in local_indices)
            rng.shuffle(batch_indices)
            yield batch_indices

    def __len__(self) -> int:
        return self._num_batches


class ConcatDataset(Dataset[T_co]):
    """Concatenate multiple datasets into one indexable dataset."""

    def __init__(self, datasets: list[Dataset]):
        if not datasets:
            raise ValueError("ConcatDataset requires at least one dataset.")
        self._datasets = datasets
        self._cumulative_sizes: list[int] = []
        total = 0
        for dataset in datasets:
            total += len(dataset)
            self._cumulative_sizes.append(total)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        idx = index.__index__()
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"index {idx} out of range for dataset of size {len(self)}")
        dataset_idx = bisect.bisect_right(self._cumulative_sizes, idx)
        local_idx = idx if dataset_idx == 0 else idx - self._cumulative_sizes[dataset_idx - 1]
        return self._datasets[dataset_idx][local_idx]

    def __len__(self) -> int:
        return self._cumulative_sizes[-1] if self._cumulative_sizes else 0


class TaskIndexedDataset(Dataset):
    """Attach a stable numeric task index to each sample."""

    def __init__(self, dataset: Dataset, task_index: int):
        self._dataset = dataset
        self._task_index = int(task_index)

    def __getitem__(self, index: SupportsIndex):
        item = dict(self._dataset[index])
        item["task_index"] = np.asarray(self._task_index, dtype=np.int32)
        return item

    def __len__(self) -> int:
        return len(self._dataset)


def _decode_parquet_image(value: dict[str, typing.Any] | PIL.Image.Image) -> torch.Tensor:
    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        if image_bytes is None:
            raise ValueError("Expected image-backed parquet value to contain raw bytes.")
        with PIL.Image.open(io.BytesIO(image_bytes)) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32)
    elif isinstance(value, PIL.Image.Image):
        array = np.asarray(value.convert("RGB"), dtype=np.float32)
    else:
        raise TypeError(f"Unsupported image-backed parquet value: {type(value)}")
    return torch.from_numpy(np.transpose(array, (2, 0, 1))).div_(255.0)


class FakeDataset(Dataset):

    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = np.random.default_rng(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
            if spec.dtype == jnp.int32:
                return rng.integers(0, 2048, size=shape, dtype=np.int32)
            return np.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


class _FastInitLeRobotDataset(lerobot_dataset.LeRobotDataset):
    """Delay image parquet decoding until individual items are fetched."""

    def load_hf_dataset(self):
        if self.episodes is None:
            path = str(self.root / "data")
            return lerobot_dataset.load_dataset("parquet", data_dir=path, split="train")
        files = [str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes]
        return lerobot_dataset.load_dataset("parquet", data_files=files, split="train")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fast_image_backed = (
            len(self.meta.image_keys) > 0 and len(self.meta.video_keys) == 0 and self.image_transforms is None
        )
        self._selected_episodes = self.episodes if self.episodes is not None else list(range(self.meta.total_episodes))
        episode_from = self.episode_data_index["from"].numpy()
        episode_to = self.episode_data_index["to"].numpy()
        self._episode_lengths = {
            int(ep_idx): int(episode_to[ep_idx] - episode_from[ep_idx])
            for ep_idx in self._selected_episodes
        }
        local_starts = []
        local_ends = []
        cursor = 0
        for ep_idx in self._selected_episodes:
            local_starts.append(cursor)
            cursor += self._episode_lengths[int(ep_idx)]
            local_ends.append(cursor)
        self._local_episode_from = np.asarray(local_starts, dtype=np.int64)
        self._local_episode_to = np.asarray(local_ends, dtype=np.int64)
        cache_size = int(os.environ.get("OPENPI_EPISODE_CACHE_SIZE", str(max(1, min(len(self._selected_episodes), 64)))))
        self._episode_cache_size = cache_size
        self._episode_cache: collections.OrderedDict[int, dict[str, typing.Any]] = collections.OrderedDict()
        if not self._fast_image_backed:
            self.hf_dataset.set_transform(hf_transform_to_torch)

    def _get_episode_data(self, ep_idx: int) -> dict[str, typing.Any]:
        if ep_idx in self._episode_cache:
            self._episode_cache.move_to_end(ep_idx)
            return self._episode_cache[ep_idx]

        table = pq.read_table(self.root / self.meta.get_data_file_path(ep_idx))
        episode_data = {}
        for key in table.column_names:
            column = table.column(key)
            if key in self.meta.image_keys:
                episode_data[key] = column.to_pylist()
            else:
                episode_data[key] = np.asarray(column.to_pylist())

        self._episode_cache[ep_idx] = episode_data
        self._episode_cache.move_to_end(ep_idx)
        while len(self._episode_cache) > self._episode_cache_size:
            self._episode_cache.popitem(last=False)
        return episode_data

    def _decode_image(self, value: dict[str, typing.Any]) -> torch.Tensor:
        return _decode_parquet_image(value)

    def __getitem__(self, idx) -> dict:
        if not self._fast_image_backed:
            return super().__getitem__(idx)

        idx = idx.__index__() if hasattr(idx, "__index__") else int(idx)
        selected_idx = int(np.searchsorted(self._local_episode_to, idx, side="right"))
        if selected_idx < 0 or selected_idx >= len(self._selected_episodes):
            raise IndexError(f"index {idx} out of range for dataset of size {len(self)}")
        ep_idx = int(self._selected_episodes[selected_idx])
        ep_start = int(self._local_episode_from[selected_idx])
        episode_len = self._episode_lengths[ep_idx]
        local_idx = idx - ep_start

        episode_data = self._get_episode_data(ep_idx)
        item = {key: episode_data[key][local_idx] for key in episode_data}

        query_indices = None
        if self.delta_indices is not None:
            query_indices = {
                key: [max(0, min(episode_len - 1, local_idx + delta)) for delta in delta_idx]
                for key, delta_idx in self.delta_indices.items()
            }
            padding = {
                f"{key}_is_pad": torch.BoolTensor(
                    [(local_idx + delta < 0) | (local_idx + delta >= episode_len) for delta in delta_idx]
                )
                for key, delta_idx in self.delta_indices.items()
            }
            item.update(padding)
            for key, q_idx in query_indices.items():
                item[key] = episode_data[key][q_idx]

        for key in self.meta.image_keys:
            item[key] = self._decode_image(item[key])

        for key, value in list(item.items()):
            if key in self.meta.image_keys or key == "task":
                continue
            if isinstance(value, str):
                continue
            array_value = np.asarray(value)
            if np.issubdtype(array_value.dtype, np.floating):
                item[key] = torch.as_tensor(value, dtype=torch.float32)
            else:
                item[key] = torch.as_tensor(value)

        task_idx = int(item["task_index"])
        item["task"] = self.meta.tasks[task_idx]
        return item


@dataclasses.dataclass(frozen=True)
class _SamplingPlan:
    sample_weights: np.ndarray | None
    metrics: dict[str, float]


@dataclasses.dataclass(frozen=True)
class _PreprocessedCacheInfo:
    cache_dir: pathlib.Path
    signature: str
    signature_payload: dict[str, typing.Any]


def _parse_resample_bins(raw_bins: str) -> np.ndarray:
    raw = str(raw_bins).strip()
    if not raw:
        return np.asarray([2.0, 4.0, 8.0, 16.0], dtype=np.float64)
    return np.asarray([float(x) for x in raw.split(",") if x.strip()], dtype=np.float64)


def _build_weighted_sampling_plan(
    *,
    dataset_len: int,
    indices: np.ndarray,
    values: np.ndarray,
    bin_upper_bounds: np.ndarray,
    gamma: float,
    max_weight_ratio: float,
    zero_weight_indices: np.ndarray | None = None,
) -> _SamplingPlan:
    if dataset_len <= 0:
        raise ValueError(f"dataset_len must be positive, got {dataset_len}.")
    if indices.ndim != 1 or values.ndim != 1:
        raise ValueError("indices and values must be 1D arrays.")
    if indices.shape[0] != values.shape[0]:
        raise ValueError(
            f"indices and values must have the same length, got {indices.shape[0]} and {values.shape[0]}."
        )

    manifest_map: dict[int, float] = {}
    for idx, value in zip(indices.tolist(), values.tolist(), strict=False):
        idx_i = int(idx)
        if 0 <= idx_i < dataset_len:
            manifest_map[idx_i] = float(value)

    covered_indices = np.asarray(sorted(manifest_map.keys()), dtype=np.int64)
    metrics: dict[str, float] = {
        "sampler_enabled": 1.0,
        "sampler_manifest_coverage": float(covered_indices.size) / float(dataset_len),
        "sampler_num_weighted_samples": float(covered_indices.size),
        "sampler_num_bins": float(bin_upper_bounds.size + 1),
    }
    if covered_indices.size == 0:
        raise ValueError("No valid manifest indices overlap with the current dataset.")

    covered_values = np.asarray([manifest_map[int(i)] for i in covered_indices], dtype=np.float64)
    bucket_ids = np.searchsorted(bin_upper_bounds, covered_values, side="left")
    num_buckets = int(bin_upper_bounds.size + 1)
    bucket_counts = np.bincount(bucket_ids, minlength=num_buckets).astype(np.float64)
    nonzero = bucket_counts > 0
    if not np.any(nonzero):
        raise ValueError("Manifest bucket counts are all zero after filtering.")

    bucket_weights = np.ones(num_buckets, dtype=np.float64)
    bucket_weights[nonzero] = np.power(bucket_counts[nonzero], -float(gamma))
    bucket_weights /= np.min(bucket_weights[nonzero])
    bucket_weights = np.minimum(bucket_weights, float(max_weight_ratio))

    sample_weights = np.ones(dataset_len, dtype=np.float64)
    sample_weights[covered_indices] = bucket_weights[bucket_ids]
    zero_indices = np.asarray([], dtype=np.int64)
    if zero_weight_indices is not None:
        zero_indices = np.asarray(zero_weight_indices, dtype=np.int64).reshape(-1)
        if zero_indices.size > 0:
            zero_indices = zero_indices[(zero_indices >= 0) & (zero_indices < dataset_len)]
            if zero_indices.size > 0:
                zero_indices = np.unique(zero_indices)
                sample_weights[zero_indices] = 0.0
    if not np.any(sample_weights > 0):
        raise ValueError("All sample weights are zero after applying horizon resample filtering.")

    uncovered_mask = np.ones(dataset_len, dtype=bool)
    uncovered_mask[covered_indices] = False
    total_weight = float(np.sum(sample_weights))
    metrics["sampler_weight_min"] = float(np.min(sample_weights))
    metrics["sampler_weight_max"] = float(np.max(sample_weights))
    metrics["sampler_uncovered_dataset_mass"] = float(np.sum(uncovered_mask)) / float(dataset_len)
    metrics["sampler_uncovered_sample_mass"] = float(np.sum(sample_weights[uncovered_mask])) / max(total_weight, 1e-12)
    metrics["sampler_zero_weight_dataset_mass"] = float(zero_indices.size) / float(dataset_len)
    metrics["sampler_zero_weight_sample_mass"] = float(np.sum(sample_weights[zero_indices])) / max(total_weight, 1e-12)

    for bucket_idx in range(num_buckets):
        bucket_mask = np.zeros(dataset_len, dtype=bool)
        covered_bucket_indices = covered_indices[bucket_ids == bucket_idx]
        bucket_mask[covered_bucket_indices] = True
        prefix = f"sampler/bin_{bucket_idx:02d}"
        upper = float(bin_upper_bounds[bucket_idx]) if bucket_idx < bin_upper_bounds.size else -1.0
        metrics[f"{prefix}_upper"] = upper
        metrics[f"{prefix}_manifest_mass"] = float(bucket_counts[bucket_idx]) / float(covered_indices.size)
        metrics[f"{prefix}_sample_mass"] = float(np.sum(sample_weights[bucket_mask])) / max(total_weight, 1e-12)
        metrics[f"{prefix}_weight"] = float(bucket_weights[bucket_idx])

    return _SamplingPlan(sample_weights=sample_weights, metrics=metrics)


def _load_horizon_sampling_plan(
    dataset: Dataset,
    config: _config.TrainConfig,
) -> _SamplingPlan:
    manifest_raw = str(getattr(config, "horizon_resample_manifest", "")).strip()
    if not manifest_raw:
        return _SamplingPlan(sample_weights=None, metrics={})

    manifest_path = pathlib.Path(manifest_raw).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"horizon_resample_manifest not found: {manifest_path}")

    with np.load(manifest_path) as data:
        if "indices" not in data:
            raise KeyError(f"Manifest {manifest_path} is missing required key 'indices'.")
        key = str(getattr(config, "horizon_resample_key", "teacher_argmax_k")).strip().lower()
        if key not in data:
            raise KeyError(f"Manifest {manifest_path} is missing required key {key!r}.")
        indices = np.asarray(data["indices"], dtype=np.int64).reshape(-1)
        values = np.asarray(data[key], dtype=np.float64).reshape(-1)
        apply_expected_h_filter = bool(
            getattr(config, "horizon_resample_filter_expected_horizon_apply", False))
        expected_h_threshold = float(
            getattr(config, "horizon_resample_filter_expected_horizon_min", 0.0))
        dropped_indices = np.asarray([], dtype=np.int64)
        if apply_expected_h_filter:
            filter_key = "teacher_expected_horizon"
            if filter_key not in data:
                raise KeyError(
                    f"Manifest {manifest_path} is missing required key {filter_key!r} "
                    "for expected-horizon filtering."
                )
            expected_h = np.asarray(data[filter_key], dtype=np.float64).reshape(-1)
            if expected_h.shape[0] != indices.shape[0]:
                raise ValueError(
                    "Manifest teacher_expected_horizon length mismatch: "
                    f"indices={indices.shape[0]}, teacher_expected_horizon={expected_h.shape[0]}"
                )
            keep_mask = expected_h > expected_h_threshold
            dropped_indices = indices[~keep_mask]
            indices = indices[keep_mask]
            values = values[keep_mask]

    plan = _build_weighted_sampling_plan(
        dataset_len=len(dataset),
        indices=indices,
        values=values,
        bin_upper_bounds=_parse_resample_bins(getattr(config, "horizon_resample_bins", "")),
        gamma=float(getattr(config, "horizon_resample_gamma", 0.5)),
        max_weight_ratio=float(getattr(config, "horizon_resample_max_weight_ratio", 10.0)),
        zero_weight_indices=dropped_indices,
    )
    metrics = dict(plan.metrics)
    metrics["sampler_manifest_present"] = 1.0
    metrics["sampler_key_teacher_argmax_k"] = 1.0 if key == "teacher_argmax_k" else 0.0
    metrics["sampler_key_teacher_expected_horizon"] = 1.0 if key == "teacher_expected_horizon" else 0.0
    metrics["sampler_filter_expected_horizon_enabled"] = 1.0 if apply_expected_h_filter else 0.0
    metrics["sampler_filter_expected_horizon_threshold"] = expected_h_threshold
    if apply_expected_h_filter:
        total_manifest = float(indices.shape[0] + dropped_indices.shape[0])
        metrics["sampler_filter_expected_horizon_kept_manifest_fraction"] = float(indices.shape[0]) / max(total_manifest, 1e-12)
        metrics["sampler_filter_expected_horizon_dropped_manifest_fraction"] = float(dropped_indices.shape[0]) / max(total_manifest, 1e-12)
    return _SamplingPlan(sample_weights=plan.sample_weights, metrics=metrics)


def _batch_sampling_metrics(sample_weight: np.ndarray | at.Array) -> dict[str, at.Array]:
    sample_weight = jnp.asarray(sample_weight, dtype=jnp.float32)
    return {
        "batch_sample_weight_mean": jnp.mean(sample_weight),
        "batch_sample_weight_std": jnp.std(sample_weight),
    }


def _batch_multitask_metrics(task_index: torch.Tensor | np.ndarray, num_tasks: int) -> dict[str, float]:
    if isinstance(task_index, torch.Tensor):
        task_index = task_index.detach().cpu().numpy()
    task_index = np.asarray(task_index, dtype=np.int64).reshape(-1)
    counts = np.bincount(task_index, minlength=num_tasks)
    if counts.shape[0] != num_tasks:
        raise ValueError(f"Batch contains an out-of-range task index: {task_index}")
    return {
        "batch_task_unique_count": float(np.count_nonzero(counts)),
        **{
            f"batch_task_count_{task_idx:02d}": float(count)
            for task_idx, count in enumerate(counts.tolist())
        },
    }


def _json_default(value):
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return np.asarray(value).tolist()
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


def _as_python_scalar(value: typing.Any) -> typing.Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def _norm_stats_fingerprint(norm_stats: dict[str, typing.Any] | None) -> str:
    if not norm_stats:
        return ""

    digest = hashlib.sha256()
    flat_stats = _transforms.flatten_dict(norm_stats)
    for key in sorted(flat_stats):
        digest.update(key.encode("utf-8"))
        stats = flat_stats[key]
        for attr in ("mean", "std", "q01", "q99"):
            value = getattr(stats, attr)
            if value is None:
                digest.update(f"{attr}:none".encode("utf-8"))
                continue
            arr = np.asarray(value)
            digest.update(f"{attr}:{arr.shape}:{arr.dtype}".encode("utf-8"))
            digest.update(arr.tobytes())
    return digest.hexdigest()


def _preprocessed_cache_signature_payload(
    config: _config.TrainConfig,
    data_config: _config.DataConfig,
) -> dict[str, typing.Any]:
    cache_config_name = str(
        getattr(config, "preprocessed_cache_compat_config_name", "") or config.name
    )
    cache_data_factory_type = str(
        getattr(config, "preprocessed_cache_compat_data_factory_type", "") or type(config.data).__name__
    )
    return {
        "cache_version": _PREPROCESSED_CACHE_VERSION,
        "train_config_name": cache_config_name,
        "repo_id": data_config.repo_id,
        "lerobot_root": data_config.lerobot_root,
        "data_factory_type": cache_data_factory_type,
        "model_config_type": type(config.model).__name__,
        "model_type": str(config.model.model_type),
        "action_horizon": int(config.model.action_horizon),
        "action_dim": int(config.model.action_dim),
        "max_token_len": int(config.model.max_token_len),
        "use_qha": bool(getattr(config.model, "use_qha", False)),
        "qha_train_mode": str(getattr(config.model, "qha_train_mode", "")),
        "qha_candidate_mode": str(getattr(config.model, "qha_candidate_mode", "")),
        "qha_history_max_length": int(getattr(config.model, "qha_history_max_length", 0)),
        "smoke_test_max_frames": config.smoke_test_max_frames,
        "default_prompt": getattr(config.data, "default_prompt", None),
        "prompt_from_task": bool(getattr(data_config, "prompt_from_task", False)),
        "prompt_index_key": str(getattr(data_config, "prompt_index_key", "task_index")),
        "adapt_to_pi": bool(getattr(config.data, "adapt_to_pi", False)),
        "use_delta_joint_actions": bool(getattr(config.data, "use_delta_joint_actions", False)),
        "use_quantile_norm": bool(getattr(data_config, "use_quantile_norm", False)),
        "norm_stats_fingerprint": _norm_stats_fingerprint(data_config.norm_stats),
        "action_sequence_keys": list(data_config.action_sequence_keys),
        "synthetic_zero_fields": list(
            getattr(config, "preprocessed_cache_synthetic_zero_fields", ())
        ),
        "transform_version": _PREPROCESSED_CACHE_VERSION,
    }


def resolve_preprocessed_cache_info(
    config: _config.TrainConfig,
    data_config: _config.DataConfig,
) -> _PreprocessedCacheInfo:
    if not data_config.repo_id:
        raise ValueError("Cannot resolve preprocessed cache path without data_config.repo_id.")

    payload = _preprocessed_cache_signature_payload(config, data_config)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    signature = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    cache_dir = config.preprocessed_cache_root_dir / str(data_config.repo_id) / signature
    return _PreprocessedCacheInfo(cache_dir=cache_dir, signature=signature, signature_payload=payload)


def _preprocessed_cache_field_filename(key: str) -> str:
    return key.replace("/", "__") + ".npy"


def _load_preprocessed_cache_manifest(cache_dir: pathlib.Path) -> dict[str, typing.Any]:
    manifest_path = cache_dir / _PREPROCESSED_CACHE_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(f"Preprocessed cache manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _validate_existing_preprocessed_cache(
    cache_info: _PreprocessedCacheInfo,
) -> pathlib.Path | None:
    if not cache_info.cache_dir.exists():
        return None
    manifest = _load_preprocessed_cache_manifest(cache_info.cache_dir)
    if manifest.get("signature") != cache_info.signature:
        raise ValueError(
            "Preprocessed cache signature mismatch. "
            f"Expected {cache_info.signature}, found {manifest.get('signature')} in {cache_info.cache_dir}."
        )
    return cache_info.cache_dir


def _unwrap_transform_pipeline(
    dataset: Dataset,
) -> tuple[Dataset, tuple[_transforms.DataTransformFn, ...]]:
    transforms: list[_transforms.DataTransformFn] = []
    base_dataset = dataset
    while isinstance(base_dataset, TransformedDataset):
        transform = base_dataset._transform
        if isinstance(transform, _transforms.CompositeTransform):
            transforms = list(transform.transforms) + transforms
        else:
            transforms = [typing.cast(_transforms.DataTransformFn, transform), *transforms]
        base_dataset = base_dataset._dataset
    return base_dataset, tuple(transforms)


def _write_preprocessed_cache_item(
    *,
    item: dict[str, typing.Any],
    row_index: int,
    num_samples: int,
    tmp_dir: pathlib.Path,
    writers: dict[str, np.memmap],
    field_specs: dict[str, dict[str, typing.Any]],
    reference_keys: list[str] | None,
    synthetic_zero_fields: frozenset[str],
) -> list[str]:
    flat_item = _transforms.flatten_dict(item)
    keys = sorted(flat_item.keys())
    if reference_keys is None:
        reference_keys = keys
    elif keys != reference_keys:
        raise ValueError(
            "Preprocessed cache build encountered inconsistent sample keys. "
            f"Expected {reference_keys}, got {keys} at row_index={row_index}."
        )

    for key in keys:
        arr = np.asarray(flat_item[key])
        if key in synthetic_zero_fields:
            if np.any(arr):
                raise ValueError(
                    f"Configured synthetic-zero field {key!r} contains non-zero values "
                    f"at row_index={row_index}."
                )
            shape = (num_samples, *arr.shape)
            if key not in field_specs:
                field_specs[key] = {
                    "synthetic": "zeros",
                    "dtype": np.dtype(arr.dtype).name,
                    "shape": list(shape),
                }
            else:
                expected_shape = tuple(field_specs[key]["shape"][1:])
                expected_dtype = np.dtype(field_specs[key]["dtype"])
                if arr.shape != expected_shape or np.dtype(arr.dtype) != expected_dtype:
                    raise ValueError(
                        f"Synthetic-zero field {key!r} changed shape or dtype "
                        f"at row_index={row_index}."
                    )
            continue
        if key not in writers:
            filename = _preprocessed_cache_field_filename(key)
            shape = (num_samples, *arr.shape)
            writers[key] = np.lib.format.open_memmap(
                tmp_dir / filename,
                mode="w+",
                dtype=arr.dtype,
                shape=shape,
            )
            field_specs[key] = {
                "filename": filename,
                "dtype": np.dtype(arr.dtype).name,
                "shape": list(shape),
            }
        else:
            expected_shape = tuple(field_specs[key]["shape"][1:])
            expected_dtype = np.dtype(field_specs[key]["dtype"])
            if arr.shape != expected_shape:
                raise ValueError(
                    f"Cached field {key!r} changed shape from {expected_shape} to {arr.shape} at row_index={row_index}."
                )
            if np.dtype(arr.dtype) != expected_dtype:
                raise ValueError(
                    f"Cached field {key!r} changed dtype from {expected_dtype} to {arr.dtype} at row_index={row_index}."
                )
        writers[key][row_index] = arr

    return reference_keys


def _transform_preprocessed_cache_chunk(
    items: list[dict[str, typing.Any]],
    transforms: tuple[_transforms.DataTransformFn, ...],
    *,
    resize_batch_size: int | None = None,
) -> list[dict[str, typing.Any]]:
    resize_indices = [
        index
        for index, transform in enumerate(transforms)
        if isinstance(transform, _transforms.ResizeImages)
    ]
    if len(resize_indices) != 1 or len(items) <= 1:
        transform = _transforms.compose(transforms)
        return [typing.cast(dict[str, typing.Any], transform(item)) for item in items]

    resize_index = resize_indices[0]
    prefix = _transforms.compose(transforms[:resize_index])
    resize = transforms[resize_index]
    suffix = _transforms.compose(transforms[resize_index + 1 :])
    prefixed = [typing.cast(dict[str, typing.Any], prefix(item)) for item in items]
    image_keys = tuple(prefixed[0]["image"])
    if any(tuple(item["image"]) != image_keys for item in prefixed):
        raise ValueError("Preprocessed cache chunk contains inconsistent image keys.")
    batched_images = {
        key: np.stack([np.asarray(item["image"][key]) for item in prefixed])
        for key in image_keys
    }
    if resize_batch_size is not None:
        if resize_batch_size < len(items):
            raise ValueError(
                f"resize_batch_size={resize_batch_size} is smaller than chunk length={len(items)}."
            )
        pad_count = resize_batch_size - len(items)
        if pad_count:
            batched_images = {
                key: np.pad(value, ((0, pad_count), *[(0, 0)] * (value.ndim - 1)))
                for key, value in batched_images.items()
            }
    if os.environ.get("OPENPI_CACHE_RESIZE_ACCELERATOR", "").lower() == "gpu":
        global _CACHE_RESIZE_ACCELERATOR_LOGGED
        device = jax.devices("gpu")[0]
        if not _CACHE_RESIZE_ACCELERATOR_LOGGED:
            logging.info(
                "Preprocessed cache resize accelerator backend=%s device=%s",
                jax.default_backend(),
                device,
            )
            _CACHE_RESIZE_ACCELERATOR_LOGGED = True
        resized_images = {
            key: _image_tools.resize_with_pad(
                jax.device_put(_transforms._ensure_channel_last_image(value), device),
                resize.height,
                resize.width,
            )
            for key, value in batched_images.items()
        }
    else:
        resized_images = typing.cast(
            dict[str, typing.Any],
            resize({"image": batched_images}),
        )["image"]
    resized_images = {
        key: np.asarray(value)[: len(items)]
        for key, value in resized_images.items()
    }

    transformed = []
    for item_index, item in enumerate(prefixed):
        item["image"] = {
            key: value[item_index]
            for key, value in resized_images.items()
        }
        transformed.append(typing.cast(dict[str, typing.Any], suffix(item)))
    return transformed


def _build_preprocessed_cache_lerobot_episodewise(
    *,
    dataset: lerobot_dataset.LeRobotDataset,
    transforms: tuple[_transforms.DataTransformFn, ...],
    cache_info: _PreprocessedCacheInfo,
    tmp_dir: pathlib.Path,
    num_samples: int,
    chunk_size: int,
    synthetic_zero_fields: frozenset[str],
) -> pathlib.Path:
    writers: dict[str, np.memmap] = {}
    field_specs: dict[str, dict[str, typing.Any]] = {}
    reference_keys: list[str] | None = None
    processed_samples = 0

    episodes = dataset.episodes if dataset.episodes is not None else list(range(dataset.meta.total_episodes))
    with tqdm.tqdm(total=num_samples, desc=f"cache:{dataset.repo_id}", dynamic_ncols=True) as pbar:
        for ep_idx in episodes:
            if processed_samples >= num_samples:
                break
            ep_start = int(dataset.episode_data_index["from"][ep_idx])
            ep_end = int(dataset.episode_data_index["to"][ep_idx])
            episode_len = ep_end - ep_start
            if episode_len <= 0:
                continue

            episode_ds = dataset.hf_dataset.select(range(ep_start, ep_end))
            episode_timestamps = [float(_as_python_scalar(ts)) for ts in episode_ds["timestamp"]]
            episode_video_frames = {}
            for vid_key in dataset.meta.video_keys:
                video_path = dataset.root / dataset.meta.get_video_file_path(ep_idx, vid_key)
                episode_video_frames[vid_key] = lerobot_dataset.decode_video_frames(
                    video_path,
                    episode_timestamps,
                    dataset.tolerance_s,
                    dataset.video_backend,
                )

            query_index_cache: dict[str, np.ndarray] = {}
            padding_cache: dict[str, np.ndarray] = {}
            query_value_cache: dict[str, torch.Tensor] = {}
            if dataset.delta_indices is not None:
                local_positions = np.arange(episode_len, dtype=np.int64)[:, None]
                for key, delta_idx in dataset.delta_indices.items():
                    offsets = np.asarray(delta_idx, dtype=np.int64)[None, :]
                    query_indices = np.clip(local_positions + offsets, 0, episode_len - 1)
                    padding = np.logical_or(local_positions + offsets < 0, local_positions + offsets >= episode_len)
                    query_index_cache[key] = query_indices
                    padding_cache[f"{key}_is_pad"] = padding
                    if key not in dataset.meta.video_keys:
                        column_values = np.asarray(list(episode_ds[key]))
                        if np.issubdtype(column_values.dtype, np.floating):
                            query_value_cache[key] = torch.as_tensor(column_values, dtype=torch.float32)
                        else:
                            query_value_cache[key] = torch.as_tensor(column_values)

            for chunk_start in range(0, episode_len, chunk_size):
                if processed_samples >= num_samples:
                    break
                chunk_end = min(chunk_start + chunk_size, episode_len)
                chunk_batch = episode_ds[chunk_start:chunk_end]

                chunk_items = []
                for local_idx in range(chunk_start, chunk_end):
                    if processed_samples >= num_samples:
                        break
                    chunk_offset = local_idx - chunk_start
                    item = {
                        key: value[chunk_offset] if isinstance(value, list) else value[chunk_offset]
                        for key, value in chunk_batch.items()
                    }
                    for vid_key in dataset.meta.video_keys:
                        item[vid_key] = episode_video_frames[vid_key][local_idx]
                    for image_key in dataset.meta.image_keys:
                        if image_key in item:
                            item[image_key] = _decode_parquet_image(item[image_key])

                    if dataset.delta_indices is not None:
                        for key, query_indices in query_index_cache.items():
                            item[f"{key}_is_pad"] = torch.as_tensor(
                                padding_cache[f"{key}_is_pad"][local_idx], dtype=torch.bool
                            )
                            if key in dataset.meta.video_keys:
                                continue
                            item[key] = query_value_cache[key][torch.as_tensor(query_indices[local_idx], dtype=torch.long)]

                    task_idx = int(_as_python_scalar(item["task_index"]))
                    item["task"] = dataset.meta.tasks[task_idx]
                    chunk_items.append(item)

                for transformed_item in _transform_preprocessed_cache_chunk(
                    chunk_items,
                    transforms,
                    resize_batch_size=chunk_size,
                ):
                    reference_keys = _write_preprocessed_cache_item(
                        item=typing.cast(dict[str, typing.Any], transformed_item),
                        row_index=processed_samples,
                        num_samples=num_samples,
                        tmp_dir=tmp_dir,
                        writers=writers,
                        field_specs=field_specs,
                        reference_keys=reference_keys,
                        synthetic_zero_fields=synthetic_zero_fields,
                    )
                    processed_samples += 1
                    pbar.update(1)
                    if (
                        processed_samples == 1
                        or processed_samples == num_samples
                        or processed_samples % _PREPROCESSED_CACHE_LOG_INTERVAL == 0
                    ):
                        logging.info(
                            "Preprocessed cache progress repo_id=%s samples=%d/%d cache_dir=%s",
                            dataset.repo_id,
                            processed_samples,
                            num_samples,
                            cache_info.cache_dir,
                        )

    if processed_samples != num_samples:
        raise ValueError(f"Preprocessed cache build wrote {processed_samples} samples, expected {num_samples}.")

    for writer in writers.values():
        writer.flush()

    manifest = {
        "cache_version": _PREPROCESSED_CACHE_VERSION,
        "signature": cache_info.signature,
        "signature_payload": cache_info.signature_payload,
        "num_samples": num_samples,
        "fields": field_specs,
    }
    with (tmp_dir / _PREPROCESSED_CACHE_MANIFEST).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=_json_default)

    tmp_dir.rename(cache_info.cache_dir)
    return cache_info.cache_dir


def build_preprocessed_cache(
    config: _config.TrainConfig,
    *,
    force: bool = False,
    skip_norm_stats: bool = False,
    batch_size: int = 1,
    num_workers: int = 0,
    prefetch_factor: int = 2,
) -> pathlib.Path:
    data_config = config.data.create(config.assets_dirs, config.model)
    cache_info = resolve_preprocessed_cache_info(config, data_config)
    existing = None
    try:
        existing = _validate_existing_preprocessed_cache(cache_info)
    except FileNotFoundError:
        existing = None
    if existing is not None and not force:
        return existing
    if existing is not None and force:
        shutil.rmtree(existing)

    if cache_info.cache_dir.exists():
        if not force:
            raise ValueError(
                "Preprocessed cache directory already exists but is invalid or mismatched: "
                f"{cache_info.cache_dir}. Re-run with force to rebuild."
            )
        shutil.rmtree(cache_info.cache_dir)

    dataset = create_dataset(data_config, config.model)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)
    base_dataset, transforms = _unwrap_transform_pipeline(dataset)

    num_samples = len(dataset)
    if config.smoke_test_max_frames is not None:
        num_samples = min(num_samples, int(config.smoke_test_max_frames))
    if num_samples <= 0:
        raise ValueError(f"Cannot build preprocessed cache for empty dataset: {data_config.repo_id}")

    cache_info.cache_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = cache_info.cache_dir.parent / f"{cache_info.cache_dir.name}.tmp-{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)

    if isinstance(base_dataset, lerobot_dataset.LeRobotDataset):
        try:
            return _build_preprocessed_cache_lerobot_episodewise(
                dataset=base_dataset,
                transforms=transforms,
                cache_info=cache_info,
                tmp_dir=tmp_dir,
                num_samples=num_samples,
                chunk_size=max(1, int(batch_size)),
                synthetic_zero_fields=frozenset(
                    getattr(config, "preprocessed_cache_synthetic_zero_fields", ())
                ),
            )
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}.")
    if prefetch_factor <= 0:
        raise ValueError(f"prefetch_factor must be positive, got {prefetch_factor}.")

    mp_context = None
    if num_workers > 0:
        mp_context = multiprocessing.get_context("spawn")
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        multiprocessing_context=mp_context,
        persistent_workers=num_workers > 0,
        collate_fn=_collate_fn,
        worker_init_fn=_worker_init_fn,
        drop_last=False,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
        loader_kwargs["pin_memory"] = False
    cache_loader = torch.utils.data.DataLoader(
        typing.cast(torch.utils.data.Dataset, dataset),
        **loader_kwargs,
    )

    writers: dict[str, np.memmap] = {}
    field_specs: dict[str, dict[str, typing.Any]] = {}
    reference_keys: list[str] | None = None
    processed_samples = 0
    try:
        progress_desc = f"cache:{data_config.repo_id}"
        with tqdm.tqdm(total=num_samples, desc=progress_desc, dynamic_ncols=True) as pbar:
            for batch in cache_loader:
                flat_batch = _transforms.flatten_dict(batch)
                keys = sorted(flat_batch.keys())
                if reference_keys is None:
                    reference_keys = keys
                elif keys != reference_keys:
                    raise ValueError(
                        "Preprocessed cache build encountered inconsistent sample keys. "
                        f"Expected {reference_keys}, got {keys} at processed_samples={processed_samples}."
                    )

                batch_size_actual: int | None = None
                for key in keys:
                    arr = np.asarray(flat_batch[key])
                    if batch_size_actual is None:
                        if arr.ndim == 0:
                            raise ValueError(f"Batched cache field {key!r} must include a batch dimension.")
                        batch_size_actual = int(arr.shape[0])
                    if key not in writers:
                        filename = _preprocessed_cache_field_filename(key)
                        shape = (num_samples, *arr.shape[1:])
                        writers[key] = np.lib.format.open_memmap(
                            tmp_dir / filename,
                            mode="w+",
                            dtype=arr.dtype,
                            shape=shape,
                        )
                        field_specs[key] = {
                            "filename": filename,
                            "dtype": np.dtype(arr.dtype).name,
                            "shape": list(shape),
                            }
                    else:
                        expected_shape = tuple(field_specs[key]["shape"][1:])
                        expected_dtype = np.dtype(field_specs[key]["dtype"])
                        if arr.shape[1:] != expected_shape:
                            raise ValueError(
                                f"Cached field {key!r} changed shape from {expected_shape} to {arr.shape[1:]} "
                                f"at processed_samples={processed_samples}."
                            )
                        if np.dtype(arr.dtype) != expected_dtype:
                            raise ValueError(
                                f"Cached field {key!r} changed dtype from {expected_dtype} to {arr.dtype} "
                                f"at processed_samples={processed_samples}."
                            )
                    writers[key][processed_samples:processed_samples + batch_size_actual] = arr

                if batch_size_actual is None:
                    raise ValueError("Encountered empty batch while building preprocessed cache.")
                processed_samples += batch_size_actual
                pbar.update(batch_size_actual)
                completed = processed_samples
                if completed == 1 or completed == num_samples or completed % _PREPROCESSED_CACHE_LOG_INTERVAL == 0:
                    logging.info(
                        "Preprocessed cache progress repo_id=%s samples=%d/%d cache_dir=%s",
                        data_config.repo_id,
                        completed,
                        num_samples,
                        cache_info.cache_dir,
                    )

        if processed_samples != num_samples:
            raise ValueError(
                f"Preprocessed cache build wrote {processed_samples} samples, expected {num_samples}."
            )

        for writer in writers.values():
            writer.flush()

        manifest = {
            "cache_version": _PREPROCESSED_CACHE_VERSION,
            "signature": cache_info.signature,
            "signature_payload": cache_info.signature_payload,
            "num_samples": num_samples,
            "fields": field_specs,
        }
        with (tmp_dir / _PREPROCESSED_CACHE_MANIFEST).open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True, default=_json_default)

        tmp_dir.rename(cache_info.cache_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    finally:
        writers.clear()

    return cache_info.cache_dir


class CachedDataset(Dataset[dict[str, typing.Any]]):

    def __init__(self, cache_info: _PreprocessedCacheInfo):
        manifest = _load_preprocessed_cache_manifest(cache_info.cache_dir)
        if manifest.get("signature") != cache_info.signature:
            raise ValueError(
                "Preprocessed cache signature mismatch. "
                f"Expected {cache_info.signature}, found {manifest.get('signature')} in {cache_info.cache_dir}."
            )

        self._cache_dir = cache_info.cache_dir
        self._num_samples = int(manifest["num_samples"])
        self._field_specs = dict(manifest["fields"])
        self._arrays = self._open_arrays()
        self._synthetic_values = self._make_synthetic_values()

    def _open_arrays(self) -> dict[str, np.ndarray]:
        """Open local cache files in the owning process as read-only memmaps."""
        return {
            key: np.load(self._cache_dir / spec["filename"], mmap_mode="r")
            for key, spec in self._field_specs.items()
            if spec.get("synthetic") is None
        }

    def _make_synthetic_values(self) -> dict[str, np.ndarray]:
        return {
            key: np.zeros(tuple(spec["shape"][1:]), dtype=np.dtype(spec["dtype"]))
            for key, spec in self._field_specs.items()
            if spec.get("synthetic") == "zeros"
        }

    def __getstate__(self):
        # PyTorch spawn workers pickle datasets. Pickling numpy.memmap objects
        # materializes the full array, which turns a small worker startup into a
        # multi-dozen-GB copy of every QHA cache. Send only cache metadata and
        # reopen local memmaps in the child instead.
        state = dict(self.__dict__)
        state["_arrays"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._arrays = self._open_arrays()
        self._synthetic_values = self._make_synthetic_values()

    def __getitem__(self, index: SupportsIndex) -> dict[str, typing.Any]:
        idx = index.__index__()
        flat_item = {key: arr[idx] for key, arr in self._arrays.items()}
        flat_item.update(self._synthetic_values)
        return typing.cast(dict[str, typing.Any], _transforms.unflatten_dict(flat_item))

    def __len__(self) -> int:
        return self._num_samples


def _cached_collate_fn(items):
    batch = torch.utils.data.default_collate(items)
    return jax.tree.map(lambda x: x.numpy() if isinstance(x, torch.Tensor) else x, batch)


def create_dataset(data_config: _config.DataConfig, model_config: _model.BaseModelConfig) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    repo_root = (
        pathlib.Path(data_config.lerobot_root) / repo_id
        if data_config.lerobot_root is not None
        else None
    )
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=repo_root)

    history_len = 0
    if getattr(model_config, "use_qha", False):
        history_len = int(getattr(model_config, "qha_history_max_length", 0))

    # Query a concatenated action window: [history(-L..-1) + future(0..H-1)] when QHA is enabled.
    # Otherwise keep the original future-only query (0..H-1).
    delta_timestamps = {
        key: [t / dataset_meta.fps for t in range(-history_len, model_config.action_horizon)]
        for key in data_config.action_sequence_keys
    }

    dataset = _FastInitLeRobotDataset(
        data_config.repo_id,
        root=repo_root,
        delta_timestamps=delta_timestamps,
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(
            dataset,
            [
                _transforms.PromptFromLeRobotTask(
                    dataset_meta.tasks,
                    index_key=data_config.prompt_index_key,
                )
            ],
        )

    return dataset


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError("Normalization stats not found. "
                             "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`.")
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
) -> DataLoader[tuple[tuple[_model.Observation, _model.Actions], dict[str, at.Array]]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    if isinstance(data_config, _config.MultiDataConfig):
        return _create_multi_data_loader(
            config,
            data_config,
            sharding=sharding,
            skip_norm_stats=skip_norm_stats,
            shuffle=shuffle,
            num_batches=num_batches,
            num_workers=num_workers,
        )

    use_cached_dataset = bool(getattr(config, "use_preprocessed_cache", False))
    if use_cached_dataset:
        cache_info = resolve_preprocessed_cache_info(config, data_config)
        if _validate_existing_preprocessed_cache(cache_info) is None:
            task_hint = ""
            repo_id = str(data_config.repo_id or "")
            if repo_id.endswith("_aloha-agilex_clean_50_repo"):
                task_hint = repo_id[: -len("_aloha-agilex_clean_50_repo")]
            build_cmd = [
                "python",
                "scripts/build_preprocessed_cache.py",
                config.name,
            ]
            if task_hint:
                build_cmd.extend(["--tasks", task_hint])
            if hasattr(config.model, "qha_train_mode"):
                build_cmd.extend(["--qha-train-mode", str(getattr(config.model, "qha_train_mode"))])
            if hasattr(config.model, "qha_candidate_mode"):
                build_cmd.extend(["--qha-candidate-mode", str(getattr(config.model, "qha_candidate_mode"))])
            if hasattr(config.model, "qha_history_max_length"):
                build_cmd.extend(["--qha-history-max-length", str(int(getattr(config.model, "qha_history_max_length")))])
            build_cmd.extend(["--max-token-len", str(int(config.model.max_token_len))])
            build_cmd.extend(["--batch-size", str(int(config.batch_size))])
            raise FileNotFoundError(
                "Required preprocessed cache is missing. "
                f"Expected cache at {cache_info.cache_dir}. Build it first, e.g. "
                f"{' '.join(build_cmd)}"
            )
        dataset = CachedDataset(cache_info)
        collate_fn = _cached_collate_fn
    else:
        dataset = create_dataset(data_config, config.model)
        dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)
        collate_fn = _collate_fn
    sampling_plan = _load_horizon_sampling_plan(dataset, config)
    instrumentation_weights = (
        np.asarray(sampling_plan.sample_weights, dtype=np.float32)
        if sampling_plan.sample_weights is not None
        else np.ones(len(dataset), dtype=np.float32)
    )
    dataset = InstrumentedDataset(dataset, instrumentation_weights)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        multiprocessing_context_name=config.dataloader_multiprocessing_context,
        prefetch_factor=config.dataloader_prefetch_factor,
        pin_memory=config.dataloader_pin_memory,
        seed=config.seed,
        sample_weights=sampling_plan.sample_weights,
        weighted_num_samples=int(getattr(config, "horizon_resample_num_samples", 0)),
        weighted_replacement=bool(getattr(config, "horizon_resample_replacement", True)),
        collate_fn=collate_fn,
    )

    class DataLoaderImpl(DataLoader):

        def __init__(
            self,
            data_config: _config.DataConfig,
            data_loader: TorchDataLoader,
            sampling_metrics: dict[str, float],
        ):
            self._data_config = data_config
            self._data_loader = data_loader
            self._sampling_metrics = sampling_metrics

        def data_config(self) -> _config.DataConfig:
            return self._data_config

        def data_configs(self) -> tuple[_config.DataConfig, ...]:
            return (self._data_config,)

        def sampling_metrics(self) -> dict[str, float]:
            return dict(self._sampling_metrics)

        def __iter__(self):
            for batch in self._data_loader:
                batch_metrics = _batch_sampling_metrics(batch[_SAMPLE_WEIGHT_KEY])
                batch_dict = dict(batch)
                batch_dict.pop(_SAMPLE_WEIGHT_KEY, None)
                batch_dict = jax.tree.map(
                    lambda x: jnp.asarray(x.numpy()) if isinstance(x, torch.Tensor) else
                    (jnp.asarray(x) if isinstance(x, np.ndarray) else x),
                    batch_dict,
                )
                yield (_model.Observation.from_dict(batch_dict), batch_dict["actions"]), batch_metrics

    return DataLoaderImpl(data_config, data_loader, sampling_plan.metrics)


def _create_multi_data_loader(
    config: _config.TrainConfig,
    multi_config: _config.MultiDataConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
) -> DataLoader[tuple[tuple[_model.Observation, _model.Actions], dict[str, at.Array]]]:
    """Create a balanced joint loader over multiple task datasets."""
    use_cached_dataset = bool(getattr(config, "use_preprocessed_cache", False))
    collate_fn = _collate_fn
    per_task_datasets: list[Dataset] = []
    per_task_lens: list[int] = []

    for task_index, sub_config in enumerate(multi_config.sub_configs):
        if use_cached_dataset:
            cache_info = resolve_preprocessed_cache_info(config, sub_config)
            if _validate_existing_preprocessed_cache(cache_info) is None:
                raise FileNotFoundError(
                    "Required preprocessed cache is missing. "
                    f"Expected cache at {cache_info.cache_dir} for repo {sub_config.repo_id}. Build it first."
                )
            dataset = CachedDataset(cache_info)
            collate_fn = _cached_collate_fn
        else:
            dataset = create_dataset(sub_config, config.model)
            dataset = transform_dataset(dataset, sub_config, skip_norm_stats=skip_norm_stats)
        per_task_datasets.append(TaskIndexedDataset(dataset, task_index))
        per_task_lens.append(len(dataset))

    dataset = ConcatDataset(per_task_datasets)
    total_len = len(dataset)
    num_tasks = len(per_task_datasets)
    task_weight = 1.0 / float(num_tasks)
    per_sample_weights = np.concatenate([
        np.full(n, task_weight / float(n), dtype=np.float32) for n in per_task_lens
    ])
    sampling_metrics = {
        "multi_task_num_tasks": float(num_tasks),
        "multi_task_total_samples": float(total_len),
    }
    dataset = InstrumentedDataset(dataset, per_sample_weights)

    multi_task_batch_mode = str(getattr(config, "multi_task_batch_mode", "mixed")).strip().lower()
    if multi_task_batch_mode not in ("mixed", "task_homogeneous", "task_balanced_mixed", "task_balanced_coverage"):
        raise ValueError(f"Unsupported multi_task_batch_mode: {multi_task_batch_mode}")
    batch_sampler = None
    torch_sample_weights = per_sample_weights
    if multi_task_batch_mode == "task_homogeneous":
        batch_sampler = TaskHomogeneousBatchSampler(
            task_lengths=per_task_lens,
            batch_size=config.batch_size // jax.process_count(),
            seed=int(config.seed),
            num_samples=int(getattr(config, "horizon_resample_num_samples", 0)),
            batches_per_task_block=int(getattr(config, "multi_task_batches_per_task_block", 1)),
        )
        torch_sample_weights = None
        sampling_metrics.update({
            "multi_task_batch_mode_task_homogeneous": 1.0,
            "multi_task_batch_mode_task_balanced_mixed": 0.0,
            "multi_task_batches_per_epoch": float(len(batch_sampler)),
            "multi_task_batches_per_task_block": float(
                max(1, int(getattr(config, "multi_task_batches_per_task_block", 1)))
            ),
        })
    elif multi_task_batch_mode == "task_balanced_mixed":
        batch_sampler = TaskBalancedMixedBatchSampler(
            task_lengths=per_task_lens,
            batch_size=config.batch_size // jax.process_count(),
            seed=int(config.seed),
            num_samples=int(getattr(config, "horizon_resample_num_samples", 0)),
        )
        torch_sample_weights = None
        sampling_metrics.update({
            "multi_task_batch_mode_task_homogeneous": 0.0,
            "multi_task_batch_mode_task_balanced_mixed": 1.0,
            "multi_task_batches_per_epoch": float(len(batch_sampler)),
        })
    elif multi_task_batch_mode == "task_balanced_coverage":
        batch_sampler = TaskBalancedCoverageBatchSampler(
            task_lengths=per_task_lens,
            batch_size=config.batch_size // jax.process_count(),
            seed=int(config.seed),
            num_samples=int(getattr(config, "horizon_resample_num_samples", 0)),
        )
        torch_sample_weights = None
        sampling_metrics.update({
            "multi_task_batch_mode_task_homogeneous": 0.0,
            "multi_task_batch_mode_task_balanced_mixed": 0.0,
            "multi_task_batch_mode_task_balanced_coverage": 1.0,
            "multi_task_batches_per_epoch": float(len(batch_sampler)),
        })
    else:
        sampling_metrics["multi_task_batch_mode_task_homogeneous"] = 0.0
        sampling_metrics["multi_task_batch_mode_task_balanced_mixed"] = 0.0
        sampling_metrics["multi_task_batch_mode_task_balanced_coverage"] = 0.0

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        multiprocessing_context_name=config.dataloader_multiprocessing_context,
        prefetch_factor=config.dataloader_prefetch_factor,
        pin_memory=config.dataloader_pin_memory,
        seed=config.seed,
        sample_weights=torch_sample_weights,
        weighted_num_samples=int(getattr(config, "horizon_resample_num_samples", 0)),
        weighted_replacement=bool(getattr(config, "horizon_resample_replacement", True)),
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
    )

    class MultiDataLoaderImpl(DataLoader):

        def __init__(
            self,
            data_configs: tuple[_config.DataConfig, ...],
            data_loader: TorchDataLoader,
            sampling_metrics: dict[str, float],
        ):
            self._data_configs = data_configs
            self._data_loader = data_loader
            self._sampling_metrics = sampling_metrics

        def data_config(self) -> _config.DataConfig:
            return self._data_configs[0]

        def data_configs(self) -> tuple[_config.DataConfig, ...]:
            return self._data_configs

        def sampling_metrics(self) -> dict[str, float]:
            return dict(self._sampling_metrics)

        def __iter__(self):
            for batch in self._data_loader:
                batch_metrics = _batch_sampling_metrics(batch[_SAMPLE_WEIGHT_KEY])
                batch_metrics.update(_batch_multitask_metrics(batch["task_index"], len(self._data_configs)))
                batch_dict = dict(batch)
                batch_dict.pop(_SAMPLE_WEIGHT_KEY, None)
                batch_dict = jax.tree.map(
                    lambda x: jnp.asarray(x.numpy()) if isinstance(x, torch.Tensor) else
                    (jnp.asarray(x) if isinstance(x, np.ndarray) else x),
                    batch_dict,
                )
                yield (_model.Observation.from_dict(batch_dict), batch_dict["actions"]), batch_metrics

    return MultiDataLoaderImpl(tuple(multi_config.sub_configs), data_loader, sampling_metrics)


class TorchDataLoader:

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_workers: int = 0,
        multiprocessing_context_name: str = "spawn",
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        seed: int = 0,
        sample_weights: np.ndarray | None = None,
        weighted_num_samples: int = 0,
        weighted_replacement: bool = True,
        batch_sampler=None,
        collate_fn=None,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B", )),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            context_name = str(multiprocessing_context_name).strip().lower()
            if context_name not in ("spawn", "forkserver"):
                raise ValueError(f"Unsupported data-loader multiprocessing context: {context_name}")
            if context_name == "forkserver":
                multiprocessing.set_forkserver_preload(
                    [
                        "openpi.training.data_loader",
                        "openpi.training.config",
                    ]
                )
            mp_context = multiprocessing.get_context(context_name)

        generator = torch.Generator()
        generator.manual_seed(seed)
        sampler = None
        loader_kwargs = dict(
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=collate_fn or _collate_fn,
            worker_init_fn=_worker_init_fn,
            generator=generator,
        )
        if batch_sampler is not None:
            loader_kwargs["batch_sampler"] = batch_sampler
        else:
            loader_kwargs.update(
                batch_size=local_batch_size,
                shuffle=bool(shuffle),
                sampler=None,
                drop_last=True,
            )
        if sample_weights is not None and batch_sampler is None:
            num_samples = int(weighted_num_samples) if int(weighted_num_samples) > 0 else len(dataset)
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=num_samples,
                replacement=bool(weighted_replacement),
                generator=generator,
            )
            loader_kwargs["sampler"] = sampler
            loader_kwargs["shuffle"] = False
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
            loader_kwargs["pin_memory"] = bool(pin_memory)
        else:
            loader_kwargs["pin_memory"] = False

        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            **loader_kwargs,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield batch


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *x: np.stack(np.asarray(x), axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
