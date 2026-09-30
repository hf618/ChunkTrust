from __future__ import annotations

import dataclasses
import gc
import json
import logging
import os
import pathlib
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

_PI05_HORIZON_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_PI05_HORIZON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PI05_HORIZON_ROOT))

import openpi.models.model as _model
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import online_teacher_workers as _teacher_workers
from openpi.training import weight_loaders as _weight_loaders
from heldout_protocol import TrainingAudit, sha256_bytes


@dataclasses.dataclass(frozen=True)
class TeacherCheckpointSpec:
    repo_id: str
    train_config_name: str
    checkpoint_root_tag: str
    model_name: str
    checkpoint_id: str
    pi0_step: int


def _as_entries(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("teachers"), list):
        return payload["teachers"]
    raise ValueError("Teacher manifest must be a JSON list or an object with a 'teachers' list.")


def load_teacher_manifest(path: str | pathlib.Path, *, default_pi0_step: int = 50) -> dict[str, TeacherCheckpointSpec]:
    manifest_path = pathlib.Path(path).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"teacher_checkpoint_manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        entries = _as_entries(json.load(f))

    specs: dict[str, TeacherCheckpointSpec] = {}
    for i, raw in enumerate(entries):
        if not isinstance(raw, dict):
            raise ValueError(f"Teacher manifest entry {i} must be an object, got {type(raw).__name__}.")
        repo_id = str(raw.get("repo_id", "")).strip()
        train_config_name = str(raw.get("train_config_name", "")).strip()
        model_name = str(raw.get("model_name", "")).strip()
        checkpoint_id = str(raw.get("checkpoint_id", "")).strip()
        checkpoint_root_tag = str(raw.get("checkpoint_root_tag", train_config_name)).strip()
        pi0_step = int(raw.get("pi0_step", default_pi0_step))
        missing = [
            name
            for name, value in (
                ("repo_id", repo_id),
                ("train_config_name", train_config_name),
                ("model_name", model_name),
                ("checkpoint_id", checkpoint_id),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"Teacher manifest entry {i} is missing required fields: {', '.join(missing)}")
        if repo_id in specs:
            raise ValueError(f"Duplicate teacher manifest repo_id: {repo_id}")
        specs[repo_id] = TeacherCheckpointSpec(
            repo_id=repo_id,
            train_config_name=train_config_name,
            checkpoint_root_tag=checkpoint_root_tag or train_config_name,
            model_name=model_name,
            checkpoint_id=checkpoint_id,
            pi0_step=pi0_step,
        )
    return specs


def _teacher_checkpoint_base_dir(config: _config.TrainConfig) -> str:
    return str(getattr(config, "teacher_checkpoint_base_dir", "") or config.checkpoint_base_dir)


def checkpoint_dir_for_spec(spec: TeacherCheckpointSpec, checkpoint_base_dir: str) -> pathlib.Path:
    return (
        pathlib.Path(checkpoint_base_dir)
        / (spec.checkpoint_root_tag or spec.train_config_name)
        / spec.model_name
        / str(spec.checkpoint_id)
    ).resolve()


def checkpoint_params_path_for_spec(spec: TeacherCheckpointSpec, checkpoint_base_dir: str) -> pathlib.Path:
    return checkpoint_dir_for_spec(spec, checkpoint_base_dir) / "params"


def maybe_apply_manifest_weight_loader(config: _config.TrainConfig) -> _config.TrainConfig:
    """Use the first routed teacher checkpoint as the train-state initializer by default."""
    mode = str(getattr(config, "teacher_runtime_mode", "")).strip().lower()
    manifest = str(getattr(config, "teacher_checkpoint_manifest", "")).strip()
    if mode != "online_per_task" or not manifest:
        return config
    if not isinstance(config.data, _config.MultiLeRobotAlohaDataConfig):
        return config

    current_loader = config.weight_loader
    should_auto_select = isinstance(current_loader, _weight_loaders.NoOpWeightLoader)
    if not should_auto_select:
        logging.info(
            "online_per_task keeping explicit weight_loader params_path=%s",
            getattr(current_loader, "params_path", type(current_loader).__name__),
        )
        return config

    repo_ids = config.data.normalized_repo_ids()
    specs = load_teacher_manifest(manifest, default_pi0_step=int(config.teacher_pi0_step_default))
    missing = [repo_id for repo_id in repo_ids if repo_id not in specs]
    if missing:
        raise ValueError("Teacher manifest is missing repo_ids: " + ", ".join(missing))
    init_spec = specs[repo_ids[0]]
    params_path = checkpoint_params_path_for_spec(init_spec, _teacher_checkpoint_base_dir(config))
    if not params_path.exists():
        raise FileNotFoundError(f"Manifest train-state initializer params not found: {params_path}")

    logging.info(
        "online_per_task auto-selected train-state initializer from manifest repo_id=%s params=%s",
        init_spec.repo_id,
        params_path,
    )
    return dataclasses.replace(
        config,
        weight_loader=_weight_loaders.CheckpointWeightLoader(str(params_path), missing_regex=r".*qha.*"),
    )


@dataclasses.dataclass
class _TeacherRuntime:
    spec: TeacherCheckpointSpec
    checkpoint_base_dir: str
    predict_fn: Any | None
    rng: jax.Array

    def _checkpoint_dir(self) -> pathlib.Path:
        return checkpoint_dir_for_spec(self.spec, self.checkpoint_base_dir)

    def _ensure_loaded(self) -> None:
        if self.predict_fn is not None:
            return
        checkpoint_dir = self._checkpoint_dir()
        params = _model.restore_params(checkpoint_dir / "params", restore_type=np.ndarray, dtype=jnp.bfloat16)
        teacher_config = dataclasses.replace(_config.get_config(self.spec.train_config_name))
        model_config = _checkpoints.resolve_model_config_for_checkpoint(
            teacher_config.model,
            checkpoint_dir,
            restored_params=params,
            qha_candidate_mode_override="dense_full",
        )
        if hasattr(model_config, "use_qha"):
            model_config = dataclasses.replace(model_config, use_qha=False, qha_candidate_mode="dense_full")
        model = model_config.load(params)
        model.eval()
        self.predict_fn = nnx_utils.module_jit(model.predict_qha_supervision_outputs)
        logging.info("Loaded routed pi05 QHA supervision backbone for repo_id=%s from %s", self.spec.repo_id, checkpoint_dir)

    def unload(self) -> None:
        if self.predict_fn is None:
            return
        self.predict_fn = None
        gc.collect()
        jax.clear_caches()
        logging.info("Unloaded routed pi05 QHA supervision backbone for repo_id=%s", self.spec.repo_id)

    def infer(self, observation: _model.Observation) -> dict[str, np.ndarray]:
        self._ensure_loaded()
        assert self.predict_fn is not None
        self.rng, call_rng = jax.random.split(self.rng)
        outputs = self.predict_fn(call_rng, observation)
        return {key: np.asarray(jax.device_get(value)) for key, value in outputs.items()}


class OnlineTeacherBank:
    """Host-side task-routed frozen pi05 checkpoint bank for joint QHA training."""

    def __init__(
        self,
        *,
        repo_ids: list[str],
        specs_by_repo_id: dict[str, TeacherCheckpointSpec],
        checkpoint_base_dir: str,
        student_candidate_horizons: tuple[int, ...],
        seed: int,
        max_loaded_teachers: int = 1,
        preload_on_startup: bool = False,
        audit: TrainingAudit | None = None,
    ):
        missing = [repo_id for repo_id in repo_ids if repo_id not in specs_by_repo_id]
        if missing:
            raise ValueError("Teacher manifest is missing repo_ids: " + ", ".join(missing))

        self._repo_ids = list(repo_ids)
        self._student_candidate_horizons = np.asarray(student_candidate_horizons, dtype=np.int32)
        self._max_loaded_teachers = max(1, int(max_loaded_teachers))
        self._preload_on_startup = bool(preload_on_startup)
        self._audit = audit
        self._lineage_batch_index = 0
        if self._preload_on_startup and self._max_loaded_teachers < len(self._repo_ids):
            raise ValueError(
                "teacher_preload_on_startup=True requires teacher_max_loaded >= number of repo_ids "
                f"({self._max_loaded_teachers} < {len(self._repo_ids)})."
            )
        self._loaded_lru: list[str] = []
        self._teachers: dict[str, _TeacherRuntime] = {}
        for task_index, repo_id in enumerate(self._repo_ids):
            spec = specs_by_repo_id[repo_id]
            params_path = checkpoint_params_path_for_spec(spec, checkpoint_base_dir)
            if not params_path.exists():
                raise FileNotFoundError(f"Teacher checkpoint params not found for {repo_id}: {params_path}")
            self._teachers[repo_id] = _TeacherRuntime(
                spec=spec,
                checkpoint_base_dir=checkpoint_base_dir,
                predict_fn=None,
                rng=jax.random.key(seed + task_index),
            )
        logging.info(
            "Initialized routed pi05 QHA supervision bank for %d repo_ids "
            "(max_loaded_teachers=%d, preload_on_startup=%s)",
            len(self._teachers),
            self._max_loaded_teachers,
            self._preload_on_startup,
        )
        if self._preload_on_startup:
            logging.warning(
                "teacher_preload_on_startup keeps routed pi05 checkpoints resident in a single process. "
                "This does not pin two teachers per GPU; placement is controlled by the JAX runtime."
            )
            self.preload_all()

    @classmethod
    def from_train_config(
        cls,
        config: _config.TrainConfig,
        *,
        audit: TrainingAudit | None = None,
    ) -> "OnlineTeacherBank | None":
        mode = str(getattr(config, "teacher_runtime_mode", "")).strip().lower()
        manifest = str(getattr(config, "teacher_checkpoint_manifest", "")).strip()
        if not mode and not manifest:
            return None
        if mode and not manifest:
            raise ValueError("teacher_checkpoint_manifest is required when teacher_runtime_mode is set.")
        if mode != "online_per_task":
            raise ValueError("teacher_runtime_mode must be online_per_task when teacher_checkpoint_manifest is set.")
        if not isinstance(config.data, _config.MultiLeRobotAlohaDataConfig):
            raise ValueError("online_per_task teacher runtime requires MultiLeRobotAlohaDataConfig.")
        if not bool(getattr(config.model, "use_qha", False)):
            raise ValueError("online_per_task teacher runtime requires config.model.use_qha=True.")
        if str(getattr(config.model, "qha_train_mode", "")).strip().lower() != "qha_only":
            raise ValueError("online_per_task teacher runtime is only supported for qha_only training.")
        if str(getattr(config.model, "qha_candidate_mode", "")).strip().lower() != "dense_full":
            raise ValueError("online_per_task teacher runtime v1 requires qha_candidate_mode=dense_full.")

        repo_ids = config.data.normalized_repo_ids()
        specs = load_teacher_manifest(manifest, default_pi0_step=int(config.teacher_pi0_step_default))
        student_candidate_horizons = tuple(range(1, int(getattr(config.model, "action_horizon", 50)) + 1))
        backend = str(getattr(config, "teacher_execution_backend", "local")).strip().lower()
        logging.info(
            "online_per_task enabled: routed pi05 checkpoints provide frozen QHA features and teacher labels; "
            "weight_loader only initializes shared train-state structure. backend=%s",
            backend,
        )
        if backend == "gpu_workers":
            return GpuWorkerTeacherBank(
                repo_ids=repo_ids,
                specs_by_repo_id=specs,
                checkpoint_base_dir=_teacher_checkpoint_base_dir(config),
                student_candidate_horizons=student_candidate_horizons,
                worker_devices=_resolve_worker_devices(str(getattr(config, "teacher_worker_devices", ""))),
                startup_timeout_s=int(getattr(config, "teacher_worker_startup_timeout_s", 900)),
                request_timeout_s=int(getattr(config, "teacher_worker_request_timeout_s", 600)),
                audit=audit,
            )
        if backend != "local":
            raise ValueError(f"Unsupported teacher_execution_backend={backend!r}.")
        return cls(
            repo_ids=repo_ids,
            specs_by_repo_id=specs,
            checkpoint_base_dir=_teacher_checkpoint_base_dir(config),
            student_candidate_horizons=student_candidate_horizons,
            seed=int(config.seed),
            max_loaded_teachers=int(getattr(config, "teacher_max_loaded", 1)),
            preload_on_startup=bool(getattr(config, "teacher_preload_on_startup", False)),
            audit=audit,
        )

    def preload_all(self) -> None:
        wall_start = time.perf_counter()
        for repo_id in self._repo_ids:
            teacher = self._teachers[repo_id]
            load_start = time.perf_counter()
            teacher._ensure_loaded()
            if repo_id in self._loaded_lru:
                self._loaded_lru.remove(repo_id)
            self._loaded_lru.append(repo_id)
            logging.info(
                "Preloaded routed pi05 QHA supervision backbone repo_id=%s checkpoint=%s load_wall_s=%.3f",
                repo_id,
                teacher._checkpoint_dir(),
                time.perf_counter() - load_start,
            )
        logging.info(
            "Preloaded %d routed pi05 QHA supervision backbones in %.3fs",
            len(self._loaded_lru),
            time.perf_counter() - wall_start,
        )

    def _loaded_teacher(self, repo_id: str) -> _TeacherRuntime:
        teacher = self._teachers[repo_id]
        if teacher.predict_fn is None:
            if self._preload_on_startup:
                raise RuntimeError(
                    f"Resident teacher bank expected repo_id={repo_id} to remain loaded, but it is unloaded."
                )
            while len(self._loaded_lru) >= self._max_loaded_teachers:
                evict_repo_id = self._loaded_lru.pop(0)
                if evict_repo_id != repo_id:
                    self._teachers[evict_repo_id].unload()
            teacher._ensure_loaded()
        if repo_id in self._loaded_lru:
            self._loaded_lru.remove(repo_id)
        self._loaded_lru.append(repo_id)
        return teacher

    def attach_to_batch(self, batch: Any) -> tuple[Any, dict[str, float]]:
        observation, actions = batch[:2]
        if observation.task_index is None:
            raise ValueError("online_per_task teacher runtime requires observation.task_index in each batch.")

        wall_start = time.perf_counter()
        task_indices = np.asarray(jax.device_get(observation.task_index), dtype=np.int32).reshape(-1)
        batch_size = int(task_indices.shape[0])
        context_tokens: np.ndarray | None = None
        context_mask: np.ndarray | None = None
        action_latents: np.ndarray | None = None
        posterior: np.ndarray | None = None
        expected_horizon = np.zeros((batch_size,), dtype=np.float32)
        argmax_horizon = np.zeros((batch_size,), dtype=np.int32)
        num_groups = 0
        lineage_groups: list[dict[str, Any]] = []

        unique_task_indices = [int(task_index) for task_index in np.unique(task_indices)]
        loaded_first = [
            task_index
            for task_index in unique_task_indices
            if 0 <= task_index < len(self._repo_ids) and self._repo_ids[task_index] in self._loaded_lru
        ]
        unloaded = [task_index for task_index in unique_task_indices if task_index not in loaded_first]

        for task_index in loaded_first + unloaded:
            repo_index = int(task_index)
            if repo_index < 0 or repo_index >= len(self._repo_ids):
                raise ValueError(f"Batch task_index {repo_index} is outside configured repo range.")
            repo_id = self._repo_ids[repo_index]
            positions = np.nonzero(task_indices == repo_index)[0]
            outputs = self._loaded_teacher(repo_id).infer(_slice_observation(observation, positions))
            horizons = np.asarray(outputs["qha_candidate_horizons"], dtype=np.int32).reshape(-1)
            if not np.array_equal(horizons, self._student_candidate_horizons):
                raise ValueError(
                    f"Teacher horizons for {repo_id} do not match student dense_full horizons: "
                    f"teacher={horizons.tolist()} student={self._student_candidate_horizons.tolist()}"
                )

            group_context_tokens = np.asarray(outputs["qha_context_tokens"], dtype=np.float32)
            group_context_mask = np.asarray(outputs["qha_context_mask"], dtype=np.bool_)
            group_action_latents = np.asarray(outputs["qha_action_latents"], dtype=np.float32)
            if context_tokens is None:
                context_tokens = np.zeros((batch_size, *group_context_tokens.shape[1:]), dtype=group_context_tokens.dtype)
                context_mask = np.zeros((batch_size, *group_context_mask.shape[1:]), dtype=group_context_mask.dtype)
                action_latents = np.zeros((batch_size, *group_action_latents.shape[1:]), dtype=group_action_latents.dtype)
            if group_context_tokens.shape[1:] != context_tokens.shape[1:]:
                raise ValueError(f"QHA context feature shape mismatch for {repo_id}.")
            if group_context_mask.shape[1:] != context_mask.shape[1:]:
                raise ValueError(f"QHA context mask shape mismatch for {repo_id}.")
            if group_action_latents.shape[1:] != action_latents.shape[1:]:
                raise ValueError(f"QHA action latent shape mismatch for {repo_id}.")
            context_tokens[positions] = group_context_tokens
            context_mask[positions] = group_context_mask
            action_latents[positions] = group_action_latents

            group_posterior = np.asarray(outputs["qha_teacher_posterior"], dtype=np.float32)
            if posterior is None:
                posterior = np.zeros((batch_size, group_posterior.shape[1]), dtype=np.float32)
            if group_posterior.shape[1] != posterior.shape[1]:
                raise ValueError(f"Teacher posterior width mismatch for {repo_id}.")
            posterior[positions] = group_posterior
            expected_horizon[positions] = np.asarray(outputs["qha_teacher_expected_horizon"], dtype=np.float32).reshape(-1)
            argmax_horizon[positions] = np.asarray(outputs["qha_teacher_argmax_horizon"], dtype=np.int32).reshape(-1)
            lineage_groups.append(
                {
                    "repo_id": repo_id,
                    "sample_count": int(len(positions)),
                    "candidate_horizons": horizons.astype(int).tolist(),
                    "teacher_posterior_sha256": sha256_bytes(np.ascontiguousarray(group_posterior).tobytes()),
                }
            )
            num_groups += 1

        if posterior is None or context_tokens is None or context_mask is None or action_latents is None:
            raise ValueError("Cannot attach teacher outputs for an empty batch.")
        if self._audit is not None:
            self._audit.record_teacher_batch(batch_index=self._lineage_batch_index, groups=lineage_groups)
        self._lineage_batch_index += 1

        qha_features = {
            "qha_context_tokens": context_tokens,
            "qha_context_mask": context_mask,
            "qha_action_latents": action_latents,
        }
        teacher_targets = {
            "qha_teacher_posterior": posterior,
            "qha_teacher_expected_horizon": expected_horizon,
            "qha_teacher_argmax_horizon": argmax_horizon,
        }
        return (observation, actions, qha_features, teacher_targets), {
            "teacher_bank_wall_s": time.perf_counter() - wall_start,
            "teacher_bank_num_groups": float(num_groups),
            "teacher_bank_loaded": float(len(self._loaded_lru)),
            "teacher_bank_preloaded": float(self._preload_on_startup),
        }

    def close(self) -> None:
        return None


def _resolve_worker_devices(raw_devices: str) -> list[str]:
    raw_devices = raw_devices.strip()
    if raw_devices:
        devices = [part.strip() for part in raw_devices.split(",") if part.strip()]
    else:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible_devices:
            devices = [part.strip() for part in visible_devices.split(",") if part.strip()]
        else:
            devices = [str(i) for i in range(jax.device_count())]
    if not devices:
        raise ValueError("teacher_execution_backend=gpu_workers requires at least one worker device.")
    return devices


class GpuWorkerTeacherBank:
    """Task-routed teacher bank backed by one subprocess per GPU worker."""

    def __init__(
        self,
        *,
        repo_ids: list[str],
        specs_by_repo_id: dict[str, TeacherCheckpointSpec],
        checkpoint_base_dir: str,
        student_candidate_horizons: tuple[int, ...],
        worker_devices: list[str],
        startup_timeout_s: int,
        request_timeout_s: int,
        audit: TrainingAudit | None = None,
    ):
        missing = [repo_id for repo_id in repo_ids if repo_id not in specs_by_repo_id]
        if missing:
            raise ValueError("Teacher manifest is missing repo_ids: " + ", ".join(missing))
        if not worker_devices:
            raise ValueError("teacher_execution_backend=gpu_workers requires non-empty worker_devices.")

        self._repo_ids = list(repo_ids)
        self._repo_to_worker_id: dict[str, int] = {}
        worker_repo_ids: list[list[str]] = [[] for _ in worker_devices]
        for index, repo_id in enumerate(self._repo_ids):
            worker_id = index % len(worker_devices)
            self._repo_to_worker_id[repo_id] = worker_id
            worker_repo_ids[worker_id].append(repo_id)

        specs_payload = {repo_id: dataclasses.asdict(specs_by_repo_id[repo_id]) for repo_id in self._repo_ids}
        for repo_id, spec in specs_by_repo_id.items():
            if repo_id not in specs_payload:
                continue
            params_path = checkpoint_params_path_for_spec(spec, checkpoint_base_dir)
            if not params_path.exists():
                raise FileNotFoundError(f"Teacher checkpoint params not found for {repo_id}: {params_path}")

        worker_specs = [
            _teacher_workers.WorkerSpec(
                worker_id=worker_id,
                device_id=device_id,
                repo_ids=tuple(assigned_repo_ids),
            )
            for worker_id, (device_id, assigned_repo_ids) in enumerate(zip(worker_devices, worker_repo_ids, strict=True))
            if assigned_repo_ids
        ]
        self._student_candidate_horizons = np.asarray(student_candidate_horizons, dtype=np.int32)
        self._audit = audit
        self._lineage_batch_index = 0
        self._num_workers = len(worker_specs)
        self._pool = _teacher_workers.GpuTeacherWorkerPool(
            worker_specs=worker_specs,
            specs_by_repo_id=specs_payload,
            checkpoint_base_dir=checkpoint_base_dir,
            startup_timeout_s=startup_timeout_s,
            request_timeout_s=request_timeout_s,
        )
        logging.info(
            "Initialized gpu_workers routed pi05 QHA supervision bank: workers=%d mapping=%s",
            self._num_workers,
            {spec.worker_id: list(spec.repo_ids) for spec in worker_specs},
        )

    def attach_to_batch(self, batch: Any) -> tuple[Any, dict[str, float]]:
        observation, actions = batch[:2]
        if observation.task_index is None:
            raise ValueError("online_per_task gpu_workers runtime requires observation.task_index in each batch.")

        wall_start = time.perf_counter()
        task_indices = np.asarray(jax.device_get(observation.task_index), dtype=np.int32).reshape(-1)
        batch_size = int(task_indices.shape[0])
        worker_groups: dict[int, list[dict[str, Any]]] = {}

        for task_index in np.unique(task_indices):
            repo_index = int(task_index)
            if repo_index < 0 or repo_index >= len(self._repo_ids):
                raise ValueError(f"Batch task_index {repo_index} is outside configured repo range.")
            repo_id = self._repo_ids[repo_index]
            positions = np.nonzero(task_indices == repo_index)[0]
            worker_id = self._repo_to_worker_id[repo_id]
            worker_groups.setdefault(worker_id, []).append(
                {
                    "repo_id": repo_id,
                    "positions": positions,
                    "observation": _slice_observation(observation, positions),
                }
            )

        results, pool_metrics = self._pool.infer(worker_groups)
        context_tokens: np.ndarray | None = None
        context_mask: np.ndarray | None = None
        action_latents: np.ndarray | None = None
        posterior: np.ndarray | None = None
        expected_horizon = np.zeros((batch_size,), dtype=np.float32)
        argmax_horizon = np.zeros((batch_size,), dtype=np.int32)
        num_groups = 0
        lineage_groups: list[dict[str, Any]] = []

        for result in results:
            repo_id = str(result["repo_id"])
            positions = np.asarray(result["positions"], dtype=np.int64)
            outputs = result["outputs"]
            horizons = np.asarray(outputs["qha_candidate_horizons"], dtype=np.int32).reshape(-1)
            if not np.array_equal(horizons, self._student_candidate_horizons):
                raise ValueError(
                    f"Teacher horizons for {repo_id} do not match student dense_full horizons: "
                    f"teacher={horizons.tolist()} student={self._student_candidate_horizons.tolist()}"
                )

            group_context_tokens = np.asarray(outputs["qha_context_tokens"], dtype=np.float32)
            group_context_mask = np.asarray(outputs["qha_context_mask"], dtype=np.bool_)
            group_action_latents = np.asarray(outputs["qha_action_latents"], dtype=np.float32)
            if context_tokens is None:
                context_tokens = np.zeros((batch_size, *group_context_tokens.shape[1:]), dtype=group_context_tokens.dtype)
                context_mask = np.zeros((batch_size, *group_context_mask.shape[1:]), dtype=group_context_mask.dtype)
                action_latents = np.zeros((batch_size, *group_action_latents.shape[1:]), dtype=group_action_latents.dtype)
            if group_context_tokens.shape[1:] != context_tokens.shape[1:]:
                raise ValueError(f"QHA context feature shape mismatch for {repo_id}.")
            if group_context_mask.shape[1:] != context_mask.shape[1:]:
                raise ValueError(f"QHA context mask shape mismatch for {repo_id}.")
            if group_action_latents.shape[1:] != action_latents.shape[1:]:
                raise ValueError(f"QHA action latent shape mismatch for {repo_id}.")
            context_tokens[positions] = group_context_tokens
            context_mask[positions] = group_context_mask
            action_latents[positions] = group_action_latents

            group_posterior = np.asarray(outputs["qha_teacher_posterior"], dtype=np.float32)
            if posterior is None:
                posterior = np.zeros((batch_size, group_posterior.shape[1]), dtype=np.float32)
            if group_posterior.shape[1] != posterior.shape[1]:
                raise ValueError(f"Teacher posterior width mismatch for {repo_id}.")
            posterior[positions] = group_posterior
            expected_horizon[positions] = np.asarray(outputs["qha_teacher_expected_horizon"], dtype=np.float32).reshape(-1)
            argmax_horizon[positions] = np.asarray(outputs["qha_teacher_argmax_horizon"], dtype=np.int32).reshape(-1)
            lineage_groups.append(
                {
                    "repo_id": repo_id,
                    "sample_count": int(len(positions)),
                    "candidate_horizons": horizons.astype(int).tolist(),
                    "teacher_posterior_sha256": sha256_bytes(np.ascontiguousarray(group_posterior).tobytes()),
                }
            )
            num_groups += 1

        if posterior is None or context_tokens is None or context_mask is None or action_latents is None:
            raise ValueError("Cannot attach gpu_workers teacher outputs for an empty batch.")
        if self._audit is not None:
            self._audit.record_teacher_batch(batch_index=self._lineage_batch_index, groups=lineage_groups)
        self._lineage_batch_index += 1

        wall_s = time.perf_counter() - wall_start
        qha_features = {
            "qha_context_tokens": context_tokens,
            "qha_context_mask": context_mask,
            "qha_action_latents": action_latents,
        }
        teacher_targets = {
            "qha_teacher_posterior": posterior,
            "qha_teacher_expected_horizon": expected_horizon,
            "qha_teacher_argmax_horizon": argmax_horizon,
        }
        return (observation, actions, qha_features, teacher_targets), {
            "teacher_bank_wall_s": wall_s,
            "teacher_bank_num_groups": float(num_groups),
            "teacher_bank_backend_gpu_workers": 1.0,
            "teacher_worker_num_workers": float(self._num_workers),
            "teacher_worker_num_groups": float(num_groups),
            "teacher_worker_wall_s": wall_s,
            "teacher_worker_max_request_wall_s": float(pool_metrics.get("teacher_worker_max_request_wall_s", 0.0)),
        }

    def close(self) -> None:
        self._pool.close()


def _slice_observation(observation: _model.Observation, indices: np.ndarray) -> _model.Observation:
    def _slice_leaf(value):
        if value is None:
            return None
        arr = np.asarray(jax.device_get(value))
        return arr[indices]

    return jax.tree.map(_slice_leaf, observation)
