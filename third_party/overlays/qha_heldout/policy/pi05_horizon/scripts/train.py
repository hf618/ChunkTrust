from __future__ import annotations

import atexit
import dataclasses
import functools
import logging
import pathlib
import platform
import queue
import sys
import threading
import time
from typing import Any

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.profiler
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.online_teachers as _online_teachers
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
from heldout_protocol import TrainingAudit


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {
        "DEBUG": "D",
        "INFO": "I",
        "WARNING": "W",
        "ERROR": "E",
        "CRITICAL": "C",
    }

    class CustomFormatter(logging.Formatter):

        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(
    config: _config.TrainConfig,
    *,
    resuming: bool,
    log_code: bool = False,
    enabled: bool = True,
):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict({
        k: v
        for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)
    })


@at.typecheck
def init_train_state(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(
            params,
            config.freeze_filter,
            lambda p: p.replace(p.value.astype(jnp.bfloat16)),
        )

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1, ),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


def _split_train_batch(batch: Any) -> tuple[Any, Any, dict[str, at.Array] | None, dict[str, at.Array] | None]:
    if len(batch) == 2:
        observation, actions = batch
        return observation, actions, None, None
    if len(batch) == 3:
        observation, actions, external_qha_teacher = batch
        return observation, actions, None, external_qha_teacher
    if len(batch) == 4:
        observation, actions, external_qha_features, external_qha_teacher = batch
        return observation, actions, external_qha_features, external_qha_teacher
    raise ValueError(f"Unsupported train batch arity: expected 2, 3, or 4 items, got {len(batch)}.")


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: Any,
    *,
    compute_horizon_alignment_metrics: bool = True,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        external_qha_features: dict[str, at.Array] | None,
        external_qha_teacher: dict[str, at.Array] | None,
    ):
        if hasattr(model, "compute_loss_and_metrics"):
            chunked_loss, extra_metrics = model.compute_loss_and_metrics(
                rng,
                observation,
                actions,
                train=True,
                compute_qha_alignment_metrics=compute_horizon_alignment_metrics,
                external_qha_features=external_qha_features,
                external_qha_teacher=external_qha_teacher,
            )
        else:
            chunked_loss = model.compute_loss(rng, observation, actions, train=True)
            extra_metrics = {"total_loss": jnp.mean(chunked_loss, axis=-1)}
        return jnp.mean(chunked_loss), extra_metrics

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions, external_qha_features, external_qha_teacher = _split_train_batch(batch)

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, extra_metrics), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model,
        train_rng,
        observation,
        actions,
        external_qha_features,
        external_qha_teacher,
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        def _ema_merge_leaf(old, new):
            # NNX state may include non-parameter leaves (e.g., RNG keys from dropout).
            # Apply EMA only to floating-point tensors; keep other leaves from the new state.
            if hasattr(old, "dtype") and jnp.issubdtype(old.dtype, jnp.inexact):
                return state.ema_decay * old + (1 - state.ema_decay) * new
            return new

        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                _ema_merge_leaf,
                state.ema_params,
                new_params,
            ),
        )

    info = {
        "grad_norm": optax.global_norm(grads),
    }
    horizon_filter = nnx_utils.PathRegex(".*qha.*")
    if hasattr(grads, "filter"):
        info["qha_grad_norm"] = optax.global_norm(grads.filter(horizon_filter))
        info["backbone_grad_norm"] = optax.global_norm(grads.filter(nnx.Not(horizon_filter)))
    else:
        info["qha_grad_norm"] = jnp.zeros((), dtype=jnp.float32)
        info["backbone_grad_norm"] = jnp.zeros((), dtype=jnp.float32)
    update_norm = optax.global_norm(updates)
    trainable_param_norm = optax.global_norm(params)
    info["update_norm"] = update_norm
    info["update_over_param_norm"] = update_norm / jnp.maximum(trainable_param_norm, 1e-12)
    info["lr"] = config.lr_schedule.create()(state.step)
    for key, value in extra_metrics.items():
        if key in _QHA_STD_LOG_KEYS:
            info[f"{key}_std"] = jnp.nanstd(jnp.asarray(value, dtype=jnp.float32))
        info[key] = jnp.mean(value)
    if "total_loss" not in info:
        info["total_loss"] = loss
    info["horizon_metrics_full_step"] = jnp.asarray(
        1.0 if compute_horizon_alignment_metrics else 0.0, dtype=jnp.float32)
    return new_state, info


@at.typecheck
def compute_horizon_diagnostics(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: Any,
) -> dict[str, at.Array]:
    model = nnx.merge(state.model_def, state.params)
    model.train()
    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions, external_qha_features, external_qha_teacher = _split_train_batch(batch)

    if hasattr(model, "compute_loss_and_metrics"):
        _, metrics = model.compute_loss_and_metrics(
            train_rng,
            observation,
            actions,
            train=True,
            compute_qha_alignment_metrics=True,
            external_qha_features=external_qha_features,
            external_qha_teacher=external_qha_teacher,
        )
    else:
        metrics = {}

    diagnostic_keys = (
        "qha_expected_horizon",
        "qha_teacher_expected_horizon",
        "qha_expected_horizon_gap_abs",
        "qha_entropy",
        "qha_teacher_entropy",
        "qha_entropy_gap_abs",
        "qha_argmax_match_rate",
        "qha_vs_batch_prior_ce_gain",
    )
    scalar_diagnostic_keys = (
        "qha_expected_horizon_corr",
        "qha_expected_horizon_std_ratio",
    )
    reduced_metrics = {}
    for key in diagnostic_keys:
        if key not in metrics:
            continue
        reduced_metrics[key] = jnp.mean(metrics[key])
        reduced_metrics[f"{key}_std"] = jnp.nanstd(jnp.asarray(metrics[key], dtype=jnp.float32))
    for key in scalar_diagnostic_keys:
        if key not in metrics:
            continue
        reduced_metrics[key] = jnp.mean(metrics[key])
    reduced_metrics["horizon_metrics_full_step"] = jnp.asarray(1.0, dtype=jnp.float32)
    return reduced_metrics


@at.typecheck
def compute_param_norm(state: training_utils.TrainState) -> at.Array:
    """Computes parameter norm for logging only (to avoid per-step overhead)."""
    model = nnx.merge(state.model_def, state.params)
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    return optax.global_norm(kernel_params)


_QHA_STD_LOG_KEYS = (
    "qha_loss",
    "qha_expected_horizon",
    "qha_teacher_expected_horizon",
    "qha_expected_horizon_gap_abs",
    "qha_entropy",
    "qha_teacher_entropy",
    "qha_entropy_gap_abs",
    "qha_argmax_match_rate",
    "qha_vs_batch_prior_ce_gain",
)


def _prefetch_iterator(iterator, prefetch_size: int):
    """Prefetch batches on a background thread to reduce host-side input stalls."""
    if prefetch_size <= 0:
        yield from iterator
        return

    q: queue.Queue[Any] = queue.Queue(maxsize=prefetch_size)
    sentinel = object()
    errors: list[BaseException] = []

    def _producer():
        try:
            for item in iterator:
                q.put(item)
        except BaseException as exc:  # surface producer errors in the consumer thread
            errors.append(exc)
        finally:
            q.put(sentinel)

    threading.Thread(target=_producer, daemon=True).start()

    while True:
        item = q.get()
        if item is sentinel:
            break
        yield item

    if errors:
        raise errors[0]


def _device_put_batch(batch, sharding_spec):
    try:
        return jax.device_put(batch, sharding_spec)
    except Exception:
        return jax.tree.map(lambda x: jax.make_array_from_process_local_data(sharding_spec, x), batch)


def _host_prefetch_iterator(iterator, prefetch_size: int):
    """Prefetch host batches and record pure iterator wait time."""
    if prefetch_size <= 0:
        while True:
            data_next_wall_start = time.perf_counter()
            item = next(iterator)
            data_next_wall_s = time.perf_counter() - data_next_wall_start
            yield item, data_next_wall_s

    q: queue.Queue[Any] = queue.Queue(maxsize=prefetch_size)
    sentinel = object()
    errors: list[BaseException] = []

    def _producer():
        try:
            while True:
                data_next_wall_start = time.perf_counter()
                item = next(iterator)
                data_next_wall_s = time.perf_counter() - data_next_wall_start
                q.put((item, data_next_wall_s))
        except StopIteration:
            pass
        except BaseException as exc:  # surface producer errors in the consumer thread
            errors.append(exc)
        finally:
            q.put(sentinel)

    threading.Thread(target=_producer, daemon=True).start()

    while True:
        item = q.get()
        if item is sentinel:
            break
        yield item

    if errors:
        raise errors[0]


def _device_stage_iterator(iterator, sharding_spec, prefetch_size: int):
    """Stage host batches to devices on a background thread and record device_put time."""
    if prefetch_size <= 0:
        for (batch, batch_metrics), data_next_wall_s in iterator:
            device_put_wall_start = time.perf_counter()
            batch = _device_put_batch(batch, sharding_spec)
            device_put_wall_s = time.perf_counter() - device_put_wall_start
            yield batch, batch_metrics, data_next_wall_s, device_put_wall_s
        return

    q: queue.Queue[Any] = queue.Queue(maxsize=prefetch_size)
    sentinel = object()
    errors: list[BaseException] = []

    def _producer():
        try:
            for (batch, batch_metrics), data_next_wall_s in iterator:
                device_put_wall_start = time.perf_counter()
                staged_batch = _device_put_batch(batch, sharding_spec)
                device_put_wall_s = time.perf_counter() - device_put_wall_start
                q.put((staged_batch, batch_metrics, data_next_wall_s, device_put_wall_s))
        except BaseException as exc:  # surface producer errors in the consumer thread
            errors.append(exc)
        finally:
            q.put(sentinel)

    threading.Thread(target=_producer, daemon=True).start()

    while True:
        item = q.get()
        if item is sentinel:
            break
        yield item

    if errors:
        raise errors[0]


def _online_teacher_iterator(iterator, teacher_bank: _online_teachers.OnlineTeacherBank | None):
    if teacher_bank is None:
        yield from iterator
        return

    for (batch, batch_metrics), data_next_wall_s in iterator:
        batch, teacher_metrics = teacher_bank.attach_to_batch(batch)
        merged_metrics = dict(batch_metrics)
        merged_metrics.update(teacher_metrics)
        merged_metrics["teacher_prefetch_enabled"] = 0.0
        merged_metrics["teacher_prefetch_wait_wall_s"] = 0.0
        merged_metrics["teacher_prefetch_queue_size"] = 0.0
        yield (batch, merged_metrics), data_next_wall_s


def _async_online_teacher_iterator(
    iterator,
    teacher_bank: _online_teachers.OnlineTeacherBank | None,
    prefetch_size: int,
):
    if teacher_bank is None or prefetch_size <= 0:
        yield from _online_teacher_iterator(iterator, teacher_bank)
        return

    q: queue.Queue[Any] = queue.Queue(maxsize=max(1, int(prefetch_size)))
    sentinel = object()
    errors: list[BaseException] = []

    def _producer():
        try:
            for (batch, batch_metrics), data_next_wall_s in iterator:
                batch, teacher_metrics = teacher_bank.attach_to_batch(batch)
                merged_metrics = dict(batch_metrics)
                merged_metrics.update(teacher_metrics)
                merged_metrics["teacher_prefetch_enabled"] = 1.0
                q.put(((batch, merged_metrics), data_next_wall_s))
        except BaseException as exc:
            errors.append(exc)
        finally:
            q.put(sentinel)

    threading.Thread(target=_producer, daemon=True).start()

    while True:
        wait_start = time.perf_counter()
        item = q.get()
        teacher_prefetch_wait_wall_s = time.perf_counter() - wait_start
        if item is sentinel:
            break
        (batch, batch_metrics), data_next_wall_s = item
        batch_metrics = dict(batch_metrics)
        batch_metrics["teacher_prefetch_wait_wall_s"] = teacher_prefetch_wait_wall_s
        batch_metrics["teacher_prefetch_queue_size"] = float(q.qsize())
        yield (batch, batch_metrics), data_next_wall_s

    if errors:
        raise errors[0]


def _canonicalize_logged_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Convert scalar metrics to a stable, sorted float dict for both terminal logs and W&B."""
    canonical: dict[str, float] = {}
    for key in sorted(metrics):
        value = metrics[key]
        arr = np.asarray(value)
        if arr.shape != ():
            continue
        canonical[key] = float(arr.item())
    return canonical


def _compile_step_functions(
    train_rng: at.KeyArrayLike,
    train_state: training_utils.TrainState,
    batch: Any,
    ptrain_step,
    pparam_norm,
) -> None:
    ptrain_step.lower(train_rng, train_state, batch).compile()
    pparam_norm.lower(train_state).compile()


def _save_training_checkpoint(
    config: _config.TrainConfig,
    checkpoint_manager,
    train_state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
    audit: TrainingAudit | None = None,
) -> None:
    if getattr(config, "save_qha_only_checkpoints", False):
        _checkpoints.save_qha_only_params(
            train_state,
            config.checkpoint_dir / str(step),
            model_config=config.model,
        )
        if audit is not None:
            audit.record_checkpoint(config.checkpoint_dir / str(step), step=step)
        return

    _checkpoints.save_state(
        checkpoint_manager,
        train_state,
        data_loader,
        step,
        model_config=config.model,
    )


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    # Rebuild freeze filter from the current model config so CLI model overrides
    # (e.g., enabling horizon-head-only training) are respected.
    if hasattr(config.model, "get_freeze_filter"):
        config = dataclasses.replace(config, freeze_filter=config.model.get_freeze_filter())
    config = _online_teachers.maybe_apply_manifest_weight_loader(config)

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}.")

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    if getattr(config, "save_qha_only_checkpoints", False) and resuming:
        raise ValueError("Resuming is not supported when save_qha_only_checkpoints=True.")
    training_audit = TrainingAudit.create(config)
    if training_audit is not None:
        logging.info("Held-out training audit enabled at %s", training_audit.audit_dir)
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        num_workers=config.num_workers,
        shuffle=True,
    )
    sampling_metrics = data_loader.sampling_metrics() if hasattr(data_loader, "sampling_metrics") else {}
    host_iter = _host_prefetch_iterator(iter(data_loader), config.host_prefetch)
    teacher_bank = _online_teachers.OnlineTeacherBank.from_train_config(config, audit=training_audit)
    if teacher_bank is not None and hasattr(teacher_bank, "close"):
        atexit.register(teacher_bank.close)
    teacher_prefetch = int(getattr(config, "teacher_prefetch", 0))
    if teacher_bank is not None and teacher_prefetch > 0:
        logging.info("Online teacher async prefetch enabled with depth=%d.", teacher_prefetch)
        host_iter = _async_online_teacher_iterator(host_iter, teacher_bank, teacher_prefetch)
    else:
        host_iter = _online_teacher_iterator(host_iter, teacher_bank)
    device_prefetch = 0 if teacher_bank is not None else config.device_prefetch
    if teacher_bank is not None and config.device_prefetch > 0:
        logging.info("Online teacher runtime enabled; disabling device prefetch for synchronous teacher inference.")
    data_iter = _device_stage_iterator(host_iter, data_sharding, device_prefetch)
    batch, batch_metrics, batch_data_next_wall_s, batch_device_put_wall_s = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    horizon_metrics_interval = max(1, int(getattr(config, "horizon_metrics_interval", 1)))
    ptrain_step = jax.jit(
        functools.partial(train_step, config, compute_horizon_alignment_metrics=False),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1, ),
    )
    pcompute_horizon_diagnostics = jax.jit(
        functools.partial(compute_horizon_diagnostics, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    param_norm_interval = int(getattr(config, "param_norm_interval", 0))
    if param_norm_interval <= 0:
        param_norm_interval = max(1, int(config.log_interval))
    pparam_norm = jax.jit(
        compute_param_norm,
        in_shardings=(train_state_sharding, ),
        out_shardings=replicated_sharding,
    )

    if config.eager_compile_step_fns:
        logging.info("Eager-compiling step functions before entering the training loop")
        with sharding.set_mesh(mesh):
            _compile_step_functions(train_rng, train_state, batch, ptrain_step, pparam_norm)

    start_step = int(train_state.step)
    if sampling_metrics:
        sampling_metrics = _canonicalize_logged_metrics(sampling_metrics)
        sampling_metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in sampling_metrics.items())
        logging.info(f"Initialized weighted sampler metrics: {sampling_metrics_str}")
        wandb.log(sampling_metrics, step=start_step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    last_param_norm: float | None = None
    profile_dir = str(config.profile_dir).strip()
    profile_enabled = bool(profile_dir) and int(config.profile_num_steps) > 0
    profile_start_step = int(config.profile_start_step)
    profile_end_step = profile_start_step + int(config.profile_num_steps)
    profiler_active = False
    def _reduce_metric(x):
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.inexact):
            mean_val = jnp.nanmean(x)
            return jnp.where(jnp.isnan(mean_val), jnp.zeros_like(mean_val), mean_val)
        return jnp.mean(x)
    for step in pbar:
        step_wall_start = time.perf_counter()
        run_horizon_diagnostics = step > start_step and (step % horizon_metrics_interval == 0)
        if profile_enabled and (not profiler_active) and step == profile_start_step:
            epath.Path(profile_dir).mkdir(parents=True, exist_ok=True)
            jax.profiler.start_trace(profile_dir)
            profiler_active = True
        train_step_wall_start = time.perf_counter()
        with jax.profiler.StepTraceAnnotation("train", step_num=step):
            with sharding.set_mesh(mesh):
                train_state, info = ptrain_step(train_rng, train_state, batch)
        train_dispatch_wall_s = time.perf_counter() - train_step_wall_start
        info = dict(info)
        info["train_step_wall_s"] = train_dispatch_wall_s
        info["train_dispatch_wall_s"] = train_dispatch_wall_s
        info["train_sync_wall_s"] = 0.0
        info["data_next_wall_s"] = batch_data_next_wall_s
        info["device_put_wall_s"] = batch_device_put_wall_s
        info["step_sync_wall_s"] = 0.0
        info.update(batch_metrics)
        if run_horizon_diagnostics:
            diagnostics_wall_start = time.perf_counter()
            with jax.profiler.StepTraceAnnotation("horizon_diagnostics", step_num=step):
                with sharding.set_mesh(mesh):
                    diagnostics_info = pcompute_horizon_diagnostics(train_rng, train_state, batch)
            diagnostics_dispatch_wall_s = time.perf_counter() - diagnostics_wall_start
            diagnostics_sync_wall_start = time.perf_counter()
            jax.block_until_ready(diagnostics_info)
            diagnostics_sync_wall_s = time.perf_counter() - diagnostics_sync_wall_start
            info["horizon_diagnostics_dispatch_wall_s"] = diagnostics_dispatch_wall_s
            info["horizon_diagnostics_sync_wall_s"] = diagnostics_sync_wall_s
            info.update(diagnostics_info)
        else:
            info["horizon_diagnostics_dispatch_wall_s"] = 0.0
            info["horizon_diagnostics_sync_wall_s"] = 0.0
        input_wait_wall_start = time.perf_counter()
        batch, batch_metrics, batch_data_next_wall_s, batch_device_put_wall_s = next(data_iter)
        input_wait_wall_s = time.perf_counter() - input_wait_wall_start

        checkpoint_wall_s = 0.0
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            checkpoint_wall_start = time.perf_counter()
            target_step = step + 1 if step == config.num_train_steps - 1 else step
            _save_training_checkpoint(
                config,
                checkpoint_manager,
                train_state,
                data_loader,
                target_step,
                audit=training_audit,
            )
            checkpoint_wall_s = time.perf_counter() - checkpoint_wall_start

        step_wall_s = time.perf_counter() - step_wall_start
        info["input_wait_wall_s"] = input_wait_wall_s
        info["checkpoint_wall_s"] = checkpoint_wall_s
        info["step_wall_s"] = step_wall_s
        infos.append(info)

        if step % config.log_interval == 0:
            logging_wall_start = time.perf_counter()
            stacked_infos = common_utils.stack_forest(infos)
            train_sync_wall_start = time.perf_counter()
            jax.block_until_ready(train_state.step)
            train_sync_wall_s = time.perf_counter() - train_sync_wall_start
            reduced_info = jax.device_get(jax.tree.map(_reduce_metric, stacked_infos))
            reduced_info["train_sync_wall_s"] = train_sync_wall_s
            reduced_info["step_sync_wall_s"] = float(time.perf_counter() - step_wall_start)
            if (last_param_norm is None) or (step % param_norm_interval == 0):
                last_param_norm = float(jax.device_get(pparam_norm(train_state)))
                reduced_info["param_norm_fresh"] = 1.0
            else:
                reduced_info["param_norm_fresh"] = 0.0
            reduced_info["param_norm"] = float(last_param_norm)
            reduced_info["logging_wall_s"] = time.perf_counter() - logging_wall_start
            reduced_info = _canonicalize_logged_metrics(reduced_info)
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []

        if profiler_active and step + 1 >= profile_end_step:
            jax.block_until_ready(train_state.step)
            if config.profile_capture_memory:
                mem_profile_path = epath.Path(profile_dir) / f"memory_step_{step + 1}.prof"
                jax.profiler.save_device_memory_profile(str(mem_profile_path))
            jax.profiler.stop_trace()
            profiler_active = False

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()
    if teacher_bank is not None and hasattr(teacher_bank, "close"):
        teacher_bank.close()
    if profiler_active:
        jax.block_until_ready(train_state.step)
        if config.profile_capture_memory:
            mem_profile_path = epath.Path(profile_dir) / f"memory_step_{config.num_train_steps}.prof"
            jax.profiler.save_device_memory_profile(str(mem_profile_path))
        jax.profiler.stop_trace()


if __name__ == "__main__":
    main(_config.cli())
