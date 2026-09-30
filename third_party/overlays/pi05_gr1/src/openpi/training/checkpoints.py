import concurrent.futures as futures
import dataclasses
import enum
import hashlib
import json
import logging
import pathlib
import shutil
from typing import Callable, Protocol

from etils import epath
import flax.nnx as nnx
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from openpi.models import qha as _qha
import openpi.models.model as _model
from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


MODEL_CONFIG_SNAPSHOT_NAME = "_train_config_model.json"
TRAIN_CONFIG_SNAPSHOT_NAME = "_train_config_resolved.json"


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str,
    *,
    keep_period: int | None,
    overwrite: bool,
    resume: bool,
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                                  "to indicate how to handle it.")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. in this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0, )]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
    *,
    model_config: _model.BaseModelConfig | None = None,
    train_config: object | None = None,
):

    def save_assets(directory: epath.Path):
        data_configs = (
            data_loader.data_configs()
            if hasattr(data_loader, "data_configs")
            else (data_loader.data_config(),)
        )
        serialized_by_asset: dict[str, str] = {}
        for data_config in data_configs:
            norm_stats = data_config.norm_stats
            asset_id = data_config.asset_id
            if norm_stats is None or asset_id is None:
                continue
            serialized = _normalize.serialize_json(norm_stats)
            if asset_id in serialized_by_asset:
                if serialized_by_asset[asset_id] != serialized:
                    raise ValueError(f"Data configs disagree on normalization stats for shared asset {asset_id}.")
                continue
            serialized_by_asset[asset_id] = serialized
            _normalize.save(directory / asset_id, norm_stats)
        if model_config is not None:
            save_model_config_snapshot(directory, model_config)
        if train_config is not None:
            save_train_config_snapshot(directory, train_config)
            save_checkpoint_metadata_files(directory, train_config)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {
            "params": params
        },
    }
    checkpoint_manager.save(step, items)


def save_qha_only_params(
    state: training_utils.TrainState,
    output_dir: epath.Path | str,
    *,
    model_config: _model.BaseModelConfig | None = None,
) -> None:
    """Extract and save only qha.* inference params."""
    output_dir = epath.Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _, params = _split_params(state)
    if isinstance(params, nnx.State):
        params = params.to_pure_dict()

    flat = flax.traverse_util.flatten_dict(params, sep="/")
    qha_flat = {key: value for key, value in flat.items() if key.startswith("qha/")}
    if not qha_flat:
        raise ValueError("No qha.* parameters found in train state. Is use_qha=True?")
    qha_params = flax.traverse_util.unflatten_dict(qha_flat, sep="/")

    def _save_assets(directory: epath.Path):
        directory.mkdir(parents=True, exist_ok=True)
        if model_config is not None:
            save_model_config_snapshot(directory, model_config)
        (directory / "_QHA_ONLY").write_text("1", encoding="utf-8")

    ckptr = ocp.PyTreeCheckpointer()
    ckptr.save(str(output_dir / "params"), {"params": qha_params})
    _save_assets(output_dir / "assets")
    logging.info("QHA-only checkpoint saved to %s (%d parameter groups)", output_dir, len(qha_flat))


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {
                    "params": params
                },
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


def _model_config_snapshot_path(assets_dir: epath.Path | str) -> epath.Path:
    return epath.Path(assets_dir) / MODEL_CONFIG_SNAPSHOT_NAME


def save_model_config_snapshot(assets_dir: epath.Path | str, model_config: _model.BaseModelConfig) -> None:
    if not dataclasses.is_dataclass(model_config):
        return
    snapshot = {
        field.name: getattr(model_config, field.name)
        for field in dataclasses.fields(model_config)
        if field.init
    }
    snapshot_path = _model_config_snapshot_path(assets_dir)
    snapshot_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")


def _json_default(value):
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, enum.Enum):
        return value.value
    if dataclasses.is_dataclass(value):
        return {
            field.name: getattr(value, field.name)
            for field in dataclasses.fields(value)
            if field.init
        }
    if isinstance(value, (set, tuple)):
        return list(value)
    return repr(value)


def save_train_config_snapshot(assets_dir: epath.Path | str, train_config: object) -> None:
    if not dataclasses.is_dataclass(train_config):
        return
    snapshot = {
        field.name: getattr(train_config, field.name)
        for field in dataclasses.fields(train_config)
        if field.init
    }
    snapshot_path = epath.Path(assets_dir) / TRAIN_CONFIG_SNAPSHOT_NAME
    snapshot_path.write_text(
        json.dumps(snapshot, default=_json_default, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def save_checkpoint_metadata_files(assets_dir: epath.Path | str, train_config: object) -> None:
    raw_paths = getattr(train_config, "checkpoint_metadata_files", ())
    if not raw_paths:
        return
    audit_dir = pathlib.Path(str(assets_dir)) / "_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    hash_lines: list[str] = []
    seen_names: set[str] = set()
    for raw_path in raw_paths:
        source = pathlib.Path(raw_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Checkpoint metadata file does not exist: {source}")
        if source.name in seen_names:
            raise ValueError(f"Duplicate checkpoint metadata basename: {source.name}")
        seen_names.add(source.name)
        destination = audit_dir / source.name
        shutil.copy2(source, destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        hash_lines.append(f"{digest}  {destination.name}")
    (audit_dir / "metadata.sha256").write_text("\n".join(sorted(hash_lines)) + "\n", encoding="utf-8")


def load_model_config_snapshot(
    assets_dir: epath.Path | str,
    base_model_config: _model.BaseModelConfig,
) -> tuple[_model.BaseModelConfig, bool]:
    snapshot_path = _model_config_snapshot_path(assets_dir)
    if not snapshot_path.exists():
        return base_model_config, False
    try:
        raw = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("Failed to read model config snapshot from %s: %s", snapshot_path, exc)
        return base_model_config, False

    allowed_fields = {field.name for field in dataclasses.fields(base_model_config) if field.init}
    updates = {key: value for key, value in raw.items() if key in allowed_fields}
    if not updates:
        return base_model_config, True
    return dataclasses.replace(base_model_config, **updates), True


def _apply_qha_candidate_mode(
    model_config: _model.BaseModelConfig,
    candidate_mode: str,
) -> _model.BaseModelConfig:
    if not hasattr(model_config, "qha_candidate_mode") or not hasattr(model_config, "qha_candidates"):
        return model_config
    action_horizon = int(getattr(model_config, "action_horizon"))
    raw_candidates = getattr(model_config, "qha_candidates")
    if candidate_mode == "sparse":
        try:
            current_candidates = _qha.parse_candidate_horizons(raw_candidates, max_horizon=action_horizon)
        except Exception:
            current_candidates = ()
        if len(current_candidates) == action_horizon and tuple(current_candidates) == tuple(range(1, action_horizon + 1)):
            raw_candidates = _qha.default_sparse_runtime_candidate_horizons(max_horizon=action_horizon)
    resolved = _qha.resolve_qha_candidate_horizons(
        raw_candidates,
        candidate_mode=candidate_mode,
        max_horizon=action_horizon,
    )
    return dataclasses.replace(
        model_config,
        qha_candidate_mode=candidate_mode,
        qha_candidates=",".join(str(v) for v in resolved),
    )


def _infer_qha_candidate_mode_from_params(
    model_config: _model.BaseModelConfig,
    restored_params: at.Params,
) -> str | None:
    if not hasattr(model_config, "qha_candidate_mode") or not hasattr(model_config, "qha_candidates"):
        return None
    flat_params = flax.traverse_util.flatten_dict(restored_params)
    bias = flat_params.get(("qha", "out_mlp", "fc2", "bias"))
    if bias is None or not hasattr(bias, "shape") or len(bias.shape) != 1:
        return None

    action_horizon = int(getattr(model_config, "action_horizon"))
    sparse_candidates = _qha.resolve_qha_candidate_horizons(
        getattr(model_config, "qha_candidates"),
        candidate_mode="sparse",
        max_horizon=action_horizon,
    )
    out_dim = int(bias.shape[0])
    if out_dim == action_horizon:
        return "dense_full"
    if out_dim == len(sparse_candidates):
        return "sparse"
    return None


def resolve_model_config_for_checkpoint(
    base_model_config: _model.BaseModelConfig,
    checkpoint_dir: epath.Path | str,
    *,
    restored_params: at.Params | None = None,
    qha_candidate_mode_override: str = "auto",
) -> _model.BaseModelConfig:
    checkpoint_dir = epath.Path(checkpoint_dir)
    model_config, has_snapshot = load_model_config_snapshot(checkpoint_dir / "assets", base_model_config)

    if not has_snapshot and hasattr(model_config, "use_qha") and bool(getattr(model_config, "use_qha", False)):
        if restored_params is None:
            restored_params = _model.restore_params(checkpoint_dir / "params", restore_type=np.ndarray)
        inferred_mode = _infer_qha_candidate_mode_from_params(model_config, restored_params)
        if inferred_mode is not None:
            model_config = _apply_qha_candidate_mode(model_config, inferred_mode)
            logging.info("Inferred qha_candidate_mode=%s from checkpoint params at %s", inferred_mode, checkpoint_dir)

    override = str(qha_candidate_mode_override).strip().lower()
    if override not in ("auto", "sparse", "dense_full"):
        raise ValueError(
            "qha_candidate_mode_override must be auto, sparse, or dense_full, "
            f"got {qha_candidate_mode_override}."
        )
    if override != "auto":
        model_config = _apply_qha_candidate_mode(model_config, override)
        logging.info("Applied qha_candidate_mode_override=%s for checkpoint at %s", override, checkpoint_dir)

    return model_config


class Callback(Protocol):

    def __call__(self, directory: epath.Path) -> None:
        ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def __init__(self):
        self._executor = futures.ThreadPoolExecutor(max_workers=1)

    def close(self):
        self._executor.shutdown()

    def save(self, directory: epath.Path, args: "CallbackSave"):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: "CallbackSave") -> list[futures.Future]:
        return [self._executor.submit(self.save, directory, args)]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs):
    ...


def _split_params(state: training_utils.TrainState, ) -> tuple[training_utils.TrainState, at.Params]:
    with at.disable_typechecking():
        if state.ema_params is not None:
            params = _serialize_prng_keys_in_state(state.ema_params)
            train_state = dataclasses.replace(state, ema_params=None)
        else:
            params = _serialize_prng_keys_in_state(state.params)
            train_state = dataclasses.replace(state, params=nnx.State({}))
    train_state = _serialize_prng_keys_in_train_state(train_state)
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    train_state = _deserialize_prng_keys_in_train_state(train_state)
    restored_params = _deserialize_prng_keys_in_state(params["params"])
    with at.disable_typechecking():
        if train_state.params:
            return dataclasses.replace(train_state, ema_params=restored_params)
        return dataclasses.replace(train_state, params=restored_params)


def _is_prng_key_dtype(value: object) -> bool:
    dtype = getattr(value, "dtype", None)
    return dtype is not None and str(dtype).startswith("key<")


def _looks_like_rng_key_path(path: str) -> bool:
    # NNX dropout/random state keys are typically stored under ".../rngs/.../key".
    return "/rngs/" in path and path.endswith("/key")


def _is_serialized_key_data(value: object) -> bool:
    if not hasattr(value, "dtype") or not hasattr(value, "shape"):
        return False
    if getattr(value, "dtype") != jnp.uint32:
        return False
    shape = getattr(value, "shape")
    return len(shape) >= 1 and shape[-1] == 2


def _map_nnx_state_leaves(
    state: nnx.State,
    fn: Callable[[str, at.Array], at.Array],
) -> nnx.State:
    copied = nnx.State(state)
    flat = flax.traverse_util.flatten_dict(copied.to_pure_dict(), sep="/")
    mapped = {k: fn(k, v) for k, v in flat.items()}
    copied.replace_by_pure_dict(flax.traverse_util.unflatten_dict(mapped, sep="/"))
    return copied


def _serialize_prng_keys_in_state(state: at.Params) -> at.Params:
    if not isinstance(state, nnx.State):
        return state

    def _serialize_leaf(path: str, value: at.Array) -> at.Array:
        if _is_prng_key_dtype(value):
            if isinstance(value, jax.ShapeDtypeStruct):
                # Restore templates use ShapeDtypeStruct leaves. Match saved dtype/shape.
                return jax.ShapeDtypeStruct(
                    shape=value.shape + (2, ),
                    dtype=jnp.uint32,
                    sharding=value.sharding,
                    weak_type=value.weak_type,
                )
            return jax.random.key_data(value)
        return value

    return _map_nnx_state_leaves(state, _serialize_leaf)


def _deserialize_prng_keys_in_state(state: at.Params) -> at.Params:
    if not isinstance(state, nnx.State):
        return state

    def _deserialize_leaf(path: str, value: at.Array) -> at.Array:
        if _looks_like_rng_key_path(path) and _is_serialized_key_data(value):
            return jax.random.wrap_key_data(value)
        return value

    return _map_nnx_state_leaves(state, _deserialize_leaf)


def _serialize_prng_keys_in_train_state(state: training_utils.TrainState) -> training_utils.TrainState:
    params = state.params
    ema_params = state.ema_params
    if isinstance(params, nnx.State):
        params = _serialize_prng_keys_in_state(params)
    if isinstance(ema_params, nnx.State):
        ema_params = _serialize_prng_keys_in_state(ema_params)
    with at.disable_typechecking():
        return dataclasses.replace(state, params=params, ema_params=ema_params)


def _deserialize_prng_keys_in_train_state(state: training_utils.TrainState) -> training_utils.TrainState:
    params = state.params
    ema_params = state.ema_params
    if isinstance(params, nnx.State):
        params = _deserialize_prng_keys_in_state(params)
    if isinstance(ema_params, nnx.State):
        ema_params = _deserialize_prng_keys_in_state(ema_params)
    with at.disable_typechecking():
        return dataclasses.replace(state, params=params, ema_params=ema_params)
