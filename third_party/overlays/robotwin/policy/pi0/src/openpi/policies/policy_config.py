from collections.abc import Sequence
import dataclasses
import logging
import pathlib
from typing import Any

import flax.traverse_util
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


@dataclasses.dataclass
class PolicyConfig:
    model: _model.BaseModel
    norm_stats: dict[str, transforms.NormStats]

    input_layers: Sequence[transforms.DataTransformFn]
    output_layers: Sequence[transforms.DataTransformFn]

    model_type: _model.ModelType = _model.ModelType.PI0
    default_prompt: str | None = None
    sample_kwargs: dict[str, Any] | None = None


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    resolved_model_config: _model.BaseModelConfig | None = None,
    qha_candidate_mode_override: str = "auto",
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    robotwin_repo_id: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    logging.info("Loading model...")
    restored_params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    model_config = resolved_model_config or _checkpoints.resolve_model_config_for_checkpoint(
        train_config.model,
        checkpoint_dir,
        restored_params=restored_params,
        qha_candidate_mode_override=qha_candidate_mode_override,
    )
    model = model_config.load(restored_params)

    data_config = train_config.data.create(train_config.assets_dirs, model_config)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        # print(f"!!!!{data_config.asset_id}")
        # print(robotwin_repo_id)
        data_config.asset_id = robotwin_repo_id
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    if sample_kwargs is None:
        sample_kwargs = {}
    if hasattr(model_config, "use_qha"):
        sample_kwargs.setdefault("return_qha", bool(getattr(model_config, "use_qha")))

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
    )


def create_trained_policy_from_base_and_qha(
    train_config: _config.TrainConfig,
    base_checkpoint_dir: pathlib.Path | str,
    qha_checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    qha_candidate_mode_override: str = "auto",
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    robotwin_repo_id: str | None = None,
) -> _policy.Policy:
    """Create a policy from a base model checkpoint plus a separate QHA head checkpoint.

    The base checkpoint provides backbone + LoRA weights.
    The QHA checkpoint provides only ``qha.*`` weights (e.g., from joint training).
    """
    repack_transforms = repack_transforms or transforms.Group()
    base_checkpoint_dir = download.maybe_download(str(base_checkpoint_dir))
    qha_checkpoint_dir = download.maybe_download(str(qha_checkpoint_dir))

    logging.info("Loading base model from %s", base_checkpoint_dir)
    # Restore split checkpoints on host first. Restoring base params directly
    # as jax.Array can place a full pi0 checkpoint on GPU before the final
    # merged model is materialized, which creates a much higher eval-time
    # memory peak than regular single-checkpoint loading.
    base_params = _model.restore_params(base_checkpoint_dir / "params", restore_type=np.ndarray, dtype=jnp.bfloat16)

    logging.info("Loading QHA head from %s", qha_checkpoint_dir)
    qha_params = _model.restore_params(qha_checkpoint_dir / "params", restore_type=np.ndarray, dtype=jnp.bfloat16)

    # Merge: base backbone + QHA head.
    flat_base = flax.traverse_util.flatten_dict(base_params, sep="/")
    flat_qha = flax.traverse_util.flatten_dict(qha_params, sep="/")
    flat_base.update(flat_qha)
    merged_params = flax.traverse_util.unflatten_dict(flat_base, sep="/")

    model_config = _checkpoints.resolve_model_config_for_checkpoint(
        train_config.model,
        qha_checkpoint_dir,
        restored_params=merged_params,
        qha_candidate_mode_override=qha_candidate_mode_override,
    )

    model = model_config.load(merged_params)

    data_config = train_config.data.create(train_config.assets_dirs, model_config)
    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        data_config.asset_id = robotwin_repo_id or data_config.asset_id
        norm_stats = _checkpoints.load_norm_stats(
            base_checkpoint_dir / "assets", data_config.asset_id)

    if sample_kwargs is None:
        sample_kwargs = {}
    if hasattr(model_config, "use_qha"):
        sample_kwargs.setdefault("return_qha", bool(getattr(model_config, "use_qha")))

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
    )
