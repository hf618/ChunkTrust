import dataclasses
import logging
import os
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


def _resolve_policy_data_config(
    data_config: _config.DataConfig | _config.MultiDataConfig,
    robotwin_repo_id: str | None,
) -> _config.DataConfig:
    if isinstance(data_config, _config.MultiDataConfig):
        if not data_config.sub_configs:
            raise ValueError("MultiDataConfig has no sub-configs.")
        if robotwin_repo_id is None:
            return data_config.sub_configs[0]
        candidate_ids = {
            robotwin_repo_id,
            f"{robotwin_repo_id}_aloha-agilex_clean_50_repo",
        }
        matches = [item for item in data_config.sub_configs if item.repo_id in candidate_ids]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one multi-task data config for {robotwin_repo_id!r}; "
                f"found {[item.repo_id for item in matches]}."
            )
        return matches[0]
    if robotwin_repo_id is not None:
        return dataclasses.replace(data_config, asset_id=robotwin_repo_id)
    return data_config


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
    pytorch_device: str | None = None,
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
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        restored_params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
        model_config = resolved_model_config or _checkpoints.resolve_model_config_for_checkpoint(
            train_config.model,
            checkpoint_dir,
            restored_params=restored_params,
            qha_candidate_mode_override=qha_candidate_mode_override,
        )
        model = model_config.load(restored_params)
    if is_pytorch:
        model_config = train_config.model
    data_config = _resolve_policy_data_config(
        train_config.data.create(train_config.assets_dirs, model_config),
        robotwin_repo_id,
    )
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

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
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
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
    pytorch_device: str | None = None,
    robotwin_repo_id: str | None = None,
) -> _policy.Policy:
    """Create a pi05_horizon policy from a frozen base checkpoint plus QHA-only weights."""
    del pytorch_device  # Split QHA overlay currently targets Orbax/JAX pi05 checkpoints.
    repack_transforms = repack_transforms or transforms.Group()
    base_checkpoint_dir = pathlib.Path(download.maybe_download(str(base_checkpoint_dir)))
    qha_checkpoint_dir = pathlib.Path(download.maybe_download(str(qha_checkpoint_dir)))

    logging.info("Loading base model from %s", base_checkpoint_dir)
    # Restore to host arrays first to avoid an unnecessary full-model device
    # placement before the merged policy is materialized.
    base_params = _model.restore_params(base_checkpoint_dir / "params", restore_type=np.ndarray, dtype=jnp.bfloat16)

    logging.info("Loading QHA head from %s", qha_checkpoint_dir)
    qha_params = _model.restore_params(qha_checkpoint_dir / "params", restore_type=np.ndarray, dtype=jnp.bfloat16)

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

    data_config = _resolve_policy_data_config(
        train_config.data.create(train_config.assets_dirs, model_config),
        robotwin_repo_id,
    )
    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(base_checkpoint_dir / "assets", data_config.asset_id)

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
        is_pytorch=False,
        pytorch_device=None,
    )
