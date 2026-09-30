from collections.abc import Sequence
import logging
import pathlib
import time
import inspect
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


def _unbatch_singleton(value):
    arr = np.asarray(value)
    if arr.ndim > 0 and arr.shape[0] == 1:
        return arr[0, ...]
    return arr


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        # Teacher-only fast path: skip SplitHistoryActions when a caller supplies
        # the future action reference separately.
        teacher_transforms = [t for t in transforms if not isinstance(t, _transforms.SplitHistoryActions)]
        self._teacher_input_transform = _transforms.compose(teacher_transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = dict(sample_kwargs) if sample_kwargs is not None else {}
        # `return_qha` is a policy-level switch: JAX models expose QHA outputs
        # through sample_actions_and_qha_outputs(), not sample_actions().
        self._return_qha = bool(self._sample_kwargs.pop("return_qha", False))
        # This changes the compiled output tree, so it is deliberately a
        # static policy construction option rather than a per-inference flag.
        self._return_qha_features = bool(self._sample_kwargs.pop("return_qha_features", False))
        qha_method = getattr(model, "sample_actions_and_qha_outputs", None)
        self._supports_qha_features = qha_method is not None and "return_qha_features" in inspect.signature(qha_method).parameters
        if self._return_qha_features and not self._supports_qha_features:
            raise ValueError("this frozen model does not expose QHA feature traces")
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            self._sample_actions_with_trace = None
            self._sample_actions_and_qha = None
            self._predict_teacher_qha_raw = None
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._sample_actions_with_trace = None
            if hasattr(model, "sample_actions_with_trace"):
                self._sample_actions_with_trace = nnx_utils.module_jit(model.sample_actions_with_trace)
            self._sample_actions_and_qha = None
            if hasattr(model, "sample_actions_and_qha_outputs"):
                self._sample_actions_and_qha = nnx_utils.module_jit(
                    getattr(model, "sample_actions_and_qha_outputs"),
                    static_argnames=("return_qha_features",) if self._supports_qha_features else (),
                )
            self._predict_teacher_qha_raw = None
            if hasattr(model, "predict_qha_teacher_outputs"):
                self._predict_teacher_qha_raw = nnx_utils.module_jit(getattr(model, "predict_qha_teacher_outputs"))
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        denoise_trace = None
        if not self._is_pytorch_model:
            trace_enabled = bool(sample_kwargs.pop("return_denoise_trace", False))
            if self._return_qha and self._sample_actions_and_qha is not None:
                outputs = {
                    "state": inputs["state"],
                    **self._sample_actions_and_qha(
                        sample_rng_or_pytorch_device,
                        observation,
                        **({"return_qha_features": self._return_qha_features} if self._supports_qha_features else {}),
                        **sample_kwargs,
                    ),
                }
            elif trace_enabled and self._sample_actions_with_trace is not None:
                actions, denoise_trace = self._sample_actions_with_trace(
                    sample_rng_or_pytorch_device,
                    observation,
                    **sample_kwargs,
                )
                outputs = {
                    "state": inputs["state"],
                    "actions": actions,
                }
            else:
                actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
                outputs = {
                    "state": inputs["state"],
                    "actions": actions,
                }
            outputs = jax.tree.map(_unbatch_singleton, outputs)
            outputs = self._output_transform(outputs)
            if denoise_trace is not None:
                outputs["denoise_trace"] = jax.tree.map(_unbatch_singleton, denoise_trace)
        else:
            sample_result = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
            outputs = {
                "state": inputs["state"],
                "actions": sample_result["actions"] if isinstance(sample_result, dict) else sample_result,
            }
            if isinstance(sample_result, dict) and sample_result.get("denoise_trace") is not None:
                outputs["denoise_trace"] = sample_result["denoise_trace"]
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
            outputs = self._output_transform(outputs)

        model_time = time.monotonic() - start_time
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def reset_rng(self, seed: int) -> None:
        """Reset the JAX sampling stream at an episode boundary."""
        if self._is_pytorch_model:
            raise RuntimeError("reset_rng is only supported by JAX policies.")
        self._rng = jax.random.key(int(seed) & 0xFFFFFFFF)

    def infer_with_forked_rng(
        self,
        obs: dict,
        *,
        fold_in: int,
        noise: np.ndarray | None = None,
    ) -> dict:
        """Infer from an independent JAX RNG branch without advancing deployment RNG."""
        if self._is_pytorch_model:
            raise RuntimeError("infer_with_forked_rng is only supported by JAX policies.")
        deployment_rng = self._rng
        self._rng = jax.random.fold_in(deployment_rng, int(fold_in) & 0xFFFFFFFF)
        try:
            return self.infer(obs, noise=noise)
        finally:
            self._rng = deployment_rng

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def infer_teacher_qha(self, obs: dict, actions_ref: np.ndarray | None = None) -> dict[str, np.ndarray] | None:
        """Compute teacher QHA quantities in raw input space."""
        if self._is_pytorch_model or self._predict_teacher_qha_raw is None:
            return None
        inputs = jax.tree.map(lambda x: x, obs)
        if actions_ref is not None:
            actions_ref = np.asarray(actions_ref)
            if actions_ref.ndim != 2:
                raise ValueError(f"actions_ref must be rank-2 [H, D], got shape {actions_ref.shape}")
            inputs["actions"] = actions_ref
        inputs = self._teacher_input_transform(inputs)
        actions = None
        if "actions" in inputs:
            actions = jnp.asarray(inputs.pop("actions"))[np.newaxis, ...]
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        self._rng, qha_rng = jax.random.split(self._rng)
        observation = _model.Observation.from_dict(inputs)
        outputs = self._predict_teacher_qha_raw(qha_rng, observation, actions)
        if outputs is None:
            return None
        result = {}
        for key, value in outputs.items():
            arr = np.asarray(value)
            if arr.ndim > 0 and arr.shape[0] == 1:
                result[key] = arr[0, ...]
            else:
                result[key] = arr
        return result


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
