from collections.abc import Sequence
import logging
import pathlib
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


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
    ):
        self._model = model
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._sample_actions_with_trace = None
        if hasattr(model, "sample_actions_with_trace"):
            self._sample_actions_with_trace = nnx_utils.module_jit(model.sample_actions_with_trace)
        self._sample_actions_and_qha = None
        if hasattr(model, "sample_actions_and_qha_outputs"):
            self._sample_actions_and_qha = nnx_utils.module_jit(model.sample_actions_and_qha_outputs)
        self._predict_teacher_qha_raw = None
        if hasattr(model, "predict_qha_teacher_outputs"):
            self._predict_teacher_qha_raw = nnx_utils.module_jit(model.predict_qha_teacher_outputs)
        self._input_transform = _transforms.compose(transforms)
        teacher_transforms = [t for t in transforms if not isinstance(t, _transforms.SplitHistoryActions)]
        self._teacher_input_transform = _transforms.compose(teacher_transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = dict(sample_kwargs) if sample_kwargs is not None else {}
        self._return_qha = bool(self._sample_kwargs.pop("return_qha", False))
        self._metadata = metadata or {}

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)

        self._rng, sample_rng = jax.random.split(self._rng)
        obs_batch = _model.Observation.from_dict(inputs)
        trace_enabled = bool(self._sample_kwargs.get("return_denoise_trace", False))
        sample_kwargs = {k: v for k, v in self._sample_kwargs.items() if k != "return_denoise_trace"}
        denoise_trace = None
        if self._return_qha and self._sample_actions_and_qha is not None:
            outputs = {
                "state": inputs["state"],
                **self._sample_actions_and_qha(sample_rng, obs_batch, **sample_kwargs),
            }
        elif trace_enabled and self._sample_actions_with_trace is not None:
            actions, denoise_trace = self._sample_actions_with_trace(sample_rng, obs_batch, **sample_kwargs)
            outputs = {
                "state": inputs["state"],
                "actions": actions,
            }
        else:
            actions = self._sample_actions(sample_rng, obs_batch, **sample_kwargs)
            outputs = {
                "state": inputs["state"],
                "actions": actions,
            }

        # Keep infer path on-device. Any host transfer should happen only at explicit boundaries.
        outputs = jax.tree.map(lambda x: x[0, ...], outputs)
        transformed_outputs = self._output_transform(outputs)
        if denoise_trace is not None:
            transformed_outputs["denoise_trace"] = jax.tree.map(lambda x: x[0, ...], denoise_trace)
        return transformed_outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def infer_teacher_qha(self, obs: dict, actions_ref: np.ndarray | None = None) -> dict[str, np.ndarray] | None:
        """Compute teacher QHA quantities in raw input space."""
        if self._predict_teacher_qha_raw is None:
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
            actions = jnp.asarray(inputs.pop("actions"))[None, ...]
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)

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
