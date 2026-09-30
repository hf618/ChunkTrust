from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats

T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):

    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(
            self,
            *,
            inputs: Sequence[DataTransformFn] = (),
            outputs: Sequence[DataTransformFn] = (),
    ) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        out = apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )
        return self._apply_hist_actions_fallback(out)

    def _normalize(self, x, stats: NormStats):
        return (x - stats.mean) / (stats.std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        return (x - stats.q01) / (stats.q99 - stats.q01 + 1e-6) * 2.0 - 1.0

    def _apply_hist_actions_fallback(self, data: DataDict) -> DataDict:
        """Align history/action spaces by reusing `actions` stats for `hist_actions` when needed."""
        if "hist_actions" not in data or self.norm_stats is None:
            return data
        flat_stats = flatten_dict(self.norm_stats)
        if "hist_actions" in flat_stats:
            return data
        action_stats = flat_stats.get("actions")
        if action_stats is None:
            return data
        out = dict(data)
        if self.use_quantiles:
            out["hist_actions"] = self._normalize_quantile(out["hist_actions"], action_stats)
        else:
            out["hist_actions"] = self._normalize(out["hist_actions"], action_stats)
        return out


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        out = apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )
        return self._apply_hist_actions_fallback(out)

    def _unnormalize(self, x, stats: NormStats):
        return x * (stats.std + 1e-6) + stats.mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        return (x + 1.0) / 2.0 * (stats.q99 - stats.q01 + 1e-6) + stats.q01

    def _apply_hist_actions_fallback(self, data: DataDict) -> DataDict:
        if "hist_actions" not in data or self.norm_stats is None:
            return data
        flat_stats = flatten_dict(self.norm_stats)
        if "hist_actions" in flat_stats:
            return data
        action_stats = flat_stats.get("actions")
        if action_stats is None:
            return data
        out = dict(data)
        if self.use_quantiles:
            out["hist_actions"] = self._unnormalize_quantile(out["hist_actions"], action_stats)
        else:
            out["hist_actions"] = self._unnormalize(out["hist_actions"], action_stats)
        return out


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        images = data["image"]
        image_keys = tuple(images)
        image_values = [np.asarray(images[key]) for key in image_keys]
        if not image_values:
            return data

        same_shape = len({arr.shape for arr in image_values}) == 1
        same_dtype = len({np.dtype(arr.dtype) for arr in image_values}) == 1
        if same_shape and same_dtype:
            stacked = np.stack(image_values, axis=0)
            resized = image_tools.resize_with_pad(stacked, self.height, self.width)
            data["image"] = {key: resized[idx] for idx, key in enumerate(image_keys)}
            return data

        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in images.items()}
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][::self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class SplitHistoryActions(DataTransformFn):
    """Split concatenated [history + future] actions into history and training targets.

    Expected inputs:
    - actions: [history_len + future_len, action_dim]
    - actions_is_pad (optional): [history_len + future_len], True means padded/out-of-episode
    """

    history_len: int
    future_len: int

    def __post_init__(self):
        if self.history_len < 0:
            raise ValueError(f"history_len must be >= 0, got {self.history_len}")
        if self.future_len <= 0:
            raise ValueError(f"future_len must be > 0, got {self.future_len}")

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        actions = np.asarray(data["actions"])
        if actions.ndim != 2:
            raise ValueError(f"actions must have rank 2 [T, D], got shape {actions.shape}")

        total_len = self.history_len + self.future_len
        if actions.shape[0] != total_len:
            raise ValueError(
                "SplitHistoryActions expected concatenated sequence length "
                f"{total_len}, got {actions.shape[0]}."
            )

        pad = data.get("actions_is_pad")
        if pad is None:
            pad = np.zeros((total_len, ), dtype=np.bool_)
        pad = np.asarray(pad).astype(np.bool_)
        if pad.shape != (total_len, ):
            raise ValueError(f"actions_is_pad must have shape ({total_len},), got {pad.shape}")

        hist_actions = actions[:self.history_len]
        hist_pad = pad[:self.history_len]
        future_actions = actions[self.history_len:]
        if future_actions.shape[0] != self.future_len:
            raise ValueError(f"future actions length mismatch, expected {self.future_len}")

        data["actions"] = future_actions
        data["hist_actions"] = hist_actions
        # True means valid history token.
        data["hist_actions_mask"] = np.logical_not(hist_pad)
        return data


def _shift_action_prefix(actions, offset, *, dims: int, sign: float):
    """Shift the masked action prefix without mutating JAX arrays in place."""
    if hasattr(actions, "at"):
        return actions.at[..., :dims].add(sign * offset)
    out = np.array(actions, copy=True)
    out[..., :dims] += sign * offset
    return out


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.mask is None:
            return data

        if "actions" not in data and "hist_actions" not in data:
            return data

        state = data["state"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        offset = np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)

        if "actions" in data:
            data["actions"] = _shift_action_prefix(data["actions"], offset, dims=dims, sign=-1.0)
        if "hist_actions" in data:
            data["hist_actions"] = _shift_action_prefix(data["hist_actions"], offset, dims=dims, sign=-1.0)

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.mask is None:
            return data

        if "actions" not in data and "hist_actions" not in data:
            return data

        state = data["state"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        offset = np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)

        if "actions" in data:
            data["actions"] = _shift_action_prefix(data["actions"], offset, dims=dims, sign=1.0)
        if "hist_actions" in data:
            data["hist_actions"] = _shift_action_prefix(data["hist_actions"], offset, dims=dims, sign=1.0)

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    _cache: dict[str, tuple[np.ndarray, np.ndarray]] = dataclasses.field(
        default_factory=dict, init=False, repr=False, compare=False, hash=False
    )

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        cached = self._cache.get(prompt)
        if cached is None:
            cached = self.tokenizer.tokenize(prompt)
            self._cache[prompt] = cached
        tokens, token_masks = cached
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        # if "task_index" not in data:
        #     raise ValueError('Cannot extract prompt without "task_index"')

        # task_index = int(data["task_index"])
        # if (prompt := self.tasks.get(task_index)) is None:
        #     raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")
        if "task" not in data:
            raise ValueError('Cannot extract prompt: "task" key not found in data')
        prompt = data["task"]

        return {**data, "prompt": prompt}


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i:i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(tree: at.PyTree[T],
               selector: at.PyTree[S],
               fn: Callable[[T, S], T],
               *,
               strict: bool = False) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99.")
