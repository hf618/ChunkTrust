"""RoboCasa GR1 Tabletop observation and action transforms for OpenPI."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses

import einops
import numpy as np

from openpi import transforms


ACTIVE_PARTS = ("left_arm", "right_arm", "left_hand", "right_hand", "waist")
RAW_SLICES = {
    "left_arm": slice(0, 7),
    "left_hand": slice(7, 13),
    "right_arm": slice(22, 29),
    "right_hand": slice(29, 35),
    "waist": slice(41, 44),
}
PART_DIMS = {
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 6,
    "right_hand": 6,
    "waist": 3,
}
RAW_DIM = 44
ACTIVE_DIM = 29


def extract_active(value: np.ndarray) -> np.ndarray:
    """Extract the frozen 29D active GR1 controls from a raw 44D vector."""
    value = np.asarray(value)
    if value.shape[-1] == ACTIVE_DIM:
        return value
    if value.shape[-1] != RAW_DIM:
        raise ValueError(f"Expected GR1 vector with last dim 44 or 29, got {value.shape}")
    return np.concatenate([value[..., RAW_SLICES[name]] for name in ACTIVE_PARTS], axis=-1)


def expand_active(value: np.ndarray) -> np.ndarray:
    """Expand a 29D active vector into the original 44D GR1 layout."""
    value = np.asarray(value)
    if value.shape[-1] == RAW_DIM:
        return value
    if value.shape[-1] != ACTIVE_DIM:
        raise ValueError(f"Expected GR1 vector with last dim 29 or 44, got {value.shape}")
    output = np.zeros((*value.shape[:-1], RAW_DIM), dtype=value.dtype)
    offset = 0
    for name in ACTIVE_PARTS:
        width = PART_DIMS[name]
        output[..., RAW_SLICES[name]] = value[..., offset : offset + width]
        offset += width
    return output


def active_to_action_dict(value: np.ndarray) -> dict[str, np.ndarray]:
    """Convert canonical active vectors to the keyed action API used by RoboCasa."""
    value = np.asarray(value)
    if value.shape[-1] != ACTIVE_DIM:
        value = extract_active(value)
    result: dict[str, np.ndarray] = {}
    offset = 0
    for name in ACTIVE_PARTS:
        width = PART_DIMS[name]
        result[f"action.{name}"] = value[..., offset : offset + width]
        offset += width
    return result


def action_dict_to_active(value: Mapping[str, np.ndarray]) -> np.ndarray:
    """Convert keyed RoboCasa actions into the frozen canonical active order."""
    arrays = []
    prefix_shape: tuple[int, ...] | None = None
    for name in ACTIVE_PARTS:
        key = f"action.{name}"
        if key not in value:
            raise KeyError(f"Missing GR1 action key: {key}")
        array = np.asarray(value[key])
        if array.shape[-1] != PART_DIMS[name]:
            raise ValueError(f"{key}: expected last dim {PART_DIMS[name]}, got {array.shape}")
        if prefix_shape is None:
            prefix_shape = array.shape[:-1]
        elif array.shape[:-1] != prefix_shape:
            raise ValueError("GR1 action dict values have inconsistent leading shapes.")
        arrays.append(array)
    return np.concatenate(arrays, axis=-1)


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected one GR1 RGB image, got shape {image.shape}")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with three channels, got shape {image.shape}")
    return image


def _empty_model_camera() -> np.ndarray:
    return np.zeros((224, 224, 3), dtype=np.uint8)


@dataclasses.dataclass(frozen=True)
class Gr1TabletopInputs(transforms.DataTransformFn):
    """Map the official LeRobot sample or inference request into OpenPI inputs."""

    def __call__(self, data: dict) -> dict:
        image = _parse_image(data["observation/image"])
        state = extract_active(np.asarray(data["observation/state"], dtype=np.float32))
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": image,
                "left_wrist_0_rgb": _empty_model_camera(),
                "right_wrist_0_rgb": _empty_model_camera(),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = extract_active(np.asarray(data["actions"], dtype=np.float32))
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class Gr1TabletopOutputs(transforms.DataTransformFn):
    """Return unnormalized 29D actions in the frozen canonical order."""

    def __call__(self, data: dict) -> dict:
        outputs = dict(data)
        outputs["actions"] = np.asarray(data["actions"])[..., :ACTIVE_DIM]
        return outputs
