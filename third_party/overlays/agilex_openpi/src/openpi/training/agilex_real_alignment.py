import dataclasses
import os

import numpy as np

import openpi.transforms as transforms


# Set a local LeRobot dataset directory when training or computing normalization stats.
# Inference loads normalization assets from the checkpoint and does not require the dataset.
AGILEX_REAL_REPO_ID = os.environ.get("CHUNKTRUST_AGILEX_DATASET", "agilex_magic_grab_bread_real")
AGILEX_REAL_ASSET_ID = "agilex_magic_grab_bread_real"
AGILEX_REAL_DEFAULT_PROMPT = (
    "Move the bread from the left side of the table to the right side of the table using both arms."
)

AGILEX_REAL_IMAGE_KEY_MAP = {
    "cam_high": "observation.images.image_top",
    "cam_left_wrist": "observation.images.image_left",
    "cam_right_wrist": "observation.images.image_right",
}

_RAW_ACTION_MIN_DIM = 14
_RAW_STATE_MIN_DIM = 26
_RIGHT_ARM_SLICE = slice(0, 7)
_LEFT_ARM_SLICE_IN_STATE = slice(13, 20)
_LEFT_ARM_SLICE_IN_ACTION = slice(7, 14)


def _validate_last_dim(values: np.ndarray, *, minimum_dim: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 0:
        raise ValueError(f"{name} must have at least 1 dimension, got shape={array.shape}")
    if array.shape[-1] < minimum_dim:
        raise ValueError(f"{name} last dimension must be at least {minimum_dim}, got shape={array.shape}")
    return array


def remap_agilex_real_state(state: np.ndarray) -> np.ndarray:
    array = _validate_last_dim(state, minimum_dim=_RAW_STATE_MIN_DIM, name="state")
    return np.concatenate([array[..., _LEFT_ARM_SLICE_IN_STATE], array[..., _RIGHT_ARM_SLICE]], axis=-1)


def remap_agilex_real_action(action: np.ndarray) -> np.ndarray:
    array = _validate_last_dim(action, minimum_dim=_RAW_ACTION_MIN_DIM, name="action")
    return np.concatenate([array[..., _LEFT_ARM_SLICE_IN_ACTION], array[..., _RIGHT_ARM_SLICE]], axis=-1)


def remap_agilex_real_batch(states: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return remap_agilex_real_state(states), remap_agilex_real_action(actions)


@dataclasses.dataclass(frozen=True)
class AgilexRealToAlohaInputs(transforms.DataTransformFn):
    """Convert raw AgileX real-robot tensors into the 14-dim ALOHA format expected by pi05."""

    def __call__(self, data: dict) -> dict:
        data["state"] = remap_agilex_real_state(data["state"])
        if "actions" in data:
            data["actions"] = remap_agilex_real_action(data["actions"])
        return data
