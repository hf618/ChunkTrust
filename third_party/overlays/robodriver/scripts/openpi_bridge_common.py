"""Shared raw26 and robot I/O helpers for the two synchronous rollout CLIs.

Derived from the workstation's smoke-action script; this module has no CLI.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


PROMPT = "Move the bread from the left side of the table to the right side of the table using both arms."
RIGHT_KEYS = [
    "joint1_right",
    "joint2_right",
    "joint3_right",
    "joint4_right",
    "joint5_right",
    "joint6_right",
    "gripper_right",
]
LEFT_KEYS = [
    "joint1_left",
    "joint2_left",
    "joint3_left",
    "joint4_left",
    "joint5_left",
    "joint6_left",
    "gripper_left",
]


def make_robot() -> Any:
    from robodriver.robots.utils import make_robot_from_config
    from robodriver.utils.import_utils import register_third_party_devices

    register_third_party_devices()
    from robodriver_robot_agilex_aloha_aio_dora.config import AgilexAlohaAIODoraRobotConfig

    robot = make_robot_from_config(AgilexAlohaAIODoraRobotConfig())
    if not robot.is_connected:
        robot.connect()
    return robot


def get_complete_observation(robot: Any, *, timeout_sec: float = 5.0, poll_sec: float = 0.05) -> dict[str, Any]:
    deadline = time.perf_counter() + timeout_sec
    last_error: Exception | None = None
    while time.perf_counter() < deadline:
        try:
            observation = robot.get_observation()
            observation_to_raw26(observation)
            return observation
        except (IndexError, KeyError, RuntimeError) as exc:
            last_error = exc
            time.sleep(poll_sec)
    raise RuntimeError(f"Timed out waiting for complete robot observation: {last_error}") from last_error


def observation_to_raw26(observation: dict[str, Any]) -> np.ndarray:
    values: list[float] = []
    for key in RIGHT_KEYS:
        values.append(float(observation[f"follower_{key}.pos"]))
    values.extend([float(observation[f"follower_pose_right_{axis}.pos"]) for axis in ["x", "y", "z", "rx", "ry", "rz"]])
    for key in LEFT_KEYS:
        values.append(float(observation[f"follower_{key}.pos"]))
    values.extend([float(observation[f"follower_pose_left_{axis}.pos"]) for axis in ["x", "y", "z", "rx", "ry", "rz"]])
    raw26 = np.asarray(values, dtype=np.float32)
    if raw26.shape != (26,):
        raise RuntimeError(f"Expected raw26 state, got {raw26.shape}")
    return raw26


def observation_to_policy_input(observation: dict[str, Any], raw26: np.ndarray) -> dict[str, Any]:
    return {
        "state": raw26,
        "images": {
            "cam_high": chw_uint8(observation["image_top"]),
            "cam_left_wrist": chw_uint8(observation["image_left"]),
            "cam_right_wrist": chw_uint8(observation["image_right"]),
        },
        "prompt": PROMPT,
    }


def chw_uint8(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise RuntimeError(f"Expected HWC or CHW image, got shape={array.shape}")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[0] == 3:
        return array
    if array.shape[-1] == 3:
        return np.transpose(array, (2, 0, 1))
    raise RuntimeError(f"Expected 3 color channels, got shape={array.shape}")


def raw26_to_left_right(raw26: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right = raw26[0:7].astype(np.float32)
    left = raw26[13:20].astype(np.float32)
    return left, right


def action_dict_from_left_right(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    action: dict[str, float] = {}
    for key, value in zip(RIGHT_KEYS, right, strict=True):
        action[f"leader_{key}.pos"] = float(value)
    for key, value in zip(LEFT_KEYS, left, strict=True):
        action[f"leader_{key}.pos"] = float(value)
    return action


def print_delta(name: str, current: np.ndarray, target: np.ndarray) -> None:
    delta = target - current
    print(
        f"{name}: joints_delta={np.array2string(delta[:6], precision=5, suppress_small=False)} "
        f"gripper_delta={delta[6]:.5f}"
    )
