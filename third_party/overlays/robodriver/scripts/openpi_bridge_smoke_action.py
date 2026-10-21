#!/usr/bin/env python3
"""Smoke-test OpenPI pi05 actions against the AgileX ALOHA Dora robot.

By default this script only previews the command. Add --execute to actually send it.
"""

from __future__ import annotations

import argparse
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
JOINT_CHOICES = [f"joint{i}" for i in range(1, 7)] + ["gripper"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["policy-first-step", "soft-reset", "manual-delta", "manual-set"], required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--state-format", choices=["raw26"], default="raw26")
    parser.add_argument("--policy-alpha", type=float, default=0.03)
    parser.add_argument("--max-joint-delta-rad", type=float, default=0.005)
    parser.add_argument("--max-gripper-delta", type=float, default=0.002)
    parser.add_argument("--single-arm-only", choices=["left", "right"], default=None)
    parser.add_argument("--arm", choices=["left", "right", "both"], default="right")
    parser.add_argument("--joint", choices=JOINT_CHOICES, default="joint5")
    parser.add_argument("--delta", type=float, default=0.02)
    parser.add_argument("--target", type=float, default=0.0)
    parser.add_argument("--disable-gripper", action="store_true")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--sleep-sec", type=float, default=0.3)
    parser.add_argument("--post-check-sec", type=float, default=0.0)
    parser.add_argument("--max-offtarget-motion-rad", type=float, default=0.04)
    parser.add_argument(
        "--manual-send-scope",
        choices=["target-arm", "both-arms"],
        default="target-arm",
        help="For manual modes, send only the selected arm by default to avoid disturbing the other arm.",
    )
    parser.add_argument("--execute", type=int, choices=[0, 1], default=0)
    return parser.parse_args()


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


def limit_target(
    current_left: np.ndarray,
    current_right: np.ndarray,
    policy_left: np.ndarray,
    policy_right: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    target_left = blend_and_clip(current_left, policy_left, args)
    target_right = blend_and_clip(current_right, policy_right, args)

    if args.single_arm_only == "left":
        target_right = current_right.copy()
    elif args.single_arm_only == "right":
        target_left = current_left.copy()

    if args.disable_gripper:
        target_left[6] = current_left[6]
        target_right[6] = current_right[6]

    return target_left, target_right


def blend_and_clip(current: np.ndarray, target: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    blended = current + args.policy_alpha * (target - current)
    delta = blended - current
    delta[:6] = np.clip(delta[:6], -args.max_joint_delta_rad, args.max_joint_delta_rad)
    delta[6] = np.clip(delta[6], -args.max_gripper_delta, args.max_gripper_delta)
    return current + delta


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


def policy_first_step(robot: Any, args: argparse.Namespace) -> None:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    observation = get_complete_observation(robot)
    raw26 = observation_to_raw26(observation)
    current_left, current_right = raw26_to_left_right(raw26)

    client = WebsocketClientPolicy(host=args.host, port=args.port)
    result = client.infer(observation_to_policy_input(observation, raw26))
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] < 14:
        raise RuntimeError(f"Expected policy actions [horizon, >=14], got {actions.shape}")

    policy_left = actions[0, 0:7]
    policy_right = actions[0, 7:14]
    target_left, target_right = limit_target(current_left, current_right, policy_left, policy_right, args)

    print(f"policy actions shape: {actions.shape}")
    print(f"selected first step; execute: {args.execute}")
    print(f"single arm: {args.single_arm_only or 'both'}; disable_gripper: {args.disable_gripper}")
    print_delta("left", current_left, target_left)
    print_delta("right", current_right, target_right)

    if args.execute:
        send_repeated(robot, action_dict_from_left_right(target_left, target_right), args)
        report_motion_check(robot, current_left, current_right, args)


def soft_reset(robot: Any, args: argparse.Namespace) -> None:
    for index in range(args.repeat):
        observation = get_complete_observation(robot)
        raw26 = observation_to_raw26(observation)
        current_left, current_right = raw26_to_left_right(raw26)
        print(f"soft-reset step {index + 1}/{args.repeat}; execute: {args.execute}")
        if args.execute:
            robot.send_action(action_dict_from_left_right(current_left, current_right))
        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)
    if args.post_check_sec > 0:
        time.sleep(args.post_check_sec)


def manual_delta(robot: Any, args: argparse.Namespace) -> None:
    observation = get_complete_observation(robot)
    raw26 = observation_to_raw26(observation)
    current_left, current_right = raw26_to_left_right(raw26)
    target_left = current_left.copy()
    target_right = current_right.copy()

    index = 6 if args.joint == "gripper" else int(args.joint.removeprefix("joint")) - 1
    if args.arm in ["left", "both"]:
        target_left[index] += args.delta
    if args.arm in ["right", "both"]:
        target_right[index] += args.delta

    print(f"manual-delta arm={args.arm} joint={args.joint} delta={args.delta:.5f}; execute: {args.execute}")
    print_delta("left", current_left, target_left)
    print_delta("right", current_right, target_right)

    if args.execute:
        send_manual_targets(robot, target_left, target_right, args)
        report_motion_check(robot, current_left, current_right, args)
        check_offtarget_motion(robot, current_left, current_right, index, args)


def manual_set(robot: Any, args: argparse.Namespace) -> None:
    observation = get_complete_observation(robot)
    raw26 = observation_to_raw26(observation)
    current_left, current_right = raw26_to_left_right(raw26)
    target_left = current_left.copy()
    target_right = current_right.copy()

    index = 6 if args.joint == "gripper" else int(args.joint.removeprefix("joint")) - 1
    if args.arm in ["left", "both"]:
        target_left[index] = args.target
    if args.arm in ["right", "both"]:
        target_right[index] = args.target
    before = (
        f"left={current_left[index]:.5f}, right={current_right[index]:.5f}"
        if args.arm == "both"
        else f"{current_left[index] if args.arm == 'left' else current_right[index]:.5f}"
    )

    print(
        f"manual-set arm={args.arm} joint={args.joint} "
        f"current={before} target={args.target:.5f}; execute: {args.execute}"
    )
    print_delta("left", current_left, target_left)
    print_delta("right", current_right, target_right)

    if args.execute:
        send_manual_targets(robot, target_left, target_right, args)
        report_motion_check(robot, current_left, current_right, args)


def send_repeated(robot: Any, action: dict[str, float], args: argparse.Namespace) -> None:
    repeat = max(1, args.repeat)
    for index in range(repeat):
        robot.send_action(action)
        print(f"sent action {index + 1}/{repeat}")
        if args.sleep_sec > 0 and index + 1 < repeat:
            time.sleep(args.sleep_sec)


def send_manual_targets(robot: Any, target_left: np.ndarray, target_right: np.ndarray, args: argparse.Namespace) -> None:
    if args.manual_send_scope == "both-arms" or args.arm == "both":
        send_repeated(robot, action_dict_from_left_right(target_left, target_right), args)
        return

    topic = f"action_joint_{args.arm}"
    target = target_left if args.arm == "left" else target_right
    repeat = max(1, args.repeat)
    for index in range(repeat):
        robot.robot_dora_node.dora_send(topic, np.asarray(target, dtype=np.float32))
        print(f"sent {topic} {index + 1}/{repeat}")
        if args.sleep_sec > 0 and index + 1 < repeat:
            time.sleep(args.sleep_sec)


def report_motion_check(
    robot: Any,
    before_left: np.ndarray,
    before_right: np.ndarray,
    args: argparse.Namespace,
) -> None:
    if args.post_check_sec > 0:
        time.sleep(args.post_check_sec)
    after = get_complete_observation(robot)
    after_left, after_right = raw26_to_left_right(observation_to_raw26(after))
    print_delta("observed-left", before_left, after_left)
    print_delta("observed-right", before_right, after_right)


def check_offtarget_motion(
    robot: Any,
    before_left: np.ndarray,
    before_right: np.ndarray,
    target_index: int,
    args: argparse.Namespace,
) -> None:
    after = get_complete_observation(robot)
    after_left, after_right = raw26_to_left_right(observation_to_raw26(after))
    for arm_name, before, after_arm in [
        ("left", before_left, after_left),
        ("right", before_right, after_right),
    ]:
        joint_motion = np.abs(after_arm[:6] - before[:6])
        allowed = np.zeros(6, dtype=bool)
        if args.arm in [arm_name, "both"] and target_index < 6:
            allowed[target_index] = True
        offtarget = joint_motion[~allowed] if target_index < 6 else joint_motion
        max_offtarget = float(np.max(offtarget)) if offtarget.size else 0.0
        if max_offtarget > args.max_offtarget_motion_rad:
            raise RuntimeError(
                f"{arm_name} arm off-target motion too large: {max_offtarget:.5f} rad. "
                "Stop policy rollout and check Piper enable/hold mode before continuing."
            )


def main() -> None:
    args = parse_args()
    robot = make_robot()
    try:
        if args.mode == "policy-first-step":
            policy_first_step(robot, args)
        elif args.mode == "soft-reset":
            soft_reset(robot, args)
        elif args.mode == "manual-delta":
            manual_delta(robot, args)
        elif args.mode == "manual-set":
            manual_set(robot, args)
        else:
            raise ValueError(args.mode)
    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
