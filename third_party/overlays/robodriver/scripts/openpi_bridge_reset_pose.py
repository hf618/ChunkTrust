#!/usr/bin/env python3
"""Save/show/reset an AgileX ALOHA reset pose through the RoboDriver Dora bridge."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from openpi_bridge_smoke_action import (  # noqa: E402
    action_dict_from_left_right,
    get_complete_observation,
    make_robot,
    observation_to_raw26,
    print_delta,
    raw26_to_left_right,
)


DEFAULT_POSE_PATH = Path(__file__).with_name("agilex_aloha_reset_pose.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["save-current", "show", "reset"], required=True)
    parser.add_argument("--pose-path", type=Path, default=DEFAULT_POSE_PATH)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.03)
    parser.add_argument("--max-gripper-step", type=float, default=0.005)
    parser.add_argument("--include-gripper", action="store_true")
    parser.add_argument("--tolerance-rad", type=float, default=0.003)
    parser.add_argument("--tolerance-gripper", type=float, default=0.003)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--hold-repeat", type=int, default=10)
    parser.add_argument("--stop-on-commanded-target", type=int, choices=[0, 1], default=1)
    parser.add_argument("--progress-style", choices=["compact", "verbose"], default="compact")
    parser.add_argument("--execute", type=int, choices=[0, 1], default=0)
    return parser.parse_args()


def pose_to_json(left: np.ndarray, right: np.ndarray) -> dict:
    return {
        "left": [float(v) for v in left],
        "right": [float(v) for v in right],
        "order": {
            "left": ["joint1_left", "joint2_left", "joint3_left", "joint4_left", "joint5_left", "joint6_left", "gripper_left"],
            "right": [
                "joint1_right",
                "joint2_right",
                "joint3_right",
                "joint4_right",
                "joint5_right",
                "joint6_right",
                "gripper_right",
            ],
        },
    }


def load_pose(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text())
    left = np.asarray(data["left"], dtype=np.float32)
    right = np.asarray(data["right"], dtype=np.float32)
    if left.shape != (7,) or right.shape != (7,):
        raise RuntimeError(f"Expected left/right shape (7,), got {left.shape}, {right.shape}")
    return left, right


def get_current_pose(robot) -> tuple[np.ndarray, np.ndarray]:
    observation = get_complete_observation(robot)
    return raw26_to_left_right(observation_to_raw26(observation))


def save_current(robot, args: argparse.Namespace) -> None:
    left, right = get_current_pose(robot)
    args.pose_path.write_text(json.dumps(pose_to_json(left, right), indent=2) + "\n")
    print(f"saved reset pose: {args.pose_path}")
    print(f"left={np.array2string(left, precision=5)}")
    print(f"right={np.array2string(right, precision=5)}")


def show_pose(args: argparse.Namespace) -> None:
    left, right = load_pose(args.pose_path)
    print(f"reset pose: {args.pose_path}")
    print(f"left={np.array2string(left, precision=5)}")
    print(f"right={np.array2string(right, precision=5)}")


def step_toward(current: np.ndarray, target: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    delta = target - current
    delta[:6] = np.clip(delta[:6], -args.max_joint_step_rad, args.max_joint_step_rad)
    if args.include_gripper:
        delta[6] = np.clip(delta[6], -args.max_gripper_step, args.max_gripper_step)
    else:
        delta[6] = 0.0
    return current + delta


def is_close(current: np.ndarray, target: np.ndarray, args: argparse.Namespace) -> bool:
    joint_close = bool(np.all(np.abs(current[:6] - target[:6]) <= args.tolerance_rad))
    if not args.include_gripper:
        return joint_close
    return joint_close and bool(abs(current[6] - target[6]) <= args.tolerance_gripper)


def max_joint_error(current: np.ndarray, target: np.ndarray) -> float:
    return float(np.max(np.abs(current[:6] - target[:6])))


def max_gripper_error(current: np.ndarray, target: np.ndarray) -> float:
    return float(abs(current[6] - target[6]))


def finish_compact_line(args: argparse.Namespace) -> None:
    if args.progress_style == "compact":
        print()


def reset_pose(robot, args: argparse.Namespace) -> None:
    target_left, target_right = load_pose(args.pose_path)
    period = 1.0 / args.hz if args.hz > 0 else 0.0
    print(f"reset target: {args.pose_path}; execute: {args.execute}")

    for step in range(args.max_steps):
        loop_start = time.perf_counter()
        current_left, current_right = get_current_pose(robot)
        if is_close(current_left, target_left, args) and is_close(current_right, target_right, args):
            finish_compact_line(args)
            print(f"reset pose reached from feedback at step {step}/{args.max_steps}")
            break

        next_left = step_toward(current_left, target_left, args)
        next_right = step_toward(current_right, target_right, args)
        command_reaches_target = is_close(next_left, target_left, args) and is_close(next_right, target_right, args)
        if command_reaches_target:
            next_left = target_left.copy()
            next_right = target_right.copy()
            if not args.include_gripper:
                next_left[6] = current_left[6]
                next_right[6] = current_right[6]

        if args.progress_style == "compact":
            line = (
                f"\rreset {step + 1:03d}/{args.max_steps:03d} | "
                f"cmd_dL={max_joint_error(current_left, next_left):.4f} "
                f"cmd_dR={max_joint_error(current_right, next_right):.4f} | "
                f"errL={max_joint_error(current_left, target_left):.4f} "
                f"errR={max_joint_error(current_right, target_right):.4f} | "
                f"gErrL={max_gripper_error(current_left, target_left):.4f} "
                f"gErrR={max_gripper_error(current_right, target_right):.4f}"
            )
            print(line, end="", flush=True)
        else:
            print(f"reset step {step + 1}/{args.max_steps}")
            print_delta("left", current_left, next_left)
            print_delta("right", current_right, next_right)

        if args.execute:
            robot.send_action(action_dict_from_left_right(next_left, next_right))

        if args.stop_on_commanded_target and command_reaches_target:
            finish_compact_line(args)
            print(f"reset target command sent at step {step + 1}/{args.max_steps}")
            break

        sleep_time = period - (time.perf_counter() - loop_start)
        if sleep_time > 0:
            time.sleep(sleep_time)

    final_left, final_right = get_current_pose(robot)
    print("final reset error:")
    print_delta("left-error", final_left, target_left)
    print_delta("right-error", final_right, target_right)

    if args.execute:
        for index in range(max(1, args.hold_repeat)):
            robot.send_action(action_dict_from_left_right(final_left, final_right))
            print(f"hold reset pose {index + 1}/{args.hold_repeat}")
            if period > 0 and index + 1 < args.hold_repeat:
                time.sleep(period)


def main() -> None:
    args = parse_args()
    if args.mode == "show":
        show_pose(args)
        return

    robot = make_robot()
    try:
        if args.mode == "save-current":
            save_current(robot, args)
        elif args.mode == "reset":
            reset_pose(robot, args)
        else:
            raise ValueError(args.mode)
    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
