#!/usr/bin/env python3
"""Safe chunk rollout bridge for OpenPI pi05 and AgileX ALOHA Dora."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from openpi_bridge_common import (  # noqa: E402
    PROMPT as DEFAULT_PROMPT,
    action_dict_from_left_right,
    get_complete_observation,
    make_robot,
    observation_to_policy_input,
    observation_to_raw26,
    print_delta,
    raw26_to_left_right,
)
from eval_video_recorder import add_recording_args, create_eval_video_recorder  # noqa: E402
from eval_episode_npz_logger import EvalEpisodeNpzLogger  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--state-format", choices=["raw26"], default="raw26")
    parser.add_argument("--rollout-mode", choices=["chunk"], default="chunk")
    parser.add_argument("--chunk-prefix-steps", type=int, default=5)
    parser.add_argument("--episode-max-action-steps", type=int, default=25)
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--policy-alpha", type=float, default=1.0)
    parser.add_argument("--action-ema", type=float, default=0.0)
    parser.add_argument("--max-joint-delta-rad", type=float, default=0.03)
    parser.add_argument("--max-gripper-delta", type=float, default=0.005)
    parser.add_argument("--allow-dual-arm", action="store_true")
    parser.add_argument("--allow-gripper", action="store_true")
    parser.add_argument("--single-arm-only", choices=["left", "right"], default=None)
    parser.add_argument("--post-check-sec", type=float, default=0.0)
    parser.add_argument("--ctrl-c-soft-reset", type=int, choices=[0, 1], default=1)
    parser.add_argument("--soft-reset-repeat", type=int, default=10)
    parser.add_argument("--soft-reset-sleep-sec", type=float, default=0.05)
    parser.add_argument("--progress-style", choices=["compact", "verbose"], default=None)
    parser.add_argument("--motion-watchdog", type=int, choices=[0, 1], default=1)
    parser.add_argument("--motion-watchdog-steps", type=int, default=20)
    parser.add_argument("--motion-command-threshold-rad", type=float, default=0.01)
    parser.add_argument("--motion-observed-threshold-rad", type=float, default=0.003)
    add_recording_args(parser)
    parser.add_argument("--execute", type=int, choices=[0, 1], default=0)
    return parser.parse_args(argv)


def clip_step(current: np.ndarray, desired: np.ndarray, *, max_joint_delta: float, max_gripper_delta: float) -> np.ndarray:
    delta = desired - current
    delta[:6] = np.clip(delta[:6], -max_joint_delta, max_joint_delta)
    delta[6] = np.clip(delta[6], -max_gripper_delta, max_gripper_delta)
    return current + delta


def target_from_policy_step(
    current_left: np.ndarray,
    current_right: np.ndarray,
    policy_step: np.ndarray,
    prev_left: np.ndarray | None,
    prev_right: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    desired_left = policy_step[0:7].astype(np.float32)
    desired_right = policy_step[7:14].astype(np.float32)

    desired_left = current_left + args.policy_alpha * (desired_left - current_left)
    desired_right = current_right + args.policy_alpha * (desired_right - current_right)

    if args.action_ema > 0 and prev_left is not None and prev_right is not None:
        ema = args.action_ema
        desired_left = ema * prev_left + (1.0 - ema) * desired_left
        desired_right = ema * prev_right + (1.0 - ema) * desired_right

    target_left = clip_step(
        current_left,
        desired_left,
        max_joint_delta=args.max_joint_delta_rad,
        max_gripper_delta=args.max_gripper_delta,
    )
    target_right = clip_step(
        current_right,
        desired_right,
        max_joint_delta=args.max_joint_delta_rad,
        max_gripper_delta=args.max_gripper_delta,
    )

    if args.single_arm_only == "left":
        target_right = current_right.copy()
    elif args.single_arm_only == "right":
        target_left = current_left.copy()
    elif not args.allow_dual_arm:
        target_left = current_left.copy()

    if not args.allow_gripper:
        target_left[6] = current_left[6]
        target_right[6] = current_right[6]

    return target_left, target_right


def progress_style(args: argparse.Namespace) -> str:
    if args.progress_style is not None:
        return args.progress_style
    return "compact" if args.execute else "verbose"


def max_joint_delta(current: np.ndarray, target: np.ndarray) -> float:
    return float(np.max(np.abs(target[:6] - current[:6])))


def gripper_delta(current: np.ndarray, target: np.ndarray) -> float:
    return float(target[6] - current[6])


def total_replans(args: argparse.Namespace) -> int:
    chunk = max(1, args.chunk_prefix_steps)
    return (args.episode_max_action_steps + chunk - 1) // chunk


def print_replan_header(
    *,
    replan_index: int,
    replan_total: int,
    actions_shape: tuple[int, ...],
    prefix_steps: int,
    executed_steps: int,
    args: argparse.Namespace,
    style: str,
) -> None:
    if style == "compact":
        print(
            f"\n[replan {replan_index:02d}/{replan_total:02d}] "
            f"policy_chunk={actions_shape}, taking={prefix_steps}, "
            f"total_step={executed_steps:03d}/{args.episode_max_action_steps}"
        )
    else:
        print(f"chunk actions shape: {actions_shape}; taking {prefix_steps}; execute: {args.execute}")


def print_step_progress(
    *,
    replan_index: int,
    replan_total: int,
    chunk_index: int,
    prefix_steps: int,
    executed_steps: int,
    current_left: np.ndarray,
    current_right: np.ndarray,
    target_left: np.ndarray,
    target_right: np.ndarray,
    args: argparse.Namespace,
    style: str,
) -> None:
    if style == "compact":
        line = (
            f"\rreplan {replan_index:02d}/{replan_total:02d} | "
            f"chunk {chunk_index + 1:02d}/{prefix_steps:02d} | "
            f"step {executed_steps + 1:04d}/{args.episode_max_action_steps:04d} | "
            f"dL={max_joint_delta(current_left, target_left):.4f} "
            f"dR={max_joint_delta(current_right, target_right):.4f} | "
            f"gL={gripper_delta(current_left, target_left):+.4f} "
            f"gR={gripper_delta(current_right, target_right):+.4f} | "
            f"{args.hz:g}Hz"
        )
        print(line, end="", flush=True)
        return

    print(f"step {executed_steps + 1}/{args.episode_max_action_steps} chunk_index={chunk_index}")
    print_delta("left", current_left, target_left)
    print_delta("right", current_right, target_right)


def finish_compact_line(style: str) -> None:
    if style == "compact":
        print()


def observed_joint_motion(start_left: np.ndarray, start_right: np.ndarray, now_left: np.ndarray, now_right: np.ndarray) -> float:
    left_motion = float(np.max(np.abs(now_left[:6] - start_left[:6])))
    right_motion = float(np.max(np.abs(now_right[:6] - start_right[:6])))
    return max(left_motion, right_motion)


def main() -> None:
    args = parse_args()
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    robot = make_robot()
    recorder = create_eval_video_recorder(args, robot, rollout_kind="safe")
    if recorder is not None:
        recorder.start()
    npz_logger = EvalEpisodeNpzLogger(recorder=recorder, rollout_kind="safe", args=args) if recorder is not None else None
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    period = 1.0 / args.hz if args.hz > 0 else 0.0
    executed_steps = 0
    replan_index = 0
    replan_total = total_replans(args)
    style = progress_style(args)
    prev_left: np.ndarray | None = None
    prev_right: np.ndarray | None = None
    watchdog_start_left: np.ndarray | None = None
    watchdog_start_right: np.ndarray | None = None
    watchdog_commanded = False

    interrupted = False
    recorder_status = "error"
    try:
        print(f"task prompt: {args.prompt}")
        while executed_steps < args.episode_max_action_steps:
            observation = get_complete_observation(robot)
            raw26 = observation_to_raw26(observation)
            policy_input = observation_to_policy_input(observation, raw26)
            policy_input["prompt"] = args.prompt
            result = client.infer(policy_input)
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] < 14:
                raise RuntimeError(f"Expected policy actions [horizon, >=14], got {actions.shape}")

            prefix_steps = min(args.chunk_prefix_steps, actions.shape[0], args.episode_max_action_steps - executed_steps)
            replan_index += 1
            if npz_logger is not None:
                npz_logger.log_replan(
                    replan_idx=replan_index - 1,
                    env_step_idx=executed_steps,
                    selected_k=prefix_steps,
                    monotonic_timestamp_ns=time.monotonic_ns(),
                )
            print_replan_header(
                replan_index=replan_index,
                replan_total=replan_total,
                actions_shape=actions.shape,
                prefix_steps=prefix_steps,
                executed_steps=executed_steps,
                args=args,
                style=style,
            )

            for chunk_index in range(prefix_steps):
                loop_start = time.perf_counter()
                step_start_time_ns = time.time_ns()
                step_start_monotonic_ns = time.monotonic_ns()
                current_obs = get_complete_observation(robot)
                current_raw26 = observation_to_raw26(current_obs)
                current_left, current_right = raw26_to_left_right(current_raw26)
                if watchdog_start_left is None or watchdog_start_right is None:
                    watchdog_start_left = current_left.copy()
                    watchdog_start_right = current_right.copy()
                target_left, target_right = target_from_policy_step(
                    current_left,
                    current_right,
                    actions[chunk_index],
                    prev_left,
                    prev_right,
                    args,
                )
                print_step_progress(
                    replan_index=replan_index,
                    replan_total=replan_total,
                    chunk_index=chunk_index,
                    prefix_steps=prefix_steps,
                    executed_steps=executed_steps,
                    current_left=current_left,
                    current_right=current_right,
                    target_left=target_left,
                    target_right=target_right,
                    args=args,
                    style=style,
                )

                if args.execute:
                    robot.send_action(action_dict_from_left_right(target_left, target_right))

                if npz_logger is not None:
                    npz_logger.log_step(
                        step_index=executed_steps,
                        replan_index=replan_index,
                        chunk_index=chunk_index,
                        step_start_time_ns=step_start_time_ns,
                        step_start_monotonic_ns=step_start_monotonic_ns,
                        state_raw26_before=current_raw26,
                        policy_action_lr14=actions[chunk_index, :14],
                        executed_action_lr14=np.concatenate([target_left, target_right]),
                        execute_enabled=bool(args.execute),
                    )

                commanded_motion = max(max_joint_delta(current_left, target_left), max_joint_delta(current_right, target_right))
                watchdog_commanded = watchdog_commanded or commanded_motion >= args.motion_command_threshold_rad
                prev_left = target_left
                prev_right = target_right
                executed_steps += 1

                sleep_time = period - (time.perf_counter() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                if (
                    args.execute
                    and args.motion_watchdog
                    and watchdog_commanded
                    and executed_steps >= args.motion_watchdog_steps
                    and watchdog_start_left is not None
                    and watchdog_start_right is not None
                ):
                    check_obs = get_complete_observation(robot)
                    check_left, check_right = raw26_to_left_right(observation_to_raw26(check_obs))
                    observed = observed_joint_motion(watchdog_start_left, watchdog_start_right, check_left, check_right)
                    if observed < args.motion_observed_threshold_rad:
                        finish_compact_line(style)
                        raise RuntimeError(
                            "Motion watchdog stopped rollout: commands were sent, but observed joint motion was "
                            f"{observed:.5f} rad after {executed_steps} steps. Check Piper enable/control mode, "
                            "Dora piper node action input, and CAN wiring before running policy again."
                        )
                    watchdog_start_left = check_left.copy()
                    watchdog_start_right = check_right.copy()
                    watchdog_commanded = False
                if executed_steps >= args.episode_max_action_steps:
                    break
            finish_compact_line(style)

        if args.post_check_sec > 0:
            time.sleep(args.post_check_sec)
        final_obs = get_complete_observation(robot)
        final_left, final_right = raw26_to_left_right(observation_to_raw26(final_obs))
        print("final state sample:")
        print(f"left={np.array2string(final_left, precision=5)}")
        print(f"right={np.array2string(final_right, precision=5)}")
        recorder_status = "completed"
    except KeyboardInterrupt:
        interrupted = True
        recorder_status = "interrupted"
        finish_compact_line(style)
        print("KeyboardInterrupt received.")
        if args.ctrl_c_soft_reset and args.execute:
            soft_reset_hold_current(robot, args)
        elif args.ctrl_c_soft_reset and not args.execute:
            print("ctrl-c soft reset skipped because execute=0")
        else:
            print("ctrl-c soft reset disabled")
    finally:
        if recorder is not None:
            recorder.stop(status=recorder_status)
        if npz_logger is not None:
            try:
                npz_path = npz_logger.save(status=recorder_status)
                if npz_path is not None:
                    print(f"episode npz saved: {npz_path}")
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: failed to save episode_data.npz: {exc}")
        if robot.is_connected:
            robot.disconnect()


def soft_reset_hold_current(robot, args: argparse.Namespace) -> None:
    print(
        f"ctrl-c soft reset: holding current pose "
        f"repeat={args.soft_reset_repeat}, sleep={args.soft_reset_sleep_sec}"
    )
    for index in range(max(1, args.soft_reset_repeat)):
        observation = get_complete_observation(robot)
        current_left, current_right = raw26_to_left_right(observation_to_raw26(observation))
        robot.send_action(action_dict_from_left_right(current_left, current_right))
        print(f"ctrl-c soft reset sent {index + 1}/{args.soft_reset_repeat}")
        if args.soft_reset_sleep_sec > 0 and index + 1 < args.soft_reset_repeat:
            time.sleep(args.soft_reset_sleep_sec)


if __name__ == "__main__":
    main()
