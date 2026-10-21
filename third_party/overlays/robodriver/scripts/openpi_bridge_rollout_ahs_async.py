#!/usr/bin/env python3
"""Async horizon-aware rollout bridge for OpenPI pi05 and AgileX ALOHA Dora."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_episode_npz_logger import EvalEpisodeNpzLogger  # noqa: E402
from eval_video_recorder import add_recording_args, create_eval_video_recorder  # noqa: E402
from openpi_bridge_rollout_ahs import (  # noqa: E402
    DEFAULT_PROMPT,
    AHSChoice,
    AHSRuntimeSelector,
    TestTimeHorizonSelector,
    action_dict_from_left_right,
    ahs_log_fields,
    beta_ab_from_snapshot,
    beta_candidates_for_choice,
    beta_state_snapshot,
    choose_ahs_exec_length,
    finish_compact_line,
    get_complete_observation,
    gripper_delta,
    make_robot,
    max_joint_delta,
    observed_joint_motion,
    observation_to_policy_input,
    observation_to_raw26,
    parse_candidates,
    print_ahs_details,
    print_delta,
    progress_style,
    raw26_to_left_right,
    soft_reset_hold_current,
    target_from_policy_step,
    total_replans,
)


@dataclass
class AsyncPlan:
    actions: np.ndarray
    choice: AHSChoice
    requested_at_step: int
    requested_monotonic_ns: int
    infer_latency_sec: float
    beta_before_horizon: dict[int, tuple[float, float]]
    beta_after_horizon: dict[int, tuple[float, float]]
    beta_before_qha: dict[int, tuple[float, float]]
    beta_after_qha: dict[int, tuple[float, float]]


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--ahs-candidates", default="10,20,30,40,50")
    parser.add_argument(
        "--ahs-exec-mode",
        choices=["expected_round", "thompson", "qmix_expected_round"],
        default="expected_round",
    )
    parser.add_argument("--ahs-expected-temp", type=float, default=1.0)
    parser.add_argument("--ahs-ts-epsilon", type=float, default=0.05)
    parser.add_argument("--ahs-ts-seed", type=int, default=0)
    parser.add_argument("--ahs-ts-update-mode", choices=["independent", "kernel_forget"], default="kernel_forget")
    parser.add_argument("--ahs-ts-kernel-bandwidth", type=float, default=10.0)
    parser.add_argument("--ahs-ts-forget-rho", type=float, default=0.99)
    parser.add_argument("--ahs-ts-update-eta", type=float, default=1.0)
    parser.add_argument("--ahs-prior-gamma", type=float, default=1.0)
    parser.add_argument("--ahs-metadata-policy", choices=["fallback", "require"], default="fallback")
    parser.add_argument("--horizon-intra-cut-t", type=float, default=0.25)
    parser.add_argument("--horizon-intra-z-mode", choices=["tau0_rms", "window_rms", "trend"], default="window_rms")
    parser.add_argument("--horizon-intra-tau-window", type=int, default=3)
    parser.add_argument("--horizon-intra-alpha", type=float, default=4.0)
    parser.add_argument("--horizon-inter-use-speed-uniformity", type=int, choices=[0, 1], default=1)
    parser.add_argument("--horizon-mix-mode", choices=["intra", "inter", "both"], default="both")
    parser.add_argument("--horizon-inter-window", type=int, default=60)
    parser.add_argument("--horizon-inter-weight", type=float, default=0.5)
    parser.add_argument("--horizon-inter-beta", type=float, default=4.0)
    parser.add_argument("--horizon-inter-fallback", choices=["ehh_only", "neutral"], default="ehh_only")
    parser.add_argument("--async-prefetch-steps", type=int, default=20)
    parser.add_argument("--async-infer-timeout-sec", type=float, default=5.0)
    parser.add_argument("--async-hold-on-empty", type=int, choices=[0, 1], default=1)
    add_recording_args(parser)
    parser.add_argument("--execute", type=int, choices=[0, 1], default=0)
    return parser.parse_args()


def make_ahs_selectors(args: argparse.Namespace, ahs_candidates: list[int]) -> tuple[AHSRuntimeSelector, TestTimeHorizonSelector]:
    ahs_selector = AHSRuntimeSelector(
        candidate_horizons=ahs_candidates,
        exec_mode=args.ahs_exec_mode,
        expected_temp=args.ahs_expected_temp,
        ts_epsilon=args.ahs_ts_epsilon,
        ts_seed=args.ahs_ts_seed,
        ts_update_mode=args.ahs_ts_update_mode,
        ts_kernel_bandwidth=args.ahs_ts_kernel_bandwidth,
        ts_forget_rho=args.ahs_ts_forget_rho,
        ts_update_eta=args.ahs_ts_update_eta,
        prior_gamma=args.ahs_prior_gamma,
    )
    horizon_selector = TestTimeHorizonSelector(
        candidate_horizons=ahs_candidates,
        exec_mode=args.ahs_exec_mode,
        expected_temp=args.ahs_expected_temp,
        ts_epsilon=args.ahs_ts_epsilon,
        ts_seed=args.ahs_ts_seed,
        ts_update_mode=args.ahs_ts_update_mode,
        ts_kernel_bandwidth=args.ahs_ts_kernel_bandwidth,
        ts_forget_rho=args.ahs_ts_forget_rho,
        ts_update_eta=args.ahs_ts_update_eta,
        intra_cut_t=args.horizon_intra_cut_t,
        intra_z_mode=args.horizon_intra_z_mode,
        intra_tau_window=args.horizon_intra_tau_window,
        intra_alpha=args.horizon_intra_alpha,
        inter_use_speed_uniformity=bool(args.horizon_inter_use_speed_uniformity),
        mix_mode=args.horizon_mix_mode,
        inter_window=args.horizon_inter_window,
        inter_weight=args.horizon_inter_weight,
        inter_beta=args.horizon_inter_beta,
        inter_fallback=args.horizon_inter_fallback,
    )
    return ahs_selector, horizon_selector


def build_policy_input(robot: Any, args: argparse.Namespace) -> dict[str, Any]:
    observation = get_complete_observation(robot)
    raw26 = observation_to_raw26(observation)
    policy_input = observation_to_policy_input(observation, raw26)
    policy_input["prompt"] = args.prompt
    return policy_input


def infer_plan(
    *,
    client: WebsocketClientPolicy,
    policy_input: dict[str, Any],
    requested_at_step: int,
    requested_monotonic_ns: int,
    remaining_steps: int,
    selector_lock: threading.Lock,
    ahs_selector: AHSRuntimeSelector,
    horizon_selector: TestTimeHorizonSelector,
    args: argparse.Namespace,
) -> AsyncPlan:
    infer_start = time.perf_counter()
    result = client.infer(policy_input)
    infer_latency = time.perf_counter() - infer_start
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] < 14:
        raise RuntimeError(f"Expected policy actions [horizon, >=14], got {actions.shape}")

    with selector_lock:
        beta_before_horizon = beta_state_snapshot(horizon_selector)
        beta_before_qha = beta_state_snapshot(ahs_selector)
        choice = choose_ahs_exec_length(
            result,
            actions=actions,
            actions_shape=actions.shape,
            remaining_steps=remaining_steps,
            selector=ahs_selector,
            horizon_selector=horizon_selector,
            args=args,
        )
        beta_after_horizon = beta_state_snapshot(horizon_selector)
        beta_after_qha = beta_state_snapshot(ahs_selector)

    return AsyncPlan(
        actions=actions,
        choice=choice,
        requested_at_step=int(requested_at_step),
        requested_monotonic_ns=int(requested_monotonic_ns),
        infer_latency_sec=float(infer_latency),
        beta_before_horizon=beta_before_horizon,
        beta_after_horizon=beta_after_horizon,
        beta_before_qha=beta_before_qha,
        beta_after_qha=beta_after_qha,
    )


def print_replan_header_async(
    *,
    replan_index: int,
    replan_total: int,
    actions_shape: tuple[int, ...],
    prefix_steps: int,
    executed_steps: int,
    choice: AHSChoice,
    args: argparse.Namespace,
    style: str,
    infer_latency_sec: float,
) -> None:
    if style == "compact":
        print(
            f"\n[replan {replan_index:02d}/{replan_total:02d}] "
            f"policy_chunk={actions_shape}, taking={prefix_steps}, "
            f"total_step={executed_steps:03d}/{args.episode_max_action_steps}, "
            f"infer={infer_latency_sec:.3f}s"
        )
    else:
        print(
            f"chunk actions shape: {actions_shape}; taking {prefix_steps}; "
            f"source={choice.source}; mode={choice.mode}; infer={infer_latency_sec:.3f}s; execute: {args.execute}"
        )
    if style == "compact":
        print(f"horizon: source={choice.source}, mode={choice.mode}, ahs_exec={choice.exec_length}")
    print_ahs_details(choice)


def make_ahs_replan_log(choice: AHSChoice, ahs_candidates: list[int], prefix_steps: int, plan: AsyncPlan) -> tuple[dict[str, Any], dict[str, Any]]:
    ahs_step_log = ahs_log_fields(choice, executed_length=prefix_steps)
    beta_candidates = beta_candidates_for_choice(choice, ahs_candidates)
    if choice.source == "denoise_trace":
        beta_before = beta_ab_from_snapshot(plan.beta_before_horizon, beta_candidates)
        beta_after = beta_ab_from_snapshot(plan.beta_after_horizon, beta_candidates)
    elif choice.source == "qha":
        beta_before = beta_ab_from_snapshot(plan.beta_before_qha, beta_candidates)
        beta_after = beta_ab_from_snapshot(plan.beta_after_qha, beta_candidates)
    else:
        beta_before = np.full((int(beta_candidates.size), 2), np.nan, dtype=np.float32)
        beta_after = np.full((int(beta_candidates.size), 2), np.nan, dtype=np.float32)

    ahs_replan_log = dict(ahs_step_log)
    ahs_replan_log.update(
        {
            "beta_candidates": beta_candidates,
            "beta_ab_before": beta_before,
            "beta_ab_after": beta_after,
        }
    )
    return ahs_step_log, ahs_replan_log


def print_step_progress_async(
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


def update_episode_metadata_async(recorder: Any, stats: dict[str, Any]) -> None:
    episode_dir = getattr(recorder, "episode_dir", None)
    if episode_dir is None:
        return
    metadata_path = Path(episode_dir) / "episode_metadata.json"
    if not metadata_path.exists():
        return
    try:
        data = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return
    data["async"] = stats
    metadata_path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def main() -> None:
    args = parse_args()
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    ahs_candidates = parse_candidates(args.ahs_candidates)
    ahs_selector, horizon_selector = make_ahs_selectors(args, ahs_candidates)
    selector_lock = threading.Lock()

    robot = make_robot()
    recorder = create_eval_video_recorder(args, robot, rollout_kind="ahs_async")
    if recorder is not None:
        recorder.start()
    npz_logger = EvalEpisodeNpzLogger(recorder=recorder, rollout_kind="ahs_async", args=args) if recorder is not None else None

    client = WebsocketClientPolicy(host=args.host, port=args.port)
    executor = ThreadPoolExecutor(max_workers=1)
    period = 1.0 / args.hz if args.hz > 0 else 0.0
    executed_steps = 0
    replan_index = 0
    replan_total = total_replans(args, ahs_candidates)
    style = progress_style(args)
    prev_left: np.ndarray | None = None
    prev_right: np.ndarray | None = None
    last_target_left: np.ndarray | None = None
    last_target_right: np.ndarray | None = None
    watchdog_start_left: np.ndarray | None = None
    watchdog_start_right: np.ndarray | None = None
    watchdog_commanded = False
    recorder_status = "error"
    future: Future[AsyncPlan] | None = None
    future_requested_step = -1
    future_requested_perf = 0.0
    hold_steps_total = 0
    pending_hold_steps = 0
    async_events: list[dict[str, Any]] = []

    def submit_replan(requested_step: int) -> Future[AsyncPlan]:
        policy_input = build_policy_input(robot, args)
        requested_monotonic_ns = time.monotonic_ns()
        remaining = max(1, args.episode_max_action_steps - requested_step)
        print(f"\nasync replan requested at step={requested_step}", flush=True)
        return executor.submit(
            infer_plan,
            client=client,
            policy_input=policy_input,
            requested_at_step=requested_step,
            requested_monotonic_ns=requested_monotonic_ns,
            remaining_steps=remaining,
            selector_lock=selector_lock,
            ahs_selector=ahs_selector,
            horizon_selector=horizon_selector,
            args=args,
        )

    def mark_future_submitted(requested_step: int) -> None:
        nonlocal future, future_requested_step, future_requested_perf
        future = submit_replan(requested_step)
        future_requested_step = requested_step
        future_requested_perf = time.perf_counter()

    def consume_ready_future() -> AsyncPlan | None:
        nonlocal future, future_requested_step, future_requested_perf
        if future is None or not future.done():
            return None
        try:
            plan = future.result()
        except Exception as exc:  # noqa: BLE001
            finish_compact_line(style)
            raise RuntimeError(f"Async replan failed: {exc}") from exc
        future = None
        future_requested_step = -1
        future_requested_perf = 0.0
        return plan

    def check_async_timeout() -> None:
        if future is None or future.done() or args.async_infer_timeout_sec <= 0:
            return
        waited = time.perf_counter() - future_requested_perf
        if waited > args.async_infer_timeout_sec:
            finish_compact_line(style)
            raise RuntimeError(
                "Async replan timed out: "
                f"requested_at_step={future_requested_step}, waited={waited:.3f}s, "
                f"timeout={args.async_infer_timeout_sec:.3f}s"
            )

    def activate_plan(plan: AsyncPlan) -> tuple[int, dict[str, Any]]:
        nonlocal replan_index, pending_hold_steps
        prefix_steps = min(plan.choice.exec_length, plan.actions.shape[0], args.episode_max_action_steps - executed_steps)
        replan_index += 1
        print_replan_header_async(
            replan_index=replan_index,
            replan_total=replan_total,
            actions_shape=plan.actions.shape,
            prefix_steps=prefix_steps,
            executed_steps=executed_steps,
            choice=plan.choice,
            args=args,
            style=style,
            infer_latency_sec=plan.infer_latency_sec,
        )
        ahs_step_log, ahs_replan_log = make_ahs_replan_log(plan.choice, ahs_candidates, prefix_steps, plan)
        if npz_logger is not None:
            npz_logger.log_replan(
                replan_idx=replan_index - 1,
                env_step_idx=executed_steps,
                selected_k=prefix_steps,
                monotonic_timestamp_ns=plan.requested_monotonic_ns,
                ahs=ahs_replan_log,
            )
            npz_logger.log_async_event(
                requested_at_step=plan.requested_at_step,
                ready_at_step=executed_steps,
                hold_steps=pending_hold_steps,
                infer_latency_sec=plan.infer_latency_sec,
            )
        async_events.append(
            {
                "requested_at_step": int(plan.requested_at_step),
                "ready_at_step": int(executed_steps),
                "hold_steps": int(pending_hold_steps),
                "infer_latency_sec": float(plan.infer_latency_sec),
            }
        )
        print(f"async replan ready latency={plan.infer_latency_sec:.3f}s at step={executed_steps}", flush=True)
        pending_hold_steps = 0
        return prefix_steps, ahs_step_log

    try:
        print(f"task prompt: {args.prompt}")
        print(
            "AHS async config: "
            f"candidates={ahs_candidates}, mode={args.ahs_exec_mode}, "
            f"metadata_policy={args.ahs_metadata_policy}, "
            f"prefetch_steps={args.async_prefetch_steps}, hold_on_empty={args.async_hold_on_empty}"
        )

        initial_policy_input = build_policy_input(robot, args)
        initial_plan = infer_plan(
            client=client,
            policy_input=initial_policy_input,
            requested_at_step=0,
            requested_monotonic_ns=time.monotonic_ns(),
            remaining_steps=args.episode_max_action_steps,
            selector_lock=selector_lock,
            ahs_selector=ahs_selector,
            horizon_selector=horizon_selector,
            args=args,
        )
        active_plan = initial_plan
        prefix_steps, ahs_step_log = activate_plan(active_plan)
        chunk_index = 0

        while executed_steps < args.episode_max_action_steps:
            if active_plan is not None and chunk_index >= prefix_steps:
                with selector_lock:
                    horizon_selector.record_executed_actions(active_plan.actions[:prefix_steps, :14])
                finish_compact_line(style)
                active_plan = None  # type: ignore[assignment]
                ready_plan = consume_ready_future()
                if ready_plan is not None:
                    active_plan = ready_plan
                    prefix_steps, ahs_step_log = activate_plan(active_plan)
                    chunk_index = 0
                elif future is None:
                    mark_future_submitted(executed_steps)

            if active_plan is None:  # type: ignore[truthy-function]
                ready_plan = consume_ready_future()
                if ready_plan is not None:
                    active_plan = ready_plan
                    prefix_steps, ahs_step_log = activate_plan(active_plan)
                    chunk_index = 0
                    continue
                check_async_timeout()
                if not args.async_hold_on_empty:
                    time.sleep(0.001)
                    continue
                loop_start = time.perf_counter()
                hold_obs = get_complete_observation(robot)
                hold_left, hold_right = raw26_to_left_right(observation_to_raw26(hold_obs))
                target_left = last_target_left.copy() if last_target_left is not None else hold_left.copy()
                target_right = last_target_right.copy() if last_target_right is not None else hold_right.copy()
                if args.execute:
                    robot.send_action(action_dict_from_left_right(target_left, target_right))
                hold_steps_total += 1
                pending_hold_steps += 1
                if pending_hold_steps == 1 or pending_hold_steps % 10 == 0:
                    print(f"\nasync hold step {pending_hold_steps} waiting for replan...", flush=True)
                sleep_time = period - (time.perf_counter() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                continue

            remaining_in_chunk = prefix_steps - chunk_index
            if (
                future is None
                and remaining_in_chunk <= max(1, int(args.async_prefetch_steps))
                and executed_steps < args.episode_max_action_steps
            ):
                mark_future_submitted(executed_steps)

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
                active_plan.actions[chunk_index],
                prev_left,
                prev_right,
                args,
            )
            print_step_progress_async(
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
                    policy_action_lr14=active_plan.actions[chunk_index, :14],
                    executed_action_lr14=np.concatenate([target_left, target_right]),
                    execute_enabled=bool(args.execute),
                    ahs=ahs_step_log,
                )

            commanded_motion = max(max_joint_delta(current_left, target_left), max_joint_delta(current_right, target_right))
            watchdog_commanded = watchdog_commanded or commanded_motion >= args.motion_command_threshold_rad
            prev_left = target_left
            prev_right = target_right
            last_target_left = target_left
            last_target_right = target_right
            executed_steps += 1
            chunk_index += 1

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

        if args.post_check_sec > 0:
            time.sleep(args.post_check_sec)
        final_obs = get_complete_observation(robot)
        final_left, final_right = raw26_to_left_right(observation_to_raw26(final_obs))
        print("final state sample:")
        print(f"left={np.array2string(final_left, precision=5)}")
        print(f"right={np.array2string(final_right, precision=5)}")
        recorder_status = "completed"
    except KeyboardInterrupt:
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
        if future is not None and not future.done():
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        async_stats = {
            "prefetch_steps": int(args.async_prefetch_steps),
            "infer_timeout_sec": float(args.async_infer_timeout_sec),
            "hold_on_empty": bool(args.async_hold_on_empty),
            "hold_steps_total": int(hold_steps_total),
            "events": async_events,
        }
        if recorder is not None:
            recorder.stop(status=recorder_status)
            update_episode_metadata_async(recorder, async_stats)
        if npz_logger is not None:
            try:
                npz_path = npz_logger.save(status=recorder_status)
                if npz_path is not None:
                    print(f"episode npz saved: {npz_path}")
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: failed to save episode_data.npz: {exc}")
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
