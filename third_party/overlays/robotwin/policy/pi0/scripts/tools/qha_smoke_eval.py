#!/usr/bin/env python3
"""Lightweight QHA smoke eval for a trained pi0 checkpoint.

This script exercises the current PI0 deploy/runtime path:
- load checkpoint through `pi_model.PI0`
- build synthetic observations
- run one or two inference passes
- optionally feed a few executed actions back into QHA history
- validate basic QHA output sanity for quick checkpoint checks
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


THIS_FILE = Path(__file__).resolve()
PI0_ROOT = THIS_FILE.parents[2]  # .../policy/pi0
if str(PI0_ROOT) not in sys.path:
    sys.path.append(str(PI0_ROOT))

from pi_model import PI0  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QHA smoke eval for pi0 checkpoints.")
    parser.add_argument("--train-config-name", type=str, default="pi0_base_aloha_robotwin_lora")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--checkpoint-id", type=int, required=True)
    parser.add_argument("--pi0-step", type=int, default=50, help="Max exec length cap used by the PI0 wrapper.")
    parser.add_argument(
        "--use-qha",
        type=str,
        choices=("auto", "true", "false"),
        default="auto",
        help="Force QHA on/off at eval time, or auto-detect from checkpoint metadata.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", type=str, default="rank the blocks by height")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--state-dim", type=int, default=14)
    parser.add_argument("--passes", type=int, default=2, help="How many inference passes to run.")
    parser.add_argument(
        "--record-actions",
        type=int,
        default=3,
        help="How many predicted actions to feed back into QHA history between passes.",
    )
    parser.add_argument(
        "--posterior-sum-tol",
        type=float,
        default=1e-2,
        help="Tolerance for posterior sum sanity check.",
    )
    parser.add_argument(
        "--print-full-posterior",
        action="store_true",
        help="Print the full posterior vector instead of only summary stats.",
    )
    return parser.parse_args()


def _make_random_observation(
    rng: np.random.Generator,
    *,
    image_size: int,
    state_dim: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    rgb_list = [
        rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8),
        rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8),
        rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8),
    ]
    state = rng.standard_normal((state_dim,)).astype(np.float32)
    return rgb_list, state


def _to_optional_bool(raw: str) -> bool | None:
    lowered = raw.strip().lower()
    if lowered == "auto":
        return None
    return lowered == "true"


def _validate_plan(plan: dict, *, posterior_sum_tol: float) -> dict[str, np.ndarray]:
    actions = np.asarray(plan["actions"])
    if actions.ndim != 2 or actions.shape[0] <= 0:
        raise AssertionError(f"Invalid actions shape: {actions.shape}")

    candidate_horizons = np.asarray(plan["qha_candidate_horizons"])
    if candidate_horizons.ndim != 1 or candidate_horizons.size <= 0:
        raise AssertionError(f"Invalid qha_candidate_horizons shape: {candidate_horizons.shape}")

    posterior = np.asarray(plan["qha_posterior"], dtype=np.float32)
    if posterior.ndim != 1 or posterior.shape[0] != candidate_horizons.shape[0]:
        raise AssertionError(
            f"Posterior/candidate mismatch: posterior={posterior.shape}, candidates={candidate_horizons.shape}"
        )
    if not np.all(np.isfinite(posterior)):
        raise AssertionError("qha_posterior contains non-finite values.")
    if abs(float(posterior.sum()) - 1.0) >= posterior_sum_tol:
        raise AssertionError(
            f"qha_posterior sum deviates too much from 1: sum={float(posterior.sum()):.6f}, "
            f"tol={posterior_sum_tol:.6f}"
        )

    exec_length = int(plan["exec_length"])
    max_exec_length = int(plan["max_exec_length"])
    if exec_length < 1 or exec_length > max_exec_length:
        raise AssertionError(f"Invalid exec length: exec_length={exec_length}, max_exec_length={max_exec_length}")

    q_mix = plan.get("qha_selector_q_mix")
    if q_mix is not None:
        q_mix = np.asarray(q_mix, dtype=np.float32)
        if q_mix.shape != posterior.shape:
            raise AssertionError(f"qha_selector_q_mix shape mismatch: {q_mix.shape} vs {posterior.shape}")

    return {
        "actions": actions,
        "candidate_horizons": candidate_horizons,
        "posterior": posterior,
        "q_mix": None if q_mix is None else q_mix,
    }


def _print_pass_summary(
    pass_idx: int,
    plan: dict,
    arrays: dict[str, np.ndarray],
    *,
    print_full_posterior: bool,
) -> None:
    print(f"\n[PASS {pass_idx}]")
    print(f"actions.shape: {arrays['actions'].shape}")
    print(f"exec_length: {int(plan['exec_length'])}")
    print(f"max_exec_length: {int(plan['max_exec_length'])}")
    print(f"qha_candidate_horizons: {arrays['candidate_horizons']}")
    print(f"qha_expected_horizon: {plan.get('qha_expected_horizon')}")
    print(f"qha_argmax_horizon: {plan.get('qha_argmax_horizon')}")
    print(f"qha_posterior_sum: {float(arrays['posterior'].sum()):.6f}")
    print(f"qha_exec_mode: {plan.get('qha_exec_mode')}")
    if print_full_posterior:
        print(f"qha_posterior: {arrays['posterior']}")
    if arrays["q_mix"] is not None:
        print(f"qha_selector_q_mix: {arrays['q_mix']}")
    runtime_scores = plan.get("qha_runtime_scores")
    if runtime_scores is not None:
        print(f"qha_runtime_scores: {np.asarray(runtime_scores, dtype=np.float32)}")


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)
    model = PI0(
        train_config_name=args.train_config_name,
        model_name=args.exp_name,
        checkpoint_id=args.checkpoint_id,
        pi0_step=args.pi0_step,
        use_qha=_to_optional_bool(args.use_qha),
    )
    model.set_language(args.prompt)

    for pass_idx in range(1, max(1, int(args.passes)) + 1):
        rgb_list, state = _make_random_observation(
            rng,
            image_size=int(args.image_size),
            state_dim=int(args.state_dim),
        )
        model.update_observation_window(rgb_list, state)
        plan = model.get_action_plan()
        arrays = _validate_plan(plan, posterior_sum_tol=float(args.posterior_sum_tol))
        _print_pass_summary(pass_idx, plan, arrays, print_full_posterior=bool(args.print_full_posterior))

        if pass_idx < int(args.passes):
            record_n = min(int(args.record_actions), int(plan["exec_length"]), arrays["actions"].shape[0])
            for i in range(record_n):
                model.record_executed_action(arrays["actions"][i])

    print("\nQHA smoke eval passed.")


if __name__ == "__main__":
    main()
