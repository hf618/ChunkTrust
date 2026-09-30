#!/usr/bin/env python3
"""Lightweight QHA smoke eval for a trained pi05_horizon checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


THIS_FILE = Path(__file__).resolve()
PI05_HORIZON_ROOT = THIS_FILE.parents[2]
for path in (PI05_HORIZON_ROOT / "src", PI05_HORIZON_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pi_model import PI0  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config-name", default="pi05_base_aloha_robotwin_full")
    parser.add_argument("--checkpoint-root-tag", default="")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--checkpoint-id", type=int, required=True)
    parser.add_argument("--pi0-step", type=int, default=50)
    parser.add_argument("--use-qha", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--qha-eval-variant", default="native",
                        choices=(
                            "native",
                            "posterior_sparse_compromise",
                            "selector_dense_interpolation",
                            "legacy_horizon_adapter",
                        ))
    parser.add_argument("--qha-hybrid-prior-gamma", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default="rank the blocks by height")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--state-dim", type=int, default=14)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--record-actions", type=int, default=3)
    parser.add_argument("--posterior-sum-tol", type=float, default=1e-2)
    return parser.parse_args()


def _to_optional_bool(raw: str) -> bool | None:
    if raw == "auto":
        return None
    return raw == "true"


def _make_observation(rng: np.random.Generator, *, image_size: int, state_dim: int):
    images = [
        rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8),
        rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8),
        rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8),
    ]
    state = rng.standard_normal((state_dim,)).astype(np.float32)
    return images, state


def _validate_plan(plan: dict, *, posterior_sum_tol: float) -> dict[str, np.ndarray]:
    actions = np.asarray(plan["actions"])
    if actions.ndim != 2 or actions.shape[0] <= 0:
        raise AssertionError(f"Invalid actions shape: {actions.shape}")
    candidates = np.asarray(plan["qha_candidate_horizons"])
    posterior = np.asarray(plan["qha_posterior"], dtype=np.float32)
    if candidates.ndim != 1 or candidates.size <= 0:
        raise AssertionError(f"Invalid candidate shape: {candidates.shape}")
    if posterior.shape != candidates.shape:
        raise AssertionError(f"Posterior/candidate mismatch: {posterior.shape} vs {candidates.shape}")
    if not np.all(np.isfinite(posterior)):
        raise AssertionError("qha_posterior contains non-finite values")
    if abs(float(posterior.sum()) - 1.0) >= posterior_sum_tol:
        raise AssertionError(f"qha_posterior sum={float(posterior.sum()):.6f}")
    exec_length = int(plan["exec_length"])
    max_exec_length = int(plan["max_exec_length"])
    if exec_length < 1 or exec_length > max_exec_length:
        raise AssertionError(f"Invalid exec length: {exec_length}/{max_exec_length}")
    q_mix = plan.get("qha_selector_q_mix")
    if q_mix is not None:
        q_mix = np.asarray(q_mix, dtype=np.float32)
        if q_mix.shape != posterior.shape:
            raise AssertionError(f"qha_selector_q_mix shape mismatch: {q_mix.shape} vs {posterior.shape}")
    return {"actions": actions, "candidates": candidates, "posterior": posterior, "q_mix": q_mix}


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)
    model = PI0(
        train_config_name=args.train_config_name,
        model_name=args.exp_name,
        checkpoint_id=args.checkpoint_id,
        pi0_step=args.pi0_step,
        checkpoint_root_tag=args.checkpoint_root_tag or None,
        use_qha=_to_optional_bool(args.use_qha),
        qha_eval_variant=args.qha_eval_variant,
        qha_hybrid_prior_gamma_override=args.qha_hybrid_prior_gamma,
    )
    model.set_language(args.prompt)

    for pass_idx in range(1, max(1, args.passes) + 1):
        images, state = _make_observation(rng, image_size=args.image_size, state_dim=args.state_dim)
        model.update_observation_window(images, state)
        plan = model.get_action_plan()
        arrays = _validate_plan(plan, posterior_sum_tol=args.posterior_sum_tol)
        print(f"[PASS {pass_idx}]")
        print(f"actions.shape={arrays['actions'].shape}")
        print(f"exec_length={int(plan['exec_length'])}/{int(plan['max_exec_length'])}")
        print(f"qha_candidate_horizons={arrays['candidates']}")
        print(f"qha_expected_horizon={plan.get('qha_expected_horizon')}")
        print(f"qha_argmax_horizon={plan.get('qha_argmax_horizon')}")
        print(f"qha_posterior_sum={float(arrays['posterior'].sum()):.6f}")
        print(f"qha_exec_mode={plan.get('qha_exec_mode')}")
        if arrays["q_mix"] is not None:
            print(f"qha_selector_q_mix_shape={arrays['q_mix'].shape}")

        if pass_idx < args.passes:
            record_n = min(args.record_actions, int(plan["exec_length"]), arrays["actions"].shape[0])
            for i in range(record_n):
                model.record_executed_action(arrays["actions"][i])

    print("QHA smoke eval passed.")


if __name__ == "__main__":
    main()
