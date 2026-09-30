"""Transparent adaptive selectors used by the pi0 rebuttal controls.

``jerk_min`` and ``variance_min`` use only one sampled action chunk.
``cross_horizon_agreement`` uses only the provisional prefixes from that same
denoising rollout. None of these methods consume AHS scores, QHA outputs, or
executed-action history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_VALID_METHODS = {"jerk_min", "variance_min", "cross_horizon_agreement"}


@dataclass(frozen=True)
class ActionOnlySelection:
    method: str
    candidate_horizons: np.ndarray
    candidate_metrics: np.ndarray
    chosen_horizon: int
    tie_count: int
    continuous_action_dimensions: np.ndarray


def validate_method(method: str | None) -> str:
    normalized = str(method or "none").strip().lower()
    if normalized not in {"none", *_VALID_METHODS}:
        raise ValueError(
            "Unsupported adaptive_selector_mode="
            f"{method!r}; expected none, jerk_min, variance_min, or cross_horizon_agreement."
        )
    return normalized


def select_action_only_horizon(
    actions: np.ndarray,
    candidate_horizons: tuple[int, ...] | list[int] | np.ndarray,
    *,
    method: str,
    gripper_indices: tuple[int, ...] | list[int] | np.ndarray = (),
) -> ActionOnlySelection:
    """Pick the minimum prefix jerk/velocity-variance candidate.

    Ties deliberately resolve to the largest candidate.  This prevents a
    hidden preference for shorter horizons and is part of the frozen rebuttal
    protocol.
    """

    method = validate_method(method)
    if method not in {"jerk_min", "variance_min"}:
        raise ValueError("select_action_only_horizon requires jerk_min or variance_min")
    chunk = np.asarray(actions, dtype=np.float64)
    if chunk.ndim != 2 or chunk.shape[0] < 3 or chunk.shape[1] < 1:
        raise ValueError(f"Expected action chunk with shape [H, D], got {chunk.shape}")
    candidates = np.asarray(candidate_horizons, dtype=np.int32).reshape(-1)
    if candidates.size == 0 or np.unique(candidates).size != candidates.size:
        raise ValueError(f"Candidate horizons must be non-empty and unique, got {candidates.tolist()}")
    if np.any(candidates < 3) or np.any(candidates > chunk.shape[0]):
        raise ValueError(f"Candidate horizons must be in [3, {chunk.shape[0]}], got {candidates.tolist()}")

    grippers = {int(index) for index in np.asarray(gripper_indices, dtype=np.int64).reshape(-1)}
    dimensions = np.asarray([index for index in range(chunk.shape[1]) if index not in grippers], dtype=np.int32)
    if dimensions.size == 0:
        raise ValueError("No continuous action dimensions remain after gripper exclusion")
    continuous = chunk[:, dimensions]
    metrics: list[float] = []
    for candidate in candidates:
        prefix = continuous[: int(candidate)]
        velocity = np.diff(prefix, axis=0)
        if method == "jerk_min":
            jerk = np.diff(velocity, axis=0)
            value = float(np.mean(np.sum(np.square(jerk), axis=1)))
        else:
            value = float(np.mean(np.var(velocity, axis=0)))
        metrics.append(value)
    metric_array = np.asarray(metrics, dtype=np.float32)
    if not np.all(np.isfinite(metric_array)):
        raise ValueError(f"{method} produced non-finite candidate metrics: {metric_array.tolist()}")
    best = float(np.min(metric_array))
    tied = candidates[np.isclose(metric_array, best, rtol=0.0, atol=1e-12)]
    return ActionOnlySelection(
        method=method,
        candidate_horizons=candidates,
        candidate_metrics=metric_array,
        chosen_horizon=int(np.max(tied)),
        tie_count=int(tied.size),
        continuous_action_dimensions=dimensions,
    )


def select_cross_horizon_agreement(
    initial_noise: np.ndarray,
    denoise_velocities: np.ndarray,
    denoise_step_sizes: np.ndarray,
    candidate_horizons: tuple[int, ...] | list[int] | np.ndarray,
    *,
    gripper_indices: tuple[int, ...] | list[int] | np.ndarray = (),
    tail_steps: int = 3,
    eps: float = 1e-8,
) -> ActionOnlySelection:
    """Select the most stable prefix across late denoising predictions.

    This is a deliberately simple agreement heuristic, not an AHS score and
    not a reproduction of a published method.  It reconstructs provisional
    chunks from one sampled denoising trajectory and scores each candidate K
    by the RMS disagreement of its continuous prefix across the final
    ``tail_steps`` Euler states.  It never receives q_mix, q_intra, p_inter,
    action history, or a learned posterior.
    """

    initial = np.asarray(initial_noise, dtype=np.float64)
    velocities = np.asarray(denoise_velocities, dtype=np.float64)
    step_sizes = np.asarray(denoise_step_sizes, dtype=np.float64).reshape(-1)
    if initial.ndim != 2 or initial.shape[0] < 3 or initial.shape[1] < 1:
        raise ValueError(f"Expected initial noise [H, D], got {initial.shape}")
    if velocities.ndim != 3 or velocities.shape[1:] != initial.shape:
        raise ValueError(
            "Expected denoise velocities [T, H, D] matching initial noise; "
            f"got velocities={velocities.shape}, initial={initial.shape}"
        )
    if step_sizes.size != velocities.shape[0] or not np.all(np.isfinite(step_sizes)):
        raise ValueError(
            "Expected one finite denoise step size per velocity state; "
            f"got step_sizes={step_sizes.shape}, velocities={velocities.shape}"
        )
    if tail_steps < 2 or velocities.shape[0] < tail_steps:
        raise ValueError(
            f"tail_steps must be in [2, {velocities.shape[0]}], got {tail_steps}"
        )

    candidates = np.asarray(candidate_horizons, dtype=np.int32).reshape(-1)
    if candidates.size == 0 or np.unique(candidates).size != candidates.size:
        raise ValueError(f"Candidate horizons must be non-empty and unique, got {candidates.tolist()}")
    if np.any(candidates < 3) or np.any(candidates > initial.shape[0]):
        raise ValueError(
            f"Candidate horizons must be in [3, {initial.shape[0]}], got {candidates.tolist()}"
        )

    grippers = {int(index) for index in np.asarray(gripper_indices, dtype=np.int64).reshape(-1)}
    dimensions = np.asarray([index for index in range(initial.shape[1]) if index not in grippers], dtype=np.int32)
    if dimensions.size == 0:
        raise ValueError("No continuous action dimensions remain after gripper exclusion")

    # Euler reconstruction preserves the exact sampled trajectory.  Keeping
    # only late states removes the intentionally high uncertainty at t=1.
    states = [initial]
    current = initial
    for velocity, step_size in zip(velocities, step_sizes, strict=True):
        current = current + float(step_size) * velocity
        states.append(current)
    late_states = np.asarray(states[-int(tail_steps):], dtype=np.float64)
    metrics: list[float] = []
    for candidate in candidates:
        prefix = late_states[:, : int(candidate), :][:, :, dimensions]
        transitions = np.diff(prefix, axis=0)
        metrics.append(float(np.sqrt(np.mean(np.square(transitions)))))
    metric_array = np.asarray(metrics, dtype=np.float32)
    if not np.all(np.isfinite(metric_array)):
        raise ValueError(f"cross_horizon_agreement produced non-finite metrics: {metric_array.tolist()}")
    best = float(np.min(metric_array))
    tied = candidates[np.isclose(metric_array, best, rtol=0.0, atol=float(eps))]
    return ActionOnlySelection(
        method="cross_horizon_agreement",
        candidate_horizons=candidates,
        candidate_metrics=metric_array,
        chosen_horizon=int(np.max(tied)),
        tie_count=int(tied.size),
        continuous_action_dimensions=dimensions,
    )
