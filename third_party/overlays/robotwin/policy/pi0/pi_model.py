#!/usr/bin/env python3
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import os
import time
import math
import dataclasses
from typing import Any
from pathlib import Path
import jax
import numpy as np
from openpi.models import qha as _qha
from openpi.policies import policy_config as _policy_config
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from action_only_selector import (
    select_action_only_horizon,
    select_cross_horizon_agreement,
    validate_method as _validate_action_only_method,
)
from denoise_logger import DenoiseLogger


def _normalize_probabilities(probs: np.ndarray) -> np.ndarray:
    arr = np.asarray(probs, dtype=np.float64).reshape(-1)
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total <= 1e-12 or not np.isfinite(total):
        return np.full((arr.size,), 1.0 / float(max(arr.size, 1)), dtype=np.float32)
    return (arr / total).astype(np.float32)


def _compress_dense_posterior_to_sparse_basis(
    dense_horizons: np.ndarray,
    posterior_dense: np.ndarray,
    sparse_basis: np.ndarray,
) -> np.ndarray:
    dense_horizons = np.asarray(dense_horizons, dtype=np.float64).reshape(-1)
    posterior_dense = np.asarray(posterior_dense, dtype=np.float64).reshape(-1)
    sparse_basis = np.asarray(sparse_basis, dtype=np.float64).reshape(-1)
    if dense_horizons.shape != posterior_dense.shape:
        raise ValueError(f"dense_horizons/posterior mismatch: {dense_horizons.shape} vs {posterior_dense.shape}")
    if sparse_basis.size <= 0:
        raise ValueError("sparse_basis must be non-empty")
    bucket_idx = np.argmin(np.abs(dense_horizons[:, None] - sparse_basis[None, :]), axis=1)
    posterior_sparse = np.zeros((sparse_basis.size,), dtype=np.float64)
    np.add.at(posterior_sparse, bucket_idx, posterior_dense)
    return _normalize_probabilities(posterior_sparse)


def _interpolate_sparse_scores_to_dense_basis(
    sparse_horizons: np.ndarray,
    sparse_scores: np.ndarray,
    dense_horizons: np.ndarray,
) -> np.ndarray:
    sparse_horizons = np.asarray(sparse_horizons, dtype=np.float64).reshape(-1)
    sparse_scores = np.asarray(sparse_scores, dtype=np.float64).reshape(-1)
    dense_horizons = np.asarray(dense_horizons, dtype=np.float64).reshape(-1)
    if sparse_horizons.shape != sparse_scores.shape:
        raise ValueError(f"sparse_horizons/sparse_scores mismatch: {sparse_horizons.shape} vs {sparse_scores.shape}")
    if sparse_horizons.size <= 0:
        raise ValueError("sparse_horizons must be non-empty")
    return np.interp(
        dense_horizons,
        sparse_horizons,
        sparse_scores,
        left=float(sparse_scores[0]),
        right=float(sparse_scores[-1]),
    ).astype(np.float32)

class PI0:

    def __init__(
        self,
        train_config_name,
        model_name,
        checkpoint_id,
        pi0_step,
        dump_denoise=0,
        denoise_log_dir=None,
        valid_action_dim=None,
        eval_type="default",
        horizon_intra_cut_t=0.25,
        horizon_candidates="10,20,30,40,50",
        horizon_intra_alpha=4.0,
        horizon_ts_alpha=None,
        horizon_ts_epsilon=0.05,
        horizon_ts_seed=0,
        horizon_ts_update_mode="kernel_forget",
        horizon_ts_kernel_bandwidth=10.0,
        horizon_ts_forget_rho=0.99,
        horizon_ts_update_eta=1.0,
        horizon_intra_z_mode="window_rms",
        horizon_exec_mode="expected_round",
        horizon_expected_temp=1.0,
        horizon_intra_tau_window=3,
        horizon_inter_use_speed_uniformity=1,
        horizon_mix_mode="both",
        horizon_inter_window=60,
        horizon_inter_weight=0.5,
        horizon_inter_beta=4.0,
        horizon_inter_fallback="ehh_only",
        fixed_exec_k=0,
        denoise_npz_compress=0,
        left_action_dim=None,
        right_action_dim=None,
        checkpoint_root_tag=None,
        use_qha=None,
        qha_eval_variant="native",
        qha_candidate_mode_override="auto",
        qha_hybrid_prior_gamma_override=None,
        base_checkpoint_root_tag=None,
        base_model_name=None,
        base_checkpoint_id=None,
        adaptive_selector_mode="none",
    ):
        self.train_config_name = train_config_name
        self.checkpoint_root_tag = checkpoint_root_tag or train_config_name
        self.model_name = model_name
        self.checkpoint_id = str(checkpoint_id)
        self.dump_denoise = bool(dump_denoise)
        self.denoise_log_dir = denoise_log_dir
        self.valid_action_dim = int(valid_action_dim) if valid_action_dim is not None else None
        self.eval_type = str(eval_type)
        self.adaptive_selector_mode = _validate_action_only_method(adaptive_selector_mode)
        if self.adaptive_selector_mode != "none" and self.eval_type == "horizon":
            raise ValueError("adaptive_selector_mode requires eval_type=default so it cannot inherit AHS state.")
        self.trace_enabled = self.dump_denoise or (self.eval_type == "horizon")
        self.horizon_intra_cut_t = float(horizon_intra_cut_t)
        self.horizon_candidates = self._parse_horizon_candidates(horizon_candidates)
        if horizon_ts_alpha is not None and horizon_intra_alpha == 4.0:
            horizon_intra_alpha = horizon_ts_alpha
        self.horizon_intra_alpha = float(horizon_intra_alpha)
        self.horizon_ts_alpha = self.horizon_intra_alpha
        self.horizon_ts_epsilon = float(horizon_ts_epsilon)
        self.horizon_ts_seed = int(horizon_ts_seed)
        self.horizon_ts_update_mode = self._parse_horizon_ts_update_mode(horizon_ts_update_mode)
        self.horizon_ts_kernel_bandwidth = float(max(1e-6, float(horizon_ts_kernel_bandwidth)))
        self.horizon_ts_forget_rho = float(np.clip(float(horizon_ts_forget_rho), 1e-6, 1.0))
        self.horizon_ts_update_eta = float(max(0.0, float(horizon_ts_update_eta)))
        self.horizon_intra_z_mode = self._parse_horizon_z_mode(horizon_intra_z_mode)
        self.horizon_exec_mode = self._parse_horizon_exec_mode(horizon_exec_mode)
        self.horizon_expected_temp = float(max(1e-6, float(horizon_expected_temp)))
        self.horizon_intra_tau_window = max(1, int(horizon_intra_tau_window))
        self.horizon_inter_use_speed_uniformity = bool(int(horizon_inter_use_speed_uniformity))
        self.horizon_mix_mode = str(horizon_mix_mode).strip().lower()
        if self.horizon_mix_mode not in ("intra", "inter", "both"):
            self.horizon_mix_mode = "both"
        self.horizon_inter_window = max(4, int(horizon_inter_window))
        self.horizon_inter_weight = float(np.clip(float(horizon_inter_weight), 0.0, 1.0))
        self.horizon_inter_beta = float(max(0.0, float(horizon_inter_beta)))
        self.horizon_inter_fallback = str(horizon_inter_fallback).strip().lower()
        if self.horizon_inter_fallback not in ("ehh_only", "neutral"):
            self.horizon_inter_fallback = "ehh_only"
        self.fixed_exec_k = int(fixed_exec_k)
        self.denoise_npz_compress = bool(int(denoise_npz_compress))
        self.left_action_dim = int(left_action_dim) if left_action_dim is not None else None
        self.right_action_dim = int(right_action_dim) if right_action_dim is not None else None
        self.last_exec_k = None

        split_loading = base_model_name is not None and base_checkpoint_id is not None
        if split_loading:
            base_checkpoint_root = self._resolve_checkpoint_root(
                self.train_config_name,
                base_model_name,
                str(base_checkpoint_id),
                checkpoint_root_tag=base_checkpoint_root_tag or self.checkpoint_root_tag,
            )
            qha_checkpoint_root = self._resolve_checkpoint_root(
                self.train_config_name,
                self.model_name,
                self.checkpoint_id,
                checkpoint_root_tag=self.checkpoint_root_tag,
            )
            metadata_info = self._inspect_checkpoint_metadata(qha_checkpoint_root / "params" / "_METADATA")
        else:
            checkpoint_root = self._resolve_checkpoint_root(
                self.train_config_name,
                self.model_name,
                self.checkpoint_id,
                checkpoint_root_tag=self.checkpoint_root_tag,
            )
            metadata_info = self._inspect_checkpoint_metadata(checkpoint_root / "params" / "_METADATA")

        if metadata_info["has_horizon_head"]:
            raise ValueError(
                "Old horizon-head checkpoints are not supported by the QHA runtime. "
                "Please load a base pi0 checkpoint or a QHA-compatible checkpoint."
            )

        config = _config.get_config(self.train_config_name)
        auto_qha = metadata_info["has_qha"]
        if use_qha is None and auto_qha is not None and hasattr(config.model, "use_qha"):
            config.model = dataclasses.replace(config.model, use_qha=auto_qha)
        if use_qha is not None and hasattr(config.model, "use_qha"):
            config.model = dataclasses.replace(config.model, use_qha=bool(use_qha))

        if split_loading:
            config.model = _checkpoints.resolve_model_config_for_checkpoint(
                config.model,
                qha_checkpoint_root,
                qha_candidate_mode_override=qha_candidate_mode_override,
            )
        else:
            config.model = _checkpoints.resolve_model_config_for_checkpoint(
                config.model,
                checkpoint_root,
                qha_candidate_mode_override=qha_candidate_mode_override,
            )
        if qha_hybrid_prior_gamma_override is not None and hasattr(config.model, "qha_hybrid_prior_gamma"):
            config.model = dataclasses.replace(
                config.model,
                qha_hybrid_prior_gamma=float(qha_hybrid_prior_gamma_override),
            )

        self.use_qha = bool(getattr(config.model, "use_qha", False))
        self.history_max_length = int(getattr(config.model, "qha_history_max_length", 0)) if self.use_qha else 0
        self.qha_infer_mode = str(getattr(config.model, "qha_infer_mode", "hybrid")).strip().lower()
        self.qha_candidate_horizons = _qha.resolve_qha_candidate_horizons(
            getattr(config.model, "qha_candidates", "10,20,30,40,50"),
            candidate_mode=str(getattr(config.model, "qha_candidate_mode", "sparse")),
            max_horizon=int(getattr(config.model, "action_horizon", 50)),
        )
        self.qha_eval_variant = str(qha_eval_variant).strip().lower()
        if self.qha_eval_variant not in ("native", "posterior_sparse_compromise", "selector_dense_interpolation"):
            raise ValueError(
                "qha_eval_variant must be native, posterior_sparse_compromise, or "
                f"selector_dense_interpolation, got {qha_eval_variant}."
            )
        self._runtime_sparse_basis = _qha.default_sparse_runtime_candidate_horizons(
            max_horizon=int(getattr(config.model, "action_horizon", 50))
        )
        self.qha_hybrid_prior_gamma = float(getattr(config.model, "qha_hybrid_prior_gamma", 1.0))
        self._runtime_selector_kwargs = {
            "exec_mode": str(getattr(config.model, "horizon_selector_exec_mode", "expected_round")),
            "expected_temp": float(getattr(config.model, "horizon_selector_expected_temp", 1.0)),
            "ts_epsilon": float(getattr(config.model, "horizon_selector_ts_epsilon", 0.05)),
            "ts_seed": int(getattr(config.model, "horizon_selector_ts_seed", 0)),
            "ts_update_mode": str(getattr(config.model, "horizon_selector_ts_update_mode", "kernel_forget")),
            "ts_kernel_bandwidth": float(getattr(config.model, "horizon_selector_ts_kernel_bandwidth", 10.0)),
            "ts_forget_rho": float(getattr(config.model, "horizon_selector_ts_forget_rho", 0.99)),
            "ts_update_eta": float(getattr(config.model, "horizon_selector_ts_update_eta", 1.0)),
            "prior_gamma": self.qha_hybrid_prior_gamma,
        }
        self._runtime_selector = self._make_runtime_selector(self.qha_candidate_horizons) if (
            self.use_qha and self.qha_infer_mode == "hybrid"
        ) else None

        sample_kwargs = {"return_denoise_trace": self.trace_enabled}
        if split_loading:
            assets_id = self._resolve_assets_id(base_checkpoint_root / "assets")
            self.policy = _policy_config.create_trained_policy_from_base_and_qha(
                config,
                base_checkpoint_root,
                qha_checkpoint_root,
                qha_candidate_mode_override=qha_candidate_mode_override,
                robotwin_repo_id=assets_id,
                sample_kwargs=sample_kwargs,
            )
        else:
            assets_id = self._resolve_assets_id(checkpoint_root / "assets")
            self.policy = _policy_config.create_trained_policy(
                config,
                checkpoint_root,
                resolved_model_config=config.model,
                qha_candidate_mode_override=qha_candidate_mode_override,
                robotwin_repo_id=assets_id,
                sample_kwargs=sample_kwargs,
            )
        print(
            "loading model success! "
            f"use_qha={self.use_qha}, "
            f"history_max_length={self.history_max_length}, "
            f"qha_infer_mode={self.qha_infer_mode}, "
            f"qha_eval_variant={self.qha_eval_variant}, "
            f"qha_hybrid_prior_gamma={self.qha_hybrid_prior_gamma}"
        )
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step
        self.denoise_logger = DenoiseLogger(enabled=self.dump_denoise, compress=self.denoise_npz_compress)
        self._episode_idx = None
        self._episode_path = None
        self._denoise_overhead_seconds = 0.0
        self._policy_infer_seconds = 0.0
        self._horizon_metric_compute_seconds = 0.0
        self._horizon_selection_seconds = 0.0
        self._horizon_ts_rng = np.random.default_rng(self.horizon_ts_seed)
        self._horizon_ts_state = {int(s): [1.0, 1.0] for s in self.horizon_candidates}  # a,b
        self._executed_action_history = np.zeros((0, 0), dtype=np.float32)
        self._episode_context: dict[str, int | str] = {}
        self._last_used_h = int(self.pi0_step)
        self._last_action_dim = int(self.valid_action_dim) if self.valid_action_dim is not None else 0
        self._reset_run_score_state()

    @staticmethod
    def _parse_horizon_candidates(raw) -> list[int]:
        if raw is None:
            return [10, 20, 30, 40, 50]
        out = []
        if isinstance(raw, (list, tuple)):
            vals = raw
        else:
            vals = str(raw).replace(";", ",").split(",")
        for x in vals:
            token = str(x).strip()
            if not token:
                continue
            range_parts = token.split(":")
            if len(range_parts) in (2, 3):
                try:
                    start = int(range_parts[0].strip())
                    end = int(range_parts[1].strip())
                    step = int(range_parts[2].strip()) if len(range_parts) == 3 else 1
                    if start > 0 and end > 0 and step > 0:
                        if start <= end:
                            out.extend(range(start, end + 1, step))
                        else:
                            out.extend(range(start, end - 1, -step))
                        continue
                except Exception:
                    pass
            try:
                v = int(token)
                if v > 0:
                    out.append(v)
            except Exception:
                continue
        if not out:
            out = [10, 20, 30, 40, 50]
        return sorted(set(out))

    @staticmethod
    def _parse_horizon_z_mode(raw) -> str:
        mode = str(raw or "window_rms").strip().lower()
        if mode not in {"tau0_rms", "window_rms", "trend"}:
            mode = "window_rms"
        return mode

    @staticmethod
    def _parse_horizon_exec_mode(raw) -> str:
        mode = str(raw or "expected_round").strip().lower()
        if mode not in {"thompson", "expected_round"}:
            mode = "expected_round"
        return mode

    @staticmethod
    def _parse_horizon_ts_update_mode(raw) -> str:
        mode = str(raw or "kernel_forget").strip().lower()
        if mode not in {"independent", "kernel_forget"}:
            mode = "kernel_forget"
        return mode

    def _horizon_policy_name(self) -> str:
        parts = ["horizon", "ehh1dpad", self.horizon_intra_z_mode]
        if self.horizon_inter_use_speed_uniformity and self.horizon_mix_mode in ("inter", "both"):
            parts.append("speedu")
        parts.append(self.horizon_mix_mode)
        parts.append(self.horizon_exec_mode)
        if self.horizon_ts_update_mode != "independent":
            parts.append(self.horizon_ts_update_mode)
        if self.horizon_exec_mode == "expected_round":
            temp_str = f"{self.horizon_expected_temp:g}".replace(".", "p")
            parts.append(f"temp{temp_str}")
        return "_".join(parts)

    def _infer_gripper_indices(self, action_dim: int | None = None) -> list[int]:
        if action_dim is None or int(action_dim) <= 0:
            action_dim = self.valid_action_dim
        if action_dim is None or int(action_dim) <= 0:
            return []
        idxs: list[int] = []
        if self.left_action_dim is not None and self.left_action_dim > 0:
            idxs.append(int(self.left_action_dim - 1))
        if self.right_action_dim is not None and self.right_action_dim > 0:
            base = int(self.left_action_dim or 0)
            idxs.append(base + int(self.right_action_dim - 1))
        idxs = [idx for idx in idxs if 0 <= idx < int(action_dim)]
        if idxs:
            return sorted(set(idxs))
        if int(action_dim) == 14:
            return [6, 13]
        return []

    def set_episode_context(
        self,
        *,
        episode_seed: int | None = None,
        instruction_hash: str | None = None,
        instruction_len: int | None = None,
        episode_manifest_id: str | None = None,
        manifest_rollout_id: int | None = None,
        manifest_sha256: str | None = None,
    ) -> None:
        ctx: dict[str, int | str] = {}
        if episode_seed is not None:
            ctx["episode_seed"] = int(episode_seed)
        if instruction_hash is not None:
            ctx["instruction_hash"] = str(instruction_hash)
        if instruction_len is not None:
            ctx["instruction_len"] = int(instruction_len)
        if episode_manifest_id is not None:
            ctx["episode_manifest_id"] = str(episode_manifest_id)
        if manifest_rollout_id is not None:
            ctx["manifest_rollout_id"] = int(manifest_rollout_id)
        if manifest_sha256 is not None:
            ctx["manifest_sha256"] = str(manifest_sha256)
        self._episode_context = ctx

    def _build_episode_meta(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "candidate_k": np.asarray(self.horizon_candidates, dtype=np.int32),
            "fixed_exec_k_requested": int(self.fixed_exec_k),
            "adaptive_selector_mode": self.adaptive_selector_mode,
            "adaptive_selector_score_source": (
                "same_rollout_denoising_prefix_agreement"
                if self.adaptive_selector_mode == "cross_horizon_agreement"
                else "single_action_chunk" if self.adaptive_selector_mode != "none" else "none"
            ),
        }
        meta.update(self._episode_context)
        gripper_indices = self._infer_gripper_indices(self.valid_action_dim)
        if gripper_indices:
            meta["gripper_action_indices"] = np.asarray(gripper_indices, dtype=np.int32)
        return meta

    def _select_adaptive_exec_k(
        self,
        actions: np.ndarray,
        used_h: int,
        trace: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        candidates = tuple(int(k) for k in self.horizon_candidates if 3 <= int(k) <= int(used_h))
        if not candidates:
            raise ValueError(
                f"adaptive_selector_mode={self.adaptive_selector_mode} has no valid candidates for action horizon {used_h}"
            )
        gripper_indices = self._infer_gripper_indices(int(actions.shape[-1]))
        if self.adaptive_selector_mode == "cross_horizon_agreement":
            if trace is None:
                raise RuntimeError("cross_horizon_agreement requires the sampled denoising trace")
            result = select_cross_horizon_agreement(
                np.asarray(jax.device_get(trace["x0"]), dtype=np.float32)[:used_h],
                np.asarray(jax.device_get(trace["v"]), dtype=np.float32)[:, :used_h],
                np.asarray(jax.device_get(trace["ds"]), dtype=np.float32),
                candidates,
                gripper_indices=gripper_indices,
            )
        else:
            result = select_action_only_horizon(
                np.asarray(actions[:used_h], dtype=np.float32),
                candidates,
                method=self.adaptive_selector_mode,
                gripper_indices=gripper_indices,
            )
        return {
            "chosen_k": int(result.chosen_horizon),
            "candidate_metrics": np.asarray(result.candidate_metrics, dtype=np.float32),
            "tie_count": int(result.tie_count),
            "continuous_action_dimensions": np.asarray(result.continuous_action_dimensions, dtype=np.int32),
        }

    def _reset_run_score_state(self) -> None:
        self._run_score_state = {
            "avg_q_intra": {"sum": 0.0, "count": 0},
            "avg_p_inter": {"sum": 0.0, "count": 0},
            "avg_q_mix": {"sum": 0.0, "count": 0},
            "avg_q_intra_contrib": {"sum": 0.0, "count": 0},
            "avg_p_inter_contrib": {"sum": 0.0, "count": 0},
            "avg_z_intra_raw": {"sum": 0.0, "count": 0},
            "avg_z_intra_norm": {"sum": 0.0, "count": 0},
            "avg_u_inter_raw": {"sum": 0.0, "count": 0},
            "avg_u_inter_norm": {"sum": 0.0, "count": 0},
        }

    def reset_run_score_state(self) -> None:
        self._reset_run_score_state()

    @staticmethod
    def _mean_finite(values: np.ndarray | list[float]) -> float | None:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size <= 0:
            return None
        finite = arr[np.isfinite(arr)]
        if finite.size <= 0:
            return None
        return float(np.mean(finite))

    def _record_run_score_mean(self, key: str, value: float | None) -> None:
        if value is None or not np.isfinite(value):
            return
        bucket = self._run_score_state[key]
        bucket["sum"] += float(value)
        bucket["count"] += 1

    @staticmethod
    def _normalize_candidate_metric(values: np.ndarray | list[float], eps: float = 1e-12) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size <= 0:
            return np.zeros((0,), dtype=np.float64)
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        finite = np.isfinite(arr)
        if not np.any(finite):
            return out
        valid = arr[finite]
        if valid.size <= 1 or float(np.max(valid) - np.min(valid)) <= eps:
            out[finite] = 0.0
            return out
        out[finite] = (valid - float(np.min(valid))) / (float(np.max(valid) - np.min(valid)) + eps)
        return out

    def _compute_effective_inter_array(self, p_inter: np.ndarray, q_intra: np.ndarray) -> np.ndarray:
        p_arr = np.asarray(p_inter, dtype=np.float64)
        q_arr = np.asarray(q_intra, dtype=np.float64)
        effective = np.array(p_arr, copy=True)
        finite_mask = np.isfinite(effective)
        if np.any(finite_mask):
            effective[finite_mask] = np.clip(effective[finite_mask], 0.0, 1.0)
        invalid_mask = ~finite_mask
        if np.any(invalid_mask):
            if self.horizon_inter_fallback == "neutral":
                effective[invalid_mask] = 0.5
            else:
                effective[invalid_mask] = q_arr[invalid_mask]
        return effective

    def _update_run_score_stats(self, horizon_info: dict | None) -> None:
        if self.eval_type != "horizon" or not horizon_info:
            return

        q_intra = np.asarray(horizon_info.get("q_intra"), dtype=np.float64)
        p_inter = np.asarray(horizon_info.get("p_inter"), dtype=np.float64)
        q_mix = np.asarray(horizon_info.get("q_mix"), dtype=np.float64)
        z_intra_raw = np.asarray(horizon_info.get("z_intra"), dtype=np.float64)
        z_intra_norm = np.asarray(horizon_info.get("z_intra_norm"), dtype=np.float64)
        u_inter_raw = np.asarray(horizon_info.get("u_inter_raw"), dtype=np.float64)
        u_inter_norm = np.asarray(horizon_info.get("u_inter_norm"), dtype=np.float64)
        if q_intra.size <= 0 or q_mix.size <= 0:
            return

        p_effective = self._compute_effective_inter_array(p_inter, q_intra)

        finite_p = np.isfinite(p_inter)
        invalid_p = ~finite_p
        q_intra_contrib = np.zeros_like(q_intra, dtype=np.float64)
        p_inter_contrib = np.zeros_like(q_intra, dtype=np.float64)

        if self.horizon_mix_mode == "intra":
            q_intra_contrib = q_mix
        elif self.horizon_mix_mode == "inter":
            if np.any(finite_p):
                p_inter_contrib[finite_p] = p_effective[finite_p]
            if np.any(invalid_p):
                if self.horizon_inter_fallback == "neutral":
                    p_inter_contrib[invalid_p] = 0.5
                else:
                    q_intra_contrib[invalid_p] = q_intra[invalid_p]
        else:
            if np.any(finite_p):
                q_intra_contrib[finite_p] = (1.0 - self.horizon_inter_weight) * q_intra[finite_p]
                p_inter_contrib[finite_p] = self.horizon_inter_weight * p_effective[finite_p]
            if np.any(invalid_p):
                if self.horizon_inter_fallback == "neutral":
                    q_intra_contrib[invalid_p] = (1.0 - self.horizon_inter_weight) * q_intra[invalid_p]
                    p_inter_contrib[invalid_p] = self.horizon_inter_weight * 0.5
                else:
                    q_intra_contrib[invalid_p] = q_intra[invalid_p]

        self._record_run_score_mean("avg_q_intra", self._mean_finite(q_intra))
        self._record_run_score_mean("avg_p_inter", self._mean_finite(p_inter))
        self._record_run_score_mean("avg_q_mix", self._mean_finite(q_mix))
        self._record_run_score_mean("avg_q_intra_contrib", self._mean_finite(q_intra_contrib))
        self._record_run_score_mean("avg_p_inter_contrib", self._mean_finite(p_inter_contrib))
        self._record_run_score_mean("avg_z_intra_raw", self._mean_finite(z_intra_raw))
        self._record_run_score_mean("avg_z_intra_norm", self._mean_finite(z_intra_norm))
        self._record_run_score_mean("avg_u_inter_raw", self._mean_finite(u_inter_raw))
        self._record_run_score_mean("avg_u_inter_norm", self._mean_finite(u_inter_norm))

    def get_run_score_averages(self) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for key, bucket in self._run_score_state.items():
            count = int(bucket["count"])
            if count <= 0:
                out[key] = None
            else:
                out[key] = float(bucket["sum"] / float(count))
        return out


    @staticmethod
    def _normalize_score_distribution(score_arr: np.ndarray, temp: float = 1.0) -> np.ndarray:
        # Temperature acts on the normalized positive weights:
        # p_T(k) propto p(k)^(1/T) == exp(log p(k) / T).
        # This is equivalent to applying a softmax temperature in log-probability
        # space, while preserving exact backward compatibility at T=1.
        arr = np.asarray(score_arr, dtype=np.float64)
        if arr.size <= 0:
            return np.zeros((0,), dtype=np.float64)
        arr = np.maximum(arr, 0.0)
        arr_sum = float(np.sum(arr))
        if arr_sum <= 1e-12 or (not np.isfinite(arr_sum)):
            return np.full((arr.size,), 1.0 / float(arr.size), dtype=np.float64)
        if abs(float(temp) - 1.0) <= 1e-12:
            return arr / arr_sum
        base = arr / arr_sum
        sharp = np.power(np.clip(base, 0.0, 1.0), 1.0 / float(max(1e-6, temp)))
        sharp_sum = float(np.sum(sharp))
        if sharp_sum <= 1e-12 or (not np.isfinite(sharp_sum)):
            return np.full((arr.size,), 1.0 / float(arr.size), dtype=np.float64)
        return sharp / sharp_sum

    def _compute_ehh_1d_padded(self, mat_h_da: np.ndarray, h_full: int, eps: float = 1e-12) -> float:
        # NOTE:
        # - We intentionally only run FFT along the chunk axis (H).
        # - action_dim is treated as channel dimension and only aggregated.
        # - zero-padding to h_full aligns frequency grids across different horizons.
        hs = int(mat_h_da.shape[0])
        da = int(mat_h_da.shape[1])
        pad_h = max(int(h_full), hs)
        if pad_h == hs:
            mat_pad = np.asarray(mat_h_da, dtype=np.float64)
        else:
            mat_pad = np.zeros((pad_h, da), dtype=np.float64)
            mat_pad[:hs, :] = np.asarray(mat_h_da, dtype=np.float64)
        spec = np.fft.rfft(mat_pad, axis=0)
        pwr = (spec.real * spec.real) + (spec.imag * spec.imag)  # (F, d_a)
        # Aggregate channels first; frequency split is only along chunk axis.
        pwr_f = np.mean(pwr, axis=1)  # (F,)
        f_t = int(pwr_f.shape[0])
        ft_c = max(1, int(self.horizon_intra_cut_t * f_t))
        e_low = float(np.sum(pwr_f[:ft_c]))
        e_high = float(np.sum(pwr_f[ft_c:]))
        e_all = e_low + e_high + eps
        return float(e_high / e_all)

    def _ehh_curve_and_mean_over_tau(
        self, v_step_prefix: np.ndarray, h_full: int | None = None, eps: float = 1e-12
    ) -> tuple[np.ndarray, float]:
        # v_step_prefix: (N_tau, Hs, d_a)
        n_tau = int(v_step_prefix.shape[0])
        if n_tau <= 0:
            return np.zeros((0,), dtype=np.float64), 0.0
        if h_full is None:
            h_full = int(v_step_prefix.shape[1])
        e_vals = np.zeros((n_tau,), dtype=np.float64)
        for tau in range(n_tau):
            mat = v_step_prefix[tau]
            e_vals[tau] = self._compute_ehh_1d_padded(mat, h_full=int(h_full), eps=eps)
        return e_vals, float(np.mean(e_vals))

    def _ehh_curves_by_horizon_batched(
        self, v_step: np.ndarray, horizons: list[int], h_full: int | None = None, eps: float = 1e-12
    ) -> dict[int, tuple[np.ndarray, float]]:
        # Exact equivalent to calling _ehh_curve_and_mean_over_tau() separately for
        # each prefix horizon, but keeps FFT in the optimized numpy kernel by
        # batching all padded prefixes together in one rfft call.
        n_tau = int(v_step.shape[0])
        h = int(v_step.shape[1])
        d_a = int(v_step.shape[2])
        if n_tau <= 0:
            return {}
        if h_full is None:
            h_full = h
        h_full = max(int(h_full), h)
        horizons = [int(s) for s in horizons]
        if not horizons:
            return {}

        batch = np.zeros((len(horizons), n_tau, h_full, d_a), dtype=np.float64)
        v64 = np.asarray(v_step, dtype=np.float64)
        for i, s in enumerate(horizons):
            s_clip = max(0, min(int(s), h))
            if s_clip > 0:
                batch[i, :, :s_clip, :] = v64[:, :s_clip, :]

        spec = np.fft.rfft(batch, axis=2)
        pwr = (spec.real * spec.real) + (spec.imag * spec.imag)
        pwr_f = np.mean(pwr, axis=3)  # (S, T, F)
        f_t = int(pwr_f.shape[2])
        ft_c = max(1, int(self.horizon_intra_cut_t * f_t))
        e_low = np.sum(pwr_f[:, :, :ft_c], axis=2)
        e_high = np.sum(pwr_f[:, :, ft_c:], axis=2)
        e_vals_all = e_high / (e_low + e_high + eps)

        out = {}
        for i, s in enumerate(horizons):
            e_vals = np.asarray(e_vals_all[i], dtype=np.float64)
            out[int(s)] = (e_vals, float(np.mean(e_vals)))
        return out

    @staticmethod
    def _z_trend_abs_slope(y: np.ndarray) -> float:
        n = int(y.size)
        if n <= 1:
            return 0.0
        x = np.arange(n, dtype=np.float64)
        x = x - np.mean(x)
        y0 = y - np.mean(y)
        denom = float(np.sum(x * x))
        if denom <= 1e-12:
            return 0.0
        slope = float(np.sum(x * y0) / denom)
        return abs(slope)

    def _compute_z_from_ehh(
        self,
        e_vals: np.ndarray,
        tau0_idx: int = 0,
        mode_override: str | None = None,
        tau_window_override: int | None = None,
    ) -> float:
        if e_vals.size <= 1:
            return 0.0
        mode = self._parse_horizon_z_mode(mode_override if mode_override is not None else self.horizon_intra_z_mode)
        tau0 = int(np.clip(tau0_idx, 0, max(0, int(e_vals.size) - 1)))
        tail = np.asarray(e_vals[tau0:], dtype=np.float64)
        if tail.size <= 1:
            return 0.0
        if mode == "trend":
            return self._z_trend_abs_slope(tail)
        if mode == "tau0_rms":
            baseline = float(e_vals[tau0])
        else:
            w = max(1, int(tau_window_override if tau_window_override is not None else self.horizon_intra_tau_window))
            hi = min(int(e_vals.size), tau0 + w)
            baseline = float(np.mean(e_vals[tau0:hi]))
        dev = tail - baseline
        return float(math.sqrt(float(np.mean(dev * dev))))

    def _speed_uniformity_proxy_by_horizon(self, xt_step: np.ndarray, horizons: list[int], eps: float = 1e-12) -> tuple[list[float], list[float], list[float]]:
        # Returns:
        # - u_raw: per-horizon speed-uniformity CV (lower is better), NaN if unavailable
        # - u_norm: candidate-local normalized continuity score input
        # - p_proxy: mapped proxy in [0,1] (higher is better)
        xt_step = np.asarray(xt_step, dtype=np.float32)
        if xt_step.ndim != 2 or xt_step.shape[1] <= 0:
            nan_list = [float("nan")] * len(horizons)
            return nan_list, nan_list, nan_list

        h = int(xt_step.shape[0])
        da = int(xt_step.shape[1])
        win = max(4, int(self.horizon_inter_window))
        history_actions = self._executed_action_history
        if history_actions.ndim != 2 or int(history_actions.shape[1]) != da:
            history_actions = np.zeros((0, da), dtype=np.float32)
            self._executed_action_history = history_actions
        history_len = int(history_actions.shape[0])
        max_horizon = max([int(s) for s in horizons], default=0)
        history_tail_len = min(history_len, win)

        tail = np.asarray(history_actions[-history_tail_len:, :], dtype=np.float32)
        xt_prefix = np.asarray(xt_step[:max_horizon, :], dtype=np.float32)
        seq = np.concatenate([tail, xt_prefix], axis=0)
        speed = np.linalg.norm(np.diff(seq, axis=0), axis=1).astype(np.float64, copy=False) if seq.shape[0] >= 2 else np.zeros((0,), dtype=np.float64)
        speed_prefix = np.concatenate([[0.0], np.cumsum(speed, dtype=np.float64)])
        speed_sq_prefix = np.concatenate([[0.0], np.cumsum(speed * speed, dtype=np.float64)])

        u_raw = []
        for s in horizons:
            s = int(s)
            if s < 1 or s > h:
                u_raw.append(float("nan"))
                continue
            pre = int(win - s)
            post = int(s)
            if pre < 1 or post < 1 or history_len < pre or h < post:
                u_raw.append(float("nan"))
                continue

            start = int(history_tail_len - pre)
            speed_start = start
            speed_end = speed_start + (win - 1)
            if speed_start < 0 or speed_end > int(speed.shape[0]):
                u_raw.append(float("nan"))
                continue

            count = float(win - 1)
            sum_speed = float(speed_prefix[speed_end] - speed_prefix[speed_start])
            sum_speed_sq = float(speed_sq_prefix[speed_end] - speed_sq_prefix[speed_start])
            mu = sum_speed / count
            if mu <= eps:
                u_raw.append(0.0)
                continue
            var = max(0.0, (sum_speed_sq / count) - (mu * mu))
            cv = float(math.sqrt(var) / (mu + eps))
            u_raw.append(cv if np.isfinite(cv) else float("nan"))

        u_norm_vals = self._normalize_candidate_metric(u_raw, eps=eps)
        p_proxy = []
        for u_norm in u_norm_vals:
            if not np.isfinite(u_norm):
                p_proxy.append(float("nan"))
                continue
            p = float(np.exp(-self.horizon_inter_beta * max(0.0, u_norm)))
            p_proxy.append(float(np.clip(p, 0.0, 1.0)))
        return u_raw, np.asarray(u_norm_vals, dtype=np.float64).tolist(), p_proxy


    @staticmethod
    def _gaussian_kernel_weights(horizons: list[int], center: float, bandwidth: float, eps: float = 1e-12) -> np.ndarray:
        vals = np.asarray(horizons, dtype=np.float64)
        if vals.size <= 0:
            return np.zeros((0,), dtype=np.float64)
        if not np.isfinite(center):
            return np.full((vals.size,), 1.0 / float(vals.size), dtype=np.float64)
        bw = max(float(bandwidth), eps)
        weights = np.exp(-((vals - float(center)) ** 2) / (2.0 * bw * bw))
        weight_sum = float(np.sum(weights))
        if weight_sum <= eps or (not np.isfinite(weight_sum)):
            return np.full((vals.size,), 1.0 / float(vals.size), dtype=np.float64)
        return weights / weight_sum

    def _apply_horizon_ts_update(
        self,
        horizons: list[int],
        q_mix_vals: list[float],
        chosen_idx: int,
        choose_mode: str,
        k_exec: int,
        expected_prob: np.ndarray | None = None,
    ) -> tuple[float, float]:
        if self.horizon_ts_update_mode == "independent":
            if choose_mode == "expected_round" and expected_prob is not None:
                for i, s in enumerate(horizons):
                    prob_i = float(expected_prob[i]) if i < int(expected_prob.size) else 0.0
                    if prob_i <= 0.0:
                        continue
                    qa_i = float(q_mix_vals[i])
                    a, b = self._horizon_ts_state.get(int(s), [1.0, 1.0])
                    self._horizon_ts_state[int(s)] = [a + (prob_i * qa_i), b + (prob_i * (1.0 - qa_i))]
                feedback = float(np.dot(np.asarray(expected_prob, dtype=np.float64), np.asarray(q_mix_vals, dtype=np.float64)))
                return float(k_exec), feedback
            qa = float(q_mix_vals[chosen_idx])
            a, b = self._horizon_ts_state.get(int(k_exec), [1.0, 1.0])
            self._horizon_ts_state[int(k_exec)] = [a + qa, b + (1.0 - qa)]
            return float(k_exec), qa

        if choose_mode == "expected_round" and expected_prob is not None:
            feedback = float(
                np.dot(
                    np.asarray(expected_prob, dtype=np.float64),
                    np.asarray(q_mix_vals, dtype=np.float64),
                )
            )
        else:
            feedback = float(q_mix_vals[chosen_idx])
        feedback = float(np.clip(feedback, 0.0, 1.0))
        update_center = float(k_exec)
        weights = self._gaussian_kernel_weights(horizons, center=update_center, bandwidth=self.horizon_ts_kernel_bandwidth)
        rho = self.horizon_ts_forget_rho
        eta = self.horizon_ts_update_eta
        for i, s in enumerate(horizons):
            a, b = self._horizon_ts_state.get(int(s), [1.0, 1.0])
            w_i = float(weights[i]) if i < int(weights.size) else 0.0
            incr = eta * w_i
            self._horizon_ts_state[int(s)] = [
                rho * a + incr * feedback,
                rho * b + incr * (1.0 - feedback),
            ]
        return update_center, feedback

    def _ehh_rms_from_tau0(self, v_step_prefix: np.ndarray, h_full: int | None = None, eps: float = 1e-12) -> float:
        # Legacy score for compatibility/debug: RMS vs tau0 single-point baseline.
        e_vals, _ = self._ehh_curve_and_mean_over_tau(v_step_prefix, h_full=h_full, eps=eps)
        if e_vals.size <= 1:
            return 0.0
        return self._compute_z_from_ehh(e_vals, tau0_idx=0, mode_override="tau0_rms")

    def _compute_horizon_info(self, v_step: np.ndarray, used_h: int, xt_step: np.ndarray | None = None) -> dict[str, Any]:
        horizons = [s for s in self.horizon_candidates if s <= int(used_h)]
        if not horizons:
            horizons = [int(used_h)]

        use_inter_signal = self.horizon_inter_use_speed_uniformity and (self.horizon_mix_mode in ("inter", "both"))
        if use_inter_signal and xt_step is not None:
            u_vals, u_norm_vals, p_vals = self._speed_uniformity_proxy_by_horizon(
                np.asarray(xt_step, dtype=np.float32),
                horizons,
            )
        else:
            u_vals = [float("nan")] * len(horizons)
            u_norm_vals = [float("nan")] * len(horizons)
            p_vals = [float("nan")] * len(horizons)

        ehh_by_horizon = self._ehh_curves_by_horizon_batched(
            np.asarray(v_step[:, :used_h, :], dtype=np.float32),
            horizons,
            h_full=used_h,
        )

        z_vals = []
        for s in horizons:
            e_vals, _ = ehh_by_horizon[int(s)]
            z = self._compute_z_from_ehh(e_vals, tau0_idx=0)
            z_vals.append(float(z))

        z_norm_vals = self._normalize_candidate_metric(z_vals)
        q_intra_vals = []
        q_mix_vals = []
        for i, _s in enumerate(horizons):
            z_norm = float(z_norm_vals[i]) if i < int(z_norm_vals.size) else float("nan")
            if np.isfinite(z_norm):
                q_intra = float(np.exp(-self.horizon_intra_alpha * max(0.0, z_norm)))
            else:
                q_intra = float("nan")
            p_inter = p_vals[i]
            if self.horizon_mix_mode == "intra":
                q_mix = q_intra
            elif self.horizon_mix_mode == "inter":
                if np.isfinite(p_inter):
                    q_mix = float(np.clip(p_inter, 0.0, 1.0))
                else:
                    q_mix = 0.5 if self.horizon_inter_fallback == "neutral" else q_intra
            else:
                if np.isfinite(p_inter):
                    q_mix = float(
                        (1.0 - self.horizon_inter_weight) * q_intra
                        + self.horizon_inter_weight * float(np.clip(p_inter, 0.0, 1.0))
                    )
                else:
                    q_mix = (
                        float((1.0 - self.horizon_inter_weight) * q_intra + self.horizon_inter_weight * 0.5)
                        if self.horizon_inter_fallback == "neutral"
                        else q_intra
                    )
            q_intra_vals.append(float(q_intra))
            q_mix_vals.append(float(q_mix))
        return {
            "horizons": np.asarray(horizons, dtype=np.int32),
            "z_intra": np.asarray(z_vals, dtype=np.float32),
            "z_intra_norm": np.asarray(z_norm_vals, dtype=np.float32),
            "u_inter_raw": np.asarray(u_vals, dtype=np.float32),
            "u_inter_norm": np.asarray(u_norm_vals, dtype=np.float32),
            "q_intra": np.asarray(q_intra_vals, dtype=np.float32),
            "p_inter": np.asarray(p_vals, dtype=np.float32),
            "q_mix": np.asarray(q_mix_vals, dtype=np.float32),
            "z_ehh_rms_from_tau0": np.asarray(z_vals, dtype=np.float32),
            "z_mode": self.horizon_intra_z_mode,
            "horizon_exec_mode": self.horizon_exec_mode,
            "horizon_ts_update_mode": self.horizon_ts_update_mode,
            "horizon_ts_kernel_bandwidth": float(self.horizon_ts_kernel_bandwidth),
            "horizon_ts_forget_rho": float(self.horizon_ts_forget_rho),
            "horizon_ts_update_eta": float(self.horizon_ts_update_eta),
            "horizon_expected_temp": float(self.horizon_expected_temp),
            "tau_window": int(self.horizon_intra_tau_window),
        }

    def _select_exec_k_horizon_from_info(self, horizon_info: dict[str, Any], used_h: int) -> tuple[int, dict[str, Any]]:
        horizons = np.asarray(horizon_info.get("horizons", []), dtype=np.int32).tolist()
        if not horizons:
            horizons = [int(used_h)]
        q_mix_vals = np.asarray(horizon_info.get("q_mix", []), dtype=np.float64).tolist()
        if not q_mix_vals:
            q_mix_vals = [1.0 for _ in horizons]

        score_vals = []
        for s, q_mix in zip(horizons, q_mix_vals, strict=True):
            a, b = self._horizon_ts_state.get(int(s), [1.0, 1.0])
            theta = float(self._horizon_ts_rng.beta(a, b))
            score_vals.append(float(theta * q_mix))

        score_arr = np.asarray(score_vals, dtype=np.float64)
        expected_prob = self._normalize_score_distribution(score_arr, temp=1.0)
        expected_horizon = None
        if self.horizon_exec_mode == "expected_round":
            expected_prob = self._normalize_score_distribution(score_arr, temp=self.horizon_expected_temp)
        if expected_prob.size > 0:
            expected_horizon = float(np.dot(expected_prob, np.asarray(horizons, dtype=np.float64)))

        eps_rand = max(0.0, min(1.0, self.horizon_ts_epsilon))
        if float(self._horizon_ts_rng.random()) < eps_rand:
            chosen_idx = int(self._horizon_ts_rng.integers(0, len(horizons)))
            choose_mode = "epsilon_random"
            k_exec = int(horizons[chosen_idx])
            update_center, update_feedback = self._apply_horizon_ts_update(
                horizons=horizons,
                q_mix_vals=q_mix_vals,
                chosen_idx=chosen_idx,
                choose_mode=choose_mode,
                k_exec=k_exec,
            )
        elif self.horizon_exec_mode == "expected_round":
            if expected_horizon is None or not np.isfinite(expected_horizon):
                expected_horizon = float(np.mean(np.asarray(horizons, dtype=np.float64)))
            k_exec = int(np.clip(int(round(expected_horizon)), 1, int(used_h)))
            choose_mode = "expected_round"
            update_center, update_feedback = self._apply_horizon_ts_update(
                horizons=horizons,
                q_mix_vals=q_mix_vals,
                chosen_idx=-1,
                choose_mode=choose_mode,
                k_exec=k_exec,
                expected_prob=expected_prob,
            )
        else:
            chosen_idx = int(np.argmax(score_arr))
            choose_mode = "thompson"
            k_exec = int(horizons[chosen_idx])
            update_center, update_feedback = self._apply_horizon_ts_update(
                horizons=horizons,
                q_mix_vals=q_mix_vals,
                chosen_idx=chosen_idx,
                choose_mode=choose_mode,
                k_exec=k_exec,
            )

        out = dict(horizon_info)
        out["scores"] = np.asarray(score_vals, dtype=np.float32)
        out["expected_horizon"] = expected_horizon
        out["choose_mode"] = choose_mode
        out["chosen_horizon"] = int(k_exec)
        out["update_center"] = float(update_center)
        out["update_feedback"] = float(update_feedback)
        return k_exec, out

    @staticmethod
    def _resolve_checkpoint_root(
        train_config_name: str,
        model_name: str,
        checkpoint_id: str,
        checkpoint_root_tag: str | None = None,
    ) -> Path:
        root_name = checkpoint_root_tag or train_config_name
        relative = Path(root_name) / model_name / str(checkpoint_id)
        module_root = Path(__file__).resolve().parent
        candidates = [
            module_root / "checkpoints" / relative,
            Path("policy/pi0/checkpoints") / relative,
            Path("checkpoints") / relative,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return candidates[0].resolve()

    @staticmethod
    def _resolve_assets_id(assets_dir: Path) -> str:
        if not assets_dir.exists():
            raise FileNotFoundError(f"assets directory not found: {assets_dir}")
        dirs = sorted([p.name for p in assets_dir.iterdir() if p.is_dir()])
        if dirs:
            return dirs[0]
        entries = sorted([p.name for p in assets_dir.iterdir()])
        if not entries:
            raise FileNotFoundError(f"assets directory is empty: {assets_dir}")
        return entries[0]

    @staticmethod
    def _inspect_checkpoint_metadata(metadata_path: Path) -> dict[str, bool | None]:
        info = {"has_qha": None, "has_horizon_head": None}
        if not metadata_path.exists():
            return info
        try:
            has_qha = False
            has_horizon_head = False
            with metadata_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "horizon_head" in line:
                        has_horizon_head = True
                    if "qha" in line:
                        has_qha = True
            info["has_qha"] = has_qha
            info["has_horizon_head"] = has_horizon_head
        except OSError:
            pass
        return info

    def _make_runtime_selector(self, candidate_horizons) -> _qha.HybridQHARuntimeSelector:
        return _qha.HybridQHARuntimeSelector(
            candidate_horizons=candidate_horizons,
            **self._runtime_selector_kwargs,
        )

    def _build_history_inputs(self):
        if self.history_max_length <= 0:
            return None, None
        hist_actions = np.zeros((self.history_max_length, 14), dtype=np.float32)
        hist_actions_mask = np.zeros((self.history_max_length,), dtype=np.bool_)
        history = np.asarray(self._executed_action_history, dtype=np.float32)
        if history.ndim == 2 and history.shape[0] > 0:
            tail = history[-self.history_max_length:, :14]
            n = int(tail.shape[0])
            hist_actions[-n:, : tail.shape[1]] = tail
            hist_actions_mask[-n:] = True
        return hist_actions, hist_actions_mask

    def _select_exec_length_qha(self, outputs: dict, *, max_exec_length: int) -> tuple[int, dict]:
        posterior_dense = np.asarray(outputs["qha_posterior"], dtype=np.float32).reshape(-1)
        candidate_horizons_dense = tuple(
            int(v) for v in np.asarray(outputs.get("qha_candidate_horizons", self.qha_candidate_horizons)).reshape(-1)
        )
        if len(candidate_horizons_dense) != int(posterior_dense.size):
            fallback_dense = tuple(int(v) for v in self.qha_candidate_horizons)
            if len(fallback_dense) == int(posterior_dense.size):
                candidate_horizons_dense = fallback_dense
            elif int(posterior_dense.size) == int(max_exec_length):
                candidate_horizons_dense = tuple(range(1, int(max_exec_length) + 1))
        runtime_candidate_horizons = candidate_horizons_dense
        runtime_posterior = posterior_dense

        if self.qha_infer_mode == "posterior_only":
            if self.qha_eval_variant == "posterior_sparse_compromise":
                runtime_candidate_horizons = self._runtime_sparse_basis
                runtime_posterior = _compress_dense_posterior_to_sparse_basis(
                    np.asarray(candidate_horizons_dense, dtype=np.int32),
                    posterior_dense,
                    np.asarray(runtime_candidate_horizons, dtype=np.int32),
                )
                expected = float(np.dot(runtime_posterior, np.asarray(runtime_candidate_horizons, dtype=np.float32)))
            else:
                expected = float(np.asarray(outputs["qha_expected_horizon"]).item())
            exec_length = int(np.clip(int(round(expected)), 1, max_exec_length))
            return exec_length, {
                "qha_exec_mode": "posterior_only",
                "qha_runtime_theta": None,
                "qha_runtime_scores": None,
                "qha_runtime_effective_scores": None,
                "qha_runtime_expected_prob": None,
                "qha_runtime_feedback": None,
                "qha_runtime_candidate_horizons": np.asarray(runtime_candidate_horizons, dtype=np.int32),
            }

        if "qha_selector_q_mix" not in outputs:
            raise KeyError("Hybrid QHA inference requires qha_selector_q_mix in policy outputs.")
        q_mix_dense = np.asarray(outputs["qha_selector_q_mix"], dtype=np.float32).reshape(-1)
        sparse_runtime_horizons = tuple(
            int(v)
            for v in np.asarray(
                outputs.get("qha_selector_candidate_horizons_sparse_runtime", self._runtime_sparse_basis),
                dtype=np.int32,
            ).reshape(-1)
        )
        q_mix_sparse_runtime = outputs.get("qha_selector_q_mix_sparse_runtime")
        if q_mix_sparse_runtime is not None:
            q_mix_sparse_runtime = np.asarray(q_mix_sparse_runtime, dtype=np.float32).reshape(-1)
        elif tuple(candidate_horizons_dense) == tuple(sparse_runtime_horizons):
            q_mix_sparse_runtime = q_mix_dense
        if q_mix_sparse_runtime is not None and (
            len(sparse_runtime_horizons) != int(q_mix_sparse_runtime.size)
            or (
                int(q_mix_sparse_runtime.size) == 1
                and len(self._runtime_sparse_basis) > 1
                and int(posterior_dense.size) > 1
            )
        ):
            fallback_basis = tuple(int(v) for v in self._runtime_sparse_basis)
            if len(fallback_basis) == int(q_mix_sparse_runtime.size):
                sparse_runtime_horizons = fallback_basis
            elif int(q_mix_dense.size) == len(fallback_basis):
                sparse_runtime_horizons = fallback_basis
                q_mix_sparse_runtime = q_mix_dense
            elif int(q_mix_dense.size) == len(candidate_horizons_dense):
                sparse_runtime_horizons = candidate_horizons_dense
                q_mix_sparse_runtime = q_mix_dense

        if self.qha_eval_variant == "posterior_sparse_compromise":
            runtime_candidate_horizons = sparse_runtime_horizons
            runtime_posterior = _compress_dense_posterior_to_sparse_basis(
                np.asarray(candidate_horizons_dense, dtype=np.int32),
                posterior_dense,
                np.asarray(runtime_candidate_horizons, dtype=np.int32),
            )
            if q_mix_sparse_runtime is None:
                raise KeyError("posterior_sparse_compromise requires qha_selector_q_mix_sparse_runtime.")
            runtime_q_mix = q_mix_sparse_runtime
        elif self.qha_eval_variant == "selector_dense_interpolation":
            runtime_candidate_horizons = candidate_horizons_dense
            runtime_posterior = posterior_dense
            if q_mix_sparse_runtime is None:
                raise KeyError("selector_dense_interpolation requires qha_selector_q_mix_sparse_runtime.")
            runtime_q_mix = _interpolate_sparse_scores_to_dense_basis(
                np.asarray(sparse_runtime_horizons, dtype=np.int32),
                q_mix_sparse_runtime,
                np.asarray(runtime_candidate_horizons, dtype=np.int32),
            )
        else:
            runtime_candidate_horizons = candidate_horizons_dense
            runtime_q_mix = q_mix_dense

        if self._runtime_selector is None or tuple(self._runtime_selector.candidate_horizons) != runtime_candidate_horizons:
            self._runtime_selector = self._make_runtime_selector(runtime_candidate_horizons)
        result = self._runtime_selector.select(runtime_posterior, runtime_q_mix, max_exec_length=max_exec_length)
        return int(result["exec_length"]), {
            "qha_exec_mode": result["choose_mode"],
            "qha_runtime_theta": result["runtime_theta"],
            "qha_runtime_scores": result["runtime_scores"],
            "qha_runtime_effective_scores": result["runtime_effective_scores"],
            "qha_runtime_expected_prob": result["runtime_expected_prob"],
            "qha_runtime_feedback": result["runtime_feedback"],
            "qha_runtime_candidate_horizons": np.asarray(runtime_candidate_horizons, dtype=np.int32),
        }

    def set_img_size(self, img_size):
        self.img_size = img_size

    # set language randomly
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, puppet_arm = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
            state,
        )
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }
        hist_actions, hist_actions_mask = self._build_history_inputs()
        if hist_actions is not None and hist_actions_mask is not None:
            self.observation_window["hist_actions"] = hist_actions
            self.observation_window["hist_actions_mask"] = hist_actions_mask

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        t_infer0 = time.time()
        infer_output = self.policy.infer(self.observation_window)
        self._policy_infer_seconds += float(time.time() - t_infer0)
        actions = infer_output["actions"]
        trace = infer_output.get("denoise_trace")
        t_dump_path0 = time.time() if (self.dump_denoise and trace is not None) else None
        if self.valid_action_dim is None:
            # Fallback for paths where left/right dims are not provided.
            self.valid_action_dim = int(self.observation_window["state"].shape[-1])
        used_h = min(int(self.pi0_step), int(actions.shape[0]))
        k_exec = used_h
        v = None
        x0 = None
        xt_full = actions[:used_h, :min(self.valid_action_dim, int(actions.shape[-1]))]
        xt_internal_full = None
        horizon_info: dict[str, Any] | None = None
        action_only_info: dict[str, Any] | None = None
        if self.use_qha and "qha_posterior" in infer_output:
            t_select0 = time.time()
            k_exec, qha_runtime_info = self._select_exec_length_qha(infer_output, max_exec_length=used_h)
            self._horizon_selection_seconds += float(time.time() - t_select0)
            horizon_info = {
                "q_intra": infer_output.get("qha_selector_q_intra"),
                "p_inter": infer_output.get("qha_selector_p_inter"),
                "q_mix": infer_output.get("qha_selector_q_mix"),
                "z_intra": infer_output.get("qha_selector_z_intra_raw"),
                "z_intra_norm": infer_output.get("qha_selector_z_intra_norm"),
                "u_inter_raw": infer_output.get("qha_selector_u_inter_raw"),
                "u_inter_norm": infer_output.get("qha_selector_u_inter_norm"),
                **qha_runtime_info,
            }
        if trace is not None and not self.use_qha:
            valid_dim = min(self.valid_action_dim, int(actions.shape[-1]), int(trace["v"].shape[-1]))
            v = trace["v"]
            if v.shape[-1] > valid_dim:
                v = v[..., :valid_dim]
            xt_full = actions[:used_h, :valid_dim]
            xt_internal_raw = trace["xt_internal"]
            xt_internal_full = xt_internal_raw[:used_h, :valid_dim]
            if "x0" in trace:
                x0_raw = trace["x0"]
                x0 = x0_raw[:used_h, :valid_dim]
            if self.adaptive_selector_mode != "none":
                t_select0 = time.time()
                action_only_info = self._select_adaptive_exec_k(xt_full, used_h, trace)
                self._horizon_selection_seconds += float(time.time() - t_select0)
                k_exec = int(action_only_info["chosen_k"])
            elif self.eval_type == "horizon":
                v_np = np.asarray(jax.device_get(v[:, :used_h, :]), dtype=np.float32)
                t_metric0 = time.time()
                horizon_info = self._compute_horizon_info(v_np, used_h, xt_step=np.asarray(xt_full, dtype=np.float32))
                self._horizon_metric_compute_seconds += float(time.time() - t_metric0)
                t_select0 = time.time()
                k_exec, horizon_info = self._select_exec_k_horizon_from_info(horizon_info, used_h)
                self._horizon_selection_seconds += float(time.time() - t_select0)
                self._update_run_score_stats(horizon_info)
        self.last_exec_k = int(np.clip(k_exec, 1, used_h))
        if self.eval_type != "horizon" and self.fixed_exec_k > 0:
            self.last_exec_k = int(np.clip(self.fixed_exec_k, 1, used_h))
        self._last_used_h = int(used_h)
        self._last_action_dim = int(xt_full.shape[-1]) if xt_full.ndim == 2 else int(self.valid_action_dim or 0)
        if self.dump_denoise and trace is not None:
            xt_log = np.array(xt_full, dtype=np.float32, copy=True)
            xt_internal_log = np.array(xt_internal_full, dtype=np.float32, copy=True)
            self.denoise_logger.log_step(
                v,
                ds=trace.get("ds"),
                x0=x0,
                xt=xt_log,
                xt_internal=xt_internal_log,
                k_exec=int(self.last_exec_k),
                monotonic_timestamp_ns=time.monotonic_ns(),
                q_intra=None if horizon_info is None else horizon_info.get("q_intra"),
                p_inter=None if horizon_info is None else horizon_info.get("p_inter"),
                q_mix=None if horizon_info is None else horizon_info.get("q_mix"),
                z_intra_raw=None if horizon_info is None else horizon_info.get("z_intra"),
                z_intra_norm=None if horizon_info is None else horizon_info.get("z_intra_norm"),
                u_inter_raw=None if horizon_info is None else horizon_info.get("u_inter_raw"),
                u_inter_norm=None if horizon_info is None else horizon_info.get("u_inter_norm"),
                action_only_candidate_metrics=None if action_only_info is None else action_only_info["candidate_metrics"],
                action_only_tie_count=None if action_only_info is None else action_only_info["tie_count"],
                action_only_continuous_action_dimensions=(
                    None if action_only_info is None else action_only_info["continuous_action_dimensions"]
                ),
            )
        if t_dump_path0 is not None:
            # Full dump-path overhead inside get_action for this replan step.
            self._denoise_overhead_seconds += float(time.time() - t_dump_path0)
        return actions

    def ensure_valid_action_dim(self, observed_dim):
        observed_dim = int(observed_dim)
        if self.valid_action_dim is None:
            self.valid_action_dim = observed_dim
            return
        # Keep logger dims consistent with actual runtime action/state vector.
        if self.valid_action_dim != observed_dim:
            self.valid_action_dim = observed_dim

    def record_executed_actions(self, executed_actions, env_step_start) -> None:
        exec_arr = np.asarray(executed_actions, dtype=np.float32)
        if exec_arr.ndim != 2:
            return
        if exec_arr.shape[0] <= 0 or exec_arr.shape[1] <= 0:
            return
        if self._executed_action_history.size == 0 or self._executed_action_history.shape[1] != exec_arr.shape[1]:
            self._executed_action_history = exec_arr.copy()
        else:
            self._executed_action_history = np.concatenate([self._executed_action_history, exec_arr], axis=0)

        pad_h = max(int(self._last_used_h), int(exec_arr.shape[0]))
        a_exec = np.zeros((pad_h, int(exec_arr.shape[1])), dtype=np.float32)
        a_exec[: exec_arr.shape[0], :] = exec_arr
        a_exec_mask = np.zeros((pad_h,), dtype=np.float32)
        a_exec_mask[: exec_arr.shape[0]] = 1.0
        payload: dict[str, Any] = {
            "A_exec": a_exec,
            "A_exec_mask": a_exec_mask,
            "replan_env_step_idx": int(env_step_start),
            "env_step_idx": int(env_step_start),
            "mp4_frame_idx": int(env_step_start),
            "K_exec": int(exec_arr.shape[0]),
            "selected_K": int(exec_arr.shape[0]),
            "executed_action_start": int(env_step_start),
            "executed_action_end": int(env_step_start) + int(exec_arr.shape[0]),
            "executed_action_range": np.asarray(
                [int(env_step_start), int(env_step_start) + int(exec_arr.shape[0])],
                dtype=np.int32,
            ),
        }
        gripper_indices = self._infer_gripper_indices(exec_arr.shape[1])
        if gripper_indices:
            payload["gripper_exec"] = np.asarray(a_exec[:, gripper_indices], dtype=np.float32)
        self.denoise_logger.update_last_step(**payload)

    def prepare_episode_logging(self, eval_video_path, episode_idx):
        t0 = time.time()
        if eval_video_path is None:
            if self.denoise_log_dir is None:
                return
            episode_path = os.path.join(self.denoise_log_dir, f"episode{episode_idx}.npz")
        else:
            episode_path = os.path.join(eval_video_path, f"episode{episode_idx}.npz")

        if self._episode_idx == episode_idx and self._episode_path == episode_path:
            return

        self._episode_idx = episode_idx
        self._episode_path = episode_path
        if self.dump_denoise:
            meta = {
                "train_config_name": self.train_config_name,
                "model_name": self.model_name,
                "checkpoint_id": self.checkpoint_id,
                "pi0_step": self.pi0_step,
                "valid_action_dim": self.valid_action_dim,
                "eval_type": self.eval_type,
                "horizon_policy": self._horizon_policy_name(),
                "horizon_candidates": self.horizon_candidates,
                "horizon_intra_alpha": self.horizon_intra_alpha,
                "horizon_ts_epsilon": self.horizon_ts_epsilon,
                "horizon_ts_seed": self.horizon_ts_seed,
                "horizon_ts_update_mode": self.horizon_ts_update_mode,
                "horizon_ts_kernel_bandwidth": self.horizon_ts_kernel_bandwidth,
                "horizon_ts_forget_rho": self.horizon_ts_forget_rho,
                "horizon_ts_update_eta": self.horizon_ts_update_eta,
                "horizon_intra_z_mode": self.horizon_intra_z_mode,
                "horizon_exec_mode": self.horizon_exec_mode,
                "horizon_expected_temp": self.horizon_expected_temp,
                "horizon_intra_tau_window": self.horizon_intra_tau_window,
                "horizon_inter_use_speed_uniformity": int(self.horizon_inter_use_speed_uniformity),
                "horizon_mix_mode": self.horizon_mix_mode,
                "horizon_inter_window": self.horizon_inter_window,
                "horizon_inter_weight": self.horizon_inter_weight,
                "horizon_inter_beta": self.horizon_inter_beta,
                "horizon_inter_fallback": self.horizon_inter_fallback,
                "checkpoint_root_tag": self.checkpoint_root_tag,
                "use_qha": int(self.use_qha),
                "qha_eval_variant": self.qha_eval_variant,
                "qha_infer_mode": self.qha_infer_mode,
                "qha_candidate_horizons": self.qha_candidate_horizons,
                "qha_hybrid_prior_gamma": self.qha_hybrid_prior_gamma,
                "adaptive_selector_mode": self.adaptive_selector_mode,
                "adaptive_selector_score_source": (
                    "same_rollout_denoising_prefix_agreement"
                    if self.adaptive_selector_mode == "cross_horizon_agreement"
                    else "single_action_chunk" if self.adaptive_selector_mode != "none" else "none"
                ),
            }
            meta.update(self._build_episode_meta())
            self.denoise_logger.start_episode(episode_path, meta=meta)
            self._denoise_overhead_seconds += float(time.time() - t0)

    def finalize_episode_logging(self, success=None, success_step=None, episode_steps=None, run_time_seconds=None):
        if not self.dump_denoise:
            # Keep a lightweight per-episode npz even when denoise dumping is disabled.
            if self._episode_path is None:
                return None
            total = float(run_time_seconds or 0.0)
            # Keep the same schema as dump_denoise=1 for apples-to-apples comparison:
            # [total_episode_seconds, denoise_overhead_seconds].
            out = {
                "run_time_seconds": np.asarray([total, 0.0], dtype=np.float32),
                "policy_infer_seconds_total": np.asarray(float(self._policy_infer_seconds), dtype=np.float32),
                "horizon_metric_compute_seconds_total": np.asarray(
                    float(self._horizon_metric_compute_seconds),
                    dtype=np.float32,
                ),
                "horizon_selection_seconds_total": np.asarray(
                    float(self._horizon_selection_seconds),
                    dtype=np.float32,
                ),
            }
            if episode_steps is not None:
                out["episode_steps"] = np.asarray(int(episode_steps), dtype=np.int32)
            np.savez(self._episode_path, **out)
            return Path(self._episode_path)
        self.denoise_logger.set_episode_info(
            success=success,
            success_step=success_step,
            episode_steps=episode_steps,
            run_time_seconds=run_time_seconds,
            denoise_overhead_seconds=self._denoise_overhead_seconds,
            policy_infer_seconds_total=self._policy_infer_seconds,
            horizon_metric_compute_seconds_total=self._horizon_metric_compute_seconds,
            horizon_selection_seconds_total=self._horizon_selection_seconds,
        )
        return self.denoise_logger.flush_episode()

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        self._episode_context = {}
        self._policy_infer_seconds = 0.0
        self._horizon_metric_compute_seconds = 0.0
        self._horizon_selection_seconds = 0.0
        if self.dump_denoise:
            self._episode_idx = None
            self._episode_path = None
            self.denoise_logger.reset_episode()
            self._denoise_overhead_seconds = 0.0
        self.last_exec_k = None
        self._last_used_h = int(self.pi0_step)
        self._last_action_dim = int(self.valid_action_dim) if self.valid_action_dim is not None else 0
        self._horizon_ts_rng = np.random.default_rng(self.horizon_ts_seed)
        self._horizon_ts_state = {int(s): [1.0, 1.0] for s in self.horizon_candidates}
        if self._runtime_selector is not None:
            self._runtime_selector.reset()
        self._executed_action_history = np.zeros((0, 0), dtype=np.float32)
        self._reset_run_score_state()
        print("successfully unset obs and language intruction")
        self.trace_enabled = self.dump_denoise or (self.eval_type == "horizon")
