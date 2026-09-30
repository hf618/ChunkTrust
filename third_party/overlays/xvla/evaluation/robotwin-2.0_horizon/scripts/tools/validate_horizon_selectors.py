#!/usr/bin/env python3
"""Validate horizon selector invariants across StarVLA / GR00T selector copies.

Checks:
1) smallest finite z_raw -> q_intra == 1.0
2) all-equal finite z_raw -> z_norm == 0 for all finite entries
3) NaN z_raw excluded from min-max normalization
4) mixed-mode fallback behavior stays consistent
"""

from __future__ import annotations
from chunktrust.paths import resolve_legacy_path as _ct_path

import importlib.util
from pathlib import Path

import numpy as np


SELECTOR_PATHS = {
    "starvla": Path(_ct_path('CHUNKTRUST_WORKSPACE_ROOT', 'starVLA/examples/Robocasa_tabletop/eval_files/stride_selector.py')),
    "gr00t_n16": Path(_ct_path('CHUNKTRUST_WORKSPACE_ROOT', 'Isaac-GR00T/gr00t/eval/stride_selector.py')),
    "gr00t_n15": Path(_ct_path('CHUNKTRUST_WORKSPACE_ROOT', 'Isaac-GR00T-n1.5/gr00t/eval/stride_selector.py')),
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _assert_close(a: float, b: float, tol: float = 1e-6) -> None:
    if np.isnan(a) and np.isnan(b):
        return
    if abs(float(a) - float(b)) > tol:
        raise AssertionError(f"{a} != {b}")


def _run_selector_invariants(module_name: str, module) -> None:
    orig_ehh = module.ehh_curves_by_stride_batched
    orig_z = module.compute_z_from_ehh
    orig_speed = module.speed_uniformity_proxy_by_stride

    try:
        def fake_ehh(_v_step, strides, h_full=None, **_kwargs):  # noqa: ARG001
            out = {}
            for s in strides:
                out[int(s)] = (np.array([float(s)], dtype=np.float32), 0.0)
            return out

        z_map = {4: 3.0, 8: 1.0, 12: 2.0}

        def fake_z(e_vals, tau0_idx=0, mode="window_rms", tau_window=3):  # noqa: ARG001
            s = int(round(float(np.asarray(e_vals).reshape(-1)[0])))
            return float(z_map[s])

        module.ehh_curves_by_stride_batched = fake_ehh
        module.compute_z_from_ehh = fake_z
        module.speed_uniformity_proxy_by_stride = (
            lambda *args, **kwargs: ([np.nan] * 3, [np.nan] * 3, [np.nan] * 3)
        )

        # Case 1: minimum finite z must map to q_intra=1.0
        _, info = module.select_exec_k_stride(
            v_step=np.zeros((2, 16, 3), dtype=np.float32),
            used_h=16,
            stride_candidates="4,8,12",
            signal_mode="intra",
            use_speed_uniformity=False,
            exec_mode="greedy",
            ts_update_mode="legacy",
            ts_epsilon=0.0,
            stride_alpha=4.0,
        )
        strides = info["strides"]
        q_intra = info["q_intra"]
        idx_min = strides.index(8)
        _assert_close(float(q_intra[idx_min]), 1.0)

        # Case 2: equal finite z -> all z_norm == 0
        z_map = {4: 5.0, 8: 5.0, 12: 5.0}
        _, info = module.select_exec_k_stride(
            v_step=np.zeros((2, 16, 3), dtype=np.float32),
            used_h=16,
            stride_candidates="4,8,12",
            signal_mode="intra",
            use_speed_uniformity=False,
            exec_mode="greedy",
            ts_update_mode="legacy",
            ts_epsilon=0.0,
            stride_alpha=4.0,
        )
        for z in info["z_intra_norm"]:
            _assert_close(float(z), 0.0)

        # Case 3: NaN excluded from normalization
        z_map = {4: np.nan, 8: 2.0, 12: 4.0}
        _, info = module.select_exec_k_stride(
            v_step=np.zeros((2, 16, 3), dtype=np.float32),
            used_h=16,
            stride_candidates="4,8,12",
            signal_mode="intra",
            use_speed_uniformity=False,
            exec_mode="greedy",
            ts_update_mode="legacy",
            ts_epsilon=0.0,
            stride_alpha=4.0,
        )
        z_norm = info["z_intra_norm"]
        if not np.isnan(z_norm[0]):
            raise AssertionError(f"{module_name}: expected NaN at z_norm[0], got {z_norm[0]}")
        _assert_close(float(z_norm[1]), 0.0)
        _assert_close(float(z_norm[2]), 1.0)

        # Case 4: both-mode fallback semantics
        z_map = {4: 0.0, 8: 0.0, 12: 0.0}
        module.speed_uniformity_proxy_by_stride = lambda *args, **kwargs: (  # noqa: E731
            [3.0, 2.0, 1.0],
            [0.1, 0.2, 0.3],
            [0.9, np.nan, 0.1],
        )
        _, info = module.select_exec_k_stride(
            v_step=np.zeros((2, 16, 3), dtype=np.float32),
            used_h=16,
            stride_candidates="4,8,12",
            signal_mode="both",
            use_speed_uniformity=True,
            speed_weight=0.5,
            speed_fallback="ehh_only",
            xt_step=np.zeros((16, 6), dtype=np.float32),
            history_actions=np.zeros((10, 6), dtype=np.float32),
            exec_mode="greedy",
            ts_update_mode="legacy",
            ts_epsilon=0.0,
            stride_alpha=4.0,
        )
        _assert_close(float(info["q_mix"][0]), 0.95)
        _assert_close(float(info["q_mix"][1]), 1.0)  # fallback ehh_only -> q_intra
        _assert_close(float(info["q_mix"][2]), 0.55)

    finally:
        module.ehh_curves_by_stride_batched = orig_ehh
        module.compute_z_from_ehh = orig_z
        module.speed_uniformity_proxy_by_stride = orig_speed


def main() -> None:
    for name, path in SELECTOR_PATHS.items():
        mod = _load_module(f"selector_{name}", path)
        _run_selector_invariants(name, mod)
        print(f"[PASS] selector invariants: {name}")
    print("ALL_SELECTOR_TESTS_PASSED")


if __name__ == "__main__":
    main()
