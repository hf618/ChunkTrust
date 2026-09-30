#!/usr/bin/env python3
"""Validate aligned horizon fields in latest info.json for each family."""

from __future__ import annotations
from chunktrust.paths import resolve_legacy_path as _ct_path

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path


REQUIRED_AVG_KEYS = [
    "avg_q_intra",
    "avg_p_inter",
    "avg_q_mix",
    "avg_q_intra_contrib",
    "avg_p_inter_contrib",
    "avg_z_intra_raw",
    "avg_z_intra_norm",
    "avg_u_inter_raw",
    "avg_u_inter_norm",
]

REQUIRED_HORIZON_KEYS = [
    "horizon_policy",
    "horizon_candidates",
    "horizon_exec_mode",
    "horizon_expected_temp",
    "horizon_intra_alpha",
    "horizon_intra_z_mode",
    "horizon_intra_tau_window",
    "horizon_inter_weight",
    "horizon_inter_beta",
    "horizon_inter_window",
    "horizon_ts_update_mode",
    "horizon_ts_kernel_bandwidth",
    "horizon_ts_forget_rho",
    "horizon_ts_update_eta",
]

FAMILY_ROOTS = {
    "xvla": [
        Path(_ct_path('CHUNKTRUST_WORKSPACE_ROOT', 'X-VLA/evaluation/robotwin-2.0/eval_results')),
        Path(_ct_path('CHUNKTRUST_RESULTS_ROOT', 'robotwin2.0/eval_results')),
    ],
    "pi": [
        Path(_ct_path('ROBOTWIN_ROOT', 'eval_result')),
        Path(_ct_path('ROBOTWIN_ROOT', 'policy/pi0/eval_result')),
        Path(_ct_path('ROBOTWIN_ROOT', 'policy/pi05_horizon/eval_result')),
        Path(_ct_path('CHUNKTRUST_RESULTS_ROOT', 'robotwin2.0/eval_results')),
    ],
    "starvla": [
        Path(_ct_path('CHUNKTRUST_WORKSPACE_ROOT', 'starVLA/examples/Robocasa_tabletop/eval_result')),
        Path(_ct_path('CHUNKTRUST_WORKSPACE_ROOT', 'starVLA/eval_result')),
        Path(_ct_path('CHUNKTRUST_RESULTS_ROOT', 'robocasa-gr1-tabletop-tasks/eval_results')),
    ],
    "gr00t_n16": [
        Path(_ct_path('CHUNKTRUST_WORKSPACE_ROOT', 'Isaac-GR00T/examples/robocasa-gr1-tabletop-tasks/eval_result')),
        Path(_ct_path('CHUNKTRUST_RESULTS_ROOT', 'robocasa-gr1-tabletop-tasks/eval_results')),
    ],
    "gr00t_n15": [
        Path(_ct_path('CHUNKTRUST_WORKSPACE_ROOT', 'Isaac-GR00T-n1.5/examples/RoboCasa/eval_result')),
        Path(_ct_path('CHUNKTRUST_RESULTS_ROOT', 'robocasa-gr1-tabletop-tasks/eval_results')),
    ],
}


def _extract_timestamp(path: Path) -> datetime:
    for part in reversed(path.parts):
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", part):
            try:
                return datetime.strptime(part, "%Y-%m-%d_%H-%M-%S")
            except ValueError:
                pass
    return datetime.fromtimestamp(path.stat().st_mtime)


def _family_matcher(family: str, info_path: Path) -> bool:
    low = str(info_path).lower()
    if "/info/info.json" not in low:
        return False
    if family == "xvla":
        return "/xvla/" in low
    if family == "pi":
        return "/xvla/" not in low and "qwen3gr00t" not in low and "gr00t" not in low
    if family == "starvla":
        return "qwen3gr00t" in low or "starvla" in low
    if family == "gr00t_n16":
        return "n1.6" in low or "n1d6" in low or "gr00t-n1.6" in low
    if family == "gr00t_n15":
        return "n1.5" in low or "gr00t-n1.5" in low or "posttrain" in low
    return False


def _find_latest_info(family: str) -> Path | None:
    candidates: list[Path] = []
    for root in FAMILY_ROOTS[family]:
        if not root.exists():
            continue
        for path in root.rglob("info.json"):
            if _family_matcher(family, path):
                candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=_extract_timestamp)


def _validate_file(path: Path, require_nonnull: bool) -> tuple[list[str], list[str]]:
    info = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in (REQUIRED_AVG_KEYS + REQUIRED_HORIZON_KEYS) if k not in info]
    null_or_nan: list[str] = []
    if require_nonnull:
        for key in REQUIRED_AVG_KEYS:
            if key not in info:
                continue
            value = info[key]
            if value is None:
                null_or_nan.append(key)
            elif isinstance(value, float) and value != value:
                null_or_nan.append(key)
    return missing, null_or_nan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families",
        nargs="+",
        default=["all"],
        choices=["all", "xvla", "pi", "starvla", "gr00t_n16", "gr00t_n15"],
    )
    parser.add_argument(
        "--require-nonnull-avg",
        action="store_true",
        help="Fail when avg_* keys exist but are null/NaN.",
    )
    args = parser.parse_args()

    families = list(FAMILY_ROOTS.keys()) if "all" in args.families else args.families
    failed = False

    for family in families:
        latest = _find_latest_info(family)
        if latest is None:
            print(f"[{family}] FAIL: no info.json found", flush=True)
            failed = True
            continue
        missing, null_or_nan = _validate_file(latest, require_nonnull=args.require_nonnull_avg)
        print(f"[{family}] latest={latest}", flush=True)
        if missing:
            print(f"  missing_keys({len(missing)}): {', '.join(missing)}", flush=True)
            failed = True
        else:
            print("  missing_keys(0): none", flush=True)
        if null_or_nan:
            print(f"  null_or_nan_avg({len(null_or_nan)}): {', '.join(null_or_nan)}", flush=True)
            failed = True
        elif args.require_nonnull_avg:
            print("  null_or_nan_avg(0): none", flush=True)

    if failed:
        print("VALIDATION_FAILED", flush=True)
        return 1
    print("VALIDATION_PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
