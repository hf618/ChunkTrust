#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _to_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_info_json(path_str: str | None) -> Dict[str, Any]:
    if not path_str:
        return {}
    path = Path(path_str)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _collect_k_exec_values(run_dir: Path) -> List[float]:
    episode_dir = run_dir / "episodes"
    if not episode_dir.exists():
        return []

    values: List[float] = []
    for npz_path in sorted(episode_dir.rglob("*.npz")):
        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception:
            continue
        try:
            if "K_exec" not in data.files:
                continue
            arr = np.asarray(data["K_exec"]).reshape(-1)
            if arr.size == 0:
                continue
            values.extend([float(x) for x in arr.tolist()])
        finally:
            data.close()
    return values


def _fmt_num(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return "N/A"
    return str(value)


def build_task_metrics(summary_csv: Path, output_csv: Path) -> Tuple[int, int]:
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    output_rows: List[Dict[str, str]] = []

    success_values: List[float] = []
    pooled_exec_values: List[float] = []

    task_count = 0
    for row in rows:
        task_name = str(row.get("task_name", "")).strip()
        if not task_name:
            continue
        task_count += 1

        info = _read_info_json(row.get("info_json"))
        run_dir = Path(str(row.get("run_dir", "")).strip())

        eval_type = str(info.get("eval_type", "default")).strip().lower()
        n_action_steps = _to_int(info.get("n_action_steps"))
        n_episodes = _to_int(info.get("n_episodes"))

        success_rate = _to_float(info.get("success_rate"))
        if success_rate is None:
            sr_percent = _to_float(row.get("success_rate_percent"))
            if sr_percent is not None:
                success_rate = sr_percent / 100.0
        if success_rate is not None:
            success_values.append(success_rate)

        exec_values: List[float] = []
        if run_dir.exists():
            # Prefer executed chunk traces for any eval mode when available.
            exec_values = _collect_k_exec_values(run_dir)
        if not exec_values:
            # Fallback for runs without per-step execution traces.
            if n_action_steps is not None and n_episodes is not None and n_episodes > 0:
                exec_values = [float(n_action_steps)] * int(n_episodes)
            elif n_action_steps is not None:
                exec_values = [float(n_action_steps)]

        exec_chunk_mean = float(np.mean(exec_values)) if exec_values else None
        exec_chunk_std = float(np.std(exec_values)) if exec_values else None
        exec_chunk_n = len(exec_values)
        pooled_exec_values.extend(exec_values)

        output_rows.append(
            {
                "task_name": task_name,
                "success_rate": _fmt_num(success_rate, 6),
                "success_rate_percent": _fmt_num(
                    success_rate * 100.0 if success_rate is not None else None, 2
                ),
                "exec_chunk_mean": _fmt_num(exec_chunk_mean, 6),
                "exec_chunk_std": _fmt_num(exec_chunk_std, 6),
                "exec_chunk_n": _fmt_int(exec_chunk_n),
                "eval_type": eval_type if eval_type else "N/A",
                "n_action_steps": _fmt_int(n_action_steps),
                "n_episodes": _fmt_int(n_episodes),
                "run_dir": str(row.get("run_dir", "")).strip() or "N/A",
                "info_json": str(row.get("info_json", "")).strip() or "N/A",
                "replan_log_path": str(row.get("replan_log_path", "")).strip() or "N/A",
                "log_path": str(row.get("log_path", "")).strip() or "N/A",
                "video_dir": str(row.get("video_dir", "")).strip() or "N/A",
            }
        )

    avg_success_rate = float(np.mean(success_values)) if success_values else None
    avg_exec_mean = float(np.mean(pooled_exec_values)) if pooled_exec_values else None
    avg_exec_std = float(np.std(pooled_exec_values)) if pooled_exec_values else None

    output_rows.append(
        {
            "task_name": "Average",
            "success_rate": _fmt_num(avg_success_rate, 6),
            "success_rate_percent": _fmt_num(
                avg_success_rate * 100.0 if avg_success_rate is not None else None, 2
            ),
            "exec_chunk_mean": _fmt_num(avg_exec_mean, 6),
            "exec_chunk_std": _fmt_num(avg_exec_std, 6),
            "exec_chunk_n": _fmt_int(len(pooled_exec_values)),
            "eval_type": "mixed",
            "n_action_steps": "N/A",
            "n_episodes": "N/A",
            "run_dir": "N/A",
            "info_json": "N/A",
            "replan_log_path": "N/A",
            "log_path": "N/A",
            "video_dir": "N/A",
        }
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_name",
        "success_rate",
        "success_rate_percent",
        "exec_chunk_mean",
        "exec_chunk_std",
        "exec_chunk_n",
        "eval_type",
        "n_action_steps",
        "n_episodes",
        "run_dir",
        "info_json",
        "replan_log_path",
        "log_path",
        "video_dir",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    return task_count, len(pooled_exec_values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build task-level success/chunk metrics CSV from summary.csv and run artifacts."
    )
    parser.add_argument("--summary-csv", required=True, help="Path to summary.csv")
    parser.add_argument("--output-csv", required=True, help="Path to output task_metrics.csv")
    args = parser.parse_args()

    summary_csv = Path(args.summary_csv)
    output_csv = Path(args.output_csv)
    if not summary_csv.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_csv}")

    task_count, pooled_count = build_task_metrics(summary_csv, output_csv)
    print(f"wrote: {output_csv}")
    print(f"tasks: {task_count}")
    print(f"pooled_exec_samples: {pooled_count}")


if __name__ == "__main__":
    main()
