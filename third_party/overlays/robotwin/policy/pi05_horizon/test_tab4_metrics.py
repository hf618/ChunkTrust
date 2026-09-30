import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from build_tab4_metrics import build_tab4


def _write_run(
    root: Path,
    task: str,
    setting: str,
    base_model: str,
    train_config: str,
    eval_tag: str,
    timestamp: str,
    info: dict,
) -> Path:
    run_dir = root / f"{task}_{setting}" / base_model / train_config / eval_tag / timestamp
    (run_dir / "info").mkdir(parents=True, exist_ok=True)
    (run_dir / "episodes").mkdir(parents=True, exist_ok=True)
    base_info = {
        "timestamp": timestamp.replace("_", " "),
        "task_name": task,
        "task_config": f"demo_{setting}",
        "base_model_type": base_model,
        "train_config_name": train_config,
        "eval_tag": eval_tag,
        "expected_rollouts": 2,
    }
    base_info.update(info)
    (run_dir / "info" / "info.json").write_text(json.dumps(base_info), encoding="utf-8")
    return run_dir


def _write_episode(run_dir: Path, idx: int, *, success: bool, replans: int, policy_total: float, metric_total: float, selection_total: float, total_time: float) -> None:
    np.savez(
        run_dir / "episodes" / f"episode{idx}.npz",
        success=np.asarray(success, dtype=np.bool_),
        K_exec=np.ones((replans,), dtype=np.int32) * 10,
        policy_infer_seconds_total=np.asarray(policy_total, dtype=np.float32),
        horizon_metric_compute_seconds_total=np.asarray(metric_total, dtype=np.float32),
        horizon_selection_seconds_total=np.asarray(selection_total, dtype=np.float32),
        run_time_seconds=np.asarray([total_time, 0.0], dtype=np.float32),
        episode_steps=np.asarray(replans * 10, dtype=np.int32),
    )


def _make_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        fixed_root=tmp_path / "fixed",
        horizon_root=tmp_path / "horizon",
        output_dir=tmp_path / "out",
        fixed_train_config="pi0_base_aloha_robotwin_lora",
        horizon_train_config="pi05_base_aloha_robotwin_full",
        fixed_eval_tag="panels_fixedk_npzv3",
        s5_tag="tab4_s5_qmix",
        s10_tag="tab4_s10_qmix",
        s50_tag="tab4_s50_qmix",
        tasks="task_a,task_b",
        settings="clean",
        fixed_values="10,20",
    )


def test_build_tab4_metrics_selects_fixed_best_and_success_only_metrics(tmp_path: Path):
    args = _make_args(tmp_path)

    for fixed_k, sr, policy_total in [(10, 0.25, 0.4), (20, 0.75, 0.8)]:
        for task in ["task_a", "task_b"]:
            run_dir = _write_run(
                args.fixed_root,
                task,
                "clean",
                "pi0",
                args.fixed_train_config,
                args.fixed_eval_tag,
                f"2026-04-27_00-00-{fixed_k}",
                {
                    "eval_type": "default",
                    "fixed_exec_k": fixed_k,
                    "success_rate": sr,
                },
            )
            _write_episode(run_dir, 0, success=True, replans=2, policy_total=policy_total, metric_total=0.0, selection_total=0.0, total_time=1.0)
            _write_episode(run_dir, 1, success=False, replans=3, policy_total=policy_total, metric_total=0.0, selection_total=0.0, total_time=2.0)

    horizon_specs = [
        ("tab4_s5_qmix", "10,20,30,40,50", [10, 20, 30, 40, 50], 0.8),
        ("tab4_s10_qmix", "5,10,15,20,25,30,35,40,45,50", list(range(5, 51, 5)), 0.6),
        ("tab4_s50_qmix", ",".join(str(x) for x in range(1, 51)), list(range(1, 51)), 0.4),
    ]
    for tag, raw_candidates, candidates, sr in horizon_specs:
        for task in ["task_a", "task_b"]:
            run_dir = _write_run(
                args.horizon_root,
                task,
                "clean",
                "pi05",
                args.horizon_train_config,
                tag,
                "2026-04-27_01-00-00",
                {
                    "eval_type": "horizon",
                    "horizon_candidates": raw_candidates,
                    "success_rate": sr,
                },
            )
            assert candidates
            _write_episode(run_dir, 0, success=True, replans=4, policy_total=0.8, metric_total=0.08, selection_total=0.04, total_time=3.0)
            _write_episode(run_dir, 1, success=False, replans=5, policy_total=1.0, metric_total=0.1, selection_total=0.05, total_time=4.0)

    main_csv, per_task_csv, sources_json = build_tab4(args)

    with main_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["candidate_set"] for row in rows] == [
        "Fixed-best / no q_mix",
        "S5 = {10,20,30,40,50}",
        "S10 = {5,10,15,...,50}",
        "S50 = {1,2,3,...,50}",
    ]

    fixed = rows[0]
    s5 = rows[1]
    assert fixed["success_rate_percent"] == "75.000000"
    assert fixed["delta_success_rate_vs_fixed_best"] == "0.000000"
    assert fixed["avg_overhead_over_base_infer_per_replan_percent"] == "0.000000"
    assert fixed["avg_replans_per_success_episode"] == "2.000000"
    assert fixed["avg_total_time_per_success_episode_ms"] == "1000.000000"
    assert s5["success_rate_percent"] == "80.000000"
    assert s5["delta_success_rate_vs_fixed_best"] == "5.000000"
    assert s5["avg_overhead_over_base_infer_per_replan_percent"] == "15.000000"
    assert s5["avg_replans_per_success_episode"] == "4.000000"
    assert s5["avg_total_time_per_success_episode_ms"] == "3000.000000"

    with per_task_csv.open("r", encoding="utf-8", newline="") as f:
        per_task_rows = list(csv.DictReader(f))
    assert len(per_task_rows) == 8

    sources = json.loads(sources_json.read_text(encoding="utf-8"))
    assert sources["fixed_best_k"] == 20
    assert sources["fixed_candidates"]["20"]["missing"] == []
