import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from build_task_metrics import build_task_metrics


def _write_run(root: Path, task_name: str, run_name: str, info: dict) -> Path:
    run_dir = root / task_name / run_name
    (run_dir / "info").mkdir(parents=True, exist_ok=True)
    (run_dir / "info" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (run_dir / "episodes").mkdir(parents=True, exist_ok=True)
    return run_dir


def test_build_task_metrics_copies_timing_fields_and_averages_latest_runs(tmp_path: Path):
    summary_csv = tmp_path / "summary.csv"
    output_csv = tmp_path / "task_metrics.csv"

    run_dir_old = _write_run(
        tmp_path / "runs",
        "task_alpha",
        "2026-04-26_00-00-00",
        {
            "timestamp": "2026-04-26_00-00-00",
            "task_config": "demo_clean",
            "model_name": "alpha_old",
            "success_rate": 0.1,
            "mean_step_policy_infer_seconds_weighted": 9.9,
        },
    )
    run_dir_a = _write_run(
        tmp_path / "runs",
        "task_alpha",
        "2026-04-27_00-00-00",
        {
            "timestamp": "2026-04-27_00-00-00",
            "task_config": "demo_clean",
            "model_name": "alpha_new",
            "success_rate": 0.8,
            "expected_rollouts": 100,
            "n_action_steps": 50,
            "mean_step_policy_infer_seconds_weighted": 0.4,
            "mean_step_policy_infer_seconds_unweighted": 0.5,
            "mean_infer_policy_infer_seconds_weighted": 0.8,
            "mean_infer_policy_infer_seconds_unweighted": 0.9,
            "mean_chunk50_policy_infer_seconds_weighted": 0.016,
            "mean_chunk50_policy_infer_seconds_unweighted": 0.018,
        },
    )
    run_dir_b = _write_run(
        tmp_path / "runs",
        "task_beta",
        "2026-04-27_01-00-00",
        {
            "timestamp": "2026-04-27_01-00-00",
            "task_config": "demo_randomized",
            "model_name": "beta",
            "success_rate": 0.6,
            "expected_rollouts": 100,
            "n_action_steps": 50,
            "mean_step_policy_infer_seconds_weighted": 0.2,
            "mean_step_policy_infer_seconds_unweighted": 0.3,
            "mean_infer_policy_infer_seconds_weighted": 0.4,
            "mean_infer_policy_infer_seconds_unweighted": 0.6,
            "mean_chunk50_policy_infer_seconds_weighted": 0.008,
            "mean_chunk50_policy_infer_seconds_unweighted": 0.012,
            "mean_infer_horizon_metric_compute_seconds_weighted": 0.05,
        },
    )

    summary_csv.write_text(
        "\n".join(
            [
                "task_name,task_config,model_name,run_dir,info_json,success_rate_percent,eval_type,replan_log_path,log_path,video_dir",
                f"task_alpha,demo_clean,alpha_old,{run_dir_old},{run_dir_old / 'info' / 'info.json'},10,horizon,N/A,N/A,N/A",
                f"task_alpha,demo_clean,alpha_new,{run_dir_a},{run_dir_a / 'info' / 'info.json'},80,horizon,N/A,N/A,N/A",
                f"task_beta,demo_randomized,beta,{run_dir_b},{run_dir_b / 'info' / 'info.json'},60,horizon,N/A,N/A,N/A",
            ]
        ),
        encoding="utf-8",
    )

    task_count, pooled_count = build_task_metrics(summary_csv, output_csv)

    assert task_count == 2
    assert pooled_count == 200

    with output_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3
    row_alpha = rows[0]
    row_beta = rows[1]
    row_avg = rows[2]

    assert row_alpha["task_name"] == "task_alpha"
    assert row_alpha["model_name"] == "alpha_new"
    assert row_alpha["mean_infer_policy_infer_seconds_weighted"] == "0.800000"
    assert row_alpha["mean_chunk50_policy_infer_seconds_unweighted"] == "0.018000"
    assert row_beta["task_name"] == "task_beta"
    assert row_beta["mean_infer_horizon_metric_compute_seconds_weighted"] == "0.050000"
    assert row_beta["mean_chunk50_horizon_metric_compute_seconds_weighted"] == "N/A"
    assert row_avg["task_name"] == "Average"
    assert row_avg["mean_step_policy_infer_seconds_weighted"] == "0.300000"
    assert row_avg["mean_infer_policy_infer_seconds_unweighted"] == "0.750000"
    assert row_avg["mean_chunk50_policy_infer_seconds_weighted"] == "0.012000"
    assert row_avg["mean_infer_horizon_metric_compute_seconds_weighted"] == "0.050000"
