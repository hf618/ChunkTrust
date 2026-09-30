from pathlib import Path

import numpy as np

from denoise_logger import DenoiseLogger
from pi_model import PI0
from runtime_hooks import collect_step_timing_stats


def _make_stub_model(tmp_path: Path, *, dump_denoise: bool) -> tuple[PI0, Path]:
    model = PI0.__new__(PI0)
    model.dump_denoise = bool(dump_denoise)
    # __new__ intentionally bypasses PI0.__init__, so establish the logging
    # invariants used by finalize_episode_logging explicitly in this fixture.
    model.episode_logging_enabled = bool(dump_denoise)
    model._episode_context = {}
    model._episode_path = str(tmp_path / ("episode_dump.npz" if dump_denoise else "episode_light.npz"))
    model._denoise_overhead_seconds = 0.25
    model._policy_infer_seconds = 1.5
    model._horizon_metric_compute_seconds = 0.5
    model._horizon_selection_seconds = 0.75
    model.pi0_step = 50
    model.denoise_logger = DenoiseLogger(enabled=bool(dump_denoise), compress=False)
    if dump_denoise:
        model.denoise_logger.start_episode(model._episode_path, meta={"pi0_step": 50})
    return model, Path(model._episode_path)


def test_finalize_episode_logging_writes_timing_fields_for_both_paths(tmp_path: Path):
    light_model, light_path = _make_stub_model(tmp_path, dump_denoise=False)
    dump_model, dump_path = _make_stub_model(tmp_path, dump_denoise=True)
    dump_model.denoise_logger.log_step(
        np.zeros((2, 2), dtype=np.float32),
        x0=np.ones((2, 2), dtype=np.float32),
        xt=np.ones((2, 2), dtype=np.float32),
        xt_internal=np.ones((2, 2), dtype=np.float32),
        k_exec=10,
        policy_infer_seconds=0.7,
        horizon_metric_compute_seconds=0.2,
        horizon_selection_seconds=0.3,
    )
    dump_model.denoise_logger.log_step(
        np.zeros((2, 2), dtype=np.float32),
        x0=np.ones((2, 2), dtype=np.float32),
        xt=np.ones((2, 2), dtype=np.float32),
        xt_internal=np.ones((2, 2), dtype=np.float32),
        k_exec=20,
        policy_infer_seconds=0.8,
        horizon_metric_compute_seconds=0.3,
        horizon_selection_seconds=0.45,
    )

    light_model.finalize_episode_logging(success=True, success_step=3, episode_steps=10, run_time_seconds=4.0)
    dump_model.finalize_episode_logging(success=True, success_step=3, episode_steps=10, run_time_seconds=4.0)

    for path, expected_overhead in ((light_path, 0.0), (dump_path, 0.25)):
        with np.load(path, allow_pickle=False) as data:
            assert "run_time_seconds" in data.files
            assert "policy_infer_seconds_total" in data.files
            assert "horizon_metric_compute_seconds_total" in data.files
            assert "horizon_selection_seconds_total" in data.files
            assert "action_chunk_size" in data.files
            assert np.allclose(np.asarray(data["run_time_seconds"]), np.asarray([4.0, expected_overhead], dtype=np.float32))
            assert np.isclose(float(np.asarray(data["policy_infer_seconds_total"]).item()), 1.5)
            assert np.isclose(float(np.asarray(data["horizon_metric_compute_seconds_total"]).item()), 0.5)
            assert np.isclose(float(np.asarray(data["horizon_selection_seconds_total"]).item()), 0.75)
            assert int(np.asarray(data["action_chunk_size"]).item()) == 50
            if path == dump_path:
                assert np.asarray(data["K_exec"]).shape == (2,)
                assert np.asarray(data["policy_infer_seconds"]).shape == (2,)
                assert np.asarray(data["horizon_metric_compute_seconds"]).shape == (2,)
                assert np.asarray(data["horizon_selection_seconds"]).shape == (2,)
                assert np.isclose(np.asarray(data["policy_infer_seconds"]).sum(), 1.5)
                assert np.isclose(np.asarray(data["horizon_metric_compute_seconds"]).sum(), 0.5)
                assert np.isclose(np.asarray(data["horizon_selection_seconds"]).sum(), 0.75)


def test_collect_step_timing_stats_reads_new_fields_and_tolerates_old_npz(tmp_path: Path):
    episodes_dir = tmp_path / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        episodes_dir / "episode0.npz",
        run_time_seconds=np.asarray([6.0, 1.0], dtype=np.float32),
        episode_steps=np.asarray(3, dtype=np.int32),
        policy_infer_seconds_total=np.asarray(1.5, dtype=np.float32),
        horizon_metric_compute_seconds_total=np.asarray(0.6, dtype=np.float32),
        horizon_selection_seconds_total=np.asarray(0.9, dtype=np.float32),
        action_chunk_size=np.asarray(50, dtype=np.int32),
        K_exec=np.asarray([10, 20], dtype=np.int32),
        policy_infer_seconds=np.asarray([0.7, 0.8], dtype=np.float32),
        horizon_metric_compute_seconds=np.asarray([0.2, 0.4], dtype=np.float32),
        horizon_selection_seconds=np.asarray([0.3, 0.6], dtype=np.float32),
    )
    np.savez(
        episodes_dir / "episode1.npz",
        run_time_seconds=np.asarray([4.0, 0.5], dtype=np.float32),
        episode_steps=np.asarray(2, dtype=np.int32),
        policy_infer_seconds_total=np.asarray(0.5, dtype=np.float32),
        K_exec=np.asarray([20], dtype=np.int32),
    )

    stats = collect_step_timing_stats(tmp_path)

    assert stats["episodes_with_timing"] == 2
    assert stats["episodes_with_step_count"] == 2
    assert stats["total_episode_steps"] == 5
    assert np.isclose(stats["mean_step_time_seconds_weighted"], 2.0)
    assert np.isclose(stats["mean_step_denoise_overhead_seconds_weighted"], 0.3)
    assert np.isclose(stats["mean_step_policy_infer_seconds_weighted"], 0.4)
    assert np.isclose(stats["mean_step_horizon_metric_compute_seconds_weighted"], 0.12)
    assert np.isclose(stats["mean_step_horizon_selection_seconds_weighted"], 0.18)
    assert np.isclose(stats["mean_step_policy_infer_seconds_unweighted"], 0.375)
    assert np.isclose(stats["mean_step_horizon_metric_compute_seconds_unweighted"], 0.2)
    assert np.isclose(stats["mean_step_horizon_selection_seconds_unweighted"], 0.3)
    assert np.isclose(stats["mean_infer_policy_infer_seconds_weighted"], 2.0 / 3.0)
    assert np.isclose(stats["mean_infer_policy_infer_seconds_unweighted"], 0.625)
    assert np.isclose(stats["mean_infer_horizon_metric_compute_seconds_weighted"], 0.3)
    assert np.isclose(stats["mean_infer_horizon_metric_compute_seconds_unweighted"], 0.3)
    assert np.isclose(stats["mean_infer_horizon_selection_seconds_weighted"], 0.45)
    assert np.isclose(stats["mean_infer_horizon_selection_seconds_unweighted"], 0.45)
    assert np.isclose(stats["mean_chunk50_policy_infer_seconds_weighted"], (2.0 / 3.0) / 50.0)
    assert np.isclose(stats["mean_chunk50_policy_infer_seconds_unweighted"], 0.625 / 50.0)
    assert np.isclose(stats["mean_chunk50_horizon_metric_compute_seconds_weighted"], 0.3 / 50.0)
    assert np.isclose(stats["mean_chunk50_horizon_metric_compute_seconds_unweighted"], 0.3 / 50.0)
    assert np.isclose(stats["mean_chunk50_horizon_selection_seconds_weighted"], 0.45 / 50.0)
    assert np.isclose(stats["mean_chunk50_horizon_selection_seconds_unweighted"], 0.45 / 50.0)
