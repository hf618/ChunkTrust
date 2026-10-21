"""Offline robot adapter checks; no device, client service, or camera is opened."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "third_party/overlays/robodriver/scripts"
CLI_SCRIPTS = (
    "openpi_bridge_smoke_action.py",
    "openpi_bridge_reset_pose.py",
    "openpi_bridge_rollout_safe.py",
    "openpi_bridge_rollout_ahs.py",
    "openpi_bridge_rollout_ahs_async.py",
)


@pytest.fixture
def bridges(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    modules = {
        name: importlib.import_module(module)
        for name, module in {
            "smoke": "openpi_bridge_smoke_action",
            "reset": "openpi_bridge_reset_pose",
            "safe": "openpi_bridge_rollout_safe",
            "ahs": "openpi_bridge_rollout_ahs",
            "async_ahs": "openpi_bridge_rollout_ahs_async",
            "video": "eval_video_recorder",
            "npz": "eval_episode_npz_logger",
        }.items()
    }
    return SimpleNamespace(**modules)


@pytest.fixture
def rollout_args(bridges, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["openpi_bridge_rollout_ahs_async.py"])
    return bridges.async_ahs.parse_args()


def test_raw26_state_and_policy_camera_mapping(bridges):
    observation = {}
    for index, key in enumerate(bridges.smoke.RIGHT_KEYS):
        observation[f"follower_{key}.pos"] = index
    for index, key in enumerate(bridges.smoke.LEFT_KEYS):
        observation[f"follower_{key}.pos"] = index + 13
    for index, axis in enumerate(("x", "y", "z", "rx", "ry", "rz")):
        observation[f"follower_pose_right_{axis}.pos"] = index + 7
        observation[f"follower_pose_left_{axis}.pos"] = index + 20
    for index, camera in enumerate(("image_top", "image_left", "image_right")):
        observation[camera] = np.full((4, 5, 3), index + 1, dtype=np.uint8)

    raw = bridges.smoke.observation_to_raw26(observation)
    np.testing.assert_array_equal(raw, np.arange(26, dtype=np.float32))
    assert raw.dtype == np.float32
    left, right = bridges.smoke.raw26_to_left_right(raw)
    np.testing.assert_array_equal(left, np.arange(13, 20))
    np.testing.assert_array_equal(right, np.arange(7))
    command = bridges.smoke.action_dict_from_left_right(left, right)
    assert command["leader_joint1_left.pos"] == 13
    assert command["leader_joint1_right.pos"] == 0
    assert command["leader_gripper_left.pos"] == 19
    assert command["leader_gripper_right.pos"] == 6
    assert len(command) == 14
    policy_input = bridges.smoke.observation_to_policy_input(observation, raw)
    assert policy_input["state"] is raw
    for index, camera in enumerate(("cam_high", "cam_left_wrist", "cam_right_wrist")):
        np.testing.assert_array_equal(policy_input["images"][camera], np.full((3, 4, 5), index + 1))
    del observation["follower_joint1_left.pos"]
    with pytest.raises(KeyError):
        bridges.smoke.observation_to_raw26(observation)


def test_float_images_are_rgb_chw_uint8(bridges):
    image = np.zeros((4, 5, 3), dtype=np.float32)
    image[..., 0], image[..., 1], image[..., 2] = -0.5, 0.5, 1.5
    output = bridges.smoke.chw_uint8(image)
    assert output.dtype == np.uint8
    np.testing.assert_array_equal(output[:, 0, 0], [0, 127, 255])
    with pytest.raises(RuntimeError, match="3 color channels"):
        bridges.smoke.chw_uint8(np.zeros((4, 5, 4)))


@pytest.mark.parametrize("module_name", ["safe", "ahs"])
@pytest.mark.parametrize(
    "single_arm,dual,gripper,active_left,active_right",
    [(None, False, False, False, True),
     (None, True, True, True, True),
     ("left", False, True, True, False),
     ("right", True, False, False, True)],
)
def test_policy_arm_gates_and_command_clipping(
    bridges, rollout_args, module_name, single_arm, dual, gripper, active_left, active_right
):
    rollout_args.single_arm_only = single_arm
    rollout_args.allow_dual_arm = dual
    rollout_args.allow_gripper = gripper
    current_left = np.full(7, 0.2, dtype=np.float32)
    current_right = np.full(7, -0.2, dtype=np.float32)
    policy = np.concatenate([np.ones(7), -np.ones(7)]).astype(np.float32)
    left, right = getattr(bridges, module_name).target_from_policy_step(
        current_left, current_right, policy, None, None, rollout_args
    )
    expected_left = current_left.copy()
    expected_right = current_right.copy()
    if active_left:
        expected_left[:6] += 0.03
        expected_left[6] += 0.005 if gripper else 0.0
    if active_right:
        expected_right[:6] -= 0.03
        expected_right[6] -= 0.005 if gripper else 0.0
    np.testing.assert_allclose(left, expected_left, atol=1e-8)
    np.testing.assert_allclose(right, expected_right, atol=1e-8)
    np.testing.assert_array_equal(current_left, np.full(7, 0.2, dtype=np.float32))
    np.testing.assert_array_equal(current_right, np.full(7, -0.2, dtype=np.float32))


def test_policy_alpha_precedes_ema_and_clipping(bridges, rollout_args):
    rollout_args.policy_alpha = 0.2
    rollout_args.action_ema = 0.25
    rollout_args.allow_dual_arm = True
    rollout_args.allow_gripper = True
    rollout_args.max_joint_delta_rad = 1.0
    rollout_args.max_gripper_delta = 1.0
    left, right = bridges.ahs.target_from_policy_step(
        np.zeros(7), np.zeros(7), np.ones(14), np.full(7, -0.1), np.full(7, 0.1), rollout_args
    )
    np.testing.assert_allclose(left, 0.125)
    np.testing.assert_allclose(right, 0.175)


def test_smoke_action_preserves_gate_defaults(bridges):
    args = SimpleNamespace(policy_alpha=0.03, max_joint_delta_rad=0.005,
                           max_gripper_delta=0.002, single_arm_only="left", disable_gripper=True)
    current = np.zeros(7, dtype=np.float32)
    left, right = bridges.smoke.limit_target(current, current, np.ones(7), -np.ones(7), args)
    np.testing.assert_allclose(left, [0.005] * 6 + [0.0])
    np.testing.assert_array_equal(right, current)


def test_reset_defaults_and_step_do_not_ship_a_robot_pose(bridges):
    assert bridges.reset.DEFAULT_POSE_PATH.parent == SCRIPTS
    assert not bridges.reset.DEFAULT_POSE_PATH.exists()
    args = SimpleNamespace(max_joint_step_rad=0.03, max_gripper_step=0.005, include_gripper=False)
    current = np.zeros(7)
    target = bridges.reset.step_toward(current, np.ones(7), args)
    np.testing.assert_allclose(target, [0.03] * 6 + [0.0])
    args.include_gripper = True
    np.testing.assert_allclose(bridges.reset.step_toward(current, np.ones(7), args), [0.03] * 6 + [0.005])


def _replay_inputs():
    tau = np.arange(5, dtype=np.float32)[:, None, None]
    t = np.arange(50, dtype=np.float32)[None, :, None]
    dim = np.arange(14, dtype=np.float32)[None, None, :]
    velocity = np.sin((t + 1) * (dim + 1) * 0.037 + tau * 0.31) * (1 + tau * 0.07)
    actions = np.sin(np.arange(50, dtype=np.float32)[:, None] * 0.11
                     + np.arange(14, dtype=np.float32)[None, :] * 0.17)
    history = np.cos(np.arange(70, dtype=np.float32)[:, None] * 0.07
                     + np.arange(14, dtype=np.float32)[None, :] * 0.13)
    return velocity, actions, history


def test_original_workstation_selector_numerical_replay(bridges, rollout_args):
    # Golden values were captured from the untouched original source's numeric
    # AST, with NumPy 1.26.4. Provenance and original source SHA are in the manifest.
    # These intentionally retain the historical shrinking FFT and candidate reset.
    _, selector = bridges.async_ahs.make_ahs_selectors(rollout_args, [10, 20, 30, 40, 50])
    velocity, actions, history = _replay_inputs()
    selector.record_executed_actions(history)
    golden = [
        (50, [10, 20, 30, 40, 50], 42, 42.21190195611295,
         [0.019163737073540688, 0.014825169928371906, 0.01146780513226986, 0.008618081919848919, 0.0069108265452086926],
         [0.056739527732133865, 0.04690634086728096, 0.2146281898021698, 0.7697522044181824, 1.0],
         [[0.9920308077377309, 0.9905810931128293], [1.0202178743222319, 0.9986465096260714],
          [1.155411054329722, 1.0373305388151777], [1.3230969585020238, 1.0853120006851007],
          [1.2367642961119611, 1.060608866757151]]),
        (25, [10, 20], 16, 15.928297567393233,
         [0.019346008077263832, 0.014728572219610214],
         [0.5091578364372253, 0.5091578364372253],
         [[1.2318605691658757, 1.2231602433551843], [1.2572972672713496, 1.2476819202075904]]),
        (7, [7], 7, 7.0, [0.05013967305421829], [1.0], [[1.99, 0.99]]),
    ]
    for remaining, horizons, exec_length, expected, z_intra, q_mix, beta in golden:
        result = selector.select({"v": velocity}, actions, max_exec_length=remaining)
        np.testing.assert_array_equal(result["horizons"], horizons)
        assert result["exec_length"] == exec_length
        assert result["expected_horizon"] == pytest.approx(expected, rel=2e-6)
        np.testing.assert_allclose(result["z_intra"], z_intra, rtol=2e-6, atol=1e-8)
        np.testing.assert_allclose(result["q_mix"], q_mix, rtol=2e-6, atol=1e-8)
        assert tuple(selector._state) == tuple(horizons)
        np.testing.assert_allclose(list(selector._state.values()), beta, rtol=2e-6, atol=1e-8)


def test_required_ahs_metadata_and_explicit_fallback(bridges, rollout_args):
    ahs, horizon = bridges.async_ahs.make_ahs_selectors(rollout_args, [10, 20, 30, 40, 50])
    actions = np.zeros((50, 14), dtype=np.float32)

    def choose(result, remaining=25):
        return bridges.ahs.choose_ahs_exec_length(
            result, actions=actions, actions_shape=actions.shape, remaining_steps=remaining,
            selector=ahs, horizon_selector=horizon, args=rollout_args
        )

    rollout_args.ahs_metadata_policy = "require"
    with pytest.raises(RuntimeError, match="AHS metadata required but unavailable"):
        choose({})
    with pytest.raises(RuntimeError, match="test-time horizon metadata unusable"):
        choose({"denoise_trace": {"v": np.zeros((3, 4))}})
    rollout_args.ahs_metadata_policy = "fallback"
    fallback = choose({})
    assert fallback.exec_length == 5
    assert fallback.source == "fallback"
    assert fallback.inactive_reason
    server_choice = choose({"exec_length": 50}, remaining=7)
    assert server_choice.exec_length == 7
    assert server_choice.source == "server_exec_length"


def test_qha_metadata_selection_preserves_effective_score_math(bridges, rollout_args):
    rollout_args.ahs_exec_mode = "qmix_expected_round"
    ahs, horizon = bridges.async_ahs.make_ahs_selectors(rollout_args, [10, 20, 50])
    actions = np.zeros((50, 14), dtype=np.float32)
    choice = bridges.ahs.choose_ahs_exec_length(
        {"qha_candidate_horizons": [10, 20, 50], "qha_posterior": [0.2, 0.3, 0.5],
         "qha_selector_q_mix": [0.1, 0.3, 0.6]},
        actions=actions, actions_shape=actions.shape, remaining_steps=50,
        selector=ahs, horizon_selector=horizon, args=rollout_args
    )
    assert choice.source == "qha"
    assert choice.expected_horizon == pytest.approx(17.0 / 0.41, rel=1e-6)
    assert choice.exec_length == 41
    assert all(value == [1.0, 1.0] for value in ahs._state.values())


def test_recorder_is_optional_and_records_default_relative_to_cwd(bridges, rollout_args):
    assert rollout_args.record_root == Path("records/eval_videos")
    assert bridges.video.create_eval_video_recorder(rollout_args, object(), rollout_kind="ahs") is None


def test_npz_keeps_policy_and_sent_commands_distinct(bridges, tmp_path):
    recorder = SimpleNamespace(episode_dir=tmp_path, task_name="offline")
    logger = bridges.npz.EvalEpisodeNpzLogger(recorder=recorder, rollout_kind="ahs", args=argparse.Namespace())
    policy = np.ones(14, dtype=np.float32)
    sent = np.zeros(14, dtype=np.float32)
    logger.log_step(step_index=0, replan_index=1, chunk_index=0, step_start_time_ns=123,
                    step_start_monotonic_ns=456, state_raw26_before=np.arange(26),
                    policy_action_lr14=policy, executed_action_lr14=sent, execute_enabled=False)
    policy[:] = 9
    sent[:] = 8
    with np.load(logger.save(status="offline"), allow_pickle=False) as data:
        np.testing.assert_array_equal(data["policy_action_lr14"], np.ones((1, 14)))
        np.testing.assert_array_equal(data["executed_action_lr14"], np.zeros((1, 14)))
        assert data["execute_enabled"].tolist() == [False]
        assert data["top_frame_index"].tolist() == [-1]


_BLOCK_OPTIONAL_IMPORTS = """
import importlib.abc
import sys
class BlockRobotDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'robodriver', 'openpi_client', 'cv2'}:
            raise ModuleNotFoundError('offline dependency forbidden: ' + fullname)
sys.meta_path.insert(0, BlockRobotDependencies())
"""


@pytest.mark.parametrize("script", CLI_SCRIPTS)
def test_cli_help_needs_no_robot_policy_client_or_opencv(script, tmp_path):
    path = SCRIPTS / script
    program = _BLOCK_OPTIONAL_IMPORTS + "\nimport runpy\nsys.argv = [sys.argv[1], '--help']\nrunpy.run_path(sys.argv[0], run_name='__main__')\n"
    result = subprocess.run([sys.executable, "-c", program, str(path)], cwd=tmp_path,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert "--execute" in result.stdout


def test_all_exported_modules_import_with_optional_dependencies_forbidden(tmp_path):
    program = _BLOCK_OPTIONAL_IMPORTS + "\nimport importlib\nsys.path.insert(0, sys.argv[1])\n"
    program += "\nfor name in sys.argv[2:]:\n    importlib.import_module(name)\n"
    names = [path.stem for path in sorted(SCRIPTS.glob("*.py"))]
    result = subprocess.run([sys.executable, "-c", program, str(SCRIPTS), *names], cwd=tmp_path,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_export_manifest_is_bounded_and_release_hashes_match():
    metadata = json.loads((ROOT / "third_party/robodriver_sources.json").read_text())
    assert metadata["origin"]["repository"] == "https://github.com/FlagOpen/RoboDriver.git"
    assert len(metadata["origin"]["revision"]) == 40
    expected_names = set(CLI_SCRIPTS) | {"eval_episode_npz_logger.py", "eval_video_recorder.py"}
    assert {Path(row["source_path"]).name for row in metadata["files"]} == expected_names
    for row in metadata["files"]:
        assert row["source_bytes"] > 0
        assert len(row["source_sha256"]) == 64
        destination = ROOT / row["destination_path"]
        assert destination.parent == SCRIPTS
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == row["release_sha256"]
    assert not list(SCRIPTS.glob("*.json"))
