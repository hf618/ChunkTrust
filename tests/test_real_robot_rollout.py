"""Run both real-robot entrypoints through disconnected, deterministic sinks."""
import importlib
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "third_party/overlays/robodriver/scripts"


class DisconnectedRobot:
    """In-memory feedback/command sink; no SDK or hardware constructor is used."""

    def __init__(self, common):
        self.is_connected = True
        self.disconnect_count = 0
        self.commands = []
        self.observation = {}
        for keys, offset in [(common.LEFT_KEYS, 11), (common.RIGHT_KEYS, 1)]:
            for index, key in enumerate(keys):
                self.observation[f"follower_{key}.pos"] = (index + offset) * 0.01
        for side in ("left", "right"):
            for axis in ("x", "y", "z", "rx", "ry", "rz"):
                self.observation[f"follower_pose_{side}_{axis}.pos"] = 0.0
        for index, name in enumerate(("image_top", "image_left", "image_right")):
            self.observation[name] = np.full((4, 5, 3), index + 1, np.uint8)

    def get_observation(self):
        return dict(self.observation)

    def send_action(self, action):
        assert self.is_connected and len(action) == 14
        assert np.isfinite(list(action.values())).all()
        self.commands.append(dict(action))
        for name, value in action.items():
            self.observation[name.replace("leader_", "follower_", 1)] = value

    def disconnect(self):
        self.disconnect_count += 1
        self.is_connected = False


def setup_disconnected_runtime(monkeypatch, module_name, *, execute, trace=True):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    common = importlib.import_module("openpi_bridge_common")
    module = importlib.import_module(module_name)
    robot = DisconnectedRobot(common)
    requests = []

    class FakePolicy:
        def __init__(self, **kwargs):
            assert kwargs["host"] == "disconnected-test"

        def infer(self, observation):
            assert observation["state"].shape == (26,)
            assert observation["prompt"] == "offline core rollout"
            assert {image.shape for image in observation["images"].values()} == {(3, 4, 5)}
            requests.append(observation["state"].copy())
            left, right = common.raw26_to_left_right(observation["state"])
            desired = np.r_[left + 0.005, right + 0.006].astype(np.float32)
            result = {"actions": np.tile(desired, (6, 1))}
            if trace:
                velocity = np.arange(5 * 6 * 32, dtype=np.float32).reshape(5, 6, 32)
                result["denoise_trace"] = {"v": np.sin(velocity * 0.01)}
            return result

    package = ModuleType("openpi_client")
    client_module = ModuleType("openpi_client.websocket_client_policy")
    client_module.WebsocketClientPolicy = FakePolicy
    monkeypatch.setitem(sys.modules, "openpi_client", package)
    monkeypatch.setitem(sys.modules, "openpi_client.websocket_client_policy", client_module)
    monkeypatch.setattr(module, "make_robot", lambda: robot)
    argv = [str(SCRIPTS / f"{module_name}.py"), "--host", "disconnected-test",
            "--prompt", "offline core rollout", "--hz", "0",
            "--episode-max-action-steps", "12", "--chunk-prefix-steps", "4",
            "--allow-dual-arm", "--allow-gripper", "--motion-watchdog", "0",
            "--execute", str(execute)]
    if module_name.endswith("_ahs"):
        argv.extend(["--ahs-candidates", "3,6", "--ahs-metadata-policy", "require"])
    monkeypatch.setattr(sys, "argv", argv)
    return module, robot, requests


@pytest.mark.parametrize("module_name", ["openpi_bridge_rollout_safe", "openpi_bridge_rollout_ahs"])
@pytest.mark.parametrize("execute", [0, 1])
def test_main_replans_commands_and_disconnects(monkeypatch, module_name, execute):
    module, robot, requests = setup_disconnected_runtime(monkeypatch, module_name, execute=execute)
    module.main()
    assert len(requests) >= 2
    assert robot.disconnect_count == 1 and not robot.is_connected
    assert len(robot.commands) == (12 if execute else 0)
    if execute:
        assert set(robot.commands[0]) == {
            f"leader_{name}_{side}.pos"
            for side in ("left", "right") for name in [*[f"joint{i}" for i in range(1, 7)], "gripper"]
        }
        assert robot.commands[0]["leader_joint1_left.pos"] == pytest.approx(0.115)
        assert robot.commands[0]["leader_joint1_right.pos"] == pytest.approx(0.016)
        assert not np.array_equal(requests[0], requests[1])


def test_required_trace_failure_sends_nothing_and_disconnects(monkeypatch):
    module, robot, requests = setup_disconnected_runtime(
        monkeypatch, "openpi_bridge_rollout_ahs", execute=1, trace=False,
    )
    with pytest.raises(RuntimeError, match="AHS metadata required"):
        module.main()
    assert len(requests) == 1 and robot.commands == []
    assert robot.disconnect_count == 1 and not robot.is_connected
