import hashlib
import json

import numpy as np
import pytest

from counterfactual_intervention import CounterfactualIntervention, CounterfactualSpec


def _write_spec(tmp_path, *, role="shrink", natural_k=30, forced_k=20):
    scout = tmp_path / "episode0.npz"
    np.savez(
        scout,
        action_plan=np.zeros((3, 50, 14), dtype=np.float32),
        selected_K=np.asarray([25, 28, natural_k], dtype=np.int32),
        episode_seed=np.asarray(123),
    )
    manifest_sha = "a" * 64
    payload = {
        "schema_version": 1,
        "protocol_id": "unit",
        "source_manifest_sha256": manifest_sha,
        "entries": [
            {
                "episode_seed": 123,
                "replan_idx": 2,
                "natural_k": natural_k,
                "forced_k": forced_k,
                "branch_role": role,
                "scout_trace_path": str(scout),
                "scout_trace_sha256": hashlib.sha256(scout.read_bytes()).hexdigest(),
            }
        ],
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, manifest_sha


def test_intervention_fires_exactly_once(tmp_path):
    path, manifest_sha = _write_spec(tmp_path)
    runtime = CounterfactualIntervention(CounterfactualSpec.load(path))
    runtime.begin_episode(episode_seed=123, manifest_sha256=manifest_sha)

    actions = np.ones((50, 14), dtype=np.float32)
    assert runtime.apply(replan_idx=0, natural_k=24, max_exec_length=50, actions=actions)[0] == 25
    assert runtime.apply(replan_idx=1, natural_k=27, max_exec_length=50, actions=actions)[0] == 28
    forced, replayed, info = runtime.apply(
        replan_idx=2, natural_k=29, max_exec_length=50, actions=actions
    )
    assert forced == 20
    assert np.all(replayed == 0)
    assert info["counterfactual_fired"] is True
    assert info["counterfactual_natural_k"] == 30
    assert info["counterfactual_live_natural_k"] == 29
    assert info["counterfactual_executed_k"] == 20
    assert runtime.apply(replan_idx=3, natural_k=22, max_exec_length=50, actions=actions)[0] == 22
    runtime.assert_complete()


def test_intervention_rejects_manifest_mismatch(tmp_path):
    path, manifest_sha = _write_spec(tmp_path)
    runtime = CounterfactualIntervention(CounterfactualSpec.load(path))
    with pytest.raises(RuntimeError, match="manifest hash mismatch"):
        runtime.begin_episode(episode_seed=123, manifest_sha256="b" * 64)

def test_intervention_rejects_skipped_or_unreached_target(tmp_path):
    path, manifest_sha = _write_spec(tmp_path)
    runtime = CounterfactualIntervention(CounterfactualSpec.load(path))
    runtime.begin_episode(episode_seed=123, manifest_sha256=manifest_sha)
    with pytest.raises(RuntimeError, match="did not reach"):
        runtime.assert_complete()
    with pytest.raises(RuntimeError, match="was skipped"):
        runtime.apply(
            replan_idx=3,
            natural_k=30,
            max_exec_length=50,
            actions=np.ones((50, 14), dtype=np.float32),
        )


@pytest.mark.parametrize(
    ("role", "natural_k", "forced_k"),
    [("shrink", 30, 30), ("keep", 30, 20), ("extend", 30, 20)],
)
def test_spec_validates_branch_direction(tmp_path, role, natural_k, forced_k):
    path, _ = _write_spec(
        tmp_path,
        role=role,
        natural_k=natural_k,
        forced_k=forced_k,
    )
    with pytest.raises(ValueError):
        CounterfactualSpec.load(path)
