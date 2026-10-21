"""Test the real-server delta without importing JAX, loading weights, or opening sockets.

AST loading executes the shipped functions against small dependency fakes. Shared
transform implementations are loaded from the released horizon layer, not copied
into these tests. Full GPU/checkpoint execution remains an integration check.
"""
import abc
import ast
import dataclasses
import enum
import hashlib
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "third_party/overlays/agilex_openpi"
SHARED = ROOT / "third_party/overlays/robotwin/policy/pi05_horizon"


def load_nodes(path, names=None, **dependencies):
    """Execute selected release nodes with postponed external type annotations."""
    tree = ast.parse(path.read_text())
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.If)):
            continue
        if names is None or getattr(node, "name", None) in names:
            nodes.append(node)
    namespace = {"__name__": __name__, "dataclasses": dataclasses, "np": np, **dependencies}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.fixture
def transforms():
    names = {
        "Group", "RepackTransform", "SplitHistoryActions", "DeltaActions", "AbsoluteActions",
        "Normalize", "Unnormalize", "make_bool_mask", "_shift_action_prefix",
    }
    return load_nodes(SHARED / "src/openpi/transforms.py", names, DataTransformFn=object)


@pytest.fixture
def alignment(transforms):
    return load_nodes(
        OVERLAY / "src/openpi/training/agilex_real_alignment.py",
        os=os, transforms=SimpleNamespace(DataTransformFn=object),
    )


def test_raw_arm_alignment_and_batch_shapes(alignment):
    states = np.arange(2 * 3 * 26).reshape(2, 3, 26)
    actions = np.arange(2 * 3 * 14).reshape(2, 3, 14)
    actual_state, actual_action = alignment.remap_agilex_real_batch(states, actions)
    np.testing.assert_array_equal(actual_state, np.concatenate([states[..., 13:20], states[..., :7]], axis=-1))
    np.testing.assert_array_equal(actual_action, np.concatenate([actions[..., 7:14], actions[..., :7]], axis=-1))
    assert actual_state.dtype == actual_action.dtype == np.float32
    result = alignment.AgilexRealToAlohaInputs()({"state": states[0, 0]})
    assert result["state"].shape == (14,)
    assert "actions" not in result


@pytest.mark.parametrize("name,values", [
    ("state", np.zeros(25)), ("action", np.zeros(13)),
    ("state", np.array(1)), ("action", np.array(1)),
])
def test_alignment_rejects_wrong_raw_layout(alignment, name, values):
    with pytest.raises(ValueError, match=name):
        getattr(alignment, "remap_agilex_real_" + name)(values)


def test_dataset_path_is_portable_and_overridable(monkeypatch, transforms):
    monkeypatch.setenv("CHUNKTRUST_AGILEX_DATASET", "/example/local/dataset")
    module = load_nodes(
        OVERLAY / "src/openpi/training/agilex_real_alignment.py",
        os=os, transforms=SimpleNamespace(DataTransformFn=object),
    )
    assert module.AGILEX_REAL_REPO_ID == "/example/local/dataset"
    assert module.AGILEX_REAL_ASSET_ID == "agilex_magic_grab_bread_real"


@pytest.fixture
def real_config(transforms, alignment, monkeypatch):
    # Use the real ALOHA encoder/decoder; adapt_to_pi=False needs no gripper conversion library.
    # The image conversion dependency is a CHW -> HWC transpose for this one code path.
    aloha = load_nodes(
        SHARED / "src/openpi/policies/aloha_policy.py",
        {"AlohaInputs", "AlohaOutputs", "_decode_aloha", "_decode_state", "_encode_actions", "_encode_actions_inv"},
        transforms=SimpleNamespace(DataTransformFn=object),
        einops=SimpleNamespace(rearrange=lambda img, pattern: np.transpose(img, (1, 2, 0))),
    )
    model_type = enum.Enum("ModelType", ["PI0", "PI05"])
    module = load_nodes(
        OVERLAY / "src/openpi/training/config.py",
        {"AssetsConfig", "DataConfig", "DataConfigFactory", "AgilexRealAlohaDataConfig"},
        abc=abc, _transforms=transforms, agilex_real_alignment=alignment, aloha_policy=aloha,
        ModelType=model_type, tyro=SimpleNamespace(MISSING=object()),
        override=lambda fn: fn, pathlib=SimpleNamespace(Path=Path), epath=SimpleNamespace(Path=Path),
        ModelTransformFactory=lambda **kw: lambda model: transforms.Group(),
    )
    monkeypatch.setattr(module.DataConfigFactory, "_load_norm_stats", lambda *args: None)
    return module, model_type


def test_pi05_coordinate_and_normalization_contract(real_config, transforms):
    module, model_type = real_config
    factory = module.AgilexRealAlohaDataConfig(repo_id="local", assets=module.AssetsConfig(asset_id="saved"))
    config = factory.create(Path("assets"), SimpleNamespace(model_type=model_type.PI05, use_qha=False))
    assert config.asset_id == "saved"
    assert config.use_quantile_norm is True
    assert factory.adapt_to_pi is False
    raw_state = np.arange(26, dtype=np.float32)
    raw_actions = np.arange(28, dtype=np.float32).reshape(2, 14)
    data = {"state": raw_state.copy(), "actions": raw_actions.copy(), "images": {
        name: np.zeros((3, 4, 5), dtype=np.uint8)
        for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")
    }}
    for transform in config.data_transforms.inputs:
        data = transform(data)
    expected_state = np.concatenate([raw_state[13:20], raw_state[:7]])
    expected_actions = np.concatenate([raw_actions[:, 7:14], raw_actions[:, :7]], axis=-1)
    np.testing.assert_array_equal(data["state"], expected_state)
    # The two grippers remain absolute; the twelve joints are deltas.
    expected_delta = expected_actions.copy()
    joint_mask = np.array(transforms.make_bool_mask(6, -1, 6, -1))
    expected_delta[:, joint_mask] -= expected_state[joint_mask]
    np.testing.assert_array_equal(data["actions"], expected_delta)
    stats = SimpleNamespace(q01=np.full(14, -10.0), q99=np.full(14, 30.0))
    normalized = transforms.Normalize._normalize_quantile(None, data["actions"], stats)
    restored = transforms.Unnormalize._unnormalize_quantile(None, normalized, stats)
    data["actions"] = np.pad(restored, ((0, 0), (0, 18)))  # pi05 padded model dimension is 32.
    for transform in config.data_transforms.outputs:
        data = transform(data)
    np.testing.assert_allclose(data["actions"], expected_actions, atol=1e-5)
    assert data["actions"].shape == (2, 14)
    assert data["image"]["base_0_rgb"].shape == (4, 5, 3)
    overridden = dataclasses.replace(factory, use_quantile_norm_override=False)
    assert not overridden.create(Path("assets"), SimpleNamespace(model_type=model_type.PI05)).use_quantile_norm


def test_qha_history_transform_is_preserved(real_config):
    module, model_type = real_config
    config = module.AgilexRealAlohaDataConfig(repo_id="local").create(
        Path("assets"), SimpleNamespace(model_type=model_type.PI05, use_qha=True,
                                        qha_history_max_length=3, action_horizon=50),
    )
    assert [type(t).__name__ for t in config.data_transforms.inputs] == [
        "AgilexRealToAlohaInputs", "SplitHistoryActions", "AlohaInputs", "DeltaActions",
    ]
    assert config.data_transforms.inputs[1].history_len == 3


@pytest.mark.parametrize("use_default", [False, True])
@pytest.mark.parametrize("trace", [False, True])
def test_serve_cli_forwards_trace_switch(use_default, trace):
    calls = []
    module = load_nodes(
        OVERLAY / "scripts/serve_policy.py", enum=enum,
        _config=SimpleNamespace(get_config=lambda name: name),
        _policy_config=SimpleNamespace(create_trained_policy=lambda *args, **kw: calls.append((args, kw))),
    )
    policy = module.Default() if use_default else module.Checkpoint("real", "checkpoint")
    module.create_policy(module.Args(policy=policy, return_denoise_trace=trace, default_prompt="move"))
    assert len(calls) == 1
    assert calls[0][1]["sample_kwargs"] == ({"return_denoise_trace": True} if trace else {})
    assert calls[0][1]["default_prompt"] == "move"


def test_singleton_norm_stats_fallback_and_exact_preference(tmp_path, caplog):
    @dataclasses.dataclass(frozen=True)
    class Config:
        asset_id: str | None

    def load_stats(folder, asset_id):
        return json.loads((folder / asset_id / "norm_stats.json").read_text())

    module = load_nodes(
        OVERLAY / "src/openpi/policies/policy_config.py", {"_load_checkpoint_norm_stats"},
        pathlib=SimpleNamespace(Path=Path), logging=logging,
        _checkpoints=SimpleNamespace(load_norm_stats=load_stats),
    )
    folder = tmp_path / "assets" / "historical_name"
    folder.mkdir(parents=True)
    (folder / "norm_stats.json").write_text('{"actions": "saved"}')
    config, stats = module._load_checkpoint_norm_stats(tmp_path, Config("configured_name"))
    assert config.asset_id == "historical_name" and stats == {"actions": "saved"}
    assert "using checkpoint asset_id=historical_name" in caplog.text
    other = tmp_path / "assets" / "configured_name"
    other.mkdir()
    (other / "norm_stats.json").write_text('{"actions": "exact"}')
    config, stats = module._load_checkpoint_norm_stats(tmp_path, Config("configured_name"))
    assert config.asset_id == "configured_name" and stats == {"actions": "exact"}
    with pytest.raises(FileNotFoundError):
        module._load_checkpoint_norm_stats(tmp_path, Config("missing"))
    with pytest.raises(ValueError, match="Asset id"):
        module._load_checkpoint_norm_stats(tmp_path, Config(None))


def tree_map(fn, value):
    if isinstance(value, dict):
        return {key: tree_map(fn, item) for key, item in value.items()}
    return fn(value)


def test_infer_trace_serialization_preserves_model_coordinates():
    def compose(transforms):
        def run(value):
            for transform in transforms:
                value = transform(value)
            return value
        return run

    jax = SimpleNamespace(
        tree=SimpleNamespace(map=tree_map),
        random=SimpleNamespace(key=lambda seed: seed, split=lambda key: (key + 1, key + 2)),
    )
    module = load_nodes(
        OVERLAY / "src/openpi/policies/policy.py",
        {"_unbatch_singleton", "_as_msgpack_safe_trace_array", "Policy"},
        BasePolicy=object, jax=jax, jnp=np, time=SimpleNamespace(monotonic=lambda: 1.0),
        inspect=__import__("inspect"), override=lambda fn: fn,
        _transforms=SimpleNamespace(compose=compose, SplitHistoryActions=type("Split", (), {})),
        _model=SimpleNamespace(Observation=SimpleNamespace(from_dict=lambda value: value)),
        nnx_utils=SimpleNamespace(module_jit=lambda fn, **kw: fn),
    )
    trace = {"v": np.full((1, 3, 5, 32), 0.25, dtype=np.float16),
             "ds": np.full((1, 3), -1 / 3, dtype=np.float64),
             "x0": np.zeros((1, 5, 32)), "xt_internal": np.ones((1, 5, 32))}
    model = SimpleNamespace(
        sample_actions=lambda *args, **kw: pytest.fail("trace sampler was not used"),
        sample_actions_with_trace=lambda *args, **kw: (np.ones((1, 5, 32)), trace),
    )
    def output_transform(value):
        return {"actions": value["actions"][:, :14] + 100}

    policy = module.Policy(model, output_transforms=[output_transform],
                           sample_kwargs={"return_denoise_trace": True})
    result = policy.infer({"state": np.zeros(32)})
    np.testing.assert_array_equal(result["actions"], np.full((5, 14), 101.0))
    assert result["denoise_trace"]["v"].shape == (3, 5, 32)
    assert all(value.dtype == np.float32 for value in result["denoise_trace"].values())
    np.testing.assert_array_equal(result["denoise_trace"]["v"], np.full((3, 5, 32), 0.25))
    np.testing.assert_array_equal(result["denoise_trace"]["xt_internal"], np.ones((5, 32)))
    integers = np.arange(3, dtype=np.int32)
    np.testing.assert_array_equal(module._as_msgpack_safe_trace_array(integers), integers)
    assert module._as_msgpack_safe_trace_array(integers).dtype == np.int32


def test_bfloat16_trace_becomes_wire_compatible():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    module = load_nodes(OVERLAY / "src/openpi/policies/policy.py", {"_as_msgpack_safe_trace_array"})
    actual = module._as_msgpack_safe_trace_array(np.array([0.5, -2], dtype=ml_dtypes.bfloat16))
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, [0.5, -2])


def test_real_delta_preserves_released_qha_and_rng_implementations():
    def methods(path):
        tree = ast.parse(path.read_text())
        policy = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Policy")
        return {node.name: ast.dump(node, include_attributes=False) for node in policy.body
                if isinstance(node, ast.FunctionDef)}

    base = methods(SHARED / "src/openpi/policies/policy.py")
    actual = methods(OVERLAY / "src/openpi/policies/policy.py")
    assert actual.keys() == base.keys()
    assert all(actual[name] == node for name, node in base.items() if name != "infer")
    for path in OVERLAY.rglob("*.py"):
        compile(path.read_text(), str(path), "exec")


def test_export_manifest_covers_exact_released_bytes():
    manifest = json.loads((ROOT / "third_party/agilex_openpi_sources.json").read_text())
    expected = {str(path.relative_to(OVERLAY)) for path in OVERLAY.rglob("*.py")}
    assert set(manifest["files"]) == expected
    for name, record in manifest["files"].items():
        assert hashlib.sha256((OVERLAY / name).read_bytes()).hexdigest() == record["release_sha256"]
        assert len(record["source"]["sha256"]) == 64
        assert not Path(record["source"]["path"]).is_absolute()
