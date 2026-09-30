#!/usr/bin/env python3
"""Serve the fixed step-30000 pi0.5 RoboCasa package with denoise traces."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import logging
from pathlib import Path
import sys
from typing import Any

import flax.nnx as nnx
import jax
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from pi05_robocasa_common import (  # noqa: E402
    AHS_CONFIG,
    CHECKPOINT,
    MODEL_ID,
    MODEL_REVISION,
    NORM_ASSET_ID,
    NORM_STATS_SHA256,
    OPTIMIZER_STEP,
    PACKAGE_ROOT,
    PACKAGE_TREE_SHA256,
    PROTOCOL_ID,
    SAMPLING_CONFIG,
    file_sha256,
    load_package_manifest,
)

RUNTIME_ROOT = PACKAGE_ROOT / "openpi_runtime"
sys.path.insert(0, str(RUNTIME_ROOT / "src"))
sys.path.insert(0, str(RUNTIME_ROOT / "packages/openpi-client/src"))

from openpi.policies import policy_config  # noqa: E402
from openpi.serving import websocket_policy_server  # noqa: E402
from openpi.training import config as _config  # noqa: E402


class NoiseAwarePolicy:
    """Pass explicit paired-noise requests without changing the packaged policy."""

    def __init__(self, policy) -> None:
        self._policy = policy

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        request = dict(request)
        noise = request.pop("policy_noise", None)
        result = self._policy.infer(request, noise=noise)
        trace = result.get("denoise_trace")
        if not isinstance(trace, dict) or "v" not in trace:
            raise RuntimeError("Fixed-trace policy response is missing denoise_trace.v")
        result["denoise_trace"] = {"v": np.asarray(trace["v"], dtype=np.float32)}
        return result

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.metadata


def bind_static_trace_steps(policy, num_steps: int) -> None:
    """Replace the delivered dynamic trace JIT with an equivalent fixed-step JIT."""
    model = policy._model
    graphdef, state = nnx.split(model)

    def fixed_trace_fn(state, rng, observation, noise):
        restored_model = nnx.merge(graphdef, state)
        return restored_model.sample_actions_with_trace(
            rng,
            observation,
            num_steps=num_steps,
            noise=noise,
        )

    jitted = jax.jit(fixed_trace_fn)

    @functools.wraps(model.sample_actions_with_trace)
    def fixed_trace(rng, observation, *, num_steps: int = num_steps, noise=None):
        if int(num_steps) != SAMPLING_CONFIG["denoise_steps"]:
            raise ValueError(f"Trace JIT is frozen to {SAMPLING_CONFIG['denoise_steps']} steps, got {num_steps}")
        return jitted(state, rng, observation, noise)

    policy._sample_actions_with_trace = fixed_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5920)
    parser.add_argument("--ready-file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_package_manifest()
    train_config = dataclasses.replace(
        _config.get_config("pi05_base_gr1_tabletop24_full"),
        assets_base_dir=str(PACKAGE_ROOT / ".no_external_training_assets"),
    )
    if train_config.data.assets.asset_id != NORM_ASSET_ID:
        raise RuntimeError(f"Unexpected configured norm asset: {train_config.data.assets.asset_id}")
    norm_stats_path = CHECKPOINT / "assets" / NORM_ASSET_ID / "norm_stats.json"
    if file_sha256(norm_stats_path) != NORM_STATS_SHA256:
        raise RuntimeError(f"Norm stats hash mismatch: {norm_stats_path}")
    policy = policy_config.create_trained_policy(
        train_config,
        CHECKPOINT,
        robotwin_repo_id=manifest["tasks"][0]["repo_id"],
        sample_kwargs={
            "num_steps": SAMPLING_CONFIG["denoise_steps"],
            "return_denoise_trace": True,
        },
    )
    bind_static_trace_steps(policy, SAMPLING_CONFIG["denoise_steps"])
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "optimizer_step": OPTIMIZER_STEP,
        "norm_asset_id": NORM_ASSET_ID,
        "norm_stats_sha256": NORM_STATS_SHA256,
        "checkpoint": str(CHECKPOINT.resolve()),
        "package_tree_sha256": PACKAGE_TREE_SHA256,
        "package_manifest_sha256": file_sha256(PACKAGE_ROOT / "PACKAGE_MANIFEST.sha256"),
        "sampling_config": SAMPLING_CONFIG,
        "ahs_config": AHS_CONFIG,
        "use_qha": False,
        "trace_num_steps_static_bound": SAMPLING_CONFIG["denoise_steps"],
        "trace_wire_fields": ["v"],
        "trace_wire_dtype": "float32",
    }
    if args.ready_file is not None:
        args.ready_file.parent.mkdir(parents=True, exist_ok=True)
        args.ready_file.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logging.info("PI05_ROBOCASA_SERVER_READY %s", json.dumps(metadata, sort_keys=True))
    websocket_policy_server.WebsocketPolicyServer(
        NoiseAwarePolicy(policy),
        host=args.host,
        port=args.port,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
