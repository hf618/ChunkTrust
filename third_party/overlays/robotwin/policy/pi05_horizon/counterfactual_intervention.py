"""Fail-closed one-shot execution-horizon interventions for paired rollouts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class InterventionEntry:
    episode_seed: int
    replan_idx: int
    natural_k: int
    forced_k: int
    branch_role: str
    scout_trace_path: str
    scout_trace_sha256: str

    @property
    def role_code(self) -> int:
        return {"shrink": -1, "keep": 0, "extend": 1}[self.branch_role]


@dataclass(frozen=True)
class CounterfactualSpec:
    path: str
    sha256: str
    protocol_id: str
    source_manifest_sha256: str
    entries: dict[int, InterventionEntry]

    @classmethod
    def load(cls, raw_path: str | Path) -> "CounterfactualSpec":
        path = Path(raw_path).expanduser().resolve()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load counterfactual spec {path}: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("counterfactual spec must be a schema_version=1 object")
        protocol_id = str(payload.get("protocol_id", "")).strip()
        manifest_sha = str(payload.get("source_manifest_sha256", "")).strip()
        if not protocol_id or len(manifest_sha) != 64:
            raise ValueError("counterfactual spec requires protocol_id and source_manifest_sha256")
        rows = payload.get("entries")
        if not isinstance(rows, list) or not rows:
            raise ValueError("counterfactual spec entries must be a non-empty list")

        entries: dict[int, InterventionEntry] = {}
        required = {
            "episode_seed",
            "replan_idx",
            "natural_k",
            "forced_k",
            "branch_role",
            "scout_trace_path",
            "scout_trace_sha256",
        }
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"counterfactual entry[{index}] must be an object")
            missing = sorted(required - set(row))
            if missing:
                raise ValueError(f"counterfactual entry[{index}] missing fields: {missing}")
            seed = row["episode_seed"]
            replan_idx = row["replan_idx"]
            natural_k = row["natural_k"]
            forced_k = row["forced_k"]
            for name, value in (
                ("episode_seed", seed),
                ("replan_idx", replan_idx),
                ("natural_k", natural_k),
                ("forced_k", forced_k),
            ):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"counterfactual entry[{index}].{name} must be an integer")
            role = str(row["branch_role"]).strip().lower()
            if role not in {"shrink", "keep", "extend"}:
                raise ValueError(f"counterfactual entry[{index}] has invalid branch_role={role!r}")
            if replan_idx < 0 or natural_k < 1 or forced_k < 1:
                raise ValueError(f"counterfactual entry[{index}] has an out-of-range integer")
            if role == "shrink" and forced_k >= natural_k:
                raise ValueError("shrink intervention must have forced_k < natural_k")
            if role == "keep" and forced_k != natural_k:
                raise ValueError("keep intervention must have forced_k == natural_k")
            if role == "extend" and forced_k <= natural_k:
                raise ValueError("extend intervention must have forced_k > natural_k")
            if seed in entries:
                raise ValueError(f"counterfactual spec has duplicate episode_seed={seed}")

            scout_path = Path(str(row["scout_trace_path"])).expanduser().resolve()
            scout_sha = str(row["scout_trace_sha256"]).strip()
            if not scout_path.is_file():
                raise ValueError(f"counterfactual scout trace does not exist: {scout_path}")
            if sha256_file(scout_path) != scout_sha:
                raise ValueError(f"counterfactual scout trace hash mismatch: {scout_path}")
            entries[seed] = InterventionEntry(
                episode_seed=seed,
                replan_idx=replan_idx,
                natural_k=natural_k,
                forced_k=forced_k,
                branch_role=role,
                scout_trace_path=str(scout_path),
                scout_trace_sha256=scout_sha,
            )
        return cls(
            path=str(path),
            sha256=sha256_file(path),
            protocol_id=protocol_id,
            source_manifest_sha256=manifest_sha,
            entries=entries,
        )


class CounterfactualIntervention:
    """Episode-scoped controller that applies exactly one pre-registered K."""

    def __init__(self, spec: CounterfactualSpec):
        self.spec = spec
        self.entry: InterventionEntry | None = None
        self.fired = False
        self._scout_action_plan: np.ndarray | None = None
        self._scout_selected_k: np.ndarray | None = None

    def reset(self) -> None:
        self.entry = None
        self.fired = False
        self._scout_action_plan = None
        self._scout_selected_k = None

    def begin_episode(self, *, episode_seed: int, manifest_sha256: str) -> InterventionEntry:
        self.reset()
        if manifest_sha256 != self.spec.source_manifest_sha256:
            raise RuntimeError(
                "counterfactual spec/episode manifest hash mismatch: "
                f"{self.spec.source_manifest_sha256} != {manifest_sha256}"
            )
        try:
            self.entry = self.spec.entries[int(episode_seed)]
        except KeyError as exc:
            raise RuntimeError(f"counterfactual spec has no entry for episode_seed={episode_seed}") from exc
        with np.load(self.entry.scout_trace_path, allow_pickle=False) as data:
            required = {"action_plan", "selected_K", "episode_seed"}
            missing = sorted(required - set(data.files))
            if missing:
                raise RuntimeError(f"counterfactual scout trace missing replay fields: {missing}")
            trace_seed = int(np.asarray(data["episode_seed"]).reshape(()).item())
            if trace_seed != int(episode_seed):
                raise RuntimeError(
                    f"counterfactual scout episode seed mismatch: {trace_seed} != {episode_seed}"
                )
            self._scout_action_plan = np.asarray(data["action_plan"], dtype=np.float32)
            self._scout_selected_k = np.asarray(data["selected_K"], dtype=np.int32).reshape(-1)
        if self._scout_action_plan.ndim != 3:
            raise RuntimeError("counterfactual scout action_plan must be [N,H,D]")
        if self._scout_selected_k.shape != (self._scout_action_plan.shape[0],):
            raise RuntimeError("counterfactual scout selected_K/action_plan lengths differ")
        if self.entry.replan_idx >= self._scout_action_plan.shape[0]:
            raise RuntimeError("counterfactual target replan is outside scout trace")
        recorded_natural = int(self._scout_selected_k[self.entry.replan_idx])
        if recorded_natural != self.entry.natural_k:
            raise RuntimeError(
                f"counterfactual spec/scout natural K mismatch: {self.entry.natural_k} != {recorded_natural}"
            )
        return self.entry

    def apply(
        self,
        *,
        replan_idx: int,
        natural_k: int,
        max_exec_length: int,
        actions: np.ndarray,
    ) -> tuple[int, np.ndarray, dict[str, Any]]:
        if self.entry is None or self._scout_action_plan is None or self._scout_selected_k is None:
            raise RuntimeError("counterfactual episode was not initialized")
        entry = self.entry
        if not self.fired and int(replan_idx) > entry.replan_idx:
            raise RuntimeError(
                f"counterfactual target replan {entry.replan_idx} was skipped before replan {replan_idx}"
            )
        active = int(replan_idx) == entry.replan_idx
        prefix_replay = int(replan_idx) <= entry.replan_idx
        executed_k = int(natural_k)
        executed_actions = np.asarray(actions)
        expected_natural_k = int(natural_k)
        if prefix_replay:
            if int(replan_idx) >= self._scout_action_plan.shape[0]:
                raise RuntimeError(f"counterfactual scout has no action plan for replan {replan_idx}")
            replay = np.asarray(self._scout_action_plan[int(replan_idx)])
            if replay.shape != executed_actions.shape:
                raise RuntimeError(
                    f"counterfactual replay action shape mismatch: {replay.shape} != {executed_actions.shape}"
                )
            executed_actions = replay.astype(executed_actions.dtype, copy=True)
            expected_natural_k = int(self._scout_selected_k[int(replan_idx)])
            executed_k = expected_natural_k
        if active:
            if self.fired:
                raise RuntimeError(f"counterfactual intervention fired twice at replan {replan_idx}")
            if entry.forced_k > int(max_exec_length):
                raise RuntimeError(
                    f"counterfactual forced K {entry.forced_k} exceeds action chunk {max_exec_length}"
                )
            executed_k = entry.forced_k
            self.fired = True
        return executed_k, executed_actions, {
            "counterfactual_enabled": True,
            "counterfactual_fired": bool(active),
            "counterfactual_prefix_replay": bool(prefix_replay),
            "counterfactual_target_replan_idx": entry.replan_idx,
            "counterfactual_natural_k": expected_natural_k,
            "counterfactual_live_natural_k": int(natural_k),
            "counterfactual_executed_k": executed_k,
            "counterfactual_role_code": entry.role_code,
        }

    def assert_complete(self) -> None:
        if self.entry is not None and not self.fired:
            raise RuntimeError(
                f"counterfactual intervention did not reach target replan {self.entry.replan_idx}"
            )
