from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


def infer_policy_name(
    config_path: str | None = None,
    config: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> str | None:
    candidates = []
    if overrides:
        candidates.append(overrides.get("policy_name"))
    if config:
        candidates.append(config.get("policy_name"))
    if config_path:
        cfg_parent = Path(config_path).parent
        if cfg_parent.name:
            candidates.append(cfg_parent.name)

    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text and text.lower() != "null":
            return text
    return None


def load_policy_hooks(policy_name: str | None) -> ModuleType | None:
    if not policy_name:
        return None

    hooks_path = Path("policy") / str(policy_name) / "runtime_hooks.py"
    if not hooks_path.is_file():
        return None

    module_name = f"_policy_runtime_hooks_{policy_name}"
    spec = importlib.util.spec_from_file_location(module_name, hooks_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load runtime hooks from {hooks_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call_optional_hook(hooks: ModuleType | None, hook_name: str, *args, **kwargs):
    if hooks is None:
        return None
    hook = getattr(hooks, hook_name, None)
    if hook is None:
        return None
    return hook(*args, **kwargs)
