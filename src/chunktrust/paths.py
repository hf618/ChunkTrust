"""Resolve machine-local research paths using explicit deployment roots."""
import os
from pathlib import Path


def resolve_legacy_path(variable: str, relative: str = "") -> str:
    defaults = {
        "ROBOTWIN_ROOT": Path.cwd(),
        "CHUNKTRUST_DATA_ROOT": Path.cwd() / "data",
        "CHUNKTRUST_RESULTS_ROOT": Path.cwd() / "outputs",
        "CHUNKTRUST_MODELS_ROOT": Path.cwd() / "checkpoints",
        "CHUNKTRUST_CACHE_ROOT": Path.home() / ".cache" / "chunktrust",
        "CHUNKTRUST_WORKSPACE_ROOT": Path.cwd().parent,
    }
    return str(Path(os.environ.get(variable, str(defaults[variable]))) / relative)
