from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional


RESULT_DIR_VERSION = "v2"

_SETTING_ALIASES = {
    "demo_clean": "clean",
    "demo_randomized": "randomized",
}

_TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"
_TIMESTAMP_INPUT_FORMATS = (
    _TIMESTAMP_FORMAT,
    "%Y-%m-%d %H:%M:%S",
)


def _clean_component(value: Any, default: Optional[str] = None) -> Optional[str]:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def normalize_eval_setting(task_config: Any) -> Optional[str]:
    setting = _clean_component(task_config)
    if setting is None:
        return None
    if setting in _SETTING_ALIASES:
        return _SETTING_ALIASES[setting]
    if setting.startswith("demo_"):
        return setting[len("demo_"):]
    return setting


def build_run_name(task_name: Any, task_config: Any) -> str:
    task = _clean_component(task_name, "unknown_task")
    normalized_setting = normalize_eval_setting(task_config)
    if normalized_setting:
        return f"{task}_{normalized_setting}"
    return task


def format_result_timestamp(timestamp: Optional[datetime] = None) -> str:
    return (timestamp or datetime.now()).strftime(_TIMESTAMP_FORMAT)


def parse_result_timestamp(timestamp_value: Any) -> datetime:
    if isinstance(timestamp_value, datetime):
        return timestamp_value

    timestamp_text = _clean_component(timestamp_value)
    if timestamp_text is None:
        raise ValueError("timestamp_value is required")

    for timestamp_format in _TIMESTAMP_INPUT_FORMATS:
        try:
            return datetime.strptime(timestamp_text, timestamp_format)
        except ValueError:
            continue

    raise ValueError(f"Unsupported timestamp format: {timestamp_text}")


def normalize_result_timestamp(timestamp_value: Any) -> str:
    return format_result_timestamp(parse_result_timestamp(timestamp_value))


def resolve_base_model_type(usr_args: Optional[Mapping[str, Any]] = None) -> str:
    usr_args = usr_args or {}
    return _clean_component(
        usr_args.get("base_model_type") or usr_args.get("policy_name"),
        "unknown_policy",
    )


def _basename_from_path(value: Any) -> str:
    text = _clean_component(value, "unknown_ckpt")
    basename = Path(text).name
    return basename or text


def resolve_ckpt_type(usr_args: Optional[Mapping[str, Any]] = None) -> str:
    usr_args = usr_args or {}
    return _clean_component(
        usr_args.get("ckpt_type")
        or usr_args.get("train_config_name")
        or usr_args.get("model_name")
        or _basename_from_path(usr_args.get("ckpt_setting")),
        "unknown_ckpt",
    )


@dataclass(frozen=True)
class EvalResultLayout:
    run_name: str
    base_model_type: str
    eval_tag: str
    ckpt_type: str
    timestamp: str
    save_dir: Path

    def metadata(self) -> dict[str, str]:
        return {
            "run_name": self.run_name,
            "base_model_type": self.base_model_type,
            "eval_tag": self.eval_tag,
            "ckpt_type": self.ckpt_type,
            "result_dir": str(self.save_dir),
            "result_dir_version": RESULT_DIR_VERSION,
        }


def build_eval_result_layout(
    task_name: Any,
    task_config: Any,
    usr_args: Optional[Mapping[str, Any]] = None,
    *,
    eval_root: Any = "eval_result",
    timestamp: Optional[Any] = None,
) -> EvalResultLayout:
    usr_args = usr_args or {}
    run_name = build_run_name(task_name, task_config)
    base_model_type = resolve_base_model_type(usr_args)
    eval_tag = _clean_component(usr_args.get("eval_tag"), "official")
    ckpt_type = resolve_ckpt_type(usr_args)
    normalized_timestamp = (
        format_result_timestamp()
        if timestamp is None
        else normalize_result_timestamp(timestamp)
    )

    save_dir = (
        Path(eval_root)
        / run_name
        / base_model_type
        / eval_tag
        / ckpt_type
        / normalized_timestamp
    )

    return EvalResultLayout(
        run_name=run_name,
        base_model_type=base_model_type,
        eval_tag=eval_tag,
        ckpt_type=ckpt_type,
        timestamp=normalized_timestamp,
        save_dir=save_dir,
    )


def build_eval_result_layout_from_args(
    usr_args: Mapping[str, Any],
    *,
    eval_root: Any = "eval_result",
    timestamp: Optional[Any] = None,
) -> EvalResultLayout:
    return build_eval_result_layout(
        usr_args.get("task_name"),
        usr_args.get("task_config"),
        usr_args,
        eval_root=eval_root,
        timestamp=timestamp,
    )


def is_legacy_result_layout(path_parts: tuple[str, ...]) -> bool:
    if len(path_parts) < 5:
        return False
    return path_parts[0] == "eval_result"
