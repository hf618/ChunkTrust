#!/usr/bin/env python3
from __future__ import annotations
from chunktrust.paths import resolve_legacy_path as _ct_path

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from build_task_metrics import build_task_metrics


UNIFIED_ROOT = Path(_ct_path('CHUNKTRUST_RESULTS_ROOT', 'robocasa-gr1-tabletop-tasks/eval_results'))
SOURCE_ROOTS = (
    Path(_ct_path('CHUNKTRUST_WORKSPACE_ROOT', 'starVLA/examples/Robocasa_tabletop/eval_result')),
    Path(_ct_path('CHUNKTRUST_WORKSPACE_ROOT', 'Isaac-GR00T/examples/robocasa-gr1-tabletop-tasks/eval_result')),
    Path(_ct_path('CHUNKTRUST_WORKSPACE_ROOT', 'Isaac-GR00T-n1.5/examples/RoboCasa/eval_result')),
)
@dataclass(frozen=True)
class RunRecord:
    info_path: Path
    source_run_dir: Path
    target_run_dir: Path
    model_name: str
    ckpt_tag: str
    eval_tag: str

    @property
    def target_summary_dir(self) -> Path:
        return UNIFIED_ROOT / "_summary" / self.model_name / self.ckpt_tag / self.eval_tag


@dataclass(frozen=True)
class SummaryRecord:
    source_dir: Path
    target_dir: Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _collect_run_records() -> List[RunRecord]:
    records: List[RunRecord] = []
    for root in SOURCE_ROOTS:
        if not root.exists():
            continue
        for info_path in sorted(root.rglob("info.json")):
            if "_summary" in info_path.parts:
                continue
            info = _load_json(info_path)
            task_name = str(info.get("task_name", "")).strip()
            model_name = str(info.get("model_name", "")).strip()
            ckpt_tag = str(info.get("ckpt_tag", "")).strip()
            eval_tag = str(info.get("eval_tag", "")).strip()
            if not all((task_name, model_name, ckpt_tag, eval_tag)):
                continue

            source_run_dir = info_path.parent
            target_run_dir = UNIFIED_ROOT / task_name / model_name / ckpt_tag / eval_tag / source_run_dir.name
            records.append(
                RunRecord(
                    info_path=info_path,
                    source_run_dir=source_run_dir,
                    target_run_dir=target_run_dir,
                    model_name=model_name,
                    ckpt_tag=ckpt_tag,
                    eval_tag=eval_tag,
                )
            )
    return records


def _collect_summary_records() -> List[SummaryRecord]:
    records: List[SummaryRecord] = []
    for root in SOURCE_ROOTS:
        summary_root = root / "_summary"
        if not summary_root.exists():
            continue
        for summary_csv in sorted(summary_root.rglob("summary.csv")):
            rel_parts = summary_csv.relative_to(summary_root).parts
            if len(rel_parts) < 4:
                continue
            model_name, eval_tag, ckpt_tag = rel_parts[:3]
            source_dir = summary_csv.parent
            target_dir = UNIFIED_ROOT / "_summary" / model_name / ckpt_tag / eval_tag
            records.append(SummaryRecord(source_dir=source_dir, target_dir=target_dir))
    return records


def _ensure_no_duplicate_targets(paths: Iterable[Path], label: str) -> None:
    seen: Dict[Path, Path] = {}
    for path in paths:
        if path in seen:
            raise RuntimeError(f"Duplicate {label} target detected: {path}")
        seen[path] = path


def _sorted_prefix_map(path_map: Dict[Path, Path]) -> List[tuple[Path, Path]]:
    return sorted(path_map.items(), key=lambda item: len(str(item[0])), reverse=True)


def _remap_path(value: str, prefix_pairs: List[tuple[Path, Path]]) -> str:
    text = value.strip()
    if not text:
        return value

    path = Path(text)
    for source_prefix, target_prefix in prefix_pairs:
        try:
            rel = path.relative_to(source_prefix)
        except ValueError:
            continue
        return str(target_prefix / rel)

    legacy_value = _remap_legacy_eval_result_path(text)
    if legacy_value is not None:
        return legacy_value
    return value


def _move_directory(source_dir: Path, target_dir: Path) -> None:
    if not source_dir.exists():
        return
    if target_dir.exists():
        raise RuntimeError(f"Target already exists: {target_dir}")
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_dir), str(target_dir))


def _rewrite_info_json(record: RunRecord) -> None:
    info_path = record.target_run_dir / "info.json"
    info = _load_json(info_path)
    info["run_dir"] = str(record.target_run_dir)
    info["log_path"] = str(record.target_run_dir / "run.log")
    info["video_dir"] = str(record.target_run_dir / "videos")

    replan_value = str(info.get("replan_log_path", "")).strip()
    info["replan_log_path"] = str(record.target_run_dir / "replan_trace.jsonl") if replan_value else ""

    server_value = str(info.get("server_log", "")).strip()
    if server_value:
        info["server_log"] = str(record.target_summary_dir / Path(server_value).name)

    _dump_json(info_path, info)


def _rewrite_summary_csv(summary_csv: Path, prefix_pairs: List[tuple[Path, Path]], target_dir: Path) -> None:
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    for row in rows:
        for key in ("run_dir", "info_json", "log_path", "video_dir", "replan_log_path", "server_log"):
            if key not in row:
                continue
            value = row.get(key, "")
            if key == "server_log" and value.strip():
                row[key] = str(target_dir / Path(value).name)
            else:
                row[key] = _remap_path(value, prefix_pairs)

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rebuild_task_metrics(summary_dir: Path) -> None:
    summary_csv = summary_dir / "summary.csv"
    task_metrics_csv = summary_dir / "task_metrics.csv"
    if summary_csv.exists():
        build_task_metrics(summary_csv, task_metrics_csv)


def _remap_legacy_eval_result_path(value: str) -> str | None:
    path = Path(value)
    parts = path.parts
    if "eval_result" not in parts:
        return None

    idx = parts.index("eval_result")
    tail = parts[idx + 1 :]
    if not tail:
        return None

    if tail[0] == "_summary":
        if len(tail) < 5:
            return None
        model_name, eval_tag, ckpt_tag = tail[1:4]
        base = UNIFIED_ROOT / "_summary" / model_name / ckpt_tag / eval_tag
        rest = tail[4:]
        return str(base.joinpath(*rest))

    if len(tail) < 5:
        return None
    task_name, model_name, eval_tag, ckpt_tag, timestamp = tail[:5]
    base = UNIFIED_ROOT / task_name / model_name / ckpt_tag / eval_tag / timestamp
    rest = tail[5:]
    return str(base.joinpath(*rest))


def _repair_unified_summaries(prefix_pairs: List[tuple[Path, Path]]) -> int:
    summary_root = UNIFIED_ROOT / "_summary"
    repaired = 0
    if not summary_root.exists():
        return repaired

    for summary_csv in sorted(summary_root.rglob("summary.csv")):
        summary_dir = summary_csv.parent
        _rewrite_summary_csv(summary_csv, prefix_pairs, summary_dir)
        _rebuild_task_metrics(summary_dir)
        repaired += 1
    return repaired


def migrate() -> None:
    run_records = _collect_run_records()
    summary_records = _collect_summary_records()

    _ensure_no_duplicate_targets((record.target_run_dir for record in run_records), "run")
    _ensure_no_duplicate_targets((record.target_dir for record in summary_records), "summary")

    run_prefix_map = {record.source_run_dir: record.target_run_dir for record in run_records}
    summary_prefix_map = {record.source_dir: record.target_dir for record in summary_records}
    prefix_pairs = _sorted_prefix_map({**run_prefix_map, **summary_prefix_map})

    UNIFIED_ROOT.mkdir(parents=True, exist_ok=True)

    for record in run_records:
        _move_directory(record.source_run_dir, record.target_run_dir)
        _rewrite_info_json(record)

    for record in summary_records:
        _move_directory(record.source_dir, record.target_dir)
        _rewrite_summary_csv(record.target_dir / "summary.csv", prefix_pairs, record.target_dir)
        _rebuild_task_metrics(record.target_dir)

    repaired_summaries = _repair_unified_summaries(prefix_pairs)

    print(f"migrated runs: {len(run_records)}")
    print(f"migrated summaries: {len(summary_records)}")
    print(f"repaired summaries: {repaired_summaries}")
    print(f"unified root: {UNIFIED_ROOT}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate RoboCasa eval results into the unified directory layout.")
    parser.parse_args()
    migrate()


if __name__ == "__main__":
    main()
