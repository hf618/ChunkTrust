#!/usr/bin/env python3
from __future__ import annotations
from chunktrust.paths import resolve_legacy_path as _ct_path

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


DEFAULT_MD_PATHS = [
    _ct_path('CHUNKTRUST_WORKSPACE_ROOT', 'starVLA/examples/Robocasa_tabletop/eval_sum/eval_sum.md'),
    _ct_path('CHUNKTRUST_WORKSPACE_ROOT', 'Isaac-GR00T/examples/robocasa-gr1-tabletop-tasks/eval_sum/eval_sum.md'),
    _ct_path('CHUNKTRUST_WORKSPACE_ROOT', 'Isaac-GR00T-n1.5/examples/RoboCasa/eval_sum/gr00t-n1.5-robocasa-tabletop-posttrain/eval_sum.md'),
    _ct_path('CHUNKTRUST_WORKSPACE_ROOT', 'Isaac-GR00T-n1.5/examples/RoboCasa/eval_sum/GR00T-N1.5-3B/eval_sum.md'),
]


def normalize_cell(s: str) -> str:
    return s.replace("`", "").replace("*", "").strip()


def normalize_task_cell(s: str) -> str:
    return normalize_cell(s).lower()


_BASE_ALIAS_TO_TAG_KEY: Dict[str, str] = {
    "default12": "official_eval_50x720x12",
    "default16": "ablation_50x720x16_default",
    "stride-both": "ablation_50x720x16_stride_expected_round_both",
    "stride-both2": "ablation_50x720x16_expected_round_both2",
    "stride-both-kf": "ablation_50x720x16_expected_round_both_kf",
    "stride-both-kf-sym": "ablation_50x720x16_expected_round_both_kf_sym",
    "stride-can1": "ablation_50x720x16_stride_expected_round_both_can1",
    "stride-can2": "ablation_50x720x16_stride_expected_round_both_can2",
    "stride-inter": "ablation_50x720x16_stride_expected_round_inter",
    "stride-intra": "ablation_50x720x16_stride_expected_round_intra",
    "greedy-inter": "ablation_50x720x16_stride_greedy_inter",
    "greedy-intra": "ablation_50x720x16_stride_greedy_intra",
}

_CFG_FIXED_CHUNK_BY_KEY: Dict[str, float] = {
    "official_eval_50x720x12": 12.0,
    "ablation_50x720x16_default": 16.0,
}


def _norm_compact_cell(s: str) -> str:
    return re.sub(r"\s+", " ", normalize_cell(s)).strip().lower()


def canonical_eval_key_from_column(col: str) -> Optional[str]:
    c = _norm_compact_cell(col)
    if not c:
        return None

    # Strip legacy wrappers used in qwen/local summaries.
    if c.startswith("local "):
        c = c[len("local ") :].strip()
    for suffix in (" chunk (cfg)", " avg exec chunk", " run dir", " run path", " sr"):
        if c.endswith(suffix):
            c = c[: -len(suffix)].strip()
            break

    if c in _BASE_ALIAS_TO_TAG_KEY:
        return _BASE_ALIAS_TO_TAG_KEY[c]

    # New canonical headers: "official eval ...", "ablation ...", and underscore variants.
    if c.startswith("official eval ") or c.startswith("ablation "):
        return c.replace(" ", "_")
    if c.startswith("official_eval_") or c.startswith("ablation_"):
        return re.sub(r"\s+", "", c)
    return None


def parse_md_table(lines: List[str], header_idx: int) -> Tuple[List[str], List[List[str]], int]:
    header = [c.strip() for c in lines[header_idx].strip().strip("|").split("|")]
    rows: List[List[str]] = []
    i = header_idx + 2
    while i < len(lines) and lines[i].startswith("|"):
        row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        if len(row) < len(header):
            row += [""] * (len(header) - len(row))
        elif len(row) > len(header):
            row = row[: len(header)]
        rows.append(row)
        i += 1
    return header, rows, i


def fmt_mean_std(mean_v: float, std_v: float) -> str:
    return f"{mean_v:.2f} ({std_v:.2f})"


def maybe_bold(value: str, template_cell: str) -> str:
    return f"**{value}**" if "**" in template_cell else value


def is_average_row(task_cell: str) -> bool:
    t = normalize_task_cell(task_cell)
    return t.startswith("average")


def chunk_key_from_column(col: str) -> Optional[str]:
    c = _norm_compact_cell(col)
    if c == "task":
        return None
    if c in {"official reported chunk", "official reported"}:
        return "official_reported"
    key = canonical_eval_key_from_column(col)
    if key is None:
        return None
    if key in _CFG_FIXED_CHUNK_BY_KEY:
        return f"cfg::{key}"
    # Any recognized non-fixed eval tag is treated as dynamic chunk source.
    if key.startswith("official_eval_") or key.startswith("ablation_"):
        return f"dyn::{key}"
    return None


def find_selected_col_for_dyn_key(sel_header: List[str], dyn_key: str) -> Optional[int]:
    for i, c in enumerate(sel_header):
        if i == 0:
            continue
        if canonical_eval_key_from_column(c) == dyn_key:
            return i
    # fallback: single run path column
    if len(sel_header) == 2:
        return 1
    return None


def collect_k_exec_values(run_dir: Path) -> List[float]:
    ep_dir = run_dir / "episodes"
    if not ep_dir.exists():
        return []
    vals: List[float] = []
    for npz_path in sorted(ep_dir.rglob("*.npz")):
        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception:
            continue
        try:
            if "K_exec" not in data.files:
                continue
            k = data["K_exec"]
            arr = np.asarray(k).reshape(-1)
            if arr.size == 0:
                continue
            vals.extend([float(x) for x in arr.tolist()])
        finally:
            data.close()
    return vals


def infer_cfg_chunk_from_col(col: str) -> Optional[float]:
    key = canonical_eval_key_from_column(col)
    if key is None:
        return None
    return _CFG_FIXED_CHUNK_BY_KEY.get(key)


def refill_one_md(md_path: Path) -> Dict[str, int]:
    lines = md_path.read_text().splitlines()
    task_header_idxs = [i for i, l in enumerate(lines) if l.startswith("| Task |")]
    if len(task_header_idxs) < 3:
        raise RuntimeError(f"{md_path}: expected >=3 Task tables, got {len(task_header_idxs)}")

    acc_idx, chunk_idx, sel_idx = task_header_idxs[:3]
    chunk_header, chunk_rows, chunk_end = parse_md_table(lines, chunk_idx)
    sel_header, sel_rows, _ = parse_md_table(lines, sel_idx)

    # map selected path rows by task
    sel_task_to_row: Dict[str, List[str]] = {}
    for r in sel_rows:
        if not r:
            continue
        sel_task_to_row[normalize_task_cell(r[0])] = r

    # per-column caches for dynamic stats
    dyn_task_stats: Dict[int, Dict[str, Tuple[float, float, int]]] = {}
    dyn_all_vals: Dict[int, List[float]] = {}

    for cidx, col in enumerate(chunk_header):
        if cidx == 0:
            continue
        key = chunk_key_from_column(col)
        if key is None:
            continue

        # Official column is textual conclusion only
        if key == "official_reported":
            for ridx, row in enumerate(chunk_rows):
                row[cidx] = maybe_bold("N/A", row[cidx]) if is_average_row(row[0]) else "N/A"
            continue

        if key.startswith("cfg::"):
            cfg_val = infer_cfg_chunk_from_col(col)
            if cfg_val is None:
                for row in chunk_rows:
                    row[cidx] = "N/A"
            else:
                base = fmt_mean_std(cfg_val, 0.0)
                for row in chunk_rows:
                    if is_average_row(row[0]):
                        row[cidx] = maybe_bold(base, row[cidx])
                    else:
                        row[cidx] = base
            continue

        # dynamic
        dyn_base = key.split("::", 1)[1]
        sel_cidx = find_selected_col_for_dyn_key(sel_header, dyn_base)
        if sel_cidx is None:
            for row in chunk_rows:
                row[cidx] = maybe_bold("N/A", row[cidx]) if is_average_row(row[0]) else "N/A"
            continue

        task_stats: Dict[str, Tuple[float, float, int]] = {}
        all_vals: List[float] = []
        for row in chunk_rows:
            task_norm = normalize_task_cell(row[0])
            if is_average_row(row[0]):
                continue
            sel_row = sel_task_to_row.get(task_norm)
            if not sel_row or sel_cidx >= len(sel_row):
                continue
            run_dir_raw = normalize_cell(sel_row[sel_cidx])
            if run_dir_raw.upper() == "N/A" or not run_dir_raw:
                continue
            run_dir = Path(run_dir_raw)
            if not run_dir.exists():
                continue
            vals = collect_k_exec_values(run_dir)
            if not vals:
                continue
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals))
            task_stats[task_norm] = (mean_v, std_v, len(vals))
            all_vals.extend(vals)

        dyn_task_stats[cidx] = task_stats
        dyn_all_vals[cidx] = all_vals

    # write filled values back to chunk rows
    for ridx, row in enumerate(chunk_rows):
        is_avg = is_average_row(row[0])
        task_norm = normalize_task_cell(row[0])
        for cidx in range(1, len(chunk_header)):
            key = chunk_key_from_column(chunk_header[cidx])
            if key is None or key.startswith("cfg::") or key == "official_reported":
                continue  # already filled

            if is_avg:
                vals = dyn_all_vals.get(cidx, [])
                if vals:
                    value = fmt_mean_std(float(np.mean(vals)), float(np.std(vals)))
                    row[cidx] = maybe_bold(value, row[cidx])
                else:
                    row[cidx] = maybe_bold("N/A", row[cidx])
            else:
                stat = dyn_task_stats.get(cidx, {}).get(task_norm)
                if stat is None:
                    row[cidx] = "N/A"
                else:
                    mean_v, std_v, _ = stat
                    row[cidx] = fmt_mean_std(mean_v, std_v)

    # rebuild chunk table block
    new_chunk_lines = []
    new_chunk_lines.append("| " + " | ".join(chunk_header) + " |")
    new_chunk_lines.append("| " + " | ".join(["---"] * len(chunk_header)) + " |")
    for r in chunk_rows:
        new_chunk_lines.append("| " + " | ".join(r) + " |")

    new_lines = lines[:chunk_idx] + new_chunk_lines + lines[chunk_end:]
    md_path.write_text("\n".join(new_lines))

    dyn_cols = sum(1 for c in chunk_header[1:] if chunk_key_from_column(c) and chunk_key_from_column(c).startswith("dyn::"))
    cfg_cols = sum(1 for c in chunk_header[1:] if chunk_key_from_column(c) and chunk_key_from_column(c).startswith("cfg::"))
    return {"dyn_cols": dyn_cols, "cfg_cols": cfg_cols, "rows": len(chunk_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute Chunk Table values from episodes/*.npz (K_exec) and refill markdown summaries.")
    parser.add_argument("--md-path", action="append", default=[], help="Path to eval_sum.md (can be passed multiple times).")
    args = parser.parse_args()

    md_paths = [Path(p) for p in (args.md_path or DEFAULT_MD_PATHS)]
    for p in md_paths:
        stat = refill_one_md(p)
        print(f"updated: {p}")
        print(f"  chunk rows={stat['rows']} cfg_cols={stat['cfg_cols']} dyn_cols={stat['dyn_cols']}")


if __name__ == "__main__":
    main()
