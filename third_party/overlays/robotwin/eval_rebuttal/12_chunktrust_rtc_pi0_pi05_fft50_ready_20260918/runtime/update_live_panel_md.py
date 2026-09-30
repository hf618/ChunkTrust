"""Refresh the human-readable live formal-panel status."""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from common import EXP
from panel_config import load_active_protocol, primary_records
from report import records, summarize


def fmt(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def main():
    if (EXP / 'artifacts/pi05_four_queue.json').exists():
        from focused_dashboard import main as update_focus
        return update_focus()
    protocol = load_active_protocol()
    rows = primary_records(records("formal"), protocol)
    total = protocol["formal_total"]
    now = datetime.datetime.now().astimezone()
    status_path = EXP / 'artifacts/pipeline_status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    running = False
    if status.get('status') == 'running' and status.get('pid'):
        try:
            os.kill(status['pid'], 0)
            running = True
        except ProcessLookupError:
            pass
    state = 'running' if running else status.get('status', 'unknown')
    if state == 'running' and not running:
        state = 'stopped (recorded PID is absent)'
    lines = [
        "# Formal panel live progress",
        "",
        f"Last refreshed: {now.isoformat()}",
        f"Panel: `{protocol['panel_protocol_id']}`",
        f"Setting: `{protocol['setting']}` (Hard / demo_randomized only)",
        f"Completed: **{len(rows):,}/{total:,} ({100*len(rows)/total:.2f}%)**",
        f"Remaining: **{total-len(rows):,}**",
        f"Backbones: π0.5 {sum(r['backbone']=='pi05' for r in rows):,}/9,600; π0 {sum(r['backbone']=='pi0' for r in rows):,}/9,600",
        f"Queue: **{state}**; phase: `{status.get('phase', '—')}`; PID: {status.get('pid', '—')}",
        "",
        "Costs use the currently completed +100 ms episodes; a cost cell is final only at n=100.",
        "SR is final only at n=100; otherwise the cell shows successes/completed.",
        "",
        "| Task | Backbone | Method | SR +0 ms | SR +100 ms | SR +200 ms | Wait % | Calls/ep | Infer s/ep |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in protocol["tasks"]:
        for backbone in ("pi05", "pi0"):
            for method in ("sync_fixed", "sync_ahs", "rtc_fixed", "rtc_ahs"):
                group = [r for r in rows if r["task"] == task and r["backbone"] == backbone and r["method"] == method]
                if not group:
                    continue
                sr = []
                for delay in (0, 100, 200):
                    cell = [r for r in group if r["extra_ms"] == delay]
                    successes = sum(r["success"] for r in cell)
                    sr.append(f"{100*successes/len(cell):.1f}%" if len(cell) == 100 else f"{successes}/{len(cell)}")
                cost = summarize([r for r in group if r["extra_ms"] == 100])
                label = {"sync_fixed": "Sync K40", "sync_ahs": "Sync AHS", "rtc_fixed": "RTC K40", "rtc_ahs": "RTC AHS"}[method]
                lines.append("| " + " | ".join([task, backbone, label, *sr,
                    fmt(cost["wait_pct"]), fmt(cost["calls_ep"]), fmt(cost["infer_s_ep"])]) + " |")
    lines += [
        "",
        "The aggregate eight-task SR and paired confidence intervals remain withheld until every task cell has 100 paired episodes.",
        "TeX is not modified by this refresh.",
        "",
    ]
    output = EXP / "artifacts/tables/formal_panel_live.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
