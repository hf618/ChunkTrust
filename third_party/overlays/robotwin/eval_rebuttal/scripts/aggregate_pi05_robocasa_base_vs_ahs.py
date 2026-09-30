#!/usr/bin/env python3
"""Verify and aggregate the complete paired pi0.5 RoboCasa Base-vs-AHS run."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from pi05_robocasa_common import (  # noqa: E402
    AHS_CONFIG,
    EXPERIMENT_ROOT,
    MODEL_ID,
    NORM_ASSET_ID,
    NORM_STATS_SHA256,
    OPTIMIZER_STEP,
    PACKAGE_TREE_SHA256,
    PROTOCOL_ID,
    atomic_write_json,
    atomic_write_text,
    file_sha256,
    manifest_rows,
    result_key,
    task_records,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260801)
    return parser.parse_args()


def load_result_files(root: Path, method: str) -> list[tuple[Path, dict[str, Any]]]:
    paths = sorted((root / f"raw_results/formal/{method}").rglob("*.json"))
    results = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "COMPLETED":
            results.append((path, payload))
    return results


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else float("nan")


def efficiency(results: list[dict[str, Any]]) -> dict[str, float]:
    calls = sum(int(row["policy_calls"]) for row in results)
    successes = sum(bool(row["outcome"]) for row in results)
    return {
        "episodes": len(results),
        "successes": successes,
        "success_rate": successes / len(results),
        "policy_calls_total": calls,
        "policy_calls_per_episode": calls / len(results),
        "policy_inference_seconds_per_episode": mean(
            [float(row["policy_inference_seconds"]) for row in results]
        ),
        "server_inference_seconds_per_episode": mean(
            [float(row["server_inference_seconds"]) for row in results]
        ),
        "policy_roundtrip_seconds_per_episode": mean(
            [float(row["policy_roundtrip_seconds"]) for row in results]
        ),
        "ahs_overhead_seconds_per_episode": mean([float(row["ahs_overhead_seconds"]) for row in results]),
        "wall_seconds_per_episode": mean([float(row["wall_seconds"]) for row in results]),
        "successes_per_1k_policy_calls": (successes * 1000.0 / calls) if calls else float("nan"),
    }


def trace_line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def validate_method(
    *,
    method: str,
    files: list[tuple[Path, dict[str, Any]]],
    manifest: list[dict[str, Any]],
    manifest_sha: str,
    protocol_sha: str,
    runtime_sha: str,
) -> dict[tuple[str, int, str], dict[str, Any]]:
    if len(files) != 1200:
        raise RuntimeError(f"{method}: expected 1200 completed records, found {len(files)}")
    by_identity: dict[tuple[str, int, str], dict[str, Any]] = {}
    allowed_tasks = {row["task_name"] for row in manifest}
    for path, result in files:
        identity = (result["task_name"], int(result["seed"]), result["instruction"])
        if identity in by_identity:
            raise RuntimeError(f"{method}: duplicate identity {identity}: {path}")
        if result["task_name"] not in allowed_tasks:
            raise RuntimeError(f"{method}: unexpected task {result['task_name']}")
        if result.get("method") != method:
            raise RuntimeError(f"{path}: method drift")
        if result.get("protocol_id") != PROTOCOL_ID:
            raise RuntimeError(f"{path}: protocol drift")
        if result.get("manifest_sha256") != manifest_sha:
            raise RuntimeError(f"{path}: manifest drift")
        if result.get("frozen_protocol_sha256") != protocol_sha:
            raise RuntimeError(f"{path}: frozen protocol drift")
        if result.get("runtime_sha256") != runtime_sha:
            raise RuntimeError(f"{path}: runtime drift")
        model = result.get("model", {})
        if (
            model.get("id") != MODEL_ID
            or int(model.get("optimizer_step", -1)) != OPTIMIZER_STEP
            or model.get("package_tree_sha256") != PACKAGE_TREE_SHA256
            or model.get("norm_asset_id") != NORM_ASSET_ID
            or model.get("norm_stats_sha256") != NORM_STATS_SHA256
        ):
            raise RuntimeError(f"{path}: checkpoint/package drift")
        selected = [int(value) for value in result.get("selected_k", [])]
        if len(selected) != int(result["policy_calls"]):
            raise RuntimeError(f"{path}: selected-K/policy-call count mismatch")
        if method == "base" and any(value != 16 for value in selected):
            raise RuntimeError(f"{path}: Base selected non-16 horizon")
        if method == "ahs" and any(value < 4 or value > 16 for value in selected):
            raise RuntimeError(f"{path}: AHS expected-round horizon outside [4,16]")
        if method == "ahs" and result.get("ahs_config") != AHS_CONFIG:
            raise RuntimeError(f"{path}: AHS configuration drift")
        trace_path = Path(result["trace_path"])
        if not trace_path.exists() or trace_line_count(trace_path) != int(result["policy_calls"]):
            raise RuntimeError(f"{path}: missing or incomplete replan trace")
        by_identity[identity] = result

    expected = {(row["task_name"], int(row["seed"]), row["instruction"]) for row in manifest}
    if set(by_identity) != expected:
        missing = sorted(expected - set(by_identity))[:10]
        extra = sorted(set(by_identity) - expected)[:10]
        raise RuntimeError(f"{method}: identity mismatch missing={missing} extra={extra}")
    return by_identity


def task_cluster_bootstrap(
    task_deltas: dict[str, float], *, replicates: int, seed: int
) -> dict[str, Any]:
    task_names = sorted(task_deltas)
    values = np.asarray([task_deltas[task] for task in task_names], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    return {
        "unit": "task",
        "task_count": len(task_names),
        "paired_identities_per_task": 50,
        "replicates": replicates,
        "seed": seed,
        "estimate": float(values.mean()),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "probability_delta_gt_zero": float(np.mean(samples > 0.0)),
    }


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    *,
    summary_rows: list[dict[str, Any]],
    base_eff: dict[str, float],
    ahs_eff: dict[str, float],
    bootstrap: dict[str, Any],
    selected_k_distribution: dict[int, int],
    manifest_sha: str,
    protocol_sha: str,
    runtime_sha: str,
) -> str:
    ci = bootstrap["ci95"]
    lines = [
        "# Pi0.5 RoboCasa GR1 Tabletop-24 Base vs AHS",
        "",
        "## Completion",
        "",
        "- Status: **PASS**",
        "- Scope: 24 official tasks x 50 paired identities x 2 methods = 2,400 formal rollouts.",
        f"- Checkpoint: `{MODEL_ID}` step `{OPTIMIZER_STEP}`; package tree `{PACKAGE_TREE_SHA256}`.",
        f"- Manifest SHA-256: `{manifest_sha}`.",
        f"- Frozen protocol SHA-256: `{protocol_sha}`.",
        f"- Runtime SHA-256: `{runtime_sha}`.",
        "- Base: fixed K=16. AHS: candidates {4,8,12,16}, expected-round T=0.8, no QHA.",
        "- Expected-round can select any integer K in [4,16]; it is not restricted to candidate anchors.",
        "",
        "## Aggregate Results",
        "",
        "| Method | Successes / 1200 | SR (%) | Calls/episode | Policy inference/episode (s) | AHS overhead/episode (s) | Wall/episode (s) | Successes/1k calls |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| Base | {base_eff['successes']} / 1200 | {fmt_pct(base_eff['success_rate'])} | "
            f"{base_eff['policy_calls_per_episode']:.2f} | {base_eff['policy_inference_seconds_per_episode']:.2f} | "
            f"0.0000 | {base_eff['wall_seconds_per_episode']:.2f} | "
            f"{base_eff['successes_per_1k_policy_calls']:.2f} |"
        ),
        (
            f"| AHS | {ahs_eff['successes']} / 1200 | {fmt_pct(ahs_eff['success_rate'])} | "
            f"{ahs_eff['policy_calls_per_episode']:.2f} | {ahs_eff['policy_inference_seconds_per_episode']:.2f} | "
            f"{ahs_eff['ahs_overhead_seconds_per_episode']:.4f} | {ahs_eff['wall_seconds_per_episode']:.2f} | "
            f"{ahs_eff['successes_per_1k_policy_calls']:.2f} |"
        ),
        "",
        (
            f"Task-macro paired AHS-Base delta: **{100.0 * bootstrap['estimate']:+.2f} pp** "
            f"(task-cluster paired-bootstrap 95% CI [{100.0 * ci[0]:+.2f}, {100.0 * ci[1]:+.2f}] pp)."
        ),
        "",
        "## Task-level Results",
        "",
        "| Task | Base | AHS | Delta (pp) | Base calls/ep | AHS calls/ep |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['task_name']} | {row['base_successes']}/50 ({row['base_sr_pct']:.1f}) | "
            f"{row['ahs_successes']}/50 ({row['ahs_sr_pct']:.1f}) | {row['delta_pp']:+.1f} | "
            f"{row['base_calls_per_episode']:.2f} | {row['ahs_calls_per_episode']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Horizon Distribution",
            "",
            "```json",
            json.dumps({str(k): v for k, v in selected_k_distribution.items()}, indent=2, sort_keys=True),
            "```",
            "",
            "## Claim Guard",
            "",
            "This experiment controls the policy **architecture/family**, not identical weights across benchmarks. "
            "The RoboCasa checkpoint is benchmark-specific and adapted to the GR1 embodiment, 44D state, 29D absolute-action interface, and RoboCasa demonstrations. "
            "The measured delta may be used only after preserving the complete task table and paired CI above.",
            "",
        ]
    )
    return "\n".join(lines)


def build_response_draft(base_eff: dict[str, float], ahs_eff: dict[str, float], bootstrap: dict[str, Any]) -> str:
    delta_pp = 100.0 * bootstrap["estimate"]
    ci = [100.0 * value for value in bootstrap["ci95"]]
    if ci[0] > 0:
        result_sentence = (
            f"AHS improves the 24-task macro success rate by **{delta_pp:+.2f} pp** "
            f"(task-cluster paired-bootstrap 95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}] pp)."
        )
    elif ci[1] < 0:
        result_sentence = (
            f"In this control, AHS changes the 24-task macro success rate by **{delta_pp:+.2f} pp** "
            f"(95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}] pp), so it does not support a positive success-rate claim."
        )
    else:
        result_sentence = (
            f"AHS changes the 24-task macro success rate by **{delta_pp:+.2f} pp** "
            f"(95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}] pp); the interval includes zero."
        )
    return "\n".join(
        [
            "# Reviewer tVFj Q3 Response Draft",
            "",
            "Two considerations motivated the original benchmark-native policy choice. First, each benchmark provides mature checkpoints or training recipes matched to its embodiment, interface, demonstrations, and evaluation stack; using these policies establishes a reasonable base-policy competence floor and avoids conflating horizon selection with an under-adapted backbone. Second, our intended claim is backbone-agnostic horizon selection across heterogeneous policy families.",
            "",
            "To disentangle policy-family choice from benchmark effects, we added a controlled RoboCasa GR1 Tabletop-24 experiment using a pi0.5 checkpoint and compared fixed K=16 against the same frozen AHS selector. The checkpoint shares the pi0.5 architecture/policy family with our RoboTwin experiment, but is separately adapted to RoboCasa's GR1 embodiment, state/action interface, and benchmark demonstrations; it is not the identical set of weights.",
            "",
            f"Across all 24 tasks and 1,200 paired episodes per method, Base achieves **{fmt_pct(base_eff['success_rate'])}%** and AHS achieves **{fmt_pct(ahs_eff['success_rate'])}%**. {result_sentence}",
            "",
            f"AHS uses {ahs_eff['policy_calls_per_episode']:.2f} policy calls/episode versus {base_eff['policy_calls_per_episode']:.2f} for Base, with {ahs_eff['ahs_overhead_seconds_per_episode']:.4f}s selector overhead/episode. The full task-level table and protocol are provided in the accompanying report.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    manifest_path = (args.manifest or root / "formal_manifest.jsonl").resolve()
    manifest = manifest_rows(manifest_path)
    if len(manifest) != 1200:
        raise RuntimeError(f"Expected 1200 formal manifest rows, found {len(manifest)}")
    expected_tasks = [item["task_name"] for item in task_records()]
    manifest_tasks = list(dict.fromkeys(row["task_name"] for row in manifest))
    if manifest_tasks != expected_tasks:
        raise RuntimeError("Formal manifest task order/membership drift")
    manifest_identities = [(row["task_name"], int(row["seed"]), row["instruction"]) for row in manifest]
    if len(set(manifest_identities)) != 1200:
        raise RuntimeError("Formal manifest contains duplicate literal identities")
    for task in expected_tasks:
        seeds = [int(row["seed"]) for row in manifest if row["task_name"] == task]
        if seeds != list(range(50)):
            raise RuntimeError(f"{task}: formal seeds are not exactly ordered 0..49")

    manifest_sha = file_sha256(manifest_path)
    protocol_path = root / "FROZEN_EVAL_PROTOCOL.json"
    runtime_path = root / "provenance/RUNTIME_SOURCE_HASHES.json"
    protocol_sha = file_sha256(protocol_path)
    runtime_payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime_sha = runtime_payload["runtime_sha256"]

    base_files = load_result_files(root, "base")
    ahs_files = load_result_files(root, "ahs")
    base = validate_method(
        method="base",
        files=base_files,
        manifest=manifest,
        manifest_sha=manifest_sha,
        protocol_sha=protocol_sha,
        runtime_sha=runtime_sha,
    )
    ahs = validate_method(
        method="ahs",
        files=ahs_files,
        manifest=manifest,
        manifest_sha=manifest_sha,
        protocol_sha=protocol_sha,
        runtime_sha=runtime_sha,
    )
    if set(base) != set(ahs):
        raise RuntimeError("Base/AHS paired identity mismatch")

    ordered_identities = [(row["task_name"], int(row["seed"]), row["instruction"]) for row in manifest]
    base_results = [base[identity] for identity in ordered_identities]
    ahs_results = [ahs[identity] for identity in ordered_identities]
    write_jsonl(root / "base_results.jsonl", base_results)
    write_jsonl(root / "ahs_results.jsonl", ahs_results)

    summary_rows = []
    task_deltas = {}
    for task in expected_tasks:
        base_task = [row for row in base_results if row["task_name"] == task]
        ahs_task = [row for row in ahs_results if row["task_name"] == task]
        base_eff_task = efficiency(base_task)
        ahs_eff_task = efficiency(ahs_task)
        delta = ahs_eff_task["success_rate"] - base_eff_task["success_rate"]
        task_deltas[task] = delta
        summary_rows.append(
            {
                "task_name": task,
                "base_successes": base_eff_task["successes"],
                "base_sr_pct": 100.0 * base_eff_task["success_rate"],
                "ahs_successes": ahs_eff_task["successes"],
                "ahs_sr_pct": 100.0 * ahs_eff_task["success_rate"],
                "delta_pp": 100.0 * delta,
                "base_calls_per_episode": base_eff_task["policy_calls_per_episode"],
                "ahs_calls_per_episode": ahs_eff_task["policy_calls_per_episode"],
                "base_inference_seconds_per_episode": base_eff_task["policy_inference_seconds_per_episode"],
                "ahs_inference_seconds_per_episode": ahs_eff_task["policy_inference_seconds_per_episode"],
                "base_wall_seconds_per_episode": base_eff_task["wall_seconds_per_episode"],
                "ahs_wall_seconds_per_episode": ahs_eff_task["wall_seconds_per_episode"],
                "base_successes_per_1k_calls": base_eff_task["successes_per_1k_policy_calls"],
                "ahs_successes_per_1k_calls": ahs_eff_task["successes_per_1k_policy_calls"],
            }
        )
    write_csv(root / "task_level_summary.csv", summary_rows)

    bootstrap = task_cluster_bootstrap(
        task_deltas,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    base_eff = efficiency(base_results)
    ahs_eff = efficiency(ahs_results)
    selected_values = [int(k) for row in ahs_results for k in row["selected_k"]]
    selected_distribution = {value: selected_values.count(value) for value in sorted(set(selected_values))}
    categories = {"both_success": 0, "base_only": 0, "ahs_only": 0, "both_fail": 0}
    for identity in ordered_identities:
        b = bool(base[identity]["outcome"])
        a = bool(ahs[identity]["outcome"])
        key = "both_success" if b and a else "base_only" if b else "ahs_only" if a else "both_fail"
        categories[key] += 1

    verification = {
        "schema_version": 1,
        "status": "PASS",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "frozen_protocol_sha256": protocol_sha,
        "runtime_sha256": runtime_sha,
        "task_count": 24,
        "identities_per_task": 50,
        "base_records": len(base_results),
        "ahs_records": len(ahs_results),
        "total_formal_rollouts": len(base_results) + len(ahs_results),
        "paired_outcome_categories": categories,
        "base": base_eff,
        "ahs": ahs_eff,
        "paired_task_cluster_bootstrap": bootstrap,
        "ahs_selected_k_distribution": selected_distribution,
    }
    atomic_write_json(root / "paired_verification.json", verification)
    atomic_write_json(root / "analysis/paired_bootstrap.json", bootstrap)
    atomic_write_json(root / "analysis/aggregate_metrics.json", verification)
    atomic_write_text(
        root / "FINAL_PI05_ROBOCASA_BASE_VS_AHS_REPORT.md",
        build_report(
            summary_rows=summary_rows,
            base_eff=base_eff,
            ahs_eff=ahs_eff,
            bootstrap=bootstrap,
            selected_k_distribution=selected_distribution,
            manifest_sha=manifest_sha,
            protocol_sha=protocol_sha,
            runtime_sha=runtime_sha,
        ),
    )
    atomic_write_text(
        root / "REVIEWER_TVFJ_Q3_RESPONSE_DRAFT.md",
        build_response_draft(base_eff, ahs_eff, bootstrap),
    )
    print(json.dumps({"status": "PASS", "root": str(root), "verification": verification}, sort_keys=True))


if __name__ == "__main__":
    main()
