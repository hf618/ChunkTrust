#!/usr/bin/env python3
"""Create a strict seed+instruction manifest for one held-out RoboTwin split.

This is deliberately a preparation tool, not an evaluator: it runs only the
expert scene-validity preflight for the literal seeds supplied by the caller.
An invalid seed aborts the command; it is never replaced by a nearby one.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
QHA_ROOT = REPO_ROOT / "policy" / "pi05_horizon"
os.chdir(REPO_ROOT)
for path in (REPO_ROOT, REPO_ROOT / "description" / "utils", QHA_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError
from generate_episode_instructions import generate_episode_descriptions
from heldout_protocol import HELDOUT_TASKS, ProtocolError, sha256_file, write_immutable_episode_manifest


def load_registered_seeds(registry_path: Path, cohort: str) -> list[int]:
    registry_path = registry_path.expanduser().resolve()
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProtocolError(f"Manifest registry does not exist: {registry_path}") from exc
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Manifest registry is invalid JSON: {registry_path}: {exc}") from exc
    if registry.get("schema_version") != "qha-heldout-v1":
        raise ProtocolError(f"Manifest registry has unsupported schema: {registry_path}")
    if registry.get("kind") != "qha_heldout_manifest_registry":
        raise ProtocolError(f"Manifest registry has unsupported kind: {registry_path}")
    selected = registry.get(cohort)
    if not isinstance(selected, dict) or not isinstance(selected.get("seeds"), list):
        raise ProtocolError(f"Manifest registry has no seed list for cohort {cohort!r}")
    try:
        seeds = [int(seed) for seed in selected["seeds"]]
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"Manifest registry cohort {cohort!r} has a non-integer seed") from exc
    expected_count = {"smoke": 5, "formal": 50}[cohort]
    if len(seeds) != expected_count or len(set(seeds)) != len(seeds):
        raise ProtocolError(
            f"Manifest registry cohort {cohort!r} must contain exactly {expected_count} unique seeds"
        )
    other_cohort = "formal" if cohort == "smoke" else "smoke"
    other = registry.get(other_cohort, {})
    other_seeds = other.get("seeds") if isinstance(other, dict) else None
    if not isinstance(other_seeds, list) or set(seeds) & {int(seed) for seed in other_seeds}:
        raise ProtocolError("Smoke and formal manifest seed sets must be disjoint")
    return seeds


def task_environment(task_name: str):
    module = __import__(f"envs.{task_name}", fromlist=[task_name])
    try:
        return getattr(module, task_name)()
    except AttributeError as exc:
        raise ProtocolError(f"No RoboTwin task class named {task_name!r}") from exc


def build_task_args(task_name: str, task_config: str) -> tuple[dict, Path]:
    config_path = REPO_ROOT / "task_config" / f"{task_config}.yml"
    if not config_path.is_file():
        raise ProtocolError(f"Task config does not exist: {config_path}")
    args = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(args, dict):
        raise ProtocolError(f"Task config is not a YAML mapping: {config_path}")
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["eval_mode"] = True
    args["render_freq"] = 0

    embodiment_type = args.get("embodiment")
    if not isinstance(embodiment_type, list) or len(embodiment_type) not in (1, 3):
        raise ProtocolError("Task config embodiment must contain one or three entries")
    embodiment_config_path = Path(CONFIGS_PATH) / "_embodiment_config.yml"
    embodiments = yaml.safe_load(embodiment_config_path.read_text(encoding="utf-8"))

    def embodiment_file(name: str) -> str:
        try:
            file_path = embodiments[name]["file_path"]
        except (KeyError, TypeError) as exc:
            raise ProtocolError(f"Embodiment {name!r} has no file_path") from exc
        if not file_path:
            raise ProtocolError(f"Embodiment {name!r} has an empty file_path")
        return str(file_path)

    if len(embodiment_type) == 1:
        args["left_robot_file"] = embodiment_file(embodiment_type[0])
        args["right_robot_file"] = embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    else:
        args["left_robot_file"] = embodiment_file(embodiment_type[0])
        args["right_robot_file"] = embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    return args, config_path


def literal_unseen_instruction(task_name: str, episode_info: dict, *, seed: int, instruction_index: int) -> str:
    # The generator internally shuffles templates and samples object phrases.
    # Freeze that generation with a local reproducible seed, then persist the
    # resulting literal string in the manifest.
    prior_state = random.getstate()
    try:
        random.seed(f"qha-heldout-manifest:{task_name}:{seed}")
        descriptions = generate_episode_descriptions(task_name, [episode_info], max_descriptions=10_000)
    finally:
        random.setstate(prior_state)
    if len(descriptions) != 1:
        raise ProtocolError(f"Seed {seed} produced no description record for task {task_name}")
    unseen = descriptions[0].get("unseen")
    if not isinstance(unseen, list) or instruction_index >= len(unseen):
        raise ProtocolError(
            f"Seed {seed} has no unseen instruction at index {instruction_index}; "
            "choose a lower --instruction-index or a different explicit seed."
        )
    instruction = str(unseen[instruction_index]).strip()
    if not instruction:
        raise ProtocolError(f"Seed {seed} produced an empty unseen instruction")
    return instruction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True, choices=HELDOUT_TASKS)
    parser.add_argument("--task-config", required=True, choices=("demo_clean", "demo_randomized"))
    parser.add_argument("--registry", required=True, type=Path, help="Pre-registered held-out seed registry")
    parser.add_argument("--cohort", required=True, choices=("smoke", "formal"))
    parser.add_argument("--instruction-index", type=int, default=0, help="Index into generated unseen instructions")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.instruction_index < 0:
        parser.error("--instruction-index must be non-negative")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ProtocolError(f"Refusing to overwrite an existing immutable manifest: {output}")
    registry_path = args.registry.expanduser().resolve()
    seeds = load_registered_seeds(registry_path, args.cohort)

    task_args, task_config_path = build_task_args(args.task_name, args.task_config)
    environment = task_environment(args.task_name)
    episodes = []
    for episode_index, seed in enumerate(seeds):
        try:
            environment.setup_demo(now_ep_num=episode_index, seed=seed, is_test=True, **task_args)
            episode_info = environment.play_once()
            if not (environment.plan_success and environment.check_success()):
                raise ProtocolError(f"Seed {seed} failed expert validity preflight")
            instruction = literal_unseen_instruction(
                args.task_name,
                episode_info["info"],
                seed=seed,
                instruction_index=args.instruction_index,
            )
        except UnStableError as exc:
            raise ProtocolError(f"Seed {seed} is unstable and cannot enter the manifest: {exc}") from exc
        finally:
            environment.close_env()
        episodes.append({
            "episode_id": f"{args.task_name}-{args.task_config}-{episode_index:03d}",
            "seed": seed,
            "instruction": instruction,
        })
        print(f"validated {episodes[-1]['episode_id']} seed={seed}")

    digest = write_immutable_episode_manifest(
        output,
        task_name=args.task_name,
        task_config=args.task_config,
        episodes=episodes,
        generator_metadata={
            "generator": str(Path(__file__).resolve()),
            "generated_unix_s": time.time(),
            "instruction_type": "unseen",
            "instruction_index": args.instruction_index,
            "task_config_sha256": sha256_file(task_config_path),
            "task_source_sha256": sha256_file(REPO_ROOT / "envs" / f"{args.task_name}.py"),
            "instruction_template_sha256": sha256_file(
                REPO_ROOT / "description" / "task_instruction" / f"{args.task_name}.json"
            ),
            "manifest_registry_path": str(registry_path),
            "manifest_registry_sha256": sha256_file(registry_path),
            "manifest_cohort": args.cohort,
            "explicit_seeds": seeds,
        },
    )
    print(json.dumps({"manifest": str(output), "sha256": digest, "episodes": len(episodes)}, indent=2))


if __name__ == "__main__":
    main()
