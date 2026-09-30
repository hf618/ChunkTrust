import sys
import os
import subprocess
import time
import json
import hashlib

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path

import yaml
from datetime import datetime
import importlib
import argparse

from generate_episode_instructions import *
from policy_runtime_hooks import call_optional_hook
from policy_runtime_hooks import infer_policy_name
from policy_runtime_hooks import load_policy_hooks

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


class EpisodeManifestError(RuntimeError):
    """Raised when a paired-manifest evaluation cannot remain reproducible."""


def _manifest_file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_episode_manifest(raw_path, *, expected_task, expected_setting, expected_instruction_type):
    """Load a fail-closed literal seed/instruction manifest for paired eval."""
    path = Path(str(raw_path)).expanduser().resolve()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EpisodeManifestError(f"cannot load episode manifest {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise EpisodeManifestError("episode manifest must be a JSON object")
    required = {
        "schema_version",
        "protocol_id",
        "task_name",
        "task_config",
        "instruction_type",
        "entries",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise EpisodeManifestError(f"episode manifest missing fields: {missing}")
    if data["schema_version"] != 1:
        raise EpisodeManifestError(f"unsupported episode manifest schema_version: {data['schema_version']!r}")
    if data["task_name"] != expected_task or data["task_config"] != expected_setting:
        raise EpisodeManifestError(
            "episode manifest task/setting mismatch: "
            f"manifest={data['task_name']}/{data['task_config']}, expected={expected_task}/{expected_setting}"
        )
    if data["instruction_type"] != expected_instruction_type:
        raise EpisodeManifestError(
            "episode manifest instruction_type mismatch: "
            f"manifest={data['instruction_type']!r}, expected={expected_instruction_type!r}"
        )
    protocol_id = str(data["protocol_id"]).strip()
    if not protocol_id:
        raise EpisodeManifestError("episode manifest protocol_id must be non-empty")
    entries = data["entries"]
    if not isinstance(entries, list) or not entries:
        raise EpisodeManifestError("episode manifest entries must be a non-empty list")
    seen_seeds = set()
    normalized_entries = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise EpisodeManifestError(f"episode manifest entry[{index}] must be an object")
        required_entry = {"rollout_id", "episode_seed", "instruction", "instruction_sha1", "precheck_status"}
        missing_entry = sorted(required_entry - entry.keys())
        if missing_entry:
            raise EpisodeManifestError(f"episode manifest entry[{index}] missing fields: {missing_entry}")
        rollout_id = entry["rollout_id"]
        episode_seed = entry["episode_seed"]
        if isinstance(rollout_id, bool) or not isinstance(rollout_id, int) or rollout_id != index:
            raise EpisodeManifestError(f"episode manifest entry[{index}].rollout_id must equal {index}")
        if isinstance(episode_seed, bool) or not isinstance(episode_seed, int):
            raise EpisodeManifestError(f"episode manifest entry[{index}].episode_seed must be an integer")
        if episode_seed in seen_seeds:
            raise EpisodeManifestError(f"episode manifest has duplicate episode_seed: {episode_seed}")
        seen_seeds.add(episode_seed)
        instruction = entry["instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise EpisodeManifestError(f"episode manifest entry[{index}].instruction must be non-empty")
        instruction_sha1 = hashlib.sha1(instruction.encode("utf-8")).hexdigest()
        if entry["instruction_sha1"] != instruction_sha1:
            raise EpisodeManifestError(f"episode manifest entry[{index}].instruction_sha1 mismatch")
        if entry["precheck_status"] != "accepted":
            raise EpisodeManifestError(
                f"episode manifest entry[{index}].precheck_status must be 'accepted', got {entry['precheck_status']!r}"
            )
        normalized_entries.append(
            {
                "rollout_id": int(rollout_id),
                "episode_seed": int(episode_seed),
                "instruction": instruction,
                "instruction_sha1": instruction_sha1,
            }
        )
    return {
        "path": str(path),
        "sha256": _manifest_file_sha256(path),
        "schema_version": int(data["schema_version"]),
        "protocol_id": protocol_id,
        "entries": normalized_entries,
    }


def _append_manifest_status(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")


def _load_completed_manifest_prefix(path, episode_manifest, episodes_dir):
    path = Path(path)
    if not path.is_file():
        return []
    completed = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EpisodeManifestError(f"invalid manifest status JSON at line {line_number}: {exc}") from exc
        if record.get("status") != "completed":
            continue
        if (
            record.get("protocol_id") != episode_manifest["protocol_id"]
            or record.get("manifest_sha256") != episode_manifest["sha256"]
        ):
            raise EpisodeManifestError("existing completed status belongs to a different immutable manifest")
        rollout_id = int(record["rollout_id"])
        if rollout_id in completed:
            raise EpisodeManifestError(f"duplicate completed manifest rollout_id={rollout_id}")
        completed[rollout_id] = record

    expected_prefix = list(range(len(completed)))
    if sorted(completed) != expected_prefix:
        raise EpisodeManifestError(
            "resume requires a contiguous completed manifest prefix; "
            f"found rollout IDs {sorted(completed)}"
        )
    episodes_dir = Path(episodes_dir)
    for rollout_id in expected_prefix:
        if not (episodes_dir / f"episode{rollout_id}.npz").is_file():
            raise EpisodeManifestError(f"completed rollout {rollout_id} is missing its episode NPZ")
    return [completed[index] for index in expected_prefix]


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except Exception:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e


def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]

    hooks = load_policy_hooks(policy_name)
    normalized_args = call_optional_hook(hooks, "normalize_usr_args", dict(usr_args))
    if normalized_args is not None:
        usr_args = normalized_args
        policy_name = usr_args["policy_name"]
        hooks = load_policy_hooks(policy_name)

    eval_type = str(usr_args.get("eval_type", "default"))
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    if "eval_video_log" in usr_args:
        args["eval_video_log"] = bool(usr_args["eval_video_log"])
    args["manifest_setup_max_attempts"] = int(usr_args.get("manifest_setup_max_attempts", 1))
    if args["manifest_setup_max_attempts"] < 1:
        raise ValueError("manifest_setup_max_attempts must be positive")

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(cur_embodiment_type):
        robot_file = _embodiment_types[cur_embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    left_cfg = args["left_embodiment_config"]
    right_cfg = args["right_embodiment_config"]
    left_arm_dim = len(left_cfg["arm_joints_name"][0])
    right_arm_dim = len(right_cfg["arm_joints_name"][1])
    left_gripper_dim = 1 if left_cfg.get("gripper_name") else 0
    right_gripper_dim = 1 if right_cfg.get("gripper_name") else 0

    usr_args["left_arm_dim"] = left_arm_dim
    usr_args["right_arm_dim"] = right_arm_dim
    usr_args["left_action_dim"] = left_arm_dim + left_gripper_dim
    usr_args["right_action_dim"] = right_arm_dim + right_gripper_dim

    layout = call_optional_hook(hooks, "resolve_run_layout", usr_args, task_name, task_config, ckpt_setting, datetime.now())
    use_hook_layout = layout is not None
    if use_hook_layout:
        save_dir = Path(layout["save_dir"])
        info_dir = Path(layout.get("info_dir", save_dir / "info"))
        episodes_dir = Path(layout.get("episodes_dir", save_dir / "episodes"))
        save_dir.mkdir(parents=True, exist_ok=True)
        info_dir.mkdir(parents=True, exist_ok=True)
        episodes_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
        save_dir.mkdir(parents=True, exist_ok=True)
        info_dir = None
        episodes_dir = None

    if args["eval_video_log"]:
        video_save_dir = episodes_dir if use_hook_layout else save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir
    elif use_hook_layout and (
        int(usr_args.get("dump_denoise", 0)) == 1
        or int(usr_args.get("lightweight_eval_logging", 0)) == 1
    ):
        # Keep lightweight manifest traces inside the canonical run directory
        # even when video creation is disabled.
        usr_args["denoise_log_dir"] = str(episodes_dir)

    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    seed = usr_args["seed"]
    st_seed = 100000 * (1 + seed)
    explicit_test_num = os.environ.get("EVAL_TEST_NUM")
    test_num = int(explicit_test_num or "100")
    if test_num <= 0:
        raise ValueError(f"EVAL_TEST_NUM must be positive, got {test_num}")

    episode_manifest = None
    manifest_status_path = None
    manifest_path = usr_args.get("episode_manifest_path")
    if manifest_path:
        if (
            int(usr_args.get("dump_denoise", 0)) != 1
            and int(usr_args.get("lightweight_eval_logging", 0)) != 1
        ):
            raise EpisodeManifestError(
                "episode_manifest_path requires --dump_denoise 1 or "
                "--lightweight_eval_logging 1 for identity verification"
            )
        episode_manifest = _load_episode_manifest(
            manifest_path,
            expected_task=task_name,
            expected_setting=task_config,
            expected_instruction_type=instruction_type,
        )
        entry_count = len(episode_manifest["entries"])
        if explicit_test_num is not None and test_num != entry_count:
            raise EpisodeManifestError(
                "EVAL_TEST_NUM must equal the manifest entry count when episode_manifest_path is set: "
                f"{test_num} != {entry_count}"
            )
        test_num = entry_count
        manifest_status_path = save_dir / "manifest_status.jsonl"

    model = get_model(usr_args)
    run_t0 = time.time()
    st_seed, suc_num = eval_policy(
        task_name,
        TASK_ENV,
        args,
        model,
        st_seed,
        test_num=test_num,
        video_size=video_size,
        instruction_type=instruction_type,
        episode_manifest=episode_manifest,
        manifest_status_path=manifest_status_path,
    )
    total_run_time_seconds = float(time.time() - run_t0)

    if use_hook_layout:
        step_timing_stats = call_optional_hook(hooks, "collect_step_timing_stats", save_dir) or {}
        base_info = {
            "timestamp": current_time,
            "task_name": task_name,
            "task_config": task_config,
            "policy_name": policy_name,
            "ckpt_setting": ckpt_setting,
            "train_config_name": usr_args.get("train_config_name"),
            "model_name": usr_args.get("model_name"),
            "checkpoint_id": usr_args.get("checkpoint_id"),
            "action_chunk_size": int(usr_args.get("pi0_step", 50)),
            "fixed_exec_k": int(usr_args.get("fixed_exec_k", 0)),
            "denoise_npz_compress": int(usr_args.get("denoise_npz_compress", 0)),
            "seed": usr_args.get("seed"),
            "instruction_type": instruction_type,
            "eval_type": eval_type,
            "dump_denoise": int(usr_args.get("dump_denoise", 0)),
            "lightweight_eval_logging": int(usr_args.get("lightweight_eval_logging", 0)),
            "manifest_setup_max_attempts": int(usr_args.get("manifest_setup_max_attempts", 1)),
            "expected_rollouts": int(test_num),
            "success_count": int(suc_num),
            "success_rate": float(suc_num / test_num),
            "total_run_time_seconds": total_run_time_seconds,
        }
        base_info.update(step_timing_stats)
        if episode_manifest is not None:
            base_info.update(
                {
                    "episode_manifest_path": episode_manifest["path"],
                    "episode_manifest_sha256": episode_manifest["sha256"],
                    "episode_manifest_schema_version": episode_manifest["schema_version"],
                    "episode_manifest_protocol_id": episode_manifest["protocol_id"],
                    "episode_manifest_entry_count": len(episode_manifest["entries"]),
                }
            )
        for key, value in layout.items():
            if key.endswith("_dir"):
                continue
            base_info[key] = str(value) if isinstance(value, Path) else value
        info = call_optional_hook(hooks, "build_run_info", dict(base_info), usr_args, save_dir, model=model)
        if info is None:
            info = base_info
        with open(info_dir / "info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=True)
        print(f"Data has been saved to {info_dir / 'info.json'}")
    else:
        file_path = os.path.join(save_dir, "_result.txt")
        with open(file_path, "w") as file:
            file.write(f"Timestamp: {current_time}\n\n")
            file.write(f"Instruction Type: {instruction_type}\n\n")
            file.write("\n".join(map(str, np.array([suc_num]) / test_num)))
        print(f"Data has been saved to {file_path}")


def _eval_policy_from_manifest(
    task_name,
    TASK_ENV,
    args,
    model,
    episode_manifest,
    manifest_status_path,
    *,
    video_size,
):
    """Execute a literal manifest without seed replacement or prompt sampling."""
    completed_prefix = _load_completed_manifest_prefix(
        manifest_status_path,
        episode_manifest,
        Path(manifest_status_path).parent / "episodes",
    )
    TASK_ENV.suc = sum(bool(record.get("success")) for record in completed_prefix)
    TASK_ENV.test_num = len(completed_prefix)
    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")
    clear_cache_freq = args["clear_cache_freq"]
    args["eval_mode"] = True

    for entry in episode_manifest["entries"]:
        rollout_id = entry["rollout_id"]
        if rollout_id < len(completed_prefix):
            continue
        episode_seed = entry["episode_seed"]
        instruction = entry["instruction"]
        instruction_hash = entry["instruction_sha1"]
        render_freq = args["render_freq"]
        video_started = False
        stage = "setup"
        _append_manifest_status(
            manifest_status_path,
            {
                "status": "started",
                "protocol_id": episode_manifest["protocol_id"],
                "manifest_sha256": episode_manifest["sha256"],
                "rollout_id": rollout_id,
                "episode_seed": episode_seed,
                "instruction_hash": instruction_hash,
            },
        )
        try:
            args["render_freq"] = 0
            setup_successes = TASK_ENV.suc
            setup_test_num = TASK_ENV.test_num
            setup_max_attempts = int(args.get("manifest_setup_max_attempts", 1))
            for setup_attempt in range(1, setup_max_attempts + 1):
                try:
                    TASK_ENV.setup_demo(now_ep_num=rollout_id, seed=episode_seed, is_test=True, **args)
                    break
                except UnStableError as exc:
                    if setup_attempt == setup_max_attempts:
                        raise
                    _append_manifest_status(
                        manifest_status_path,
                        {
                            "status": "setup_retry",
                            "protocol_id": episode_manifest["protocol_id"],
                            "manifest_sha256": episode_manifest["sha256"],
                            "rollout_id": rollout_id,
                            "episode_seed": episode_seed,
                            "instruction_hash": instruction_hash,
                            "setup_attempt": setup_attempt,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        },
                    )
                    try:
                        TASK_ENV.close_env()
                    except Exception:
                        pass
                    TASK_ENV = class_decorator(task_name)
                    TASK_ENV.suc = setup_successes
                    TASK_ENV.test_num = setup_test_num
            args["render_freq"] = render_freq
            TASK_ENV.set_instruction(instruction=instruction)

            if TASK_ENV.eval_video_path is not None:
                stage = "start_video"
                ffmpeg = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-y",
                        "-loglevel",
                        "error",
                        "-f",
                        "rawvideo",
                        "-pixel_format",
                        "rgb24",
                        "-video_size",
                        video_size,
                        "-framerate",
                        "10",
                        "-i",
                        "-",
                        "-pix_fmt",
                        "yuv420p",
                        "-vcodec",
                        "libx264",
                        "-crf",
                        "23",
                        f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                    ],
                    stdin=subprocess.PIPE,
                )
                TASK_ENV._set_eval_video_ffmpeg(ffmpeg)
                video_started = True

            stage = "execute"
            succ = False
            ep_t0 = time.time()
            reset_func(model)
            if not hasattr(model, "set_episode_context"):
                raise EpisodeManifestError("policy model does not support episode manifest provenance")
            model.set_episode_context(
                episode_seed=episode_seed,
                instruction_hash=instruction_hash,
                instruction_len=len(instruction),
                episode_manifest_id=episode_manifest["protocol_id"],
                manifest_rollout_id=rollout_id,
                manifest_sha256=episode_manifest["sha256"],
            )
            while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
                observation = TASK_ENV.get_obs()
                eval_func(TASK_ENV, model, observation)
                if TASK_ENV.eval_success:
                    succ = True
                    break

            if video_started:
                TASK_ENV._del_eval_video_ffmpeg()
                video_started = False

            stage = "finalize"
            run_time_seconds = float(time.time() - ep_t0)
            if hasattr(model, "finalize_episode_logging"):
                model.finalize_episode_logging(
                    success=succ,
                    success_step=(TASK_ENV.take_action_cnt if succ else -1),
                    episode_steps=TASK_ENV.take_action_cnt,
                    run_time_seconds=run_time_seconds,
                )

            if succ:
                TASK_ENV.suc += 1
                print()
                print("\033[92mSuccess!\033[0m")
            else:
                print()
                print("\033[91mFail!\033[0m")

            TASK_ENV.close_env(clear_cache=((rollout_id + 1) % clear_cache_freq == 0))
            if TASK_ENV.render_freq:
                TASK_ENV.viewer.close()
            TASK_ENV.test_num += 1
            _append_manifest_status(
                manifest_status_path,
                {
                    "status": "completed",
                    "protocol_id": episode_manifest["protocol_id"],
                    "manifest_sha256": episode_manifest["sha256"],
                    "rollout_id": rollout_id,
                    "episode_seed": episode_seed,
                    "instruction_hash": instruction_hash,
                    "success": bool(succ),
                },
            )
            print(
                f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | "
                f"\033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
                f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => "
                f"\033[95m{round(TASK_ENV.suc / TASK_ENV.test_num * 100, 1)}%\033[0m, "
                f"current seed: \033[90m{episode_seed}\033[0m\n"
            )
        except Exception as exc:
            args["render_freq"] = render_freq
            if video_started:
                try:
                    TASK_ENV._del_eval_video_ffmpeg()
                except Exception:
                    pass
            try:
                TASK_ENV.close_env()
            except Exception:
                pass
            _append_manifest_status(
                manifest_status_path,
                {
                    "status": "error",
                    "protocol_id": episode_manifest["protocol_id"],
                    "manifest_sha256": episode_manifest["sha256"],
                    "rollout_id": rollout_id,
                    "episode_seed": episode_seed,
                    "instruction_hash": instruction_hash,
                    "stage": stage,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            raise EpisodeManifestError(
                f"manifest rollout {rollout_id} failed during {stage}; refusing seed replacement"
            ) from exc
        finally:
            args["render_freq"] = render_freq

    last_seed = episode_manifest["entries"][-1]["episode_seed"]
    return int(last_seed) + 1, TASK_ENV.suc


def eval_policy(
    task_name,
    TASK_ENV,
    args,
    model,
    st_seed,
    test_num=100,
    video_size=None,
    instruction_type=None,
    episode_manifest=None,
    manifest_status_path=None,
):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    if episode_manifest is not None:
        if manifest_status_path is None:
            raise EpisodeManifestError("manifest_status_path is required for manifest evaluation")
        return _eval_policy_from_manifest(
            task_name,
            TASK_ENV,
            args,
            model,
            episode_manifest,
            manifest_status_path,
            video_size=video_size,
        )

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    clear_cache_freq = args["clear_cache_freq"]
    args["eval_mode"] = True

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError:
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception:
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        ep_t0 = time.time()
        reset_func(model)
        if hasattr(model, "set_episode_context"):
            model.set_episode_context(
                episode_seed=int(now_seed),
                instruction_hash=hashlib.sha1(instruction.encode("utf-8")).hexdigest(),
                instruction_len=len(instruction),
            )
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                succ = True
                break
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        run_time_seconds = float(time.time() - ep_t0)
        if hasattr(model, "finalize_episode_logging"):
            model.finalize_episode_logging(
                success=succ,
                success_step=(TASK_ENV.take_action_cnt if succ else -1),
                episode_steps=TASK_ENV.take_action_cnt,
                run_time_seconds=run_time_seconds,
            )

        if succ:
            TASK_ENV.suc += 1
            print()
            print("\033[92mSuccess!\033[0m")
        else:
            print()
            print("\033[91mFail!\033[0m")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc / TASK_ENV.test_num * 100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        now_seed += 1

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--eval_tag",
        type=str,
        default=None,
        help="Optional evaluation tag used by pi05_horizon batch eval launchers.",
    )
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.eval_tag is not None:
        config["eval_tag"] = args.eval_tag

    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except Exception:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        effective_policy_name = infer_policy_name(args.config, config, overrides)
        hooks = load_policy_hooks(effective_policy_name)
        call_optional_hook(hooks, "validate_overrides", config, overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST

    Sapien_TEST()
    usr_args = parse_args_and_config()
    main(usr_args)
