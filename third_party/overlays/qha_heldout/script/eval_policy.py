import sys
import os
import subprocess
import time
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

_QHA_ROOT = Path(__file__).resolve().parents[1] / "policy" / "pi05_horizon"
if str(_QHA_ROOT) not in sys.path:
    sys.path.insert(0, str(_QHA_ROOT))
from heldout_protocol import ProtocolError, load_episode_manifest

from generate_episode_instructions import *
from eval_result_paths import build_eval_result_layout_from_args

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
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


def parse_bool_value(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def main(usr_args):
    # Record start time
    start_time = datetime.now()
    result_layout = build_eval_result_layout_from_args(
        usr_args,
        eval_root=usr_args.get("eval_root", "eval_result"),
        timestamp=start_time,
    )
    start_timestamp = result_layout.timestamp
    start_time_iso = start_time.isoformat() + "Z"
    
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    video_save_dir = None
    video_size = None
    
    # Setup logging with batch log directory support
    batch_log_dir = usr_args.get("batch_log_dir")
    
    if batch_log_dir:
        # Use batch log structure (pi0-style)
        log_dir = Path(batch_log_dir) / task_name
        log_dir.mkdir(parents=True, exist_ok=True)
        # Log file already created by auto_eval.sh, but we set the path for reference
        usr_args["log_dir"] = batch_log_dir
    else:
        # Fallback to old structure
        log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(f"policy/{policy_name}/logs/eval/{policy_name}/{task_name}")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"eval_{task_config}_{log_timestamp}.log"
        usr_args["log_dir"] = str(log_dir.parent.parent)

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting
    args["skip_get_obs_within_replan"] = usr_args.get(
        "skip_get_obs_within_replan",
        args.get("skip_get_obs_within_replan", False),
    )
    if "eval_video_log" in usr_args and usr_args["eval_video_log"] is not None:
        args["eval_video_log"] = parse_bool_value(usr_args["eval_video_log"])

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
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

    ckpt_name = Path(ckpt_setting).name if ckpt_setting else "unknown_ckpt"
    save_dir = result_layout.save_dir
    episodes_dir = save_dir / "episodes"
    info_dir = save_dir / "info"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)
    usr_args["result_dir"] = str(save_dir)

    # A manifest is an evaluation contract, not a seed generator. It must be
    # loaded before model construction so every policy gets the same log path.
    episode_manifest = None
    episode_manifest_sha256 = None
    episode_manifest_path = usr_args.get("episode_manifest_path")
    if episode_manifest_path:
        episode_manifest, episode_manifest_sha256 = load_episode_manifest(
            episode_manifest_path,
            task_name=task_name,
            task_config=task_config,
        )
        if int(usr_args.get("rollouts_parallel", 1)) != 1:
            raise ProtocolError("Immutable seed+instruction manifests require rollouts_parallel=1.")
        usr_args["denoise_log_dir"] = str(episodes_dir)

    if args["eval_video_log"]:
        video_save_dir = episodes_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        args["eval_video_save_dir"] = video_save_dir

    # Output evaluation header
    print("=" * 60)
    print(f"Evaluating model: {ckpt_setting}")
    print(f"Task: {task_name}")
    print(f"Config: {task_config}")
    print(f"Result Dir: {save_dir}")
    print("=" * 60)
    
    # Output camera config
    print("\n============= Config =============")
    print(f"Messy Table: {args['domain_randomization']['cluttered_table']}")
    print(f"Random Background: {args['domain_randomization']['random_background']}")
    if args["domain_randomization"]["random_background"]:
        print(f" - Clean Background Rate: {args['domain_randomization']['clean_background_rate']}")
    print(f"Random Light: {args['domain_randomization']['random_light']}")
    if args["domain_randomization"]["random_light"]:
        print(f" - Crazy Random Light Rate: {args['domain_randomization']['crazy_random_light_rate']}")
    print(f"Random Table Height: {args['domain_randomization']['random_table_height']}")
    print(f"Random Head Camera Distance: {args['domain_randomization']['random_head_camera_dis']}")
    print(f"Head Camera Config: {args['camera']['head_camera_type']}, {args['camera']['collect_head_camera']}")
    print(f"Wrist Camera Config: {args['camera']['wrist_camera_type']}, {args['camera']['collect_wrist_camera']}")
    print(f"Embodiment Config: {embodiment_name}")
    print("==================================\n")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = len(episode_manifest["episodes"]) if episode_manifest is not None else int(
        usr_args.get("eval_num_episodes", 100)
    )
    if test_num <= 0:
        raise ValueError(f"`eval_num_episodes` must be positive, got {test_num}")
    topk = 1

    # Get rollouts_parallel parameter
    rollouts_parallel = usr_args.get("rollouts_parallel", 1)
    
    print(f"\n\033[33m=== Parallel Configuration ===\033[0m")
    print(f"Rollouts Parallel: {rollouts_parallel}")
    episode_outcomes = []

    # The manifest has already been validated above. Construct the policy before
    # dispatching rollouts so the immutable path cannot reference an undefined
    # model or fall through into the legacy seed-based evaluator.
    model = get_model(usr_args)
    if episode_manifest is not None:
        st_seed, suc_num, total_steps, episode_steps_list = eval_policy_manifest(
            task_name,
            TASK_ENV,
            args,
            model,
            episodes=episode_manifest["episodes"],
            video_size=video_size,
            outcome_records=episode_outcomes,
        )
    elif rollouts_parallel > 1:
        print(f"\033[32m✓ Parallel execution enabled ({rollouts_parallel} workers)\033[0m")
        st_seed, suc_num, total_steps, episode_steps_list = eval_policy_parallel(
            task_name,
            TASK_ENV,
            args,
            model,
            st_seed,
            test_num=test_num,
            video_size=video_size,
            instruction_type=instruction_type,
            rollouts_parallel=rollouts_parallel,
            usr_args=usr_args,
        )
    else:
        print(f"Serial execution (single worker)")
        st_seed, suc_num, total_steps, episode_steps_list = eval_policy(
            task_name,
            TASK_ENV,
            args,
            model,
            st_seed,
            test_num=test_num,
            video_size=video_size,
            instruction_type=instruction_type,
        )
    print("================================\n")

    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    # Save result.txt
    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {start_timestamp}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")
    
    # Calculate runtime
    end_time = datetime.now()
    end_time_iso = end_time.isoformat() + "Z"
    runtime_seconds = int((end_time - start_time).total_seconds())
    
    # Format runtime as human-readable string
    hours = runtime_seconds // 3600
    minutes = (runtime_seconds % 3600) // 60
    seconds = runtime_seconds % 60
    
    if hours > 0:
        runtime_formatted = f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        runtime_formatted = f"{minutes}m {seconds}s"
    else:
        runtime_formatted = f"{seconds}s"
    
    # Calculate mean step time
    mean_step_time_seconds = runtime_seconds / total_steps if total_steps > 0 else 0
    
    # Generate info.json with runtime statistics and new fields
    import json
    import shutil
    info_data = {
        "timestamp": start_timestamp,
        "task_name": task_name,
        "task_config": task_config,
        "policy_name": policy_name,
        "ckpt_name": ckpt_name,
        "ckpt_path": ckpt_setting,
        "instruction_type": instruction_type,
        "expected_rollouts": test_num,
        "success_count": suc_num,
        "success_rate": round(suc_num / test_num, 4),
        "seed": seed if seed >= 0 else None,
        "run_start_time": start_time_iso,
        "run_end_time": end_time_iso,
        "total_runtime_seconds": runtime_seconds,
        "total_runtime_formatted": runtime_formatted,
        "total_episode_steps": total_steps,
        "mean_step_time_seconds": round(mean_step_time_seconds, 4),
        **result_layout.metadata(),
    }
    if episode_manifest is not None:
        outcomes_path = info_dir / "episode_outcomes.jsonl"
        with outcomes_path.open("w", encoding="utf-8") as handle:
            for record in episode_outcomes:
                handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
        info_data.update({
            "immutable_episode_manifest_path": str(Path(episode_manifest_path).resolve()),
            "immutable_episode_manifest_sha256": episode_manifest_sha256,
            "immutable_episode_count": len(episode_manifest["episodes"]),
            "episode_outcomes_path": str(outcomes_path),
            "heldout_runtime_config": {
                key: usr_args.get(key)
                for key in (
                    "eval_type",
                    "use_qha",
                    "qha_infer_mode_override",
                    "qha_candidate_mode_override",
                    "qha_selector_exec_mode_override",
                    "qha_selector_expected_temp_override",
                    "qha_hybrid_prior_gamma_override",
                    "horizon_candidates",
                    "horizon_exec_mode",
                    "horizon_expected_temp",
                    "checkpoint_root_tag",
                    "model_name",
                    "checkpoint_id",
                    "base_checkpoint_root_tag",
                    "base_model_name",
                    "base_checkpoint_id",
                )
            },
        })
    
    info_path = info_dir / "info.json"
    with open(info_path, "w") as f:
        json.dump(info_data, f, indent=2, ensure_ascii=False)
    
    print(f"Info saved to {info_path}")
    print(f"Total runtime: {runtime_formatted} ({runtime_seconds}s)")
    print(f"Total steps: {total_steps}, Mean step time: {mean_step_time_seconds:.4f}s")
    
    # Copy robotwin.yml to info directory
    robotwin_yml_src = Path(f"policy/{policy_name}/utils/robotwin.yml")
    robotwin_yml_dst = info_dir / "robotwin.yml"
    
    if robotwin_yml_src.exists():
        shutil.copy2(robotwin_yml_src, robotwin_yml_dst)
        print(f"Copied robotwin.yml to {robotwin_yml_dst}")
    else:
        print(f"Warning: robotwin.yml not found at {robotwin_yml_src}")
    
    # Copy paths_config.yml to info directory
    paths_config_src = Path(f"policy/{policy_name}/paths_config.yml")
    paths_config_dst = info_dir / "paths_config.yml"
    
    if paths_config_src.exists():
        shutil.copy2(paths_config_src, paths_config_dst)
        print(f"Copied paths_config.yml to {paths_config_dst}")
    else:
        print(f"Warning: paths_config.yml not found at {paths_config_src}")
    # return task_reward


def eval_policy_parallel(task_name,
                        TASK_ENV,
                        args,
                        model,
                        st_seed,
                        test_num=100,
                        video_size=None,
                        instruction_type=None,
                        rollouts_parallel=2,
                        usr_args=None):
    """
    Parallel evaluation using multiprocessing.
    Splits rollouts across multiple worker processes.
    """
    print(f"\033[32m[Parallel Mode] Launching {rollouts_parallel} workers...\033[0m")
    
    # Split rollouts across workers
    rollouts_per_worker = test_num // rollouts_parallel
    remainder = test_num % rollouts_parallel
    
    # Create worker assignments
    worker_assignments = []
    current_seed = st_seed
    for worker_id in range(rollouts_parallel):
        worker_rollouts = rollouts_per_worker + (1 if worker_id < remainder else 0)
        worker_assignments.append({
            'worker_id': worker_id,
            'start_seed': current_seed,
            'num_rollouts': worker_rollouts,
            'task_name': task_name,
            'args': args,
            'video_size': video_size,
            'instruction_type': instruction_type,
            'usr_args': usr_args
        })
        current_seed += worker_rollouts * 1000  # Ensure seed separation
        print(f"  Worker {worker_id}: {worker_rollouts} rollouts (seed {worker_assignments[worker_id]['start_seed']})")
    
    # Use multiprocessing Pool for parallel execution
    print(f"\033[32m[Parallel Mode] Starting parallel execution...\033[0m")
    
    # Note: Due to SAPIEN's complexity, we'll use the serial implementation for now
    # and add a warning that parallel execution requires additional setup
    print(f"\033[33m[Warning] Parallel rollouts execution requires additional SAPIEN configuration.\033[0m")
    print(f"\033[33m[Warning] Falling back to serial execution for safety.\033[0m")
    print(f"\033[33m[Warning] To enable true parallel execution, please contact the developers.\033[0m")
    
    # Fall back to serial execution
    return eval_policy(task_name, TASK_ENV, args, model, st_seed, test_num, video_size, instruction_type)


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]
    skip_get_obs_within_replan = parse_bool_value(args.get("skip_get_obs_within_replan", False))

    # Track episode steps
    total_steps = 0
    episode_steps_list = []

    args["eval_mode"] = True

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                # stack_trace = traceback.format_exc()
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        TASK_ENV.current_eval_seed = now_seed
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

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
        reset_func(model)
        episode_steps = 0
        observation = None
        episode_wall_start = time.time()
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            needs_observation = True
            if skip_get_obs_within_replan and hasattr(model, "should_request_observation"):
                try:
                    needs_observation = bool(model.should_request_observation())
                except Exception:
                    needs_observation = True
            if needs_observation:
                observation = TASK_ENV.get_obs()
                eval_func(TASK_ENV, model, observation)
            else:
                eval_func(TASK_ENV, model, None)
            episode_steps += 1
            if TASK_ENV.eval_success:
                succ = True
                break
        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if hasattr(model, "finalize_episode_logging"):
            try:
                model.finalize_episode_logging(
                    success=succ,
                    success_step=episode_steps if succ else None,
                    episode_steps=episode_steps,
                    run_time_seconds=time.time() - episode_wall_start,
                )
            except Exception as exc:
                print(f"Warning: finalize_episode_logging failed: {exc}")

        # Record episode steps
        total_steps += episode_steps
        episode_steps_list.append(episode_steps)

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1

    return now_seed, TASK_ENV.suc, total_steps, episode_steps_list


def eval_policy_manifest(
    task_name,
    TASK_ENV,
    args,
    model,
    *,
    episodes,
    video_size=None,
    outcome_records=None,
):
    """Evaluate literal manifest episodes without seed or instruction substitution."""
    print(f"\033[34mTask Name: {args['task_name']} (immutable manifest)\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0
    args["eval_mode"] = True
    clear_cache_freq = args["clear_cache_freq"]
    skip_get_obs_within_replan = parse_bool_value(args.get("skip_get_obs_within_replan", False))
    eval_func = eval_function_decorator(args["policy_name"], "eval")
    reset_func = eval_function_decorator(args["policy_name"], "reset_model")
    total_steps = 0
    episode_steps_list = []
    outcome_records = outcome_records if outcome_records is not None else []

    for episode_index, manifest_episode in enumerate(episodes):
        literal_seed = int(manifest_episode["seed"])
        literal_instruction = str(manifest_episode["instruction"])
        episode_id = str(manifest_episode["episode_id"])
        render_freq = args["render_freq"]
        args["render_freq"] = 0
        try:
            # This is a validity check only. Unlike the legacy path, a bad seed
            # is a protocol failure and is never replaced by a nearby seed.
            TASK_ENV.setup_demo(now_ep_num=episode_index, seed=literal_seed, is_test=True, **args)
            TASK_ENV.play_once()
            if not (TASK_ENV.plan_success and TASK_ENV.check_success()):
                raise ProtocolError(
                    f"Manifest episode {episode_id} seed={literal_seed} failed expert validity preflight."
                )
        except Exception as exc:
            TASK_ENV.close_env()
            if isinstance(exc, ProtocolError):
                raise
            raise ProtocolError(
                f"Manifest episode {episode_id} seed={literal_seed} failed expert validity preflight: {exc}"
            ) from exc
        TASK_ENV.close_env()
        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=episode_index, seed=literal_seed, is_test=True, **args)
        TASK_ENV.current_eval_seed = literal_seed
        TASK_ENV.set_instruction(instruction=literal_instruction)
        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                    "-pixel_format", "rgb24", "-video_size", video_size, "-framerate", "10",
                    "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        reset_func(model)
        succ = False
        episode_steps = 0
        observation = None
        episode_wall_start = time.time()
        try:
            while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
                needs_observation = True
                if skip_get_obs_within_replan and hasattr(model, "should_request_observation"):
                    try:
                        needs_observation = bool(model.should_request_observation())
                    except Exception:
                        needs_observation = True
                if needs_observation:
                    observation = TASK_ENV.get_obs()
                    eval_func(TASK_ENV, model, observation)
                else:
                    eval_func(TASK_ENV, model, None)
                episode_steps += 1
                if TASK_ENV.eval_success:
                    succ = True
                    break
        finally:
            if TASK_ENV.eval_video_path is not None:
                TASK_ENV._del_eval_video_ffmpeg()

        episode_wall_seconds = time.time() - episode_wall_start
        if hasattr(model, "finalize_episode_logging"):
            model.finalize_episode_logging(
                success=succ,
                success_step=episode_steps if succ else None,
                episode_steps=episode_steps,
                run_time_seconds=episode_wall_seconds,
            )
        total_steps += episode_steps
        episode_steps_list.append(episode_steps)
        if succ:
            TASK_ENV.suc += 1
        outcome_records.append({
            "episode_id": episode_id,
            "seed": literal_seed,
            "instruction": literal_instruction,
            "success": bool(succ),
            "episode_steps": int(episode_steps),
            "episode_wall_seconds": float(episode_wall_seconds),
            "replan_log": str(Path(getattr(model, "_replan_log_path", ""))) if getattr(model, "_replan_log_path", None) else None,
        })
        TASK_ENV.close_env(clear_cache=((episode_index + 1) % clear_cache_freq == 0))
        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()
        TASK_ENV.test_num += 1
        print(
            f"\033[93m{task_name}\033[0m immutable | "
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m, "
            f"seed: \033[90m{literal_seed}\033[0m, episode_id: {episode_id}"
        )

    return int(episodes[-1]["seed"]), TASK_ENV.suc, total_steps, episode_steps_list


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--eval_tag", type=str, default="official",
                       help="Evaluation tag for tracking experiments")
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["eval_tag"] = args.eval_tag

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
