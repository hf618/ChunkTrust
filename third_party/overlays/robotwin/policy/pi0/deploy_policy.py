import numpy as np
import torch
import dill
import os, sys

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

from pi_model import *


# Encode observation for the model
def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]

    return input_rgb_arr, input_state


def get_model(usr_args):
    train_config_name, model_name, checkpoint_id, pi0_step = (usr_args["train_config_name"], usr_args["model_name"],
                                                              usr_args["checkpoint_id"], usr_args["pi0_step"])
    dump_denoise = usr_args.get("dump_denoise", 0)
    denoise_log_dir = usr_args.get("denoise_log_dir")
    valid_action_dim = None
    left_action_dim = usr_args.get("left_action_dim")
    right_action_dim = usr_args.get("right_action_dim")
    if ("left_action_dim" in usr_args) and ("right_action_dim" in usr_args):
        valid_action_dim = int(usr_args["left_action_dim"]) + int(usr_args["right_action_dim"])
    elif ("left_arm_dim" in usr_args) and ("right_arm_dim" in usr_args):
        valid_action_dim = int(usr_args["left_arm_dim"]) + int(usr_args["right_arm_dim"])
    eval_type = usr_args.get("eval_type", "default")
    return PI0(
        train_config_name,
        model_name,
        checkpoint_id,
        pi0_step,
        dump_denoise=dump_denoise,
        denoise_log_dir=denoise_log_dir,
        valid_action_dim=valid_action_dim,
        eval_type=eval_type,
        horizon_intra_cut_t=usr_args.get("horizon_intra_cut_t", 0.25),
        horizon_candidates=usr_args.get("horizon_candidates", "10,20,30,40,50"),
        horizon_intra_alpha=usr_args.get("horizon_intra_alpha", usr_args.get("horizon_ts_alpha", 4.0)),
        horizon_ts_epsilon=usr_args.get("horizon_ts_epsilon", 0.05),
        horizon_ts_seed=usr_args.get("horizon_ts_seed", usr_args.get("seed", 0)),
        horizon_ts_update_mode=usr_args.get("horizon_ts_update_mode", "kernel_forget"),
        horizon_ts_kernel_bandwidth=usr_args.get("horizon_ts_kernel_bandwidth", 10.0),
        horizon_ts_forget_rho=usr_args.get("horizon_ts_forget_rho", 0.99),
        horizon_ts_update_eta=usr_args.get("horizon_ts_update_eta", 1.0),
        horizon_intra_z_mode=usr_args.get("horizon_intra_z_mode", "window_rms"),
        horizon_exec_mode=usr_args.get("horizon_exec_mode", "expected_round"),
        horizon_expected_temp=usr_args.get("horizon_expected_temp", 1.0),
        horizon_intra_tau_window=usr_args.get("horizon_intra_tau_window", 3),
        horizon_inter_use_speed_uniformity=usr_args.get("horizon_inter_use_speed_uniformity", 1),
        horizon_mix_mode=usr_args.get("horizon_mix_mode", "both"),
        horizon_inter_window=usr_args.get("horizon_inter_window", 60),
        horizon_inter_weight=usr_args.get("horizon_inter_weight", 0.5),
        horizon_inter_beta=usr_args.get("horizon_inter_beta", 4.0),
        horizon_inter_fallback=usr_args.get("horizon_inter_fallback", "ehh_only"),
        fixed_exec_k=usr_args.get("fixed_exec_k", 0),
        denoise_npz_compress=usr_args.get("denoise_npz_compress", 0),
        left_action_dim=left_action_dim,
        right_action_dim=right_action_dim,
        checkpoint_root_tag=usr_args.get("checkpoint_root_tag"),
        use_qha=usr_args.get("use_qha"),
        qha_eval_variant=usr_args.get("qha_eval_variant", "native"),
        qha_candidate_mode_override=usr_args.get("qha_candidate_mode_override", "auto"),
        qha_hybrid_prior_gamma_override=usr_args.get("qha_hybrid_prior_gamma_override"),
        base_checkpoint_root_tag=usr_args.get("base_checkpoint_root_tag"),
        base_model_name=usr_args.get("base_model_name"),
        base_checkpoint_id=usr_args.get("base_checkpoint_id"),
        adaptive_selector_mode=usr_args.get("adaptive_selector_mode", "none"),
    )


def eval(TASK_ENV, model, observation):
    if hasattr(model, "prepare_episode_logging"):
        model.prepare_episode_logging(getattr(TASK_ENV, "eval_video_path", None), getattr(TASK_ENV, "test_num", None))
    if hasattr(model, "ensure_valid_action_dim"):
        try:
            observed_dim = observation["joint_action"]["vector"].shape[-1]
            model.ensure_valid_action_dim(observed_dim)
        except Exception:
            pass

    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)

    # ======== Get Action ========

    actions_full = model.get_action()[:model.pi0_step]
    exec_len = getattr(model, "last_exec_k", len(actions_full))
    exec_len = max(1, min(int(exec_len), int(len(actions_full))))
    actions = actions_full[:exec_len]
    env_step_start = int(getattr(TASK_ENV, "take_action_cnt", 0))

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)
    if hasattr(model, "record_executed_actions"):
        model.record_executed_actions(actions, env_step_start)

    # ============================


def reset_model(model):
    model.reset_obsrvationwindows()
