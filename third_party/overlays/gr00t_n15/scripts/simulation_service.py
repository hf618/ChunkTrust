# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import numpy as np

from gr00t.eval.robot import RobotInferenceServer
from gr00t.eval.simulation import (
    MultiStepConfig,
    SimulationConfig,
    SimulationInferenceClient,
    StrideEvalConfig,
    VideoConfig,
)
from gr00t.model.policy import Gr00tPolicy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model checkpoint directory.",
        default="<PATH_TO_YOUR_MODEL>",
    )
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="The embodiment tag for the model.",
        default="<EMBODIMENT_TAG>",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        help="Name of the environment to run.",
        default="<ENV_NAME>",
    )
    parser.add_argument("--port", type=int, help="Port number for the server.", default=5555)
    parser.add_argument(
        "--host", type=str, help="Host address for the server.", default="localhost"
    )
    parser.add_argument("--video_dir", type=str, help="Directory to save videos.", default=None)
    parser.add_argument("--n_episodes", type=int, help="Number of episodes to run.", default=2)
    parser.add_argument("--n_envs", type=int, help="Number of parallel environments.", default=1)
    parser.add_argument(
        "--n_action_steps",
        type=int,
        help="Number of action steps per environment step.",
        default=16,
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        help="Maximum number of steps per episode.",
        default=1440,
    )
    parser.add_argument("--eval_type", type=str, default="default")
    parser.add_argument("--dump_denoise", type=int, default=0)
    parser.add_argument("--horizon_candidates", type=str, default="4,8,12,16")
    parser.add_argument("--horizon_exec_mode", type=str, default="expected_round")
    parser.add_argument("--horizon_expected_temp", type=float, default=1.0)
    parser.add_argument("--horizon_intra_alpha", type=float, default=4.0)
    parser.add_argument("--horizon_intra_cut_t", type=float, default=0.25)
    parser.add_argument("--horizon_ts_epsilon", type=float, default=0.05)
    parser.add_argument("--horizon_ts_seed", type=int, default=0)
    parser.add_argument("--horizon_ts_update_mode", type=str, default="kernel_forget")
    parser.add_argument("--horizon_ts_kernel_bandwidth", type=float, default=10.0)
    parser.add_argument("--horizon_ts_forget_rho", type=float, default=0.99)
    parser.add_argument("--horizon_ts_update_eta", type=float, default=1.0)
    parser.add_argument("--horizon_intra_z_mode", type=str, default="window_rms")
    parser.add_argument("--horizon_intra_tau_window", type=int, default=1)
    parser.add_argument("--horizon_mix_mode", type=str, default="both")
    parser.add_argument("--horizon_inter_use_speed_uniformity", type=int, default=1)
    parser.add_argument("--horizon_inter_window", type=int, default=20)
    parser.add_argument("--horizon_inter_weight", type=float, default=0.5)
    parser.add_argument("--horizon_inter_beta", type=float, default=4.0)
    parser.add_argument("--horizon_inter_fallback", type=str, default="ehh_only")
    parser.add_argument("--replan_log_path", type=str, default="")
    parser.add_argument("--run_summary_path", type=str, default="")
    # server mode
    parser.add_argument("--server", action="store_true", help="Run the server.")
    # client mode
    parser.add_argument("--client", action="store_true", help="Run the client")
    args = parser.parse_args()

    if args.server:
        # Create a policy
        policy = Gr00tPolicy(
            model_path=args.model_path,
            embodiment_tag=args.embodiment_tag,
        )

        # Start the server
        server = RobotInferenceServer(policy, port=args.port)
        server.run()

    elif args.client:
        eval_type = str(args.eval_type).strip().lower()
        supported_eval_types = {"default", "horizon"}
        if eval_type not in supported_eval_types:
            raise ValueError(
                f"Unsupported eval_type: {args.eval_type}. Supported values: {sorted(supported_eval_types)}"
            )

        # Create a simulation client
        simulation_client = SimulationInferenceClient(host=args.host, port=args.port)

        print("Available modality configs:")
        modality_config = simulation_client.get_modality_config()
        print(modality_config.keys())

        # Create simulation configuration
        config = SimulationConfig(
            env_name=args.env_name,
            n_episodes=args.n_episodes,
            n_envs=args.n_envs,
            video=VideoConfig(
                video_dir=args.video_dir,
                max_episode_steps=args.max_episode_steps,
                file_name_mode="episode_success",
            ),
            multistep=MultiStepConfig(
                n_action_steps=args.n_action_steps,
                max_episode_steps=args.max_episode_steps,
            ),
            eval=StrideEvalConfig(
                eval_type=eval_type,
                dump_denoise=bool(args.dump_denoise),
                horizon_candidates=args.horizon_candidates,
                horizon_exec_mode=str(args.horizon_exec_mode).strip().lower(),
                horizon_expected_temp=args.horizon_expected_temp,
                horizon_intra_alpha=args.horizon_intra_alpha,
                horizon_intra_cut_t=args.horizon_intra_cut_t,
                horizon_ts_epsilon=args.horizon_ts_epsilon,
                horizon_ts_seed=args.horizon_ts_seed,
                horizon_ts_update_mode=str(args.horizon_ts_update_mode).strip().lower(),
                horizon_ts_kernel_bandwidth=args.horizon_ts_kernel_bandwidth,
                horizon_ts_forget_rho=args.horizon_ts_forget_rho,
                horizon_ts_update_eta=args.horizon_ts_update_eta,
                horizon_intra_z_mode=str(args.horizon_intra_z_mode).strip().lower(),
                horizon_intra_tau_window=args.horizon_intra_tau_window,
                horizon_mix_mode=str(args.horizon_mix_mode).strip().lower(),
                horizon_inter_use_speed_uniformity=bool(args.horizon_inter_use_speed_uniformity),
                horizon_inter_window=args.horizon_inter_window,
                horizon_inter_weight=args.horizon_inter_weight,
                horizon_inter_beta=args.horizon_inter_beta,
                horizon_inter_fallback=str(args.horizon_inter_fallback).strip().lower(),
                replan_log_path=args.replan_log_path,
                run_summary_path=args.run_summary_path,
            ),
        )

        # Run the simulation
        print(f"Running simulation for {args.env_name}...")
        env_name, episode_successes = simulation_client.run_simulation(config)

        # Print results
        print(f"Results for {env_name}:")
        print(f"Success rate: {np.mean(episode_successes):.2f}")

    else:
        raise ValueError("Please specify either --server or --client")
