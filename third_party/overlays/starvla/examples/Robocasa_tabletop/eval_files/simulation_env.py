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
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import tyro
import dataclasses
import json
import logging
import os

# Required for robocasa environments
import robocasa  # noqa: F401
import robosuite  # noqa: F401
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401
from examples.Robocasa_tabletop.eval_files.base_config import ModalityConfig
from examples.Robocasa_tabletop.eval_files.base_config import BasePolicy
from examples.Robocasa_tabletop.eval_files.wrappers.multistep_wrapper import MultiStepWrapper
from examples.Robocasa_tabletop.eval_files.wrappers.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)

from examples.Robocasa_tabletop.eval_files.model2robocasa_interface import PolicyWarper


@dataclass
class VideoConfig:
    """Configuration for video recording settings."""

    video_dir: Optional[str] = None
    steps_per_render: int = 2 # 和10 是什么关系
    fps: int = 10 # BUG 数据集中应该是 20？
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 1


@dataclass
class MultiStepConfig:
    """Configuration for multi-step environment settings."""

    video_delta_indices: np.ndarray = field(default=np.array([0]))
    state_delta_indices: np.ndarray = field(default=np.array([0]))
    n_action_steps: int = 16
    max_episode_steps: int = 1440


@dataclass
class SimulationConfig:
    """Main configuration for simulation environment."""

    env_name: str
    n_episodes: int = 2
    n_envs: int = 1
    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)


class SimulationInferenceEnv:
    """Client for running simulations with a model."""

    def __init__(self, model: Optional[BasePolicy] = None):
        """Initialize the simulation client with a model."""
        self.model = model
        self.env = None

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Get action from the model based on observations."""
        # NOTE(YL)!
        # hot fix to change the video.ego_view_bg_crop_pad_res256_freq20 to video.ego_view
        if "video.ego_view_bg_crop_pad_res256_freq20" in observations: # BUG @JinhuiYE here only one viwes
            observations["video.ego_view"] = observations.pop(
                "video.ego_view_bg_crop_pad_res256_freq20"
            )
        return self.model.step(observations)

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """Get modality configuration from the model."""
        return self.model.get_modality_config()

    def _begin_model_episode(self, video_dir: Optional[str], env_idx: int = 0) -> None:
        if not hasattr(self.model, "begin_episode"):
            return
        try:
            self.model.begin_episode(video_dir=video_dir, env_idx=env_idx)
        except TypeError:
            self.model.begin_episode()

    def _finalize_model_episode(
        self,
        *,
        env_idx: int = 0,
        success: bool,
        success_step: Optional[int],
        episode_steps: int,
        run_time_seconds: float,
    ) -> None:
        if not hasattr(self.model, "finalize_episode_logging"):
            return
        self.model.finalize_episode_logging(
            env_idx=env_idx,
            success=success,
            success_step=success_step,
            episode_steps=episode_steps,
            run_time_seconds=run_time_seconds,
        )

    def setup_environment(self, config: SimulationConfig) -> gym.vector.VectorEnv:
        """Set up the simulation environment based on the provided configuration."""
        # Create environment functions for each parallel environment
        env_fns = [partial(_create_single_env, config=config, idx=i) for i in range(config.n_envs)]
        # Create vector environment (sync for single env, async for multiple)
        if config.n_envs == 1:
            return gym.vector.SyncVectorEnv(env_fns)
        else:
            return gym.vector.AsyncVectorEnv(
                env_fns,
                shared_memory=False,
                context="spawn",
            )

    def run_simulation(self, config: SimulationConfig, model: Optional[BasePolicy] = None) -> Tuple[str, List[bool]]:
        """Run the simulation for the specified number of episodes.
        
        Args:
            config: Configuration for the simulation
            model: The model to use for inference. If None, uses the model from __init__
        """
        # Use the provided model or fall back to the instance model
        if model is not None:
            self.model = model
        
        if self.model is None:
            raise ValueError("No model provided. Please provide a model either in __init__ or run_simulation")
        
        start_time = time.time()
        print(
            f"Running {config.n_episodes} episodes for {config.env_name} with {config.n_envs} environments"
        )
        # Set up the environment
        self.env = self.setup_environment(config)
        # Initialize tracking variables
        episode_lengths = []
        current_rewards = [0] * config.n_envs
        current_lengths = [0] * config.n_envs
        episode_start_times = [0.0] * config.n_envs
        first_success_steps: list[Optional[int]] = [None] * config.n_envs
        completed_episodes = 0
        current_successes = [False] * config.n_envs
        episode_successes = []
        # Initial environment reset
        obs, _ = self.env.reset()
        episode_start_times = [time.time()] * config.n_envs
        for env_idx in range(config.n_envs):
            env_video_dir = _video_dir_for_env(config.video.video_dir, env_idx, config.n_envs)
            self._begin_model_episode(video_dir=env_video_dir, env_idx=env_idx)
        # Main simulation loop
        stop_collection = False
        while completed_episodes < config.n_episodes:
            # Process observations and get actions from the model
            actions = self._get_actions_from_model(obs)
            # Step the environment
            next_obs, rewards, terminations, truncations, env_infos = self.env.step(actions)
            # Update episode tracking
            for env_idx in range(config.n_envs):
                current_lengths[env_idx] += 1
                step_success = bool(env_infos["success"][env_idx][0])
                current_successes[env_idx] |= step_success
                if step_success and first_success_steps[env_idx] is None:
                    first_success_steps[env_idx] = current_lengths[env_idx]
                current_rewards[env_idx] += rewards[env_idx]
                # If episode ended, store results
                if terminations[env_idx] or truncations[env_idx]:
                    episode_lengths.append(current_lengths[env_idx])
                    episode_successes.append(current_successes[env_idx])
                    episode_runtime = float(time.time() - episode_start_times[env_idx])
                    env_video_dir = _video_dir_for_env(config.video.video_dir, env_idx, config.n_envs)
                    self._finalize_model_episode(
                        env_idx=env_idx,
                        success=current_successes[env_idx],
                        success_step=first_success_steps[env_idx],
                        episode_steps=current_lengths[env_idx],
                        run_time_seconds=episode_runtime,
                    )
                    current_successes[env_idx] = False
                    completed_episodes += 1
                    if completed_episodes < config.n_episodes:
                        self._begin_model_episode(video_dir=env_video_dir, env_idx=env_idx)
                    else:
                        stop_collection = True
                    # Reset trackers for this environment
                    current_rewards[env_idx] = 0
                    current_lengths[env_idx] = 0
                    first_success_steps[env_idx] = None
                    episode_start_times[env_idx] = time.time()
                    if stop_collection:
                        break
            obs = next_obs
            if stop_collection:
                break
        # Clean up
        self.env.reset()
        self.env.close()
        self.env = None
        print(
            f"Collecting {config.n_episodes} episodes took {time.time() - start_time:.2f} seconds"
        )
        assert (
            len(episode_successes) == config.n_episodes
        ), f"Expected exactly {config.n_episodes} episodes, got {len(episode_successes)}"
        return config.env_name, episode_successes

    def _get_actions_from_model(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Process observations and get actions from the model."""
        # Get actions from the model
        action_dict = self.get_action(observations)
        # Extract actions from the response
        if "actions" in action_dict:
            actions = action_dict["actions"]
        else:
            actions = action_dict
        # Add batch dimension to actions
        return actions


def _create_single_env(config: SimulationConfig, idx: int) -> gym.Env:
    """Create a single environment with appropriate wrappers."""
    # Create base environment
    env = gym.make(config.env_name, enable_render=True)
    # Add video recording wrapper if needed (only for the first environment)
    if config.video.video_dir is not None:
        env_video_dir = _video_dir_for_env(config.video.video_dir, idx, config.n_envs)
        video_recorder = VideoRecorder.create_h264(
            fps=config.video.fps,
            codec=config.video.codec,
            input_pix_fmt=config.video.input_pix_fmt,
            crf=config.video.crf,
            thread_type=config.video.thread_type,
            thread_count=config.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=env_video_dir,
            steps_per_render=config.video.steps_per_render,
        )
    # Add multi-step wrapper
    env = MultiStepWrapper(
        env,
        video_delta_indices=config.multistep.video_delta_indices,
        state_delta_indices=config.multistep.state_delta_indices,
        n_action_steps=config.multistep.n_action_steps,
        max_episode_steps=config.multistep.max_episode_steps,
    )
    return env


def _video_dir_for_env(video_dir: Optional[str], env_idx: int, n_envs: int) -> Optional[Path]:
    if video_dir is None:
        return None
    video_dir_path = Path(video_dir)
    if int(n_envs) <= 1:
        return video_dir_path
    return video_dir_path / f"env{int(env_idx):02d}"


def run_evaluation(
    env_name: str,
    model: BasePolicy,
    video_dir: Optional[str] = None,
    n_episodes: int = 2,
    n_envs: int = 1,
    n_action_steps: int = 2,
    max_episode_steps: int = 100,
) -> Tuple[str, List[bool]]:
    """
    Simple entry point to run a simulation evaluation.
    Args:
        env_name: Name of the environment to run
        model: The model to use for inference
        video_dir: Directory to save videos (None for no videos)
        n_episodes: Number of episodes to run
        n_envs: Number of parallel environments
        n_action_steps: Number of action steps per environment step
        max_episode_steps: Maximum number of steps per episode
    Returns:
        Tuple of environment name and list of episode success flags
    """
    # Create configuration
    config = SimulationConfig(
        env_name=env_name,
        n_episodes=n_episodes,
        n_envs=n_envs,
        video=VideoConfig(video_dir=video_dir),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps, max_episode_steps=max_episode_steps
        ),
    )
    # Create client and run simulation
    client = SimulationInferenceEnv(model=model)
    results = client.run_simulation(config)
    # Print results
    print(f"Results for {env_name}:")
    print(f"Success rate: {np.mean(results[1]):.2f}")
    return results


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 5678
    resize_size = [224,224]

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    env_name: str = "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    n_episodes: int = 50  # Number of steps to wait for objects to stabilize i n sim
    n_envs: int = 1  # Number of rollouts per task
    max_episode_steps: int = 360 # 
    n_action_steps: int = 3

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/1029_qwenGR00T_fourier_gr1_unified_1000_PnPMilkToMicrowaveClose_gpus_woPretrain_wState/checkpoints/steps_20000_pytorch_model.pt.log/gr1_unified/logs/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = "results/Checkpoints/1029_qwenGR00T_fourier_gr1_unified_1000_PnPMilkToMicrowaveClose_gpus_woPretrain_wState/checkpoints/steps_20000_pytorch_model.pt"
    eval_type: str = "default"
    dump_denoise: bool = False
    horizon_candidates: str = "4,8,12,16"
    horizon_exec_mode: str = "expected_round"
    horizon_expected_temp: float = 1.0
    horizon_intra_alpha: float = 4.0
    horizon_intra_cut_t: float = 0.25
    horizon_ts_epsilon: float = 0.05
    horizon_ts_seed: int = 0
    horizon_ts_update_mode: str = "kernel_forget"
    horizon_ts_kernel_bandwidth: float = 10.0
    horizon_ts_forget_rho: float = 0.99
    horizon_ts_update_eta: float = 1.0
    horizon_intra_z_mode: str = "window_rms"
    horizon_intra_tau_window: int = 1
    horizon_mix_mode: str = "both"
    horizon_inter_use_speed_uniformity: bool = True
    horizon_inter_window: int = 20
    horizon_inter_weight: float = 0.5
    horizon_inter_beta: float = 4.0
    horizon_inter_fallback: str = "ehh_only"
    replan_log_path: Optional[str] = None
    run_summary_path: Optional[str] = None



def eval_gr1_unified(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")
    if os.getenv("DEBUG", False):
        start_debugpy_once()

    model = PolicyWarper(
        policy_ckpt_path=args.pretrained_path, # to get unnormalization stats
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
        n_action_steps=args.n_action_steps,
        eval_type=args.eval_type,
        dump_denoise=args.dump_denoise,
        horizon_candidates=args.horizon_candidates,
        horizon_exec_mode=args.horizon_exec_mode,
        horizon_expected_temp=args.horizon_expected_temp,
        horizon_intra_alpha=args.horizon_intra_alpha,
        horizon_intra_cut_t=args.horizon_intra_cut_t,
        horizon_ts_epsilon=args.horizon_ts_epsilon,
        horizon_ts_seed=args.horizon_ts_seed,
        horizon_ts_update_mode=args.horizon_ts_update_mode,
        horizon_ts_kernel_bandwidth=args.horizon_ts_kernel_bandwidth,
        horizon_ts_forget_rho=args.horizon_ts_forget_rho,
        horizon_ts_update_eta=args.horizon_ts_update_eta,
        horizon_intra_z_mode=args.horizon_intra_z_mode,
        horizon_intra_tau_window=args.horizon_intra_tau_window,
        horizon_mix_mode=args.horizon_mix_mode,
        horizon_inter_use_speed_uniformity=args.horizon_inter_use_speed_uniformity,
        horizon_inter_window=args.horizon_inter_window,
        horizon_inter_weight=args.horizon_inter_weight,
        horizon_inter_beta=args.horizon_inter_beta,
        horizon_inter_fallback=args.horizon_inter_fallback,
        replan_log_path=args.replan_log_path,
    )
    results = run_evaluation(
        env_name=args.env_name,
        model=model,
        video_dir=args.video_out_path,
        n_episodes=args.n_episodes,
        n_envs=args.n_envs,
        n_action_steps=args.n_action_steps,
        max_episode_steps=args.max_episode_steps,
    )
    if args.run_summary_path:
        summary_path = Path(args.run_summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = model.get_horizon_run_summary() if hasattr(model, "get_horizon_run_summary") else {}
        summary["env_name"] = args.env_name
        summary["n_episodes"] = int(args.n_episodes)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return results

def start_debugpy_once():
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True

if __name__ == "__main__":
    tyro.cli(eval_gr1_unified)
