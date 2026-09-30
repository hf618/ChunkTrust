from __future__ import annotations
from chunktrust.paths import resolve_legacy_path as _ct_path

import re
import shutil
import textwrap
from pathlib import Path


THIS_REPO = Path(__file__).resolve().parents[2]
TARGET_REPO = Path(_ct_path('CHUNKTRUST_WORKSPACE_ROOT', 'Isaac-GR00T-n1.5'))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def should_preserve_horizon_file(path: Path, existing: str) -> bool:
    if path.name not in {"simulation_service.py", "simulation.py", "batch_eval_single_gpu.sh"}:
        return False
    return "horizon_candidates" in existing or "HORIZON_CANDIDATES" in existing


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if should_preserve_horizon_file(path, existing):
            content = existing
    path.write_text(content, encoding="utf-8")


def replace_once(content: str, old: str, new: str, *, path: Path) -> str:
    if old not in content:
        raise RuntimeError(f"Expected snippet not found in {path}")
    return content.replace(old, new, 1)


def replace_span(
    content: str,
    start_marker: str,
    end_marker: str,
    replacement: str,
    *,
    path: Path,
) -> str:
    start = content.find(start_marker)
    if start < 0:
        raise RuntimeError(f"Start marker not found in {path}")
    end = content.find(end_marker, start)
    if end < 0:
        raise RuntimeError(f"End marker not found in {path}")
    return content[:start] + replacement + content[end:]


def rewrite_action_head() -> None:
    path = TARGET_REPO / "gr00t/model/action_head/flow_matching_action_head.py"
    content = read_text(path)
    new = textwrap.dedent(
        """
            @torch.no_grad()
            def get_action(
                self,
                backbone_output: BatchFeature,
                action_input: BatchFeature,
                return_trace: bool = False,
            ) -> BatchFeature:

                backbone_output = self.process_backbone_output(backbone_output)

                # Get vision and language embeddings.
                vl_embs = backbone_output.backbone_features
                embodiment_id = action_input.embodiment_id

                # Embed state.
                state_features = self.state_encoder(action_input.state, embodiment_id)

                # Set initial actions as the sampled noise.
                batch_size = vl_embs.shape[0]
                device = vl_embs.device
                actions = torch.randn(
                    size=(batch_size, self.config.action_horizon, self.config.action_dim),
                    dtype=vl_embs.dtype,
                    device=device,
                )

                num_steps = self.num_inference_timesteps
                dt = 1.0 / num_steps
                trace_v = []
                trace_ds = []

                # Run denoising steps.
                for t in range(num_steps):
                    t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
                    t_discretized = int(t_cont * self.num_timestep_buckets)

                    # Embed noised action trajectory.
                    timesteps_tensor = torch.full(
                        size=(batch_size,), fill_value=t_discretized, device=device
                    )
                    action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
                    # Maybe add position embedding.
                    if self.config.add_pos_embed:
                        pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                        pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                        action_features = action_features + pos_embs

                    # Join vision, language, state and action embedding along sequence dimension.
                    future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
                    sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

                    # Run model forward.
                    model_output = self.model(
                        hidden_states=sa_embs,
                        encoder_hidden_states=vl_embs,
                        timestep=timesteps_tensor,
                    )
                    pred = self.action_decoder(model_output, embodiment_id)

                    pred_velocity = pred[:, -self.action_horizon :]
                    if return_trace:
                        trace_v.append(pred_velocity.detach().clone())
                        trace_ds.append(t_discretized)

                    # Update actions using euler integration.
                    actions = actions + dt * pred_velocity

                result = {"action_pred": actions}
                if return_trace:
                    result["denoise_trace"] = {
                        "v": torch.stack(trace_v, dim=1),
                        "xt_internal": actions.detach().clone(),
                        "x0": actions.detach().clone(),
                        "ds": torch.tensor(trace_ds, dtype=torch.int32, device=device),
                    }
                return BatchFeature(data=result)
        """
    ).strip()
    start = content.find("@torch.no_grad()", content.find("return BatchFeature(data=output_dict)"))
    if start < 0:
        raise RuntimeError(f"Start marker not found in {path}")
    end = content.find("\n    @property\n    def device(self):", start)
    if end < 0:
        raise RuntimeError(f"End marker not found in {path}")
    content = content[:start] + textwrap.indent(new, "    ") + "\n" + content[end:]
    write_text(path, content)


def rewrite_gr00t_n1() -> None:
    path = TARGET_REPO / "gr00t/model/gr00t_n1.py"
    content = read_text(path)
    new = textwrap.dedent(
        """
            def get_action(
                self,
                inputs: dict,
                return_trace: bool = False,
            ) -> BatchFeature:
                backbone_inputs, action_inputs = self.prepare_input(inputs)
                # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
                backbone_outputs = self.backbone(backbone_inputs)
                action_head_outputs = self.action_head.get_action(
                    backbone_outputs,
                    action_inputs,
                    return_trace=return_trace,
                )
                self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
                return action_head_outputs
        """
    ).strip()
    cls_start = content.find("class GR00T_N1_5")
    start = content.find("def get_action(", cls_start)
    if start < 0:
        raise RuntimeError(f"Start marker not found in {path}")
    end = content.find("\n    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:", start)
    if end < 0:
        raise RuntimeError(f"End marker not found in {path}")
    content = content[:start] + textwrap.indent(new, "    ") + "\n" + content[end:]
    write_text(path, content)


def rewrite_policy() -> None:
    path = TARGET_REPO / "gr00t/model/policy.py"
    content = read_text(path)
    content = replace_once(
        content,
        "def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:",
        "def get_action(self, observations: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> Any:",
        path=path,
    )

    new_block = textwrap.dedent(
        """
            def get_action(
                self,
                observations: Dict[str, Any],
                options: Optional[Dict[str, Any]] = None,
            ) -> Any:
                \"""
                Make a prediction with the model.
                Args:
                    obs (Dict[str, Any]): The observation to make a prediction for.

                e.g. obs = {
                    "video.<>": np.ndarray,  # (T, H, W, C)
                    "state.<>": np.ndarray, # (T, D)
                    "annotation.<>": np.ndarray, # (T, )
                }

                or with batched input:
                e.g. obs = {
                    "video.<>": np.ndarray,, # (B, T, H, W, C)
                    "state.<>": np.ndarray, # (B, T, D)
                    "annotation.<>": np.ndarray, # (B, T, )
                }

                Returns:
                    Dict[str, Any]: The predicted action.
                \"""
                if (
                    options is None
                    and isinstance(observations, dict)
                    and "observation" in observations
                    and ("options" in observations or set(observations.keys()) <= {"observation", "options"})
                ):
                    options = observations.get("options")
                    observations = observations.get("observation", {})

                return_trace = bool((options or {}).get("return_trace", False))

                # Create a copy to avoid mutating input
                obs_copy = observations.copy()

                is_batch = self._check_state_is_batched(obs_copy)
                if not is_batch:
                    obs_copy = unsqueeze_dict_values(obs_copy)

                # Convert to numpy arrays
                for k, v in obs_copy.items():
                    if not isinstance(v, np.ndarray):
                        obs_copy[k] = np.array(v)

                normalized_input = self.apply_transforms(obs_copy)
                normalized_action, info = self._get_action_from_normalized_input(
                    normalized_input,
                    return_trace=return_trace,
                )
                unnormalized_action = self._get_unnormalized_action(normalized_action)

                if not is_batch:
                    unnormalized_action = squeeze_dict_values(unnormalized_action)

                if return_trace or options is not None:
                    return unnormalized_action, info
                return unnormalized_action

            def _get_action_from_normalized_input(
                self,
                normalized_input: Dict[str, Any],
                return_trace: bool = False,
            ) -> tuple[torch.Tensor, Dict[str, Any]]:
                # Set up autocast context if needed
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                    model_pred = self.model.get_action(normalized_input, return_trace=return_trace)

                normalized_action = model_pred["action_pred"].float()
                info = {}
                if return_trace and "denoise_trace" in model_pred:
                    denoise_trace = {}
                    for key, value in model_pred["denoise_trace"].items():
                        if isinstance(value, torch.Tensor):
                            tensor = value.detach()
                            if torch.is_floating_point(tensor):
                                tensor = tensor.float()
                            denoise_trace[key] = tensor.cpu().numpy()
                        else:
                            denoise_trace[key] = value
                    info["denoise_trace"] = denoise_trace
                return normalized_action, info
        """
    ).strip()
    cls_start = content.find("class Gr00tPolicy")
    start = content.find("def get_action(", cls_start)
    if start < 0:
        raise RuntimeError(f"Start marker not found in {path}")
    end = content.find("\n    def _get_unnormalized_action(self, normalized_action: torch.Tensor) -> Dict[str, Any]:", start)
    if end < 0:
        raise RuntimeError(f"End marker not found in {path}")
    content = content[:start] + textwrap.indent(new_block, "    ") + "\n" + content[end:]
    write_text(path, content + "\n")


def rewrite_multistep_wrapper() -> None:
    path = TARGET_REPO / "gr00t/eval/wrappers/multistep_wrapper.py"
    content = read_text(path)
    content = replace_once(
        content,
        "        self._action_space = repeated_space(env.action_space, n_action_steps)\n",
        (
            "        self._action_space = repeated_space(env.action_space, n_action_steps)\n"
            "        self._action_space[\"exec_horizon\"] = spaces.Box(\n"
            "            low=np.array([1], dtype=np.int32),\n"
            "            high=np.array([n_action_steps], dtype=np.int32),\n"
            "            shape=(1,),\n"
            "            dtype=np.int32,\n"
            "        )\n"
        ),
        path=path,
    )

    new = textwrap.dedent(
        """
            def step(self, action):
                \"""
                action: dict: key-value pairs where the values are of shape (n_action_steps,) + action_shape
                \"""
                exec_horizon = self.n_action_steps
                if "exec_horizon" in action:
                    exec_value = np.asarray(action["exec_horizon"]).reshape(-1)
                    if exec_value.size > 0:
                        exec_horizon = int(exec_value[0])
                exec_horizon = max(1, min(int(exec_horizon), int(self.n_action_steps)))

                states = []
                rewards = []
                dones = []
                truncated = False
                env_state = {"states": [], "model": []}
                for step in range(exec_horizon):
                    act = {}
                    for key, value in action.items():
                        if key == "exec_horizon":
                            continue
                        act[key] = value[step, :]
                    if len(self.done) > 0 and self.done[-1]:
                        # termination
                        break
                    observation, reward, done, truncated, info = super().step(act)
                    env_state = {"states": [], "model": []}
                    states.append(env_state["states"])
                    rewards.append(reward)
                    dones.append(done)
                    self.obs.append(observation)
                    self.reward.append(reward)
                    if (self.max_episode_steps is not None) and (
                        len(self.reward) >= self.max_episode_steps
                    ):
                        # truncation
                        done = True
                    self.done.append(done)
                    self._add_info(info)

                observation = self._get_obs(self.video_delta_indices, self.state_delta_indices)
                reward = aggregate(self.reward, self.reward_agg_method)
                done = aggregate(self.done, "max")
                info = dict_take_last_n(self.info, self.max_steps_needed)
                states = np.array(states)
                rewards = np.array(rewards)
                dones = np.array(dones)
                info["states"] = states
                info["rewards"] = rewards
                info["model"] = env_state["model"]
                info["actions"] = action
                info["dones"] = dones
                info["exec_horizon"] = np.array([exec_horizon], dtype=np.int32)
                return observation, reward, done, truncated, info
        """
    ).strip()
    start = content.find("def step(self, action):\n")
    if start < 0:
        raise RuntimeError(f"Start marker not found in {path}")
    end = content.find("\n    def _get_obs(self, video_delta_indices, state_delta_indices):", start)
    if end < 0:
        raise RuntimeError(f"End marker not found in {path}")
    content = content[:start] + textwrap.indent(new, "    ") + "\n" + content[end:]
    write_text(path, content + "\n")


def rewrite_simulation_service() -> None:
    path = TARGET_REPO / "scripts/simulation_service.py"
    content = textwrap.dedent(
        """
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
            parser.add_argument("--stride_candidates", type=str, default="2,4,6,8,10,12")
            parser.add_argument("--stride_exec_mode", type=str, default="expected_round")
            parser.add_argument("--stride_expected_temp", type=float, default=0.75)
            parser.add_argument("--stride_ts_alpha", type=float, default=80.0)
            parser.add_argument("--stride_ts_epsilon", type=float, default=0.05)
            parser.add_argument("--stride_ts_seed", type=int, default=0)
            parser.add_argument("--stride_ts_update_mode", type=str, default="legacy")
            parser.add_argument("--stride_ts_kernel_bandwidth", type=float, default=10.0)
            parser.add_argument("--stride_ts_forget_rho", type=float, default=1.0)
            parser.add_argument("--stride_ts_update_eta", type=float, default=1.0)
            parser.add_argument("--stride_z_mode", type=str, default="window_rms")
            parser.add_argument("--stride_tau_window", type=int, default=3)
            parser.add_argument("--stride_signal_mode", type=str, default="intra")
            parser.add_argument("--stride_use_speed_uniformity", type=int, default=1)
            parser.add_argument("--stride_speed_window", type=int, default=60)
            parser.add_argument("--stride_speed_weight", type=float, default=0.3)
            parser.add_argument("--stride_speed_beta", type=float, default=4.0)
            parser.add_argument("--stride_speed_fallback", type=str, default="ehh_only")
            parser.add_argument("--replan_log_path", type=str, default="")
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
                        eval_type=str(args.eval_type).strip().lower(),
                        dump_denoise=bool(args.dump_denoise),
                        stride_candidates=args.stride_candidates,
                        stride_exec_mode=str(args.stride_exec_mode).strip().lower(),
                        stride_expected_temp=args.stride_expected_temp,
                        stride_ts_alpha=args.stride_ts_alpha,
                        stride_ts_epsilon=args.stride_ts_epsilon,
                        stride_ts_seed=args.stride_ts_seed,
                        stride_ts_update_mode=str(args.stride_ts_update_mode).strip().lower(),
                        stride_ts_kernel_bandwidth=args.stride_ts_kernel_bandwidth,
                        stride_ts_forget_rho=args.stride_ts_forget_rho,
                        stride_ts_update_eta=args.stride_ts_update_eta,
                        stride_z_mode=str(args.stride_z_mode).strip().lower(),
                        stride_tau_window=args.stride_tau_window,
                        stride_signal_mode=str(args.stride_signal_mode).strip().lower(),
                        stride_use_speed_uniformity=bool(args.stride_use_speed_uniformity),
                        stride_speed_window=args.stride_speed_window,
                        stride_speed_weight=args.stride_speed_weight,
                        stride_speed_beta=args.stride_speed_beta,
                        stride_speed_fallback=str(args.stride_speed_fallback).strip().lower(),
                        replan_log_path=args.replan_log_path,
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
        """
    ).lstrip()
    write_text(path, content)


def rewrite_simulation() -> None:
    path = TARGET_REPO / "gr00t/eval/simulation.py"
    content = textwrap.dedent(
        """
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
        import json
        import time
        from dataclasses import dataclass, field
        from functools import partial
        from pathlib import Path
        from typing import Any, Dict, List, Optional, Tuple

        import gymnasium as gym
        import numpy as np
        from tqdm import tqdm

        # Required for robocasa environments
        import robocasa  # noqa: F401
        import robosuite  # noqa: F401
        from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401

        from gr00t.data.dataset import ModalityConfig
        from gr00t.eval.denoise_logger import DenoiseLogger
        from gr00t.eval.service import BaseInferenceClient
        from gr00t.eval.stride_selector import select_exec_k_stride
        from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
        from gr00t.eval.wrappers.video_recording_wrapper import (
            VideoRecorder,
            VideoRecordingWrapper,
        )
        from gr00t.model.policy import BasePolicy


        @dataclass
        class VideoConfig:
            \"""Configuration for video recording settings.\"""

            video_dir: Optional[str] = None
            steps_per_render: int = 2
            max_episode_steps: int = 720
            fps: int = 10
            codec: str = "h264"
            input_pix_fmt: str = "rgb24"
            crf: int = 22
            thread_type: str = "FRAME"
            thread_count: int = 1
            file_name_mode: str = "episode_success"


        @dataclass
        class MultiStepConfig:
            \"""Configuration for multi-step environment settings.\"""

            video_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
            state_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
            n_action_steps: int = 16
            max_episode_steps: int = 1440


        @dataclass
        class StrideEvalConfig:
            eval_type: str = "default"
            dump_denoise: bool = False
            stride_candidates: str = "2,4,6,8,10,12"
            stride_exec_mode: str = "expected_round"
            stride_expected_temp: float = 0.75
            stride_ts_alpha: float = 80.0
            stride_ts_epsilon: float = 0.05
            stride_ts_seed: int = 0
            stride_ts_update_mode: str = "legacy"
            stride_ts_kernel_bandwidth: float = 10.0
            stride_ts_forget_rho: float = 1.0
            stride_ts_update_eta: float = 1.0
            stride_z_mode: str = "window_rms"
            stride_tau_window: int = 3
            stride_signal_mode: str = "intra"
            stride_use_speed_uniformity: bool = True
            stride_speed_window: int = 60
            stride_speed_weight: float = 0.3
            stride_speed_beta: float = 4.0
            stride_speed_fallback: str = "ehh_only"
            replan_log_path: str = ""


        @dataclass
        class SimulationConfig:
            \"""Main configuration for simulation environment.\"""

            env_name: str
            n_episodes: int = 2
            n_envs: int = 1
            video: VideoConfig = field(default_factory=VideoConfig)
            multistep: MultiStepConfig = field(default_factory=MultiStepConfig)
            eval: StrideEvalConfig = field(default_factory=StrideEvalConfig)


        def _video_dir_for_env(video_dir: Path, env_idx: int, total_n_envs: int) -> Path:
            if total_n_envs <= 1:
                return video_dir
            return video_dir / f"env{env_idx:02d}"


        def _normalize_task_descriptions(
            observations: dict[str, Any],
            batch_size: int,
        ) -> list[str | None]:
            raw_task_description = None
            for key in (
                "annotation.human.action.task_description",
                "annotation.human.coarse_action",
                "task",
            ):
                if key in observations:
                    raw_task_description = observations[key]
                    break

            if raw_task_description is None:
                return [None] * batch_size

            if isinstance(raw_task_description, np.ndarray):
                raw_task_description = raw_task_description.tolist()

            if isinstance(raw_task_description, (list, tuple)):
                task_descriptions = list(raw_task_description)
            else:
                task_descriptions = [raw_task_description]

            normalized = []
            for item in task_descriptions:
                if isinstance(item, np.ndarray):
                    item = item.tolist()
                if isinstance(item, (list, tuple)) and len(item) == 1:
                    item = item[0]
                normalized.append(None if item is None else str(item))

            if not normalized:
                normalized = [None] * batch_size
            elif len(normalized) == 1 and batch_size > 1:
                normalized = normalized * batch_size
            elif len(normalized) != batch_size:
                raise RuntimeError(
                    "Task description batch size mismatch: "
                    f"got {len(normalized)} descriptions for batch size {batch_size}"
                )
            return normalized


        def _flatten_action_chunks(actions: dict[str, Any], action_modality_keys: list[str]) -> np.ndarray:
            flat_chunks = []
            horizon = None
            for key in action_modality_keys:
                parsed_key = key if key.startswith("action.") else f"action.{key}"
                if parsed_key not in actions:
                    raise KeyError(f"Missing action key {parsed_key} in policy output")
                arr = np.asarray(actions[parsed_key], dtype=np.float32)
                if arr.ndim != 3:
                    raise ValueError(f"Unexpected action chunk shape for {parsed_key}: {arr.shape}")
                if horizon is None:
                    horizon = arr.shape[1]
                else:
                    horizon = min(horizon, arr.shape[1])
                flat_chunks.append(arr)

            if not flat_chunks or horizon is None:
                raise ValueError("No action chunks found to flatten")

            return np.concatenate([arr[:, :horizon, :] for arr in flat_chunks], axis=-1)


        class EvalTraceManager:
            def __init__(
                self,
                *,
                stride_config: StrideEvalConfig,
                n_envs: int,
                n_action_steps: int,
            ):
                self.stride_config = stride_config
                self.n_envs = int(n_envs)
                self.n_action_steps = int(n_action_steps)
                self.enabled = bool(
                    self.stride_config.dump_denoise
                    or self.stride_config.eval_type == "stride"
                    or self.stride_config.replan_log_path
                )
                self.replan_log_path = (
                    Path(self.stride_config.replan_log_path)
                    if self.stride_config.replan_log_path
                    else None
                )
                if self.replan_log_path is not None:
                    self.replan_log_path.parent.mkdir(parents=True, exist_ok=True)
                self._denoise_loggers = {
                    env_idx: DenoiseLogger(enabled=self.stride_config.dump_denoise)
                    for env_idx in range(self.n_envs)
                }
                self._denoise_overhead_seconds = {env_idx: 0.0 for env_idx in range(self.n_envs)}
                self._episode_paths: dict[int, Path | None] = {
                    env_idx: None for env_idx in range(self.n_envs)
                }
                self._episode_indices = {env_idx: 0 for env_idx in range(self.n_envs)}
                self._replan_indices = {env_idx: 0 for env_idx in range(self.n_envs)}
                self.executed_action_history = [
                    np.zeros((0, 0), dtype=np.float32) for _ in range(self.n_envs)
                ]
                self.stride_ts_states = [dict() for _ in range(self.n_envs)]
                self.stride_ts_rngs = [
                    np.random.default_rng(int(self.stride_config.stride_ts_seed) + env_idx)
                    for env_idx in range(self.n_envs)
                ]

            @staticmethod
            def _json_safe(value: Any) -> Any:
                if isinstance(value, np.generic):
                    return value.item()
                if isinstance(value, np.ndarray):
                    return value.tolist()
                if isinstance(value, Path):
                    return str(value)
                if isinstance(value, tuple):
                    return [EvalTraceManager._json_safe(v) for v in value]
                if isinstance(value, list):
                    return [EvalTraceManager._json_safe(v) for v in value]
                if isinstance(value, dict):
                    return {str(k): EvalTraceManager._json_safe(v) for k, v in value.items()}
                return value

            @staticmethod
            def _float_list_or_nan(values) -> np.ndarray:
                return np.asarray(
                    [np.nan if value is None else float(value) for value in values],
                    dtype=np.float32,
                )

            def _stride_info_to_npz_extras(
                self, stride_info: dict[str, Any] | None
            ) -> dict[str, np.ndarray]:
                if not stride_info:
                    return {}

                return {
                    "stride_values": np.asarray(stride_info.get("strides", []), dtype=np.int32),
                    "stride_z_ehh_rms_from_tau0": np.asarray(
                        stride_info.get("z_ehh_rms_from_tau0", []),
                        dtype=np.float32,
                    ),
                    "stride_q_proxy": np.asarray(stride_info.get("q_proxy", []), dtype=np.float32),
                    "stride_q_mix": np.asarray(stride_info.get("q_mix", []), dtype=np.float32),
                    "stride_score": np.asarray(stride_info.get("score", []), dtype=np.float32),
                    "stride_speed_uniformity_cv": self._float_list_or_nan(
                        stride_info.get("speed_uniformity_cv", [])
                    ),
                    "stride_speed_uniformity_proxy": self._float_list_or_nan(
                        stride_info.get("speed_uniformity_proxy", [])
                    ),
                    "stride_score_prob": np.asarray(
                        stride_info.get("score_prob", []),
                        dtype=np.float32,
                    ),
                    "stride_expected_prob": np.asarray(
                        stride_info.get("expected_prob", []),
                        dtype=np.float32,
                    ),
                    "stride_theta_sample": np.asarray(
                        stride_info.get("theta_sample", []),
                        dtype=np.float32,
                    ),
                    "stride_posterior_alpha_before": np.asarray(
                        stride_info.get("posterior_alpha_before", []),
                        dtype=np.float32,
                    ),
                    "stride_posterior_beta_before": np.asarray(
                        stride_info.get("posterior_beta_before", []),
                        dtype=np.float32,
                    ),
                    "stride_posterior_alpha_after": np.asarray(
                        stride_info.get("posterior_alpha_after", []),
                        dtype=np.float32,
                    ),
                    "stride_posterior_beta_after": np.asarray(
                        stride_info.get("posterior_beta_after", []),
                        dtype=np.float32,
                    ),
                    "stride_expected_stride": np.asarray(
                        float(stride_info.get("expected_stride", np.nan)),
                        dtype=np.float32,
                    ),
                    "stride_chosen_stride": np.asarray(
                        int(stride_info.get("chosen_stride", 0)),
                        dtype=np.int32,
                    ),
                    "stride_update_center": np.asarray(
                        float(stride_info.get("update_center", np.nan)),
                        dtype=np.float32,
                    ),
                    "stride_update_feedback": np.asarray(
                        float(stride_info.get("update_feedback", np.nan)),
                        dtype=np.float32,
                    ),
                    "stride_ehh_curve_by_stride": np.asarray(
                        stride_info.get("ehh_curve_by_stride", []),
                        dtype=np.float32,
                    ),
                    "stride_selector_mode": np.asarray(
                        str(stride_info.get("selector_mode", "")),
                    ),
                    "stride_choose_mode": np.asarray(
                        str(stride_info.get("choose_mode", "")),
                    ),
                    "stride_info_json": np.asarray(
                        json.dumps(self._json_safe(stride_info), ensure_ascii=False)
                    ),
                }

            def _resolve_episode_path(self, video_dir: str | None, env_idx: int) -> Path | None:
                if video_dir:
                    video_dir_path = Path(video_dir)
                    if video_dir_path.parent.name.lower() == "videos":
                        episode_dir = video_dir_path.parent.parent / "episodes" / video_dir_path.name
                    elif video_dir_path.name.lower() == "videos":
                        episode_dir = video_dir_path.parent / "episodes"
                    else:
                        episode_dir = video_dir_path.parent / "episodes"
                elif self.replan_log_path is not None:
                    if self.n_envs > 1:
                        episode_dir = self.replan_log_path.parent / "episodes" / f"env{env_idx:02d}"
                    else:
                        episode_dir = self.replan_log_path.parent / "episodes"
                else:
                    return None

                episode_dir.mkdir(parents=True, exist_ok=True)
                episode_idx = int(self._episode_indices[env_idx])
                return episode_dir / f"episode{episode_idx}.npz"

            def begin_episode(self, env_idx: int, video_dir: str | None = None) -> None:
                if not self.enabled:
                    return
                env_idx = int(env_idx)
                self._episode_indices[env_idx] += 1
                self._replan_indices[env_idx] = 0
                self._denoise_overhead_seconds[env_idx] = 0.0
                self.executed_action_history[env_idx] = np.zeros((0, 0), dtype=np.float32)
                self.stride_ts_states[env_idx] = {}
                self.stride_ts_rngs[env_idx] = np.random.default_rng(
                    int(self.stride_config.stride_ts_seed) + env_idx
                )
                logger = self._denoise_loggers[env_idx]
                logger.reset_episode()
                episode_path = self._resolve_episode_path(video_dir=video_dir, env_idx=env_idx)
                self._episode_paths[env_idx] = episode_path
                if episode_path is not None:
                    logger.start_episode(episode_path)

            def ensure_action_history(self, action_dim: int) -> None:
                for env_idx in range(self.n_envs):
                    history = self.executed_action_history[env_idx]
                    if history.shape[1] != int(action_dim):
                        self.executed_action_history[env_idx] = np.zeros(
                            (0, int(action_dim)),
                            dtype=np.float32,
                        )

            def append_executed_actions(
                self, flat_actions: np.ndarray, exec_horizon: np.ndarray
            ) -> None:
                if not self.enabled:
                    return
                for env_idx, k_exec in enumerate(np.asarray(exec_horizon).reshape(-1).tolist()):
                    executed = np.asarray(flat_actions[env_idx, : int(k_exec), :], dtype=np.float32)
                    if executed.size <= 0:
                        continue
                    history = self.executed_action_history[env_idx]
                    if history.size == 0 or history.shape[1] != executed.shape[1]:
                        self.executed_action_history[env_idx] = executed.copy()
                    else:
                        self.executed_action_history[env_idx] = np.concatenate(
                            [history, executed],
                            axis=0,
                        )

            def log_replan(
                self,
                *,
                task_descriptions: list[str | None],
                exec_horizon: np.ndarray,
                stride_infos: list[dict[str, Any] | None] | None,
            ) -> None:
                if self.replan_log_path is None:
                    return

                timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
                records = []
                for env_idx, k_exec in enumerate(np.asarray(exec_horizon).reshape(-1)):
                    stride_info = None
                    if stride_infos is not None and env_idx < len(stride_infos):
                        stride_info = stride_infos[env_idx]
                    records.append(
                        {
                            "timestamp": timestamp,
                            "episode_index": int(self._episode_indices[env_idx]),
                            "replan_index": int(self._replan_indices[env_idx]),
                            "batch_index": int(env_idx),
                            "task_description": self._json_safe(task_descriptions[env_idx]),
                            "eval_type": self.stride_config.eval_type,
                            "exec_horizon": int(k_exec),
                            "stride_info": self._json_safe(stride_info),
                        }
                    )

                with self.replan_log_path.open("a", encoding="utf-8") as fp:
                    for record in records:
                        fp.write(json.dumps(record, ensure_ascii=False) + "\\n")

            def log_denoise_step(
                self,
                *,
                denoise_trace: dict[str, Any] | None,
                flat_actions: np.ndarray,
                exec_horizon: np.ndarray,
                stride_infos: list[dict[str, Any] | None] | None,
            ) -> None:
                if not self.stride_config.dump_denoise or denoise_trace is None:
                    return

                trace_v = np.asarray(denoise_trace["v"], dtype=np.float32)
                used_h = min(int(self.n_action_steps), int(flat_actions.shape[1]))
                trace_ds = denoise_trace.get("ds")
                trace_x0 = denoise_trace.get("x0")
                trace_xt_internal = denoise_trace.get("xt_internal")

                for env_idx in range(flat_actions.shape[0]):
                    episode_path = self._episode_paths.get(int(env_idx))
                    if episode_path is None:
                        continue

                    k_exec = int(np.asarray(exec_horizon).reshape(-1)[env_idx])
                    xt_log = np.array(flat_actions[env_idx, :used_h, :], dtype=np.float32, copy=True)
                    if k_exec < xt_log.shape[0]:
                        xt_log[k_exec:, :] = 0.0

                    xt_internal_log = None
                    if trace_xt_internal is not None:
                        xt_internal_log = np.asarray(
                            trace_xt_internal[env_idx, :used_h, :],
                            dtype=np.float32,
                        ).copy()
                        if k_exec < xt_internal_log.shape[0]:
                            xt_internal_log[k_exec:, :] = 0.0

                    x0 = None
                    if trace_x0 is not None:
                        x0 = np.asarray(trace_x0[env_idx, :used_h, :], dtype=np.float32)

                    ds = None
                    if trace_ds is not None:
                        ds_arr = np.asarray(trace_ds)
                        if ds_arr.ndim == 2:
                            ds = np.asarray(ds_arr[env_idx], dtype=np.float32)
                        else:
                            ds = np.asarray(ds_arr, dtype=np.float32)

                    extras = {}
                    if self.stride_config.eval_type == "stride" and stride_infos is not None:
                        extras = self._stride_info_to_npz_extras(stride_infos[env_idx])

                    t0 = time.time()
                    self._denoise_loggers[env_idx].log_step(
                        np.asarray(trace_v[env_idx, :, :used_h, :], dtype=np.float32),
                        ds=ds,
                        x0=x0,
                        xt=xt_log,
                        xt_internal=xt_internal_log,
                        k_exec=k_exec if self.stride_config.eval_type == "stride" else None,
                        extras=extras,
                    )
                    self._denoise_overhead_seconds[env_idx] += float(time.time() - t0)

            def finalize_episode(
                self,
                *,
                env_idx: int,
                success: bool,
                success_step: int | None,
                episode_steps: int,
                run_time_seconds: float,
            ) -> None:
                if not self.enabled:
                    return
                logger = self._denoise_loggers[int(env_idx)]
                logger.set_episode_info(
                    success=success,
                    success_step=success_step,
                    episode_steps=episode_steps,
                    run_time_seconds=run_time_seconds,
                    denoise_overhead_seconds=self._denoise_overhead_seconds[int(env_idx)],
                )
                logger.flush_episode()

            def advance_replan_indices(self) -> None:
                if not self.enabled:
                    return
                for env_idx in range(self.n_envs):
                    self._replan_indices[env_idx] += 1


        def _success_from_value(value: Any) -> bool:
            if value is None:
                return False
            if isinstance(value, np.ndarray):
                return bool(np.any(value))
            if isinstance(value, list):
                return bool(np.any(np.asarray(value)))
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            if isinstance(value, (int, np.integer)):
                return bool(value)
            return bool(value)


        class SimulationInferenceClient(BaseInferenceClient, BasePolicy):
            \"""Client for running simulations and communicating with the inference server.\"""

            def __init__(self, host: str = "localhost", port: int = 5555):
                \"""Initialize the simulation client with server connection details.\"""
                super().__init__(host=host, port=port)
                self.env = None

            def get_action(
                self,
                observations: Dict[str, Any],
                options: Dict[str, Any] | None = None,
            ) -> Any:
                \"""Get action from the inference server based on observations.\"""
                obs_copy = observations.copy()
                if "video.ego_view_bg_crop_pad_res256_freq20" in obs_copy:
                    obs_copy["video.ego_view"] = obs_copy.pop(
                        "video.ego_view_bg_crop_pad_res256_freq20"
                    )
                payload = obs_copy if options is None else {"observation": obs_copy, "options": options}
                return self.call_endpoint("get_action", payload)

            def get_modality_config(self) -> Dict[str, ModalityConfig]:
                \"""Get modality configuration from the inference server.\"""
                return self.call_endpoint("get_modality_config", requires_input=False)

            def setup_environment(self, config: SimulationConfig) -> gym.vector.VectorEnv:
                \"""Set up the simulation environment based on the provided configuration.\"""
                env_fns = [
                    partial(_create_single_env, config=config, idx=i) for i in range(config.n_envs)
                ]
                if config.n_envs == 1:
                    return gym.vector.SyncVectorEnv(env_fns)
                return gym.vector.AsyncVectorEnv(
                    env_fns,
                    shared_memory=False,
                    context="spawn",
                )

            def run_simulation(self, config: SimulationConfig) -> Tuple[str, List[bool]]:
                \"""Run the simulation for the specified number of episodes.\"""
                start_time = time.time()
                print(
                    f"Running {config.n_episodes} episodes for {config.env_name} with {config.n_envs} environments"
                )

                self.env = self.setup_environment(config)
                action_modality_keys = self.get_modality_config()["action"].modality_keys
                trace_manager = EvalTraceManager(
                    stride_config=config.eval,
                    n_envs=config.n_envs,
                    n_action_steps=config.multistep.n_action_steps,
                )

                current_rewards = [0] * config.n_envs
                current_lengths = [0] * config.n_envs
                current_successes = [False] * config.n_envs
                first_success_steps: list[int | None] = [None] * config.n_envs
                episode_start_times = [time.time()] * config.n_envs
                episode_successes = []
                completed_episodes = 0

                obs, _ = self.env.reset()
                for env_idx in range(config.n_envs):
                    env_video_dir = None
                    if config.video.video_dir is not None:
                        env_video_dir = str(
                            _video_dir_for_env(Path(config.video.video_dir), env_idx, config.n_envs)
                        )
                    trace_manager.begin_episode(env_idx=env_idx, video_dir=env_video_dir)

                pbar = tqdm(total=config.n_episodes, desc="Episodes")
                stop_collection = False
                while completed_episodes < config.n_episodes:
                    batch_size = int(
                        len(
                            next(
                                value
                                for key, value in obs.items()
                                if key.startswith("video.") or key.startswith("state.")
                            )
                        )
                    )
                    task_descriptions = _normalize_task_descriptions(obs, batch_size)
                    policy_options = None
                    if config.eval.eval_type == "stride" or config.eval.dump_denoise:
                        policy_options = {"return_trace": True}

                    actions, policy_info = self._get_actions_from_server(obs, options=policy_options)
                    exec_horizon = np.full(
                        (batch_size,),
                        config.multistep.n_action_steps,
                        dtype=np.int32,
                    )
                    stride_infos: list[dict[str, Any] | None] | None = None
                    flat_actions = None
                    if (
                        config.eval.eval_type == "stride"
                        or config.eval.dump_denoise
                        or trace_manager.replan_log_path is not None
                    ):
                        flat_actions = _flatten_action_chunks(actions, action_modality_keys)
                        trace_manager.ensure_action_history(flat_actions.shape[-1])
                        if config.eval.eval_type == "stride":
                            denoise_trace = policy_info.get("denoise_trace")
                            if denoise_trace is None:
                                raise RuntimeError(
                                    "eval_type='stride' requires denoise_trace from the policy output."
                                )
                            v_trace = np.asarray(denoise_trace["v"], dtype=np.float32)
                            if v_trace.ndim != 4:
                                raise RuntimeError(
                                    f"Unexpected denoise_trace['v'] shape: {v_trace.shape}"
                                )
                            stride_infos = []
                            used_h = min(
                                int(config.multistep.n_action_steps),
                                int(flat_actions.shape[1]),
                            )
                            for env_idx in range(batch_size):
                                k_exec, stride_info = select_exec_k_stride(
                                    v_trace[env_idx, :, :used_h, :],
                                    used_h,
                                    stride_candidates=config.eval.stride_candidates,
                                    exec_mode=config.eval.stride_exec_mode,
                                    expected_temp=config.eval.stride_expected_temp,
                                    stride_alpha=config.eval.stride_ts_alpha,
                                    ts_epsilon=config.eval.stride_ts_epsilon,
                                    ts_update_mode=config.eval.stride_ts_update_mode,
                                    ts_kernel_bandwidth=config.eval.stride_ts_kernel_bandwidth,
                                    ts_forget_rho=config.eval.stride_ts_forget_rho,
                                    ts_update_eta=config.eval.stride_ts_update_eta,
                                    z_mode=config.eval.stride_z_mode,
                                    tau_window=config.eval.stride_tau_window,
                                    xt_step=np.asarray(
                                        flat_actions[env_idx, :used_h, :],
                                        dtype=np.float32,
                                    ),
                                    history_actions=trace_manager.executed_action_history[env_idx],
                                    signal_mode=config.eval.stride_signal_mode,
                                    use_speed_uniformity=config.eval.stride_use_speed_uniformity,
                                    speed_window=config.eval.stride_speed_window,
                                    speed_weight=config.eval.stride_speed_weight,
                                    speed_beta=config.eval.stride_speed_beta,
                                    speed_fallback=config.eval.stride_speed_fallback,
                                    ts_state=trace_manager.stride_ts_states[env_idx],
                                    ts_rng=trace_manager.stride_ts_rngs[env_idx],
                                )
                                exec_horizon[env_idx] = int(k_exec)
                                stride_infos.append(stride_info)
                            trace_manager.append_executed_actions(flat_actions, exec_horizon)

                        actions["exec_horizon"] = exec_horizon[:, None]
                        trace_manager.log_replan(
                            task_descriptions=task_descriptions,
                            exec_horizon=exec_horizon,
                            stride_infos=stride_infos,
                        )
                        trace_manager.log_denoise_step(
                            denoise_trace=policy_info.get("denoise_trace"),
                            flat_actions=flat_actions,
                            exec_horizon=exec_horizon,
                            stride_infos=stride_infos,
                        )
                        trace_manager.advance_replan_indices()

                    next_obs, rewards, terminations, truncations, env_infos = self.env.step(actions)
                    for env_idx in range(config.n_envs):
                        if "success" in env_infos:
                            current_successes[env_idx] |= _success_from_value(
                                env_infos["success"][env_idx]
                            )
                        if (
                            "final_info" in env_infos
                            and env_infos["final_info"][env_idx] is not None
                            and "success" in env_infos["final_info"][env_idx]
                        ):
                            current_successes[env_idx] |= _success_from_value(
                                env_infos["final_info"][env_idx]["success"]
                            )
                        if current_successes[env_idx] and first_success_steps[env_idx] is None:
                            first_success_steps[env_idx] = current_lengths[env_idx] + 1

                        current_rewards[env_idx] += rewards[env_idx]
                        current_lengths[env_idx] += 1

                        if terminations[env_idx] or truncations[env_idx]:
                            episode_successes.append(bool(current_successes[env_idx]))
                            trace_manager.finalize_episode(
                                env_idx=env_idx,
                                success=bool(current_successes[env_idx]),
                                success_step=first_success_steps[env_idx],
                                episode_steps=current_lengths[env_idx],
                                run_time_seconds=float(time.time() - episode_start_times[env_idx]),
                            )
                            current_successes[env_idx] = False
                            first_success_steps[env_idx] = None
                            current_rewards[env_idx] = 0
                            current_lengths[env_idx] = 0
                            episode_start_times[env_idx] = time.time()
                            completed_episodes += 1
                            pbar.update(1)

                            if completed_episodes >= config.n_episodes:
                                stop_collection = True
                                break

                            env_video_dir = None
                            if config.video.video_dir is not None:
                                env_video_dir = str(
                                    _video_dir_for_env(
                                        Path(config.video.video_dir),
                                        env_idx,
                                        config.n_envs,
                                    )
                                )
                            trace_manager.begin_episode(env_idx=env_idx, video_dir=env_video_dir)

                    if stop_collection:
                        break
                    obs = next_obs

                pbar.close()
                self.env.reset()
                self.env.close()
                self.env = None
                print(
                    f"Collecting {config.n_episodes} episodes took {time.time() - start_time:.2f} seconds"
                )
                assert len(episode_successes) == config.n_episodes, (
                    f"Expected exactly {config.n_episodes} episodes, got {len(episode_successes)}"
                )
                return config.env_name, episode_successes

            def _get_actions_from_server(
                self,
                observations: Dict[str, Any],
                options: Dict[str, Any] | None = None,
            ) -> tuple[Dict[str, Any], Dict[str, Any]]:
                \"""Process observations and get actions from the inference server.\"""
                response = self.get_action(observations, options=options)
                info: Dict[str, Any] = {}
                action_dict: Dict[str, Any]

                if isinstance(response, (list, tuple)) and len(response) == 2:
                    action_dict = response[0]
                    info = response[1] if isinstance(response[1], dict) else {}
                else:
                    action_dict = response

                if isinstance(action_dict, dict) and "actions" in action_dict:
                    info = action_dict.get("info", info if isinstance(info, dict) else {})
                    actions = action_dict["actions"]
                else:
                    actions = action_dict
                return actions, info if isinstance(info, dict) else {}


        def _create_single_env(config: SimulationConfig, idx: int) -> gym.Env:
            \"""Create a single environment with appropriate wrappers.\"""
            env = gym.make(config.env_name, enable_render=True)
            if config.video.video_dir is not None:
                base_video_dir = Path(config.video.video_dir)
                per_env_video_dir = _video_dir_for_env(base_video_dir, idx, config.n_envs)
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
                    video_dir=per_env_video_dir,
                    steps_per_render=config.video.steps_per_render,
                    max_episode_steps=config.video.max_episode_steps,
                    file_name_mode=config.video.file_name_mode,
                )
            env = MultiStepWrapper(
                env,
                video_delta_indices=config.multistep.video_delta_indices,
                state_delta_indices=config.multistep.state_delta_indices,
                n_action_steps=config.multistep.n_action_steps,
                max_episode_steps=config.multistep.max_episode_steps,
            )
            return env


        def run_evaluation(
            env_name: str,
            host: str = "localhost",
            port: int = 5555,
            video_dir: Optional[str] = None,
            n_episodes: int = 2,
            n_envs: int = 1,
            n_action_steps: int = 2,
            max_episode_steps: int = 100,
            eval_config: StrideEvalConfig | None = None,
        ) -> Tuple[str, List[bool]]:
            \"""
            Simple entry point to run a simulation evaluation.
            \"""
            config = SimulationConfig(
                env_name=env_name,
                n_episodes=n_episodes,
                n_envs=n_envs,
                video=VideoConfig(
                    video_dir=video_dir,
                    max_episode_steps=max_episode_steps,
                    file_name_mode="episode_success",
                ),
                multistep=MultiStepConfig(
                    n_action_steps=n_action_steps, max_episode_steps=max_episode_steps
                ),
                eval=eval_config or StrideEvalConfig(),
            )
            client = SimulationInferenceClient(host=host, port=port)
            results = client.run_simulation(config)
            print(f"Results for {env_name}:")
            print(f"Success rate: {np.mean(results[1]):.2f}")
            return results


        if __name__ == "__main__":
            run_evaluation(
                env_name="robocasa_gr1_arms_only_fourier_hands/TwoArmPnPCarPartBrakepedal_GR1ArmsOnlyFourierHands_Env",
                host="localhost",
                port=5555,
                video_dir="./videos",
            )
        """
    ).lstrip()
    write_text(path, content)


def rewrite_batch_script() -> None:
    path = TARGET_REPO / "examples/RoboCasa/batch_eval_single_gpu.sh"
    content = textwrap.dedent(
        """
        #!/usr/bin/env bash

        set -euo pipefail

        # Single-GPU serial evaluation for the 24 RoboCasa GR1 tabletop tasks (N1.5 branch).
        # Output layout:
        #   /home/hfd24/Fanding/United/robocasa-gr1-tabletop-tasks/eval_results/<task_name>/<model_name>/<ckpt_tag>/<eval_tag>/<timestamp>/
        # Summary layout:
        #   /home/hfd24/Fanding/United/robocasa-gr1-tabletop-tasks/eval_results/_summary/<model_name>/<ckpt_tag>/<eval_tag>/

        ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
        cd "${ROOT_DIR}"

        PYTHON_BIN="${PYTHON_BIN:-python}"
        SIM_PYTHON="${SIM_PYTHON:-${PYTHON_BIN}}"
        MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/playground/Pretrained_models/gr00t-n1.5-robocasa-tabletop-posttrain}"
        EVAL_TAG="${EVAL_TAG:-official_eval_50x720x16}"
        CKPT_TAG="${CKPT_TAG:-$(basename "${MODEL_PATH}")}"
        DATA_CONFIG="${DATA_CONFIG:-fourier_gr1_arms_waist}"
        PORT="${PORT:-5555}"
        GPU_ID="${GPU_ID:-0}"
        HOST="${HOST:-127.0.0.1}"
        N_EPISODES="${N_EPISODES:-50}"
        N_ENVS="${N_ENVS:-1}"
        MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
        N_ACTION_STEPS="${N_ACTION_STEPS:-16}"
        EVAL_TYPE="${EVAL_TYPE:-default}"
        DUMP_DENOISE="${DUMP_DENOISE:-0}"
        STRIDE_CANDIDATES="${STRIDE_CANDIDATES:-2,4,6,8,10,12}"
        STRIDE_EXEC_MODE="${STRIDE_EXEC_MODE:-expected_round}"
        STRIDE_EXPECTED_TEMP="${STRIDE_EXPECTED_TEMP:-0.75}"
        STRIDE_TS_ALPHA="${STRIDE_TS_ALPHA:-80.0}"
        STRIDE_TS_EPSILON="${STRIDE_TS_EPSILON:-0.05}"
        STRIDE_TS_SEED="${STRIDE_TS_SEED:-0}"
        STRIDE_TS_UPDATE_MODE="${STRIDE_TS_UPDATE_MODE:-legacy}"
        STRIDE_TS_KERNEL_BANDWIDTH="${STRIDE_TS_KERNEL_BANDWIDTH:-10.0}"
        STRIDE_TS_FORGET_RHO="${STRIDE_TS_FORGET_RHO:-1.0}"
        STRIDE_TS_UPDATE_ETA="${STRIDE_TS_UPDATE_ETA:-1.0}"
        STRIDE_Z_MODE="${STRIDE_Z_MODE:-window_rms}"
        STRIDE_TAU_WINDOW="${STRIDE_TAU_WINDOW:-3}"
        STRIDE_SIGNAL_MODE="${STRIDE_SIGNAL_MODE:-intra}"
        STRIDE_USE_SPEED_UNIFORMITY="${STRIDE_USE_SPEED_UNIFORMITY:-1}"
        STRIDE_SPEED_WINDOW="${STRIDE_SPEED_WINDOW:-60}"
        STRIDE_SPEED_WEIGHT="${STRIDE_SPEED_WEIGHT:-0.3}"
        STRIDE_SPEED_BETA="${STRIDE_SPEED_BETA:-4.0}"
        STRIDE_SPEED_FALLBACK="${STRIDE_SPEED_FALLBACK:-ehh_only}"
        RESULT_ROOT="${RESULT_ROOT:-/home/hfd24/Fanding/United/robocasa-gr1-tabletop-tasks/eval_results}"
        SERVER_READY_TIMEOUT_SEC="${SERVER_READY_TIMEOUT_SEC:-600}"

        infer_model_name() {
          local model_path="$1"
          python - "$model_path" <<'PY'
        import json
        import sys
        from pathlib import Path

        model_path = Path(sys.argv[1])
        base = model_path.name

        def normalize_from_text(text: str):
            t = text.lower()
            if "n1.6" in t or "n1_6" in t or "n1d6" in t:
                return "GR00T-N1.6-3B"
            if "n1.5" in t or "n1_5" in t:
                return "GR00T-N1.5-3B"
            return None

        for candidate in [base, str(model_path)]:
            guess = normalize_from_text(candidate)
            if guess is not None:
                print(guess)
                raise SystemExit(0)

        config_path = model_path / "config.json"
        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
                model_type = str(cfg.get("model_type", "")).lower()
                if model_type == "gr00t_n1_5":
                    print("GR00T-N1.5-3B")
                    raise SystemExit(0)
                if model_type in {"gr00t_n1d6", "gr00t_n1_6"}:
                    print("GR00T-N1.6-3B")
                    raise SystemExit(0)
            except Exception:
                pass

        print(base)
        PY
        }

        MODEL_NAME="${MODEL_NAME:-$(infer_model_name "${MODEL_PATH}")}"

        SUMMARY_DIR="${RESULT_ROOT}/_summary/${MODEL_NAME}/${CKPT_TAG}/${EVAL_TAG}"
        SUMMARY_CSV="${SUMMARY_DIR}/summary.csv"
        SERVER_LOG="${SUMMARY_DIR}/server.log"

        is_truthy() {
          local value="${1:-}"
          case "${value,,}" in
            1|true|yes|on) return 0 ;;
            0|false|no|off|"") return 1 ;;
            *)
              echo "Invalid boolean value: ${value}" >&2
              return 1
              ;;
          esac
        }

        ENV_NAMES=(
          gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env
          gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
          gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
          gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env
          gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env
          gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env
          gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
          gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
          gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env
          gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env
          gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
          gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
          gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
          gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
          gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
          gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
          gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
          gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env
          gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
          gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
          gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
          gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env
          gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
          gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
        )

        server_pid=""

        cleanup() {
          if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
            kill "${server_pid}" 2>/dev/null || true
            wait "${server_pid}" 2>/dev/null || true
          fi
        }

        trap cleanup EXIT

        task_name_from_env() {
          local env_name="$1"
          local task_name
          task_name="$(basename "${env_name}")"
          task_name="${task_name%_GR1ArmsAndWaistFourierHands_Env}"
          task_name="$(echo "${task_name}" | sed \
            -e 's/^PnP/pnp_/' \
            -e 's/^PosttrainPnPNovel/posttrain_pnp_novel_/' \
            -e 's/To/_to_/g' \
            -e 's/From/from_/g' \
            -e 's/SplitA//g' \
            -e 's/\\([a-z0-9]\\)\\([A-Z]\\)/\\1_\\2/g' \
            | tr '[:upper:]' '[:lower:]' \
            | sed 's/__/_/g; s/_$//')"
          echo "${task_name}"
        }

        require_file() {
          local path="$1"
          local hint="$2"
          if [[ ! -e "${path}" ]]; then
            echo "Missing required path: ${path}" >&2
            echo "${hint}" >&2
            exit 1
          fi
        }

        wait_for_server() {
          local retries=$((SERVER_READY_TIMEOUT_SEC / 2))
          local attempt
          for attempt in $(seq 1 "${retries}"); do
            if [[ -n "${server_pid}" ]] && ! kill -0 "${server_pid}" 2>/dev/null; then
              echo "Server exited early. Check ${SERVER_LOG}" >&2
              return 1
            fi
            if grep -q "Server is ready and listening on" "${SERVER_LOG}" 2>/dev/null; then
              return 0
            fi
            sleep 2
          done
          echo "Server did not become ready within ${SERVER_READY_TIMEOUT_SEC}s. Check ${SERVER_LOG}" >&2
          return 1
        }

        extract_success_rate() {
          local log_path="$1"
          python - "$log_path" <<'PY'
        import re
        import sys
        from pathlib import Path

        text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
        matches = re.findall(r"Success rate:\\s*([0-9]*\\.?[0-9]+)", text, flags=re.IGNORECASE)
        print(matches[-1] if matches else "nan")
        PY
        }

        require_file "${MODEL_PATH}" "Download the target checkpoint and set MODEL_PATH=/abs/path/to/model"

        mkdir -p "${SUMMARY_DIR}"
        echo "task_name,env_name,success_rate_percent,run_dir,info_json,log_path,video_dir,replan_log_path,server_log" > "${SUMMARY_CSV}"

        echo "Starting policy server on GPU ${GPU_ID}, port ${PORT}"
        PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
          "${PYTHON_BIN}" scripts/inference_service.py --server \
          --model_path "${MODEL_PATH}" \
          --data_config "${DATA_CONFIG}" \
          --port "${PORT}" \
          --host "0.0.0.0" \
          > "${SERVER_LOG}" 2>&1 &
        server_pid=$!

        wait_for_server

        for ENV_NAME in "${ENV_NAMES[@]}"; do
          TASK_NAME="$(task_name_from_env "${ENV_NAME}")"
          TASK_DIR="${RESULT_ROOT}/${TASK_NAME}/${MODEL_NAME}/${CKPT_TAG}/${EVAL_TAG}"
          RUN_TIMESTAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
          RUN_DIR="${TASK_DIR}/${RUN_TIMESTAMP}"
          VIDEO_DIR="${RUN_DIR}/videos"
          LOG_PATH="${RUN_DIR}/run.log"
          INFO_PATH="${RUN_DIR}/info.json"
          REPLAN_LOG_PATH=""
          if is_truthy "${DUMP_DENOISE}" || [[ "${EVAL_TYPE}" == "stride" ]]; then
            REPLAN_LOG_PATH="${RUN_DIR}/replan_trace.jsonl"
          fi
          START_TIME_ISO="$(date --iso-8601=seconds)"
          START_TIME_EPOCH="$(date +%s)"

          mkdir -p "${VIDEO_DIR}"

          echo "Running ${ENV_NAME}"
          MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES="${GPU_ID}" \
            "${SIM_PYTHON}" scripts/simulation_service.py --client \
            --host "${HOST}" \
            --port "${PORT}" \
            --env_name "${ENV_NAME}" \
            --video_dir "${VIDEO_DIR}" \
            --max_episode_steps "${MAX_EPISODE_STEPS}" \
            --n_episodes "${N_EPISODES}" \
            --n_envs "${N_ENVS}" \
            --n_action_steps "${N_ACTION_STEPS}" \
            --eval_type "${EVAL_TYPE}" \
            --dump_denoise "${DUMP_DENOISE}" \
            --stride_candidates "${STRIDE_CANDIDATES}" \
            --stride_exec_mode "${STRIDE_EXEC_MODE}" \
            --stride_expected_temp "${STRIDE_EXPECTED_TEMP}" \
            --stride_ts_alpha "${STRIDE_TS_ALPHA}" \
            --stride_ts_epsilon "${STRIDE_TS_EPSILON}" \
            --stride_ts_seed "${STRIDE_TS_SEED}" \
            --stride_ts_update_mode "${STRIDE_TS_UPDATE_MODE}" \
            --stride_ts_kernel_bandwidth "${STRIDE_TS_KERNEL_BANDWIDTH}" \
            --stride_ts_forget_rho "${STRIDE_TS_FORGET_RHO}" \
            --stride_ts_update_eta "${STRIDE_TS_UPDATE_ETA}" \
            --stride_z_mode "${STRIDE_Z_MODE}" \
            --stride_tau_window "${STRIDE_TAU_WINDOW}" \
            --stride_signal_mode "${STRIDE_SIGNAL_MODE}" \
            --stride_use_speed_uniformity "${STRIDE_USE_SPEED_UNIFORMITY}" \
            --stride_speed_window "${STRIDE_SPEED_WINDOW}" \
            --stride_speed_weight "${STRIDE_SPEED_WEIGHT}" \
            --stride_speed_beta "${STRIDE_SPEED_BETA}" \
            --stride_speed_fallback "${STRIDE_SPEED_FALLBACK}" \
            --replan_log_path "${REPLAN_LOG_PATH}" \
            2>&1 | tee "${LOG_PATH}"

          RATE="$(extract_success_rate "${LOG_PATH}")"
          RATE_PERCENT="$(python - "${RATE}" <<'PY'
        import math
        import sys
        rate = sys.argv[1]
        try:
            value = float(rate)
            if math.isnan(value):
                raise ValueError
            print(f"{value * 100:.1f}")
        except Exception:
            print("nan")
        PY
        )"
          END_TIME_ISO="$(date --iso-8601=seconds)"
          END_TIME_EPOCH="$(date +%s)"
          DURATION_SECONDS="$((END_TIME_EPOCH - START_TIME_EPOCH))"

          cat > "${INFO_PATH}" <<EOF
        {
          "task_name": "${TASK_NAME}",
          "env_name": "${ENV_NAME}",
          "model_name": "${MODEL_NAME}",
          "eval_tag": "${EVAL_TAG}",
          "ckpt_tag": "${CKPT_TAG}",
          "run_dir": "${RUN_DIR}",
          "model_path": "${MODEL_PATH}",
          "host": "${HOST}",
          "port": ${PORT},
          "gpu_id": ${GPU_ID},
          "n_episodes": ${N_EPISODES},
          "n_envs": ${N_ENVS},
          "max_episode_steps": ${MAX_EPISODE_STEPS},
          "n_action_steps": ${N_ACTION_STEPS},
          "eval_type": "${EVAL_TYPE}",
          "dump_denoise": ${DUMP_DENOISE},
          "stride_candidates": "${STRIDE_CANDIDATES}",
          "stride_exec_mode": "${STRIDE_EXEC_MODE}",
          "stride_expected_temp": ${STRIDE_EXPECTED_TEMP},
          "stride_ts_alpha": ${STRIDE_TS_ALPHA},
          "stride_ts_epsilon": ${STRIDE_TS_EPSILON},
          "stride_ts_seed": ${STRIDE_TS_SEED},
          "stride_ts_update_mode": "${STRIDE_TS_UPDATE_MODE}",
          "stride_ts_kernel_bandwidth": ${STRIDE_TS_KERNEL_BANDWIDTH},
          "stride_ts_forget_rho": ${STRIDE_TS_FORGET_RHO},
          "stride_ts_update_eta": ${STRIDE_TS_UPDATE_ETA},
          "stride_z_mode": "${STRIDE_Z_MODE}",
          "stride_tau_window": ${STRIDE_TAU_WINDOW},
          "stride_signal_mode": "${STRIDE_SIGNAL_MODE}",
          "stride_use_speed_uniformity": ${STRIDE_USE_SPEED_UNIFORMITY},
          "stride_speed_window": ${STRIDE_SPEED_WINDOW},
          "stride_speed_weight": ${STRIDE_SPEED_WEIGHT},
          "stride_speed_beta": ${STRIDE_SPEED_BETA},
          "stride_speed_fallback": "${STRIDE_SPEED_FALLBACK}",
          "success_rate": ${RATE},
          "success_rate_percent": "${RATE_PERCENT}",
          "start_time": "${START_TIME_ISO}",
          "end_time": "${END_TIME_ISO}",
          "duration_seconds": ${DURATION_SECONDS},
          "log_path": "${LOG_PATH}",
          "video_dir": "${VIDEO_DIR}",
          "replan_log_path": "${REPLAN_LOG_PATH}",
          "server_log": "${SERVER_LOG}"
        }
        EOF

          echo "${TASK_NAME},${ENV_NAME},${RATE_PERCENT},${RUN_DIR},${INFO_PATH},${LOG_PATH},${VIDEO_DIR},${REPLAN_LOG_PATH},${SERVER_LOG}" >> "${SUMMARY_CSV}"
        done

        echo "Evaluation finished."
        echo "Summary CSV: ${SUMMARY_CSV}"
        """
    ).lstrip()
    write_text(path, content)


def copy_helper_files() -> None:
    helper_files = [
        ("gr00t/eval/denoise_logger.py", "gr00t/eval/denoise_logger.py"),
        ("gr00t/eval/stride_selector.py", "gr00t/eval/stride_selector.py"),
    ]
    for src_rel, dst_rel in helper_files:
        src = THIS_REPO / src_rel
        dst = TARGET_REPO / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def main() -> None:
    if not TARGET_REPO.exists():
        raise SystemExit(f"Target repo not found: {TARGET_REPO}")

    copy_helper_files()
    rewrite_action_head()
    rewrite_gr00t_n1()
    rewrite_policy()
    rewrite_multistep_wrapper()
    rewrite_simulation_service()
    rewrite_simulation()
    rewrite_batch_script()
    print(f"Patched stride eval support into {TARGET_REPO}")


if __name__ == "__main__":
    main()
