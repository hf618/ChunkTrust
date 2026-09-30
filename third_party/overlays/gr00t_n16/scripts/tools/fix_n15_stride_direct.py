from __future__ import annotations

import shutil
import sys
import textwrap
from pathlib import Path

THIS_REPO = Path(__file__).resolve().parents[2]
if str(THIS_REPO) not in sys.path:
    sys.path.insert(0, str(THIS_REPO))

from scripts.tools import patch_n15_stride_ts as base


TARGET_REPO = base.TARGET_REPO


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def replace_range(
    path: Path,
    *,
    start: int,
    end: int,
    replacement: str,
) -> None:
    content = read_text(path)
    write_text(path, content[:start] + replacement + content[end:])


def replace_between(
    path: Path,
    *,
    start_marker: str,
    end_marker: str,
    replacement: str,
    search_start: int = 0,
) -> None:
    content = read_text(path)
    start = content.find(start_marker, search_start)
    if start < 0:
        raise RuntimeError(f"Start marker not found in {path}: {start_marker!r}")
    end = content.find(end_marker, start)
    if end < 0:
        raise RuntimeError(f"End marker not found in {path}: {end_marker!r}")
    write_text(path, content[:start] + replacement + content[end:])


def ensure_contains(path: Path, needle: str, replacement: str) -> None:
    content = read_text(path)
    if needle in content:
        return
    if replacement not in content:
        raise RuntimeError(f"Needle not found in {path}: {needle!r}")
    write_text(path, content.replace(replacement, replacement + needle, 1))


def repair_action_head() -> None:
    path = TARGET_REPO / "gr00t/model/action_head/flow_matching_action_head.py"
    content = read_text(path)
    loss_end = content.find('return BatchFeature(data=output_dict)')
    if loss_end < 0:
        raise RuntimeError(f"Loss-return anchor not found in {path}")
    start = content.find("@torch.no_grad()", loss_end)
    if start < 0:
        raise RuntimeError(f"get_action start not found in {path}")
    start = content.rfind("\n", 0, start) + 1
    end = content.find("\n    @property\n    def device(self):", start)
    if end < 0:
        raise RuntimeError(f"device property anchor not found in {path}")
    replacement = textwrap.indent(
        textwrap.dedent(
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
        + "\n",
        "    ",
    )
    replace_range(path, start=start, end=end, replacement=replacement)


def repair_gr00t_n1() -> None:
    path = TARGET_REPO / "gr00t/model/gr00t_n1.py"
    content = read_text(path)
    class_start = content.find("class GR00T_N1_5")
    if class_start < 0:
        raise RuntimeError(f"class anchor not found in {path}")
    start = content.find("def get_action(", class_start)
    if start < 0:
        raise RuntimeError(f"get_action start not found in {path}")
    end = content.find("\n    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:", start)
    if end < 0:
        raise RuntimeError(f"prepare_input anchor not found in {path}")
    replacement = textwrap.indent(
        textwrap.dedent(
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
        + "\n",
        "    ",
    )
    replace_range(path, start=start, end=end, replacement=replacement)


def repair_policy() -> None:
    path = TARGET_REPO / "gr00t/model/policy.py"
    content = read_text(path)
    class_start = content.find("class Gr00tPolicy")
    if class_start < 0:
        raise RuntimeError(f"class anchor not found in {path}")
    start = content.find("    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:", class_start)
    if start < 0:
        raise RuntimeError(f"policy get_action start not found in {path}")
    end = content.find(
        "\n    def _get_unnormalized_action(self, normalized_action: torch.Tensor) -> Dict[str, Any]:",
        start,
    )
    if end < 0:
        raise RuntimeError(f"policy unnormalize anchor not found in {path}")
    replacement = textwrap.indent(
        textwrap.dedent(
            """
            def get_action(
                self,
                observations: Dict[str, Any],
                options: Optional[Dict[str, Any]] = None,
            ) -> Any:
                \"\"\"
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
                \"\"\"
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
        + "\n",
        "    ",
    )
    replace_range(path, start=start, end=end, replacement=replacement)


def repair_multistep_wrapper() -> None:
    path = TARGET_REPO / "gr00t/eval/wrappers/multistep_wrapper.py"
    content = read_text(path)
    needle = """        self._action_space = repeated_space(env.action_space, n_action_steps)\n"""
    if '"exec_horizon"' not in content:
        content = content.replace(
            needle,
            needle
            + """        self._action_space["exec_horizon"] = spaces.Box(
            low=np.array([1], dtype=np.int32),
            high=np.array([n_action_steps], dtype=np.int32),
            shape=(1,),
            dtype=np.int32,
        )
""",
            1,
        )
        write_text(path, content)
        content = read_text(path)

    start = content.find("    def step(self, action):\n")
    if start < 0:
        raise RuntimeError(f"step start not found in {path}")
    end = content.find("\n    def _get_obs(self, video_delta_indices, state_delta_indices):", start)
    if end < 0:
        raise RuntimeError(f"_get_obs anchor not found in {path}")
    replacement = textwrap.indent(
        textwrap.dedent(
            """
            def step(self, action):
                \"\"\"
                action: dict: key-value pairs where the values are of shape (n_action_steps,) + action_shape
                \"\"\"
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
        + "\n",
        "    ",
    )
    replace_range(path, start=start, end=end, replacement=replacement)


def repair_robot_client() -> None:
    path = TARGET_REPO / "gr00t/eval/robot.py"
    content = read_text(path)
    old = """    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        return self.call_endpoint("get_action", observations)
"""
    new = """    def get_action(
        self,
        observations: Dict[str, Any],
        options: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        payload = observations if options is None else {"observation": observations, "options": options}
        return self.call_endpoint("get_action", payload)
"""
    if old in content:
        write_text(path, content.replace(old, new, 1))


def main() -> None:
    if not TARGET_REPO.exists():
        raise SystemExit(f"Target repo not found: {TARGET_REPO}")

    base.copy_helper_files()
    repair_action_head()
    repair_gr00t_n1()
    repair_policy()
    repair_multistep_wrapper()
    repair_robot_client()
    base.rewrite_simulation()
    base.rewrite_simulation_service()
    base.rewrite_batch_script()
    print(f"Directly repaired stride support in {TARGET_REPO}")


if __name__ == "__main__":
    main()
