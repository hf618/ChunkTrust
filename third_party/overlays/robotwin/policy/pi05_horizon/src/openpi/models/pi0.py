import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.qha as _qha
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")
_DEFAULT_PI0_DENOISE_STEPS = 10


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        self.qha = None
        if config.use_qha:
            self.qha = _qha.QueryBasedHorizonAdapter(
                context_dim=paligemma_config.width,
                action_dim=action_expert_config.width,
                config=_qha.QHAConfig(
                    candidate_horizons=_qha.parse_candidate_horizons(
                        config.qha_candidates,
                        max_horizon=config.action_horizon,
                    ),
                    d_model=config.qha_d_model,
                    num_queries=config.qha_num_queries,
                    num_heads=config.qha_num_heads,
                    num_bridge_layers=config.qha_num_bridge_layers,
                    use_fusion_gate=config.qha_use_fusion_gate,
                    dropout=config.qha_dropout,
                    loss_type=config.qha_loss_type,
                    eps=config.qha_eps,
                ),
                rngs=rngs,
            )

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def _compute_flow_loss(
        self,
        noise_rng: at.KeyArrayLike,
        time_rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
    ) -> at.Float[at.Array, "*b ah"]:
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        del prefix_out
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    def _prepare_prefix_context_cache(
        self,
        observation: _model.Observation,
    ) -> tuple[
        at.Bool[at.Array, "b p"],
        at.Float[at.Array, "b p d"],
        at.Array,
    ]:
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_hidden, _), kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        assert prefix_hidden is not None
        return prefix_mask, prefix_hidden, kv_cache

    def _run_suffix_with_prefix_cache(
        self,
        observation: _model.Observation,
        x_t: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        *,
        prefix_mask: at.Bool[at.Array, "b p"],
        kv_cache: at.Array,
    ) -> tuple[at.Float[at.Array, "b h d"], _model.Actions]:
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, timestep)
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None
        action_hidden = suffix_out[:, -self.action_horizon :]
        velocity = self.action_out_proj(action_hidden)
        return action_hidden, velocity

    def _get_hist_actions_and_mask(
        self, observation: _model.Observation
    ) -> tuple[at.Float[at.Array, "b l ad"], at.Bool[at.Array, "b l"]]:
        if observation.hist_actions is not None:
            hist_actions = observation.hist_actions
            if observation.hist_actions_mask is None:
                hist_mask = jnp.ones(hist_actions.shape[:2], dtype=jnp.bool_)
            else:
                hist_mask = observation.hist_actions_mask
            return hist_actions, hist_mask
        hist_actions = observation.state[:, None, :]
        hist_mask = jnp.ones(hist_actions.shape[:2], dtype=jnp.bool_)
        return hist_actions, hist_mask

    def _qha_candidate_horizons(self) -> tuple[int, ...]:
        return _qha.resolve_qha_candidate_horizons(
            self.config.qha_candidates,
            candidate_mode=self.config.qha_candidate_mode,
            max_horizon=self.action_horizon,
        )

    def _qha_runtime_sparse_candidate_horizons(self) -> tuple[int, ...]:
        return _qha.default_sparse_runtime_candidate_horizons(max_horizon=self.action_horizon)

    def _qha_selector_config(self) -> _qha.SelectorConfig:
        return _qha.SelectorConfig(
            candidate_horizons=self._qha_candidate_horizons(),
            exec_mode=self.config.horizon_selector_exec_mode,
            expected_temp=self.config.horizon_selector_expected_temp,
            intra_cut_t=self.config.horizon_selector_intra_cut_t,
            intra_alpha=self.config.horizon_selector_intra_alpha,
            intra_z_mode=self.config.horizon_selector_intra_z_mode,
            intra_tau_window=self.config.horizon_selector_intra_tau_window,
            mix_mode=self.config.horizon_selector_mix_mode,
            inter_use_speed_uniformity=bool(self.config.horizon_selector_inter_use_speed_uniformity),
            inter_window=self.config.horizon_selector_inter_window,
            inter_weight=self.config.horizon_selector_inter_weight,
            inter_beta=self.config.horizon_selector_inter_beta,
            inter_fallback=self.config.horizon_selector_inter_fallback,
            eps=self.config.qha_eps,
        )

    def _qha_runtime_sparse_selector_config(self) -> _qha.SelectorConfig:
        return _qha.SelectorConfig(
            candidate_horizons=self._qha_runtime_sparse_candidate_horizons(),
            exec_mode=self.config.horizon_selector_exec_mode,
            expected_temp=self.config.horizon_selector_expected_temp,
            intra_cut_t=self.config.horizon_selector_intra_cut_t,
            intra_alpha=self.config.horizon_selector_intra_alpha,
            intra_z_mode=self.config.horizon_selector_intra_z_mode,
            intra_tau_window=self.config.horizon_selector_intra_tau_window,
            mix_mode=self.config.horizon_selector_mix_mode,
            inter_use_speed_uniformity=bool(self.config.horizon_selector_inter_use_speed_uniformity),
            inter_window=self.config.horizon_selector_inter_window,
            inter_weight=self.config.horizon_selector_inter_weight,
            inter_beta=self.config.horizon_selector_inter_beta,
            inter_fallback=self.config.horizon_selector_inter_fallback,
            eps=self.config.qha_eps,
        )

    def _default_num_steps(self, num_steps: int | at.Int[at.Array, ""] | None) -> int:
        if num_steps is None:
            return _DEFAULT_PI0_DENOISE_STEPS
        if isinstance(num_steps, int):
            steps = int(num_steps)
        else:
            steps = int(jax.device_get(num_steps))
        if steps <= 0:
            raise ValueError(f"num_steps must be positive, got {steps}")
        return steps

    def _build_qha_rollout(
        self,
        observation: _model.Observation,
        rollout_rng: at.KeyArrayLike,
        *,
        num_steps: int = _DEFAULT_PI0_DENOISE_STEPS,
    ) -> dict[str, at.Array]:
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rollout_rng, (batch_size, self.action_horizon, self.action_dim))
        prefix_mask, context_tokens, kv_cache = self._prepare_prefix_context_cache(observation)
        dt = jnp.asarray(-1.0 / num_steps, dtype=noise.dtype)
        time_grid = jnp.asarray(1.0, dtype=noise.dtype) + dt * jnp.arange(num_steps, dtype=noise.dtype)

        def scan_step(
            x_t: _model.Actions,
            time_scalar: at.Float[at.Array, ""],
        ) -> tuple[_model.Actions, _model.Actions]:
            time_batch = jnp.broadcast_to(time_scalar, (batch_size,))
            _, v_t = self._run_suffix_with_prefix_cache(
                observation,
                x_t,
                time_batch,
                prefix_mask=prefix_mask,
                kv_cache=kv_cache,
            )
            next_x_t = x_t + dt * v_t
            return next_x_t, v_t

        final_actions, v_seq = jax.lax.scan(scan_step, noise, xs=time_grid)
        final_action_latents, _ = self._run_suffix_with_prefix_cache(
            observation,
            final_actions,
            jnp.zeros((batch_size,), dtype=noise.dtype),
            prefix_mask=prefix_mask,
            kv_cache=kv_cache,
        )
        return {
            "actions": final_actions,
            "fm_tensor": jnp.swapaxes(v_seq, 0, 1),
            "context_tokens": context_tokens,
            "context_mask": prefix_mask,
            "action_latents": final_action_latents,
        }

    def _compute_qha_outputs_from_rollout(
        self,
        observation: _model.Observation,
        rollout_rng: at.KeyArrayLike,
        *,
        num_steps: int = _DEFAULT_PI0_DENOISE_STEPS,
        return_qha_features: bool = False,
    ) -> dict[str, at.Array]:
        if self.qha is None:
            raise RuntimeError("QHA is disabled for this model.")

        rollout = self._build_qha_rollout(observation, rollout_rng, num_steps=num_steps)
        qha_outputs = self.qha(
            rollout["context_tokens"],
            rollout["action_latents"],
            context_mask=rollout["context_mask"],
        )
        hist_actions, hist_mask = self._get_hist_actions_and_mask(observation)
        selector_outputs = _qha.selector_scores_from_rollout(
            rollout["fm_tensor"],
            rollout["actions"],
            hist_actions,
            hist_mask,
            self._qha_selector_config(),
        )
        outputs = {
            "actions": rollout["actions"],
            "fm_tensor": rollout["fm_tensor"],
            **qha_outputs,
            "qha_selector_q_intra": selector_outputs["q_intra"],
            "qha_selector_p_inter": selector_outputs["p_inter"],
            "qha_selector_q_mix": selector_outputs["q_mix"],
            "qha_selector_z_intra_raw": selector_outputs["z_intra_raw"],
            "qha_selector_z_intra_norm": selector_outputs["z_intra_norm"],
            "qha_selector_u_inter_raw": selector_outputs["u_inter_raw"],
            "qha_selector_u_inter_norm": selector_outputs["u_inter_norm"],
            "qha_teacher_posterior": selector_outputs["teacher_posterior"],
            "qha_teacher_expected_horizon": selector_outputs["teacher_expected_horizon"],
            "qha_teacher_argmax_horizon": selector_outputs["teacher_argmax_horizon"],
        }
        # These frozen-backbone tensors are large.  Keep them off the normal
        # deployment return path and expose them only for an explicitly
        # configured SHIFT training trace.
        if return_qha_features:
            outputs.update(
                {
                    "qha_context_tokens": rollout["context_tokens"],
                    "qha_context_mask": rollout["context_mask"],
                    "qha_action_latents": rollout["action_latents"],
                }
            )
        if self.config.qha_candidate_mode == "dense_full":
            sparse_selector_outputs = _qha.selector_scores_from_rollout(
                rollout["fm_tensor"],
                rollout["actions"],
                hist_actions,
                hist_mask,
                self._qha_runtime_sparse_selector_config(),
            )
            outputs.update(
                {
                    "qha_selector_candidate_horizons_sparse_runtime": sparse_selector_outputs["candidate_horizons"],
                    "qha_selector_q_intra_sparse_runtime": sparse_selector_outputs["q_intra"],
                    "qha_selector_p_inter_sparse_runtime": sparse_selector_outputs["p_inter"],
                    "qha_selector_q_mix_sparse_runtime": sparse_selector_outputs["q_mix"],
                }
            )
        return outputs

    def _empty_qha_alignment_metrics(
        self,
        batch_size: int,
        dtype: at.Array | jnp.dtype,
        *,
        fill_nan: bool = False,
    ) -> dict[str, at.Array]:
        dtype = jnp.asarray(0, dtype=dtype).dtype
        base = jnp.full((batch_size,), jnp.nan if fill_nan else 0.0, dtype=dtype)
        scalar_base = jnp.asarray(jnp.nan if fill_nan else 0.0, dtype=dtype)
        return {
            "qha_expected_horizon": base,
            "qha_teacher_expected_horizon": base,
            "qha_expected_horizon_gap_abs": base,
            "qha_entropy": base,
            "qha_teacher_entropy": base,
            "qha_entropy_gap_abs": base,
            "qha_argmax_match_rate": base,
            "qha_vs_batch_prior_ce_gain": base,
            "qha_expected_horizon_corr": scalar_base,
            "qha_expected_horizon_std_ratio": scalar_base,
        }

    def _compute_qha_alignment_metrics(
        self,
        posterior: at.Float[at.Array, "b s"],
        teacher: at.Float[at.Array, "b s"],
        horizons: at.Int[at.Array, "s"],
    ) -> dict[str, at.Array]:
        eps = self.config.qha_eps
        horizon_vals = horizons.astype(posterior.dtype)
        expected = jnp.sum(posterior * horizon_vals[None, :], axis=-1)
        teacher_expected = jnp.sum(teacher * horizon_vals[None, :], axis=-1)
        posterior_clipped = jnp.clip(posterior, eps, 1.0)
        teacher_clipped = jnp.clip(teacher, eps, 1.0)
        teacher_batch_prior = jnp.mean(teacher, axis=0)
        teacher_batch_prior = teacher_batch_prior / jnp.maximum(jnp.sum(teacher_batch_prior), eps)
        teacher_batch_prior_clipped = jnp.clip(teacher_batch_prior, eps, 1.0)
        student_ce = -jnp.sum(teacher * jnp.log(posterior_clipped), axis=-1)
        batch_prior_ce = -jnp.sum(teacher * jnp.log(teacher_batch_prior_clipped[None, :]), axis=-1)
        expected_centered = expected - jnp.mean(expected)
        teacher_expected_centered = teacher_expected - jnp.mean(teacher_expected)
        corr_denom = jnp.sqrt(
            jnp.sum(jnp.square(expected_centered)) * jnp.sum(jnp.square(teacher_expected_centered))
        )
        expected_corr = jnp.where(
            corr_denom > eps,
            jnp.sum(expected_centered * teacher_expected_centered) / corr_denom,
            jnp.asarray(0.0, dtype=posterior.dtype),
        )
        expected_std_ratio = jnp.std(expected) / jnp.maximum(jnp.std(teacher_expected), eps)
        entropy = -jnp.sum(posterior_clipped * jnp.log(posterior_clipped), axis=-1)
        teacher_entropy = -jnp.sum(teacher_clipped * jnp.log(teacher_clipped), axis=-1)
        return {
            "qha_expected_horizon": expected,
            "qha_teacher_expected_horizon": teacher_expected,
            "qha_expected_horizon_gap_abs": jnp.abs(expected - teacher_expected),
            "qha_entropy": entropy,
            "qha_teacher_entropy": teacher_entropy,
            "qha_entropy_gap_abs": jnp.abs(entropy - teacher_entropy),
            "qha_argmax_match_rate": (
                jnp.argmax(posterior, axis=-1) == jnp.argmax(teacher, axis=-1)
            ).astype(posterior.dtype),
            "qha_vs_batch_prior_ce_gain": batch_prior_ce - student_ce,
            "qha_expected_horizon_corr": expected_corr,
            "qha_expected_horizon_std_ratio": expected_std_ratio,
        }

    def _compute_qha_training_outputs(
        self,
        observation: _model.Observation,
        rollout_rng: at.KeyArrayLike,
        *,
        stop_gradient_backbone: bool,
        compute_teacher: bool = True,
        num_steps: int = _DEFAULT_PI0_DENOISE_STEPS,
    ) -> dict[str, at.Array]:
        if self.qha is None:
            raise RuntimeError("QHA is disabled for this model.")

        rollout = self._build_qha_rollout(observation, rollout_rng, num_steps=num_steps)
        context_tokens = rollout["context_tokens"]
        context_mask = rollout["context_mask"]
        action_latents = rollout["action_latents"]
        fm_tensor = rollout["fm_tensor"]
        final_actions = rollout["actions"]
        if stop_gradient_backbone:
            context_tokens = jax.lax.stop_gradient(context_tokens)
            action_latents = jax.lax.stop_gradient(action_latents)
            fm_tensor = jax.lax.stop_gradient(fm_tensor)
            final_actions = jax.lax.stop_gradient(final_actions)

        qha_outputs = self.qha(
            context_tokens,
            action_latents,
            context_mask=context_mask,
        )
        outputs = {
            "actions": final_actions,
            "fm_tensor": fm_tensor,
            **qha_outputs,
        }
        if not compute_teacher:
            return outputs

        hist_actions, hist_mask = self._get_hist_actions_and_mask(observation)
        selector_outputs = _qha.selector_scores_from_rollout(
            fm_tensor,
            final_actions,
            hist_actions,
            hist_mask,
            self._qha_selector_config(),
        )
        teacher_posterior = jax.lax.stop_gradient(selector_outputs["teacher_posterior"])
        outputs.update({
            "qha_selector_q_intra": selector_outputs["q_intra"],
            "qha_selector_p_inter": selector_outputs["p_inter"],
            "qha_selector_q_mix": selector_outputs["q_mix"],
            "qha_selector_z_intra_raw": selector_outputs["z_intra_raw"],
            "qha_selector_z_intra_norm": selector_outputs["z_intra_norm"],
            "qha_selector_u_inter_raw": selector_outputs["u_inter_raw"],
            "qha_selector_u_inter_norm": selector_outputs["u_inter_norm"],
            "qha_teacher_posterior": teacher_posterior,
            "qha_teacher_expected_horizon": jax.lax.stop_gradient(selector_outputs["teacher_expected_horizon"]),
            "qha_teacher_argmax_horizon": jax.lax.stop_gradient(selector_outputs["teacher_argmax_horizon"]),
        })
        return outputs

    def _compute_qha_supervision_outputs_from_rollout(
        self,
        observation: _model.Observation,
        rollout_rng: at.KeyArrayLike,
        *,
        num_steps: int = _DEFAULT_PI0_DENOISE_STEPS,
    ) -> dict[str, at.Array]:
        """Build frozen-backbone QHA features plus selector supervision labels."""
        rollout = self._build_qha_rollout(observation, rollout_rng, num_steps=num_steps)
        hist_actions, hist_mask = self._get_hist_actions_and_mask(observation)
        selector_outputs = _qha.selector_scores_from_rollout(
            rollout["fm_tensor"],
            rollout["actions"],
            hist_actions,
            hist_mask,
            self._qha_selector_config(),
        )
        return {
            "qha_context_tokens": rollout["context_tokens"],
            "qha_context_mask": rollout["context_mask"],
            "qha_action_latents": rollout["action_latents"],
            "qha_candidate_horizons": selector_outputs["candidate_horizons"],
            "qha_teacher_posterior": selector_outputs["teacher_posterior"],
            "qha_teacher_expected_horizon": selector_outputs["teacher_expected_horizon"],
            "qha_teacher_argmax_horizon": selector_outputs["teacher_argmax_horizon"],
            "qha_selector_q_intra": selector_outputs["q_intra"],
            "qha_selector_p_inter": selector_outputs["p_inter"],
            "qha_selector_q_mix": selector_outputs["q_mix"],
            "qha_selector_z_intra_raw": selector_outputs["z_intra_raw"],
            "qha_selector_z_intra_norm": selector_outputs["z_intra_norm"],
            "qha_selector_u_inter_raw": selector_outputs["u_inter_raw"],
            "qha_selector_u_inter_norm": selector_outputs["u_inter_norm"],
        }

    def predict_qha_teacher_outputs(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions_ref: _model.Actions | None = None,
        *,
        num_steps: int | at.Int[at.Array, ""] | None = None,
    ) -> dict[str, at.Array] | None:
        del actions_ref
        if self.qha is None:
            return None
        observation = _model.preprocess_observation(None, observation, train=False)
        outputs = self._compute_qha_training_outputs(
            observation,
            rng,
            stop_gradient_backbone=True,
            num_steps=self._default_num_steps(num_steps),
        )
        return {
            "qha_candidate_horizons": outputs["qha_candidate_horizons"],
            "qha_teacher_posterior": outputs["qha_teacher_posterior"],
            "qha_teacher_expected_horizon": outputs["qha_teacher_expected_horizon"],
            "qha_teacher_argmax_horizon": outputs["qha_teacher_argmax_horizon"],
            "qha_selector_q_intra": outputs["qha_selector_q_intra"],
            "qha_selector_p_inter": outputs["qha_selector_p_inter"],
            "qha_selector_q_mix": outputs["qha_selector_q_mix"],
            "qha_selector_z_intra_raw": outputs["qha_selector_z_intra_raw"],
            "qha_selector_z_intra_norm": outputs["qha_selector_z_intra_norm"],
            "qha_selector_u_inter_raw": outputs["qha_selector_u_inter_raw"],
            "qha_selector_u_inter_norm": outputs["qha_selector_u_inter_norm"],
        }

    def predict_qha_supervision_outputs(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions_ref: _model.Actions | None = None,
        *,
        num_steps: int | at.Int[at.Array, ""] | None = None,
    ) -> dict[str, at.Array] | None:
        del actions_ref
        observation = _model.preprocess_observation(None, observation, train=False)
        return self._compute_qha_supervision_outputs_from_rollout(
            observation,
            rng,
            num_steps=self._default_num_steps(num_steps),
        )

    def sample_actions_and_qha_outputs(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] | None = None,
        return_qha_features: bool = False,
    ) -> dict[str, at.Array]:
        observation = _model.preprocess_observation(None, observation, train=False)
        steps = self._default_num_steps(num_steps)
        if self.qha is None:
            return {
                "actions": self.sample_actions(rng, observation, num_steps=steps),
            }
        return self._compute_qha_outputs_from_rollout(
            observation,
            rng,
            num_steps=steps,
            return_qha_features=return_qha_features,
        )

    def compute_loss_and_metrics(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        compute_qha_alignment_metrics: bool = True,
        compute_horizon_alignment_metrics: bool | None = None,
        external_qha_features: dict[str, at.Array] | None = None,
        external_qha_teacher: dict[str, at.Array] | None = None,
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        if compute_horizon_alignment_metrics is not None:
            compute_qha_alignment_metrics = bool(compute_horizon_alignment_metrics)
        preprocess_rng, noise_rng, time_rng, qha_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        if self.qha is not None and self.config.qha_train_mode == "qha_only":
            flow_loss = jnp.zeros(actions.shape[:-1], dtype=actions.dtype)
            if external_qha_features is None:
                qha_outputs = self._compute_qha_training_outputs(
                    observation,
                    qha_rng,
                    stop_gradient_backbone=True,
                    compute_teacher=external_qha_teacher is None,
                    num_steps=self._default_num_steps(None),
                )
            else:
                context_tokens = jax.lax.stop_gradient(external_qha_features["qha_context_tokens"])
                action_latents = jax.lax.stop_gradient(external_qha_features["qha_action_latents"])
                context_mask = jax.lax.stop_gradient(external_qha_features["qha_context_mask"])
                qha_outputs = self.qha(
                    context_tokens,
                    action_latents,
                    context_mask=context_mask,
                )
            if external_qha_teacher is None:
                if external_qha_features is not None:
                    raise ValueError("external_qha_features requires external_qha_teacher for qha_only training.")
                teacher_posterior = qha_outputs["qha_teacher_posterior"]
                teacher_expected_horizon = qha_outputs["qha_teacher_expected_horizon"]
                teacher_argmax_horizon = qha_outputs["qha_teacher_argmax_horizon"]
            else:
                teacher_posterior = jax.lax.stop_gradient(external_qha_teacher["qha_teacher_posterior"])
                teacher_expected_horizon = jax.lax.stop_gradient(
                    external_qha_teacher["qha_teacher_expected_horizon"]
                )
                teacher_argmax_horizon = jax.lax.stop_gradient(
                    external_qha_teacher["qha_teacher_argmax_horizon"]
                )
            qha_loss = _qha.kl_or_soft_ce(
                qha_outputs["qha_posterior"],
                teacher_posterior,
                loss_type=self.config.qha_loss_type,
                eps=self.config.qha_eps,
            )
            total_loss = flow_loss + self.config.qha_loss_weight * qha_loss[..., None]
            metrics = {
                "flow_loss": jnp.mean(flow_loss, axis=-1),
                "qha_loss": qha_loss,
                "total_loss": jnp.mean(total_loss, axis=-1),
            }
            if compute_qha_alignment_metrics:
                metrics.update(
                    self._compute_qha_alignment_metrics(
                        qha_outputs["qha_posterior"],
                        teacher_posterior,
                        qha_outputs["qha_candidate_horizons"],
                    )
                )
                metrics["qha_teacher_expected_horizon"] = teacher_expected_horizon
                metrics["qha_teacher_argmax_horizon"] = teacher_argmax_horizon
            else:
                metrics.update(
                    self._empty_qha_alignment_metrics(
                        qha_outputs["qha_posterior"].shape[0],
                        qha_outputs["qha_posterior"].dtype,
                        fill_nan=True,
                    )
                )
            return total_loss, metrics

        flow_loss = self._compute_flow_loss(noise_rng, time_rng, observation, actions)
        metrics = {
            "flow_loss": jnp.mean(flow_loss, axis=-1),
        }

        if self.qha is None:
            metrics["qha_loss"] = jnp.zeros_like(metrics["flow_loss"])
            metrics.update(
                self._empty_qha_alignment_metrics(
                    metrics["flow_loss"].shape[0],
                    metrics["flow_loss"].dtype,
                    fill_nan=False,
                )
            )
            metrics["total_loss"] = metrics["flow_loss"]
            return flow_loss, metrics

        qha_outputs = self._compute_qha_training_outputs(
            observation,
            qha_rng,
            stop_gradient_backbone=False,
            num_steps=self._default_num_steps(None),
        )
        qha_loss = _qha.kl_or_soft_ce(
            qha_outputs["qha_posterior"],
            qha_outputs["qha_teacher_posterior"],
            loss_type=self.config.qha_loss_type,
            eps=self.config.qha_eps,
        )
        total_loss = flow_loss + self.config.qha_loss_weight * qha_loss[..., None]
        metrics["qha_loss"] = qha_loss
        metrics["total_loss"] = jnp.mean(total_loss, axis=-1)
        if compute_qha_alignment_metrics:
            metrics.update(
                self._compute_qha_alignment_metrics(
                    qha_outputs["qha_posterior"],
                    qha_outputs["qha_teacher_posterior"],
                    qha_outputs["qha_candidate_horizons"],
                )
            )
        else:
            metrics.update(
                self._empty_qha_alignment_metrics(
                    qha_outputs["qha_posterior"].shape[0],
                    qha_outputs["qha_posterior"].dtype,
                    fill_nan=True,
                )
            )
        return total_loss, metrics

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        loss, _ = self.compute_loss_and_metrics(rng, observation, actions, train=train)
        return loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    @at.typecheck
    def sample_actions_with_trace(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> tuple[_model.Actions, dict[str, at.Array]]:
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def scan_step(carry, _):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation,
                x_t,
                jnp.broadcast_to(time, (batch_size,)),
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            next_x_t = x_t + dt * v_t
            next_time = time + dt
            return (next_x_t, next_time), v_t

        (x_t, _), v_seq = jax.lax.scan(
            scan_step,
            (noise, jnp.asarray(1.0, dtype=jnp.float32)),
            xs=None,
            length=int(num_steps),
        )
        v_trace = jnp.swapaxes(v_seq, 0, 1)

        trace = {
            "v": v_trace,
            "ds": jnp.full((batch_size, int(num_steps)), jnp.asarray(dt, dtype=jnp.float32), dtype=jnp.float32),
            "x0": noise,
            "xt_internal": x_t,
        }
        return x_t, trace
