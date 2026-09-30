import dataclasses
import functools
import logging
import numpy as np

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

import openpi.models.qha as _qha
from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")
_DEFAULT_PI0_DENOISE_STEPS = 10


def _siglip_token_count(image_resolution: tuple[int, int], patch_size: int = 14) -> int:
    height, width = image_resolution
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Image resolution {image_resolution} must be divisible by patch size {patch_size} "
            "to build a static prefix attention template."
        )
    return (height // patch_size) * (width // patch_size)


def build_prefix_ar_mask(*, num_image_streams: int, image_tokens_per_stream: int, max_token_len: int) -> np.ndarray:
    """Build the static prefix autoregressive mask used by embed_prefix()."""
    prefix_len = int(num_image_streams) * int(image_tokens_per_stream) + int(max_token_len)
    return np.zeros((prefix_len,), dtype=np.bool_)


def build_suffix_ar_mask(*, action_horizon: int) -> np.ndarray:
    """Build the static suffix autoregressive mask used by _embed_suffix_from_state()."""
    return np.asarray([True] + [True] + ([False] * (int(action_horizon) - 1)), dtype=np.bool_)


@functools.lru_cache(maxsize=None)
def get_prefix_attn_template(*, max_token_len: int) -> np.ndarray:
    prefix_ar_mask = build_prefix_ar_mask(
        num_image_streams=len(_model.IMAGE_KEYS),
        image_tokens_per_stream=_siglip_token_count(_model.IMAGE_RESOLUTION),
        max_token_len=max_token_len,
    )
    return make_attn_template(prefix_ar_mask)


@functools.lru_cache(maxsize=None)
def get_suffix_attn_template(*, action_horizon: int) -> np.ndarray:
    return make_attn_template(build_suffix_ar_mask(action_horizon=action_horizon))


def make_attn_template(mask_ar: np.ndarray | at.Array) -> np.ndarray:
    """Build a static [S, S] attention template from a 1D autoregressive mask."""
    mask_ar_np = np.asarray(mask_ar, dtype=np.int32).reshape(-1)
    cumsum = np.cumsum(mask_ar_np, axis=0)
    return (cumsum[None, :] <= cumsum[:, None]).astype(np.bool_)


def combine_input_mask_with_attn_template(
    input_mask: at.Bool[at.Array, "b s"],
    attn_template: np.ndarray | at.Array,
) -> at.Bool[at.Array, "b s s"]:
    """Apply a precomputed static attention template to a dynamic batch input mask."""
    template = jnp.asarray(attn_template, dtype=jnp.bool_)
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(template[None, :, :], valid_mask)


def build_suffix_mask(*, batch_size: int, action_horizon: int) -> at.Bool[at.Array, "b s"]:
    return jnp.ones((int(batch_size), int(action_horizon) + 1), dtype=jnp.bool_)


def build_suffix_positions(
    prefix_mask: at.Bool[at.Array, "b p"],
    *,
    action_horizon: int,
) -> at.Int[at.Array, "b s"]:
    prefix_len = jnp.sum(prefix_mask, axis=-1, dtype=jnp.int32)
    suffix_offsets = jnp.arange(int(action_horizon) + 1, dtype=jnp.int32)[None, :]
    return prefix_len[:, None] + suffix_offsets


def build_suffix_full_attn_mask(
    prefix_mask: at.Bool[at.Array, "b p"],
    *,
    action_horizon: int,
) -> tuple[at.Bool[at.Array, "b s"], at.Bool[at.Array, "b s q"]]:
    suffix_mask = build_suffix_mask(batch_size=int(prefix_mask.shape[0]), action_horizon=action_horizon)
    suffix_attn_mask = combine_input_mask_with_attn_template(
        suffix_mask,
        get_suffix_attn_template(action_horizon=action_horizon),
    )
    prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_mask.shape[1])
    full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
    return suffix_mask, full_attn_mask


def build_rollout_timestep_grid(num_steps: int, *, dtype: jnp.dtype) -> at.Float[at.Array, "t"]:
    if int(num_steps) <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    dt = jnp.asarray(-1.0 / int(num_steps), dtype=dtype)
    integration_t0 = jnp.asarray(1.0, dtype=dtype)
    return integration_t0 + dt * jnp.arange(int(num_steps), dtype=dtype)


def build_rollout_time_emb_table(
    *,
    batch_size: int,
    num_steps: int,
    embedding_dim: int,
    dtype: jnp.dtype,
) -> at.Float[at.Array, "t b emb"]:
    rollout_t_grid = build_rollout_timestep_grid(num_steps, dtype=dtype)
    time_emb = posemb_sincos(rollout_t_grid, embedding_dim, min_period=4e-3, max_period=4.0)
    return einops.repeat(time_emb, "t emb -> t b emb", b=int(batch_size))


def build_rollout_time_tokens_table(
    time_emb_table: at.Float[at.Array, "t b emb"],
    *,
    action_horizon: int,
) -> at.Float[at.Array, "t b h emb"]:
    return einops.repeat(time_emb_table, "t b emb -> t b h emb", h=int(action_horizon))


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
def posemb_sincos(pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float,
                  max_period: float) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period)**fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 64
    use_qha: bool = False
    qha_train_mode: str = "end_to_end"  # end_to_end | qha_only
    qha_candidate_mode: str = "sparse"  # sparse | dense_full
    qha_candidates: str = "10,20,30,40,50"
    qha_num_queries: int = 8
    qha_d_model: int = 256
    qha_num_heads: int = 8
    qha_num_bridge_layers: int = 1
    qha_use_fusion_gate: bool = True
    qha_dropout: float = 0.0
    qha_loss_type: str = "KL"
    qha_loss_weight: float = 1.0
    qha_infer_mode: str = "hybrid"  # hybrid | posterior_only
    qha_hybrid_prior_gamma: float = 1.0
    qha_history_max_length: int = 64
    qha_eps: float = 1e-6

    horizon_selector_exec_mode: str = "expected_round"  # expected_round | thompson
    horizon_selector_expected_temp: float = 1.0
    horizon_selector_intra_cut_t: float = 0.25
    horizon_selector_intra_alpha: float = 4.0
    horizon_selector_intra_z_mode: str = "window_rms"
    horizon_selector_intra_tau_window: int = 3
    horizon_selector_mix_mode: str = "both"  # intra | inter | both
    horizon_selector_inter_use_speed_uniformity: bool = True
    horizon_selector_inter_window: int = 60
    horizon_selector_inter_weight: float = 0.5
    horizon_selector_inter_beta: float = 4.0
    horizon_selector_inter_fallback: str = "ehh_only"
    horizon_selector_ts_epsilon: float = 0.05
    horizon_selector_ts_seed: int = 0
    horizon_selector_ts_update_mode: str = "kernel_forget"
    horizon_selector_ts_kernel_bandwidth: float = 10.0
    horizon_selector_ts_forget_rho: float = 0.99
    horizon_selector_ts_update_eta: float = 1.0

    def __post_init__(self) -> None:
        train_mode = self.qha_train_mode.strip().lower()
        if train_mode not in ("end_to_end", "qha_only"):
            raise ValueError(f"qha_train_mode must be end_to_end or qha_only, got {self.qha_train_mode}.")
        object.__setattr__(self, "qha_train_mode", train_mode)

        infer_mode = self.qha_infer_mode.strip().lower()
        if infer_mode not in ("hybrid", "posterior_only"):
            raise ValueError(f"qha_infer_mode must be hybrid or posterior_only, got {self.qha_infer_mode}.")
        object.__setattr__(self, "qha_infer_mode", infer_mode)

        candidate_mode = self.qha_candidate_mode.strip().lower()
        if candidate_mode not in ("sparse", "dense_full"):
            raise ValueError(
                f"qha_candidate_mode must be sparse or dense_full, got {self.qha_candidate_mode}."
            )
        object.__setattr__(self, "qha_candidate_mode", candidate_mode)

        if self.qha_num_queries <= 0:
            raise ValueError(f"qha_num_queries must be positive, got {self.qha_num_queries}.")
        if self.qha_d_model <= 0:
            raise ValueError(f"qha_d_model must be positive, got {self.qha_d_model}.")
        if self.qha_num_heads <= 0:
            raise ValueError(f"qha_num_heads must be positive, got {self.qha_num_heads}.")
        if self.qha_d_model % self.qha_num_heads != 0:
            raise ValueError(
                f"qha_d_model ({self.qha_d_model}) must be divisible by qha_num_heads ({self.qha_num_heads}).")
        if self.qha_num_bridge_layers <= 0:
            raise ValueError(f"qha_num_bridge_layers must be positive, got {self.qha_num_bridge_layers}.")
        if self.qha_history_max_length <= 0:
            raise ValueError(f"qha_history_max_length must be positive, got {self.qha_history_max_length}.")
        if self.qha_loss_weight < 0:
            raise ValueError(f"qha_loss_weight must be >= 0, got {self.qha_loss_weight}.")
        if self.qha_hybrid_prior_gamma < 0:
            raise ValueError(f"qha_hybrid_prior_gamma must be >= 0, got {self.qha_hybrid_prior_gamma}.")

        loss_type = self.qha_loss_type.strip().lower()
        if loss_type not in ("kl", "soft_ce"):
            raise ValueError(f"qha_loss_type must be KL or soft_ce, got {self.qha_loss_type}.")
        object.__setattr__(self, "qha_loss_type", "KL" if loss_type == "kl" else "soft_ce")

        candidates = _qha.resolve_qha_candidate_horizons(
            self.qha_candidates,
            candidate_mode=self.qha_candidate_mode,
            max_horizon=self.action_horizon,
        )
        object.__setattr__(self, "qha_candidates", ",".join(str(v) for v in candidates))

        selector_mode = self.horizon_selector_exec_mode.strip().lower()
        if selector_mode not in ("expected_round", "thompson"):
            raise ValueError(
                f"horizon_selector_exec_mode must be expected_round or thompson, got {self.horizon_selector_exec_mode}.")
        object.__setattr__(self, "horizon_selector_exec_mode", selector_mode)

        selector_z_mode = self.horizon_selector_intra_z_mode.strip().lower()
        if selector_z_mode not in ("tau0_rms", "window_rms", "trend"):
            raise ValueError(
                "horizon_selector_intra_z_mode must be tau0_rms, window_rms, or trend. "
                f"Got {self.horizon_selector_intra_z_mode}."
            )
        object.__setattr__(self, "horizon_selector_intra_z_mode", selector_z_mode)

        selector_mix_mode = self.horizon_selector_mix_mode.strip().lower()
        if selector_mix_mode not in ("intra", "inter", "both"):
            raise ValueError(
                f"horizon_selector_mix_mode must be intra, inter, or both. Got {self.horizon_selector_mix_mode}.")
        object.__setattr__(self, "horizon_selector_mix_mode", selector_mix_mode)

        inter_fallback = self.horizon_selector_inter_fallback.strip().lower()
        if inter_fallback not in ("ehh_only", "neutral"):
            raise ValueError(
                "horizon_selector_inter_fallback must be ehh_only or neutral. "
                f"Got {self.horizon_selector_inter_fallback}."
            )
        object.__setattr__(self, "horizon_selector_inter_fallback", inter_fallback)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        hist_len = self.qha_history_max_length if self.use_qha else 1

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                hist_actions=jax.ShapeDtypeStruct([batch_size, hist_len, self.action_dim], jnp.float32),
                hist_actions_mask=jax.ShapeDtypeStruct([batch_size, hist_len], jnp.bool_),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        if self.use_qha and self.qha_train_mode == "qha_only":
            return nnx.Not(nnx_utils.PathRegex(".*qha.*"))

        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(gemma_params_filter, )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(nnx.Not(action_expert_params_filter), )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(action_expert_params_filter, )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")), )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


class Pi0(_model.BaseModel):

    def __init__(self, config: Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
            ))
        llm.lazy_init(rngs=rngs, method="init")
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            ))
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
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
            input_mask.append(einops.repeat(
                obs.image_masks[name],
                "b -> b s",
                s=image_tokens.shape[1],
            ))
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
    def _embed_suffix_from_state(
        self, state: at.Float[at.Array, "b ad"], noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = self._compute_action_time_embedding(timestep)
        tokens = self._embed_suffix_tokens_from_time_emb(state, noisy_actions, time_emb)
        input_mask = build_suffix_mask(batch_size=int(state.shape[0]), action_horizon=self.action_horizon)
        ar_mask = jnp.asarray(build_suffix_ar_mask(action_horizon=self.action_horizon))
        return tokens, input_mask, ar_mask

    def _compute_action_time_embedding(
        self,
        timestep: at.Float[at.Array, " b"],
    ) -> at.Float[at.Array, "b emb"]:
        return posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)

    def _embed_suffix_tokens_from_time_emb(
        self,
        state: at.Float[at.Array, "b ad"],
        noisy_actions: _model.Actions,
        time_emb: at.Float[at.Array, "b emb"],
    ) -> at.Float[at.Array, "b s emb"]:
        # add a single state token
        state_token = self.state_proj(state)[:, None, :]
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        action_time_tokens = self._compute_action_time_tokens(noisy_actions, time_tokens)
        return jnp.concatenate([state_token, action_time_tokens], axis=1)

    def _compute_action_time_tokens(
        self,
        noisy_actions: _model.Actions,
        time_tokens: at.Float[at.Array, "b h emb"],
    ) -> at.Float[at.Array, "b h emb"]:
        action_tokens = self.action_in_proj(noisy_actions)
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)
        return action_time_tokens

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        return self._embed_suffix_from_state(obs.state, noisy_actions, timestep)

    def _predict_velocity(
        self,
        observation: _model.Observation,
        x_t: _model.Actions,
        diffusion_t: at.Float[at.Array, " b"],
    ) -> _model.Actions:
        prefix_mask, kv_cache = self._prepare_prefix_cache(observation)
        return self._predict_velocity_with_prefix_cache(
            observation,
            x_t,
            diffusion_t,
            prefix_mask=prefix_mask,
            kv_cache=kv_cache,
        )

    def _prepare_prefix_cache(
        self,
        observation: _model.Observation,
    ) -> tuple[at.Bool[at.Array, "b p"], at.Array]:
        """Build a prefix KV cache once and reuse it across multiple suffix evaluations."""
        prefix_mask, _, kv_cache = self._prepare_prefix_context_cache(observation)
        return prefix_mask, kv_cache

    def _prepare_prefix_context_cache(
        self,
        observation: _model.Observation,
    ) -> tuple[at.Bool[at.Array, "b p"], at.Float[at.Array, "b p d"], at.Array]:
        prefix_tokens, prefix_mask, _ = self.embed_prefix(observation)
        prefix_attn_mask = combine_input_mask_with_attn_template(
            prefix_mask,
            get_prefix_attn_template(max_token_len=self.config.max_token_len),
        )
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_hidden, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
        )
        assert prefix_hidden is not None
        return prefix_mask, prefix_hidden, kv_cache

    def _predict_velocity_with_prefix_cache(
        self,
        observation: _model.Observation,
        x_t: _model.Actions,
        diffusion_t: at.Float[at.Array, " b"],
        *,
        prefix_mask: at.Bool[at.Array, "b p"],
        kv_cache: at.Array,
    ) -> _model.Actions:
        return self._predict_velocity_from_state_with_prefix_cache(
            observation.state,
            x_t,
            diffusion_t,
            prefix_mask=prefix_mask,
            kv_cache=kv_cache,
        )

    def _predict_velocity_from_state_with_prefix_cache(
        self,
        state: at.Float[at.Array, "b ad"],
        x_t: _model.Actions,
        diffusion_t: at.Float[at.Array, " b"],
        *,
        prefix_mask: at.Bool[at.Array, "b p"],
        kv_cache: at.Array,
    ) -> _model.Actions:
        _, velocity = self._run_action_suffix_with_prefix_cache(
            state,
            x_t,
            diffusion_t,
            prefix_mask=prefix_mask,
            kv_cache=kv_cache,
        )
        return velocity

    def _run_action_suffix_with_prefix_cache(
        self,
        state: at.Float[at.Array, "b ad"],
        x_t: _model.Actions,
        diffusion_t: at.Float[at.Array, " b"],
        *,
        prefix_mask: at.Bool[at.Array, "b p"],
        kv_cache: at.Array,
    ) -> tuple[at.Float[at.Array, "b h d"], _model.Actions]:
        suffix_context = self._prepare_suffix_runtime_context(prefix_mask)
        time_emb = self._compute_action_time_embedding(diffusion_t)
        return self._run_action_suffix_prepared(
            state,
            x_t,
            time_emb=time_emb,
            full_attn_mask=suffix_context["full_attn_mask"],
            positions=suffix_context["positions"],
            kv_cache=kv_cache,
        )

    def _prepare_suffix_runtime_context(
        self,
        prefix_mask: at.Bool[at.Array, "b p"],
        state: at.Float[at.Array, "b ad"] | None = None,
    ) -> dict[str, at.Array]:
        suffix_mask, full_attn_mask = build_suffix_full_attn_mask(
            prefix_mask,
            action_horizon=self.action_horizon,
        )
        positions = build_suffix_positions(prefix_mask, action_horizon=self.action_horizon)
        context = {
            "suffix_mask": suffix_mask,
            "full_attn_mask": full_attn_mask,
            "positions": positions,
        }
        if state is not None:
            context["state_token"] = self.state_proj(state)[:, None, :]
        return context

    def _run_action_suffix_prepared(
        self,
        state: at.Float[at.Array, "b ad"],
        x_t: _model.Actions,
        *,
        time_emb: at.Float[at.Array, "b emb"],
        full_attn_mask: at.Bool[at.Array, "b s q"],
        positions: at.Int[at.Array, "b s"],
        kv_cache: at.Array,
    ) -> tuple[at.Float[at.Array, "b h d"], _model.Actions]:
        state_token = self.state_proj(state)[:, None, :]
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        return self._run_action_suffix_prepared_from_state_token(
            state_token,
            x_t,
            time_tokens=time_tokens,
            full_attn_mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
        )

    def _run_action_suffix_prepared_from_state_token(
        self,
        state_token: at.Float[at.Array, "b 1 emb"],
        x_t: _model.Actions,
        *,
        time_tokens: at.Float[at.Array, "b h emb"],
        full_attn_mask: at.Bool[at.Array, "b s q"],
        positions: at.Int[at.Array, "b s"],
        kv_cache: at.Array,
    ) -> tuple[at.Float[at.Array, "b h d"], _model.Actions]:
        action_time_tokens = self._compute_action_time_tokens(x_t, time_tokens)
        suffix_tokens = jnp.concatenate([state_token, action_time_tokens], axis=1)
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
        )
        assert prefix_out is None
        action_hidden = suffix_out[:, -self.action_horizon:]
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
        # Fallback keeps the interface backward-compatible when history is unavailable.
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

    def _build_qha_rollout(
        self,
        observation: _model.Observation,
        rollout_rng: at.KeyArrayLike,
        *,
        num_steps: int = _DEFAULT_PI0_DENOISE_STEPS,
        prefix_mask: at.Bool[at.Array, "b p"] | None = None,
        context_tokens: at.Float[at.Array, "b p d"] | None = None,
        kv_cache: at.Array | None = None,
    ) -> dict[str, at.Array]:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rollout_rng, (batch_size, self.action_horizon, self.action_dim))
        if prefix_mask is None or context_tokens is None or kv_cache is None:
            prefix_mask, context_tokens, kv_cache = self._prepare_prefix_context_cache(observation)
        suffix_context = self._prepare_suffix_runtime_context(prefix_mask, observation.state)
        time_emb_table = build_rollout_time_emb_table(
            batch_size=batch_size,
            num_steps=num_steps,
            embedding_dim=self.action_in_proj.out_features,
            dtype=noise.dtype,
        )
        time_tokens_table = build_rollout_time_tokens_table(
            time_emb_table,
            action_horizon=self.action_horizon,
        )
        state_token = suffix_context["state_token"]
        euler_dt = jnp.asarray(-1.0 / num_steps, dtype=noise.dtype)

        def _scan_step(
            x_t: _model.Actions,
            time_tokens: at.Float[at.Array, "b h emb"],
        ) -> tuple[_model.Actions, _model.Actions]:
            _, v_t = self._run_action_suffix_prepared_from_state_token(
                state_token,
                x_t,
                time_tokens=time_tokens,
                full_attn_mask=suffix_context["full_attn_mask"],
                positions=suffix_context["positions"],
                kv_cache=kv_cache,
            )
            next_x_t = x_t + euler_dt * v_t
            return next_x_t, v_t

        final_actions, v_step = jax.lax.scan(
            _scan_step,
            noise,
            xs=time_tokens_table,
            unroll=2,
        )
        final_time_emb = self._compute_action_time_embedding(jnp.zeros((batch_size,), dtype=noise.dtype))
        final_time_tokens = einops.repeat(final_time_emb, "b emb -> b s emb", s=self.action_horizon)
        action_latents, _ = self._run_action_suffix_prepared_from_state_token(
            state_token,
            final_actions,
            time_tokens=final_time_tokens,
            full_attn_mask=suffix_context["full_attn_mask"],
            positions=suffix_context["positions"],
            kv_cache=kv_cache,
        )
        return {
            "actions": final_actions,
            "fm_tensor": jnp.swapaxes(v_step, 0, 1),
            "x0": noise,
            "context_tokens": context_tokens,
            "context_mask": prefix_mask,
            "action_latents": action_latents,
        }

    def _compute_qha_outputs_from_rollout(
        self,
        observation: _model.Observation,
        rollout_rng: at.KeyArrayLike,
        *,
        num_steps: int = _DEFAULT_PI0_DENOISE_STEPS,
        prefix_mask: at.Bool[at.Array, "b p"] | None = None,
        context_tokens: at.Float[at.Array, "b p d"] | None = None,
        kv_cache: at.Array | None = None,
    ) -> dict[str, at.Array]:
        if self.qha is None:
            raise RuntimeError("QHA is disabled for this model.")
        rollout = self._build_qha_rollout(
            observation,
            rollout_rng,
            num_steps=num_steps,
            prefix_mask=prefix_mask,
            context_tokens=context_tokens,
            kv_cache=kv_cache,
        )
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

    def _compute_qha_training_outputs(
        self,
        observation: _model.Observation,
        rollout_rng: at.KeyArrayLike,
        *,
        stop_gradient_backbone: bool,
        compute_teacher: bool = True,
        num_steps: int = _DEFAULT_PI0_DENOISE_STEPS,
        prefix_mask: at.Bool[at.Array, "b p"] | None = None,
        context_tokens: at.Float[at.Array, "b p d"] | None = None,
        kv_cache: at.Array | None = None,
    ) -> dict[str, at.Array]:
        if self.qha is None:
            raise RuntimeError("QHA is disabled for this model.")

        rollout = self._build_qha_rollout(
            observation,
            rollout_rng,
            num_steps=num_steps,
            prefix_mask=prefix_mask,
            context_tokens=context_tokens,
            kv_cache=kv_cache,
        )
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
        outputs.update({
            "qha_selector_q_intra": selector_outputs["q_intra"],
            "qha_selector_p_inter": selector_outputs["p_inter"],
            "qha_selector_q_mix": selector_outputs["q_mix"],
            "qha_selector_z_intra_raw": selector_outputs["z_intra_raw"],
            "qha_selector_z_intra_norm": selector_outputs["z_intra_norm"],
            "qha_selector_u_inter_raw": selector_outputs["u_inter_raw"],
            "qha_selector_u_inter_norm": selector_outputs["u_inter_norm"],
            "qha_teacher_posterior": jax.lax.stop_gradient(selector_outputs["teacher_posterior"]),
            "qha_teacher_expected_horizon": jax.lax.stop_gradient(selector_outputs["teacher_expected_horizon"]),
            "qha_teacher_argmax_horizon": jax.lax.stop_gradient(selector_outputs["teacher_argmax_horizon"]),
        })
        return outputs

    def _compute_qha_teacher_outputs_from_rollout(
        self,
        observation: _model.Observation,
        rollout_rng: at.KeyArrayLike,
        *,
        num_steps: int = _DEFAULT_PI0_DENOISE_STEPS,
    ) -> dict[str, at.Array]:
        rollout = self._build_qha_rollout(
            observation,
            rollout_rng,
            num_steps=num_steps,
        )
        hist_actions, hist_mask = self._get_hist_actions_and_mask(observation)
        selector_outputs = _qha.selector_scores_from_rollout(
            rollout["fm_tensor"],
            rollout["actions"],
            hist_actions,
            hist_mask,
            self._qha_selector_config(),
        )
        return {
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

    def _compute_qha_supervision_outputs_from_rollout(
        self,
        observation: _model.Observation,
        rollout_rng: at.KeyArrayLike,
        *,
        num_steps: int = _DEFAULT_PI0_DENOISE_STEPS,
    ) -> dict[str, at.Array]:
        """Build frozen-backbone QHA features plus selector supervision labels."""
        rollout = self._build_qha_rollout(
            observation,
            rollout_rng,
            num_steps=num_steps,
        )
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
        observation = _model.preprocess_observation(None, observation, train=False)
        outputs = self._compute_qha_teacher_outputs_from_rollout(
            observation,
            rng,
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
    ) -> dict[str, at.Array]:
        observation = _model.preprocess_observation(None, observation, train=False)
        steps = self._default_num_steps(num_steps)
        if self.qha is None:
            return {
                "actions": self._build_qha_rollout(observation, rng, num_steps=steps)["actions"],
            }
        return self._compute_qha_outputs_from_rollout(observation, rng, num_steps=steps)

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
        preprocess_rng, noise_rng, flow_t_rng, qha_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        if self.qha is not None and self.config.qha_train_mode == "qha_only":
            flow_loss = jnp.zeros(actions.shape[:-1], dtype=actions.dtype)
            if external_qha_features is None:
                prefix_mask, context_tokens, kv_cache = self._prepare_prefix_context_cache(observation)
                qha_outputs = self._compute_qha_training_outputs(
                    observation,
                    qha_rng,
                    stop_gradient_backbone=True,
                    compute_teacher=external_qha_teacher is None,
                    prefix_mask=prefix_mask,
                    context_tokens=context_tokens,
                    kv_cache=kv_cache,
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
                    ))
                metrics["qha_teacher_expected_horizon"] = teacher_expected_horizon
                metrics["qha_teacher_argmax_horizon"] = teacher_argmax_horizon
            else:
                metrics.update(
                    self._empty_qha_alignment_metrics(
                        qha_outputs["qha_posterior"].shape[0],
                        qha_outputs["qha_posterior"].dtype,
                        fill_nan=True,
                    ))
            return total_loss, metrics

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        flow_t = jax.random.beta(flow_t_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        flow_t_expanded = flow_t[..., None, None]
        x_t = flow_t_expanded * noise + (1 - flow_t_expanded) * actions
        u_t = noise - actions

        prefix_mask = None
        context_tokens = None
        kv_cache = None
        if self.qha is not None:
            prefix_mask, context_tokens, kv_cache = self._prepare_prefix_context_cache(observation)
            v_t = self._predict_velocity_from_state_with_prefix_cache(
                observation.state,
                x_t,
                flow_t,
                prefix_mask=prefix_mask,
                kv_cache=kv_cache,
            )
        else:
            v_t = self._predict_velocity(observation, x_t, flow_t)
        flow_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
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
                ))
            metrics["total_loss"] = metrics["flow_loss"]
            return flow_loss, metrics

        qha_outputs = self._compute_qha_training_outputs(
            observation,
            qha_rng,
            stop_gradient_backbone=False,
            prefix_mask=prefix_mask,
            context_tokens=context_tokens,
            kv_cache=kv_cache,
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
                ))
        else:
            metrics.update(
                self._empty_qha_alignment_metrics(
                    qha_outputs["qha_posterior"].shape[0],
                    qha_outputs["qha_posterior"].dtype,
                    fill_nan=True,
                ))
        return total_loss, metrics

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        loss, _ = self.compute_loss_and_metrics(rng, observation, actions, train=train)
        return loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        return self._build_qha_rollout(
            observation,
            rng,
            num_steps=self._default_num_steps(num_steps),
        )["actions"]

    @at.typecheck
    def sample_actions_with_trace(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] | None = None,
    ) -> tuple[_model.Actions, dict[str, at.Array]]:
        """Sample actions while preserving the local denoise-trace eval interface."""
        observation = _model.preprocess_observation(None, observation, train=False)
        steps = self._default_num_steps(num_steps)
        rollout = self._build_qha_rollout(observation, rng, num_steps=steps)
        trace = {
            "v": rollout["fm_tensor"],
            "ds": jnp.full(
                (observation.state.shape[0], steps),
                jnp.asarray(-1.0 / steps, dtype=rollout["actions"].dtype),
                dtype=rollout["actions"].dtype,
            ),
            "x0": rollout["x0"],
            "xt_internal": rollout["actions"],
        }
        return rollout["actions"], trace
