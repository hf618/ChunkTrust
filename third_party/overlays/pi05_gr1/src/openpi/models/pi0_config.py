import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.qha as _qha
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
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

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
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
                f"qha_d_model ({self.qha_d_model}) must be divisible by qha_num_heads ({self.qha_num_heads})."
            )
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
                "horizon_selector_exec_mode must be expected_round or thompson, "
                f"got {self.horizon_selector_exec_mode}."
            )
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
                f"horizon_selector_mix_mode must be intra, inter, or both. Got {self.horizon_selector_mix_mode}."
            )
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
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

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
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
