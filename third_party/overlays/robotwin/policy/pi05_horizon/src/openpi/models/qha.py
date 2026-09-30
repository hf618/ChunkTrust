import dataclasses
import math
from collections.abc import Sequence
from typing import Any

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.shared.array_typing as at


DEFAULT_QHA_SPARSE_CANDIDATE_HORIZONS: tuple[int, ...] = (10, 20, 30, 40, 50)


def _safe_log(x: at.Array, eps: float) -> at.Array:
    return jnp.log(jnp.clip(x, eps, None))


def kl_or_soft_ce(
    pred: at.Float[at.Array, "b s"],
    target: at.Float[at.Array, "b s"],
    *,
    loss_type: str = "KL",
    eps: float = 1e-6,
) -> at.Float[at.Array, "b"]:
    p = jnp.clip(pred, eps, 1.0)
    q = jnp.clip(target, eps, 1.0)
    if loss_type.upper() == "KL":
        return jnp.sum(q * (_safe_log(q, eps) - _safe_log(p, eps)), axis=-1)
    return -jnp.sum(q * _safe_log(p, eps), axis=-1)


def parse_candidate_horizons(raw: str | Sequence[int] | None, *, max_horizon: int | None = None) -> tuple[int, ...]:
    if raw is None:
        values: list[int] = list(DEFAULT_QHA_SPARSE_CANDIDATE_HORIZONS)
    elif isinstance(raw, str):
        values = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            values.append(int(token))
    else:
        values = [int(v) for v in raw]
    values = [v for v in values if v > 0]
    if max_horizon is not None:
        values = [v for v in values if v <= int(max_horizon)]
    if not values:
        fallback = int(max_horizon) if max_horizon is not None else 1
        values = [max(1, fallback)]
    return tuple(sorted(dict.fromkeys(values)))


def default_sparse_runtime_candidate_horizons(*, max_horizon: int) -> tuple[int, ...]:
    return parse_candidate_horizons(DEFAULT_QHA_SPARSE_CANDIDATE_HORIZONS, max_horizon=max_horizon)


def resolve_qha_candidate_horizons(
    raw: str | Sequence[int] | None,
    *,
    candidate_mode: str = "sparse",
    max_horizon: int,
) -> tuple[int, ...]:
    mode = str(candidate_mode).strip().lower()
    if mode == "sparse":
        return parse_candidate_horizons(raw, max_horizon=max_horizon)
    if mode == "dense_full":
        return tuple(range(1, int(max_horizon) + 1))
    raise ValueError(f"Unsupported qha candidate_mode: {candidate_mode}")


def normalize_score_distribution(
    score_arr: at.Array,
    *,
    temp: float = 1.0,
    eps: float = 1e-12,
) -> at.Array:
    arr = jnp.maximum(jnp.asarray(score_arr, dtype=jnp.float32), 0.0)
    arr_sum = jnp.sum(arr, axis=-1, keepdims=True)
    base = jnp.where(arr_sum > eps, arr / jnp.clip(arr_sum, eps, None), 0.0)
    if abs(float(temp) - 1.0) <= 1e-12:
        uniform = jnp.full_like(base, 1.0 / float(max(base.shape[-1], 1)))
        return jnp.where(arr_sum > eps, base, uniform)
    sharp = jnp.power(jnp.clip(base, 0.0, 1.0), 1.0 / float(max(temp, 1e-6)))
    sharp_sum = jnp.sum(sharp, axis=-1, keepdims=True)
    uniform = jnp.full_like(sharp, 1.0 / float(max(sharp.shape[-1], 1)))
    return jnp.where(sharp_sum > eps, sharp / jnp.clip(sharp_sum, eps, None), uniform)


def _sinusoidal_positional_encoding(length: int, dim: int, *, dtype=jnp.float32) -> at.Float[at.Array, "l d"]:
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    positions = jnp.arange(length, dtype=dtype)[:, None]
    half = (dim + 1) // 2
    inv_freq = jnp.exp(-jnp.log(10000.0) * jnp.arange(half, dtype=dtype) / jnp.maximum(1, half - 1))
    sinusoid = positions * inv_freq[None, :]
    emb = jnp.concatenate([jnp.sin(sinusoid), jnp.cos(sinusoid)], axis=-1)
    return emb[:, :dim]


class _TinySelfAttention(nnx.Module):

    def __init__(self, d_model: int, num_heads: int, *, dropout: float = 0.0, rngs: nnx.Rngs):
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.k_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.v_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs) if dropout > 0 else None

    def __call__(
        self,
        x: at.Float[at.Array, "b n d"],
        mask: at.Bool[at.Array, "b n"] | None = None,
    ) -> at.Float[at.Array, "b n d"]:
        q = einops.rearrange(self.q_proj(x), "b n (h d) -> b h n d", h=self.num_heads)
        k = einops.rearrange(self.k_proj(x), "b n (h d) -> b h n d", h=self.num_heads)
        v = einops.rearrange(self.v_proj(x), "b n (h d) -> b h n d", h=self.num_heads)
        scale = 1.0 / jnp.sqrt(float(self.head_dim))
        attn = jnp.einsum("bhid,bhjd->bhij", q, k) * scale
        if mask is not None:
            key_mask = mask[:, None, None, :]
            attn = jnp.where(key_mask, attn, jnp.finfo(attn.dtype).min)
        attn = jax.nn.softmax(attn, axis=-1)
        if mask is not None:
            query_mask = mask[:, None, :, None]
            attn = attn * query_mask
        out = jnp.einsum("bhij,bhjd->bhid", attn, v)
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        out = self.out_proj(out)
        if self.dropout is not None:
            out = self.dropout(out)
        if mask is not None:
            out = out * mask[..., None]
        return out


class _TinyCrossAttention(nnx.Module):

    def __init__(self, d_model: int, num_heads: int, *, dropout: float = 0.0, rngs: nnx.Rngs):
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.k_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.v_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs) if dropout > 0 else None

    def __call__(
        self,
        query: at.Float[at.Array, "b nq d"],
        kv: at.Float[at.Array, "b nk d"],
        kv_mask: at.Bool[at.Array, "b nk"] | None = None,
    ) -> at.Float[at.Array, "b nq d"]:
        q = einops.rearrange(self.q_proj(query), "b n (h d) -> b h n d", h=self.num_heads)
        k = einops.rearrange(self.k_proj(kv), "b n (h d) -> b h n d", h=self.num_heads)
        v = einops.rearrange(self.v_proj(kv), "b n (h d) -> b h n d", h=self.num_heads)
        scale = 1.0 / jnp.sqrt(float(self.head_dim))
        attn = jnp.einsum("bhid,bhjd->bhij", q, k) * scale
        if kv_mask is not None:
            key_mask = kv_mask[:, None, None, :]
            attn = jnp.where(key_mask, attn, jnp.finfo(attn.dtype).min)
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum("bhij,bhjd->bhid", attn, v)
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        out = self.out_proj(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class _TinyMLP(nnx.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        rngs: nnx.Rngs,
    ):
        hidden = hidden_dim or in_dim
        self.fc1 = nnx.Linear(in_dim, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, out_dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs) if dropout > 0 else None

    def __call__(self, x: at.Array) -> at.Array:
        x = nnx.swish(self.fc1(x))
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc2(x)


@dataclasses.dataclass(frozen=True)
class QHAConfig:
    candidate_horizons: tuple[int, ...]
    d_model: int = 256
    num_queries: int = 8
    num_heads: int = 8
    num_bridge_layers: int = 1
    use_fusion_gate: bool = True
    dropout: float = 0.0
    loss_type: str = "KL"
    eps: float = 1e-6


class QueryBasedHorizonAdapter(nnx.Module):

    def __init__(
        self,
        context_dim: int,
        action_dim: int,
        config: QHAConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.context_in = _TinyMLP(context_dim, config.d_model, hidden_dim=config.d_model, dropout=config.dropout, rngs=rngs)
        self.action_in = _TinyMLP(action_dim, config.d_model, hidden_dim=config.d_model, dropout=config.dropout, rngs=rngs)
        self.query_in = nnx.Linear(config.num_queries, config.d_model, rngs=rngs)

        ctx_cross_names: list[str] = []
        act_cross_names: list[str] = []
        refine_names: list[str] = []
        for i in range(config.num_bridge_layers):
            ctx_name = f"context_cross_{i}"
            act_name = f"action_cross_{i}"
            refine_name = f"refine_{i}"
            setattr(self, ctx_name, _TinyCrossAttention(config.d_model, config.num_heads, dropout=config.dropout, rngs=rngs))
            setattr(self, act_name, _TinyCrossAttention(config.d_model, config.num_heads, dropout=config.dropout, rngs=rngs))
            setattr(self, refine_name, _TinySelfAttention(config.d_model, config.num_heads, dropout=config.dropout, rngs=rngs))
            ctx_cross_names.append(ctx_name)
            act_cross_names.append(act_name)
            refine_names.append(refine_name)
        self._context_cross_names = tuple(ctx_cross_names)
        self._action_cross_names = tuple(act_cross_names)
        self._refine_names = tuple(refine_names)

        self.gate_mlp = _TinyMLP(config.d_model, config.d_model, hidden_dim=config.d_model, dropout=config.dropout, rngs=rngs)
        self.out_mlp = _TinyMLP(config.d_model, len(config.candidate_horizons), hidden_dim=config.d_model, dropout=config.dropout, rngs=rngs)

    def __call__(
        self,
        context_tokens: at.Float[at.Array, "b n dc"],
        action_latents: at.Float[at.Array, "b h da"],
        *,
        context_mask: at.Bool[at.Array, "b n"] | None = None,
    ) -> dict[str, at.Array]:
        b = context_tokens.shape[0]
        context = self.context_in(context_tokens)
        action = self.action_in(action_latents)
        context = context + _sinusoidal_positional_encoding(context.shape[1], self.config.d_model, dtype=context.dtype)[None, :, :]
        action = action + _sinusoidal_positional_encoding(action.shape[1], self.config.d_model, dtype=action.dtype)[None, :, :]

        eye = jnp.eye(self.config.num_queries, dtype=context.dtype)
        queries = self.query_in(eye)[None, :, :]
        queries = jnp.broadcast_to(queries, (b, self.config.num_queries, self.config.d_model))

        stage1 = queries
        stage2 = queries
        for ctx_name, act_name, refine_name in zip(
            self._context_cross_names,
            self._action_cross_names,
            self._refine_names,
            strict=True,
        ):
            stage1 = stage1 + getattr(self, ctx_name)(stage1, context, kv_mask=context_mask)
            stage2 = stage1 + getattr(self, act_name)(stage1, action)
            stage2 = stage2 + getattr(self, refine_name)(stage2)
            stage1 = stage2

        pooled_action = jnp.mean(stage2, axis=1)
        pooled_context = jnp.mean(stage1, axis=1)
        if self.config.use_fusion_gate:
            gate = jax.nn.sigmoid(self.gate_mlp(pooled_action))
            fused = gate * pooled_action + (1.0 - gate) * pooled_context
        else:
            fused = pooled_action
        logits = self.out_mlp(fused)
        posterior = jax.nn.softmax(logits, axis=-1)
        horizons = jnp.asarray(self.config.candidate_horizons, dtype=posterior.dtype)
        expected = jnp.sum(posterior * horizons[None, :], axis=-1)
        argmax = horizons[jnp.argmax(posterior, axis=-1)]
        return {
            "qha_logits": logits,
            "qha_posterior": posterior,
            "qha_expected_horizon": expected,
            "qha_argmax_horizon": argmax.astype(jnp.int32),
            "qha_candidate_horizons": horizons.astype(jnp.int32),
        }


@dataclasses.dataclass(frozen=True)
class SelectorConfig:
    candidate_horizons: tuple[int, ...]
    exec_mode: str = "expected_round"
    expected_temp: float = 1.0
    intra_cut_t: float = 0.25
    intra_alpha: float = 4.0
    intra_z_mode: str = "window_rms"
    intra_tau_window: int = 3
    mix_mode: str = "both"
    inter_use_speed_uniformity: bool = True
    inter_window: int = 60
    inter_weight: float = 0.5
    inter_beta: float = 4.0
    inter_fallback: str = "ehh_only"
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.exec_mode not in ("expected_round", "thompson"):
            raise ValueError(f"Unsupported selector exec_mode: {self.exec_mode}")
        if self.intra_z_mode not in ("tau0_rms", "window_rms", "trend"):
            raise ValueError(f"Unsupported selector intra_z_mode: {self.intra_z_mode}")
        if self.mix_mode not in ("intra", "inter", "both"):
            raise ValueError(f"Unsupported selector mix_mode: {self.mix_mode}")
        if self.inter_fallback not in ("ehh_only", "neutral"):
            raise ValueError(f"Unsupported selector inter_fallback: {self.inter_fallback}")


def _normalize_candidate_metric(values: at.Array, eps: float = 1e-12) -> at.Array:
    arr = jnp.asarray(values, dtype=jnp.float32)
    arr_min = jnp.min(arr, axis=-1, keepdims=True)
    arr_max = jnp.max(arr, axis=-1, keepdims=True)
    span = arr_max - arr_min
    out = jnp.where(span > eps, (arr - arr_min) / (span + eps), jnp.zeros_like(arr))
    if arr.ndim == 1:
        return out.reshape(arr.shape)
    return out


def _compute_ehh_curve(
    v_step_prefix: at.Array,
    *,
    h_full: int,
    cut_t: float,
    eps: float,
) -> at.Array:
    hs = int(v_step_prefix.shape[1])
    da = int(v_step_prefix.shape[2])
    pad_h = max(int(h_full), hs)
    if pad_h == hs:
        mat_pad = jnp.asarray(v_step_prefix, dtype=jnp.float32)
    else:
        mat_pad = jnp.pad(jnp.asarray(v_step_prefix, dtype=jnp.float32), ((0, 0), (0, pad_h - hs), (0, 0)))
    spec = jnp.fft.rfft(mat_pad, axis=1)
    pwr = (jnp.real(spec) ** 2) + (jnp.imag(spec) ** 2)
    pwr_f = jnp.mean(pwr, axis=2)
    f_t = int(pwr_f.shape[1])
    ft_c = max(1, int(cut_t * f_t))
    e_low = jnp.sum(pwr_f[:, :ft_c], axis=1)
    e_high = jnp.sum(pwr_f[:, ft_c:], axis=1)
    return e_high / (e_low + e_high + eps)


def _z_trend_abs_slope(y: at.Array) -> at.Array:
    n = int(y.shape[-1])
    if n <= 1:
        return jnp.zeros(y.shape[:-1], dtype=jnp.float32)
    x = jnp.arange(n, dtype=jnp.float32)
    x = x - jnp.mean(x)
    y0 = y - jnp.mean(y, axis=-1, keepdims=True)
    denom = jnp.sum(x * x)
    slope = jnp.where(denom > 1e-12, jnp.sum(x * y0, axis=-1) / denom, 0.0)
    return jnp.abs(slope)


def _compute_z_from_ehh(
    e_vals: at.Array,
    *,
    mode: str,
    tau_window: int,
) -> at.Array:
    tail = jnp.asarray(e_vals, dtype=jnp.float32)
    if int(tail.shape[-1]) <= 1:
        return jnp.zeros(tail.shape[:-1], dtype=jnp.float32)
    if mode == "trend":
        return _z_trend_abs_slope(tail)
    if mode == "tau0_rms":
        baseline = tail[..., :1]
    else:
        hi = min(int(tail.shape[-1]), max(1, int(tau_window)))
        baseline = jnp.mean(tail[..., :hi], axis=-1, keepdims=True)
    dev = tail - baseline
    return jnp.sqrt(jnp.mean(dev * dev, axis=-1))


def _candidate_prefix_masks(
    horizons: Sequence[int],
    *,
    max_horizon: int,
) -> at.Bool[at.Array, "k h"]:
    horizon_arr = jnp.asarray(horizons, dtype=jnp.int32)
    pos = jnp.arange(max_horizon, dtype=jnp.int32)[None, :]
    return pos < horizon_arr[:, None]


def _compute_q_intra_batched(
    fm_tensor: at.Float[at.Array, "b t h d"],
    horizons: Sequence[int],
    *,
    h_full: int,
    cut_t: float,
    intra_alpha: float,
    intra_z_mode: str,
    intra_tau_window: int,
    eps: float,
) -> tuple[at.Array, at.Array, at.Array]:
    candidate_masks = _candidate_prefix_masks(horizons, max_horizon=h_full)
    masked = jnp.asarray(fm_tensor, dtype=jnp.float32)[:, None, :, :, :] * candidate_masks[None, :, None, :, None]
    spec = jnp.fft.rfft(masked, axis=3)
    pwr = (jnp.real(spec) ** 2) + (jnp.imag(spec) ** 2)
    pwr_f = jnp.mean(pwr, axis=-1)
    f_t = int(pwr_f.shape[-1])
    ft_c = max(1, int(cut_t * f_t))
    e_low = jnp.sum(pwr_f[..., :ft_c], axis=-1)
    e_high = jnp.sum(pwr_f[..., ft_c:], axis=-1)
    e_vals = e_high / (e_low + e_high + eps)
    z_raw = _compute_z_from_ehh(e_vals, mode=intra_z_mode, tau_window=intra_tau_window)
    z_norm = _normalize_candidate_metric(z_raw, eps=eps)
    q_intra = jnp.exp(-float(intra_alpha) * jnp.maximum(0.0, z_norm))
    return z_raw, z_norm, q_intra


def _speed_uniformity_proxy_batched(
    history_actions: at.Array,
    history_mask: at.Array,
    xt_step: at.Array,
    horizons: Sequence[int],
    *,
    inter_window: int,
    inter_beta: float,
    eps: float,
) -> tuple[at.Array, at.Array, at.Array]:
    history_actions = jnp.asarray(history_actions, dtype=jnp.float32)
    history_mask = jnp.asarray(history_mask, dtype=jnp.bool_)
    xt_step = jnp.asarray(xt_step, dtype=jnp.float32)
    h = int(xt_step.shape[1])
    w = int(inter_window)
    horizon_arr = jnp.asarray(horizons, dtype=jnp.int32)
    pre = jnp.maximum(0, w - horizon_arr)
    post = horizon_arr
    valid_candidate = (horizon_arr >= 1) & (horizon_arr <= h) & (pre >= 1) & (post >= 1) & (h >= post)

    if history_actions.shape[1] >= w:
        hist_window = history_actions[:, -w:, :]
        hist_window_mask = history_mask[:, -w:]
    else:
        pad = w - int(history_actions.shape[1])
        hist_window = jnp.pad(history_actions, ((0, 0), (pad, 0), (0, 0)))
        hist_window_mask = jnp.pad(history_mask, ((0, 0), (pad, 0)), constant_values=False)

    if xt_step.shape[1] >= w:
        xt_window = xt_step[:, :w, :]
    else:
        pad = w - int(xt_step.shape[1])
        xt_window = jnp.pad(xt_step, ((0, 0), (0, pad), (0, 0)))

    pos = jnp.arange(w, dtype=jnp.int32)[None, :]
    hist_src_idx = jnp.clip((w - pre)[:, None] + pos, 0, max(w - 1, 0))
    hist_tokens = jnp.take_along_axis(
        hist_window[:, None, :, :],
        hist_src_idx[None, :, :, None],
        axis=2,
    )
    hist_token_mask = jnp.take_along_axis(
        hist_window_mask[:, None, :],
        hist_src_idx[None, :, :],
        axis=2,
    )

    xt_src_idx = jnp.clip(pos - pre[:, None], 0, max(w - 1, 0))
    xt_tokens = jnp.take_along_axis(
        xt_window[:, None, :, :],
        xt_src_idx[None, :, :, None],
        axis=2,
    )
    use_hist = pos < pre[:, None]
    seq = jnp.where(use_hist[None, :, :, None], hist_tokens, xt_tokens)
    speed = jnp.linalg.norm(jnp.diff(seq, axis=2), axis=-1)
    count = max(1, int(speed.shape[-1]))
    mu = jnp.sum(speed, axis=-1) / float(count)
    var = jnp.maximum(0.0, jnp.mean(speed * speed, axis=-1) - (mu * mu))
    cv = jnp.where(mu > eps, jnp.sqrt(var) / (mu + eps), 0.0)
    valid_history = jnp.all(jnp.where(use_hist[None, :, :], hist_token_mask, True), axis=-1)
    valid = jnp.logical_and(valid_history, valid_candidate[None, :])
    u_raw = jnp.where(valid, cv, jnp.full_like(cv, jnp.nan))
    u_norm = _normalize_candidate_metric(u_raw, eps=eps)
    p_proxy = jnp.exp(-float(inter_beta) * jnp.maximum(0.0, u_norm))
    return u_raw, u_norm, p_proxy


def _speed_uniformity_proxy_single(
    history_actions: at.Array,
    history_mask: at.Array,
    xt_step: at.Array,
    horizons: Sequence[int],
    *,
    inter_window: int,
    inter_beta: float,
    eps: float,
) -> tuple[at.Array, at.Array, at.Array]:
    history_actions = jnp.asarray(history_actions, dtype=jnp.float32)
    history_mask = jnp.asarray(history_mask, dtype=jnp.bool_)
    xt_step = jnp.asarray(xt_step, dtype=jnp.float32)
    h = int(xt_step.shape[0])
    u_raw_vals: list[at.Array] = []
    for s in horizons:
        s = int(s)
        pre = max(0, int(inter_window - s))
        post = int(s)
        if s < 1 or s > h or pre < 1 or post < 1 or h < post:
            u_raw_vals.append(jnp.asarray(jnp.nan, dtype=jnp.float32))
            continue
        hist_tail = history_actions[-pre:, :]
        hist_tail_mask = history_mask[-pre:]
        xt_prefix = xt_step[:post, :]
        seq = jnp.concatenate([hist_tail, xt_prefix], axis=0)
        speed = jnp.linalg.norm(jnp.diff(seq, axis=0), axis=1)
        count = max(1, int(speed.shape[0]))
        mu = jnp.sum(speed) / float(count)
        var = jnp.maximum(0.0, jnp.mean(speed * speed) - (mu * mu))
        cv = jnp.where(mu > eps, jnp.sqrt(var) / (mu + eps), 0.0)
        valid_history = jnp.all(hist_tail_mask)
        u_raw_vals.append(jnp.where(valid_history, cv, jnp.asarray(jnp.nan, dtype=jnp.float32)))
    u_raw = jnp.stack(u_raw_vals, axis=0)
    u_norm = _normalize_candidate_metric(u_raw, eps=eps)
    p_proxy = jnp.exp(-float(inter_beta) * jnp.maximum(0.0, u_norm))
    return u_raw, u_norm, p_proxy


def selector_scores_from_rollout(
    fm_tensor: at.Float[at.Array, "b t h d"],
    final_actions: at.Float[at.Array, "b h d"],
    hist_actions: at.Float[at.Array, "b l d"],
    hist_mask: at.Bool[at.Array, "b l"],
    config: SelectorConfig,
) -> dict[str, at.Array]:
    horizons = tuple(int(v) for v in config.candidate_horizons if int(v) <= int(final_actions.shape[1]))
    if not horizons:
        horizons = (int(final_actions.shape[1]),)
    horizon_arr = jnp.asarray(horizons, dtype=jnp.int32)
    z_raw, z_norm, q_intra = _compute_q_intra_batched(
        fm_tensor,
        horizons,
        h_full=int(final_actions.shape[1]),
        cut_t=config.intra_cut_t,
        intra_alpha=config.intra_alpha,
        intra_z_mode=config.intra_z_mode,
        intra_tau_window=config.intra_tau_window,
        eps=config.eps,
    )

    if config.inter_use_speed_uniformity and config.mix_mode in ("inter", "both"):
        u_raw, u_norm, p_inter = _speed_uniformity_proxy_batched(
            hist_actions,
            hist_mask,
            final_actions,
            horizons,
            inter_window=config.inter_window,
            inter_beta=config.inter_beta,
            eps=config.eps,
        )
    else:
        batch_size = int(fm_tensor.shape[0])
        shape = (batch_size, len(horizons))
        u_raw = jnp.full(shape, jnp.nan, dtype=jnp.float32)
        u_norm = jnp.full(shape, jnp.nan, dtype=jnp.float32)
        p_inter = jnp.full(shape, jnp.nan, dtype=jnp.float32)

    if config.mix_mode == "intra":
        q_mix = q_intra
    elif config.mix_mode == "inter":
        p_clipped = jnp.clip(p_inter, 0.0, 1.0)
        fallback = jnp.full_like(q_intra, 0.5 if config.inter_fallback == "neutral" else 0.0)
        if config.inter_fallback != "neutral":
            fallback = q_intra
        q_mix = jnp.where(jnp.isfinite(p_inter), p_clipped, fallback)
    else:
        p_clipped = jnp.clip(p_inter, 0.0, 1.0)
        mixed = (1.0 - float(config.inter_weight)) * q_intra + float(config.inter_weight) * p_clipped
        if config.inter_fallback == "neutral":
            fallback = (1.0 - float(config.inter_weight)) * q_intra + float(config.inter_weight) * 0.5
        else:
            fallback = q_intra
        q_mix = jnp.where(jnp.isfinite(p_inter), mixed, fallback)

    teacher = normalize_score_distribution(
        q_mix,
        temp=config.expected_temp if config.exec_mode == "expected_round" else 1.0,
    )
    expected = jnp.sum(teacher * horizon_arr.astype(teacher.dtype)[None, :], axis=-1)
    argmax = horizon_arr[jnp.argmax(teacher, axis=-1)]

    return {
        "candidate_horizons": horizon_arr,
        "q_intra": q_intra,
        "p_inter": p_inter,
        "q_mix": q_mix,
        "z_intra_raw": z_raw,
        "z_intra_norm": z_norm,
        "u_inter_raw": u_raw,
        "u_inter_norm": u_norm,
        "teacher_posterior": teacher,
        "teacher_expected_horizon": expected,
        "teacher_argmax_horizon": argmax.astype(jnp.int32),
    }


def _normalize_score_distribution_np(score_arr: np.ndarray, temp: float = 1.0) -> np.ndarray:
    arr = np.asarray(score_arr, dtype=np.float64)
    arr = np.maximum(arr, 0.0)
    arr_sum = float(np.sum(arr))
    if arr_sum <= 1e-12 or (not np.isfinite(arr_sum)):
        return np.full((arr.size,), 1.0 / float(max(arr.size, 1)), dtype=np.float64)
    base = arr / arr_sum
    if abs(float(temp) - 1.0) <= 1e-12:
        return base
    sharp = np.power(np.clip(base, 0.0, 1.0), 1.0 / float(max(1e-6, temp)))
    sharp_sum = float(np.sum(sharp))
    if sharp_sum <= 1e-12 or (not np.isfinite(sharp_sum)):
        return np.full((arr.size,), 1.0 / float(max(arr.size, 1)), dtype=np.float64)
    return sharp / sharp_sum


class HybridQHARuntimeSelector:

    def __init__(
        self,
        *,
        candidate_horizons: Sequence[int],
        exec_mode: str = "expected_round",
        expected_temp: float = 1.0,
        ts_epsilon: float = 0.05,
        ts_seed: int = 0,
        ts_update_mode: str = "kernel_forget",
        ts_kernel_bandwidth: float = 10.0,
        ts_forget_rho: float = 0.99,
        ts_update_eta: float = 1.0,
        prior_gamma: float = 1.0,
    ):
        self.candidate_horizons = tuple(int(v) for v in candidate_horizons)
        self.exec_mode = str(exec_mode).strip().lower()
        self.expected_temp = float(max(expected_temp, 1e-6))
        self.ts_epsilon = float(np.clip(ts_epsilon, 0.0, 1.0))
        self.ts_update_mode = str(ts_update_mode).strip().lower()
        self.ts_kernel_bandwidth = float(max(ts_kernel_bandwidth, 1e-6))
        self.ts_forget_rho = float(np.clip(ts_forget_rho, 1e-6, 1.0))
        self.ts_update_eta = float(max(ts_update_eta, 0.0))
        self.prior_gamma = float(max(prior_gamma, 0.0))
        self._seed = int(ts_seed)
        self._rng = np.random.default_rng(self._seed)
        self._state = {int(s): [1.0, 1.0] for s in self.candidate_horizons}

    def reset(self, seed: int | None = None) -> None:
        for key in self._state:
            self._state[key] = [1.0, 1.0]
        episode_seed = self._seed if seed is None else int(seed)
        self._rng = np.random.default_rng(np.random.SeedSequence([self._seed, episode_seed]))

    def _gaussian_kernel_weights(self, center: float) -> np.ndarray:
        vals = np.asarray(self.candidate_horizons, dtype=np.float64)
        weights = np.exp(-((vals - float(center)) ** 2) / (2.0 * self.ts_kernel_bandwidth * self.ts_kernel_bandwidth))
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-12 or (not np.isfinite(weight_sum)):
            return np.full((vals.size,), 1.0 / float(max(vals.size, 1)), dtype=np.float64)
        return weights / weight_sum

    def _apply_update(
        self,
        effective_scores: np.ndarray,
        *,
        chosen_idx: int,
        k_exec: int,
        expected_prob: np.ndarray | None = None,
    ) -> float:
        if self.ts_update_mode == "independent":
            if expected_prob is not None:
                for i, s in enumerate(self.candidate_horizons):
                    prob_i = float(expected_prob[i]) if i < expected_prob.size else 0.0
                    if prob_i <= 0.0:
                        continue
                    qa_i = float(effective_scores[i])
                    a, b = self._state[int(s)]
                    self._state[int(s)] = [a + (prob_i * qa_i), b + (prob_i * (1.0 - qa_i))]
                return float(np.dot(expected_prob, effective_scores))
            qa = float(effective_scores[chosen_idx])
            a, b = self._state[int(k_exec)]
            self._state[int(k_exec)] = [a + qa, b + (1.0 - qa)]
            return qa

        feedback = float(np.dot(expected_prob, effective_scores)) if expected_prob is not None else float(effective_scores[chosen_idx])
        feedback = float(np.clip(feedback, 0.0, 1.0))
        weights = self._gaussian_kernel_weights(float(k_exec))
        for i, s in enumerate(self.candidate_horizons):
            a, b = self._state[int(s)]
            incr = self.ts_update_eta * float(weights[i])
            self._state[int(s)] = [
                self.ts_forget_rho * a + incr * feedback,
                self.ts_forget_rho * b + incr * (1.0 - feedback),
            ]
        return feedback

    def select(self, posterior: np.ndarray, q_mix: np.ndarray, *, max_exec_length: int) -> dict[str, Any]:
        posterior = np.asarray(posterior, dtype=np.float64).reshape(-1)
        q_mix = np.asarray(q_mix, dtype=np.float64).reshape(-1)
        if posterior.shape != q_mix.shape:
            raise ValueError(f"posterior/q_mix shape mismatch: {posterior.shape} vs {q_mix.shape}")
        if posterior.shape[0] != len(self.candidate_horizons):
            raise ValueError(
                f"posterior length must match candidate_horizons. got {posterior.shape[0]} vs {len(self.candidate_horizons)}"
            )

        prior = np.power(np.clip(posterior, 1e-12, 1.0), self.prior_gamma)
        effective = np.clip(q_mix, 0.0, 1.0) * prior
        theta = np.asarray(
            [self._rng.beta(*self._state[int(s)]) for s in self.candidate_horizons],
            dtype=np.float64,
        )
        runtime_scores = theta * effective
        expected_prob = None

        if float(self._rng.random()) < self.ts_epsilon:
            chosen_idx = int(self._rng.integers(0, len(self.candidate_horizons)))
            k_exec = int(self.candidate_horizons[chosen_idx])
            choose_mode = "epsilon_random"
            feedback = self._apply_update(effective, chosen_idx=chosen_idx, k_exec=k_exec)
        elif self.exec_mode == "expected_round":
            expected_prob = _normalize_score_distribution_np(runtime_scores, temp=self.expected_temp)
            expected_horizon = float(np.dot(expected_prob, np.asarray(self.candidate_horizons, dtype=np.float64)))
            k_exec = int(np.clip(int(round(expected_horizon)), 1, int(max_exec_length)))
            choose_mode = "expected_round"
            chosen_idx = -1
            feedback = self._apply_update(effective, chosen_idx=chosen_idx, k_exec=k_exec, expected_prob=expected_prob)
        else:
            chosen_idx = int(np.argmax(runtime_scores))
            k_exec = int(self.candidate_horizons[chosen_idx])
            choose_mode = "thompson"
            feedback = self._apply_update(effective, chosen_idx=chosen_idx, k_exec=k_exec)

        return {
            "exec_length": int(np.clip(k_exec, 1, int(max_exec_length))),
            "choose_mode": choose_mode,
            "runtime_theta": theta.astype(np.float32),
            "runtime_scores": runtime_scores.astype(np.float32),
            "runtime_effective_scores": effective.astype(np.float32),
            "runtime_expected_prob": None if expected_prob is None else expected_prob.astype(np.float32),
            "runtime_feedback": float(feedback),
        }
