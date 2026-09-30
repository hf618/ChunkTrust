"""Inference-time RTC Eq. 2--5, arXiv:2506.07339v2, in OpenPI t=1->0 convention.

Local v is minus the paper's v. Clean estimate = x - t*v; integrate v - w*VJP.
The main AHS signal is the uncorrected network velocity along this guided path.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import einops
from openpi.models import model as base
from openpi.models.pi0 import make_attn_mask

def sample(self, observation, noise, target, weights, *, guided=False, num_steps=10):
    observation = base.preprocess_observation(None, observation, train=False)
    b = noise.shape[0]
    pi0 = hasattr(self, '_prepare_suffix_runtime_context')
    if pi0:
        prefix_mask, _, kv = self._prepare_prefix_context_cache(observation)
        context = self._prepare_suffix_runtime_context(prefix_mask, observation.state)
        from openpi.models.pi0 import build_rollout_time_emb_table, build_rollout_time_tokens_table
        emb = build_rollout_time_emb_table(batch_size=b, num_steps=num_steps,
                                          embedding_dim=self.action_in_proj.out_features, dtype=noise.dtype)
        time_tokens = build_rollout_time_tokens_table(emb, action_horizon=self.action_horizon)
    else:
        tok, prefix_mask, ar = self.embed_prefix(observation)
        _, kv = self.PaliGemma.llm([tok, None], mask=make_attn_mask(prefix_mask, ar),
                                  positions=jnp.cumsum(prefix_mask, axis=1) - 1)

    def velocity(x, t, step):
        if pi0:
            _, v = self._run_action_suffix_prepared_from_state_token(
                context['state_token'], x, time_tokens=time_tokens[step],
                full_attn_mask=context['full_attn_mask'], positions=context['positions'], kv_cache=kv)
            return v
        tok, mask, ar, cond = self.embed_suffix(observation, x, jnp.broadcast_to(t, (b,)))
        attention = jnp.concatenate([einops.repeat(prefix_mask, 'b p -> b s p', s=tok.shape[1]),
                                     make_attn_mask(mask, ar)], axis=-1)
        pos = jnp.sum(prefix_mask, axis=-1)[:,None] + jnp.cumsum(mask, axis=-1) - 1
        (_, suffix), _ = self.PaliGemma.llm([None, tok], mask=attention, positions=pos,
                                           kv_cache=kv, adarms_cond=[None, cond])
        return self.action_out_proj(suffix[:, -self.action_horizon:])

    if pi0 and not guided:
        # Preserve the original pi0 scan operands and Euler expression exactly.
        # Indexing the embedding table inside a different scan changes BF16
        # compiler fusion and produced a measurable baseline discrepancy.
        dt = jnp.asarray(-1.0 / num_steps, dtype=noise.dtype)
        def original_step(x, tokens):
            _, v = self._run_action_suffix_prepared_from_state_token(
                context['state_token'], x, time_tokens=tokens,
                full_attn_mask=context['full_attn_mask'],positions=context['positions'],kv_cache=kv)
            return x + dt * v, v
        actions, raw = jax.lax.scan(original_step,noise,xs=time_tokens,unroll=2)
        raw = jnp.swapaxes(raw,0,1)
        return actions, {'v':raw,'guided_update':raw,'correction_rms':jnp.zeros((b,num_steps)),
                         'xt_internal':actions}

    def step(carry, idx):
        x, t = carry
        if guided:
            def clean(x_):
                v_ = velocity(x_, t, idx)
                return x_ - t * v_, v_
            predicted, pullback, raw = jax.vjp(clean, x, has_aux=True)
            error = (target - predicted) * weights
            grad = pullback(error)[0]
            # paper tau=1-t; r^2=t^2/((1-t)^2+t^2).
            coefficient = jnp.minimum(10., (t*t + (1-t)**2) / jnp.maximum(t*(1-t), 1e-8))
            correction = coefficient * grad
            update = raw - correction
        else:
            raw = velocity(x, t, idx)
            update = raw
            correction = jnp.zeros_like(raw)
        return (x - update / num_steps, t - 1. / num_steps), (raw, update, jnp.sqrt(jnp.mean(correction**2, axis=(1,2))))

    (actions, _), (raw, update, correction_rms) = jax.lax.scan(
        step, (noise, jnp.asarray(1., jnp.float32)), jnp.arange(num_steps), unroll=2 if pi0 else 1)
    return actions, {'v': jnp.swapaxes(raw,0,1), 'guided_update': jnp.swapaxes(update,0,1),
                     'correction_rms': jnp.swapaxes(correction_rms,0,1), 'xt_internal': actions}
