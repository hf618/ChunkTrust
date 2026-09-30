from __future__ import annotations
import copy
import dataclasses
import time
import types
import numpy as np
import jax
import jax.numpy as jnp
from openpi.training import config, checkpoints
from openpi.policies import policy_config
from openpi.models import model as base
from openpi.shared import nnx_utils
from common import checkpoint, CONFIGS, soft_mask
from sampler import sample
from integrity import verify_upstream,verify_checkpoint

class Engine:
    def __init__(self, backbone, task):
        start = time.perf_counter()
        self.backbone = backbone
        self.path = checkpoint(backbone, task)
        verify_upstream()
        verify_checkpoint(self.path)
        cfg = config.get_config(CONFIGS[backbone][1])
        resolved = checkpoints.resolve_model_config_for_checkpoint(cfg.model, self.path)
        if getattr(resolved, 'use_qha', False):
            raise ValueError('QHA checkpoint/config is forbidden in this panel')
        cfg = dataclasses.replace(cfg, model=resolved)
        assets = list((self.path / 'assets').glob('*/norm_stats.json'))
        if len(assets) != 1:
            raise ValueError(f'expected unique original normalization stats, got {assets}')
        self.policy = policy_config.create_trained_policy(cfg, self.path, resolved_model_config=resolved,
                                                          robotwin_repo_id=assets[0].parent.name)
        self.model = self.policy._model
        if self.model.action_horizon != 50:
            raise ValueError(f'expected H=50, got {self.model.action_horizon}')
        self.forward = nnx_utils.module_jit(types.MethodType(sample, self.model),
                                           static_argnames=('guided','num_steps'))
        self.load_s = time.perf_counter() - start
        self.compiled = set()

    def prepare(self, requests):
        inputs, targets, masks, noises = [], [], [], []
        for req in requests:
            raw = copy.deepcopy(req['observation'])
            prev = np.asarray(req.get('previous', np.empty((0,14))), np.float32)
            n = len(prev)
            raw['actions'] = np.zeros((50,14), np.float32)
            raw['actions'][:n] = prev
            inp = self.policy._input_transform(raw)
            target = np.asarray(inp.pop('actions'))
            weight = np.zeros_like(target)
            weight[:,:14] = soft_mask(req.get('frozen',0),n)[:,None]
            inputs.append(inp); targets.append(target); masks.append(weight)
            # Request identity, never batch slot or padding, determines the noise.
            noises.append(jax.random.normal(jax.random.key(req['rng_seed']), (50,self.model.action_dim)))
        data = jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)), *inputs)
        return inputs, base.Observation.from_dict(data), jnp.stack(noises), jnp.asarray(np.stack(targets)), jnp.asarray(np.stack(masks))

    def infer(self, requests, guided=False, batch_size=None):
        start = time.perf_counter()
        # Empty initial queues have no RTC objective. Use the identical original
        # numerical path (VJP compilation can otherwise change BF16 GEMM kernels).
        guided = guided and any(len(r.get('previous',())) for r in requests)
        n = len(requests)
        bs = batch_size or n
        if bs < n:
            raise ValueError((bs,n))
        padded = requests + [requests[-1]] * (bs - n)
        inputs, obs, noise, target, mask = self.prepare(padded)
        jax.block_until_ready((obs,noise,target,mask))
        pre = time.perf_counter() - start
        tick = time.perf_counter()
        actions, trace = self.forward(obs,noise,target,mask,guided=guided,num_steps=10)
        jax.block_until_ready((actions,trace))
        model_s = time.perf_counter() - tick
        first_compile = (bs,guided) not in self.compiled
        self.compiled.add((bs,guided))
        actions, trace = jax.device_get((actions,trace))
        outputs = []
        for i in range(n):
            out = self.policy._output_transform({'actions':actions[i], 'state':np.asarray(inputs[i]['state'])})
            action = np.asarray(out['actions'], np.float32)
            if action.shape != (50,14) or not np.isfinite(action).all():
                raise FloatingPointError(f'invalid actions {action.shape}')
            outputs.append({'actions':action,'trace':{k:np.asarray(v[i]) for k,v in trace.items()},
                            'model_batch_s':model_s,'model_amortized_s':model_s/n,
                            'active_batch':n,'padded_batch':bs,'jit_included':first_compile})
        elapsed = time.perf_counter()-start
        for out in outputs:
            out.update(preprocess_batch_s=pre, end_to_end_batch_s=elapsed)
        return outputs

    def roundtrip(self, request, absolute):
        raw = copy.deepcopy(request['observation']); raw['actions'] = absolute.copy()
        inp = self.policy._input_transform(raw)
        out = self.policy._output_transform({'actions':np.asarray(inp['actions']), 'state':np.asarray(inp['state'])})
        return float(np.max(np.abs(out['actions']-absolute)))
