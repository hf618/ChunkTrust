#!/usr/bin/env python3
"""CPU QHA-head optimization and checkpoint roundtrip on synthetic frozen features.

Run in the OpenPI environment. This checks the head, not a full policy rollout
or the paper's online teacher training protocol.
"""
import tempfile
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from openpi.models.qha import QueryBasedHorizonAdapter,QHAConfig,kl_or_soft_ce

head=QueryBasedHorizonAdapter(32,32,QHAConfig(tuple(range(1,51))),rngs=nnx.Rngs(0))
rng=np.random.default_rng(7)
context=jnp.asarray(rng.normal(size=(2,8,32)),dtype=jnp.float32)
actions=jnp.asarray(rng.normal(size=(2,50,32)),dtype=jnp.float32)
target=jax.nn.softmax(jnp.asarray(rng.normal(size=(2,50)),dtype=jnp.float32),axis=-1)
optimizer=nnx.Optimizer(head,optax.adamw(5e-5))

def loss(m):return kl_or_soft_ce(m(context,actions)['qha_posterior'],target).mean()

losses=[]
for _ in range(3):
    value,grads=nnx.value_and_grad(loss)(head)
    assert np.isfinite(float(value))
    optimizer.update(grads);losses.append(float(value))
# Serialize only the trainable head's arrays. The paper's released checkpoints
# use Orbax; this lightweight check independently tests a parameter roundtrip.
graph,params,other=nnx.split(head,nnx.Param,...)

leaves,structure=jax.tree_util.tree_flatten(params)
with tempfile.TemporaryDirectory() as tmp:
    file=Path(tmp)/'head.npz'
    np.savez(file,**{f'p{i}':np.asarray(v) for i,v in enumerate(leaves)})
    with np.load(file,allow_pickle=False) as data:
        restored=jax.tree_util.tree_unflatten(structure,[jnp.asarray(data[f'p{i}']) for i in range(len(leaves))])
    rebuilt=nnx.merge(graph,restored,other)
    np.testing.assert_array_equal(np.asarray(head(context,actions)['qha_posterior']),np.asarray(rebuilt(context,actions)['qha_posterior']))
print({'head_optimization_steps':3,'losses':losses,'parameter_roundtrip':'PASS','input_kind':'synthetic frozen features'})
