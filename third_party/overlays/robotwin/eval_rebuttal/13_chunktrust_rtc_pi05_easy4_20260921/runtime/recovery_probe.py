"""Isolated allocator recovery evidence. Never writes formal rollout results."""
import argparse
import multiprocessing as mp
import os
import pickle
import time
import traceback
import numpy as np
from common import EXP, write_json


def scenes(conn):
    os.environ['JAX_PLATFORMS'] = 'cpu'
    from scene import create, state_signature
    import json
    envs=[]
    try:
        entries=json.loads((EXP/'manifests/formal_extra50/place_can_basket.json').read_text())['entries'][:10]
        for entry in entries:
            env, obs=create('place_can_basket', entry['episode_seed'], entry['rollout_id'], entry['instruction'])
            envs.append(env)
            assert state_signature(env,obs)==entry['signature']
            conn.send(('prepared',entry['rollout_id']))
        conn.send(('ready',10))
        assert conn.recv()=='close'
    except BaseException:
        conn.send(('error',traceback.format_exc()))
        raise
    finally:
        for env in envs:env.close_env()
        conn.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--scenes',action='store_true');a=p.parse_args()
    import jax
    from engine import Engine
    assert jax.default_backend()=='gpu', jax.devices()
    print('GPU_CONFIRMED',jax.devices(),flush=True)
    engine=Engine('pi05','place_can_basket')
    print('MODEL_LOADED',jax.devices()[0].memory_stats(),flush=True)
    process=None
    try:
        if a.scenes:
            ctx=mp.get_context('spawn');conn,child=ctx.Pipe()
            process=ctx.Process(target=scenes,args=(child,));process.start();child.close()
            while True:
                if not conn.poll(300):raise TimeoutError('scene probe setup')
                tag,data=conn.recv();print(tag,data,flush=True)
                if tag=='error':raise RuntimeError(data)
                if tag=='ready':break
        with (EXP/'artifacts/observation_place_bread_basket.pkl').open('rb') as f:obs=pickle.load(f)
        request=dict(observation=obs,rng_seed=17)
        plain=engine.infer([request],guided=False,batch_size=10)[0]
        guided_req=dict(request,previous=plain['actions'][20:],frozen=5)
        guided=engine.infer([guided_req],guided=True,batch_size=10)[0]
        timings={}
        for flag,req in ((False,request),(True,guided_req)):
            timings[str(flag)]=[engine.infer([req],guided=flag,batch_size=10)[0]['model_batch_s'] for _ in range(3)]
        arrays={f'{label}_{k}':np.asarray(v) for label,result in (('plain',plain),('guided',guided)) for k,v in dict(actions=result['actions'],**result['trace']).items()}
        out=EXP/'artifacts/recovery_20260921';out.mkdir(exist_ok=True)
        np.savez(out/f'{a.name}.npz',**arrays)
        report=dict(status='PASS',name=a.name,scene_count=10 if a.scenes else 0,device=str(jax.devices()[0]),timings=timings,memory=jax.devices()[0].memory_stats(),environment={k:os.environ.get(k) for k in ('JAX_PLATFORMS','XLA_PYTHON_CLIENT_MEM_FRACTION','XLA_PYTHON_CLIENT_PREALLOCATE','XLA_FLAGS')})
        write_json(out/f'{a.name}.json',report);print(report,flush=True)
    finally:
        if process:
            if process.is_alive():conn.send('close')
            process.join(15)
            if process.is_alive():process.terminate();process.join(5)

if __name__=='__main__':main()
