from __future__ import annotations
import argparse
import multiprocessing as mp
import pickle
import time
import traceback
import numpy as np
from common import EXP,CAL_TASKS,METHODS,write_json,rng_seed
from selector import Selector

def client(conn, backbone, observations):
    try:
        measured=[]
        for method in METHODS:
            selector=Selector(seed=3)
            previous=None
            for idx in range(220):
                obs=observations[idx%len(observations)]
                request=dict(observation=obs,rng_seed=rng_seed(backbone,'profile',0,idx))
                if previous is not None and method.startswith('rtc'):
                    request.update(previous=previous[20:],frozen=5)
                start=time.perf_counter()
                conn.send(('infer',method,request))
                out=conn.recv()
                rpc_s=time.perf_counter()-start
                selection=0.
                if method.endswith('ahs'):
                    t=time.perf_counter()
                    offset=5 if request.get('frozen') else 0
                    selector.select(out['trace']['v'][:,offset:],out['actions'][offset:],[] if previous is None else previous[:20])
                    selection=time.perf_counter()-t
                elapsed=time.perf_counter()-start
                if idx>=20:
                    if out['jit_included']:raise AssertionError('profiling JIT within timed requests')
                    measured.append(dict(method=method,request=idx-20,e2e_s=elapsed,rpc_model_s=rpc_s,
                                  model_s=out['model_batch_s'],selector_s=selection,
                                  preprocess_s=out['preprocess_batch_s']))
                previous=out['actions']
            print('PROFILE_METHOD_DONE',method,flush=True)
        conn.send(('done',measured))
    except BaseException:
        conn.send(('error',traceback.format_exc()))
        raise
    finally:conn.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--backbone',required=True);p.add_argument('--repeat',type=int,required=True);a=p.parse_args()
    observations=[]
    for task in CAL_TASKS:
        with (EXP/'artifacts'/f'observation_{task}.pkl').open('rb') as f:observations.append(pickle.load(f))
    from engine import Engine
    engine=Engine(a.backbone,'place_bread_basket')
    ctx=mp.get_context('spawn');parent,child=ctx.Pipe()
    process=ctx.Process(target=client,args=(child,a.backbone,observations));process.start();child.close()
    try:
        while True:
            if not parent.poll(300):raise TimeoutError('profile client timeout')
            msg=parent.recv()
            if msg[0]=='done':records=msg[1];break
            if msg[0]=='error':raise RuntimeError(msg[1])
            _,method,request=msg
            parent.send(engine.infer([request],guided=method.startswith('rtc'))[0])
        summary={}
        for method in METHODS:
            values=[r['e2e_s'] for r in records if r['method']==method]
            summary[method]=dict(n=len(values),p50_s=float(np.quantile(values,.5)),p95_s=float(np.quantile(values,.95)),p99_s=float(np.quantile(values,.99)))
        write_json(EXP/'artifacts'/f'profile_{a.backbone}_{a.repeat}.json',
                   dict(backbone=a.backbone,repeat=a.repeat,warmup_per_method=20,load_s=engine.load_s,
                        timing='B1 actual multiprocessing Pipe roundtrip including transform, model, VJP, transfer and CPU AHS',
                        summary=summary,requests=records),immutable=True)
        print('PROFILE_OK',summary,flush=True)
    finally:
        parent.close();process.join(timeout=5)
        if process.is_alive():process.terminate();process.join(timeout=5)

if __name__=='__main__':main()
