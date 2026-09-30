import argparse
import copy
import pickle
import time
import numpy as np
import jax
from common import EXP, write_json
from engine import Engine

def main():
    p=argparse.ArgumentParser(); p.add_argument('--backbone',required=True); p.add_argument('--batch-size',type=int,default=10); a=p.parse_args()
    with (EXP/'artifacts/observation_place_bread_basket.pkl').open('rb') as f:obs=pickle.load(f)
    request={'observation':obs,'rng_seed':17}
    engine=Engine(a.backbone,'place_bread_basket')
    print('MODEL_LOADED',engine.load_s,flush=True)
    plain=engine.infer([request])[0]
    np.save(EXP/'artifacts'/f'probe_actions_{a.backbone}.npy',plain['actions'])
    print('PLAIN_OK',plain['model_batch_s'],flush=True)
    # Direct original sampler equivalence uses the exact same JAX key and noise.
    _,ob,noise,_,_=engine.prepare([request])
    # Original wrapper does not mark num_steps static; its frozen default is 10.
    kwargs={}
    if a.backbone=='pi05':kwargs['noise']=noise
    ref,reftrace=engine.policy._sample_actions_with_trace(jax.random.key(17),ob,**kwargs)
    ref=np.asarray(ref)
    error=float(np.max(np.abs(ref[0]-plain['trace']['xt_internal'])))
    if not np.allclose(ref[0],plain['trace']['xt_internal'],atol=2e-4,rtol=2e-4):
        raise AssertionError(f'original sampler mismatch: {error}')
    raw_error=float(np.max(np.abs(np.asarray(reftrace['v'][0],np.float32)-np.asarray(plain['trace']['v'],np.float32))))
    if not np.allclose(np.asarray(reftrace['v'][0],np.float32),np.asarray(plain['trace']['v'],np.float32),atol=2e-4,rtol=2e-4):
        raise AssertionError(f'original raw velocity mismatch: {raw_error}')
    empty=engine.infer([request],guided=True)[0]
    zero_error=float(np.max(np.abs(empty['actions']-plain['actions'])))
    if not np.allclose(empty['actions'],plain['actions'],atol=2e-4,rtol=2e-4):
        raise AssertionError(f'empty-guidance mismatch {zero_error}')
    rt=engine.roundtrip(request,plain['actions'])
    if rt>2e-4:raise AssertionError(f'action roundtrip {rt}')
    request.update(previous=plain['actions'][20:],frozen=5)
    guided=engine.infer([request],guided=True)[0]
    batch=engine.infer([request],guided=True,batch_size=a.batch_size)[0]
    # BF16 GEMM algorithms can vary with batch size; report error, require bounded drift.
    batch_error=float(np.max(np.abs(batch['actions']-guided['actions'])))
    if not np.allclose(batch['actions'],guided['actions'],atol=.03,rtol=.03):
        raise AssertionError(f'batch numerical mismatch {batch_error}')
    permutation=engine.infer([dict(request,rng_seed=111),request],guided=True,batch_size=a.batch_size)[1]
    slot_error=float(np.max(np.abs(permutation['actions']-batch['actions'])))
    if slot_error>2e-5:raise AssertionError(f'RNG/batch slot mismatch {slot_error}')
    result=dict(backbone=a.backbone,load_s=engine.load_s,original_max_error=error,original_raw_velocity_max_error=raw_error,zero_mask_max_error=zero_error,
                action_roundtrip_max_error=rt,batch_max_error=batch_error,slot_max_error=slot_error,
                batch_size=a.batch_size,action_shape=list(batch['actions'].shape),
                trace_shape=list(batch['trace']['v'].shape),correction_rms=guided['trace']['correction_rms'],
                devices=[str(d) for d in jax.devices()],memory_stats=jax.devices()[0].memory_stats())
    write_json(EXP/'artifacts'/f'model_probe_{a.backbone}.json',result)
    print('MODEL_PROBE_OK',result,flush=True)

if __name__=='__main__':main()
