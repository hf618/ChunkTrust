from __future__ import annotations
import argparse
import hashlib
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
import pickle
import time
from pathlib import Path
from common import EXP,PROTOCOL_ID,write_json,append_json
from worker import simulator_host

def run_wave(engine, jobs, batch_size=10):
    ctx=mp.get_context('spawn')
    pairs=[];pending={};alive=set();results=[];process=None
    try:
        children=[]
        for job in jobs:
            parent,child=ctx.Pipe()
            pairs.append((parent,job));children.append(child);alive.add(parent)
        process=ctx.Process(target=simulator_host,args=(children,jobs))
        process.start()
        for child in children:child.close()
        ready=set();deadline=time.monotonic()+300
        while len(ready)<len(pairs):
            if time.monotonic()>deadline:raise TimeoutError('scene worker setup timeout')
            for conn in wait(list(alive-ready),timeout=1):
                tag,data=conn.recv()
                if tag!='ready':raise RuntimeError((tag,data))
                ready.add(conn)
        # Compile before virtual clocks start. Shape remains exactly 10, including
        # waves with fewer than 10 active episodes and late-episode underfill.
        with (EXP/'artifacts/observation_place_bread_basket.pkl').open('rb') as f:obs=pickle.load(f)
        request=dict(observation=obs,rng_seed=0)
        guided=jobs[0]['method'].startswith('rtc')
        warm=engine.infer([request],guided=False,batch_size=batch_size)[0]
        if guided:
            request.update(previous=warm['actions'][20:],frozen=5)
            engine.infer([request],guided=True,batch_size=batch_size)
        for conn in alive:conn.send('go')
        last=time.monotonic()
        while alive:
            if time.monotonic()-last>600:raise TimeoutError('no rollout progress for 600 seconds')
            readable=wait(list(alive-pending.keys()),timeout=.05) if alive-pending.keys() else []
            for conn in readable:
                tag,data=conn.recv();last=time.monotonic()
                if tag=='infer':pending[conn]=data
                elif tag=='done':results.append(data);alive.remove(conn)
                else:raise RuntimeError((tag,data))
            # Short bounded gather window; clocks are independent of host queueing.
            if pending:
                until=time.monotonic()+.04
                while len(pending)<len(alive) and time.monotonic()<until:
                    for conn in wait(list(alive-pending.keys()),timeout=.005):
                        tag,data=conn.recv()
                        if tag=='infer':pending[conn]=data
                        elif tag=='done':results.append(data);alive.remove(conn)
                        else:raise RuntimeError((tag,data))
                # Empty and guided queues have different numerical paths. Keep
                # their batches separate so another environment cannot change
                # the output of a zero-mask request.
                groups={False:[],True:[]}
                for conn,req in pending.items():groups[guided and len(req.get('previous',()))>0].append(conn)
                for use_guidance,order in groups.items():
                    if not order:continue
                    outputs=engine.infer([pending[c] for c in order],guided=use_guidance,batch_size=batch_size)
                    for conn,output in zip(order,outputs,strict=True):
                        output['server_ready_wall']=time.monotonic()
                        conn.send(('prediction',output))
                    append_json(EXP/'logs/batches.jsonl',dict(time=time.time(),active=len(order),padded=batch_size,
                                method=jobs[0]['method'],stage=jobs[0]['stage'],model_s=outputs[0]['model_batch_s']))
                pending.clear();last=time.monotonic()
        return results
    finally:
        for conn,_ in pairs:conn.close()
        if process is not None:
            process.join(timeout=3)
            if process.is_alive():process.terminate();process.join(timeout=5)

def run(engine, jobs, workers=10):
    remaining=[]
    for job in jobs:
        output=Path(job['output'])
        if (output/'result.json').exists():
            previous=json.loads((output/'result.json').read_text())
            for key in ('stage','backbone','task','method','delay_s','fixed_k','candidates','timeout_s','execution_hash','manifest_hash','settings_hash'):
                if previous[key]!=job[key]:raise RuntimeError(f'resume fingerprint mismatch {key}')
            continue
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f'incomplete attempt requires explicit archived retry, refusing overwrite: {output}')
        remaining.append(job)
    for i in range(0,len(remaining),workers):
        wave=remaining[i:i+workers]
        begin=time.time();results=run_wave(engine,wave,workers)
        append_json(EXP/'logs/waves.jsonl',dict(stage=wave[0]['stage'],backbone=wave[0]['backbone'],
                    task=wave[0]['task'],method=wave[0]['method'],n=len(wave),wall_s=time.time()-begin,
                    successes=sum(r['success'] for r in results),completed_at=time.time()))
        write_json(EXP/'artifacts/live_status.json',dict(stage=wave[0]['stage'],backbone=wave[0]['backbone'],
                   task=wave[0]['task'],method=wave[0]['method'],block_completed=i+len(wave),
                   block_total=len(remaining),last_wave_wall_s=time.time()-begin,
                   last_wave_successes=sum(r['success'] for r in results),updated_at=time.time()))
        print('WAVE_DONE',wave[0]['stage'],wave[0]['method'],i+len(wave),len(remaining),time.time()-begin,flush=True)

def execution_fingerprint():
    runtime=Path(__file__).parent
    code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(runtime.glob('*.py'))
          if p.stem in ('common','controller','selector','sampler','engine','scene','worker','run_block','integrity','lazy_planner')}
    return hashlib.sha256(json.dumps(code,sort_keys=True).encode()).hexdigest()

def jobs_for(args, settings):
    manifest_path=EXP/'manifests'/args.stage/f'{args.task}.json'
    manifest=json.loads(manifest_path.read_text())
    execution_hash=execution_fingerprint()
    manifest_hash=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    settings_hash=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
    jobs=[]
    for extra in args.extra_ms:
        for method in args.methods:
            for entry in manifest['entries']:
                tag=f'{args.backbone}/{args.task}/{method}/K{args.fixed_k}/extra{extra}/ep{entry["rollout_id"]:03d}'
                jobs.append(dict(stage=args.stage,backbone=args.backbone,task=args.task,method=method,
                    protocol_id=PROTOCOL_ID,fft_horizon=50,handoff='first_ready_waypoint_boundary',
                    execution_hash=execution_hash,manifest_hash=manifest_hash,settings_hash=settings_hash,
                    fixed_k=args.fixed_k,candidates=settings['candidates'],
                    delay_s=settings['d0_s']+extra/1000,extra_ms=extra,
                    waypoint_lower_s=settings['waypoint_lower_s'],timeout_s=settings['timeout_s'],
                    entry=entry,output=str(EXP/'results'/args.stage/tag)))
                jobs[-1]['clock_mode']='native' if args.stage in ('native','native_check') else 'controlled'
                if args.stage in ('native','native_check'):
                    jobs[-1]['delay_s']=settings['method_b1_p95_s'][method]
                if args.stage=='native_check':jobs[-1]['calibration_action_limit']=80
                if args.stage in ('calibration','memory'):
                    jobs[-1]['calibration_action_limit']=100 if args.stage=='calibration' else 20
    return jobs

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backbone',required=True);p.add_argument('--task',required=True)
    p.add_argument('--stage',required=True);p.add_argument('--methods',nargs='+',required=True)
    p.add_argument('--extra-ms',nargs='+',type=int,default=[0]);p.add_argument('--fixed-k',type=int,default=20)
    p.add_argument('--settings',type=Path,required=True)
    a=p.parse_args();settings=json.loads(a.settings.read_text())
    if a.stage in ('formal','strong_fixed','native'):
        from semantic_gate import verify_gate
        for stage in ('native_check','smoke','pilot'):verify_gate(stage)
        frozen=json.loads((EXP/'artifacts/formal_protocol.json').read_text())
        assert frozen['protocol_id']==PROTOCOL_ID and frozen['execution_hash']==execution_fingerprint()
    from engine import Engine
    engine=Engine(a.backbone,a.task)
    jobs=jobs_for(a,settings)
    for i in range(0,len(jobs),len(json.loads((EXP/'manifests'/a.stage/f'{a.task}.json').read_text())['entries'])):
        n=len(json.loads((EXP/'manifests'/a.stage/f'{a.task}.json').read_text())['entries'])
        run(engine,jobs[i:i+n],workers=1 if a.stage in ('native','native_check') else 10)
