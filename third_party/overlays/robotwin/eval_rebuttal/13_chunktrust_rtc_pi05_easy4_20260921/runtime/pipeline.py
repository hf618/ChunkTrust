"""Persistent, fail-closed execution of the approved full experiment panel.

No SR-dependent early stop. A failed technical check stops the pipeline with
its evidence preserved. Existing successful episodes are immutable.
"""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from common import EXP,ROOT,PROTOCOL_ID,TASKS,CAL_TASKS,METHODS,CONFIGS,write_json,append_json
from manifests import STAGES

RUNTIME=EXP/'runtime'
CURRENT=None

def env_for(bb='pi05',cpu=False):
    env=os.environ.copy()
    env.update(LEROBOT_SKIP_TIMESTAMP_CHECK='1',XLA_PYTHON_CLIENT_MEM_FRACTION='.40',
               XLA_PYTHON_CLIENT_PREALLOCATE='true',
               XLA_FLAGS='--xla_gpu_enable_command_buffer=',OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',XDG_RUNTIME_DIR='/tmp',PYTHONHASHSEED='0')
    if cpu:env['JAX_PLATFORMS']='cpu'
    else:env.pop('JAX_PLATFORMS',None)
    return env

def python(bb='pi05'):
    return str(ROOT/'policy'/CONFIGS[bb][0]/'.venv/bin/python')

def retryable_scene_setup_failure(script, evidence):
    # Only the pre-GO scene handshake is retryable. In-rollout errors, identity
    # mismatches and model OOM remain fail-closed; never replace a seed/result.
    return (script in ('panel_run_block.py','easy_run_block.py') and "if tag!='ready'" in evidence
            and 'cannot create buffer' in evidence)


def run(name,script,args=(),bb='pi05',cpu=False):
    retries=0
    log_path=EXP/'logs'/f'{name}.log'
    while True:
        offset=log_path.stat().st_size if log_path.exists() else 0
        completed_before=len(list((EXP/'results/formal').rglob('result.json')))
        try:
            return run_once(name,script,args,bb,cpu)
        except RuntimeError:
            with log_path.open('rb') as stream:
                stream.seek(offset);evidence=stream.read().decode(errors='replace')
            if not retryable_scene_setup_failure(script,evidence):raise
            completed_after=len(list((EXP/'results/formal').rglob('result.json')))
            if completed_after>completed_before:retries=0
            retries+=1
            if retries>2:raise
            append_json(EXP/'logs/setup_retries.jsonl',dict(time=time.time(),phase=name,
                retry=retries,completed_before=completed_before,completed_after=completed_after,
                reason='pre-GO renderer buffer allocation; restart process with identical jobs/config'))
            print('RETRY_SCENE_SETUP',name,retries,flush=True)
            time.sleep(5)


def run_once(name,script,args=(),bb='pi05',cpu=False):
    global CURRENT
    command=[python(bb),str(RUNTIME/script),*map(str,args)]
    start=time.time()
    status=dict(status='running',phase=name,pid=os.getpid(),command=command,started_at=start)
    write_json(EXP/'artifacts/pipeline_status.json',status)
    print('START',name,flush=True)
    with (EXP/'logs'/f'{name}.log').open('a') as log:
        CURRENT=subprocess.Popen(command,cwd=ROOT,env=env_for(bb,cpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while CURRENT.poll() is None:
            status.update(child_pid=CURRENT.pid,heartbeat=time.time(),elapsed_s=time.time()-start)
            write_json(EXP/'artifacts/pipeline_status.json',status)
            try:
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
                append_json(EXP/'logs/hardware.jsonl',dict(time=time.time(),phase=name,gpu=gpu))
            except Exception as exc:append_json(EXP/'logs/hardware.jsonl',dict(time=time.time(),error=str(exc)))
            time.sleep(5)
        code=CURRENT.returncode;CURRENT=None
    append_json(EXP/'logs/pipeline_events.jsonl',dict(phase=name,command=command,exit_code=code,wall_s=time.time()-start))
    if code:raise RuntimeError(f'{name} failed (exit {code}); see logs/{name}.log')
    print('DONE',name,time.time()-start,flush=True)

def manifest(stage):
    for task in STAGES[stage][2]:
        if not (EXP/'manifests'/stage/f'{task}.json').exists():
            run(f'manifest_{stage}_{task}','manifests.py',['--stage',stage,'--task',task],cpu=True)

def block(stage,bb,task,methods,extras,k=20,settings=None):
    settings=settings or EXP/'artifacts'/f'settings_{bb}.json'
    run(f'{stage}_{bb}_{task}_K{k}_{"-".join(methods)}','run_block.py',
        ['--stage',stage,'--backbone',bb,'--task',task,'--methods',*methods,
         '--extra-ms',*extras,'--fixed-k',k,'--settings',settings],bb=bb)
    run('report','report.py',cpu=True)

def rotated(methods,bb,task):
    offset=(TASKS.index(task)+(0 if bb=='pi0' else 2))%len(methods)
    return list(methods[offset:])+list(methods[:offset])

def main():
    lock=(EXP/'pipeline.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.write(str(os.getpid()));lock.flush()
    source_hashes={str(p.relative_to(EXP)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(RUNTIME.glob('*.py'))}
    write_json(EXP/'artifacts/execution_sources.json',source_hashes)
    write_json(EXP/'artifacts/control_protocol.json',dict(
        protocol_id=PROTOCOL_ID,fft_horizon=50,
        scoring='original AHS formulas on the actually available suffix; fixed H50 FFT padding; actual executed target history',
        handoff='first legal waypoint boundary at or after full prediction availability',
        controlled_availability='request_tick + ceil(specified_delay / physics_dt); never inspect output earlier',
        native_availability='complete prediction received by background receiver; wall clock timestamp',
        forecast='frozen waypoint P05-based prefix estimate guides inference and request timing, but does not compel execution',
        k_semantics='selected K is the target and maximum old execution budget; record actual actions separately; never extend K to hide a late result',
        late_fallback='hold last drive target and continue physics after selected old budget is exhausted',
        limitations='variable-duration waypoint adaptation; cannot interrupt TOPP or claim fixed-Hz/native-deployment equivalence',
        required_gates=['unit_tests','native_check','smoke','pilot']),immutable=True)
    run('unit_tests','run_tests.py',cpu=True)
    if not (EXP/'artifacts/checkpoint_inventory.json').exists():run('provenance','provenance.py',cpu=True)
    audit=json.loads((EXP/'artifacts/seed_overlap_audit.json').read_text())
    if audit['overlaps']:raise RuntimeError('historical seed overlap')
    manifest('calibration')
    for bb in ('pi05','pi0'):
        if not (EXP/'artifacts'/f'model_probe_{bb}.json').exists():run(f'model_probe_{bb}','probe_model.py',['--backbone',bb],bb=bb)
    for task in ('place_bread_basket','handover_block'):
        if not (EXP/'artifacts'/f'controller_audit_{task}.json').exists():run(f'controller_audit_{task}','controller_audit.py',['--task',task],cpu=True)
    # Three fresh model-service processes per backbone. Never overlap profiling
    # with simulator setup or another model service on this single GPU.
    for bb in ('pi05','pi0'):
        for rep in range(3):
            if not (EXP/'artifacts'/f'profile_{bb}_{rep}.json').exists():
                run(f'profile_{bb}_{rep}','latency_profile.py',['--backbone',bb,'--repeat',rep],bb=bb)
    bootstrap=EXP/'artifacts/settings_calibration.json'
    write_json(bootstrap,dict(d0_s=0.,candidates=[10,20,30,40],waypoint_lower_s=.004,timeout_s=3600.),immutable=True)
    manifest('memory')
    for bb in ('pi05','pi0'):
        block('memory',bb,'place_bread_basket',['rtc_fixed'],[0],settings=bootstrap)
        for task in CAL_TASKS:block('calibration',bb,task,['sync_fixed'],[0],settings=bootstrap)
    if not (EXP/'artifacts/latency_protocol.json').exists():run('freeze_latency','calibrate.py',['latency'],cpu=True)
    # Reuses the first memory-probe scene for an isolated B1 technical check.
    # This is not an additional independent success-rate cohort.
    manifest('native_check')
    for bb in ('pi05','pi0'):
        block('native_check',bb,'place_bread_basket',['rtc_fixed','rtc_ahs'],[0])
    run('gate_native_check','semantic_gate.py',['native_check'],cpu=True)
    manifest('smoke')
    for bb in ('pi05','pi0'):
        for task in STAGES['smoke'][2]:block('smoke',bb,task,rotated(METHODS,bb,task),[0,200])
    run('gate_smoke','semantic_gate.py',['smoke'],cpu=True)
    manifest('pilot')
    for bb in ('pi05','pi0'):
        for task in CAL_TASKS:block('pilot',bb,task,rotated(METHODS,bb,task),[0,100,200])
    run('gate_pilot','semantic_gate.py',['pilot'],cpu=True)
    manifest('validation')
    for bb in ('pi05','pi0'):
        candidates=json.loads((EXP/'artifacts'/f'settings_{bb}.json').read_text())['candidates']
        for task in CAL_TASKS:
            for k in candidates:block('validation',bb,task,rotated(('sync_fixed','rtc_fixed'),bb,task),[0,200],k)
            block('validation',bb,task,rotated(('sync_ahs','rtc_ahs'),bb,task),[0,200])
    if not (EXP/'artifacts/selected_k.json').exists():run('choose_k','calibrate.py',['choose_k'],cpu=True)
    selected=json.loads((EXP/'artifacts/selected_k.json').read_text())
    # Four formal-only tasks lack K-selection validation trajectories. Use an
    # independent reference cohort for all eight tasks, keeping timeout estimation
    # out of both K selection and formal SR. 2 x8 x20 =320 additional episodes.
    manifest('timeout')
    for bb in ('pi05','pi0'):
        for task in TASKS:block('timeout',bb,task,['sync_fixed'],[200],selected['fixed_k'][bb])
    if not (EXP/'artifacts/formal_protocol.json').exists():run('freeze_formal','calibrate.py',['freeze'],cpu=True)
    frozen=json.loads((EXP/'artifacts/formal_protocol.json').read_text())
    manifest('formal')
    from semantic_gate import verify_gate
    for stage in ('native_check','smoke','pilot'):verify_gate(stage)
    for bb in ('pi05','pi0'):
        for task in TASKS:
            settings=dict(frozen['latency'][bb],timeout_s=frozen['timeout_s'][task])
            path=EXP/'artifacts'/f'formal_settings_{bb}_{task}.json';write_json(path,settings,immutable=True)
            block('formal',bb,task,rotated(METHODS,bb,task),[0,100,200],selected['fixed_k'][bb],path)
    # Strong fixed controls use the identical formal identities but stay outside
    # the strict four-row primary table.
    strong_dir=EXP/'manifests/strong_fixed';strong_dir.mkdir(exist_ok=True)
    for task in CAL_TASKS:
        dest=strong_dir/f'{task}.json'
        if not dest.exists():dest.symlink_to(EXP/'manifests/formal'/f'{task}.json')
    for bb in ('pi05','pi0'):
        for task in CAL_TASKS:
            for k in selected['strong_fixed_k'][bb]:
                block('strong_fixed',bb,task,['rtc_fixed'],[0,200],k,EXP/'artifacts'/f'formal_settings_{bb}_{task}.json')
    manifest('native')
    for bb in ('pi05','pi0'):
        for task in STAGES['native'][2]:
            block('native',bb,task,rotated(METHODS,bb,task),[0],selected['fixed_k'][bb],EXP/'artifacts'/f'formal_settings_{bb}_{task}.json')
    run('report','report.py',cpu=True)
    progress=json.loads((EXP/'artifacts/progress.json').read_text())
    if progress['formal']['n']!=9600:raise AssertionError('incomplete formal panel')
    write_json(EXP/'artifacts/pipeline_status.json',dict(status='completed',formal_episodes=9600,completed_at=time.time()))
    print('FULL_PIPELINE_COMPLETED',flush=True)

def stop(signum,frame):
    if CURRENT is not None:
        try:os.killpg(CURRENT.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    raise KeyboardInterrupt

if __name__=='__main__':
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:main()
    except BaseException as exc:
        write_json(EXP/'artifacts/pipeline_status.json',dict(status='stopped_on_error',pid=os.getpid(),error=str(exc),traceback=traceback.format_exc(),at=time.time()))
        traceback.print_exc();raise
