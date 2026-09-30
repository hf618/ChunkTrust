import argparse
import json
import math
import numpy as np
from common import EXP,PROTOCOL_ID,TASKS,CAL_TASKS,METHODS,write_json
from report import records

def latency_settings():
    settings={}
    calibration=records('calibration')
    for bb in ('pi0','pi05'):
        timed=[]
        for rep in range(3):
            p=json.loads((EXP/'artifacts'/f'profile_{bb}_{rep}.json').read_text())
            if len(p['requests'])!=800:raise AssertionError('incomplete latency profile')
            timed.extend(p['requests'])
        p95={m:float(np.quantile([x['e2e_s'] for x in timed if x['method']==m],.95)) for m in METHODS}
        d0=max(p95.values())
        by_task={}
        for task in CAL_TASKS:
            rs=[r for r in calibration if r['backbone']==bb and r['task']==task]
            if len(rs)!=2:raise AssertionError(f'missing calibration {bb}/{task}: {len(rs)}')
            durations=[d for r in rs for d in r['waypoint_durations_s']]
            if len(durations)<10:raise AssertionError('insufficient waypoint calibration')
            by_task[task]=dict(p05_s=float(np.quantile(durations,.05)),p50_s=float(np.median(durations)),n=len(durations))
        lower=min(x['p05_s'] for x in by_task.values())
        lead=math.ceil((d0+.2)/lower)
        candidates=[k for k in (10,20,30,40) if lead<=k and k+lead<=50]
        evidence=dict(protocol_id=PROTOCOL_ID,fft_horizon=50,d0_s=d0,method_b1_p95_s=p95,waypoint_lower_s=lower,waypoint_calibration=by_task,
                      max_delay_lead=lead,candidates=candidates,timeout_s=3600.,
                      timeout_status='provisional technical budget; formal timeout frozen from independent cohort',
                      forecast='ceil(total delay / minimum task P05 calibrated waypoint duration)',
                      handoff='first legal waypoint boundary at/after prediction availability; forecast is not a commitment',
                      horizon_semantics='K is the target/upper execution budget; early ready handoff can execute fewer actions; log both',
                      late_result_fallback='hold last drive target after old selected K is exhausted; never extend K')
        write_json(EXP/'artifacts'/f'latency_feasibility_{bb}.json',evidence)
        if not candidates:raise RuntimeError(f'no feasible horizon for {bb}, lead={lead}; see feasibility evidence')
        if 20 not in candidates:raise RuntimeError(f'pilot K20 not in feasible candidates {candidates}; technical protocol needs revision')
        write_json(EXP/'artifacts'/f'settings_{bb}.json',evidence,immutable=True)
        settings[bb]=evidence
    write_json(EXP/'artifacts/latency_protocol.json',settings,immutable=True)

def choose_k():
    rows=records('validation');chosen={};strong={};evidence={}
    for bb in ('pi0','pi05'):
        params=json.loads((EXP/'artifacts'/f'settings_{bb}.json').read_text());cs=params['candidates']
        ranking=[];rtc=[]
        for k in cs:
            rs=[r for r in rows if r['backbone']==bb and r['fixed_k']==k and r['method'] in ('sync_fixed','rtc_fixed')]
            expected=4*2*2*20
            if len(rs)!=expected:raise AssertionError(f'incomplete K validation {bb}/{k}: {len(rs)} != {expected}')
            score=float(np.mean([r['success'] for r in rs]));calls=float(np.mean([r['calls'] for r in rs]))
            ranking.append(dict(k=k,sr=score,calls=calls))
            rr=[r for r in rs if r['method']=='rtc_fixed']
            rtc.append(dict(k=k,sr=float(np.mean([r['success'] for r in rr])),calls=float(np.mean([r['calls'] for r in rr])),
                            infer=float(np.mean([r['model_infer_s'] for r in rr]))))
        chosen[bb]=max(ranking,key=lambda x:(x['sr'],-x['calls'],x['k']))['k']
        ahs=[r for r in rows if r['backbone']==bb and r['method']=='rtc_ahs']
        if len(ahs)!=4*2*20:raise AssertionError('incomplete RTC AHS validation')
        ahs_calls=float(np.mean([r['calls'] for r in ahs]))
        best=max(rtc,key=lambda x:(x['sr'],-x['calls'],x['k']))['k']
        budget=min(rtc,key=lambda x:(abs(x['calls']-ahs_calls),-x['sr'],-x['k']))['k']
        strong[bb]=sorted({best,budget}-{chosen[bb]})
        evidence[bb]=dict(common_ranking=ranking,rtc_ranking=rtc,rtc_ahs_calls=ahs_calls,
                          rtc_best_k=best,nearest_call_budget_k=budget)
    write_json(EXP/'artifacts/selected_k.json',dict(fixed_k=chosen,strong_fixed_k=strong,evidence=evidence),immutable=True)

def freeze():
    from semantic_gate import verify_gate
    from run_block import execution_fingerprint
    for stage in ('native_check','smoke','pilot'):verify_gate(stage)
    selected=json.loads((EXP/'artifacts/selected_k.json').read_text())
    timings=records('timeout');limits={}
    for task in TASKS:
        rs=[r for r in timings if r['task']==task]
        if len(rs)!=40:raise AssertionError(f'incomplete independent timeout cohort for {task}')
        limits[task]=math.ceil(1.2*float(np.quantile([r['physics_s'] for r in rs],.99))/.004)*.004
    protocol=dict(schema_version=2,protocol_id=PROTOCOL_ID,execution_hash=execution_fingerprint(),status='FROZEN_BEFORE_FORMAL',tasks=list(TASKS),setting='demo_randomized',
                  methods=list(METHODS),formal_n_per_cell=50,formal_total=9600,
                  **selected,timeout_s=limits,noninferiority_margin_pp=-3.,bootstrap_repetitions=10000,bootstrap_seed=0,
                  latency=json.loads((EXP/'artifacts/latency_protocol.json').read_text()),
                  execution_contract=json.loads((EXP/'artifacts/control_protocol.json').read_text()),
                  timeout_rule='per task, 1.2*P99 pooled 2 backbones x20 independent sync_fixed K* episodes at D0+200ms, rounded up to physical tick')
    write_json(EXP/'artifacts/formal_protocol.json',protocol,immutable=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('step',choices=('latency','choose_k','freeze'));a=p.parse_args()
    {'latency':latency_settings,'choose_k':choose_k,'freeze':freeze}[a.step]()
