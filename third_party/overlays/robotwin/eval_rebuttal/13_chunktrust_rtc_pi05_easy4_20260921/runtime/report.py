from __future__ import annotations
import csv
import datetime
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from common import EXP,TASKS,METHODS,write_json
from panel_config import load_active_protocol, primary_records

def records(stage):
    out=[]
    for p in (EXP/'results'/stage).glob('**/result.json'):
        row=json.loads(p.read_text());row['result_path']=str(p)
        if row['status']=='completed':out.append(row)
    return out

def summarize(rows):
    if not rows:return dict(n=0,sr=None,wait_pct=None,calls_ep=None,infer_s_ep=None)
    task_sr=[np.mean([r['success'] for r in rows if r['task']==task]) for task in sorted({r['task'] for r in rows})]
    return dict(n=len(rows),tasks=len(task_sr),sr=100*float(np.mean(task_sr)),
                wait_pct=100*sum(r['wait_s'] for r in rows)/max(sum(r['physics_s'] for r in rows),1e-10),
                calls_ep=float(np.mean([r['calls'] for r in rows])),
                infer_s_ep=float(np.mean([r['model_infer_s'] for r in rows])))

def timing_summary(rows):
    handoff=[v for r in rows for v in r['handoff_delay_s']]
    age=[v for r in rows for v in r['request_to_commit_s']]
    selected=[v for r in rows for v in r['selected_k_before_handoff']]
    actual=[v for r in rows for v in r['actual_k_before_handoff']]
    return dict(n=len(rows),
        request_to_commit_p50_ms=1000*float(np.quantile(age,.5)) if age else None,
        request_to_commit_p95_ms=1000*float(np.quantile(age,.95)) if age else None,
        handoff_after_ready_p95_ms=1000*float(np.quantile(handoff,.95)) if handoff else None,
        action_age_mean_ms=1000*sum(r['action_age_sum_s'] for r in rows)/max(sum(r['action_age_count'] for r in rows),1),
        selected_k_nonterminal_mean=float(np.mean(selected)) if selected else None,
        actual_k_nonterminal_mean=float(np.mean(actual)) if actual else None,
        shortened_nonterminal_chunks=sum(a<k for a,k in zip(actual,selected)),
        mean_physical_s_ep=float(np.mean([r['physics_s'] for r in rows])) if rows else None)

def paired_ci(rows,backbone,delay,comparison,tasks=TASKS,n_per_task=50):
    a,b=comparison
    data=[];cost=[]
    for task in tasks:
        arm={m:{r['episode_seed']:r for r in rows if r['backbone']==backbone and r['task']==task and r['method']==m and r['extra_ms']==delay} for m in (a,b)}
        for method in (a,b):
            count=sum(r['backbone']==backbone and r['task']==task and r['method']==method and r['extra_ms']==delay for r in rows)
            if count!=len(arm[method]):raise AssertionError('duplicate paired identity')
        if set(arm[a])!=set(arm[b]) or len(arm[a])!=n_per_task:return None
        # Both arms must also have identical initial physical/rendered state.
        for seed in arm[a]:
            if arm[a][seed]['reset_signature']!=arm[b][seed]['reset_signature']:raise AssertionError('paired reset mismatch')
        data.append(np.array([int(arm[a][s]['success'])-int(arm[b][s]['success']) for s in sorted(arm[a])]))
        cost.append(np.array([[[arm[m][s][k] for k in ('wait_s','physics_s','calls','model_infer_s')] for m in (a,b)] for s in sorted(arm[a])]))
    data=np.asarray(data);cost=np.asarray(cost);rng=np.random.default_rng(0)
    draws=[];cost_draws=[]
    def cost_delta(c):
        total=c.reshape(-1,2,4).sum(axis=0)
        return [100*(total[0,0]/max(total[0,1],1e-10)-total[1,0]/max(total[1,1],1e-10)),
                (total[0,2]-total[1,2])/(c.size/8),(total[0,3]-total[1,3])/(c.size/8)]
    for _ in range(10000):
        ts=rng.integers(0,len(tasks),len(tasks))
        ids=rng.integers(0,n_per_task,(len(tasks),n_per_task))
        draws.append(float(data[ts[:,None],ids].mean())*100)
        cost_draws.append(cost_delta(cost[ts[:,None],ids]))
    lo,hi=np.quantile(draws,[.025,.975])
    cost_ci=np.quantile(cost_draws,[.025,.975],axis=0)
    return dict(backbone=backbone,extra_ms=delay,comparison=f'{a} - {b}',n_pairs=n_per_task*len(tasks),
                delta_pp=100*float(data.mean()),ci95_pp=[float(lo),float(hi)],
                cost_delta=dict(zip(('wait_pp','calls_ep','infer_s_ep'),cost_delta(cost))),
                cost_ci95=dict(zip(('wait_pp','calls_ep','infer_s_ep'),cost_ci.T.tolist())),
                less_waiting=bool(cost_ci[1,0]<0),
                noninferiority_margin_pp=-3.,noninferior=bool(lo>-3.),bootstrap='hierarchical paired tasks then seeds; 10000; rng=0')

def main():
    output=EXP/'artifacts/tables';output.mkdir(parents=True,exist_ok=True)
    frozen=load_active_protocol()
    n_per_task=frozen.get('formal_n_per_cell',50) if frozen else 50
    n_per_row=len(TASKS)*n_per_task
    formal_total=2*len(METHODS)*3*n_per_row
    if frozen:assert formal_total==frozen['formal_total']
    raw_formal=records('formal')
    allrows=primary_records(raw_formal,frozen) if frozen else []
    table=[];detail=[];cis=[]
    for backbone in ('pi0','pi05'):
        for method in METHODS:
            selected_k=frozen['fixed_k'][backbone] if frozen else None
            subset=[r for r in allrows if r['backbone']==backbone and r['method']==method]
            row=dict(policy=backbone,execution='RTC' if method.startswith('rtc') else 'Sync',
                     horizon='AHS' if method.endswith('ahs') else f'K={selected_k}' if selected_k is not None else 'K=pending')
            for delay in (0,100,200):
                rs=[r for r in subset if r['extra_ms']==delay];s=summarize(rs)
                row[f'n_plus_{delay}']=s['n']
                row[f'sr_plus_{delay}_pct']=s['sr'] if s['n']==n_per_row and s['tasks']==len(TASKS) else None
                detail.append(dict(backbone=backbone,method=method,extra_ms=delay,**s))
                if delay==100:
                    for key in ('wait_pct','calls_ep','infer_s_ep'):row[key]=s[key] if s['n']==n_per_row else None
            table.append(row)
        for delay in (0,100,200):
            for pair in (('rtc_ahs','sync_ahs'),('rtc_ahs','rtc_fixed'),('rtc_fixed','sync_fixed'),('sync_ahs','sync_fixed')):
                if frozen:
                    ci=paired_ci(allrows,backbone,delay,pair,n_per_task=n_per_task)
                    if ci:cis.append(ci)
    with (output/'table5.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    write_json(output/'all_delay_costs.json',detail)
    write_json(output/'paired_ci.json',cis)
    strong_cis=[]
    if frozen:
        from common import CAL_TASKS
        strong=records('strong_fixed')
        selection_path=EXP/frozen.get('supplementary_selection_file','artifacts/strong_control_selection.json')
        selection=json.loads(selection_path.read_text()) if selection_path.exists() else frozen
        if selection_path.exists():
            assert selection['panel_protocol_id']==frozen['panel_protocol_id']
            assert selection['fixed_k']==frozen['fixed_k']
        for bb,ks in selection['strong_fixed_k'].items():
            for k in ks:
                alias=f'rtc_fixed_K{k}'
                merged=[r for r in allrows if r['backbone']==bb and r['method']=='rtc_ahs']
                merged += [dict(r,method=alias) for r in strong if r['backbone']==bb and r['fixed_k']==k]
                for delay in (0,200):
                    ci=paired_ci(merged,bb,delay,('rtc_ahs',alias),tasks=CAL_TASKS,n_per_task=n_per_task)
                    if ci:strong_cis.append(ci)
    write_json(output/'strong_fixed_paired_ci.json',strong_cis)
    strong_groups=defaultdict(list)
    for row in records('strong_fixed'):
        strong_groups[(row['backbone'],row['fixed_k'],row['extra_ms'])].append(row)
    write_json(output/'strong_fixed_costs.json',[dict(backbone=k[0],fixed_k=k[1],extra_ms=k[2],**summarize(v)) for k,v in sorted(strong_groups.items())])
    groups=defaultdict(list)
    for r in allrows:groups[(r['backbone'],r['task'],r['method'],r['extra_ms'],r['fixed_k'])].append(r)
    write_json(output/'per_task.json',[dict(backbone=k[0],task=k[1],method=k[2],extra_ms=k[3],fixed_k=k[4],**summarize(v)) for k,v in sorted(groups.items())])
    def fmt(x):return '—' if x is None else f'{x:.2f}' if isinstance(x,float) else str(x)
    text=['# Standalone Table 5 — ChunkTrust × RTC','',
          f'Formal results recorded: {len(allrows)}/{formal_total}. A cell is shown only after all {n_per_row} paired episodes finish ({n_per_task} per task).','',
          '| Policy | Execution | Horizon | SR +0 ms ↑ | SR +100 ms ↑ | SR +200 ms ↑ | Wait %† ↓ | Calls/ep† | Infer s/ep† |',
          '|---|---|---|---:|---:|---:|---:|---:|---:|']
    columns=('policy','execution','horizon','sr_plus_0_pct','sr_plus_100_pct','sr_plus_200_pct','wait_pct','calls_ep','infer_s_ep')
    for row in table:text.append('| '+' | '.join(fmt(row[c]) for c in columns)+' |')
    text+=['','† Costs use +100 ms, all outcomes included. +0 is measured D0, not zero inference latency.',
           'Controlled virtual physical-time results; model seconds are blocked model-call time amortized across active batch requests, not B1 deployment latency.',
           'RTC uses the first legal waypoint boundary after prediction availability. It never starts another old waypoint after that boundary; an active TOPP trajectory is not interrupted.',
           'K denotes the selected target/maximum execution budget. Early readiness can shorten actual execution. FFT padding stays H=50. See timing_diagnostics.md for action age, handoff delay and selected versus actual K.',
           'Wait percentage is the ratio of total inference-induced hold time to total simulated episode time. No TeX has been generated or modified.','']
    if frozen and frozen.get('panel_protocol_id'):
        text += [f'Active panel: {frozen["panel_protocol_id"]}. Fixed K: {frozen["fixed_k"]}; specified comparator, not validation-optimal K*. Original action limits and common 3600 s physical cap.',
                 frozen.get('reporting_warning','K optimization and strong fixed controls follow the primary panel.'),'']
    (output/'table5.md').write_text('\n'.join(text))
    # Refresh the task-level view under the same frozen sample-size target.
    task_panel=[]
    for task in TASKS:
        for bb in ('pi05','pi0'):
            if not any(r['task']==task and r['backbone']==bb for r in allrows):continue
            for method in METHODS:
                item=dict(backbone=bb,task=task,method=method,horizon='AHS' if method.endswith('ahs') else f'K={frozen["fixed_k"][bb]}')
                for delay in (0,100,200):
                    rs=[r for r in allrows if (r['task'],r['backbone'],r['method'],r['extra_ms'])==(task,bb,method,delay)]
                    complete=len(rs)==n_per_task;s=summarize(rs)
                    item.update({f'n_plus_{delay}':len(rs),f'successes_plus_{delay}':sum(r['success'] for r in rs) if complete else None,
                                 f'sr_plus_{delay}_pct':s['sr'] if complete else None})
                    if delay==100:
                        item.update({key:s[key] if complete else None for key in ('wait_pct','calls_ep','infer_s_ep')})
                task_panel.append(item)
    metadata=dict(generated_at=datetime.datetime.now().astimezone().isoformat(),panel_protocol_id=frozen['panel_protocol_id'] if frozen else None,
                  primary_completed=len(allrows),primary_total=formal_total,n_per_task=n_per_task,cost_delay_ms=100,
                  raw_formal_completed=len(raw_formal),excluded_k20_reference_completed=len(raw_formal)-len(allrows),
                  scope='Completed task cells only. Previous observations retained once; eight-task aggregate requires all tasks.')
    write_json(output/'formal_task_panel.json',dict(metadata=metadata,panel=task_panel))
    if task_panel:
        with (output/'formal_task_panel.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(task_panel[0]));w.writeheader();w.writerows(task_panel)
    lines=['# Current formal task panel','',f'As of {metadata["generated_at"]}; primary {len(allrows)}/{formal_total}; target {n_per_task} per task and condition.','',
           '| Model | Task | Method | Horizon | SR +0 ms | SR +100 ms | SR +200 ms | Wait %† | Calls/ep† | Infer s/ep† |',
           '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    for item in task_panel:
        sr=[f'{item[f"sr_plus_{d}_pct"]:.2f}% ({item[f"successes_plus_{d}"]}/{n_per_task})' if item[f'sr_plus_{d}_pct'] is not None
            else f'pending ({item[f"n_plus_{d}"]}/{n_per_task} completed)' for d in (0,100,200)]
        lines.append('| '+' | '.join([item[k] for k in ('backbone','task','method','horizon')]+sr+[fmt(item[k]) for k in ('wait_pct','calls_ep','infer_s_ep')])+' |')
    lines+=['','† Costs use +100 ms, all outcomes included. Model seconds are amortized batch time. +0 includes measured D0.',
            frozen.get('reporting_warning','') if frozen else '','']
    (output/'formal_task_panel.md').write_text('\n'.join(lines))
    timing=[]
    for bb in ('pi0','pi05'):
        for method in METHODS:
            rs=[r for r in allrows if r['backbone']==bb and r['method']==method and r['extra_ms']==100]
            complete=len(rs)==n_per_row
            data=timing_summary(rs) if complete else {}
            timing.append(dict(backbone=bb,method=method,n=len(rs),**{key:data.get(key) for key in (
                'request_to_commit_p50_ms','request_to_commit_p95_ms','handoff_after_ready_p95_ms',
                'action_age_mean_ms','selected_k_nonterminal_mean','actual_k_nonterminal_mean',
                'shortened_nonterminal_chunks','mean_physical_s_ep')}))
    with (output/'timing_diagnostics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(timing[0]));w.writeheader();w.writerows(timing)
    clock_text=['# Timing diagnostics accompanying Table 5','',
        f'Formal +100 ms only; controlled physical clock; all outcomes included. Empty until {n_per_row} episodes per row.','',
        '| Backbone | Method | Request→commit P95 ms | Ready→handoff P95 ms | Mean action age ms | Selected K | Actual K |',
        '|---|---|---:|---:|---:|---:|---:|']
    keys=('backbone','method','request_to_commit_p95_ms','handoff_after_ready_p95_ms',
          'action_age_mean_ms','selected_k_nonterminal_mean','actual_k_nonterminal_mean')
    clock_text+=['| '+' | '.join(fmt(r[k]) for k in keys)+' |' for r in timing]
    clock_text+=['','Latency quantiles pool accepted requests; mean action age weights executed actions. K means exclude episode-terminal chunks.',
                 'Request→commit includes specified inference delay and legal-boundary alignment. Ready→handoff isolates boundary alignment.',
                 'These metrics complement idle wait; none alone establishes deployment speed or SR gains. Native wall-clock results are in native_audit.json.','']
    (output/'timing_diagnostics.md').write_text('\n'.join(clock_text))
    progress={stage:summarize(allrows if stage=='formal' else records(stage)) for stage in ('calibration','native_check','smoke','pilot','validation','timeout','formal','strong_fixed','native')}
    reference=[r for r in raw_formal if r['method'].endswith('fixed') and r['fixed_k']!=frozen['fixed_k'][r['backbone']]] if frozen else []
    refs=defaultdict(list)
    for r in reference:refs[(r['backbone'],r['task'],r['method'],r['fixed_k'],r['extra_ms'])].append(r)
    write_json(output/'fixed_reference_per_task.json',[dict(backbone=k[0],task=k[1],method=k[2],fixed_k=k[3],extra_ms=k[4],n=len(rs),
        summary=summarize(rs) if len(rs)==50 else None) for k,rs in sorted(refs.items())])
    waves_path=EXP/'logs/waves.jsonl'
    waves=[json.loads(line) for line in waves_path.read_text().splitlines()] if waves_path.exists() else []
    eta={}
    for bb in ('pi0','pi05'):
        ws=[w for w in waves if w['backbone']==bb and w['stage'] in ('smoke','pilot','validation','formal')][-20:]
        if len(ws)>=2:
            sec_per_ep=sum(w['wall_s'] for w in ws)/sum(w['n'] for w in ws)
            completed=sum(r['backbone']==bb for r in allrows)
            eta[bb]=dict(observed_wave_count=len(ws),wall_s_per_episode=sec_per_ep,
                         remaining_formal_hours=(formal_total//2-completed)*sec_per_ep/3600,
                         scope='active primary formal only; excludes supplementary experiments, manifest setup and checkpoint switches; recent wave mix may include the previous fixed K')
        else:eta[bb]=dict(status='awaiting at least two non-calibration waves')
    write_json(EXP/'artifacts/eta.json',eta)
    write_json(EXP/'artifacts/progress.json',progress)
    native=primary_records(records('native'),frozen) if frozen else [];native_summary=[]
    for bb in ('pi0','pi05'):
        for method in METHODS:
            rs=[r for r in native if r['backbone']==bb and r['method']==method]
            if not rs:continue
            infer_lat=[];commit_lat=[];overlap=total_infer=0.
            for r in rs:
                event_path=Path(r['result_path']).parent/'events.jsonl'
                events=[json.loads(line) for line in event_path.read_text().splitlines()]
                motion=[e for e in events if e['event']=='action']
                for e in events:
                    if e['event'] not in ('commit','discard'):continue
                    start,end=e['request_wall'],e['available_wall']
                    infer_lat.append(end-start)
                    if e['event']=='commit':commit_lat.append(e['commit_wall']-start)
                    total_infer+=end-start
                    overlap+=sum(max(0.,min(end,a['end_wall'])-max(start,a['start_wall'])) for a in motion)
            native_summary.append(dict(backbone=bb,method=method,**summarize(rs),timing=timing_summary(rs),
                software_ready_p95_s=float(np.quantile(infer_lat,.95)) if infer_lat else None,
                commit_age_p95_s=float(np.quantile(commit_lat,.95)) if commit_lat else None,
                infer_execution_overlap_ratio=overlap/max(total_infer,1e-9),
                realtime_factor=sum(r['physics_s'] for r in rs)/sum(r['rollout_wall_s'] for r in rs),
                tick_deadline_misses=sum(r['native_deadline_misses'] for r in rs)))
    write_json(output/'native_audit.json',native_summary)
    print(json.dumps(progress,indent=2))

if __name__=='__main__':main()
