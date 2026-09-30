"""Separate Pi05 Hard/Easy and completed Pi0 views; queue scope stays unchanged."""
import datetime
import hashlib
import json
from pathlib import Path
from common import EXP
from panel_config import load_active_protocol, primary_records
from report import records,summarize


def config():return json.loads((EXP/'artifacts/pi05_four_queue.json').read_text())


def easy_rows(root):
    root=Path(root);pointer=json.loads((root/'artifacts/active_panel.json').read_text())
    p=json.loads((root/pointer['protocol_file']).read_text());rows=[];identities=set();cache={}
    def manifest(task,cohort):
        key=(task,cohort)
        if key not in cache:
            path=root/'manifests'/cohort/f'{task}.json';raw=path.read_bytes();data=json.loads(raw)
            assert data['setting']=='demo_clean'
            cache[key]=(hashlib.sha256(raw).hexdigest(),{e['rollout_id']:e for e in data['entries']})
        return cache[key]
    for path in (root/'results/formal').glob('**/result.json'):
        row=json.loads(path.read_text())
        assert row['status']=='completed' and row['backbone']=='pi05' and row['task'] in p['tasks']
        assert row['execution_hash']==p['execution_hash']
        mode='ahs' if row['method'].endswith('ahs') else 'fixed'
        settings=json.loads((root/p['settings_files']['pi05'][row['task']][mode]).read_text())
        assert row['settings_hash']==hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
        assert row['fixed_k']==(20 if mode=='ahs' else 40)
        mh,entries=manifest(row['task'],'formal' if row['rollout_id']<50 else 'formal_extra50')
        entry=entries[row['rollout_id']]
        assert row['manifest_hash']==mh and row['episode_seed']==entry['episode_seed'] and row['reset_signature']==entry['signature']
        identity=(row['task'],row['method'],row['extra_ms'],row['episode_seed'])
        assert identity not in identities;identities.add(identity);rows.append(row)
    return rows


def scoped_rows(c):
    hard=[r for r in primary_records(records('formal'),load_active_protocol()) if r['backbone']=='pi05' and r['task'] in c['tasks']]
    return {'Hard':hard,'Easy':easy_rows(c['easy_root'])}


def main():
    c=config();scopes=scoped_rows(c);s=json.loads((EXP/'artifacts/pipeline_status.json').read_text())
    protocol=load_active_protocol()
    pi0_rows=[r for r in primary_records(records('formal'),protocol) if r['backbone']=='pi0']
    pi0_tasks=[task for task in protocol['tasks'] if any(r['task']==task for r in pi0_rows)]
    easy_status=json.loads((Path(c['easy_root'])/'artifacts/pipeline_status.json').read_text())
    now=datetime.datetime.now().astimezone().isoformat();n=sum(map(len,scopes.values()))
    lines=['# Formal live panels — π0.5 and π0','',f'Last refreshed: {now}',
        f'Queue: **{s["status"]}**; phase: `{s.get("phase","—")}`; PID: {s.get("pid","—")}',
        f'Current requested scope: **{n:,}/9,600**; Hard **{len(scopes["Hard"]):,}/4,800**; Easy **{len(scopes["Easy"]):,}/4,800**.',
        f'Easy phase: `{easy_status.get("phase",easy_status["status"])}`.',
        f'π0 historical completed episodes: **{len(pi0_rows):,}**; shown separately below and excluded from the active π0.5 progress count.',
        'Order: finish all four π0.5 Hard tasks, then the same four Easy tasks. π0 and the remaining four Hard tasks are deferred; historical results are preserved.',
        'Hard = demo_randomized; Easy = demo_clean. H=50, fixed K=40, AHS candidates 10/20/30/40, 100 paired rollouts per method/delay/task, 10 scenes and padded model batch 10.',
        'Costs use +100 ms episodes only; final at n=100 per task/method. Partial SR is successes/completed. Hard and Easy are separate cohorts.', '']
    labels={'sync_fixed':'Sync K40','sync_ahs':'Sync AHS','rtc_fixed':'RTC K40','rtc_ahs':'RTC AHS'}
    def fmt(x):return '—' if x is None else f'{x:.2f}'
    tables=[(f'π0.5 — {setting}',rows,c['tasks']) for setting,rows in scopes.items()]
    tables.append(('π0 — Hard (completed records; queue deferred)',pi0_rows,pi0_tasks))
    for title,rows,tasks in tables:
        lines += [f'## {title}','', '| Task | Method | SR +0 ms | SR +100 ms | SR +200 ms | Wait % | Calls/ep | Infer s/ep |','|---|---|---:|---:|---:|---:|---:|---:|']
        for task in tasks:
            for method,label in labels.items():
                group=[r for r in rows if r['task']==task and r['method']==method];sr=[]
                for delay in (0,100,200):
                    cell=[r for r in group if r['extra_ms']==delay];success=sum(r['success'] for r in cell)
                    assert len(cell)<=100
                    sr.append(f'{success:.1f}%' if len(cell)==100 else f'{success}/{len(cell)}')
                cost=summarize([r for r in group if r['extra_ms']==100])
                lines.append('| '+' | '.join([task,label,*sr,fmt(cost['wait_pct']),fmt(cost['calls_ep']),fmt(cost['infer_s_ep'])])+' |')
        lines.append('')
    lines += ['Automatic refresh approximately every minute. TeX is not edited.', '']
    path=EXP/'artifacts/tables/formal_panel_live.md';path.write_text('\n'.join(lines));print(path)

if __name__=='__main__':main()
