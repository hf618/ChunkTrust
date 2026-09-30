"""Read-only result observer; publish fresh counts without interrupting rollouts."""
import json
import time
from collections import defaultdict
from common import EXP,write_json
from panel_config import load_active_protocol,primary_records

def main():
    seen={}
    last_md=0.
    while True:
        for path in (EXP/'results').glob('**/result.json'):
            if str(path) in seen:continue
            row=json.loads(path.read_text())
            if row['status']!='completed':continue
            seen[str(path)]={k:row[k] for k in ('stage','backbone','task','method','fixed_k','extra_ms','success','calls','episode_seed','execution_hash','settings_hash','rollout_id','manifest_hash','reset_signature')}
        cells=defaultdict(list)
        raw_formal=[r for r in seen.values() if r['stage']=='formal']
        protocol=load_active_protocol()
        active_formal=primary_records(raw_formal,protocol) if protocol else raw_formal
        current=[r for r in seen.values() if r['stage']!='formal']+active_formal
        for row in current:
            key=tuple(row[k] for k in ('stage','backbone','task','method','fixed_k','extra_ms'))
            cells[key].append(row)
        items=[];stages=defaultdict(lambda:dict(completed=0,successes=0,calls=0))
        for key,rows in sorted(cells.items()):
            stage,bb,task,method,k,extra=key
            manifest=json.loads((EXP/'manifests'/stage/f'{task}.json').read_text())
            total=protocol['formal_n_per_cell'] if stage=='formal' and protocol else len(manifest['entries'])
            successes=sum(r['success'] for r in rows)
            items.append(dict(stage=stage,backbone=bb,task=task,method=method,fixed_k=k,extra_ms=extra,
                              completed=len(rows),expected=total,successes=successes,
                              sr_pct_if_cell_complete=100*successes/total if len(rows)==total else None))
            stages[stage]['completed']+=len(rows);stages[stage]['successes']+=successes
            stages[stage]['calls']+=sum(r['calls'] for r in rows)
        status=json.loads((EXP/'artifacts/pipeline_status.json').read_text())
        write_json(EXP/'artifacts/live_progress.json',dict(updated_at=time.time(),pipeline=status,stages=dict(stages),
                    cells=items,active_panel=protocol.get('panel_protocol_id') if protocol else None,
                    raw_formal_completed=len(raw_formal),fixed_reference_completed=len(raw_formal)-len(active_formal),
                    note='Formal counts refer to the active panel only. Previous fixed-K results remain reference data. AHS record K is a storage label, not its selected length. Incomplete cells have no SR.'))
        if time.monotonic()-last_md>=60 or status['status'] in ('completed','stopped_on_error'):
            from update_live_panel_md import main as update_md
            update_md();last_md=time.monotonic()
        if status['status'] in ('completed','stopped_on_error'):break
        time.sleep(10)

if __name__=='__main__':main()
