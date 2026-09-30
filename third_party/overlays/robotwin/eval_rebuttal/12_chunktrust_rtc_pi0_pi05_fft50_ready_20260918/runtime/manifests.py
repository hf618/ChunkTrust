from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
import traceback
from common import EXP, TASKS, CAL_TASKS, write_json, append_json

STAGES = {'calibration':(910000,2,CAL_TASKS),'smoke':(920000,5,('place_bread_basket','handover_block')),
          'pilot':(930000,10,CAL_TASKS),'validation':(940000,20,CAL_TASKS),
          'formal':(950000,50,TASKS),'native':(960000,10,('handover_block','hanging_mug')),
          'memory':(970000,10,('place_bread_basket',)), 'timeout':(980000,20,TASKS),
          'native_check':(970000,1,('place_bread_basket',))}

def build(stage, task):
    from scene import create,state_signature
    from envs.utils.create_actor import UnStableError
    start,count,tasks=STAGES[stage]
    if task not in tasks:raise ValueError((stage,task))
    output=EXP/'manifests'/stage/f'{task}.json'
    if output.exists():
        data=json.loads(output.read_text())
        if len(data['entries'])!=count:raise RuntimeError('manifest size mismatch')
        return
    entries=[];rejections=[]
    # Each task has a distinct candidate interval; all cohorts are disjoint.
    first=start+TASKS.index(task)*1000
    for seed in range(first,first+1000):
        env=None
        try:
            env,obs=create(task,seed,len(entries))
            signature=state_signature(env,obs)
            entry=dict(rollout_id=len(entries),episode_seed=seed,instruction=obs['prompt'],
                       signature=signature,precheck_status='accepted')
            entries.append(entry)
            print('SCENE_ACCEPTED',stage,task,len(entries),count,seed,flush=True)
            write_json(output.with_suffix('.progress.json'),dict(accepted=len(entries),required=count,last_seed=seed))
            if stage=='calibration' and len(entries)==1:
                with (EXP/'artifacts'/f'observation_{task}.pkl').open('wb') as f:pickle.dump(obs,f)
        except UnStableError as exc:
            rejections.append(dict(seed=seed,type=type(exc).__name__,message=str(exc)))
        finally:
            if env is not None:env.close_env()
        if len(entries)==count:break
    if len(entries)!=count:raise RuntimeError(f'only {len(entries)}/{count} scene-valid entries')
    write_json(output,dict(schema_version=1,protocol_id='chunktrust-rtc-v1',stage=stage,task=task,
               setting='demo_randomized',precheck='setup_only_no_policy_or_expert_outcomes',
               seed_candidate_start=first,entries=entries,rejections=rejections),immutable=True)
    print('MANIFEST_OK',stage,task,len(entries),len(rejections),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=STAGES,required=True);p.add_argument('--task',required=True)
    a=p.parse_args();build(a.stage,a.task)
