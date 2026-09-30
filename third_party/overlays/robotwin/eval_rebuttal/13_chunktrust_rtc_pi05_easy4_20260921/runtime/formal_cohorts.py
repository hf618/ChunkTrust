"""Append formal seeds 50..99 while keeping the first-50 manifest byte-identical."""
import argparse
import hashlib
import json
from pathlib import Path
from common import EXP,TASKS,write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_extension(base,extra):
    assert len(base['entries'])==len(extra['entries'])==50
    assert [e['rollout_id'] for e in base['entries']]==list(range(50))
    assert [e['rollout_id'] for e in extra['entries']]==list(range(50,100))
    first={e['episode_seed'] for e in base['entries']}
    second={e['episode_seed'] for e in extra['entries']}
    assert len(first)==len(second)==50 and first.isdisjoint(second)
    last_checked=max([e['episode_seed'] for e in base['entries']]+[e['seed'] for e in base.get('rejections',[])])
    assert min(second)>last_checked
    start=base['seed_candidate_start']
    assert all(start<=seed<start+1000 for seed in first|second)


def write_combined(task):
    base_path=EXP/'manifests/formal'/f'{task}.json'
    extra_path=EXP/'manifests/formal_extra50'/f'{task}.json'
    base=json.loads(base_path.read_text());extra=json.loads(extra_path.read_text())
    assert extra['base_manifest_sha256']==digest(base_path)
    validate_extension(base,extra)
    result=dict(base,stage='formal100',entries=base['entries']+extra['entries'],
                rejections=base.get('rejections',[])+extra.get('rejections',[]),
                cohort_manifests={str(path.relative_to(EXP)):digest(path) for path in (base_path,extra_path)})
    write_json(EXP/'manifests/formal100'/f'{task}.json',result,immutable=True)
    return result


def build(task):
    from scene import create,state_signature
    from envs.utils.create_actor import UnStableError
    base_path=EXP/'manifests/formal'/f'{task}.json'
    base=json.loads(base_path.read_text());assert len(base['entries'])==50
    extra_path=EXP/'manifests/formal_extra50'/f'{task}.json'
    if extra_path.exists():write_combined(task);return
    start=max([e['episode_seed'] for e in base['entries']]+[e['seed'] for e in base.get('rejections',[])])+1
    entries=[];rejections=[]
    for seed in range(start,base['seed_candidate_start']+1000):
        env=None
        try:
            rollout_id=50+len(entries)
            env,obs=create(task,seed,rollout_id)
            entries.append(dict(rollout_id=rollout_id,episode_seed=seed,instruction=obs['prompt'],signature=state_signature(env,obs),precheck_status='accepted'))
            print('SCENE_ACCEPTED formal_extra50',task,len(entries),50,seed,flush=True)
            write_json(extra_path.with_suffix('.progress.json'),dict(accepted=len(entries),required=50,last_seed=seed))
        except UnStableError as exc:rejections.append(dict(seed=seed,type=type(exc).__name__,message=str(exc)))
        finally:
            if env is not None:env.close_env()
        if len(entries)==50:break
    assert len(entries)==50, 'Not enough setup-valid candidates within the original task seed interval'
    extra=dict(schema_version=1,protocol_id='chunktrust-rtc-v2-formal-extension',stage='formal_extra50',task=task,
        setting=base['setting'],precheck='setup_only_no_policy_or_expert_outcomes',
        seed_candidate_start=start,base_manifest_sha256=digest(base_path),entries=entries,rejections=rejections)
    validate_extension(base,extra)
    write_json(extra_path,extra,immutable=True);write_combined(task)
    print('FORMAL_100_IDENTITIES_READY',task,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=TASKS,required=True)
    build(parser.parse_args().task)
