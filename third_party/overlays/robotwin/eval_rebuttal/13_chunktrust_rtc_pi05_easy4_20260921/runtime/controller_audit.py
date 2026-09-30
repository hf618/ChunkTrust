"""Actual simulator reset/controller equivalence; standalone CPU JAX process."""
import argparse
import copy
import json
import pickle
import time
import numpy as np
from common import EXP, write_json, observation_input
from scene import create,state_signature
from controller import take_action_ticks

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',default='place_bread_basket');a=p.parse_args()
    entry=json.loads((EXP/'manifests/calibration'/f'{a.task}.json').read_text())['entries'][0]
    runs=[]
    actions=np.load(EXP/'artifacts'/f'probe_actions_pi05.npy')[:10]
    for ticked in (False,True):
        env,obs=create(a.task,entry['episode_seed'],entry['rollout_id'],entry['instruction'],lazy_planning=ticked)
        try:
            sig=state_signature(env,obs)
            if sig!=entry['signature']:raise AssertionError('reset hash mismatch against precheck')
            snapshots=[]
            for action in actions:
                if ticked:
                    for _ in take_action_ticks(env,action):pass
                else:env.take_action(action)
                obs=observation_input(env.get_obs(),entry['instruction'])
                snapshots.append(dict(signature=state_signature(env,obs),success=env.eval_success,action_count=env.take_action_cnt))
                if env.eval_success:break
            runs.append(snapshots)
        finally:env.close_env()
    if runs[0]!=runs[1]:
        write_json(EXP/'artifacts'/f'controller_mismatch_{a.task}.json',runs)
        raise AssertionError('real controller trajectory mismatch')
    reference=EXP/'artifacts/controller_reference'/f'controller_audit_{a.task}.json'
    if reference.exists() and json.loads(reference.read_text())['snapshots']!=runs[0]:
        raise AssertionError('trajectory differs from original eager CuRobo setup')
    write_json(EXP/'artifacts'/f'controller_audit_{a.task}.json',dict(passed=True,waypoints=len(runs[0]),snapshots=runs[0]))
    print('REAL_CONTROLLER_EQUIVALENCE_OK',a.task,len(runs[0]),flush=True)

if __name__=='__main__':main()
