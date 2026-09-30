import argparse
import pickle
import time
import numpy as np
from common import EXP, write_json
from scene import create, state_signature
from controller import take_action_ticks

def main():
    p=argparse.ArgumentParser(); p.add_argument('--task',default='place_bread_basket'); p.add_argument('--seed',type=int,default=910000); a=p.parse_args()
    start=time.perf_counter()
    env,obs=create(a.task,a.seed)
    try:
        sig=state_signature(env,obs)
        path=EXP/'artifacts'/f'observation_{a.task}.pkl'
        with path.open('wb') as f:pickle.dump(obs,f)
        action=np.asarray(obs['state']).copy()
        for _ in take_action_ticks(env,action):pass
        write_json(EXP/'artifacts'/f'scene_probe_{a.task}.json',dict(seed=a.seed,signature=sig,elapsed_s=time.perf_counter()-start,hold_waypoint_ticks=env._rtc_ticks,step_lim=env.step_lim,timestep=env.scene.get_timestep()))
        print('SCENE_PROBE_OK',env._rtc_ticks,flush=True)
    finally:env.close_env()

if __name__=='__main__':main()
