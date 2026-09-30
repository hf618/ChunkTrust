from __future__ import annotations
import sys
import random
import hashlib
import json
import numpy as np
from common import ROOT, observation_input, observation_hash
sys.path[:0] = [str(ROOT),str(ROOT/'description/utils'),str(ROOT/'eval_rebuttal/scripts')]
import build_table4_episode_manifest as source
from build_clean50_literal_manifests import scene_info
from generate_episode_instructions import generate_episode_descriptions

def create(task, seed, rollout_id=0, instruction=None, *, lazy_planning=True):
    from lazy_planner import install
    install(enabled=lazy_planning)
    random.seed(seed)
    np.random.seed(seed)
    args = source._build_task_args(task,'demo_clean')
    args.update(collect_data=False,eval_video_log=False,eval_mode=True,render_freq=0)
    env = source._create_environment(task)
    try:
        env.setup_demo(now_ep_num=rollout_id,seed=seed,is_test=True,**args)
        if instruction is None:
            random.seed(seed)
            instruction = generate_episode_descriptions(task,[scene_info(task,env)],max_descriptions=1)[0]['unseen'][0]
            if '{' in instruction or '}' in instruction:
                raise ValueError(f'unresolved prompt: {instruction}')
        env.set_instruction(instruction=instruction)
        env._rtc_ticks = 0
        return env, observation_input(env.get_obs(),instruction)
    except BaseException:
        env.close_env()
        raise

def state_signature(env, obs):
    actors = sorted(env.scene.get_all_actors(),key=lambda x: (x.name, tuple(x.get_pose().p), tuple(x.get_pose().q)))
    physical = []
    for a in actors:
        physical.append([a.name, np.asarray(a.get_pose().p).tolist(), np.asarray(a.get_pose().q).tolist()])
    for a in env.scene.get_all_articulations():
        physical.append([a.name, np.asarray(a.get_qpos()).tolist(), np.asarray(a.get_qvel()).tolist()])
    sig = observation_hash(obs)
    sig['scene_sha256'] = hashlib.sha256(json.dumps(physical,sort_keys=True).encode()).hexdigest()
    return sig
