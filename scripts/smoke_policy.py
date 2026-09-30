#!/usr/bin/env python3
"""Load a real pi0.5 base plus QHA checkpoint and check two traced predictions."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--policy-root',type=Path,required=True)
p.add_argument('--base',type=Path,required=True,help='base checkpoint directory ending in its step')
p.add_argument('--qha',type=Path,required=True,help='QHA-only checkpoint directory ending in its step')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
sys.path[:0]=[str(a.policy_root.resolve()),str(a.policy_root.resolve()/'src')]
from pi_model import PI0
base=a.base.resolve();qha=a.qha.resolve()
model=PI0(train_config_name='pi05_base_aloha_robotwin_full',model_name=qha.parent.name,
          checkpoint_id=qha.name,checkpoint_root_tag=str(qha.parent.parent),pi0_step=50,
          base_checkpoint_root_tag=str(base.parent.parent),base_model_name=base.parent.name,
          base_checkpoint_id=base.name,use_qha=True,eval_type='horizon',valid_action_dim=14,
          qha_candidate_mode_override='dense_full',qha_hybrid_prior_gamma_override=1.0,
          qha_selector_expected_temp_override=1.0)
model.set_language('place the bread into the basket')
rng=np.random.default_rng(7);results=[]
for i in range(2):
    model.update_observation_window([rng.integers(0,256,(224,224,3),dtype=np.uint8) for _ in range(3)],np.zeros(14,np.float32))
    plan=model.get_action_plan();actions=np.asarray(plan['actions']);prior=np.asarray(plan['qha_posterior'])
    assert actions.shape==(50,14) and np.isfinite(actions).all()
    # The frozen inference path emits bfloat16 softmax values; check its
    # dtype precision in float32 without renormalizing or changing the policy.
    tolerance = 2 / 128 if str(prior.dtype) == 'bfloat16' else 1e-5
    prior_sum = float(prior.astype(np.float32).sum())
    assert prior.shape==(50,) and np.isfinite(prior).all() and abs(prior_sum-1)<tolerance, {'shape':prior.shape,'dtype':str(prior.dtype),'sum':prior_sum}
    assert 1<=int(plan['exec_length'])<=50
    for action in actions[:int(plan['exec_length'])]:model.record_executed_action(action)
    results.append({'actions_shape':list(actions.shape),'selected_k':int(plan['exec_length']),'prior_sum':prior_sum,'prior_dtype':str(prior.dtype),'sum_tolerance':tolerance})
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps({'status':'PASS','input_kind':'synthetic observations','rollout':False,'predictions':results},indent=2)+'\n')
print('Real checkpoint prediction smoke passed')
