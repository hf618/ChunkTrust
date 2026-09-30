import copy
import sys
import unittest
import tempfile
import json
import time
import ast
import math
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from common import soft_mask,rng_seed
from selector import Selector
from controller import take_action_ticks
from original_controller import take_action

class Planner:
    def __init__(self,n,fail=False):self.n=n; self.fail=fail
    def TOPP(self,path,dt,verbose):
        if self.fail:raise RuntimeError('synthetic planner failure')
        p=np.linspace(path[0],path[-1],self.n);return None,p,np.zeros_like(p),None,None

class Fake:
    def __init__(self,ln=3,rn=5,success_at=999,fail=False):
        self.take_action_cnt=0;self.step_lim=30;self.eval_success=False;self.eval_video_path=None;self.render_freq=0
        self._rtc_ticks=0;self.ticks=0;self.success_at=success_at;self.events=[]
        self.scene=SimpleNamespace(step=self.step)
        self.robot=SimpleNamespace(get_left_arm_jointState=lambda:[0.]*7,get_right_arm_jointState=lambda:[0.]*7,
             get_left_gripper_val=lambda:0.,get_right_gripper_val=lambda:0.,
             left_mplib_planner=Planner(ln,fail),right_mplib_planner=Planner(rn),
             set_arm_joints=lambda p,v,side:self.events.append(('arm',side,tuple(p),tuple(v))),
             set_gripper=lambda p,side:self.events.append(('grip',side,float(p))))
    def step(self):self.ticks+=1;self.events.append(('step',self.ticks))
    def _update_render(self):self.events.append(('render',self.ticks))
    def check_success(self):return self.ticks>=self.success_at
    def get_obs(self):self.events.append(('obs',self.ticks))

class Contracts(unittest.TestCase):
    def test_metrics_include_failures_ratio_of_totals_and_paired_ci(self):
        from report import summarize,paired_ci
        rows=[]
        for seed in range(50):
            for method,wait,succ in [('rtc_ahs',1.,True),('sync_ahs',3.,False)]:
                rows.append(dict(backbone='pi0',task='toy',method=method,extra_ms=100,episode_seed=seed,
                       reset_signature={},success=succ,wait_s=wait,physics_s=10.,calls=2,model_infer_s=.3))
        s=summarize(rows)
        self.assertEqual(s['n'],100);self.assertEqual(s['sr'],50.);self.assertEqual(s['wait_pct'],20.)
        ci=paired_ci(rows,'pi0',100,('rtc_ahs','sync_ahs'),tasks=['toy'])
        self.assertEqual(ci['ci95_pp'],[100.,100.])
        self.assertAlmostEqual(ci['cost_delta']['wait_pp'],-20.)
        self.assertTrue(ci['noninferior']);self.assertTrue(ci['less_waiting'])
        self.assertIsNone(paired_ci(rows[:-1],'pi0',100,('rtc_ahs','sync_ahs'),tasks=['toy']))

    def test_frozen_selector_matches_both_original_backbones(self):
        root=Path(__file__).resolve().parents[3]
        meta=json.loads((Path(__file__).resolve().parents[1]/'artifacts/source_extraction.json').read_text())
        for backbone in ('pi0','pi05_horizon'):
            tree=ast.parse((root/'policy'/backbone/'pi_model.py').read_text())
            source=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PI0')
            methods=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in meta['selector_methods']]
            code='from __future__ import annotations\nclass Original:\n'+''.join('    '+line+'\n' for n in methods for line in ast.unparse(n).splitlines())
            ns={'np':np,'math':math};exec(code,ns)
            current=Selector(seed=11);ref=ns['Original']();ref.__dict__=copy.deepcopy(current.__dict__)
            rg=np.random.default_rng(31)
            for i in range(30):
                v=rg.normal(size=(10,50,14));a=rg.normal(size=(50,14));history=rg.normal(size=(i*3,14)).astype(np.float32)
                ref._executed_action_history=history[-60:]
                expected_info=ref._compute_horizon_info(v,50,xt_step=a)
                expected_k,expected_info=ref._select_exec_k_horizon_from_info(expected_info,50)
                k,info=current.select(v,a,history)
                self.assertEqual(k,expected_k)
                np.testing.assert_array_equal(info['scores'],expected_info['scores'])
                self.assertEqual(current._horizon_ts_state,ref._horizon_ts_state)

    def test_physical_clock_causality_and_rtc_expired_indices(self):
        from worker import episode
        results={}
        for method in ('sync_fixed','rtc_fixed','sync_ahs','rtc_ahs'):
            env=Fake(3,5);env.step_lim=5
            env.scene.get_timestep=lambda:.004
            obs={'state':np.zeros(14,dtype=np.float32),'images':{},'prompt':'test'}
            raw={'joint_action':{'vector':np.zeros(14)},'observation':{c:{'rgb':np.zeros((8,8,3),np.uint8)} for c in ('head_camera','left_camera','right_camera')}}
            env.get_obs=lambda:raw;env.close_env=lambda:None
            class Conn:
                def __init__(self):self.request_tick=None;self.messages=[]
                def send(self,msg):
                    self.messages.append(msg)
                    if msg[0]=='infer':self.request_tick=env._rtc_ticks
                def recv(self):
                    if self.request_tick is None:return 'go'
                    # Receiving the result itself is forbidden before release.
                    if env._rtc_ticks < self.request_tick+2:raise AssertionError('future output accessed early')
                    return ('prediction',dict(jit_included=False,actions=np.zeros((50,14)),
                         trace=dict(v=np.ones((10,50,14)),guided_update=np.ones((10,50,14)),correction_rms=np.zeros(10)),
                         model_amortized_s=.1,model_batch_s=1.,active_batch=10,padded_batch=10,server_ready_wall=time.monotonic()))
            with tempfile.TemporaryDirectory() as d:
                job=dict(backbone='pi0',task='toy',entry=dict(episode_seed=1,rollout_id=1,instruction='test',signature={}),
                         output=d,timeout_s=3.,delay_s=.008,waypoint_lower_s=.02,method=method,candidates=[1,2],fixed_k=2)
                scene=SimpleNamespace(create=lambda *args:(env,obs),state_signature=lambda *args:{})
                with patch.dict(sys.modules,{'scene':scene}):episode(Conn(),job)
                result=json.loads((Path(d)/'result.json').read_text())
                events=[json.loads(l) for l in (Path(d)/'events.jsonl').read_text().splitlines()]
                commits=[e for e in events if e['event']=='commit']
                self.assertTrue(all(e['commit_tick']>=e['ready_tick'] for e in commits))
                self.assertEqual(result['physics_ticks'],env.ticks)
                if method.startswith('rtc'):
                    self.assertTrue(any(e['expired']==1 for e in commits[1:]))
                    for c in commits[1:]:
                        matching=[e for e in events if e['event']=='action' and e['request_id']==c['request_id']]
                        if matching:self.assertEqual(matching[0]['original_index'],c['expired'])
                results[method]=result
        self.assertLess(results['rtc_fixed']['wait_s'],results['sync_fixed']['wait_s'])

    def test_controller_original_equivalence(self):
        for ln,rn,success,fail in [(3,5,999,False),(5,3,999,False),(3,5,2,False),(0,3,999,False),(3,5,999,True)]:
            a,b=Fake(ln,rn,success,fail),Fake(ln,rn,success,fail)
            action=np.linspace(-.1,.2,14)
            take_action(a,action)
            for _ in take_action_ticks(b,action):pass
            self.assertEqual(a.events,b.events)
            self.assertEqual(a.eval_success,b.eval_success)
            self.assertEqual(b._rtc_ticks,b.ticks)
    def test_one_tick_per_yield_and_terminal_tick_counted(self):
        f=Fake(success_at=2);g=take_action_ticks(f,np.zeros(14));next(g)
        self.assertEqual(f.ticks,1)
        with self.assertRaises(StopIteration):next(g)
        self.assertEqual(f._rtc_ticks,2)
    def test_mask(self):
        w=soft_mask(5,30)
        np.testing.assert_array_equal(w[:5],1)
        np.testing.assert_array_equal(w[30:],0)
        self.assertTrue(np.all(np.diff(w)<=0))
        self.assertTrue(np.all((w[5:30]>0)&(w[5:30]<1)))
        with self.assertRaises(ValueError):soft_mask(31,30)
    def test_rng_identity_and_no_method_dependency(self):
        self.assertEqual(rng_seed('pi0','a',1,0),rng_seed('pi0','a',1,0))
        self.assertNotEqual(rng_seed('pi0','a',1,0),rng_seed('pi0','a',1,1))
    def test_selector_expired_prefix_and_independent_state(self):
        rg=np.random.default_rng(5);v=rg.normal(size=(10,50,14));a=rg.normal(size=(50,14))
        left,right=Selector(seed=7),Selector(seed=7)
        k,info=left.select(v[:,5:],a[5:],[])
        v[:,:5]=np.nan
        k2,info2=right.select(v[:,5:],a[5:],[])
        self.assertEqual(k,k2);np.testing.assert_array_equal(info['scores'],info2['scores'])
        self.assertTrue(10<=k<=40)
        with self.assertRaises(ValueError):left.select(v[:,11:],a[11:],[])

if __name__=='__main__':unittest.main()
