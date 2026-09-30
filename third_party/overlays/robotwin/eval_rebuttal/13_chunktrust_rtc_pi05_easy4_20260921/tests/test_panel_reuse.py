"""Check adaptive reuse and prevent mixing fixed K20 into a K40 panel."""
import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from test_contracts import Fake
from worker import episode
from panel_config import primary_records


class PanelReuse(unittest.TestCase):
    def rollout(self,method,label):
        env=Fake(success_at=10**9);env.step_lim=85
        env.scene.get_timestep=lambda:.004
        raw={'joint_action':{'vector':np.zeros(14)},'observation':{c:{'rgb':np.zeros((8,8,3),np.uint8)} for c in ('head_camera','left_camera','right_camera')}}
        env.get_obs=lambda:raw;env.close_env=lambda:None
        obs={'state':np.zeros(14),'images':{},'prompt':'test'}
        def take_ticks(env,action):
            env.take_action_cnt+=1
            for _ in range(20):env.scene.step();env._rtc_ticks+=1;yield
        requests=[]
        class Conn:
            request=None
            def send(self,msg):
                if msg[0]=='infer':
                    self.request=msg[1];self.tick=env._rtc_ticks
                    requests.append((self.request['rng_seed'],self.request['frozen'],self.request['previous'].tolist()))
            def recv(self):
                if self.request is None:return 'go'
                assert env.take_action_cnt>=env.step_lim or env._rtc_ticks>=self.tick+math.ceil(.244/.004-1e-10)
                rng=np.random.default_rng(self.request['rng_seed'])
                return ('prediction',dict(jit_included=False,actions=rng.normal(size=(50,14)),
                    trace=dict(v=rng.normal(size=(10,50,14)),guided_update=np.ones((10,50,14)),correction_rms=np.zeros(10)),
                    model_amortized_s=.1,model_batch_s=1.,active_batch=10,padded_batch=10,server_ready_wall=time.monotonic()))
        with tempfile.TemporaryDirectory() as directory:
            job=dict(backbone='pi0',task='toy',entry=dict(episode_seed=13,rollout_id=0,instruction='test',signature={}),
                     output=directory,timeout_s=30.,delay_s=.244,waypoint_lower_s=.08,method=method,candidates=[10,20,30,40],fixed_k=label)
            scene=SimpleNamespace(create=lambda *a:(env,obs),state_signature=lambda *a:{})
            with patch.dict(sys.modules,{'scene':scene}),patch('controller.take_action_ticks',take_ticks),contextlib.redirect_stdout(io.StringIO()):
                episode(Conn(),job)
            result=json.loads((Path(directory)/'result.json').read_text())
            events=[json.loads(line) for line in (Path(directory)/'events.jsonl').read_text().splitlines()]
        keys=('event','request_id','request_tick','ready_tick','commit_tick','expired','selected_k','original_index','start_tick','end_tick','target')
        trajectory=[{k:e[k] for k in keys if k in e} for e in events]
        return requests,trajectory,{k:result[k] for k in ('success','calls','physics_ticks','wait_ticks','selected_k_by_chunk','actual_actions_by_chunk')}

    def test_real_adaptive_worker_ignores_fixed_control_label(self):
        for method in ('sync_ahs','rtc_ahs'):
            self.assertEqual(self.rollout(method,20),self.rollout(method,40))

    def test_fixed_worker_really_changes_execution_budget(self):
        for method in ('sync_fixed','rtc_fixed'):
            a=self.rollout(method,20);b=self.rollout(method,40)
            self.assertEqual(set(a[2]['selected_k_by_chunk']),{20})
            self.assertEqual(set(b[2]['selected_k_by_chunk']),{40})
            self.assertGreater(a[2]['calls'],b[2]['calls'])

    def test_panel_excludes_old_fixed_rows_but_keeps_shared_ahs(self):
        protocol=dict(fixed_k={'pi0':40},ahs_record_k={'pi0':20},execution_hash='code')
        rows=[dict(backbone='pi0',task='toy',method=m,fixed_k=k,extra_ms=0,episode_seed=1,execution_hash='code',settings_hash='settings')
              for m,ks in [('sync_fixed',[20,40]),('rtc_fixed',[20,40]),('sync_ahs',[20]),('rtc_ahs',[20])] for k in ks]
        with patch('panel_config.settings_digest',return_value='settings'):
            result=primary_records(rows,protocol)
            self.assertEqual(len(result),4)
            self.assertTrue(all(r['fixed_k']==40 for r in result if r['method'].endswith('fixed')))
            with self.assertRaisesRegex(RuntimeError,'duplicate'):primary_records(rows+[rows[-1]],protocol)
            with self.assertRaisesRegex(RuntimeError,'settings hash'):primary_records([dict(rows[-1],settings_hash='changed')],protocol)


if __name__=='__main__':unittest.main()
