"""Exercise append-only manifests, old-record resume and the n=100 estimator."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
import formal_cohorts
import panel_run_block
from report import paired_ci
from common import write_json


class N100Resume(unittest.TestCase):
    def manifests(self):
        base=dict(task='toy',stage='formal',setting='demo_randomized',seed_candidate_start=950000,
                  entries=[dict(rollout_id=i,episode_seed=950000+i,signature={'seed':i},instruction='test') for i in range(50)],
                  rejections=[dict(seed=950050)])
        extra=dict(entries=[dict(rollout_id=i,episode_seed=950001+i,signature={'seed':i},instruction='test') for i in range(50,100)])
        return base,extra

    def test_extension_rejects_overlaps_and_wrong_numbering(self):
        base,extra=self.manifests();formal_cohorts.validate_extension(base,extra)
        for field,value in [('episode_seed',950000),('episode_seed',950050),('rollout_id',0)]:
            changed=copy.deepcopy(extra);changed['entries'][0][field]=value
            with self.assertRaises(AssertionError):formal_cohorts.validate_extension(base,changed)

    def test_real_job_builder_reuses_prefix_and_keeps_extension_hash_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base,extra=self.manifests()
            path=root/'manifests/formal/toy.json';write_json(path,base)
            before=path.read_bytes();extra['base_manifest_sha256']=hashlib.sha256(before).hexdigest()
            extra_path=root/'manifests/formal_extra50/toy.json';write_json(extra_path,extra)
            settings_path=root/'settings.json'
            write_json(settings_path,dict(candidates=[10,20,30,40],d0_s=.15,waypoint_lower_s=.08,timeout_s=3600.))
            protocol=dict(fixed_k={'pi0':40},ahs_record_k={'pi0':20},formal_n_per_cell=50)
            args=argparse.Namespace(stage='formal',backbone='pi0',task='toy',methods=['rtc_fixed','rtc_ahs'],extra_ms=[0,100,200],fixed_k=40)
            with patch('panel_run_block.EXP',root),patch('run_block.EXP',root),patch('formal_cohorts.EXP',root),patch('panel_run_block.settings_path',return_value=settings_path):
                old=panel_run_block.panel_jobs(args,protocol)
                for group in old:
                    job=group[0]
                    result={k:v for k,v in job.items() if k not in ('entry','output')}
                    result.update(status='completed',episode_seed=job['entry']['episode_seed'],reset_signature=job['entry']['signature'])
                    write_json(Path(job['output'])/'result.json',result)
                new=panel_run_block.panel_jobs(args,dict(protocol,formal_n_per_cell=100))
                self.assertEqual(len(new),6)
                for previous,extended in zip(old,new):
                    self.assertEqual(extended[:50],previous)
                    self.assertEqual(len(extended),100)
                    self.assertEqual(len({j['output'] for j in extended}),100)
                    self.assertEqual({j['manifest_hash'] for j in extended[50:]},{formal_cohorts.digest(extra_path)})
                    self.assertTrue(all(j['stage']=='formal' and j['clock_mode']=='controlled' for j in extended))
                self.assertEqual(panel_run_block.verify_saved_jobs(new),594)
                self.assertEqual(path.read_bytes(),before)
                combined=json.loads((root/'manifests/formal100/toy.json').read_text())
                self.assertEqual(combined['entries'][:50],base['entries'])
                saved=Path(new[0][0]['output'])/'result.json';result=json.loads(saved.read_text());result['manifest_hash']='wrong';write_json(saved,result)
                with self.assertRaisesRegex(RuntimeError,'manifest_hash'):panel_run_block.verify_saved_jobs(new)

    def test_n100_paired_estimator_requires_all_pairs_and_rejects_duplicates(self):
        rows=[dict(backbone='pi0',task='toy',method=m,extra_ms=100,episode_seed=i,
                   reset_signature={'seed':i},success=(i%2==0),wait_s=1.,physics_s=10.,calls=3,model_infer_s=.2)
              for m in ('rtc_ahs','sync_ahs') for i in range(100)]
        args=('pi0',100,('rtc_ahs','sync_ahs'))
        ci=paired_ci(rows,*args,tasks=['toy'],n_per_task=100)
        self.assertEqual(ci['n_pairs'],100);self.assertEqual(ci['ci95_pp'],[0.,0.])
        self.assertIsNone(paired_ci(rows[:-1],*args,tasks=['toy'],n_per_task=100))
        with self.assertRaisesRegex(AssertionError,'duplicate'):paired_ci(rows+[rows[0]],*args,tasks=['toy'],n_per_task=100)


if __name__=='__main__':unittest.main()
