import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('summary',ROOT/'scripts/summarize_results.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_released_cohorts_have_complete_cells():
    for name,n in [('robotwin50/episode_metrics.csv',20),('qha_heldout/episodes.csv',100),('robocasa_pi05/episodes.csv',50),('rtc_pi05/hard_episodes.csv',100),('rtc_pi05/easy_episodes.csv',100)]:
        for row in module.summarize(ROOT/'results'/name):assert row['cell_counts']==[n]


def test_recomputed_robotwin50_matches_verified_aggregate():
    rows=module.summarize(ROOT/'results/robotwin50/episode_metrics.csv')
    assert {r['method']:round(r['sr_percent'],2) for r in rows}=={'base':56.7,'ahs':63.5}


def test_duplicate_seed_rejected(tmp_path):
    p=tmp_path/'bad.csv';p.write_text('task,setting,method,episode_seed,success\na,easy,ahs,1,1\na,easy,ahs,1,0\n')
    with pytest.raises(ValueError,match='duplicate'):module.summarize(p)


def test_equal_task_weight_not_pooled(tmp_path):
    p=tmp_path/'unequal.csv';p.write_text('task,setting,method,episode_seed,success\na,easy,ahs,1,1\na,easy,ahs,2,1\nb,easy,ahs,1,0\n')
    assert module.summarize(p)[0]['sr_percent']==50
