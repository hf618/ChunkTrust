#!/usr/bin/env python3
"""Recompute equal-task success rates from released episode outcomes."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT=Path(__file__).resolve().parents[1]


def rate(value):
    if value in ('1','True','true'):return 1
    if value in ('0','False','false'):return 0
    raise ValueError(f'invalid outcome: {value!r}')


def summarize(path):
    rows=list(csv.DictReader(path.open()))
    groups=defaultdict(list);identities=set()
    for r in rows:
        setting=r.get('setting',path.stem.split('_')[0])
        ident=(r['task'],setting,r['method'],r.get('extra_ms',''),r['episode_seed'])
        if ident in identities:raise ValueError(f'duplicate episode identity: {ident}')
        identities.add(ident)
        groups[(r['method'],r.get('extra_ms',''))].append(dict(r,setting=setting))
    output=[]
    for (method,delay),values in sorted(groups.items()):
        cells=defaultdict(list)
        for r in values:cells[(r['task'],r['setting'])].append(rate(r['success']))
        tasks=defaultdict(list)
        for (task,setting),outcomes in cells.items():tasks[task].append(mean(outcomes))
        result={'method':method,'extra_ms':int(delay) if delay else None,'episodes':len(values),'tasks':len(tasks),'sr_percent':100*mean(mean(v) for v in tasks.values()),'cell_counts':sorted(set(map(len,cells.values())))}
        if 'wait_s' in values[0]:
            n=len(values);seconds=sum(float(r['physics_s']) for r in values)
            age_count=sum(int(r['action_age_count']) for r in values)
            result.update(wait_percent=100*sum(float(r['wait_s']) for r in values)/seconds,
                          wait_s_ep=mean(float(r['wait_s']) for r in values),calls_ep=mean(float(r['calls']) for r in values),
                          infer_s_ep=mean(float(r['model_infer_s']) for r in values),
                          observation_age_mean_s=sum(float(r['action_age_sum_s']) for r in values)/age_count)
        output.append(result)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);a=p.parse_args()
    files={'robotwin50':'robotwin50/episode_metrics.csv','qha_heldout':'qha_heldout/episodes.csv','robocasa_pi05':'robocasa_pi05/episodes.csv','rtc_hard':'rtc_pi05/hard_episodes.csv','rtc_easy':'rtc_pi05/easy_episodes.csv'}
    result={key:summarize(ROOT/'results'/name) for key,name in files.items()}
    text=json.dumps(result,indent=2)+'\n'
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
    else:print(text,end='')
