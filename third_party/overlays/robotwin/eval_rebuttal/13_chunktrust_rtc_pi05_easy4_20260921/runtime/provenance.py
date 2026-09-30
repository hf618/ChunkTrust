import hashlib
import json
import re
import subprocess
from pathlib import Path
from common import EXP,ROOT,TASKS,CONFIGS,checkpoint,write_json

def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(8*1024*1024):h.update(block)
    return h.hexdigest()

def main():
    inventory=[]
    for backbone in CONFIGS:
        for task in TASKS:
            folder=checkpoint(backbone,task)
            if not (folder/'params').is_dir() or not (folder/'assets').is_dir():raise RuntimeError(folder)
            files=[]
            for p in sorted(folder.rglob('*')):
                if p.is_file():files.append(dict(path=str(p.relative_to(folder)),bytes=p.stat().st_size,sha256=file_hash(p)))
            inventory.append(dict(backbone=backbone,task=task,path=str(folder),files=files))
            print('CHECKPOINT_HASHED',backbone,task,flush=True)
    write_json(EXP/'artifacts/checkpoint_inventory.json',inventory,immutable=True)
    sources={}
    for folder in (ROOT/'envs',ROOT/'task_config',ROOT/'description',EXP/'runtime'):
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.suffix in ('.py','.json','.yml','.yaml'):sources[str(p.relative_to(ROOT))]=file_hash(p)
    for bb,(folder,_,_) in CONFIGS.items():
        for p in sorted((ROOT/'policy'/folder/'src/openpi').rglob('*.py')):sources[str(p.relative_to(ROOT))]=file_hash(p)
    write_json(EXP/'artifacts/source_inventory.json',sources)
    # Audit literal episode identities in historical JSON/JSONL, excluding this
    # experiment. Other integers (step counts, timestamps) are not seed evidence.
    old=[];overlap=[]
    files=subprocess.check_output(['rg','--files','--hidden','--no-ignore',str(ROOT/'eval_rebuttal')],text=True).splitlines()
    for name in files:
        p=Path(name)
        if EXP in p.parents or p.suffix not in ('.json','.jsonl'):continue
        if 'checkpoint_download' in p.parts or 'staging' in p.parts:continue
        with p.open(errors='replace') as f:
            matched=False
            for line in f:
                for value in re.findall(r'"episode_seed"\s*:\s*(\d+)',line):
                    matched=True;seed=int(value)
                    if 910000<=seed<990000:overlap.append(dict(file=str(p),seed=seed))
        if matched:old.append(str(p))
    write_json(EXP/'artifacts/seed_overlap_audit.json',dict(historical_files=old,overlaps=overlap,reserved_interval=[910000,990000]))
    if overlap:raise RuntimeError('new candidate seed range overlaps historical cohort')
    print('PROVENANCE_OK',len(inventory),len(old),flush=True)

if __name__=='__main__':main()
