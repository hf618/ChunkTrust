#!/usr/bin/env python3
"""Run one full-suite cell from the archived command recipe in a prepared checkout."""
import argparse,json,os,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--task',required=True);p.add_argument('--setting',choices=['demo_clean','demo_randomized'],required=True)
p.add_argument('--method',choices=['base','ahs'],required=True);p.add_argument('--backend-root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',default='0');p.add_argument('--dry-run',action='store_true')
a=p.parse_args();key=f'{a.task}__{a.setting}__{a.method}'
recipes=json.loads((ROOT/'configs/robotwin50/commands.json').read_text())
if key not in recipes:p.error('task/setting/method not in frozen command recipes')
replacements={'${CHUNKTRUST_ROOT}':str(ROOT),'${ROBOTWIN_ROOT}':str(a.backend_root.resolve()),'${OUTPUT_ROOT}':str(a.output.resolve())}
argv=recipes[key].copy()
for i,s in enumerate(argv):
 for k,v in replacements.items():s=s.replace(k,v)
 argv[i]=s
# eval.sh positional argument 6 is the GPU id; seed remains untouched.
argv[7]=a.gpu
print(shlex.join(argv),flush=True)
if not a.dry_run:
 env=os.environ.copy();env['PYTHONHASHSEED']='0';env['ROBOTWIN_ROOT']=str(a.backend_root.resolve())
 subprocess.run(argv,cwd=a.backend_root.resolve(),env=env,check=True)
