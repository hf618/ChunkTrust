#!/usr/bin/env python3
"""Check release file integrity, internal documentation links and accidental credentials."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]

def tracked_candidates():
    # The release is intentionally an allowlist, not a scan of workspaces/caches.
    roots=['src','configs','scripts','examples','environments','third_party','tests','docs','results','assets','.github']
    for name in roots:
        for p in (ROOT/name).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and not any(x.endswith('.egg-info') for x in p.parts) and p.suffix!='.pyc':yield p
    for name in ['README.md','LICENSE','pyproject.toml','.gitignore','CITATION.cff']:
        p=ROOT/name
        if p.is_file():yield p


def main():
    files=list(tracked_candidates());errors=[]
    secret=re.compile(r'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{25,}|ms-[a-f0-9]{8}-[a-f0-9-]{20,})')
    for p in files:
        if p.suffix in ['.png','.jpg','.npz']:continue
        s=p.read_text(errors='replace')
        if secret.search(s):errors.append(f'possible credential: {p.relative_to(ROOT)}')
        if p.suffix=='.md':
            for target in re.findall(r'\]\(([^)]+)\)',s):
                if '://' in target or target.startswith('#'):continue
                path=target.split('#',1)[0]
                if not (p.parent/path).exists():errors.append(f'broken local link: {p.relative_to(ROOT)} -> {path}')
    for record in json.loads((ROOT/'configs/assets.json').read_text()).values():
        if record['status']=='published':
            if not re.fullmatch(r'[a-f0-9]{40}',record['revision']):errors.append('asset revision must be immutable')
            for f in record['files']:
                if len(f['sha256'])!=64:errors.append('invalid asset checksum')
    if errors:raise SystemExit('\n'.join(errors))
    print(f'PASS: {len(files)} release files checked; no credential patterns or broken local Markdown links')

if __name__=='__main__':main()
