#!/usr/bin/env python3
"""Download a released HF artifact at a pinned revision and verify each file."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('asset');p.add_argument('--destination',required=True,type=Path)
    p.add_argument('--catalog',type=Path,default=ROOT/'configs/assets.json',help='optional verified-assets catalog downloaded from HF')
    a=p.parse_args();catalog=json.loads(a.catalog.read_text())
    if a.asset not in catalog: p.error('unknown asset; available: '+', '.join(catalog))
    spec=catalog[a.asset]
    if spec.get('status')!='published':p.error('asset is not yet published; see docs/models.md')
    for item in spec['files']:
        dest=a.destination/item['local_path'];dest.parent.mkdir(exist_ok=True,parents=True)
        def matches(path):
            if not path.is_file() or path.stat().st_size!=item['bytes']:return False
            h=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
            return h.hexdigest()==item['sha256']
        if matches(dest):continue
        partial=dest.with_name(dest.name+'.partial')
        url=f'https://huggingface.co/{spec["repo_id"]}/resolve/{spec["revision"]}/{item["remote_path"]}'
        with urllib.request.urlopen(url,timeout=120) as src,partial.open('wb') as out:
            while block:=src.read(8*1024*1024):out.write(block)
        if not matches(partial):raise RuntimeError(f'checksum mismatch: {dest.name}')
        partial.replace(dest)
        print('Verified',item['local_path'])
    print('Asset complete:',a.asset)

if __name__=='__main__':main()
