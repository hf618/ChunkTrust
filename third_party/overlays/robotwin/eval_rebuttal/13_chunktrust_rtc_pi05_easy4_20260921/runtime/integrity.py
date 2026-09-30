import hashlib
import json
from common import EXP,ROOT

def verify_upstream():
    inventory=json.loads((EXP/'artifacts/source_inventory.json').read_text())
    for name,expected in inventory.items():
        path=ROOT/name
        if EXP in path.parents:continue
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'upstream source changed after provenance freeze: {name}')

def verify_checkpoint(path):
    inventory=json.loads((EXP/'artifacts/checkpoint_inventory.json').read_text())
    match=next(x for x in inventory if x['path']==str(path))
    for entry in match['files']:
        file=path/entry['path']
        if file.stat().st_size!=entry['bytes']:raise RuntimeError(f'checkpoint size changed: {file}')
        h=hashlib.sha256()
        with file.open('rb') as stream:
            while chunk:=stream.read(8*1024*1024):h.update(chunk)
        if h.hexdigest()!=entry['sha256']:raise RuntimeError(f'checkpoint content changed: {file}')
