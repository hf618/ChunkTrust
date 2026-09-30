"""Verify the clean-setting extension against the frozen Hard algorithm."""
import hashlib
import json
from pathlib import Path
from common import EXP, ROOT
from panel_config import load_active_protocol
from run_block import execution_fingerprint
from integrity import verify_upstream


def verify():
    p=load_active_protocol();parent=Path(p['parent_hard']['directory'])
    assert p['setting']=='demo_clean' and p['backbones']==['pi05']
    assert p['tasks']==['place_a2b_left','place_bread_basket','place_bread_skillet','place_can_basket']
    assert p['formal_total']==4800 and p['formal_n_per_cell']==100
    assert p['execution_hash']==execution_fingerprint()
    original=json.loads((parent/p['parent_hard']['protocol_file']).read_text())
    for name in p['algorithm_sources']:
        hard=(parent/'runtime'/name).read_bytes();easy=(EXP/'runtime'/name).read_bytes()
        if name=='scene.py':
            assert hard.count(b"'demo_randomized'")==1
            assert easy==hard.replace(b"'demo_randomized'",b"'demo_clean'")
        else:assert easy==hard,name
    for task in p['tasks']:
        for mode,path in p['settings_files']['pi05'][task].items():
            assert (EXP/path).read_bytes()==(parent/original['settings_files']['pi05'][task][mode]).read_bytes()
    tests=json.loads((EXP/'artifacts/unit_test_report.json').read_text())
    assert tests['status']=='PASS' and tests['execution_hash']==p['execution_hash']
    verify_upstream()
    for stage in ('native_check','smoke','pilot'):
        gate=json.loads((parent/'artifacts'/f'gate_{stage}.json').read_text())
        assert gate['status']=='PASS' and gate['execution_hash']==p['parent_hard']['execution_hash']
        entries={str(f.relative_to(parent)):hashlib.sha256(f.read_bytes()).hexdigest()
                 for f in sorted((parent/'results'/stage).rglob('*')) if f.name in ('result.json','events.jsonl')}
        assert gate['evidence_hash']==hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest()
    return p

if __name__=='__main__':print(verify()['panel_protocol_id'])
