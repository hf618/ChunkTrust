"""Technical, outcome-independent gates required before formal evaluations."""
import argparse
import hashlib
import json
import time
from common import EXP, PROTOCOL_ID, METHODS, write_json
from manifests import STAGES
from report import records
from run_block import execution_fingerprint
from audit_semantics import audit


def evidence_digest(stage):
    entries={}
    for path in sorted((EXP/'results'/stage).rglob('*')):
        if path.name in ('result.json','events.jsonl'):
            entries[str(path.relative_to(EXP))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest()


def verify_tests():
    tests=json.loads((EXP/'artifacts/unit_test_report.json').read_text())
    assert tests['status']=='PASS' and tests['execution_hash']==execution_fingerprint()
    assert tests['protocol_id']==PROTOCOL_ID


def verify_gate(stage):
    verify_tests()
    proof=json.loads((EXP/'artifacts'/f'gate_{stage}.json').read_text())
    assert proof['status']=='PASS' and proof['protocol_id']==PROTOCOL_ID
    assert proof['execution_hash']==execution_fingerprint()
    assert proof['evidence_hash']==evidence_digest(stage)
    return proof


def create_gate(stage):
    if (EXP/'artifacts'/f'gate_{stage}.json').exists():
        print(json.dumps(verify_gate(stage),indent=2));return
    verify_tests()
    methods=('rtc_fixed','rtc_ahs') if stage=='native_check' else METHODS
    extras=(0,) if stage=='native_check' else (0,200) if stage=='smoke' else (0,100,200)
    expected=set()
    for bb in ('pi0','pi05'):
        for task in STAGES[stage][2]:
            manifest=json.loads((EXP/'manifests'/stage/f'{task}.json').read_text())
            for method in methods:
                for extra in extras:
                    for entry in manifest['entries']:expected.add((bb,task,method,extra,entry['episode_seed']))
    rows=records(stage)
    actual=[tuple(r[k] for k in ('backbone','task','method','extra_ms','episode_seed')) for r in rows]
    assert len(actual)==len(set(actual)) and set(actual)==expected, 'missing or unexpected stage identities'
    assert not list((EXP/'results'/stage).rglob('error.json'))
    paired={}
    for row in rows:
        assert row['execution_hash']==execution_fingerprint() and row['fixed_k']==20
        key=(row['backbone'],row['task'],row['episode_seed'])
        if key in paired:assert paired[key]==row['reset_signature']
        paired[key]=row['reset_signature']
    proof=audit(stage)
    assert proof['counts']['episodes']==len(expected)
    assert proof['counts']['ahs_trace_scores_recomputed']>0
    if stage=='native_check':assert proof['counts']['native_commits_checked']>4
    else:
        assert proof['counts']['first_ready_boundaries_checked']>0
        assert proof['counts']['forecasts_larger_than_expired']>0
    proof.update(execution_hash=execution_fingerprint(),evidence_hash=evidence_digest(stage),expected_episodes=len(expected),
                 decision_rule='complete identities, strict reset pairing, fixed FFT50, first-ready-boundary causality and actual history; never gate on SR or significance')
    write_json(EXP/'artifacts'/f'gate_{stage}.json',proof,immutable=True)
    print(json.dumps(proof,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('native_check','smoke','pilot'));args=p.parse_args()
    create_gate(args.stage)
