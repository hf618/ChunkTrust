"""Run the frozen K20 primary panel before supplementary K optimization.

User-requested schedule/protocol amendment, 2026-09-19. The execution code,
technical gates, fresh formal identities and four-way panel remain unchanged.
The primary comparator is explicitly K20, not validation-optimal K*.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import yaml

from common import EXP, ROOT, PROTOCOL_ID, TASKS, CAL_TASKS, METHODS, write_json
import pipeline as runner
from report import records
from run_block import execution_fingerprint
from semantic_gate import verify_gate

PANEL_ID = 'chunktrust-rtc-v2-primary-k20-staticcap-20260919'


def archive_interrupted_validation():
    """Preserve interrupted attempts, so later retries use the same identities."""
    pending = [p for p in (EXP/'results/validation').glob('**/ep*')
               if p.is_dir() and any(p.iterdir()) and not (p/'result.json').exists()]
    if not pending:
        return
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    archive = EXP/'interrupted_attempts'/f'priority_{stamp}'
    entries = []
    for source in sorted(pending):
        relative = source.relative_to(EXP)
        destination = archive/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        entries.append(dict(original=str(relative), archived=str(destination.relative_to(EXP)),
                            files={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest()
                                   for f in source.rglob('*') if f.is_file()}))
        source.rename(destination)
    write_json(archive/'archive_manifest.json', dict(
        reason='User requested formal-first scheduling; interruption is not an episode failure.',
        retry='Same validation identity, settings and RNG; no replacement seeds.', entries=entries), immutable=True)


def freeze_primary():
    gates = {stage: verify_gate(stage) for stage in ('native_check', 'smoke', 'pilot')}
    path = EXP/'artifacts/formal_protocol.json'
    if path.exists():
        protocol = json.loads(path.read_text())
        assert protocol['panel_protocol_id'] == PANEL_ID
        assert protocol['execution_hash'] == execution_fingerprint()
        assert protocol['fixed_k'] == {'pi0':20, 'pi05':20}
        return protocol
    assert not list((EXP/'results/formal').glob('**/result.json')), 'Never revise protocol after formal outcomes'
    assert not any((EXP/'results/formal').glob('**/events.jsonl')), 'Never revise after formal execution starts'
    limits_path = ROOT/'task_config/_eval_step_limit.yml'
    if not limits_path.exists():
        limits_path = ROOT/'config/_eval_step_limit.yml'
    limits = yaml.safe_load(limits_path.read_text())
    latency = copy.deepcopy(json.loads((EXP/'artifacts/latency_protocol.json').read_text()))
    for bb in ('pi0', 'pi05'):
        latency[bb].update(timeout_s=3600., timeout_status='frozen uniform 3600 s physical cap; original task action limit retained')
    selected = dict(
        fixed_k={'pi0':20, 'pi05':20}, strong_fixed_k={'pi0':[], 'pi05':[]},
        selection_mode='previously used K20 fixed for primary panel, not chosen by validation or formal outcomes',
        evidence='Original pilot/debug reference K20. User requested earlier formal evaluation after partial validation. Amendment is prospective for all formal identities.',
        supplementary_selection_file='artifacts/strong_control_selection.json')
    protocol = dict(
        schema_version=3, protocol_id=PROTOCOL_ID, panel_protocol_id=PANEL_ID,
        execution_hash=execution_fingerprint(), status='FROZEN_BEFORE_FORMAL',
        frozen_at=datetime.datetime.now().astimezone().isoformat(),
        tasks=list(TASKS), setting='demo_randomized', methods=list(METHODS),
        formal_n_per_cell=50, formal_total=9600, **selected,
        timeout_s={task:3600. for task in TASKS}, latency=latency,
        action_limits={task:int(limits.get(task, 1000)) for task in TASKS},
        action_limit_source=dict(path=str(limits_path), sha256=hashlib.sha256(limits_path.read_bytes()).hexdigest()),
        timeout_rule='Original per-task action budget plus uniform 3600 s physical cap, shared by all methods/backbones/delays. No fitted P99 deadline in this panel.',
        noninferiority_margin_pp=-3., bootstrap_repetitions=10000, bootstrap_seed=0,
        execution_contract=json.loads((EXP/'artifacts/control_protocol.json').read_text()),
        technical_gates={stage:dict(execution_hash=g['execution_hash'], evidence_hash=g['evidence_hash']) for stage,g in gates.items()},
        formal_seed_rule='Existing formal stage: 950000 + task index * 1000; first 50 setup-valid identities; no policy/outcome screening.',
        amendment_reason='User requested faster entry into formal evaluation on 2026-09-19.',
        outcome_visibility_before_amendment={stage:len(records(stage)) for stage in ('smoke','pilot','validation','formal')},
        scope='Four-way latency comparison against K20 under original action budgets. No claim against optimized fixed K or under a fitted deadline until corresponding supplemental evidence exists.',
        deferred=['remaining validation grid', 'validation-selected strong fixed controls', 'native wall-clock audit'],
        removed_prerequisite='Independent 320-episode P99 timeout cohort replaced by the explicit fixed cap; it does not silently change this formal panel later.')
    write_json(EXP/'artifacts/selected_k.json', selected, immutable=True)
    write_json(path, protocol, immutable=True)
    write_json(EXP/'artifacts/priority_amendment.json', protocol, immutable=True)
    return protocol


def ensure_manifest(stage, task):
    if not (EXP/'manifests'/stage/f'{task}.json').exists():
        runner.run(f'manifest_{stage}_{task}', 'manifests.py', ['--stage',stage,'--task',task], cpu=True)


def ensure_formal_settings(protocol, bb, task):
    settings = dict(protocol['latency'][bb], timeout_s=protocol['timeout_s'][task],
                    panel_protocol_id=PANEL_ID)
    path = EXP/'artifacts'/f'formal_settings_{bb}_{task}.json'
    write_json(path, settings, immutable=True)
    return path


def select_supplementary_controls(primary_k=20, panel_id=PANEL_ID, output_path=None):
    """Select only from the complete independent validation cohort."""
    output = output_path or EXP/'artifacts/strong_control_selection.json'
    if output.exists():
        result = json.loads(output.read_text())
        assert result['panel_protocol_id'] == panel_id
        assert result['fixed_k'] == {'pi0':primary_k,'pi05':primary_k}
        return result
    rows = records('validation')
    assert len(rows) == 3200, 'Require the complete balanced validation cohort'
    selected = {}; evidence = {}
    for bb in ('pi0','pi05'):
        ranking = []; rtc = []
        for k in (10,20,30,40):
            rs = [r for r in rows if r['backbone']==bb and r['fixed_k']==k and r['method'] in ('sync_fixed','rtc_fixed')]
            assert len(rs)==320
            ranking.append(dict(k=k,sr=float(np.mean([r['success'] for r in rs])),calls=float(np.mean([r['calls'] for r in rs]))))
            rr = [r for r in rs if r['method']=='rtc_fixed']
            rtc.append(dict(k=k,sr=float(np.mean([r['success'] for r in rr])),calls=float(np.mean([r['calls'] for r in rr]))))
        ahs = [r for r in rows if r['backbone']==bb and r['method']=='rtc_ahs']
        assert len(ahs)==160
        ahs_calls = float(np.mean([r['calls'] for r in ahs]))
        best = max(rtc,key=lambda x:(x['sr'],-x['calls'],x['k']))['k']
        budget = min(rtc,key=lambda x:(abs(x['calls']-ahs_calls),-x['sr'],-x['k']))['k']
        selected[bb] = sorted({best,budget}-{primary_k})
        evidence[bb] = dict(common_ranking=ranking,rtc_ranking=rtc,rtc_ahs_calls=ahs_calls,
                            rtc_best_k=best,nearest_call_budget_k=budget,
                            diagnostic_common_k=max(ranking,key=lambda x:(x['sr'],-x['calls'],x['k']))['k'])
    result = dict(panel_protocol_id=panel_id, fixed_k={'pi0':primary_k,'pi05':primary_k}, strong_fixed_k=selected,
                  selection_source='Complete independent validation only; formal outcomes never read for K selection.',
                  evidence=evidence)
    write_json(output,result,immutable=True)
    return result


def main(prepare_only=False):
    lock=(EXP/'pipeline.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
    archive_interrupted_validation()
    protocol=freeze_primary()
    write_json(EXP/'artifacts/priority_execution_sources.json', {
        p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((EXP/'runtime').glob('*.py'))})
    for task in TASKS:
        for bb in ('pi05','pi0'):
            ensure_formal_settings(protocol,bb,task)
    if prepare_only:
        print('PRIMARY_PROTOCOL_FROZEN',PANEL_ID,flush=True)
        return
    # Create each task's setup-only manifest immediately before its first block;
    # do not wait for all eight manifests before starting real formal episodes.
    for task in TASKS:
        ensure_manifest('formal',task)
        for bb in ('pi05','pi0'):
            runner.block('formal',bb,task,runner.rotated(METHODS,bb,task),[0,100,200],20,
                         EXP/'artifacts'/f'formal_settings_{bb}_{task}.json')
    assert len(records('formal'))==9600
    write_json(EXP/'artifacts/primary_completed.json',dict(panel_protocol_id=PANEL_ID,episodes=9600,completed_at=time.time()))
    # Resume all saved validation identities; run_block verifies saved hashes and
    # skips completed episodes. Interrupted attempts were archived, never erased.
    for bb in ('pi05','pi0'):
        for task in CAL_TASKS:
            ensure_manifest('validation',task)
            for k in (10,20,30,40):
                runner.block('validation',bb,task,runner.rotated(('sync_fixed','rtc_fixed'),bb,task),[0,200],k)
            runner.block('validation',bb,task,runner.rotated(('sync_ahs','rtc_ahs'),bb,task),[0,200])
    selected=select_supplementary_controls()
    strong_dir=EXP/'manifests/strong_fixed';strong_dir.mkdir(exist_ok=True)
    for task in CAL_TASKS:
        dest=strong_dir/f'{task}.json'
        if not dest.exists():dest.symlink_to(EXP/'manifests/formal'/f'{task}.json')
        for bb in ('pi05','pi0'):
            for k in selected['strong_fixed_k'][bb]:
                runner.block('strong_fixed',bb,task,['rtc_fixed'],[0,200],k,
                             EXP/'artifacts'/f'formal_settings_{bb}_{task}.json')
    for task in ('handover_block','hanging_mug'):
        ensure_manifest('native',task)
        for bb in ('pi05','pi0'):
            runner.block('native',bb,task,runner.rotated(METHODS,bb,task),[0],20,
                         EXP/'artifacts'/f'formal_settings_{bb}_{task}.json')
    runner.run('report','report.py',cpu=True)
    write_json(EXP/'artifacts/pipeline_status.json',dict(status='completed',panel_protocol_id=PANEL_ID,
               formal_episodes=9600,completed_at=time.time()))
    print('FULL_PRIORITY_PIPELINE_COMPLETED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    signal.signal(signal.SIGTERM,runner.stop);signal.signal(signal.SIGINT,runner.stop)
    try:
        main(args.prepare_only)
    except BaseException as exc:
        write_json(EXP/'artifacts/pipeline_status.json',dict(status='stopped_on_error',pid=os.getpid(),
                   error=str(exc),traceback=traceback.format_exc(),at=time.time()))
        traceback.print_exc();raise
