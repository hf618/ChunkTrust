"""Explicit post-outcome K40 amendment, with immutable K20 reference data."""
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
from common import EXP,TASKS,CAL_TASKS,METHODS,write_json
from panel_config import primary_records
from report import records
from run_block import execution_fingerprint
from semantic_gate import verify_gate
import pipeline as runner
from priority_pipeline import ensure_manifest,select_supplementary_controls

PANEL_ID='chunktrust-rtc-v2-primary-k40-staticcap-20260919'
PROTOCOL_FILE='artifacts/formal_protocol_k40.json'


def archive_interrupted_formal():
    pending=[p for p in (EXP/'results/formal').glob('**/ep*') if p.is_dir() and any(p.iterdir()) and not (p/'result.json').exists()]
    if not pending:return
    destination=EXP/'interrupted_attempts'/('k40_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    entries=[]
    for source in sorted(pending):
        relative=source.relative_to(EXP);target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        entries.append(dict(original=str(relative),archived=str(target.relative_to(EXP)),
                            hashes={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source.rglob('*') if f.is_file()}))
        source.rename(target)
    write_json(destination/'archive_manifest.json',dict(reason='User selected K40; partial attempts preserved, never counted as failures or replaced by new seeds.',entries=entries),immutable=True)


def freeze():
    gates={stage:verify_gate(stage) for stage in ('native_check','smoke','pilot')}
    path=EXP/PROTOCOL_FILE
    if path.exists():
        protocol=json.loads(path.read_text())
        assert protocol['execution_hash']==execution_fingerprint() and protocol['panel_protocol_id']==PANEL_ID
        return protocol
    raw=records('formal')
    assert not any(r['method'].endswith('fixed') and r['fixed_k']==40 for r in raw), 'Freeze before any K40 formal fixed outcome'
    assert not list((EXP/'results/formal').glob('**/K40/**/events.jsonl')), 'Freeze before K40 starts'
    old_path=EXP/'artifacts/formal_protocol.json';old=json.loads(old_path.read_text())
    assert old['fixed_k']=={'pi0':20,'pi05':20} and old['execution_hash']==execution_fingerprint()
    protocol=copy.deepcopy(old)
    files={bb:{} for bb in ('pi0','pi05')}
    for bb in files:
        for task in TASKS:
            ahs=EXP/'artifacts'/f'formal_settings_{bb}_{task}.json'
            fixed=EXP/'artifacts'/f'formal_settings_k40_{bb}_{task}.json'
            settings=json.loads(ahs.read_text());settings['panel_protocol_id']=PANEL_ID
            write_json(fixed,settings,immutable=True)
            files[bb][task]=dict(ahs=str(ahs.relative_to(EXP)),fixed=str(fixed.relative_to(EXP)))
    protocol.update(
        schema_version=4,panel_protocol_id=PANEL_ID,status='FROZEN_BEFORE_K40_FIXED_ROLLOUTS_AFTER_PARTIAL_K20_OUTCOMES',
        fixed_k={'pi0':40,'pi05':40},ahs_record_k={'pi0':20,'pi05':20},settings_files=files,
        frozen_at=datetime.datetime.now().astimezone().isoformat(),
        selection_mode='User-specified long-execution K40 comparator after inspecting partial validation and K20 formal results; not validation-optimal or outcome-blind.',
        evidence='User: 那我感觉把基线设置为 K=40 感觉会好一点吧. Existing K20 protocol and results remain immutable reference evidence.',
        amendment_reason='User selected K40 to study adaptive replanning relative to a less frequent fixed replanning comparator.',
        predecessor=dict(protocol_file=str(old_path.relative_to(EXP)),sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),panel_protocol_id=old['panel_protocol_id']),
        inherited_ahs='Reuse every completed formal AHS identity with unchanged settings/execution hashes. K20 in AHS paths is a legacy label; the AHS branch never reads fixed_k. New AHS uses identical settings and RNG.',
        outcome_visibility_at_change=[dict(backbone=bb,method=method,completed=sum(r['backbone']==bb and r['method']==method for r in raw)) for bb in ('pi0','pi05') for method in METHODS],
        technical_gates={stage:dict(execution_hash=g['execution_hash'],evidence_hash=g['evidence_hash']) for stage,g in gates.items()},
        supplementary_selection_file='artifacts/strong_control_selection_k40.json',
        scope='Four-way comparison against specified K40 under unchanged action/time budgets. Baseline choice followed partial observed outcomes; no claim of validation optimality or fully outcome-blind confirmatory selection.',
        reporting_warning='K40 was selected after partial validation/K20 formal outcomes were observed. Original K20 results remain available; AHS rows are shared observations, never counted as independent replications. Strong fixed controls follow independent validation.')
    # Remove predecessor fields whose names could imply this amendment preceded all formal data.
    protocol.pop('outcome_visibility_before_amendment',None)
    write_json(path,protocol,immutable=True)
    reuse=[r for r in primary_records(raw,protocol) if r['method'].endswith('ahs')]
    write_json(EXP/'artifacts/k40_reuse_inventory.json',dict(panel_protocol_id=PANEL_ID,
        completed_ahs_reused=len(reuse),fixed_k20_reference_completed=sum(r['method'].endswith('fixed') for r in raw),
        outcomes_not_used_to_filter_reuse=True,
        source_results={str(Path(r['result_path']).relative_to(EXP)):hashlib.sha256(Path(r['result_path']).read_bytes()).hexdigest() for r in reuse}),immutable=True)
    return protocol


def block(stage,bb,task,methods,extras,k=40):
    runner.run(f'{stage}_{bb}_{task}_K{k}_panel40_{"-".join(methods)}','panel_run_block.py',
        ['--stage',stage,'--backbone',bb,'--task',task,'--methods',*methods,'--extra-ms',*extras,
         '--fixed-k',k,'--panel-id',PANEL_ID],bb=bb)
    runner.run('report','report.py',cpu=True)


def main(prepare_only=False):
    lock=(EXP/'pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
    # Only the initial, explicitly requested transition archives old interruptions.
    # A later execution failure requires a separate audited retry.
    if not (EXP/PROTOCOL_FILE).exists():archive_interrupted_formal()
    protocol=freeze()
    write_json(EXP/'artifacts/active_panel.json',dict(panel_protocol_id=PANEL_ID,protocol_file=PROTOCOL_FILE),immutable=True)
    write_json(EXP/'artifacts/k40_execution_sources.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((EXP/'runtime').glob('*.py'))})
    if prepare_only:print('K40_PANEL_FROZEN',flush=True);return
    for task in TASKS:
        ensure_manifest('formal',task)
        for bb in ('pi05','pi0'):block('formal',bb,task,runner.rotated(METHODS,bb,task),[0,100,200])
    assert len(primary_records(records('formal'),protocol))==9600
    write_json(EXP/'artifacts/primary_completed_k40.json',dict(panel_protocol_id=PANEL_ID,episodes=9600,completed_at=time.time()))
    for bb in ('pi05','pi0'):
        for task in CAL_TASKS:
            for k in (10,20,30,40):runner.block('validation',bb,task,runner.rotated(('sync_fixed','rtc_fixed'),bb,task),[0,200],k)
            runner.block('validation',bb,task,runner.rotated(('sync_ahs','rtc_ahs'),bb,task),[0,200])
    selected=select_supplementary_controls(primary_k=40,panel_id=PANEL_ID,output_path=EXP/protocol['supplementary_selection_file'])
    for task in CAL_TASKS:
        dest=EXP/'manifests/strong_fixed'/f'{task}.json';dest.parent.mkdir(exist_ok=True)
        if not dest.exists():dest.symlink_to(EXP/'manifests/formal'/f'{task}.json')
        for bb in ('pi05','pi0'):
            for k in selected['strong_fixed_k'][bb]:block('strong_fixed',bb,task,['rtc_fixed'],[0,200],k)
    for task in ('handover_block','hanging_mug'):
        ensure_manifest('native',task)
        for bb in ('pi05','pi0'):block('native',bb,task,runner.rotated(METHODS,bb,task),[0])
    runner.run('report','report.py',cpu=True)
    write_json(EXP/'artifacts/pipeline_status.json',dict(status='completed',panel_protocol_id=PANEL_ID,formal_episodes=9600,completed_at=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    signal.signal(signal.SIGTERM,runner.stop);signal.signal(signal.SIGINT,runner.stop)
    try:main(args.prepare_only)
    except BaseException as exc:
        write_json(EXP/'artifacts/pipeline_status.json',dict(status='stopped_on_error',pid=os.getpid(),error=str(exc),traceback=traceback.format_exc(),at=time.time()))
        traceback.print_exc();raise
