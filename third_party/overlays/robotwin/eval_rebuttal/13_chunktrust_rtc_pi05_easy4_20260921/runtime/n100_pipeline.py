"""Resume the K40 panel and append paired identities 50..99 per task."""
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
from formal_cohorts import write_combined
import pipeline as runner
from priority_pipeline import ensure_manifest,select_supplementary_controls

PANEL_ID='chunktrust-rtc-v2-primary-k40-n100-staticcap-20260919'
PROTOCOL_FILE='artifacts/formal_protocol_k40_n100.json'
PRESERVED_FILE='artifacts/n100_transition_20260919_141426/preserved_files.json'


def verify_preserved():
    inventory=json.loads((EXP/PRESERVED_FILE).read_text())
    for relative,expected in inventory.items():
        assert hashlib.sha256((EXP/relative).read_bytes()).hexdigest()==expected, relative
    return len(inventory)


def freeze():
    verify_preserved()
    for stage in ('native_check','smoke','pilot'):verify_gate(stage)
    path=EXP/PROTOCOL_FILE
    if path.exists():
        protocol=json.loads(path.read_text())
        assert protocol['execution_hash']==execution_fingerprint() and protocol['panel_protocol_id']==PANEL_ID
        return protocol
    old_path=EXP/'artifacts/formal_protocol_k40.json';old=json.loads(old_path.read_text())
    assert old['formal_n_per_cell']==50 and old['execution_hash']==execution_fingerprint()
    assert not list((EXP/'manifests/formal_extra50').glob('*.json')), 'Freeze before extending identities'
    raw=records('formal');existing=primary_records(raw,old)
    protocol=copy.deepcopy(old)
    protocol.update(schema_version=5,panel_protocol_id=PANEL_ID,
        status='FROZEN_SAMPLE_SIZE_AMENDMENT_AFTER_PARTIAL_N50_OUTCOMES',
        frozen_at=datetime.datetime.now().astimezone().isoformat(),
        formal_n_per_cell=100,formal_total=19200,strong_fixed_n_per_cell=100,
        amendment_reason='User requested 100 rollouts and resume using existing seeds; include all prior completed identities without outcome filtering.',
        predecessor=dict(protocol_file=str(old_path.relative_to(EXP)),sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),panel_protocol_id=old['panel_protocol_id']),
        formal_seed_rule='Keep original first 50 setup-valid entries byte-identical. Continue the same per-task 1000-seed candidate interval to obtain the next 50 setup-valid identities. Reject only native UnStableError during scene setup; no policy or expert screening.',
        formal_cohorts=[dict(directory='formal',rollout_ids=[0,49]),dict(directory='formal_extra50',rollout_ids=[50,99])],
        combined_manifest_directory='formal100',
        outcome_visibility_at_sample_size_change=dict(primary_completed=len(existing),raw_formal_completed=len(raw)),
        preserved_inventory=PRESERVED_FILE,
        supplementary_selection_file='artifacts/strong_control_selection_k40_n100.json',
        reporting_warning=old['reporting_warning']+' Sample size increased from 50 to 100 per task after partial outcomes were available, at user request. All prior completed observations are retained once; this is a disclosed sample-size amendment, not a new independent replication.')
    write_json(path,protocol,immutable=True)
    return protocol


def block(stage,bb,task,methods,extras,k=40):
    runner.run(f'{stage}_{bb}_{task}_K{k}_n100_{"-".join(methods)}','panel_run_block.py',
        ['--stage',stage,'--backbone',bb,'--task',task,'--methods',*methods,'--extra-ms',*extras,
         '--fixed-k',k,'--panel-id',PANEL_ID],bb=bb)
    runner.run('report','report.py',cpu=True)


def verify_formal_complete(protocol):
    rows=primary_records(records('formal'),protocol)
    assert len(rows)==protocol['formal_total']
    for task in TASKS:
        seeds={e['episode_seed'] for e in write_combined(task)['entries']}
        for bb in ('pi0','pi05'):
            for method in METHODS:
                for extra in (0,100,200):
                    cell=[r for r in rows if (r['task'],r['backbone'],r['method'],r['extra_ms'])==(task,bb,method,extra)]
                    assert len(cell)==100 and {r['episode_seed'] for r in cell}==seeds
    verify_preserved()


def main(prepare_only=False):
    lock=(EXP/'pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
    protocol=freeze()
    pointer=EXP/'artifacts/active_panel.json'
    previous=json.loads(pointer.read_text())
    assert previous['panel_protocol_id'] in (protocol['predecessor']['panel_protocol_id'],PANEL_ID)
    write_json(pointer,dict(panel_protocol_id=PANEL_ID,protocol_file=PROTOCOL_FILE))
    write_json(EXP/'artifacts/n100_execution_sources.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((EXP/'runtime').glob('*.py'))})
    if prepare_only:print('N100_PANEL_FROZEN',flush=True);return
    # Finish one backbone's complete 9,600-episode panel before starting the
    # other backbone. This is a scheduling change only; every block remains
    # append-only and the resume verifier skips completed identities.
    for bb in ('pi05','pi0'):
        for task in TASKS:
            ensure_manifest('formal',task)
            if not (EXP/'manifests/formal100'/f'{task}.json').exists():
                runner.run(f'manifest_formal_extra50_{task}','formal_cohorts.py',['--task',task],cpu=True)
            write_combined(task)
            block('formal',bb,task,runner.rotated(METHODS,bb,task),[0,100,200])
    verify_formal_complete(protocol)
    write_json(EXP/'artifacts/primary_completed_k40_n100.json',dict(panel_protocol_id=PANEL_ID,episodes=19200,completed_at=time.time()))
    for bb in ('pi05','pi0'):
        for task in CAL_TASKS:
            for k in (10,20,30,40):runner.block('validation',bb,task,runner.rotated(('sync_fixed','rtc_fixed'),bb,task),[0,200],k)
            runner.block('validation',bb,task,runner.rotated(('sync_ahs','rtc_ahs'),bb,task),[0,200])
    selected=select_supplementary_controls(primary_k=40,panel_id=PANEL_ID,output_path=EXP/protocol['supplementary_selection_file'])
    for task in CAL_TASKS:
        source=EXP/'manifests/formal100'/f'{task}.json'
        dest=EXP/'manifests/strong_fixed'/f'{task}.json';dest.parent.mkdir(exist_ok=True)
        if dest.exists():assert dest.resolve()==source.resolve()
        else:dest.symlink_to(source)
        for bb in ('pi05','pi0'):
            for k in selected['strong_fixed_k'][bb]:block('strong_fixed',bb,task,['rtc_fixed'],[0,200],k)
    for task in ('handover_block','hanging_mug'):
        ensure_manifest('native',task)
        for bb in ('pi05','pi0'):block('native',bb,task,runner.rotated(METHODS,bb,task),[0])
    runner.run('report','report.py',cpu=True)
    verify_preserved()
    write_json(EXP/'artifacts/pipeline_status.json',dict(status='completed',panel_protocol_id=PANEL_ID,formal_episodes=19200,completed_at=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    signal.signal(signal.SIGTERM,runner.stop);signal.signal(signal.SIGINT,runner.stop)
    try:main(args.prepare_only)
    except BaseException as exc:
        write_json(EXP/'artifacts/pipeline_status.json',dict(status='stopped_on_error',pid=os.getpid(),error=str(exc),traceback=traceback.format_exc(),at=time.time()))
        traceback.print_exc();raise
