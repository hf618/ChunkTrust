"""Four-task pi05 Easy queue; paired n=100, append-only resume."""
import argparse
import fcntl
import json
import os
import signal
import time
import traceback
from common import EXP, METHODS, write_json
from easy_contract import verify
from formal_cohorts import write_combined
from panel_run_block import panel_jobs,verify_saved_jobs
import pipeline as runner


def main(prepare_only=False):
    with (EXP/'pipeline.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
        p=verify()
        if prepare_only:print('EASY_QUEUE_CONTRACT_PASS',p['execution_hash'],flush=True);return
        for task in p['tasks']:
            if not (EXP/'manifests/formal'/f'{task}.json').exists():
                runner.run(f'easy_manifest_{task}','manifests.py',['--stage','formal','--task',task],cpu=True)
            if not (EXP/'manifests/formal100'/f'{task}.json').exists():
                runner.run(f'easy_extend_{task}','formal_cohorts.py',['--task',task],cpu=True)
            manifest=write_combined(task);assert manifest['setting']=='demo_clean'
            methods=runner.rotated(METHODS,'pi05',task)
            args=['--stage','formal','--backbone','pi05','--task',task,'--methods',*methods,'--extra-ms',0,100,200,'--fixed-k',40,'--panel-id',p['panel_protocol_id']]
            runner.run(f'easy_pi05_{task}_n100','easy_run_block.py',args)
            check=argparse.Namespace(stage='formal',backbone='pi05',task=task,methods=methods,extra_ms=[0,100,200],fixed_k=40)
            assert verify_saved_jobs(panel_jobs(check,p))==0
        write_json(EXP/'artifacts/pipeline_status.json',dict(status='completed',pid=os.getpid(),formal_episodes=4800,completed_at=time.time()))
        print('EASY_FOUR_TASKS_COMPLETED',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');a=parser.parse_args()
    signal.signal(signal.SIGTERM,runner.stop);signal.signal(signal.SIGINT,runner.stop)
    try:main(a.prepare_only)
    except BaseException as e:
        write_json(EXP/'artifacts/pipeline_status.json',dict(status='stopped_on_error',pid=os.getpid(),error=str(e),traceback=traceback.format_exc(),at=time.time()))
        raise
