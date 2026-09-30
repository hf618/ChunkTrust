"""User-requested scope: finish pi05 Hard four tasks, then Easy four tasks."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import time
import traceback
from common import EXP,METHODS,write_json
from n100_pipeline import freeze,block
from panel_run_block import panel_jobs,verify_saved_jobs
from focused_dashboard import config,scoped_rows
import pipeline as runner

ADOPTED=None

def identity(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
        if stat[0]=='Z':return None
        return dict(start_ticks=stat[19],command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode().strip())
    except FileNotFoundError:return None


def validate(c):
    assert c['backbones']==['pi05'] and c['tasks']==['place_a2b_left','place_bread_basket','place_bread_skillet','place_can_basket']
    assert c['order']==['Hard','Easy'] and c['rollouts_per_cell']==100
    p=freeze();assert p['execution_hash']==c['hard_execution_hash']
    e=Path(c['easy_root']);assert (e/'runtime/easy_pipeline.py').is_file()
    return p


def main(adopt_record=None,prepare_only=False):
    global ADOPTED
    c=config();p=validate(c)
    if prepare_only:print('FOUR_TASK_HARD_THEN_EASY_QUEUE_VALIDATED');return
    with (EXP/'pipeline.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
        (EXP/'pipeline.pid').write_text(str(os.getpid())+'\n')
        if adopt_record:
            a=json.loads(Path(adopt_record).read_text());pid=a['child_pid'];expected=a['child_identity']
            current=identity(pid)
            if current is not None:
                assert current==expected,'adoption PID identity changed'
                assert 'panel_run_block.py' in current['command'] and '--backbone pi05' in current['command']
                assert '--task place_can_basket' in current['command']
                ADOPTED=pid
                print('ADOPT_EXISTING_HARD_BLOCK',pid,flush=True)
                while identity(pid)==expected:
                    write_json(EXP/'artifacts/pipeline_status.json',dict(status='running',phase='hard_pi05_place_can_basket_adopted',pid=os.getpid(),child_pid=pid,command=a['command'],heartbeat=time.time(),started_at=a['started_at'],scope=c['queue_id']))
                    time.sleep(5)
                ADOPTED=None
        for task in c['tasks']:
            block('formal','pi05',task,runner.rotated(METHODS,'pi05',task),[0,100,200])
            args=argparse.Namespace(stage='formal',backbone='pi05',task=task,methods=list(METHODS),extra_ms=[0,100,200],fixed_k=40)
            assert verify_saved_jobs(panel_jobs(args,p))==0
        rows=scoped_rows(c);assert len(rows['Hard'])==4800
        write_json(EXP/'artifacts/pi05_four_hard_completed.json',dict(episodes=4800,completed_at=time.time(),scope=c['queue_id']))
        print('HARD_FOUR_COMPLETE_STARTING_EASY',flush=True)
        runner.run('easy_pi05_four_tasks',str(Path(c['easy_root'])/'runtime/easy_pipeline.py'),cpu=True)
        rows=scoped_rows(c);assert len(rows['Easy'])==4800
        write_json(EXP/'artifacts/pipeline_status.json',dict(status='completed',phase='pi05_four_hard_and_easy_complete',pid=os.getpid(),scope=c['queue_id'],hard_episodes=4800,easy_episodes=4800,completed_at=time.time()))
        print('REQUESTED_PI05_FOUR_HARD_EASY_QUEUE_COMPLETE',flush=True)


def stop(signum,frame):
    if ADOPTED:
        try:os.killpg(ADOPTED,signal.SIGTERM)
        except ProcessLookupError:pass
    runner.stop(signum,frame)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--adopt-record',type=Path);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:main(a.adopt_record,a.prepare_only)
    except BaseException as e:
        write_json(EXP/'artifacts/pipeline_status.json',dict(status='stopped_on_error',pid=os.getpid(),error=str(e),traceback=traceback.format_exc(),at=time.time()))
        raise
