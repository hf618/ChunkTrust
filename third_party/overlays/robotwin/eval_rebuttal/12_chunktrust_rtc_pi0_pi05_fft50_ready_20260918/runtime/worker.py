from __future__ import annotations
import math
import os
import threading
import time
import traceback
from pathlib import Path
import numpy as np
from common import EXP, append_json, write_json, rng_seed, observation_input
from selector import Selector
from contextlib import nullcontext

class NativeResponse:
    """Availability means the complete prediction arrived, not a Pipe header."""
    def __init__(self, conn):
        self.ready = threading.Event()
        self.response = self.error = self.ready_wall = None
        def receive():
            try:self.response = conn.recv()
            except BaseException as exc:self.error = exc
            finally:
                self.ready_wall = time.monotonic()
                self.ready.set()
        self.thread = threading.Thread(target=receive, daemon=True, name='native-prediction-receiver')
        self.thread.start()

    def get(self):
        self.thread.join()
        if self.error is not None:raise self.error
        return self.response

def episode(conn, job, prepared=None, render_lock=None):
    from scene import create, state_signature
    from controller import take_action_ticks
    start = time.perf_counter()
    entry = job['entry']
    env, initial = prepared if prepared is not None else create(job['task'],entry['episode_seed'],entry['rollout_id'],entry['instruction'])
    lock = render_lock if render_lock is not None else nullcontext()
    output = Path(job['output'])
    output.mkdir(parents=True,exist_ok=True)
    try:
        with lock:signature = state_signature(env,initial)
        if signature != entry['signature']:
            write_json(output/'reset_mismatch.json',dict(expected=entry['signature'],actual=signature))
            raise RuntimeError('fresh reset signature differs from frozen manifest')
        dt = float(env.scene.get_timestep())
        max_ticks = math.ceil(job['timeout_s']/dt)
        action_limit = min(env.step_lim,job.get('calibration_action_limit',env.step_lim))
        delay_ticks = math.ceil(job['delay_s']/dt - 1e-10)
        rtc, ahs = job['method'].split('_')
        rtc, ahs = rtc=='rtc', ahs=='ahs'
        candidates = job['candidates']
        selector = Selector(candidates,rng_seed(job['backbone'],job['task'],entry['episode_seed'],0,'ahs'))
        history, durations, actions = [], [], []
        source_indices = []
        cursor = committed_end = 0
        request_id = 0
        wait_ticks = 0
        model_s = selector_s = 0.
        calls = discarded = 0
        previous_source = None
        current_request_tick = 0
        current_request_wall = 0.
        action_ages, handoff_delays, request_ages = [], [], []
        selected_ks, actual_ks = [], []
        events = output/'events.jsonl'
        conn.send(('ready',dict(signature=signature,pid=os.getpid(),episode_seed=entry['episode_seed'])))
        if conn.recv() != 'go':raise RuntimeError('invalid start handshake')
        native = job.get('clock_mode','controlled') == 'native'
        wall_origin = time.monotonic()
        deadline_misses = 0
        max_tick_lag_s = 0.

        def pace():
            nonlocal deadline_misses,max_tick_lag_s
            if not native:return
            lag=time.monotonic()-(wall_origin+env._rtc_ticks*dt)
            max_tick_lag_s=max(max_tick_lag_s,lag)
            if lag>dt:deadline_misses+=1
            if lag<0:time.sleep(-lag)

        def done():
            return env.eval_success or env.take_action_cnt>=action_limit or env._rtc_ticks>=max_ticks

        def execute_one():
            nonlocal cursor
            tick = env._rtc_ticks
            action_wall = time.monotonic()
            action = actions[cursor]
            gen = take_action_ticks(env,action)
            completed = False
            while True:
                with lock:
                    try:next(gen)
                    except StopIteration:
                        completed = not env.eval_success
                        break
                pace()
                if env._rtc_ticks >= max_ticks:break
            elapsed = env._rtc_ticks-tick
            if elapsed:
                durations.append(elapsed*dt)
                history.append(np.asarray(action,np.float32))
            age = action_wall-current_request_wall if native else (tick-current_request_tick)*dt
            action_ages.append(age)
            append_json(events,dict(event='action',request_id=previous_source,original_index=int(source_indices[cursor]),
                      observation_tick=current_request_tick,start_tick=tick,end_tick=env._rtc_ticks,
                      start_wall=action_wall,end_wall=time.monotonic(),
                      observation_age_s=age,target=action,completed=completed))
            cursor += 1

        def finalize_chunk(reason):
            if previous_source is None:return
            selected_ks.append(committed_end);actual_ks.append(cursor)
            append_json(events,dict(event='chunk_end',request_id=previous_source,reason=reason,
                        selected_k=committed_end,actual_actions_started=cursor,
                        unexecuted_selected_actions=committed_end-cursor,end_tick=env._rtc_ticks))

        def hold_one():
            nonlocal wait_ticks
            with lock:
                env.scene.step();env._rtc_ticks+=1;wait_ticks+=1
                env._update_render()
                if env.check_success():env.eval_success=True
            pace()

        while not done():
            # Requests start at waypoint boundaries. Forecast uses frozen independent
            # calibration only, never TOPP-planning of future states.
            lead = math.ceil(job['delay_s']/job['waypoint_lower_s']) if rtc else 0
            if rtc and lead > min(candidates):
                raise RuntimeError(f'infeasible frozen lead={lead} for candidates={candidates}')
            trigger = max(cursor,committed_end-lead) if rtc else committed_end
            while cursor < trigger and not done():execute_one()
            if done():break
            request_tick = env._rtc_ticks
            remaining = committed_end-cursor if rtc else 0
            old = np.asarray(actions[cursor:],np.float32) if rtc and len(actions) else np.empty((0,14),np.float32)
            with lock:obs = observation_input(env.get_obs(),entry['instruction'])
            request = dict(observation=obs,previous=old,frozen=remaining,
                           rng_seed=rng_seed(job['backbone'],job['task'],entry['episode_seed'],request_id))
            request_wall = time.monotonic()
            conn.send(('infer',request))
            receiver = NativeResponse(conn) if native else None
            calls += 1
            ready_tick = request_tick+delay_ticks
            expired = 0
            old_cursor_at_request = cursor
            def prediction_ready():
                return receiver.ready.is_set() if native else env._rtc_ticks >= ready_tick
            # A forecast is a guidance mask, not a promise to run all masked
            # waypoints. At each legal boundary, prefer the newly available plan.
            # Do not interrupt an already-started TOPP trajectory. If late, never
            # extend the selected old K: hold its last drive target instead.
            while cursor < committed_end and not done() and not prediction_ready():
                execute_one();expired+=1
            while not prediction_ready() and not done():hold_one()
            response = receiver.get() if native else conn.recv()
            if response[0] != 'prediction':raise RuntimeError(str(response))
            result = response[1]
            response_wall = time.monotonic()
            if native:
                # Wall availability and an overloaded simulator's physical clock
                # are different coordinates. Keep the actual wall timestamp;
                # do not pretend its ideal 250 Hz index is a measured sim tick.
                ready_tick = None
            if result['jit_included']:raise RuntimeError('JIT warm-up leaked into rollout')
            model_s += result['model_amortized_s']
            if done():
                discarded+=1
                append_json(events,dict(event='discard',request_id=request_id,request_tick=request_tick,
                                        ready_tick=ready_tick,expired=expired,request_wall=request_wall,
                                        guidance_prefix_steps=remaining,
                                        available_wall=receiver.ready_wall if native else None,
                                        server_ready_wall=result['server_ready_wall'],response_wall=response_wall,
                                        model_amortized_s=result['model_amortized_s']))
                break
            if not 0 <= expired <= remaining:raise AssertionError('actual expired prefix exceeds selected old budget')
            new_actions = result['actions'][expired:]
            if len(new_actions) < max(candidates):
                raise RuntimeError('expired suffix cannot support the common candidate set')
            tick = time.perf_counter()
            if ahs:
                k, info = selector.select(result['trace']['v'][:,expired:],new_actions,history)
            else:
                k, info = job['fixed_k'],{}
            select_elapsed = time.perf_counter()-tick
            selector_s += select_elapsed
            commit_wall = time.monotonic()
            handoff_delay = commit_wall-receiver.ready_wall if native else (env._rtc_ticks-ready_tick)*dt
            request_age = commit_wall-request_wall if native else (env._rtc_ticks-request_tick)*dt
            handoff_delays.append(handoff_delay);request_ages.append(request_age)
            finalize_chunk('replaced_at_first_ready_boundary')
            append_json(events,dict(event='commit',request_id=request_id,request_tick=request_tick,
                         ready_tick=ready_tick,commit_tick=env._rtc_ticks,expired=expired,
                         guidance_prefix_steps=remaining,forecast_error_steps=expired-remaining,
                         old_cursor_at_request=old_cursor_at_request,
                         skipped_old_selected_actions=remaining-expired,
                         available_wall=receiver.ready_wall if native else None,commit_wall=commit_wall,
                         handoff_delay_s=handoff_delay,request_to_commit_s=request_age,
                         request_wall=request_wall,server_ready_wall=result['server_ready_wall'],response_wall=response_wall,
                         available=len(new_actions),selected_k=k,ahs=info,selector_s=select_elapsed,
                         model_amortized_s=result['model_amortized_s'],model_batch_s=result['model_batch_s'],
                         active_batch=result['active_batch'],padded_batch=result['padded_batch'],
                         correction_rms=result['trace']['correction_rms'],
                         raw_velocity_rms=float(np.sqrt(np.mean(result['trace']['v'][:,expired:,:14]**2))),
                         guided_update_rms=float(np.sqrt(np.mean(result['trace']['guided_update'][:,expired:,:14]**2)))))
            if entry['rollout_id']==0 and request_id<5:
                np.savez_compressed(output/f'trace_{request_id:04d}.npz',**result['trace'],actions=result['actions'],expired=expired)
            actions = new_actions
            source_indices = list(range(expired,50))
            cursor, committed_end = 0,k
            previous_source = request_id
            current_request_tick = request_tick
            current_request_wall = request_wall
            request_id += 1
        finalize_chunk('episode_terminal')
        result = {k:v for k,v in job.items() if k not in ('entry','output')}
        result.update(status='completed',episode_seed=entry['episode_seed'],rollout_id=entry['rollout_id'],
                      success=bool(env.eval_success),reason='success' if env.eval_success else 'physical_timeout' if env._rtc_ticks>=max_ticks else 'action_budget',
                      reset_signature=signature,physics_ticks=env._rtc_ticks,physics_s=env._rtc_ticks*dt,
                      wait_ticks=wait_ticks,wait_s=wait_ticks*dt,calls=calls,discarded_calls=discarded,
                      model_infer_s=model_s,selector_s=selector_s,wall_s=time.perf_counter()-start,
                      native_deadline_misses=deadline_misses,native_max_tick_lag_s=max_tick_lag_s,
                      handoff_delay_s=handoff_delays,request_to_commit_s=request_ages,
                      action_age_sum_s=sum(action_ages),action_age_count=len(action_ages),
                      action_age_p95_s=float(np.quantile(action_ages,.95)) if action_ages else None,
                      selected_k_by_chunk=selected_ks,actual_actions_by_chunk=actual_ks,
                      selected_k_before_handoff=selected_ks[:-1],actual_k_before_handoff=actual_ks[:-1],
                      controlled_clock=not native,rollout_wall_s=time.monotonic()-wall_origin,
                      realtime_factor=env._rtc_ticks*dt/max(time.monotonic()-wall_origin,1e-9),
                      actions_started=env.take_action_cnt,waypoint_durations_s=durations)
        write_json(output/'result.json',result,immutable=True)
        conn.send(('done',result))
    finally:
        with lock:env.close_env()

def simulator_host(connections,jobs):
    """Ten independently scheduled episode workers share one Vulkan/OIDN context.

    Setup is serialized before any policy clock starts (global setup RNGs).
    Each worker owns its scene, clock, queue, selector and request stream. Renderer
    access and individual PhysX ticks are interleaved under a lock; inference runs
    concurrently in the other process. No episode waits for another to finish.
    """
    import threading
    os.environ['JAX_PLATFORMS']='cpu'
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    os.environ['XDG_RUNTIME_DIR']='/tmp'
    log=EXP/'logs'/f'simulator_host_{os.getpid()}.log'
    with log.open('w',buffering=1) as stream:
        os.dup2(stream.fileno(),1);os.dup2(stream.fileno(),2)
        from scene import create
        prepared=[]
        try:
            for job in jobs:
                en=job['entry']
                prepared.append(create(job['task'],en['episode_seed'],en['rollout_id'],en['instruction']))
                print('ENV_PREPARED',len(prepared),len(jobs),en['episode_seed'],flush=True)
            lock=threading.RLock();errors=[]
            def work(conn,job,initial):
                try:episode(conn,job,initial,lock)
                except BaseException as exc:
                    error=dict(status='error',error_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
                    errors.append(error)
                    write_json(Path(job['output'])/'error.json',error)
                    traceback.print_exc()
                    try:conn.send(('error',error))
                    except Exception:pass
                finally:conn.close()
            threads=[threading.Thread(target=work,args=(c,j,e),name=f'episode-{j["entry"]["rollout_id"]}')
                     for c,j,e in zip(connections,jobs,prepared,strict=True)]
            for thread in threads:thread.start()
            for thread in threads:thread.join()
            if errors:raise RuntimeError(f'{len(errors)} episode worker errors')
        except BaseException as exc:
            error=dict(status='error',error_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
            for conn in connections:
                try:conn.send(('error',error))
                except Exception:pass
            raise

def worker_main(conn, job):
    os.environ['JAX_PLATFORMS']='cpu'
    os.environ['OMP_NUM_THREADS']='1'
    os.environ['OPENBLAS_NUM_THREADS']='1'
    os.environ['XDG_RUNTIME_DIR']='/tmp'
    log = Path(job['output'])/'worker.log'
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w',buffering=1) as stream:
        os.dup2(stream.fileno(),1);os.dup2(stream.fileno(),2)
        try:episode(conn,job)
        except BaseException as exc:
            error=dict(status='error',error_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
            write_json(log.parent/'error.json',error)
            traceback.print_exc()
            try:conn.send(('error',error))
            except Exception:pass
            raise
        finally:conn.close()
