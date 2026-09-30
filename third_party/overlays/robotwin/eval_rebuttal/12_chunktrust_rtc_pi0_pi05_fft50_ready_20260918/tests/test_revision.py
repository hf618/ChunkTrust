import json
import math
import sys
import tempfile
import time
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from selector import Selector
from worker import episode, NativeResponse
from test_contracts import Fake


class RevisionContracts(unittest.TestCase):
    def test_native_availability_requires_complete_transfer(self):
        header = threading.Event()
        payload = threading.Event()
        class Conn:
            def recv(self):
                header.set()
                if not payload.wait(timeout=2):raise TimeoutError('test payload timeout')
                return ('prediction', {'complete': True})
        receiver = NativeResponse(Conn())
        self.assertTrue(header.wait(timeout=2))
        self.assertFalse(receiver.ready.is_set())
        payload.set()
        self.assertEqual(receiver.get(), ('prediction', {'complete': True}))
        self.assertTrue(receiver.ready.is_set())
        self.assertIsInstance(receiver.ready_wall, float)
    def test_fft_grid_stays_at_h50_when_available_suffix_shrinks(self):
        rng = np.random.default_rng(42)
        velocity = rng.normal(size=(10, 50, 14))
        actions = rng.normal(size=(50, 14))
        history = rng.normal(size=(60, 14))
        reference_k, reference = Selector(seed=4).select(velocity, actions, history)
        for available in (50, 49, 48, 47, 46, 45, 40):
            k, info = Selector(seed=4).select(velocity[:, :available], actions[:available], history)
            np.testing.assert_array_equal(info['z_intra'], reference['z_intra'])
            np.testing.assert_array_equal(info['q_mix'], reference['q_mix'])
            self.assertEqual(k, reference_k)

    def run_timeline(self, method, durations, delay_s, lower_s):
        env = Fake()
        env.step_lim = 7
        env.scene.get_timestep = lambda: .004
        obs = {'state': np.zeros(14), 'images': {}, 'prompt': 'test'}
        raw = {'joint_action': {'vector': np.zeros(14)}, 'observation': {
            c: {'rgb': np.zeros((8, 8, 3), np.uint8)} for c in ('head_camera', 'left_camera', 'right_camera')}}
        env.get_obs = lambda: raw
        env.close_env = lambda: None
        def take_ticks(env, action):
            n = durations[min(env.take_action_cnt, len(durations)-1)]
            env.take_action_cnt += 1
            for _ in range(n):
                env.scene.step()
                env._rtc_ticks += 1
                yield
        class Conn:
            request_tick = None
            def send(self, message):
                if message[0] == 'infer': self.request_tick = env._rtc_ticks
            def recv(self):
                if self.request_tick is None: return 'go'
                # Terminal requests are drained for accounting only; their
                # predictions never influence the terminated environment.
                assert env.take_action_cnt >= env.step_lim or env._rtc_ticks >= self.request_tick + math.ceil(delay_s/.004-1e-10)
                return ('prediction', dict(jit_included=False, actions=np.zeros((50, 14)),
                    trace=dict(v=np.ones((10, 50, 14)), guided_update=np.ones((10, 50, 14)), correction_rms=np.zeros(10)),
                    model_amortized_s=.1, model_batch_s=1., active_batch=10, padded_batch=10, server_ready_wall=time.monotonic()))
        with tempfile.TemporaryDirectory() as directory:
            job = dict(backbone='pi0', task='toy', entry=dict(episode_seed=1, rollout_id=1, instruction='test', signature={}),
                output=directory, timeout_s=3., delay_s=delay_s, waypoint_lower_s=lower_s, method=method, candidates=[4], fixed_k=4)
            scene = SimpleNamespace(create=lambda *args: (env, obs), state_signature=lambda *args: {})
            with patch.dict(sys.modules, {'scene': scene}), patch('controller.take_action_ticks', take_ticks):
                episode(Conn(), job)
            events = [json.loads(x) for x in (Path(directory)/'events.jsonl').read_text().splitlines()]
            result = json.loads((Path(directory)/'result.json').read_text())
        return events, result

    def test_ready_mid_waypoint_does_not_start_another_old_waypoint(self):
        for method in ('rtc_fixed', 'rtc_ahs'):
            events, result = self.run_timeline(method, [1, 1, 4, 1, 1, 1, 1], .012, .004)
            commits = [e for e in events if e['event'] == 'commit']
            next_commit = commits[1]
            # Forecast three old waypoints; output becomes ready during the
            # second. Switch immediately after it, leaving the third unstarted.
            self.assertEqual(next_commit['expired'], 2)
            self.assertEqual(next_commit['commit_tick'], next_commit['request_tick']+5)
            old = [e for e in events if e['event'] == 'action' and e['request_id'] == 0]
            self.assertTrue(all(e['start_tick'] < next_commit['ready_tick'] for e in old))
            new = next(e for e in events if e['event'] == 'action' and e['request_id'] == 1)
            self.assertEqual(new['original_index'], 2)
            self.assertEqual(new['start_tick'], next_commit['commit_tick'])

    def test_ready_exactly_at_boundary_switches_without_one_extra_action(self):
        events, result = self.run_timeline('rtc_fixed', [1]*7, .008, .004)
        commits = [e for e in events if e['event'] == 'commit']
        self.assertEqual(commits[1]['expired'], 2)
        self.assertEqual(commits[1]['commit_tick'], commits[1]['ready_tick'])

    def test_late_result_holds_after_old_k_budget_exhausted(self):
        events, result = self.run_timeline('rtc_ahs', [1]*7, .020, .008)
        commits = [e for e in events if e['event'] == 'commit']
        self.assertEqual(commits[1]['expired'], 3)
        self.assertEqual(commits[1]['commit_tick'], commits[1]['ready_tick'])
        old = [e for e in events if e['event'] == 'action' and e['request_id'] == 0]
        self.assertEqual(len(old), 4)
        self.assertGreater(result['wait_ticks'], 5)


if __name__ == '__main__': unittest.main()
