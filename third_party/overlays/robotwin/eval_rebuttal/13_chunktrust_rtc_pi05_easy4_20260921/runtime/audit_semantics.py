"""Offline AHS replay and causal audit of completed episodes; no model/GPU use."""
import ast
import argparse
import copy
import json
import math
import time

import numpy as np

from common import EXP, ROOT, PROTOCOL_ID, rng_seed, write_json
from integrity import verify_upstream
from selector import Selector
from trace_io import load_trace


def original_selector(backbone, current):
    folder = 'pi0' if backbone == 'pi0' else 'pi05_horizon'
    tree = ast.parse((ROOT / 'policy' / folder / 'pi_model.py').read_text())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PI0')
    names = json.loads((EXP / 'artifacts/source_extraction.json').read_text())['selector_methods']
    methods = [n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name in names]
    source = 'from __future__ import annotations\nclass Original:\n' + ''.join(
        '    ' + line + '\n' for n in methods for line in ast.unparse(n).splitlines())
    namespace = {'np': np, 'math': math}
    exec(source, namespace)
    ref = namespace['Original']()
    ref.__dict__ = copy.deepcopy(current.__dict__)
    return ref


def comparable(value):
    return np.asarray([np.nan if x is None else x for x in value], dtype=np.float64)


def audit(stage=None, paths=None):
    verify_upstream()
    counts = dict(episodes=0, commits=0, actions=0, ahs_decisions_replayed=0,
                  ahs_trace_scores_recomputed=0, rtc_trace_scores_recomputed=0,
                  rtc_guided_vs_raw_k_differences=0, rtc_fft50_vs_suffix_k_differences=0,
                  rtc_fft50_vs_suffix_score_differences=0, inter_available_decisions=0,
                  frozen_prefix_exclusions=0, early_output_violations=0,
                  forecasts_larger_than_expired=0, first_ready_boundaries_checked=0,
                  finalized_chunks=0, native_commits_checked=0)
    boundary_delay = []
    by_backbone = {}
    for path in sorted(paths if paths is not None else (EXP / 'results').rglob('result.json')):
        row = json.loads(path.read_text())
        if stage is not None and row['stage'] != stage:continue
        if row['stage'] in ('memory', 'calibration'):
            continue
        assert row['protocol_id'] == PROTOCOL_ID and row['fft_horizon'] == 50
        events = [json.loads(line) for line in (path.parent / 'events.jsonl').read_text().splitlines()]
        ahs = row['method'].endswith('ahs')
        current = Selector(row['candidates'], rng_seed(row['backbone'], row['task'], row['episode_seed'], 0, 'ahs'))
        reference = original_selector(row['backbone'], current) if ahs else None
        history, commits, first_action = [], {}, set()
        all_actions = [e for e in events if e['event'] == 'action']
        ended = [e for e in events if e['event'] == 'chunk_end']
        assert sum(e['actual_actions_started'] for e in ended) == len(all_actions)
        assert len(ended) == len([e for e in events if e['event'] == 'commit'])
        for e in ended:
            assert 0 <= e['actual_actions_started'] <= e['selected_k']
            assert e['actual_actions_started'] == sum(a['request_id'] == e['request_id'] for a in all_actions)
            counts['finalized_chunks'] += 1
        counts['episodes'] += 1
        for event in events:
            if event['event'] == 'action':
                counts['actions'] += 1
                commit = commits[event['request_id']]
                assert event['start_tick'] >= commit['commit_tick']
                assert commit['expired'] <= event['original_index'] < commit['expired'] + commit['selected_k']
                if event['request_id'] not in first_action:
                    assert event['original_index'] == commit['expired']
                    first_action.add(event['request_id'])
                if event['end_tick'] > event['start_tick']:
                    history.append(np.asarray(event['target'], np.float32))
                continue
            if event['event'] != 'commit':
                continue
            counts['commits'] += 1
            commits[event['request_id']] = event
            if row['controlled_clock']:
                assert event['commit_tick'] >= event['ready_tick'] >= event['request_tick']
                old = [e for e in all_actions if e['request_id'] == event['request_id']-1
                       and e['start_tick'] >= event['request_tick']]
                assert len(old) == event['expired']
                assert all(e['start_tick'] < event['ready_tick'] for e in old)
                expected_boundary = max([event['ready_tick']] + [e['end_tick'] for e in old])
                assert event['commit_tick'] == expected_boundary
                counts['first_ready_boundaries_checked'] += 1
                if row['method'].startswith('rtc'):
                    dt = row['physics_s'] / row['physics_ticks']
                    boundary_delay.append((event['commit_tick'] - event['ready_tick']) * dt)
            else:
                assert event['commit_wall'] >= event['available_wall'] >= event['request_wall']
                counts['native_commits_checked'] += 1
            assert 0 <= event['expired'] <= event['guidance_prefix_steps']
            counts['forecasts_larger_than_expired'] += int(event['expired'] < event['guidance_prefix_steps'])
            assert event['available'] == 50 - event['expired']
            assert min(row['candidates']) <= event['selected_k'] <= max(row['candidates']) <= event['available']
            if not ahs:
                continue
            saved = event['ahs']
            assert saved['fft_horizon'] == 50 and saved['available_horizon'] == event['available']
            if any(x is not None for x in saved['p_inter']):
                counts['inter_available_decisions'] += 1
            trace_path = path.parent / f"trace_{event['request_id']:04d}.npz"
            before = copy.deepcopy(current)
            if trace_path.exists():
                trace = load_trace(trace_path)
                offset = event['expired']
                assert trace['expired'] == offset
                velocity = trace['v'][:, offset:, :14]
                actions = trace['actions'][offset:]
                reference._executed_action_history = np.asarray(history, np.float32).reshape(-1, 14)[-60:]
                expected = reference._compute_horizon_info(velocity, 50, xt_step=actions)
                expected_k, expected = reference._select_exec_k_horizon_from_info(expected, len(actions))
                k, info = current.select(velocity, actions, history)
                for name in ('z_intra', 'q_intra', 'p_inter', 'u_inter_raw', 'q_mix', 'scores'):
                    np.testing.assert_allclose(info[name], comparable(saved[name]), atol=2e-6, rtol=2e-6, equal_nan=True)
                    np.testing.assert_array_equal(info[name], expected[name])
                assert k == expected_k
                counts['ahs_trace_scores_recomputed'] += 1
                by_backbone[row['backbone']] = by_backbone.get(row['backbone'], 0) + 1
                if row['method'] == 'rtc_ahs' and offset:
                    counts['rtc_trace_scores_recomputed'] += 1
                    counts['frozen_prefix_exclusions'] += offset
                    alternate = copy.deepcopy(before)
                    guided_k, _ = alternate.select(trace['guided_update'][:, offset:], actions, history)
                    counts['rtc_guided_vs_raw_k_differences'] += int(guided_k != k)
                    alternate = copy.deepcopy(before)
                    # Diagnostic only: replay the superseded v1 suffix-length
                    # frequency grid. The live v2 always retains H=50.
                    alternate.fft_horizon = len(actions)
                    fft_k, fft_info = alternate.select(velocity, actions, history)
                    counts['rtc_fft50_vs_suffix_k_differences'] += int(fft_k != k)
                    counts['rtc_fft50_vs_suffix_score_differences'] += int(not np.allclose(fft_info['q_intra'], info['q_intra'], atol=1e-6))
            else:
                # Past the five saved traces, replay posterior/RNG decisions
                # using logged scores; this is not a raw-score recomputation.
                k, info = current._select_exec_k_horizon_from_info(saved, event['available'])
                expected_k, expected = reference._select_exec_k_horizon_from_info(saved, event['available'])
                assert k == expected_k
            assert k == event['selected_k']
            np.testing.assert_allclose(info['scores'], saved['scores'], atol=2e-6, rtol=2e-6)
            assert current._horizon_ts_state == reference._horizon_ts_state
            counts['ahs_decisions_replayed'] += 1
    result = dict(checked_at=time.time(),status='PASS',stage=stage,protocol_id=PROTOCOL_ID,counts=counts,
                  trace_recomputations_by_backbone=by_backbone,
                  rtc_extra_boundary_delay_ms_quantiles=dict(zip(('p50','p95','max'),
                      (1000*np.quantile(boundary_delay,[.5,.95,1.])).tolist())) if boundary_delay else {},
                  scope='Causal checks on all completed non-M0 episodes; original AHS formulas replayed at this panel settings. Only saved first-five traces support raw-score and shadow recomputation.',
                  qualification='PASS verifies H50 scoring and first-ready-boundary scheduling, not SR gains or fixed-Hz RTC equivalence. Target K and actual execution lengths are recorded separately; waypoint boundary granularity remains.')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage');args=parser.parse_args()
    result=audit(args.stage)
    write_json(EXP/'artifacts'/f'semantic_audit{("_"+args.stage) if args.stage else ""}.json',result)
    print(json.dumps(result,indent=2))
