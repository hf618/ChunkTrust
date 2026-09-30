"""Replay stored real-robot decisions from recorded scores, without loading a model.

This audits recorded selection arithmetic; it cannot recover raw velocity spectra
or reproduce physical success from a trace of this kind.
"""
from pathlib import Path
import json
import numpy as np
from chunktrust._pi_ahs import Selector

p=Path(__file__).parent/'data/duck_replans.json'
d=json.loads(p.read_text())
for i,(scores,candidates,k) in enumerate(zip(d['ahs_replan_scores'],d['ahs_replan_candidates'],d['selected_K'],strict=True)):
    prob=Selector._normalize_score_distribution(np.asarray(scores),d['temperature'])
    expected=int(round(float(np.dot(prob,candidates))))
    assert expected==k,(i,expected,k)
    print(f'Replan {i:02d}: execute {k:2d} actions')
print(f'Checked {len(d["selected_K"])} recorded horizon decisions. {d["purpose"]}')
