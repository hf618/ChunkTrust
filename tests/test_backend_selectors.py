import importlib
import numpy as np
import pytest


@pytest.mark.parametrize('backend',['starvla_ahs','gr00t_ahs'])
def test_gr1_selector_retains_episode_state(backend):
    selector=importlib.import_module('chunktrust.'+backend)
    rng=np.random.default_rng(22)
    v=rng.normal(size=(10,16,29)).astype(np.float32)
    actions=rng.normal(size=(16,29)).astype(np.float32)
    history=rng.normal(size=(20,29)).astype(np.float32)
    state={4:[1.,1.],8:[1.,1.],12:[1.,1.],16:[1.,1.]}
    common=dict(horizon_candidates=[4,8,12,16],xt_step=actions,history_actions=history,ts_state=state,ts_rng=np.random.default_rng(7))
    common['mix_mode' if backend=='starvla_ahs' else 'signal_mode']='both'
    k,info=selector.select_exec_k_horizon(v,16,**common)
    assert 1<=k<=16
    assert np.isfinite(info['q_mix']).all()
    assert any(a!=1 or b!=1 for a,b in state.values())
