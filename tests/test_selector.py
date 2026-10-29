import ast
import copy
import json
import math
from pathlib import Path
from typing import Any
import numpy as np
import pytest
from chunktrust import AHS, AHSConfig, HybridQHARuntimeSelector
from chunktrust._pi_ahs import Selector

ROOT = Path(__file__).resolve().parents[1]


def original_class(path, names, class_name='PI0'):
    tree=ast.parse(path.read_text())
    source=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name)
    members=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in names]
    missing=set(names)-{n.name for n in members}
    assert missing <= {'_uses_horizon_posterior'}  # PI0 always uses its posterior in this path.
    code='from __future__ import annotations\nclass Original:\n'+''.join('    '+line+'\n' for n in members for line in ast.unparse(n).splitlines())
    ns={'np':np,'math':math,'Any':Any};exec(code,ns)
    return ns['Original']


@pytest.mark.parametrize('policy',['pi0','pi05_horizon'])
def test_full_numerical_agreement_with_policy(policy):
    path=ROOT/f'third_party/overlays/robotwin/policy/{policy}/pi_model.py'
    names=[n.name for n in ast.parse((ROOT/'src/chunktrust/_pi_ahs.py').read_text()).body[-1].body if isinstance(n,ast.FunctionDef) and n.name not in {'__init__','select'}]
    Original=original_class(path,names)
    current=AHS(AHSConfig(candidates=(10,20,30,40)),seed=11)
    ref=Original();ref.__dict__=copy.deepcopy(current.__dict__)
    rng=np.random.default_rng(31)
    for i in range(30):
        v=rng.normal(size=(10,50,14));a=rng.normal(size=(50,14));history=rng.normal(size=(i*3,14)).astype(np.float32)
        ref._executed_action_history=history[-60:]
        expected=ref._compute_horizon_info(v,50,xt_step=a)
        ek,expected=ref._select_exec_k_horizon_from_info(expected,50)
        k,got=current.select(v,a,history)
        assert k==ek
        for field in ['z_intra','q_intra','p_inter','q_mix','scores']:
            np.testing.assert_array_equal(got[field],expected[field])
        assert current._horizon_ts_state==ref._horizon_ts_state


def test_suffix_fft_remains_full_horizon():
    rng=np.random.default_rng(42)
    v=rng.normal(size=(10,50,14));a=rng.normal(size=(50,14));history=rng.normal(size=(60,14))
    c=AHSConfig(candidates=(10,20,30,40))
    expected=AHS(c,seed=4).select(v,a,history)
    for available in [50,49,45,40]:
        k,info=AHS(c,seed=4).select(v[:,:available],a[:available],history)
        assert k==expected[0]
        np.testing.assert_array_equal(info['q_mix'],expected[1]['q_mix'])
    with pytest.raises(ValueError):AHS(c).select(v[:,:39],a[:39],history)


def test_json_resume_and_episode_reset():
    rng=np.random.default_rng(52);v=rng.normal(size=(10,50,14));a=rng.normal(size=(50,14))
    original=AHS(seed=7)
    original.select(v,a)
    resumed=AHS(seed=100)
    resumed.load_state_dict(json.loads(json.dumps(original.state_dict())))
    for _ in range(5):
        k1,i1=original.select(v,a);k2,i2=resumed.select(v,a)
        assert k1==k2
        np.testing.assert_array_equal(i1['scores'],i2['scores'])
    resumed.reset(7);k1,i1=AHS(seed=7).select(v,a);k2,i2=resumed.select(v,a)
    assert k1==k2
    np.testing.assert_array_equal(i1['scores'],i2['scores'])
    with pytest.raises(ValueError):AHS(AHSConfig(temperature=1)).load_state_dict(original.state_dict())


def test_prior_fusion_is_same_as_native_qha():
    text=(ROOT/'third_party/overlays/robotwin/policy/pi05_horizon/src/openpi/models/qha.py').read_text()
    nodes=[n for n in ast.parse(text).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in {'_normalize_score_distribution_np','HybridQHARuntimeSelector'}]
    ns={'np':np};exec('from __future__ import annotations\n'+'\n\n'.join(ast.get_source_segment(text,n) for n in nodes),ns)
    candidates=tuple(range(1,51));ref=ns['HybridQHARuntimeSelector'](candidate_horizons=candidates,ts_seed=8);got=HybridQHARuntimeSelector(candidate_horizons=candidates,ts_seed=8)
    rng=np.random.default_rng(13)
    for _ in range(20):
        p=rng.dirichlet(np.ones(50));q=rng.random(50)
        a=ref.select(p,q,max_exec_length=50);b=got.select(p,q,max_exec_length=50)
        for key in a:
            np.testing.assert_array_equal(a[key],b[key])
        assert ref._state==got._state


def test_evidence_does_not_advance_rng_or_memory():
    a=AHS();before=a.state_dict()
    info=a.evidence(np.ones((10,50,14)),np.zeros((50,14)))
    assert a.state_dict()==before
    assert np.isfinite(info['q_mix']).all()


@pytest.mark.parametrize('bad', [np.nan,np.inf])
def test_nonfinite_input_rejected(bad):
    v=np.ones((10,50,14));v[0,0,0]=bad
    with pytest.raises(ValueError):AHS().select(v,np.zeros((50,14)))


@pytest.mark.parametrize("field", ["horizon", "action_dim", "history_window"])
@pytest.mark.parametrize("bad", [True, False, np.bool_(True), 14.0, "14", None, np.nan, np.inf, 1j])
def test_config_dimensions_require_integers(field, bad):
    with pytest.raises(ValueError, match=field):
        AHSConfig(**{field: bad})


@pytest.mark.parametrize("bad", [None, "10,20", (), (True, 20), (np.bool_(True), 20),
                                  (10.0, 20), (10.5, 20), ("10", 20), (10, 10),
                                  (20, 10), (0, 20), (10, 51)])
def test_config_candidates_require_a_valid_integer_basis(bad):
    with pytest.raises(ValueError, match="candidate"):
        AHSConfig(candidates=bad)


@pytest.mark.parametrize("field", ["temperature", "epsilon", "kernel_bandwidth", "forget_rho", "update_eta"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, True, np.bool_(False), "0.8", None, 1j, 10**1000])
def test_config_parameters_require_finite_real_numbers(field, bad):
    with pytest.raises(ValueError, match=field):
        AHSConfig(**{field: bad})


@pytest.mark.parametrize("values", [
    {"horizon": 0}, {"action_dim": 0}, {"history_window": 3},
    {"temperature": 0}, {"temperature": -1}, {"epsilon": -0.1}, {"epsilon": 1.1},
    {"kernel_bandwidth": 0}, {"kernel_bandwidth": -1},
    {"forget_rho": 0}, {"forget_rho": 1.1}, {"update_eta": -1},
])
def test_config_preserves_original_parameter_bounds(values):
    with pytest.raises(ValueError):
        AHSConfig(**values)


@pytest.mark.parametrize("epsilon", [0, 1])
def test_config_accepts_numeric_endpoints_and_numpy_scalars(epsilon):
    config = AHSConfig(
        horizon=np.int64(50), action_dim=np.int32(14), history_window=np.int64(60),
        candidates=np.asarray([10, 20, 30, 40], dtype=np.int32),
        temperature=np.float64(1), epsilon=epsilon, kernel_bandwidth=10,
        forget_rho=1, update_eta=0,
    )
    assert type(config.horizon) is type(config.action_dim) is type(config.history_window) is int
    assert all(type(value) is int for value in config.candidates)
    for name in ("temperature", "epsilon", "kernel_bandwidth", "forget_rho", "update_eta"):
        assert type(getattr(config, name)) is float
    selector = AHS(config, seed=17)
    state = json.loads(json.dumps(selector.state_dict(), allow_nan=False))
    AHS(config).load_state_dict(state)


@pytest.mark.parametrize("field", ["velocity", "actions", "history"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_invalid_inputs_do_not_advance_posterior_or_rng(field, bad):
    selector = AHS(seed=11)
    inputs = {
        "velocity": np.ones((10, 50, 14)), "actions": np.zeros((50, 14)),
        "history": np.zeros((60, 14)),
    }
    selector.select(**inputs)
    before = selector.state_dict()
    inputs[field].flat[0] = bad
    with pytest.raises(ValueError, match="finite"):
        selector.select(**inputs)
    assert selector.state_dict() == before
