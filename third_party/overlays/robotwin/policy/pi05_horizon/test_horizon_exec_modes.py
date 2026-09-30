import numpy as np

from pi_model import PI0


class _ForbiddenPosteriorRng:
    def beta(self, *args, **kwargs):
        raise AssertionError("qmix_expected_round should not sample posterior")

    def random(self, *args, **kwargs):
        raise AssertionError("qmix_expected_round should not use epsilon randomness")

    def integers(self, *args, **kwargs):
        raise AssertionError("qmix_expected_round should not sample random horizons")


def _make_stub_model(exec_mode: str, mix_mode: str = "inter") -> PI0:
    model = PI0.__new__(PI0)
    model.horizon_candidates = [10, 20, 30]
    model.horizon_inter_use_speed_uniformity = True
    model.horizon_mix_mode = mix_mode
    model.horizon_inter_fallback = "ehh_only"
    model.horizon_inter_weight = 0.5
    model.horizon_inter_beta = 4.0
    model.horizon_intra_alpha = 4.0
    model.horizon_exec_mode = exec_mode
    model.horizon_expected_temp = 1.0
    model.horizon_ts_epsilon = 0.95
    model.horizon_ts_update_mode = "kernel_forget"
    model.horizon_ts_kernel_bandwidth = 10.0
    model.horizon_ts_forget_rho = 0.99
    model.horizon_ts_update_eta = 1.0
    model.horizon_intra_z_mode = "window_rms"
    model.horizon_intra_tau_window = 3
    model.horizon_intra_cut_t = 0.25
    model._horizon_ts_state = {10: [1000.0, 1.0], 20: [1.0, 1000.0], 30: [500.0, 500.0]}
    model._horizon_ts_rng = _ForbiddenPosteriorRng()
    model._speed_uniformity_proxy_by_horizon = lambda xt_step, horizons: (
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.1, 0.6, 0.3],
    )
    model._ehh_curves_by_horizon_batched = lambda v_step, horizons, h_full=None: {
        int(s): (np.asarray([0.0, 0.0], dtype=np.float64), 0.0) for s in horizons
    }
    model._compute_z_from_ehh = lambda e_vals, tau0_idx=0: 0.0
    return model


def test_qmix_expected_round_ignores_posterior_and_epsilon():
    model = _make_stub_model("qmix_expected_round")
    state_before = {k: list(v) for k, v in model._horizon_ts_state.items()}

    k_exec, info = model._select_exec_k_horizon(
        v_step=np.zeros((2, 30, 1), dtype=np.float32),
        used_h=30,
        xt_step=np.zeros((30, 1), dtype=np.float32),
    )

    assert k_exec == 22
    assert info["choose_mode"] == "qmix_expected_round"
    assert np.allclose(info["q_mix"], np.asarray([0.1, 0.6, 0.3], dtype=np.float32))
    assert np.allclose(info["scores"], info["q_mix"])
    assert np.isclose(info["expected_horizon"], 22.0)
    assert np.isnan(info["update_center"])
    assert np.isnan(info["update_feedback"])
    assert model._horizon_ts_state == state_before


def test_qmix_expected_round_policy_name_omits_posterior_update_mode():
    model = _make_stub_model("qmix_expected_round", mix_mode="both")

    assert model._horizon_policy_name() == "horizon_ehh1dpad_window_rms_speedu_both_qmix_expected_round_temp1"
