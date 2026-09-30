"""Independent analytic checks of the actual sampler against RTC Eq. 2--5.

Run with JAX_PLATFORMS=cpu to leave the evaluation GPU untouched.
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from common import soft_mask
from sampler import sample


class AffinePolicy:
    action_horizon = 4

    def __init__(self, matrix, bias, time_bias):
        self.matrix = jnp.asarray(matrix)
        self.bias = jnp.asarray(bias)
        self.time_bias = jnp.asarray(time_bias)
        self.PaliGemma = SimpleNamespace(llm=lambda streams, **kwargs: (tuple(streams), None))

    def embed_prefix(self, observation):
        return jnp.zeros((1, 1, 2)), jnp.ones((1, 1), bool), jnp.zeros((1,), bool)

    def embed_suffix(self, observation, x, t):
        v = (x.reshape(1, -1) @ self.matrix.T + self.bias + t[:, None] * self.time_bias).reshape(x.shape)
        return v, jnp.ones(x.shape[:2], bool), jnp.zeros((4,), bool), None

    def action_out_proj(self, x):
        return x


class RTCEquations(unittest.TestCase):
    def test_actual_sampler_matches_analytic_paper_time_direction_and_vjp(self):
        rng = np.random.default_rng(13)
        matrix = rng.normal(0, .03, (8, 8)) + np.eye(8) * .2
        bias, time_bias = rng.normal(0, .1, (2, 8))
        noise, target = rng.normal(0, .3, (2, 1, 4, 2))
        weights = np.array([1., .4, .1, 0.])[None, :, None] * np.ones((1, 4, 2))
        policy = AffinePolicy(matrix, bias, time_bias)
        with patch('sampler.base.preprocess_observation', side_effect=lambda _, o, **kwargs: o):
            actual, trace = sample(policy, None, jnp.asarray(noise, dtype=jnp.float32),
                                   jnp.asarray(target, dtype=jnp.float32),
                                   jnp.asarray(weights, dtype=jnp.float32), guided=True)
        # Independent Eq. 2--4 in the paper's tau=0->1 convention, using an
        # explicit analytic Jacobian instead of JAX autodiff.
        x = noise.reshape(-1).copy()
        expected_raw, expected_update = [], []
        for step in range(10):
            tau = step / 10
            paper_velocity = -(matrix @ x + bias + (1 - tau) * time_bias)
            clean = x + (1 - tau) * paper_velocity
            error = (target.reshape(-1) - clean) * weights.reshape(-1)
            gradient = error @ (np.eye(8) - (1 - tau) * matrix)
            coefficient = 10. if step == 0 else min(10., ((1 - tau) ** 2 + tau ** 2) / (tau * (1 - tau)))
            corrected = paper_velocity + coefficient * gradient
            expected_raw.append(-paper_velocity)
            expected_update.append(-corrected)
            x += corrected / 10
        np.testing.assert_allclose(np.asarray(actual).reshape(-1), x, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(np.asarray(trace['v']).reshape(10, 8), expected_raw, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(np.asarray(trace['guided_update']).reshape(10, 8), expected_update, atol=3e-6, rtol=3e-6)

    def test_mask_matches_paper_equation_five(self):
        for frozen, overlap in [(0, 0), (0, 30), (2, 32), (5, 15), (5, 5), (5, 50)]:
            expected = []
            for i in range(50):
                c = (overlap - i) / (overlap - frozen + 1)
                expected.append(1. if i < frozen else c * (np.exp(c) - 1) / (np.e - 1) if i < overlap else 0.)
            np.testing.assert_allclose(soft_mask(frozen, overlap), expected, atol=3e-8, rtol=1e-7)


if __name__ == '__main__':
    unittest.main()
