"""Regression tests for participant-cluster uncertainty in the exploratory checks."""
import unittest

import numpy as np

from analyze_reviewer_checks import cluster_weight_draws, logit_fit


class ParticipantBootstrapTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.X = np.column_stack([np.ones(60), rng.normal(size=(60, 2))])
        self.y = 1 / (1 + np.exp(-(self.X @ np.array([.2, .7, -.3]))))
        self.y += rng.uniform(-.06, .06, 60)
        self.pid = np.repeat([10, 20, 30, 40, 50, 60], 10)

    def test_cluster_weights_equal_explicit_resampling(self):
        seed = 4
        ids, draws = cluster_weight_draws(self.X, self.y, self.pid, n_boot=1, seed=seed)
        selected = np.random.default_rng(seed).integers(len(ids), size=len(ids))
        rows = np.concatenate([np.flatnonzero(self.pid == ids[i]) for i in selected])
        np.testing.assert_allclose(draws[0], logit_fit(self.X[rows], self.y[rows]), atol=1e-10)

    def test_identical_agents_have_zero_paired_bootstrap_difference(self):
        _, first = cluster_weight_draws(self.X, self.y, self.pid, n_boot=20)
        # Different input ordering must preserve the shared participant draws.
        ix = np.arange(len(self.y))[::-1]
        _, second = cluster_weight_draws(self.X[ix], self.y[ix], self.pid[ix], n_boot=20)
        np.testing.assert_allclose(first, second, atol=1e-10)

    def test_single_participant_is_not_an_independent_sample(self):
        with self.assertRaises(ValueError):
            cluster_weight_draws(self.X, self.y, np.ones(len(self.y)), n_boot=1)


if __name__ == '__main__':
    unittest.main()
