"""Tests for the gradient-inversion attack demo."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np

from model.attack import reconstruct_single_sample, relative_error, run_attack


class TestReconstruct(unittest.TestCase):
    def test_exact_recovery(self):
        x = np.array([0.2, -0.5, 0.1, 0.3, 0.0, 0.4])
        # g = (p - y) * [1; x]; choose a scale c != 0
        c = -0.7
        g = c * np.concatenate([[1.0], x])
        np.testing.assert_allclose(reconstruct_single_sample(g), x, rtol=1e-12)

    def test_zero_bias_gradient_returns_none(self):
        self.assertIsNone(reconstruct_single_sample(np.zeros(7)))

    def test_relative_error(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertAlmostEqual(relative_error(x, x), 0.0, places=12)
        self.assertEqual(relative_error(x, None), float("inf"))


class TestAttack(unittest.TestCase):
    def _case(self):
        w = np.zeros(7)  # p = 0.5, so delta[0] = 0.5 - y != 0
        x = np.array([0.3, -0.2, 0.5, 0.1, -0.4, 0.6])
        y = 0.0
        return w, x, y

    def test_plain_attack_recovers(self):
        w, x, y = self._case()
        r = run_attack(w, x, y)
        self.assertTrue(r["without_privacy"]["recovered"])
        self.assertLess(r["without_privacy"]["feature_error"], 0.01)

    def test_protected_attack_fails(self):
        w, x, y = self._case()
        r = run_attack(w, x, y)
        self.assertFalse(r["with_privacy"]["recovered"])
        # masked+DP reconstruction is orders of magnitude worse than the raw one
        self.assertGreater(r["with_privacy"]["feature_error"], 1.0)
        self.assertGreater(
            r["with_privacy"]["feature_error"],
            r["without_privacy"]["feature_error"] * 50,
        )

    def test_shape_matches_trace_contract(self):
        w, x, y = self._case()
        r = run_attack(w, x, y)
        self.assertEqual(set(r.keys()), {"target", "without_privacy", "with_privacy"})
        for k in ("without_privacy", "with_privacy"):
            self.assertEqual(set(r[k].keys()), {"recovered", "feature_error"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
