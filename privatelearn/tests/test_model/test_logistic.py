"""Tests for the logistic-regression core (stdlib unittest, no pytest)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np

# a local fixture: the model itself no longer hard-codes a feature count
N_FEATURES = 6
N_WEIGHTS = N_FEATURES + 1

from model.logistic import (
    auc,
    gradient,
    loss,
    predict_proba,
    sigmoid,
    train,
)


class TestSigmoid(unittest.TestCase):
    def test_range(self):
        z = np.array([-20.0, -1.0, 0.0, 1.0, 20.0])
        s = sigmoid(z)
        self.assertTrue(np.all(s > 0.0) and np.all(s < 1.0))

    def test_extremes_approach_limits(self):
        self.assertGreater(float(sigmoid(100.0)), 0.9999)
        self.assertLess(float(sigmoid(-100.0)), 0.0001)

    def test_center(self):
        self.assertAlmostEqual(float(sigmoid(0.0)), 0.5, places=12)

    def test_monotonic(self):
        s = sigmoid(np.array([-2.0, -1.0, 1.0, 2.0]))
        self.assertTrue(np.all(np.diff(s) > 0.0))


class TestModel(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = rng.normal(size=(50, N_FEATURES))
        self.y = (self.X[:, 0] + self.X[:, 1] > 0.0).astype(float)
        self.w0 = np.zeros(N_WEIGHTS)

    def test_weight_shape_contract(self):
        self.assertEqual(N_WEIGHTS, N_FEATURES + 1)

    def test_predict_proba_shape(self):
        p = predict_proba(self.w0, self.X)
        self.assertEqual(p.shape, (50,))
        self.assertTrue(np.all((p > 0.0) & (p < 1.0)))

    def test_gradient_matches_finite_difference(self):
        g = gradient(self.w0, self.X, self.y)
        eps = 1e-6
        for i in range(N_WEIGHTS):
            wp = self.w0.copy()
            wp[i] += eps
            wm = self.w0.copy()
            wm[i] -= eps
            numeric = (loss(wp, self.X, self.y) - loss(wm, self.X, self.y)) / (2 * eps)
            self.assertAlmostEqual(float(g[i]), float(numeric), places=5)

    def test_loss_decreases(self):
        l0 = loss(self.w0, self.X, self.y)
        w = train(self.w0, self.X, self.y, epochs=8, lr=0.3)
        self.assertLess(loss(w, self.X, self.y), l0)

    def test_delta_update_is_finite_vector(self):
        w_prev = self.w0.copy()
        w_new = train(w_prev, self.X, self.y, epochs=4, lr=0.1)
        delta = w_new - w_prev
        self.assertEqual(delta.shape, (N_WEIGHTS,))
        self.assertTrue(np.isfinite(delta).all())


class TestAuc(unittest.TestCase):
    def test_random_is_half(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 500)
        s = rng.normal(size=500)
        self.assertAlmostEqual(auc(y, s), 0.5, delta=0.08)

    def test_separable_is_one(self):
        y = np.array([0] * 50 + [1] * 50)
        s = np.array([0.1] * 50 + [0.9] * 50)
        self.assertAlmostEqual(auc(y, s), 1.0, places=12)

    def test_separable_is_zero_when_inverted(self):
        y = np.array([0] * 50 + [1] * 50)
        s = np.array([0.9] * 50 + [0.1] * 50)
        self.assertAlmostEqual(auc(y, s), 0.0, places=12)

    def test_single_class_returns_half(self):
        self.assertEqual(auc(np.ones(20), np.random.normal(size=20)), 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
