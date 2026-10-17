"""Tests for aggregation strategies and the adaptive selector."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np

from model.aggregation import (
    aggregate,
    fedavg,
    fedprox,
    select_strategy,
    trimmed_mean,
)


class TestFedAvg(unittest.TestCase):
    def test_equal_weights_is_plain_mean(self):
        d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        out = fedavg(d, [100, 100, 100])
        np.testing.assert_allclose(out, [3.0, 4.0])

    def test_weighted_by_n(self):
        d = np.array([[1.0, 0.0], [0.0, 1.0]])
        # client 0 has 3x the data
        out = fedavg(d, [300, 100])
        np.testing.assert_allclose(out, [0.75, 0.25])

    def test_single_client_equals_its_delta(self):
        d = np.array([[0.4, -0.2, 0.1, 0.0, 0.0, 0.0, 0.0]])
        out = fedavg(d, [120])
        np.testing.assert_allclose(out, d[0])
        # 1-client FedAvg must equal local SGD's update (the plan's acceptance)
        np.testing.assert_allclose(aggregate(d, [120], "fedavg"), d[0])


class TestTrimmedMean(unittest.TestCase):
    def test_drops_largest_norm(self):
        d = np.array([[1.0, 0.0], [0.0, 1.0], [100.0, 100.0]])  # third is garbage
        out = trimmed_mean(d, [120, 120, 120], k=1)
        np.testing.assert_allclose(out, [0.5, 0.5])

    def test_rejects_too_few_clients(self):
        with self.assertRaises(ValueError):
            trimmed_mean(np.array([[1.0]]), [120], k=1)


class TestFedProx(unittest.TestCase):
    def test_caps_norm(self):
        d = np.array([[0.0, 3.0], [0.0, 4.0]])  # norm 3 and 4, cap at 2
        out = fedprox(d, [120, 120], mu=2.0)
        self.assertLessEqual(float(np.linalg.norm(out)), 2.0)

    def test_small_updates_unchanged(self):
        d = np.array([[0.5, 0.0], [0.0, 0.5]])
        out = fedprox(d, [120, 120], mu=10.0)
        np.testing.assert_allclose(out, [0.25, 0.25])


class TestSelector(unittest.TestCase):
    def _d(self):
        return np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

    def test_outlier_loss_selects_trimmed(self):
        s, _ = select_strategy([0.4, 0.41, 5.0], self._d())
        self.assertEqual(s, "trimmed_mean")

    def test_tight_losses_select_fedavg(self):
        agreeing = np.array([[1.0, 0.1], [0.9, 0.0], [1.0, -0.1]])  # all ~norm 1
        s, reason = select_strategy([0.40, 0.41, 0.39], agreeing)
        self.assertEqual(s, "fedavg")
        self.assertIn("FedAvg", reason)

    def test_spread_losses_select_fedprox(self):
        s, _ = select_strategy([0.30, 0.45, 0.60], self._d(), tau=0.05)
        self.assertEqual(s, "fedprox")

    def test_reason_is_a_string(self):
        _, reason = select_strategy([0.4, 0.41, 0.39], self._d())
        self.assertIsInstance(reason, str)
        self.assertTrue(len(reason) > 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
