"""Tests for aggregation strategies and the adaptive selector."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np

from model.aggregation import (
    FEDPROX_MU,
    aggregate,
    fedavg,
    fedprox,
    select_strategy,
    trimmed_mean,
)
from model.logistic import auc, train

# a local fixture: the model itself no longer hard-codes a feature count
N_FEATURES = 6
N_WEIGHTS = N_FEATURES + 1


class TestFedAvg(unittest.TestCase):
    def test_equal_weights_is_plain_mean(self):
        d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_allclose(fedavg(d, [100, 100, 100]), [3.0, 4.0])

    def test_weighted_by_n(self):
        d = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_allclose(fedavg(d, [300, 100]), [0.75, 0.25])

    def test_single_client_equals_local_sgd(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, N_FEATURES))
        y = (X[:, 0] > 0).astype(float)
        w0 = np.zeros(N_WEIGHTS)
        delta = train(w0, X, y) - w0
        np.testing.assert_allclose(w0 + aggregate([delta], [40], "fedavg"), train(w0, X, y))

    def test_rejects_bad_shapes(self):
        with self.assertRaises(ValueError):
            fedavg(np.ones(3), [1, 1, 1])
        with self.assertRaises(ValueError):
            fedavg(np.ones((2, 3)), [1])


class TestTrimmedMean(unittest.TestCase):
    def test_drops_flagged_client(self):
        d = np.array([[1.0, 0.0], [0.0, 1.0], [100.0, 100.0]])
        np.testing.assert_allclose(trimmed_mean(d, [120, 120, 120], drop=[2]), [0.5, 0.5])

    def test_drop_wins_over_norm(self):
        # client 0 is flagged even though client 2 has the largest norm
        d = np.array([[1.0, 0.0], [0.0, 1.0], [5.0, 5.0]])
        np.testing.assert_allclose(trimmed_mean(d, [1, 1, 1], drop=[0]), [2.5, 3.0])

    def test_norm_fallback_for_unmasked_updates(self):
        d = np.array([[1.0, 0.0], [0.0, 1.0], [100.0, 100.0]])
        np.testing.assert_allclose(trimmed_mean(d, [120, 120, 120]), [0.5, 0.5])

    def test_rejects_dropping_everyone_or_bad_index(self):
        with self.assertRaises(ValueError):
            trimmed_mean(np.ones((2, 2)), [1, 1], drop=[0, 1])
        with self.assertRaises(ValueError):
            trimmed_mean(np.ones((2, 2)), [1, 1], drop=[5])
        with self.assertRaises(ValueError):
            trimmed_mean(np.array([[1.0]]), [120])


class TestFedProx(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.X = rng.normal(size=(60, N_FEATURES))
        self.y = (self.X[:, 1] - self.X[:, 2] > 0).astype(float)
        self.w_global = np.full(N_WEIGHTS, 0.2)

    def test_server_side_is_weighted_average(self):
        d = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_allclose(fedprox(d, [300, 100]), fedavg(d, [300, 100]))

    def test_proximal_term_keeps_local_model_near_global(self):
        free = train(self.w_global, self.X, self.y, epochs=20, lr=0.3)
        prox = train(self.w_global, self.X, self.y, epochs=20, lr=0.3, mu=1.0, w_anchor=self.w_global)
        self.assertLess(np.linalg.norm(prox - self.w_global), np.linalg.norm(free - self.w_global))

    def test_mu_zero_is_plain_training(self):
        np.testing.assert_allclose(
            train(self.w_global, self.X, self.y, mu=0.0, w_anchor=self.w_global),
            train(self.w_global, self.X, self.y))

    def test_mu_requires_anchor(self):
        with self.assertRaises(ValueError):
            train(self.w_global, self.X, self.y, mu=FEDPROX_MU)


class TestSelector(unittest.TestCase):
    def test_outlier_loss_selects_trimmed_and_flags_that_client(self):
        s, reason, flagged = select_strategy([0.4, 5.0, 0.41], names=["Hospital A", "Hospital B", "Hospital C"])
        self.assertEqual(s, "trimmed_mean")
        self.assertEqual(flagged, (1,))
        self.assertIn("Hospital B", reason)

    def test_outlier_detected_even_when_other_losses_tie(self):
        s, _, flagged = select_strategy([0.4, 0.4, 5.0])
        self.assertEqual((s, flagged), ("trimmed_mean", (2,)))

    def test_honest_non_iid_gaps_are_not_outliers(self):
        # measured probe losses on the real split (round 4) and the all-equal round 1
        for losses in ([0.5333, 0.6292, 0.6224], [0.6931, 0.6931, 0.6932], [0.60, 0.62, 0.70]):
            self.assertEqual(select_strategy(losses).flagged, (), losses)

    def test_tight_losses_select_fedavg(self):
        s, reason, flagged = select_strategy([0.40, 0.41, 0.39])
        self.assertEqual((s, flagged), ("fedavg", ()))
        self.assertIn("FedAvg", reason)

    def test_spread_losses_select_fedprox(self):
        s, _, flagged = select_strategy([0.30, 0.45, 0.60], tau=0.05)
        self.assertEqual((s, flagged), ("fedprox", ()))

    def test_works_without_deltas_for_masked_rounds(self):
        self.assertEqual(select_strategy([0.40, 0.41, 0.39], deltas=None).strategy, "fedavg")

    def test_norm_spread_only_with_unmasked_deltas(self):
        d = np.array([[0.01, 0.0], [0.0, 0.01], [3.0, 3.0]])
        self.assertEqual(select_strategy([0.40, 0.41, 0.39], deltas=d).strategy, "fedprox")

    def test_aggregate_uses_flagged(self):
        d = np.array([[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]])
        dec = select_strategy([0.4, 0.41, 3.0])
        np.testing.assert_allclose(aggregate(d, [1, 1, 1], dec.strategy, drop=dec.flagged), [0.5, 0.5])

    def test_unknown_strategy(self):
        with self.assertRaises(ValueError):
            aggregate(np.ones((2, 2)), [1, 1], "magic")


class TestAucTies(unittest.TestCase):
    def test_all_tied_scores_give_half(self):
        self.assertAlmostEqual(auc(np.array([0, 1, 0, 1]), np.zeros(4)), 0.5)

    def test_partial_ties_match_pairwise_definition(self):
        y = np.array([0, 0, 1, 1, 1, 0])
        s = np.array([0.1, 0.5, 0.5, 0.9, 0.2, 0.2])
        pos, neg = s[y == 1], s[y == 0]
        pairwise = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
        self.assertAlmostEqual(auc(y, s), float(pairwise), places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
