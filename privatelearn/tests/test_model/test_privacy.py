"""Tests for secure aggregation (masking) and differential privacy."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np

from model.privacy import (
    add_dp_noise,
    clip_norm,
    epsilon,
    make_pair_secrets,
    mask_update,
    noise_rng,
    prepare_update,
    record_sensitivity,
    sigma_for_epsilon,
    train_dp,
    unmask_sum,
)

IDS = [0, 1, 2]
RAW = [
    np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
    np.array([-1.0, 0.5, 0.0, 0.1, -0.2, 0.3, 0.0]),
    np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
]


class TestMasking(unittest.TestCase):
    def setUp(self):
        self.secrets = make_pair_secrets(IDS, seed="hospital-handshake")

    def masked(self, round_=1, participants=IDS, raw=RAW):
        return [mask_update(raw[i], i, participants, round_, self.secrets) for i in participants]

    def test_masks_cancel_exactly_in_sum(self):
        np.testing.assert_allclose(unmask_sum(self.masked()), np.sum(RAW, axis=0), rtol=0, atol=1e-9)

    def test_masked_update_looks_like_noise(self):
        for i, m in enumerate(self.masked()):
            self.assertGreater(float(np.linalg.norm(m - RAW[i])), 1.0)

    def test_single_participant_has_no_masks(self):
        np.testing.assert_allclose(mask_update(RAW[0], 0, [0], 1, self.secrets), RAW[0])

    def test_masks_change_every_round(self):
        # same update in two rounds: the difference must not reveal 0 (i.e. masks differ)
        m1 = mask_update(RAW[0], 0, IDS, 1, self.secrets)
        m2 = mask_update(RAW[0], 0, IDS, 2, self.secrets)
        self.assertGreater(float(np.linalg.norm(m2 - m1)), 1.0)
        # and differencing two rounds of different updates does not leak their difference
        m3 = mask_update(RAW[0] * 3, 0, IDS, 2, self.secrets)
        self.assertFalse(np.allclose(m3 - m1, RAW[0] * 2))

    def test_coordinator_cannot_strip_masks_without_the_secrets(self):
        guessed = make_pair_secrets(IDS, seed=42)  # public default-style guess
        stripped = self.masked()[0] - mask_update(np.zeros(7), 0, IDS, 1, guessed)
        self.assertFalse(np.allclose(stripped, RAW[0], atol=1e-3))

    def test_dropout_handled_by_fixing_participants_before_masking(self):
        # h2 offline: participants announced as [0, 1] BEFORE masking -> exact sum
        two = self.masked(participants=[0, 1])
        np.testing.assert_allclose(unmask_sum(two), RAW[0] + RAW[1], rtol=0, atol=1e-9)
        # whereas dropping h2 AFTER masking for all three breaks cancellation
        broken = unmask_sum(self.masked()[:2])
        self.assertGreater(float(np.linalg.norm(broken - RAW[0] - RAW[1])), 0.1)

    def test_hospital_needs_only_its_own_pair_secrets(self):
        own = {k: v for k, v in self.secrets.items() if 0 in k}
        np.testing.assert_allclose(mask_update(RAW[0], 0, IDS, 1, own), self.masked()[0])
        with self.assertRaises(ValueError):
            mask_update(RAW[2], 2, IDS, 1, own)  # lacks the (1, 2) secret

    def test_client_must_be_a_participant(self):
        with self.assertRaises(ValueError):
            mask_update(RAW[2], 2, [0, 1], 1, self.secrets)


class TestClipAndNoise(unittest.TestCase):
    def test_clip_caps_norm(self):
        self.assertLessEqual(float(np.linalg.norm(clip_norm(np.array([3.0, 4.0]), 2.0))), 2.0 + 1e-9)

    def test_clip_leaves_small_unchanged(self):
        u = np.array([0.5, 0.5])
        np.testing.assert_allclose(clip_norm(u, 2.0), u)

    def test_noise_std_matches_sigma(self):
        rng = np.random.default_rng(0)
        samples = add_dp_noise(np.zeros(20000), 0.5, rng)
        self.assertAlmostEqual(float(samples.std()), 0.5, delta=0.02)

    def test_rng_is_required(self):
        with self.assertRaises(TypeError):
            add_dp_noise(np.zeros(7), 0.5, None)

    def test_noise_differs_per_hospital_and_round(self):
        z = np.zeros(7)
        a = add_dp_noise(z, 1.0, noise_rng("s", 0, 1))
        self.assertFalse(np.allclose(a, add_dp_noise(z, 1.0, noise_rng("s", 1, 1))))
        self.assertFalse(np.allclose(a, add_dp_noise(z, 1.0, noise_rng("s", 0, 2))))
        np.testing.assert_allclose(a, add_dp_noise(z, 1.0, noise_rng("s", 0, 1)))  # reproducible


class TestEpsilon(unittest.TestCase):
    def test_monotonic(self):
        self.assertGreater(epsilon(0.5, 1.0), epsilon(1.0, 1.0))
        self.assertGreater(epsilon(1.0, 2.0), epsilon(1.0, 1.0))

    def test_infinite_at_zero_noise(self):
        self.assertEqual(epsilon(0.0, 1.0), float("inf"))

    def test_tighter_than_classic_bound_where_classic_is_valid(self):
        sigma = 6.0  # classic ε ≈ 0.81 < 1, so the classic bound applies
        classic = math.sqrt(2 * math.log(1.25 / 1e-5)) / sigma
        self.assertLess(epsilon(sigma, 1.0), classic)
        self.assertGreater(epsilon(sigma, 1.0), 0.5 * classic)

    def test_epsilon_achieves_requested_delta(self):
        from model.privacy import _delta_of
        for sigma in (0.5, 1.0, 3.0):
            eps = epsilon(sigma, 1.0)
            self.assertLessEqual(_delta_of(eps, sigma, 1.0), 1e-5 * (1 + 1e-6))
            self.assertGreater(_delta_of(eps * 0.99, sigma, 1.0), 1e-5)

    def test_sigma_for_epsilon_round_trips(self):
        for eps in (0.5, 1.5, 4.0):
            self.assertAlmostEqual(epsilon(sigma_for_epsilon(eps, 1.0), 1.0), eps, places=6)


class TestRecordLevelDp(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.X = rng.normal(size=(120, 6)) * 2.0
        self.y = (self.X[:, 0] > 0).astype(float)

    def test_record_sensitivity_formula(self):
        self.assertAlmostEqual(record_sensitivity(120, lr=0.1, epochs=4, example_clip=1.0), 0.8 / 120)
        with self.assertRaises(ValueError):
            record_sensitivity(0)

    def test_changing_one_row_moves_delta_within_sensitivity(self):
        w0 = np.zeros(7)
        base = train_dp(w0, self.X, self.y)
        sens = record_sensitivity(len(self.y))
        rng = np.random.default_rng(9)
        for _ in range(20):
            X2, y2 = self.X.copy(), self.y.copy()
            k = int(rng.integers(len(y2)))
            X2[k] = rng.normal(size=6) * 10.0
            y2[k] = 1.0 - y2[k]
            self.assertLessEqual(float(np.linalg.norm(train_dp(w0, X2, y2) - base)), sens + 1e-12)

    def test_large_clip_equals_plain_training(self):
        from model.logistic import train
        np.testing.assert_allclose(train_dp(np.zeros(7), self.X, self.y, example_clip=1e9),
                                   train(np.zeros(7), self.X, self.y), atol=1e-12)


class TestPipeline(unittest.TestCase):
    def test_prepare_update_sums_to_noised_total(self):
        secrets = make_pair_secrets(IDS, seed="x")
        sigma = sigma_for_epsilon(1.5, 1.0)
        outs = [prepare_update(RAW[i] * 0.01, i, IDS, 3, secrets, noise_seed="n", C=1.0, sigma=sigma)
                for i in IDS]
        total = unmask_sum([m for m, _ in outs])
        expected = sum(clip_norm(RAW[i] * 0.01, 1.0) + noise_rng("n", i, 3).normal(0.0, sigma, 7) for i in IDS)
        np.testing.assert_allclose(total, expected, rtol=0, atol=1e-9)
        for _, eps in outs:
            self.assertAlmostEqual(eps, 1.5, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
