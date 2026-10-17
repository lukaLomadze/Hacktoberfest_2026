"""Tests for secure aggregation (masking) and differential privacy."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np

from model.privacy import (
    add_dp_noise,
    clip_norm,
    epsilon,
    mask_update,
    prepare_update,
    unmask_sum,
)


class TestMasking(unittest.TestCase):
    def test_masks_cancel_exactly_in_sum(self):
        ids = [0, 1, 2]
        raw = [
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            np.array([-1.0, 0.5, 0.0, 0.1, -0.2, 0.3, 0.0]),
            np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        ]
        masked = [mask_update(raw[i], ids[i], ids, base_seed=7) for i in range(3)]
        s = unmask_sum(masked)
        np.testing.assert_allclose(s, np.sum(raw, axis=0), rtol=1e-9, atol=1e-9)

    def test_masked_update_looks_like_noise(self):
        raw = np.ones(7) * 0.1
        masked = mask_update(raw, 0, [0, 1, 2], base_seed=7)
        self.assertGreater(float(np.linalg.norm(masked - raw)), 1.0)

    def test_single_client_has_no_masks(self):
        raw = np.arange(7.0)
        masked = mask_update(raw, 0, [0], base_seed=7)
        np.testing.assert_allclose(masked, raw)


class TestClipAndNoise(unittest.TestCase):
    def test_clip_caps_norm(self):
        u = clip_norm(np.array([3.0, 4.0]), 2.0)
        self.assertLessEqual(float(np.linalg.norm(u)), 2.0 + 1e-9)

    def test_clip_leaves_small_unchanged(self):
        u = np.array([0.5, 0.5])
        np.testing.assert_allclose(clip_norm(u, 2.0), u)

    def test_noise_std_matches_sigma(self):
        rng = np.random.default_rng(0)
        samples = np.array([add_dp_noise(np.zeros(1), 0.5, rng)[0] for _ in range(20000)])
        self.assertAlmostEqual(float(samples.std()), 0.5, delta=0.02)

    def test_epsilon_monotonic(self):
        self.assertGreater(epsilon(0.1, 1.0), epsilon(0.5, 1.0))
        self.assertGreater(epsilon(0.5, 2.0), epsilon(0.5, 1.0))

    def test_epsilon_infinity_at_zero_noise(self):
        self.assertEqual(epsilon(0.0, 1.0), float("inf"))


class TestPipeline(unittest.TestCase):
    def test_prepare_update_shape_and_eps(self):
        masked, eps = prepare_update(np.ones(7) * 0.05, 0, [0, 1, 2], C=1.0, sigma=0.1, base_seed=11)
        self.assertEqual(masked.shape, (7,))
        self.assertGreater(eps, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
