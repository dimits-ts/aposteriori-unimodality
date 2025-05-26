import unittest
import numpy as np
from src.aposteriori import dfu


class TestDFU(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_unimodal_gaussian_returns_low_dfu(self):
        data = self.rng.normal(loc=0, scale=1, size=1000)
        score = dfu(data, bins=10, normalized=False)
        self.assertGreaterEqual(score, 0)
        self.assertLess(score, 1)
        norm_score = dfu(data, bins=10, normalized=True)
        self.assertGreaterEqual(norm_score, 0)
        self.assertLess(norm_score, 0.3)

    def test_uniform_distribution_returns_low_ndfu(self):
        data = self.rng.uniform(1, 5, size=1000)
        score = dfu(data, bins=5, normalized=True)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 0.3)

    def test_bimodal_distribution_high_ndfu(self):
        mode1 = self.rng.normal(loc=-2, scale=0.3, size=500)
        mode2 = self.rng.normal(loc=2, scale=0.3, size=500)
        data = np.hstack([mode1, mode2])
        score = dfu(data, bins=10, normalized=True)
        self.assertGreater(score, 0.5)

    def test_discrete_likert_unimodal(self):
        data = [3] * 100 + [2] * 10 + [4] * 10
        score = dfu(data, bins=5, normalized=True)
        self.assertGreaterEqual(score, 0)
        self.assertLess(score, 0.3)

    def test_discrete_likert_bimodal(self):
        data = [1] * 50 + [5] * 50
        score = dfu(data, bins=5, normalized=True)
        self.assertGreaterEqual(score, 0.9)
        self.assertLessEqual(score, 1.0)

    def test_empty_input_raises_value_error(self):
        with self.assertRaises(ValueError):
            dfu([], bins=5, normalized=True)

    def test_single_value_input_zero_dfu(self):
        data = [3.0] * 100
        score = dfu(data, bins=5, normalized=False)
        self.assertEqual(score, 0)

    def test_random_binning_does_not_crash(self):
        data = self.rng.normal(0, 1, size=100)
        for bins in [1, 3, 5, 10, 20]:
            score = dfu(data, bins=bins, normalized=True)
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)

    def test_normalization_changes_scale(self):
        data = self.rng.normal(0, 1.5, size=1000)
        raw = dfu(data, bins=10, normalized=False)
        norm = dfu(data, bins=10, normalized=True)
        self.assertAlmostEqual(raw, norm)
        self.assertGreaterEqual(norm, 0)
        self.assertLessEqual(norm, 1)

    def test_three_point_polarized(self):
        data = [1] * 50 + [5] * 50  # assuming bin centers at 1, 3, 5
        score = dfu(data, bins=3, normalized=True)
        self.assertAlmostEqual(score, 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
