import unittest
import numpy as np
from ..ndfu import ndfu, _to_hist


class TestNDFUFunctions(unittest.TestCase):

    def test_to_hist_normalized(self):
        scores = [1, 2, 2, 3, 3, 3, 4, 4, 5]
        num_bins = 5
        expected = np.array([1 / 9, 2 / 9, 3 / 9, 2 / 9, 1 / 9])  # Normalized counts
        result = _to_hist(scores, num_bins=num_bins, normed=True)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_to_hist_not_normalized(self):
        scores = [1, 2, 2, 3, 3, 3, 4, 4, 5]
        num_bins = 5
        expected = np.array([1, 2, 3, 2, 1])  # Raw counts
        result = _to_hist(scores, num_bins=num_bins, normed=False)
        np.testing.assert_array_equal(result, expected)

    def test_to_hist_empty_scores(self):
        with self.assertRaises(ValueError):
            _to_hist([], num_bins=5)

    def test_ndfu_uniform_distribution(self):
        # A uniform distribution should have low nDFU.
        scores = [1, 2, 3, 4, 5]
        result = ndfu(scores, num_bins=5)
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_ndfu_single_peak(self):
        # A distribution with a single peak
        scores = [1, 2, 2, 3, 3, 3, 4, 4, 5]
        result = ndfu(scores, num_bins=5)
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_ndfu_multimodal_distribution(self):
        # A clearly multimodal distribution
        scores = [1, 1, 5, 5]
        result = ndfu(scores, num_bins=5)
        self.assertAlmostEqual(result, 1.0)

    def test_ndfu_edge_case_low_bins(self):
        # Edge case: fewer bins than unique values in data
        scores = [1, 2, 3, 4, 5]
        result = ndfu(scores, num_bins=3)
        self.assertGreaterEqual(result, 0.0)

    def test_ndfu_edge_case_single_bin(self):
        # Edge case: single bin
        scores = [1, 2, 3, 4, 5]
        result = ndfu(scores, num_bins=1)
        self.assertEqual(result, 0.0)

    def test_ndfu_empty_scores(self):
        with self.assertRaises(ValueError):
            ndfu([], num_bins=5)


if __name__ == "__main__":
    unittest.main()
