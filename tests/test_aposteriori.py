import unittest
import numpy as np

from src.aposteriori import (
    _polarization_stat,
    _raw_significance,
    _correct_significance,
    aposteriori_unimodality,
)


class TestAposterioriUnimodality(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_output_is_dict(self):
        annotations = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        factor_group = ["A"] * 5 + ["B"] * 5
        comment_group = ["c1"] * 5 + ["c2"] * 5
        result = aposteriori_unimodality(
            annotations, factor_group, comment_group, bins=5
        )
        self.assertIsInstance(result, dict)
        for key, value in result.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, float)

    def test_output_keys_match_factor_values(self):
        annotations = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        factor_group = ["A"] * 5 + ["B"] * 5
        comment_group = ["c1"] * 5 + ["c2"] * 5
        result = aposteriori_unimodality(
            annotations, factor_group, comment_group, bins=5
        )
        self.assertEqual(set(result.keys()), {"A", "B"})

    def test_empty_inputs_raise_value_error(self):
        with self.assertRaises(ValueError):
            aposteriori_unimodality([], [], [], bins=5)

    def test_mismatched_lengths_raise_value_error(self):
        annotations = [1, 2, 3]
        factor_group = ["A", "B"]
        comment_group = ["c1", "c1", "c1"]
        with self.assertRaises(ValueError):
            aposteriori_unimodality(
                annotations, factor_group, comment_group, bins=5
            )

    def test_single_group_raise_value_error(self):
        annotations = [1, 2, 3, 4, 5]
        factor_group = ["solo"] * 5
        comment_group = ["c1"] * 5
        with self.assertRaises(ValueError):
            aposteriori_unimodality(
                annotations, factor_group, comment_group, bins=5
            )

    def test_partitioned_bimodal_data_low_pvalue(self):
        annotations = [1] * 50 + [5] * 50
        factor_group = ["left"] * 50 + ["right"] * 50
        comment_group = ["c1"] * 50 + ["c2"] * 50
        result = aposteriori_unimodality(
            annotations, factor_group, comment_group, bins=5
        )
        for group, p in result.items():
            self.assertLess(p, 0.05)

    def test_random_noise_returns_high_pvalues(self):
        annotations = self.rng.normal(loc=3, scale=1, size=100).tolist()
        factor_group = ["X"] * 50 + ["Y"] * 50
        comment_group = ["c1"] * 50 + ["c2"] * 50
        result = aposteriori_unimodality(
            annotations, factor_group, comment_group, bins=5
        )
        for group, p in result.items():
            self.assertGreater(p, 0.05)

    def test_multiple_comments_are_aggregated(self):
        annotations = [1, 5, 1, 5, 1, 5, 2, 4, 2, 4]
        factor_group = ["A", "B"] * 5
        comment_group = [
            "c1",
            "c1",
            "c2",
            "c2",
            "c3",
            "c3",
            "c4",
            "c4",
            "c5",
            "c5",
        ]
        result = aposteriori_unimodality(
            annotations, factor_group, comment_group, bins=5
        )
        self.assertEqual(set(result.keys()), {"A", "B"})
        self.assertTrue(all(0 <= p <= 1 for p in result.values()))


class TestPolarizationStat(unittest.TestCase):

    def test_basic_functionality(self):
        annotations = np.array([1, 2, 3, 1, 2, 3])
        groups = np.array(["A", "A", "A", "B", "B", "B"])
        result = _polarization_stat(annotations, groups, bins=3)
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), {"A", "B"})
        for val in result.values():
            self.assertIsInstance(val, float)

    def test_single_group(self):
        annotations = np.array([1, 1, 1])
        groups = np.array(["A", "A", "A"])
        result = _polarization_stat(annotations, groups, bins=3)
        self.assertEqual(set(result.keys()), {"A"})

    def test_all_same_annotation(self):
        annotations = np.array([2, 2, 2, 2])
        groups = np.array(["X", "Y", "X", "Y"])
        result = _polarization_stat(annotations, groups, bins=3)
        self.assertEqual(set(result.keys()), {"X", "Y"})
        for val in result.values():
            self.assertIsInstance(val, float)

    def test_empty_input(self):
        annotations = np.array([])
        groups = np.array([])
        with self.assertRaises(ValueError):
            _polarization_stat(annotations, groups, bins=3)

    def test_mismatched_lengths(self):
        annotations = np.array([1, 2])
        groups = np.array(["A"])
        with self.assertRaises(ValueError):
            _polarization_stat(annotations, groups, bins=2)

    def test_bins_parameter_effect(self):
        annotations = np.array([1, 2, 3, 4])
        groups = np.array(["A", "A", "B", "B"])
        # Make sure it runs and output stays valid for different bin values
        for bins in [2, 3, 4]:
            result = _polarization_stat(annotations, groups, bins=bins)
            self.assertEqual(set(result.keys()), {"A", "B"})
            for val in result.values():
                self.assertIsInstance(val, float)


class TestRawSignificance(unittest.TestCase):

    def test_output_type_and_keys(self):
        global_ndfus = [0.2, 0.3]
        stats_by_factor = {"A": [0.3, 0.35], "B": [0.1, 0.15]}
        result = _raw_significance(global_ndfus, stats_by_factor)
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), {"A", "B"})
        for val in result.values():
            self.assertIsInstance(val, float)

    def test_empty_inputs(self):
        result = _correct_significance({})
        self.assertEqual(result, {})

    def test_only_one_factor(self):
        global_ndfus = [0.1, 0.2]
        stats_by_factor = {"A": [0.25, 0.3]}
        result = _raw_significance(global_ndfus, stats_by_factor)
        self.assertEqual(set(result.keys()), {"A"})
        self.assertIsInstance(result["A"], float)

    def test_mismatched_distribution_sizes(self):
        global_ndfus = [0.1, 0.2, 0.35, 0.6]
        stats_by_factor = {"A": [0.1, 0.2, 0.3], "B": [0.5]}
        with self.assertRaises(ValueError):
            _raw_significance(global_ndfus, stats_by_factor)

    def test_constant_global_distribution(self):
        global_ndfus = [0.3, 0.3]
        stats_by_factor = {"A": [0.3, 0.4], "B": [0.2, 0.1]}
        # Should still work even if global has no variance
        result = _raw_significance(global_ndfus, stats_by_factor)
        self.assertTrue(all(isinstance(val, float) for val in result.values()))

    def test_nan_or_invalid_values(self):
        global_ndfus = [0.3, float("nan"), 0.2]
        stats_by_factor = {"A": [0.3, 0.4]}
        with self.assertRaises(ValueError):
            _raw_significance(global_ndfus, stats_by_factor)


class TestCorrectSignificance(unittest.TestCase):

    def test_output_format_and_keys(self):
        raw_pvals = {"A": 0.01, "B": 0.04, "C": 0.2}
        result = _correct_significance(raw_pvals)
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), {"A", "B", "C"})
        for val in result.values():
            self.assertIsInstance(val, float)

    def test_correction_is_applied(self):
        raw_pvals = {"A": 0.01, "B": 0.02, "C": 0.03}
        result = _correct_significance(raw_pvals, alpha=0.05)
        for key in result:
            self.assertGreaterEqual(result[key], raw_pvals[key])

    def test_alpha_parameter_does_not_affect_output_values(self):
        # Alpha often affects decision thresholds, not p-value correction itself
        raw_pvals = {"A": 0.01, "B": 0.04}
        result1 = _correct_significance(raw_pvals, alpha=0.05)
        result2 = _correct_significance(raw_pvals, alpha=0.01)
        self.assertEqual(result1, result2)

    def test_edge_case_all_zeros(self):
        raw_pvals = {"X": 0.0, "Y": 0.0}
        result = _correct_significance(raw_pvals)
        for val in result.values():
            self.assertEqual(val, 0.0)

    def test_edge_case_all_ones(self):
        raw_pvals = {"X": 1.0, "Y": 1.0}
        result = _correct_significance(raw_pvals)
        for val in result.values():
            self.assertEqual(val, 1.0)

    def test_invalid_pvalue_range(self):
        raw_pvals = {"X": -0.1, "Y": 1.2}
        with self.assertRaises(ValueError):
            _correct_significance(raw_pvals)

    def test_empty_input(self):
        result = _correct_significance({})
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
