import unittest
import numpy as np

from src.aposteriori import (
    _factor_dfu_stat,
    aposteriori_unimodality,
)


class TestAposterioriUnimodality(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(42)

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
        self.assertTrue(
            all(
                np.isnan(p.pvalue) or 0 <= p.pvalue <= 1
                for p in result.values()
            )
        )


class TestPolarizationStat(unittest.TestCase):

    def test_basic_functionality(self):
        annotations = np.array([1, 2, 3, 1, 2, 3])
        groups = np.array(["A", "A", "A", "B", "B", "B"])
        result = _factor_dfu_stat(annotations, groups, bins=3)
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), {"A", "B"})
        for val in result.values():
            self.assertIsInstance(val, float)

    def test_single_group(self):
        annotations = np.array([1, 1, 1])
        groups = np.array(["A", "A", "A"])
        result = _factor_dfu_stat(annotations, groups, bins=3)
        self.assertEqual(set(result.keys()), {"A"})

    def test_all_same_annotation(self):
        annotations = np.array([2, 2, 2, 2])
        groups = np.array(["X", "Y", "X", "Y"])
        result = _factor_dfu_stat(annotations, groups, bins=3)
        self.assertEqual(set(result.keys()), {"X", "Y"})
        for val in result.values():
            self.assertIsInstance(val, float)

    def test_empty_input(self):
        annotations = np.array([])
        groups = np.array([])
        with self.assertRaises(ValueError):
            _factor_dfu_stat(annotations, groups, bins=3)

    def test_mismatched_lengths(self):
        annotations = np.array([1, 2])
        groups = np.array(["A"])
        with self.assertRaises(ValueError):
            _factor_dfu_stat(annotations, groups, bins=2)

    def test_bins_parameter_effect(self):
        annotations = np.array([1, 2, 3, 4])
        groups = np.array(["A", "A", "B", "B"])
        # Make sure it runs and output stays valid for different bin values
        for bins in [2, 3, 4]:
            result = _factor_dfu_stat(annotations, groups, bins=bins)
            self.assertEqual(set(result.keys()), {"A", "B"})
            for val in result.values():
                self.assertIsInstance(val, float)


if __name__ == "__main__":
    unittest.main()
