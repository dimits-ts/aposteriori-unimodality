import unittest
import numpy as np
from ..aposteriori import (
    aposteriori_unimodality,
    _bootstrap_level_aposteriori_unit,
    _discussion_aposteriori,
    _aposteriori_unit,
)


class TestAposterioriUnimodality(unittest.TestCase):
    def test_aposteriori_unimodality_basic(self):
        """Test with simple grouped annotations and annotator groups."""
        annotations = [
            np.array([1, 2, 3, 4]),
            np.array([2, 3, 2, 3])
        ]
        annotator_group = [
            np.array(["A", "A", "B", "B"]),
            np.array(["A", "A", "B", "B"])
        ]
        result = aposteriori_unimodality(annotations, annotator_group)
        self.assertIsInstance(result, dict, "The result should be a dictionary.")
        self.assertGreaterEqual(result["A"], 0.0, "P-value should be non-negative.")
        self.assertGreaterEqual(result["B"], 0.0, "P-value should be non-negative.")

    def test_aposteriori_unimodality_mismatched_lengths(self):
        """Test that mismatched lengths raise an error."""
        annotations = [np.array([1, 2, 3])]
        annotator_group = [np.array(["A", "B"])]
        with self.assertRaises(ValueError):
            aposteriori_unimodality(annotations, annotator_group)

    def test_aposteriori_unimodality_empty_input(self):
        """Test that empty input returns an empty dictionary."""
        annotations = []
        annotator_group = []
        result = aposteriori_unimodality(annotations, annotator_group)
        self.assertEqual(result, {}, "Empty input should return an empty dictionary.")


class TestBootstrapLevelAposterioriUnit(unittest.TestCase):
    def test_bootstrap_level_aposteriori_unit_basic(self):
        """Test computation of aposteriori statistics via bootstrap."""
        annotations = np.array([1, 2, 3, 4])
        annotator_group = np.array(["A", "A", "B", "B"])
        level = "A"
        result = _bootstrap_level_aposteriori_unit(
            annotations, annotator_group, level, sample_ratio=0.5, bootstrap_steps=10
        )
        self.assertIsInstance(result, float, "Result should be a float.")
        self.assertGreaterEqual(result, 0.0, "Result should be non-negative.")


class TestDiscussionAposteriori(unittest.TestCase):
    def test_discussion_aposteriori_basic(self):
        """Test the Wilcoxon test with valid statistics."""
        stats = [0.1, 0.2, 0.3, 0.4]
        result = _discussion_aposteriori(stats)
        self.assertGreaterEqual(result, 0.0, "P-value should be non-negative.")
        self.assertLessEqual(result, 1.0, "P-value should be at most 1.")

    def test_discussion_aposteriori_no_difference(self):
        """Test when there is no difference between statistics and zero."""
        stats = [0.0, 0.0, 0.0]
        result = _discussion_aposteriori(stats)
        self.assertEqual(result, 1.0, "Should return 1.0 when there is no difference.")


class TestAposterioriUnit(unittest.TestCase):
    def test_aposteriori_unit_basic(self):
        """Test computation of aposteriori unit differences."""
        global_annotations = np.array([1, 2, 3, 4])
        level_annotations = np.array([1, 1, 2, 2])
        result = _aposteriori_unit(global_annotations, level_annotations)
        self.assertIsInstance(result, float, "Result should be a float.")


if __name__ == "__main__":
    unittest.main()
