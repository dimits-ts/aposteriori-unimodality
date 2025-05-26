import unittest
import numpy as np

from src.aposteriori import _ListDict


class TestListDict(unittest.TestCase):

    def setUp(self):
        self.ld = _ListDict()

    def test_set_and_get_single_value(self):
        self.ld['a'] = 1
        self.assertEqual(self.ld['a'], [1])

    def test_set_multiple_values_same_key(self):
        self.ld['a'] = 1
        self.ld['a'] = 2
        self.ld['a'] = 3
        self.assertEqual(self.ld['a'], [1, 2, 3])

    def test_set_multiple_keys(self):
        self.ld['a'] = 1
        self.ld['b'] = 2
        self.assertEqual(self.ld['a'], [1])
        self.assertEqual(self.ld['b'], [2])

    def test_keys(self):
        self.ld['a'] = 1
        self.ld['b'] = 2
        self.assertEqual(set(self.ld.keys()), {'a', 'b'})

    def test_values(self):
        self.ld['a'] = 1
        self.ld['a'] = 2
        self.ld['b'] = 3
        values = list(self.ld.values())
        self.assertIn([1, 2], values)
        self.assertIn([3], values)

    def test_items(self):
        self.ld['a'] = 1
        self.ld['a'] = 2
        self.ld['b'] = 3
        items = dict(self.ld.items())
        self.assertEqual(items['a'], [1, 2])
        self.assertEqual(items['b'], [3])

    def test_get_nonexistent_key_raises(self):
        with self.assertRaises(KeyError):
            _ = self.ld['missing']

    def test_update_with_factors_basic(self):
        stats = {'a': 10, 'b': 20}
        factors = ['a', 'b', 'c']
        self.ld.update_with_factors(stats, factors)
        self.assertEqual(self.ld['a'], [10])
        self.assertEqual(self.ld['b'], [20])
        self.assertTrue(np.isnan(self.ld['c'][0]))

    def test_update_with_factors_multiple_calls(self):
        factors = ['x', 'y']
        self.ld.update_with_factors({'x': 1}, factors)
        self.ld.update_with_factors({'y': 2}, factors)
        self.assertEqual(self.ld['x'], [1, np.nan])
        self.assertEqual(self.ld['y'], [np.nan, 2])

    def test_update_with_factors_all_nan(self):
        factors = ['k1', 'k2']
        self.ld.update_with_factors({}, factors)
        self.assertTrue(np.isnan(self.ld['k1'][0]))
        self.assertTrue(np.isnan(self.ld['k2'][0]))

    def test_all_lists_equal_length(self):
        factors = ['a', 'b', 'c']
        self.ld.update_with_factors({'a': 1}, factors)
        self.ld.update_with_factors({'b': 2}, factors)
        self.ld.update_with_factors({'c': 3}, factors)
        lengths = [len(self.ld[f]) for f in factors]
        self.assertEqual(len(set(lengths)), 1)  # All lengths equal


if __name__ == '__main__':
    unittest.main()
