import unittest

from src.aposteriori import _ListDict


class TestListDict(unittest.TestCase):

    def setUp(self):
        self.ld = _ListDict()

    def test_set_and_get_single_value(self):
        self.ld["a"] = 1
        self.assertEqual(self.ld["a"], [1])

    def test_set_multiple_values_same_key(self):
        self.ld["a"] = 1
        self.ld["a"] = 2
        self.ld["a"] = 3
        self.assertEqual(self.ld["a"], [1, 2, 3])

    def test_set_multiple_keys(self):
        self.ld["a"] = 1
        self.ld["b"] = 2
        self.assertEqual(self.ld["a"], [1])
        self.assertEqual(self.ld["b"], [2])

    def test_keys(self):
        self.ld["a"] = 1
        self.ld["b"] = 2
        self.assertEqual(set(self.ld.keys()), {"a", "b"})

    def test_values(self):
        self.ld["a"] = 1
        self.ld["a"] = 2
        self.ld["b"] = 3
        values = list(self.ld.values())
        self.assertIn([1, 2], values)
        self.assertIn([3], values)

    def test_items(self):
        self.ld["a"] = 1
        self.ld["b"] = 2
        items = dict(self.ld.items())
        self.assertEqual(items["a"], [1])
        self.assertEqual(items["b"], [2])

    def test_get_nonexistent_key_raises(self):
        with self.assertRaises(KeyError):
            _ = self.ld["missing"]

    def test_add_dict_single_call(self):
        self.ld.add_dict({"a": 1, "b": 2})
        self.assertEqual(self.ld["a"], [1])
        self.assertEqual(self.ld["b"], [2])

    def test_add_dict_multiple_calls(self):
        self.ld.add_dict({"a": 1})
        self.ld.add_dict({"a": 2, "b": 3})
        self.assertEqual(self.ld["a"], [1, 2])
        self.assertEqual(self.ld["b"], [3])

    def test_len(self):
        self.ld["a"] = 1
        self.ld["b"] = 2
        self.assertEqual(len(self.ld), 2)


if __name__ == "__main__":
    unittest.main()
