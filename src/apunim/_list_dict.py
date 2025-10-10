from typing import TypeVar, Generic
import json


K = TypeVar("K")  # Key type
V = TypeVar("V")  # Value type


class _ListDict(Generic[K, V]):
    """
    A dictionary appending multiple values with the same key
    to a list instead of overwriting them.
    """

    # TODO: Properly implement key error
    def __init__(self):
        """
        Create a new ListDict which will hold arrays of equal length
        for the values of each key.

        :param all_factors: List of all possible factors
            (ensures all keys updated)
        :type all_factors: list
        """
        self.dict = {}

    def keys(self) -> list[K, list[V]]:
        return self.dict.keys()

    def values(self) -> list[K, list[V]]:
        return self.dict.values()

    def items(self) -> list[tuple[K, list[V]]]:
        return self.dict.items()
    
    def add_dict(self, new_stats: dict[K, V]):
        """
        Update the _ListDict with at most one extra value per factor, keeping
        all internal lists at the same length.
        :param new_stats: Dictionary of {factor: stat}
        """
        for factor in new_stats:
            self[factor] = new_stats[factor]

    def __getitem__(self, key) -> list[V]:
        return self.dict[key]

    def __setitem__(self, key: K, value: V):
        if key in self.dict:
            self.dict[key].append(value)
        else:
            self.dict[key] = [value]

    def __contains__(self, key: K) -> bool:
        return key in self.dict

    def __len__(self):
        return len(self.dict)

    def __str__(self):
        return json.dumps(self.dict)
