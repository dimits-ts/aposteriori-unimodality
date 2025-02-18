from typing import Iterable
import numpy as np


# code adapted from John Pavlopoulos https://github.com/ipavlopoulos/ndfu/blob/main/src/__init__.py
def ndfu(input_data: Iterable[float], num_bins: int = 5) -> float:
    """The normalized Distance From Unimodality measure
    :param: input_data: a list of annotations, not necessarily discrete
    :param: histogram_input: False to compute rel. frequencies (ratings as input)
    :raises ValueError: if input_data is empty
    :return: the nDFU score
    """
    # compute DFU
    hist = _to_hist(input_data, num_bins=num_bins)
    max_value = max(hist)
    pos_max = np.where(hist == max_value)[0][0]
    # right search
    max_diff = 0
    for i in range(pos_max, len(hist) - 1):
        diff = hist[i + 1] - hist[i]
        if diff > max_diff:
            max_diff = diff
    for i in range(pos_max, 0, -1):
        diff = hist[i - 1] - hist[i]
        if diff > max_diff:
            max_diff = diff

    # return normalized dfu
    return max_diff / max_value


def _to_hist(scores: Iterable[float], num_bins: int, normed: bool = True) -> np.ndarray:
    """Creating a normalised histogram
    :param: scores: the ratings (not necessarily discrete)
    :param: num_bins: the number of bins to create
    :param: normed: whether to normalise the counts or not, by default true
    :return: the histogram
    """
    scores_array = np.array(scores)
    if len(scores_array) == 0:
        raise ValueError("Annotation list can not be empty.")

    # not keeping the values order when bins are not created
    counts, bins = np.histogram(a=scores_array, bins=num_bins)
    counts_normed = counts / counts.sum()
    return counts_normed if normed else counts
