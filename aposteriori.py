from typing import Any, Iterable

import numpy as np
import scipy


class _ListDict:
    def __init__(self):
        self.dict = {}

    def keys(self) -> list[Any, list[Any]]:
        return self.dict.keys()

    def values(self) -> list[Any, list[Any]]:
        return self.dict.values()

    def items(self) -> list[tuple[Any, list]]:
        return self.dict.items()

    def __getitem__(self, key):
        return self.dict[key]

    def __setitem__(self, key, value):
        if key in self.dict:
            self.dict[key].append(value)
        else:
            self.dict[key] = [value]


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


def aposteriori_unimodality(
    annotations: list[int], factor_group: list[Any], comment_group: list[Any]
) -> float:
    # data prep
    _validate_input(annotations, factor_group, comment_group)
    annotations = np.array(annotations)
    factor_group = np.array(factor_group)
    comment_group = np.array(comment_group)

    stats_by_factor = (
        _ListDict()
    )  # keeps list for each factor, each value in the list is a comment
    global_ndfus = []  # ndfus when not partioned by any factor
    all_factors = np.unique(factor_group)

    # select comment
    for curr_comment_id in np.unique(comment_group):
        # select only annotations relevant to this comment
        is_in_curr_comment = comment_group == curr_comment_id
        all_comment_annotations = annotations[is_in_curr_comment]
        comment_annotator_groups = factor_group[is_in_curr_comment]

        # update results for each factor according to new comment
        comment_factor_ndfus = _calculate_comment_factor_ndfus(
            all_comment_annotations, comment_annotator_groups
        )
        _update_stats_by_factor(stats_by_factor, comment_factor_ndfus, all_factors)

        # update comment ndfu
        global_ndfus.append(ndfu(all_comment_annotations))

    return _significance(global_ndfus, stats_by_factor)


def _update_stats_by_factor(
    stats_by_factor: _ListDict, new_stats: dict, all_factors: list
) -> None:
    for factor, ndfu in new_stats.items():
        stats_by_factor[factor] = ndfu

    # keep size of all groups the same even if no annotation from that factor was observed in a comment
    for factor in all_factors:
        if factor not in new_stats.keys():
            stats_by_factor[factor] = np.nan


def _calculate_comment_factor_ndfus(
    all_comment_annotations: np.ndarray, feature_group: np.ndarray
) -> dict[Any, float]:
    """
    Generate the aposteriori stat (ndfu diff stat) for each annotation level, for one comment.

    :param all_comment_annotations: An array containing all annotations for the current comment
    :type all_comment_annotations: np.ndarray
    :param annotator_group: An array where each value is a distinct level of the currently considered factor
    :type annotator_group: np.ndarray
    :return: A numpy array containing the ndfu stats for each level of the currently considered factor, for one comment
    :rtype: np.ndarray
    """
    stats = {}
    for factor in np.unique(feature_group):
        factor_annotations = all_comment_annotations[feature_group == factor]
        if len(factor_annotations) == 0:
            stats[factor] = np.nan
        else:
            stats[factor] = ndfu(factor_annotations)

    return stats


def _significance(global_ndfus: list[float], stats_by_factor: _ListDict) -> float:
    """
    Performs a Wilcoxon signed-rank test to determine the significance of differences
    in aposteriori statistics for a specific group.

    Args:
        level_aposteriori_statistics (list[float]): A list of aposteriori statistics for a group.

    Returns:
        float: The p-value of the Wilcoxon test. If no difference is detected, returns 1.
    """
    pvalues_by_factor = {}

    for factor, factor_ndfus in stats_by_factor.items():
        x = global_ndfus
        y = factor_ndfus
        pvalue = scipy.stats.wilcoxon(
            x, y, alternative="greater", nan_policy="omit"
        ).pvalue
        pvalues_by_factor[factor] = pvalue

    return pvalues_by_factor


def _validate_input(
    annotations: list[int], annotator_group: list[Any], comment_group: list[Any]
) -> None:
    if not (len(annotations) == len(annotator_group) == len(comment_group)):
        raise ValueError(
            "Length of provided lists must be the same, "
            + f"but len(annotations)=={len(annotations)}, "
            + f"len(annotator_group)=={len(annotator_group)}, "
            + f"len(comment_group)=={len(comment_group)}"
        )


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
