from typing import Any, Iterable
import copy

import numpy as np
import scipy
import statsmodels.stats.multitest


class _ListDict:
    """
    A dictionary appending multiple values with the same key
    to a list instead of overwriting them.
    """

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


# code adapted from John Pavlopoulos
# https://github.com/ipavlopoulos/ndfu/blob/main/src/__init__.py
def ndfu(input_data: Iterable[float], bins: int) -> float:
    """The normalized Distance From Unimodality measure
    :param: input_data: a sequence of annotations, not necessarily discrete
    :raises ValueError: if input_data is empty
    :return: the nDFU score of the sequence
    """
    # compute DFU
    hist = _to_hist(input_data, bins=bins)

    if len(hist) == 0:
        return np.nan
    else:
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
    annotations: list[int],
    factor_group: list[Any],
    comment_group: list[Any],
    bins: int,
) -> float:
    # data prep
    _validate_input(annotations, factor_group, comment_group)
    annotations = np.array(annotations)
    factor_group = np.array(factor_group)
    comment_group = np.array(comment_group)

    stats_by_factor = (
        _ListDict()
    )  # keeps list for each factor, each value in the list is a comment
    global_ndfus = []  # ndfus when not partitioned by any factor
    all_factors = np.unique(factor_group)

    # select comment
    for curr_comment_id in np.unique(comment_group):
        # select only annotations relevant to this comment
        is_in_curr_comment = comment_group == curr_comment_id
        all_comment_annotations = annotations[is_in_curr_comment]
        comment_annotator_groups = factor_group[is_in_curr_comment]

        # update results for each factor according to new comment
        comment_factor_ndfus = _polarization_stat(
            all_comment_annotations, comment_annotator_groups, bins=bins
        )
        stats_by_factor = _update_stats_by_factor(
            stats_by_factor, comment_factor_ndfus, all_factors
        )

        # update comment ndfu
        global_ndfus.append(ndfu(all_comment_annotations, bins=bins))

    raw_pvalues = _raw_significance(global_ndfus, stats_by_factor)
    corrected_pvalues = _correct_significance(raw_pvalues, alpha=0.001)
    return corrected_pvalues


def _update_stats_by_factor(
    old_stats: _ListDict, new_stats: dict[Any, float], all_factors: list
) -> _ListDict:
    """
    Update the ListDict with at most one extra value per factor, keeping all
    internal arrays at same length. If a factor isn't present, it is replaced
    by a nan.

    :param old_stats: The ListDict to be updated
    :type stats_by_factor: _ListDict
    :param new_stats: A dictionary holding pairs of factor:stat
    :type new_stats: dict[Any, float]
    :param all_factors: A list of all possible factors in the feature.
        Needed to ensure that all internal arrays have the same length.
    :type all_factors: list
    """
    updated_dict = copy.copy(old_stats)

    for factor, ndfu in new_stats.items():
        updated_dict[factor] = ndfu

    # keep size of all groups the same even if no annotation from that factor
    # was observed in a comment
    for factor in all_factors:
        if factor not in new_stats.keys():
            updated_dict[factor] = np.nan

    return updated_dict


def _polarization_stat(
    all_comment_annotations: np.ndarray, feature_group: np.ndarray, bins: int
) -> dict[Any, float]:
    """
    Generate the polarization stat (ndfu diff stat) for each factor of the
    selected feature, for one comment.

    :param all_comment_annotations: An array containing all annotations
        for the current comment
    :type all_comment_annotations: np.ndarray
    :param feature_group: An array where each value is a distinct level of
        the currently considered factor
    :type annotator_group: np.ndarray
    :param bins: number of annotation levels
    :return: The polarization stats for each level of the currently considered
        factor, for one comment
    :rtype: np.ndarray
    """
    stats = {}
    for factor in np.unique(feature_group):
        factor_annotations = all_comment_annotations[feature_group == factor]
        if len(factor_annotations) == 0:
            stats[factor] = np.nan
        else:
            stats[factor] = ndfu(factor_annotations, bins=bins)

    return stats


def _correct_significance(
    raw_pvalues: Iterable[float], alpha: float = 0.05
) -> dict[Any, float]:
    # place each pvalue in an ordered list
    keys, raw_pvalue_ls = zip(*raw_pvalues.items())  # keep key-value order
    corrected_pvalue_ls = _apply_correction(raw_pvalue_ls, alpha)
    # repackage dictionary
    corrected_pvalues_dict = dict(zip(keys, corrected_pvalue_ls))
    return corrected_pvalues_dict


def _raw_significance(
    global_ndfus: list[float], stats_by_factor: _ListDict
) -> dict[Any, float]:
    """
    Performs a means test to determine the significance of
    differences in aposteriori statistics for a specific feature.

    :param global_ndfus: A list of aposteriori
        statistics for a feature.
    :type level_aposteriori_statistics: (list[float])
    :return: The aposteriori unimodality significance for each factor
    :rtype: dict[Any, float]
    """
    pvalues_by_factor = {}

    for factor, factor_ndfus in stats_by_factor.items():
        x = global_ndfus
        y = factor_ndfus
        pvalue = scipy.stats.ttest_rel(
            x, y, alternative="greater", nan_policy="omit"
        ).pvalue
        pvalues_by_factor[factor] = pvalue

    return pvalues_by_factor


def _apply_correction(pvalues: Iterable[float], alpha: float) -> np.ndarray:
    corrected_stats = statsmodels.stats.multitest.multipletests(
        np.array(pvalues),
        alpha=alpha,
        method="bonferroni",
        is_sorted=False,
        returnsorted=False,
    )
    return corrected_stats[1]


def _validate_input(
    annotations: list[int],
    annotator_group: list[Any],
    comment_group: list[Any],
) -> None:
    if not (len(annotations) == len(annotator_group) == len(comment_group)):
        raise ValueError(
            "Length of provided lists must be the same, "
            + f"but len(annotations)=={len(annotations)}, "
            + f"len(annotator_group)=={len(annotator_group)}, "
            + f"len(comment_group)=={len(comment_group)}"
        )


def _to_hist(scores: Iterable[float], bins: int) -> np.ndarray:
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
    # normalize results using density=True to make stat invariant to scale
    counts, bins = np.histogram(a=scores_array, bins=bins, density=True)
    return counts / counts.sum()
