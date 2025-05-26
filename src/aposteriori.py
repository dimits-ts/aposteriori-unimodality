from typing import Any, TypeVar
from collections.abc import Collection

import numpy as np
import scipy
import statsmodels.stats.multitest


FactorType = TypeVar("Factor Type")


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

    def update_with_factors(
        self, new_stats: dict[Any, float], all_factors: list
    ):
        """
        Update the _ListDict with at most one extra value per factor, keeping
        all internal lists at the same length.
        If a factor isn't present in new_stats, append np.nan instead.

        :param new_stats: Dictionary of {factor: stat}
        :param all_factors: List of all possible factors
            (ensures all keys updated)
        """
        for factor in all_factors:
            if factor in new_stats:
                self[factor] = new_stats[factor]
            else:
                self[factor] = np.nan


# code adapted from John Pavlopoulos
# https://github.com/ipavlopoulos/ndfu/blob/main/src/__init__.py
def dfu(x: Collection[float], bins: int, normalized: bool = False) -> float:
    """
    Computes the Distance From Unimodality measure for a list of annotations
    :param: x: a sequence of annotations, not necessarily discrete
    :type x: Collection[float]
    :param bins: number of bins. If data is discrete, it is advisable to use
        the number of modes. Example: An annotation task in the 1-5 LIKERT
        scale should use 5 bins.
    :type bins: int
    :param normalized: set to true to normalize the measure to the [0,1] range
        (normalized Distance From Unimodality - nDFU)
    :type normalized: bool
    :raises ValueError: if input_data is empty or number of bins is less than 1
    :return: the DFU score of the sequence
    """
    if bins <= 1:
        raise ValueError("Number of bins must be at least two.")

    hist = _to_hist(x, bins=bins)
    if hist.size == 0:
        return np.nan

    max_value = np.max(hist)
    pos_max = np.argmax(hist)

    # right search
    right_diffs = hist[pos_max + 1 :] - hist[pos_max:-1]
    max_rdiff = right_diffs.max(initial=0)

    # left search
    if pos_max > 0:
        left_diffs = hist[0:pos_max] - hist[1 : pos_max + 1]
        max_ldiff = left_diffs[left_diffs > 0].max(initial=0)
    else:
        max_ldiff = 0

    max_diff = max(max_rdiff, max_ldiff)
    dfu_stat = max_diff / max_value if normalized else max_diff
    return float(dfu_stat)


def aposteriori_unimodality(
    annotations: Collection[float],
    factor_group: Collection[FactorType],
    comment_group: Collection[FactorType],
    bins: int,
) -> float:
    """
    Perform the Aposteriori Unimodality Test to identify whether any annotator
    group, defined by a particular Socio-Demographic Beackground (SDB)
    attribute (e.g., gender, age), contributes significantly to the observed
    polarization in a discussion.

    This method tests whether partitioning annotations by a specific factor
    (such as gender or age group) systematically reduces within-group
    polarization (as measured by Distance from Unimodality, DFU), relative to
    the global polarization. It aggregates comment-level polarization
    differences and performs statistical testing across the discussion.

    :param annotations:
        A list of annotation scores, where each element corresponds to an
        annotation (e.g., a toxicity score) made by an annotator.
        Needs not be discrete.
    :type annotations: list[float]

    :param factor_group:
        A list indicating the group assignment (e.g., 'male', 'female') of
        the annotator who produced each annotation. For example, if two
        annotations were made by a male and female annotator respectively,
        the provided factor_group would be ["male", "female"].
        female annotator
    :type factor_group: list[`FactorType`]

    :param comment_group:
        A list of comment identifiers, where each element associates an
        annotation with a specific comment in the discussion.
    :type comment_group: list[`FactorType`]

    :param bins:
        The number of bins to use when computing the DFU polarization metric.
        If data is discrete, it is advisable to use the number of modes.
        Example: An annotation task in the 1-5 LIKERT scale should use 5 bins.
    :type bins: int

    :returns:
        A list of pvalues for each factor of the selected SDB dimension.
        A low p-value indicates that the group likely contributes to the
        observed polarization.
    :rtype: float

    :raises ValueError:
        If the given lists are not the same length, or are empty.

    .. seealso::
        - :func:`dfu` – Computes the Distance from Unimodality.

    .. note::
        The test is conservative by design and well-suited to annotation tasks
        with a small number of group comparisons.
        Recommended to use with discussions having numerous comments.
        The test is relatively robust even with a small number of annotations
        per comment.
    """
    # data prep
    _validate_input(annotations, factor_group, comment_group)
    annotations = np.array(annotations)
    factor_group = np.array(factor_group)
    comment_group = np.array(comment_group)

    # keeps list for each factor, each value in the list is a comment
    factor_dict = _ListDict()
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
        factor_dict.update_with_factors(comment_factor_ndfus, all_factors)

        # update comment ndfu
        global_ndfus.append(dfu(all_comment_annotations, bins=bins))

    raw_pvalues = _raw_significance(global_ndfus, factor_dict)
    corrected_pvalues = _correct_significance(raw_pvalues, alpha=0.001)
    return corrected_pvalues


def _validate_input(
    annotations: Collection[int],
    annotator_group: Collection[FactorType],
    comment_group: Collection[FactorType],
) -> None:
    if not (len(annotations) == len(annotator_group) == len(comment_group)):
        raise ValueError(
            "Length of provided lists must be the same, "
            + f"but len(annotations)=={len(annotations)}, "
            + f"len(annotator_group)=={len(annotator_group)}, "
            + f"len(comment_group)=={len(comment_group)}"
        )


def _polarization_stat(
    all_comment_annotations: np.ndarray[float],
    feature_group: np.ndarray[FactorType],
    bins: int,
) -> dict[FactorType, float]:
    """
    Generate the polarization stat (dfu diff stat) for each factor of the
    selected feature, for one comment.

    :param all_comment_annotations: An array containing all annotations
        for the current comment
    :type all_comment_annotations: np.ndarray[float]
    :param feature_group: An array where each value is a distinct level of
        the currently considered factor
    :type annotator_group: np.ndarray[`FactorType`]
    :param bins: number of annotation levels
    :type bins: int
    :return: The polarization stats for each level of the currently considered
        factor, for one comment
    :rtype: np.ndarray
    """
    if all_comment_annotations.shape != feature_group.shape:
        raise ValueError("Value and group arrays must be the same length.")

    if len(all_comment_annotations) == 0:
        raise ValueError("Empty annotation list given.")

    stats = {}
    for factor in np.unique(feature_group):
        factor_annotations = all_comment_annotations[feature_group == factor]
        if len(factor_annotations) == 0:
            stats[factor] = np.nan
        else:
            stats[factor] = dfu(factor_annotations, bins=bins)

    return stats


def _raw_significance(
    global_ndfus: list[float], stats_by_factor: _ListDict
) -> dict[FactorType, float]:
    """
    Performs a means test to determine the significance of
    differences in aposteriori statistics for a specific feature.

    :param global_ndfus: A list of aposteriori statistics for a feature.
    :type global_ndfus: `FactorType`
    :return: The aposteriori unimodality significance for each factor
    :rtype: dict[`FactorType`, float]
    :raises ValueError: if there is a mismatch between the number of comments 
        in the provided dictionary and the global_ndfus for any factor
    """
    if len(global_ndfus) == 0:
        return {}

    pvalues_by_factor = {}

    for factor, factor_ndfus in stats_by_factor.items():
        x = global_ndfus
        y = factor_ndfus

        if len(x) != len(y):
            raise ValueError(
                f"Number of comments ({len(y)}) "
                f"is different than number of global dfus ({len(x)}) "
                f"for factor {factor}."
            )

        pvalue = scipy.stats.ttest_rel(
            x, y, alternative="greater", nan_policy="omit"
        ).pvalue
        pvalues_by_factor[factor] = pvalue

    return pvalues_by_factor


def _correct_significance(
    raw_pvalues: dict[FactorType, float], alpha: float = 0.05
) -> dict[FactorType, float]:
    """
    Apply a statistical correction to pvalues from multiple alternative
    hypotheses.

    :param raw_pvalues: the pvalue of each hypothesis
    :type raw_pvalues: dict[`FactorType`, float]
    :param alpha: the target significance, defaults to 0.05
    :type alpha: float, optional
    :return: the corrected pvalues for each hypothesis
    :rtype: dict[`FactorType`, float]
    """
    if len(raw_pvalues) == 0:
        return {}

    if not np.any([0 <= np.array(x) <= 1 for x in raw_pvalues.values()]):
        raise ValueError("Invalid pvalues given for correction.")

    # place each pvalue in an ordered list
    keys, raw_pvalue_ls = zip(*raw_pvalues.items())  # keep key-value order
    corrected_pvalue_ls = _apply_correction(raw_pvalue_ls, alpha)
    # repackage dictionary
    corrected_pvalues_dict = dict(zip(keys, corrected_pvalue_ls))
    return corrected_pvalues_dict


def _apply_correction(pvalues: Collection[float], alpha: float) -> np.ndarray:
    corrected_stats = statsmodels.stats.multitest.multipletests(
        np.array(pvalues),
        alpha=alpha,
        method="bonferroni",
        is_sorted=False,
        returnsorted=False,
    )
    return corrected_stats[1]


def _to_hist(scores: np.ndarray[float], bins: int) -> np.ndarray:
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
    counts, bins = np.histogram(a=scores_array, bins=bins, density=True)
    return counts
