from typing import TypeVar, Iterable, Any
from collections.abc import Collection

import numpy as np

from . import _list_dict


FactorType = TypeVar("Factor Type")


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

    max_value = np.max(hist)
    pos_max = np.argmax(hist)

    # right search
    right_diffs = hist[pos_max+1:] - hist[pos_max:-1]
    max_rdiff = right_diffs.max(initial=0)

    # left search
    if pos_max > 0:
        left_diffs = hist[0:pos_max] - hist[1:pos_max + 1]
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
    iterations: int = 100,
) -> dict[FactorType, float]:
    """
    Perform the Aposteriori Unimodality Test to identify whether any annotator
    group, defined by a particular Socio-Demographic Beackground (SDB)
    attribute (e.g., gender, age), contributes significantly to the observed
    polarization in a discussion.

    This method tests whether partitioning annotations by a specific factor
    (such as gender or age group) systematically reduces within-group
    polarization (as measured by Distance from Unimodality, DFU), relative to
    the global polarization.

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
    :param iterations:
        The number of randomized groups compared against the original groups.
        A larger number makes the method more accurate,
        but also more computationally expensive.
    :type iterations: int
    :returns:
        The apunim kappa value for each factor of the selected SDB dimension.
        If kappa~=0, the polarization can be explained by chance.
        If kappa>0, increased polarization can not be explained by chance,
        but rather must be partially caused by differences between
        the SDB groups.
        If kappa<0, the decrease in polarization is partially caused by
        differences between the SDB groups.
    :rtype: dict[`FactorType`, float]
    :raises ValueError:
        If the given lists are not the same length, are empty,
        are comprised of a single group, or a single comment.

    .. seealso::
        - :func:`dfu` - Computes the Distance from Unimodality.

    .. note::
        The test is relatively robust even with a small number of annotations
        per comment.
    """
    # data prep
    _validate_input(annotations, factor_group, comment_group)
    annotations = np.array(annotations)
    factor_group = np.array(factor_group)
    comment_group = np.array(comment_group)

    # keeps list for each factor, each value in the list is a comment
    all_factors = _unique(factor_group)
    factor_dict = _list_dict._ListDict()
    randomized_ndfu_dict = _list_dict._ListDict()

    # select comment
    for curr_comment_id in _unique(comment_group):
        # select only annotations relevant to this comment
        is_in_curr_comment = comment_group == curr_comment_id
        all_comment_annotations = annotations[is_in_curr_comment]
        comment_annotator_groups = factor_group[is_in_curr_comment]
        lengths_by_factor = {
            factor: np.sum(comment_annotator_groups == factor)
            for factor in all_factors
        }

        # update results for each factor according to new comment
        comment_factor_ndfus = _factor_polarization_stat(
            all_comment_annotations, comment_annotator_groups, bins=bins
        )
        factor_dict.add_dict(comment_factor_ndfus)

        # get ndfu of randomized comments
        comment_randomized_ndfus = _random_polarization_stat(
            # number of observations per factor
            annotations=all_comment_annotations,
            group_sizes=lengths_by_factor,
            all_factors=all_factors,
            bins=bins,
            iterations=iterations,
        )
        randomized_ndfu_dict.add_dict(comment_randomized_ndfus)

    apunim_kappa = _apunim_kappa(
        observed_factors=factor_dict,
        randomized_factors=randomized_ndfu_dict,
        all_factors=all_factors,
    )
    return apunim_kappa


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

    if len(annotations) == 0:
        raise ValueError("No annotations given.")

    if len(_unique(annotator_group)) < 2:
        raise ValueError("Only one group was provided.")

    if len(_unique(comment_group)) < 2:
        raise ValueError(
            "Only one comment was provided. "
            "The Aposteriori Unimodality Test is defined for discussions, "
            "not individual comments."
        )


def _factor_polarization_stat(
    all_comment_annotations: np.ndarray[float],
    annotator_group: np.ndarray[FactorType],
    bins: int,
) -> dict[FactorType, float]:
    """
    Generate the polarization stat (dfu diff stat) for each factor of the
    selected feature, for one comment.

    :param all_comment_annotations: An array containing all annotations
        for the current comment
    :type all_comment_annotations: np.ndarray[float]
    :param annotator_group: An array where each value is a distinct level of
        the currently considered factor
    :type annotator_group: np.ndarray[`FactorType`]
    :param bins: number of annotation levels
    :type bins: int
    :return: The polarization stats for each level of the currently considered
        factor, for one comment
    :rtype: np.ndarray
    """
    if all_comment_annotations.shape != annotator_group.shape:
        raise ValueError("Value and group arrays must be the same length.")

    if len(all_comment_annotations) == 0:
        raise ValueError("Empty annotation list given.")

    stats = {}
    for factor in _unique(annotator_group):
        factor_annotations = all_comment_annotations[annotator_group == factor]
        if len(factor_annotations) == 0:
            stats[factor] = np.nan
        else:
            stats[factor] = dfu(factor_annotations, bins=bins, normalized=True)

    return stats


def _random_polarization_stat(
    annotations: np.ndarray[float],
    group_sizes: dict[FactorType, int],
    all_factors: Iterable[FactorType],
    bins: int,
    iterations: int,
) -> dict[FactorType, float]:
    # Split annotations in len(group_lengths) groups,
    # each of which has a length equal to the entries in group_lengths.
    # Then for each group run _factor_polarization_stat
    all_random_ndfus = _list_dict._ListDict()
    for i in range(iterations):
        random_groups = _random_partition(
            annotations, np.array(list(group_sizes.values()))
        )
        random_annotations = np.hstack(random_groups)
        pseudo_groups = np.array(
            [
                factor
                for factor, size in group_sizes.items()
                for _ in range(size)
            ]
        )
        random_ndfus = _factor_polarization_stat(
            random_annotations, pseudo_groups, bins
        )
        all_random_ndfus.add_dict(random_ndfus)

    # return the average of all polarization stats for each factor
    mean_random_ndfu_dict = {
        factor: np.mean(stats) for factor, stats in all_random_ndfus.items()
    }
    return mean_random_ndfu_dict


def _random_partition(
    arr: np.ndarray, sizes: np.ndarray[int]
) -> list[np.ndarray]:
    """
    Randomly partition a numpy array into groups of given sizes.

    Parameters:
    - arr: numpy array to be partitioned.
    - sizes: list of integers indicating the size of each group.

    Returns:
    - List of numpy arrays, each with the size specified in `sizes`.

    Raises:
    - ValueError: if the sum of sizes does not match the length of arr.
    """
    if np.sum(sizes) != len(arr):
        raise ValueError(
            f"Sum of sizes ({np.sum(sizes)}) must equal length "
            f"of input array ({len(arr)})."
        )

    shuffled = np.random.default_rng().permutation(arr)
    partitions = []
    start = 0
    for size in sizes:
        end = start + size
        partitions.append(shuffled[start:end])
        start = end

    return partitions


def _apunim_kappa(
    observed_factors: dict[FactorType, float],
    randomized_factors: dict[FactorType, float],
    all_factors: Iterable[FactorType],
) -> float:
    observed_means = {
        f: np.nanmean(vals) for f, vals in observed_factors.items()
    }
    randomized_means = {
        f: np.nanmean(vals) for f, vals in randomized_factors.items()
    }

    # Cohen's kappa style formula
    apunim_kappa = {}
    for f in all_factors:
        O_f = observed_means[f]
        E_f = randomized_means[f]
        if np.isnan(O_f) or np.isnan(E_f) or E_f >= 1.0:
            apunim_kappa[f] = np.nan
        else:
            apunim_kappa[f] = (O_f - E_f) / (1.0 - E_f)

    return apunim_kappa


def _to_hist(scores: np.ndarray[float], bins: int) -> np.ndarray:
    """
    Creates a normalised histogram. Used for DFU calculation.
    :param: scores: the ratings (not necessarily discrete)
    :param: num_bins: the number of bins to create
    :param: normed: whether to normalise the counts or not, by default true
    :return: the histogram
    """
    scores_array = np.array(scores)
    if len(scores_array) == 0:
        raise ValueError("Annotation list can not be empty.")

    counts, bins = np.histogram(a=scores_array, bins=bins, density=True)
    return counts


def _unique(x: Iterable[Any]) -> Iterable[Any]:
    return set(x)
