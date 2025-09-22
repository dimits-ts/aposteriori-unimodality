from typing import TypeVar, Iterable, Any
from collections import namedtuple
from collections.abc import Collection

import statsmodels.stats.multitest
import numpy as np

from . import _list_dict


ApunimResult = namedtuple("ApunimResult", ["value", "pvalue"])
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
    alpha: float = 0.05,
) -> dict[FactorType, ApunimResult]:
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
        A named tuple containing the apunim metric ("value")
        and pvalue ("pvalue") for each factor of the selected SDB dimension.
        If value~=0, the polarization can be explained by chance.
        If value>0, increased polarization can not be explained by chance,
        but rather must be partially caused by differences between
        the SDB groups.
        If value<0, the decrease in polarization is partially caused by
        differences between the SDB groups.
    :rtype: dict[`FactorType`, ApunimResult]
    :raises ValueError:
        If the given lists are not the same length, are empty,
        are comprised of a single group, or a single comment.

    .. seealso::
        - :func:`dfu` - Computes the Distance from Unimodality.

    .. note::
        The test is relatively robust even with a small number of annotations
        per comment. The pvalue estimation is non-parametric.
    """
    # data prep
    _validate_input(annotations, factor_group, comment_group, iterations, bins)
    annotations = np.array(annotations)
    factor_group = np.array(factor_group)
    comment_group = np.array(comment_group)

    all_factors = _unique(factor_group)
    observed_dfu_dict = _list_dict._ListDict()
    apriori_dfu_dict = _list_dict._ListDict()

    # gather stats per comment
    for curr_comment_id in _unique(comment_group):
        is_in_curr_comment = comment_group == curr_comment_id
        all_comment_annotations = annotations[is_in_curr_comment]
        comment_annotator_groups = factor_group[is_in_curr_comment]
        lengths_by_factor = {
            factor: np.sum(comment_annotator_groups == factor)
            for factor in all_factors
        }

        observed_dfu_dict.add_dict(
            _factor_dfu_stat(
                all_comment_annotations,
                comment_annotator_groups,
                bins=bins,
            )
        )

        apriori_dfu_dict.add_dict(
            _apriori_polarization_stat(
                annotations=all_comment_annotations,
                group_sizes=lengths_by_factor,
                bins=bins,
                iterations=iterations,
            )
        )

    # compute raw results per factor
    raw_results = {}
    for factor in all_factors:
        res = _aposteriori_polarization_stat(
            observed_dfu_dict[factor],
            apriori_dfu_dict[factor],
        )
        raw_results[factor] = res

    # extract valid p-values for correction
    factors_with_pvals = [
        f
        for f, res in raw_results.items()
        if res.pvalue is not None and not np.isnan(res.pvalue)
    ]
    raw_pvals = [raw_results[f].pvalue for f in factors_with_pvals]

    # apply correction
    corrected_pvals = _apply_correction_to_results(raw_pvals, alpha)
    corrected_results = {}
    for factor in all_factors:
        # reassemble results dict
        corrected_results[factor] = ApunimResult(
            value=raw_results[factor].value,
            pvalue=(
                corrected_pvals[factors_with_pvals.index(factor)]
                if factor in factors_with_pvals
                else np.nan
            ),
        )
    return corrected_results


def _validate_input(
    annotations: Collection[int],
    annotator_group: Collection[FactorType],
    comment_group: Collection[FactorType],
    iterations: int,
    bins: int,
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

    if iterations < 1:
        raise ValueError("iterations must be at least 1.")

    if bins < 2:
        raise ValueError("Number of bins has to be at least 2.")


def _factor_dfu_stat(
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
    :rtype: dict[FactorType, float]
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


def _apriori_polarization_stat(
    annotations: np.ndarray[float],
    group_sizes: dict[FactorType, int],
    bins: int,
    iterations: int,
) -> dict[FactorType, list[float]]:
    """
    For a single comment's annotations, generate `iterations` random partitions
    that respect the given group_sizes, compute the normalized DFU for each
    resulting group, and return a dict mapping factor -> list of DFU values
    (one value per iteration).

    :param annotations: 1D numpy array of annotation values for the comment
    :param group_sizes:
        dict mapping factor -> size for that factor in this comment
    :param bins: number of bins to use when computing DFU
    :param iterations: number of random partitions to sample
    :return: dict mapping factor -> list[float] (length == iterations)
    """
    # order of factors must be preserved so results align
    factors = list(group_sizes.keys())
    sizes = np.array([group_sizes[f] for f in factors], dtype=int)

    if np.sum(sizes) != len(annotations):
        raise ValueError(
            "Sum of provided group sizes must equal the number of annotations."
        )

    # prepare result lists
    results: dict[FactorType, list[float]] = {f: [] for f in factors}

    for _ in range(iterations):
        partitions = _random_partition(annotations, sizes)
        # partitions is a list of numpy arrays in the same order as `factors`
        for f, part in zip(factors, partitions):
            if part.size == 0:
                results[f].append(np.nan)
            else:
                results[f].append(dfu(part, bins=bins))

    return results


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


def _aposteriori_polarization_stat(
    observed_dfus: list[float],
    randomized_dfus: list[list[float]],
) -> ApunimResult:
    """
    Compute Cohen's kappa-style AP-unimodality statistic and
    non-parametric p-value for a single factor.
    """
    # TODO: the error is here. we should give means to the stat estimation,
    # but not the pvalue estimation
    O_f = np.nanmean(observed_dfus)
    E_f = np.nanmean([np.nanmean(r) for r in randomized_dfus])

    if np.isnan(O_f) or np.isnan(E_f):
        print(O_f, E_f)
        return ApunimResult(np.nan, np.nan)

    kappa = (O_f - E_f) / (1.0 - E_f)

    # compute kappa for each randomization (null distribution)
    kappa_null = []
    for r in randomized_dfus:
        O_r = np.nanmean(r)
        E_r = np.nanmean(
            [np.nanmean(rr) for rr in randomized_dfus if rr is not r]
        )
        if not (np.isnan(O_r) or np.isnan(E_r)):
            kappa_null.append((O_r - E_r) / (1.0 - E_r))

    kappa_null = np.array(kappa_null)

    # two sided test
    p_value = np.mean(np.abs(kappa_null) >= abs(kappa))

    return ApunimResult(kappa, p_value)


def _apply_correction_to_results(
    pvalues: Collection[float], alpha: float = 0.05
) -> np.ndarray:
    """
    Apply multiple hypothesis correction to a list of p-values.
    Returns corrected p-values in the same order.
    """
    if len(pvalues) == 0:
        return np.array([])

    if np.any((np.array(pvalues) < 0) | (np.array(pvalues) > 1)):
        raise ValueError("Invalid pvalues given for correction.")

    return _apply_correction(pvalues, alpha)


def _apply_correction(pvalues: Collection[float], alpha: float) -> np.ndarray:
    corrected_stats = statsmodels.stats.multitest.multipletests(
        np.array(pvalues),
        alpha=alpha,
        method="fdr_bh",
        is_sorted=False,
        returnsorted=False,
    )
    return corrected_stats[1]


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
    # preserve first-seen order
    return list(dict.fromkeys(x))
