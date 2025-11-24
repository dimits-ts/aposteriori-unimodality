import warnings
import math
from collections import namedtuple
from collections.abc import Collection
from typing import Any, TypeVar

import statsmodels.stats.multitest
import scipy.stats
import numpy as np
from numpy.typing import NDArray

from . import _list_dict


FactorType = TypeVar("FactorType")
ApunimResult = namedtuple("ApunimResult", ["apunim", "pvalue"])


# code adapted from John Pavlopoulos
# https://github.com/ipavlopoulos/ndfu/blob/main/src/__init__.py
def dfu(x: Collection[float], bins: int, normalized: bool = True) -> float:
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
    right_diffs = hist[pos_max + 1:] - hist[pos_max:-1]
    max_rdiff = right_diffs.max(initial=0)

    # left search
    if pos_max > 0:
        left_diffs = hist[0:pos_max] - hist[1: pos_max + 1]
        max_ldiff = left_diffs[left_diffs > 0].max(initial=0)
    else:
        max_ldiff = 0

    max_diff = max(max_rdiff, max_ldiff)
    dfu_stat = max_diff / max_value if normalized else max_diff
    return float(dfu_stat)


def aposteriori_unimodality(
    annotations: Collection[float],
    factor_group: Collection[FactorType],  # type: ignore
    comment_group: Collection[FactorType],  # type: ignore
    num_bins: int | None = None,
    iterations: int = 100,
    alpha: float | None = 0.05,
    pvalue_estimation: str = "both",
    two_sided: bool = True,
    seed: int | None = None,
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
    :param num_bins:
        The number of bins to use when computing the DFU polarization metric.
        If data is discrete, it is advisable to use the number of modes.
        Example: An annotation task in the 1-5 LIKERT scale should use 5 bins.
        None to create as many bins as the distinct values in the annotations.
        WARNING: If set to None, check whether all possible values are
        represented at least once in the provided annotation.
    :type num_bins: int
    :param iterations:
        The number of randomized groups compared against the original groups.
        A larger number makes the method more accurate,
        but also more computationally expensive.
    :type iterations: int
    :param alpha:
        The target statistical significance. Used to apply pvalue correction
        for multiple comparisons. None to disable pvalue corrections.
    :type alpha: float | None
    :param pvalue_estimation:
        Which pvalue estimation method to use.
        "parametric", "non parametric", "both" or "none"
    :type pvalue_estimation: str
    :param two_sided:
        Whether the statistical tests run for both less and
        greater polarization, or just greater.
    :type two_sided: bool
    :param seed: The random seed used, None for non-deterministic outputs.
    :type seed: int | None
    :returns:
        A dictionary containing the apunim result ("apunim"),
        the parametric p-value ("p_param")
        and non-parametric p-value ("p_nonparam"),
        depending on the pvalue_estimation parameter.
        If apunim~=0, the polarization can be explained by chance.
        If apunim>0, increased polarization can not be explained by chance,
        but rather must be partially caused by differences between
        the sociodemographic groups.
        If apunim<0, the decrease in polarization is partially caused by
        differences between the sociodemographic groups.
    :rtype: dict[str, dict[FactorType, float]]
    :raises ValueError:
        If the given lists are not the same length, are empty,
        are comprised of a single group, or a single comment.

    .. seealso::
        - :func:`dfu` - Computes the Distance from Unimodality.

    .. note::
        The test is relatively robust even with a small number of annotations
        per comment. The pvalue estimation is non-parametric.
    """
    rng = np.random.default_rng(seed=seed)
    bins = num_bins if num_bins is not None else len(_unique(annotations))

    _validate_input(
        annotations,
        factor_group,
        comment_group,
        iterations,
        bins,
        alpha,
        pvalue_estimation,
    )

    annotations = np.array(annotations)
    factor_group: NDArray[Any] = np.array(factor_group)
    comment_group: NDArray[Any] = np.array(comment_group)
    all_factors = _unique(factor_group)

    # Identify comments with actual polarization
    valid_comments = _get_valid_comments(
        annotations=annotations,
        comment_group=comment_group,
        factor_group=factor_group,
        bins=bins,
    )
    if not valid_comments:
        raise ValueError("No polarized comments found.")

    (
        annotations,
        factor_group,
        comment_group,
        all_factors,
    ) = _filter_to_valid_comments(
        annotations,
        factor_group,
        comment_group,
        valid_comments,
    )

    observed_dfu_dict, apriori_dfu_dict = _compute_dfu_distributions(
        valid_comments,
        annotations,
        factor_group,
        comment_group,
        all_factors,
        bins,
        iterations,
        rng,
    )

    results = _compute_factor_results(
        observed_dfu_dict,
        apriori_dfu_dict,
        all_factors,
        two_sided,
    )

    # Apply p-value correction if needed
    if alpha is not None:
        results = _correct_pvalues(results, alpha)

    return results


def _filter_to_valid_comments(
    annotations,
    factor_group,
    comment_group,
    valid_comments,
):
    valid_mask = np.isin(comment_group, valid_comments)
    annotations = annotations[valid_mask]
    factor_group = factor_group[valid_mask]
    comment_group = comment_group[valid_mask]

    # update factors after filtering
    all_factors = _unique(factor_group)

    return annotations, factor_group, comment_group, all_factors


def _compute_dfu_distributions(
    valid_comments,
    annotations,
    factor_group,
    comment_group,
    all_factors,
    bins,
    iterations,
    rng,
):
    observed_dfu_dict = _list_dict._ListDict()
    apriori_dfu_dict = _list_dict._ListDict()

    for curr_comment in valid_comments:
        mask = comment_group == curr_comment
        comment_ann = annotations[mask]
        comment_groups = factor_group[mask]

        # counts per factor
        lengths_by_factor = {
            factor: int(np.count_nonzero(comment_groups == factor))
            for factor in all_factors
        }

        # observed DFUs
        observed_dfu_dict.add_dict(
            _factor_dfu_stat(comment_ann, comment_groups, bins=bins)
        )

        # randomized apriori DFUs
        apriori_dfu_dict.add_dict(
            _apriori_polarization_stat(
                annotations=comment_ann,
                group_sizes=lengths_by_factor,
                bins=bins,
                iterations=iterations,
                rng=rng,
            )
        )

    return observed_dfu_dict, apriori_dfu_dict


def _compute_factor_results(
    observed_dfu_dict,
    apriori_dfu_dict,
    all_factors,
    two_sided,
):
    results = {}

    for factor in all_factors:
        apunim = _aposteriori_polarization_stat(
            observed_dfus=observed_dfu_dict[factor],
            randomized_dfus=apriori_dfu_dict[factor],
        )

        pvalue = _aposteriori_pvalue_parametric(
            randomized_dfus=apriori_dfu_dict[factor],
            kappa=apunim,
            two_sided=two_sided,
        )

        results[factor] = ApunimResult(apunim=apunim, pvalue=pvalue)

    return results


def _correct_pvalues(results, alpha):
    factors, result_objs = zip(*results.items())
    pvals = [r.pvalue for r in result_objs]

    corrected = _apply_correction_to_results(pvals, alpha)

    return {
        f: ApunimResult(r.apunim, cp)
        for f, r, cp in zip(factors, result_objs, corrected)
    }


def _get_valid_comments(
    annotations: NDArray[np.float64],
    comment_group: NDArray[np.int64],
    factor_group: NDArray[Any],
    bins: int,
) -> list[int]:
    # --- FIRST LOOP: Identify valid comments ---
    valid_comments = []
    for curr_comment_id in _unique(comment_group):
        is_in_curr_comment = comment_group == curr_comment_id
        all_comment_annotations = annotations[is_in_curr_comment]
        comment_annotator_groups = factor_group[is_in_curr_comment]

        if len(all_comment_annotations) > 0 and _comment_is_valid(
            comment_annotations=all_comment_annotations,
            comment_annotator_groups=comment_annotator_groups,
            bins=bins,
        ):
            valid_comments.append(curr_comment_id)

    return valid_comments


def _validate_input(
    annotations: Collection[float],
    annotator_group: Collection[FactorType],
    comment_group: Collection[FactorType],
    iterations: int,
    bins: int,
    alpha: float | None,
    pvalue_estimation: str,
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

    valid = ["parametric", "non parametric", "both", "none"]
    if pvalue_estimation not in valid:
        raise ValueError(
            "pvalue_estimation must be one of the following: ", valid
        )
    if alpha is not None and (alpha < 0 or alpha > 1):
        raise ValueError("Alpha should be between 0 and 1.")


def _comment_is_valid(
    comment_annotations: Collection[float],
    comment_annotator_groups: Collection[FactorType],
    bins: int,
) -> bool:
    """
    A comment is valid if:
      1. It shows polarization (DFU > 0)
      2. It has at least two distinct annotator groups
    """

    # --- Check for polarization ---
    has_polarization = not np.isclose(
        dfu(comment_annotations, bins=bins, normalized=True),
        0,
        atol=0.01,
    )

    # --- annotator groups ---
    groups = [x for x in comment_annotator_groups if _is_not_none(x)]
    sufficient_groups = len(_unique(groups)) >= 2

    return has_polarization and sufficient_groups


def _factor_dfu_stat(
    all_comment_annotations: NDArray[np.float64],
    annotator_group: NDArray[Any],
    bins: int,
) -> dict[object, float]:
    """
    Generate the polarization stat (dfu diff stat) for each factor of the
    selected feature, for one comment.

    :param all_comment_annotations: An array containing all annotations
        for the current comment
    :type all_comment_annotations:  NDArray[float]
    :param annotator_group: An array where each value is a distinct level of
        the currently considered factor
    :type annotator_group:  NDArray[`FactorType`]
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
            stats[factor] = dfu(factor_annotations, bins=bins)

    return stats


def _apriori_polarization_stat(
    annotations: NDArray[np.float64],
    group_sizes: dict[Any, int],
    bins: int,
    iterations: int,
    rng: np.random.Generator,
) -> dict[object, list[float]]:
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
    results: dict[object, list[float]] = {f: [] for f in factors}

    for _ in range(iterations):
        partitions = _random_partition(arr=annotations, sizes=sizes, rng=rng)
        # partitions is a list of numpy arrays in the same order as `factors`
        for f, part in zip(factors, partitions):
            if part.size == 0:
                results[f].append(np.nan)
            else:
                results[f].append(dfu(part, bins=bins))
    return results


def _random_partition(
    arr: NDArray,
    sizes: NDArray[np.int64],
    rng: np.random.Generator,
) -> list[NDArray]:
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

    shuffled = rng.permutation(arr)
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
) -> float:
    """
    Compute AP-unimodality statistic and p-value.
    """
    if len(observed_dfus) == 0 or np.all(np.isnan(observed_dfus)):
        return np.nan

    O_f = np.nanmean(observed_dfus)

    # expected mean from randomizations
    # filters out all-nan expected values which may crop up
    means = [_safe_nanmean(r) for r in randomized_dfus]
    means = [m for m in means if not np.isnan(m)]
    if len(means) == 0:
        warnings.warn(
            "Apunim statistic is NaN because all randomized DFU estimates "
            "were NaN. This typically means that randomized groups were empty "
            "or had no variation in annotations.",
            RuntimeWarning,
        )
        return np.nan

    E_f = np.mean(means)
    if np.isclose(E_f, 1, atol=10e-3):
        warnings.warn(
            "Estimated polarization is very close to max. "
            "The aposteriori test may be unreliable."
        )
    if E_f == 1:
        warnings.warn(
            "Apunim statistic is NaN because the expected DFU (E_f) is 1, "
            "meaning all random partitions were already maximally polarized.",
            RuntimeWarning,
        )
        return np.nan

    apunim = (O_f - E_f) / (1.0 - E_f)
    return float(apunim)


def _aposteriori_pvalue_parametric(
    randomized_dfus: list[list[float]], kappa: float, two_sided: bool
) -> float:
    """
    Parametric p-value estimation for κ using a normal approximation.
    """
    if np.isnan(kappa):
        warnings.warn(
            "p-value could not be computed because the apunim statistic "
            "is NaN. This usually happens when a factor has no valid "
            "annotations.",
            RuntimeWarning,
        )
        return np.nan

    # compute null distribution of kappa as before
    kappa_null = []
    for i, r in enumerate(randomized_dfus):
        if len(r) == 0 or np.all(np.isnan(r)):
            continue
        O_r = np.nanmean(r)
        other_means = [
            _safe_nanmean(rr) for j, rr in enumerate(randomized_dfus) if j != i
        ]
        other_means = [m for m in other_means if not np.isnan(m)]
        if len(other_means) == 0:
            continue
        E_r = np.mean(other_means)
        kappa_null.append((O_r - E_r) / (1.0 - E_r))

    kappa_null = np.array(kappa_null)
    if len(kappa_null) < 2:
        warnings.warn(
            "p-value is NaN because the null distribution for kappa "
            "could not be estimated (fewer than two valid randomized "
            "DFU values). This often occurs when comments contain too few "
            "annotations per group.",
            RuntimeWarning,
        )
        return np.nan

    # use a one-sample t-test comparing kappa_null to the observed kappa
    # H0: mean(kappa_null) == kappa
    # We compute test statistic for the difference from kappa
    p_value = scipy.stats.ttest_1samp(
        kappa_null, kappa, alternative="two-sided" if two_sided else "larger"
    ).pvalue  # type: ignore

    return float(p_value)


def _safe_nanmean(x):
    """Helper to compute nanmean safely."""
    return np.nanmean(x) if len(x) > 0 and not np.all(np.isnan(x)) else np.nan


def _apply_correction_to_results(
    pvalues: Collection[float], alpha: float = 0.05
) -> NDArray:
    """
    Apply multiple hypothesis correction to a list of p-values.
    Returns corrected p-values in the same order.

    NaN p-values are excluded from correction (because FDR procedures
    cannot operate on undefined hypotheses). They are restored to NaN
    in their original positions after correction.
    """
    pvals = np.array(pvalues, dtype=float)

    if np.any((pvals[~np.isnan(pvals)] < 0) | (pvals[~np.isnan(pvals)] > 1)):
        raise ValueError("Invalid pvalues given for correction.")

    return _apply_correction(pvals, alpha)


def _apply_correction(pvalues: NDArray, alpha: float) -> NDArray:
    """
    FDR correction that handles NaNs safely.

    Steps:
    1. Identify valid (non-NaN) p-values.
    2. Apply FDR-BH only to valid p-values.
    3. Restore NaN positions to the output.
    """
    pvals = np.array(pvalues, dtype=float)
    valid_mask = ~np.isnan(pvals)
    valid_pvals = pvals[valid_mask]

    # If no valid values exist, return array of NaN
    if len(valid_pvals) == 0:
        warnings.warn(
            "All p-values are NaN; skipping multiple-testing correction.",
            RuntimeWarning,
        )
        return pvals  # all NaN array

    # Perform FDR correction on valid p-values
    corrected_valid = statsmodels.stats.multitest.multipletests(
        valid_pvals,
        alpha=alpha,
        method="fdr_bh",
        is_sorted=False,
        returnsorted=False,
    )[1]

    # Create output array and restore NaNs
    corrected = np.full_like(pvals, np.nan)
    corrected[valid_mask] = corrected_valid

    return corrected


def _to_hist(scores: Collection[float], bins: int) -> NDArray:
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

    counts, _ = np.histogram(a=scores_array, bins=bins, density=True)
    return counts


def _unique(x: Collection[Any]) -> list[Any]:
    # preserve first-seen order
    return list(dict.fromkeys(x))


def _is_not_none(x: Any) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))
