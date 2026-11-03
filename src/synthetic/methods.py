"""
Systematic-disagreement detection methods compared in the experiments:
apunim, Krippendorff delta-alpha, the original aposteriori unimodality,
chi-squared and GMM clustering.

The method_* functions take a dense annotators x items matrix on the
synthetic rating scale (simulation.N_LEVELS). apunim_min_pvalue and
krippendorff_alpha are also used by real_data_comparison.py.
"""

import apunim
import krippendorff
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from ..lib import labels
from .simulation import N_LEVELS


# Significance level for counting a detection.
ALPHA = 0.05

METHOD_ORDER = [
    "apunim",
    "Krippendorff delta-alpha",
    "aposteriori unimodality (2024)",
    "chi-squared (Akhtar et al. 2019)",
    "GMM clustering (Checco/Mignemi)",
]

# Labels for matplotlib figures. Matplotlib's LaTeX run does not load the
# paper's acronym package, so \ac{...} cannot be used here.
LEGEND_LABEL = {
    "apunim": "apunim (ours)",
    "Krippendorff delta-alpha": r"Krippendorff $\Delta\alpha$",
    "aposteriori unimodality (2024)": labels.AU.short,
    "chi-squared (Akhtar et al. 2019)": r"$\chi^2$",
    "GMM clustering (Checco/Mignemi)": labels.GMM.short,
}

# Labels for LaTeX tables included in the paper, where \ac{...} works.
TABLE_LABEL = {
    **LEGEND_LABEL,
    "aposteriori unimodality (2024)": labels.AU.tex,
}


def _long(matrix, groups):
    n_ann, n_items = matrix.shape
    return (
        matrix.T.ravel(),
        np.tile(groups, n_items),
        np.repeat(np.arange(n_items), n_ann),
    )


def apunim_min_pvalue(annotations, groups, comments, num_bins, seed):
    """
    apunim aggregated to one (stat, pvalue) pair per SDB feature: the factor
    with the smallest p-value, Bonferroni-corrected over the factors that
    could be tested. Inputs are in long format (one entry per annotation).

    Shared by the synthetic experiments (method_apunim below) and the real
    data comparison (real_data_comparison.method_apunim).
    """
    try:
        res = apunim.aposteriori_unimodality(
            annotations,
            groups,
            comments,
            num_bins=num_bins,
            iterations=100,
            # raw p-values: the Bonferroni correction over the factors
            # below is the only correction (apunim's default alpha=0.05
            # would apply BH on top of it)
            alpha=None,
            seed=seed,
        )
    except ValueError:
        return 0.0, 1.0

    # factors apunim could not test have a NaN p-value; they are neither
    # candidates for the minimum nor counted in the correction
    res = {k: v for k, v in res.items() if not np.isnan(v.pvalue)}

    if not res:
        return 0.0, 1.0

    best = min(res.values(), key=lambda r: r.pvalue)
    return best.apunim, min(1.0, best.pvalue * len(res))


def method_apunim(matrix, groups, seed):
    return apunim_min_pvalue(
        *_long(matrix, groups), num_bins=N_LEVELS, seed=seed
    )


def krippendorff_alpha(matrix):
    return (
        np.nan
        if matrix.shape[0] < 2
        else krippendorff.alpha(
            reliability_data=matrix,
            level_of_measurement="ordinal",
        )
    )


def _delta_alpha(matrix, groups):
    overall = krippendorff_alpha(matrix)
    within = [
        alpha
        for alpha in (
            krippendorff_alpha(matrix[groups == group])
            for group in np.unique(groups)
        )
        if not np.isnan(alpha)
    ]
    return (
        (float(np.mean(within)) - overall, overall)
        if within
        else (np.nan, np.nan)
    )


def method_delta_alpha(matrix, groups, seed, n_perm=200):
    obs, overall = _delta_alpha(matrix, groups)

    if np.isnan(obs):
        return np.nan, 1.0, overall

    rng = np.random.default_rng(seed)
    null = np.array(
        [
            _delta_alpha(matrix, rng.permutation(groups))[0]
            for _ in range(n_perm)
        ]
    )

    return (
        obs,
        (1 + np.sum(null >= obs)) / (1 + n_perm),
        overall,
    )


def _frac_explained(matrix, groups):
    levels = np.unique(groups)
    hits = 0
    eligible = 0

    for c in range(matrix.shape[1]):
        col = matrix[:, c]

        if apunim.dfu(col, bins=N_LEVELS) <= 0:
            continue

        eligible += 1

        if all(
            apunim.dfu(col[groups == group], bins=N_LEVELS) <= 0
            for group in levels
        ):
            hits += 1

    return hits / eligible if eligible else np.nan


def method_original_au(matrix, groups, seed, n_perm=100):
    obs = _frac_explained(matrix, groups)

    if np.isnan(obs):
        return np.nan, 1.0

    rng = np.random.default_rng(seed)
    null = np.array(
        [
            _frac_explained(matrix, rng.permutation(groups))
            for _ in range(n_perm)
        ]
    )
    null = null[~np.isnan(null)]

    return (
        obs,
        (1 + np.sum(null >= obs)) / (1 + len(null)),
    )


def method_chi2_variance(matrix, groups, seed):
    """
    Chi-squared test of independence between group membership and
    pooled annotation level.

    Note that this tests a global association between group and
    annotation level. Under the bidirectional simulation, positive and
    negative comment effects can cancel each other out in the pooled
    distribution.
    """
    levels = np.arange(1, N_LEVELS + 1)
    unique_groups = np.unique(groups)

    table = np.array(
        [
            [(matrix[groups == g] == lvl).sum() for lvl in levels]
            for g in unique_groups
        ]
    )

    if table.shape[0] < 2 or np.any(table.sum(axis=1) == 0):
        return np.nan, 1.0

    table = table[:, table.sum(axis=0) > 0]

    if table.shape[1] < 2:
        return np.nan, 1.0

    try:
        chi2_stat, pvalue, _, _ = chi2_contingency(table)
    except ValueError:
        return np.nan, 1.0

    return chi2_stat, pvalue


def method_mixture_clustering(matrix, groups, seed, n_perm=200):
    """
    Fit a 2-component Gaussian mixture to annotator mean ratings.

    Under the bidirectional simulation, the direction of the group effect
    can vary across comments, so annotator means may no longer cleanly
    separate into two groups.
    """
    unique_groups = np.unique(groups)

    if len(unique_groups) < 2 or matrix.shape[0] < 4:
        return np.nan, 1.0

    ann_means = matrix.mean(axis=1).reshape(-1, 1)

    try:
        gmm = GaussianMixture(n_components=2, random_state=seed, n_init=3)
        cluster_labels = gmm.fit_predict(ann_means)
    except ValueError:
        return np.nan, 1.0

    obs_ari = adjusted_rand_score(groups, cluster_labels)

    rng = np.random.default_rng(seed)
    null = np.array(
        [
            adjusted_rand_score(rng.permutation(groups), cluster_labels)
            for _ in range(n_perm)
        ]
    )

    return (
        obs_ari,
        (1 + np.sum(null >= obs_ari)) / (1 + n_perm),
    )


def run_methods(matrix, groups, rep, apunim_label):
    """
    Run all methods on a single (matrix, groups) pair.

    Returns a list of (method_label, stat, pvalue) tuples.
    """
    rows = [
        (
            apunim_label,
            *method_apunim(matrix, groups, rep),
        )
    ]

    stat, pvalue, _ = method_delta_alpha(matrix, groups, rep)

    rows += [
        ("Krippendorff delta-alpha", stat, pvalue),
        (
            "aposteriori unimodality (2024)",
            *method_original_au(matrix, groups, rep),
        ),
        (
            "chi-squared (Akhtar et al. 2019)",
            *method_chi2_variance(matrix, groups, rep),
        ),
        (
            "GMM clustering (Checco/Mignemi)",
            *method_mixture_clustering(matrix, groups, rep),
        ),
    ]

    return rows