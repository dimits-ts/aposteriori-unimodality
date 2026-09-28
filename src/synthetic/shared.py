"""
Shared constants, simulation, and method functions for synthetic experiments.

Imported by metric_comparison.py and metric_comparison_multigroup.py.
"""

import zlib

import apunim
import krippendorff
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture


# ---------------------------------------------------------------- constants

N_LEVELS = 5
SIGMA = 1.8
ALPHA = 0.05

DELTAS = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
]

# simulation labels
SIMULATION_UNIDIRECTIONAL = "unidirectional"
SIMULATION_BIDIRECTIONAL = "bidirectional"

# Line styles keyed by simulation label.
SIMULATION_LINESTYLE = {
    SIMULATION_UNIDIRECTIONAL: "--",
    SIMULATION_BIDIRECTIONAL: "-",
}

# Human-readable legend labels for simulation types.
SIMULATION_LEGEND_LABEL = {
    SIMULATION_UNIDIRECTIONAL: "Unidirectional",
    SIMULATION_BIDIRECTIONAL: "Bidirectional",
}

METHOD_ORDER = [
    "apunim",
    "Krippendorff delta-alpha",
    "aposteriori unimodality (2024)",
    "chi-squared (Akhtar et al. 2019)",
    "GMM clustering (Checco/Mignemi)",
]

LEGEND_LABEL = {
    "Krippendorff delta-alpha": r"Krippendorff $\Delta\alpha$",
    "aposteriori unimodality (2024)": "aposteriori unim.",
    "chi-squared (Akhtar et al. 2019)": r"$\chi^2$",
    "GMM clustering (Checco/Mignemi)": "GMM",
}


# --------------------------------------------------------------- generators


def _groups(n_ann, minority_frac):
    """Two groups: minority 'B' at minority_frac, majority 'A' for the rest."""
    n_min = max(1, int(round(minority_frac * n_ann)))
    return np.array(["B"] * n_min + ["A"] * (n_ann - n_min))


def _groups_multigroup(n_ann, k, minority_frac):
    """
    Build a group label array for k groups.

    The polarizing group 'G0' has size minority_frac * n_ann.
    The remaining k-1 non-polarizing groups ('G1'..'Gk-1') share the
    rest of the annotators as equally as possible.
    """
    n_pol = max(1, int(round(minority_frac * n_ann)))
    n_rest = n_ann - n_pol

    # Distribute remaining annotators across k-1 non-polarizing groups
    base, extra = divmod(n_rest, k - 1)
    group_sizes = [base + (1 if i < extra else 0) for i in range(k - 1)]

    labels = ["G0"] * n_pol
    for i, size in enumerate(group_sizes):
        labels += [f"G{i + 1}"] * size

    return np.array(labels)


def simulate(
    n_items,
    n_ann,
    delta,
    minority_frac,
    rng,
    unidirectional: bool,
    sigma=SIGMA,
):
    """
    Generate annotations with a comment-specific group effect (two groups).

    Model:

        y_ij = b_j + g_i * p_j + epsilon_ij

    where:

        b_j:
            Baseline rating for item j, drawn from Uniform(2.5, 3.5).

        g_i:
            Group membership: -1/2 for group A, +1/2 for group B.

        p_j:
            Item-specific polarization effect.
            unidirectional=True:  p_j ~ Uniform(0, delta)
            unidirectional=False: p_j ~ Uniform(-delta, delta)

        epsilon_ij:
            Independent annotation noise ~ Normal(0, sigma).
    """
    groups = _groups(n_ann, minority_frac)
    base = rng.uniform(2.5, 3.5, size=n_items)

    if delta == 0:
        polarization = np.zeros(n_items)
    elif unidirectional:
        polarization = rng.uniform(0, delta, size=n_items)
    else:
        polarization = rng.uniform(-delta, delta, size=n_items)

    group_code = np.where(groups == "B", 0.5, -0.5)
    shift = group_code[:, None] * polarization[None, :]
    vals = rng.normal(base[None, :] + shift, sigma)
    vals = np.clip(np.round(vals), 1, N_LEVELS)

    return vals, groups


def simulate_multigroup(
    n_items,
    n_ann,
    k,
    delta,
    minority_frac,
    rng,
    unidirectional: bool,
    sigma=SIGMA,
):
    """
    Generate annotations with k groups, only one of which is polarizing.

    Model:

        y_ij = b_j + mu_{g_i, j} + epsilon_ij

    where:

        b_j:
            Baseline rating for item j, drawn from Uniform(2.5, 3.5).

        mu_{g, j}:
            Group- and item-specific offset (Option B).
            For the polarizing group G0:
                unidirectional:  mu_{G0, j} ~ Uniform(0, delta)
                bidirectional:   mu_{G0, j} ~ Uniform(-delta, delta)
            For all other groups G1..Gk-1:
                mu_{g, j} = 0

        epsilon_ij:
            Independent annotation noise ~ Normal(0, sigma).
    """
    groups = _groups_multigroup(n_ann, k, minority_frac)
    base = rng.uniform(2.5, 3.5, size=n_items)

    # Build per-annotator shift matrix (n_ann x n_items)
    shift = np.zeros((n_ann, n_items))

    pol_mask = groups == "G0"

    if delta > 0:
        if unidirectional:
            pol_effect = rng.uniform(0, delta, size=n_items)
        else:
            pol_effect = rng.uniform(-delta, delta, size=n_items)

        shift[pol_mask] = pol_effect[None, :]

    vals = rng.normal(base[None, :] + shift, sigma)
    vals = np.clip(np.round(vals), 1, N_LEVELS)

    return vals, groups


# ------------------------------------------------------------------- seeding


def _seed(*args):
    """
    Deterministic seed across processes and runs.
    hash() is salted per process, so use crc32 instead.
    """
    key = "|".join(str(a) for a in args).encode()
    return zlib.crc32(key)


# ------------------------------------------------------------------- methods


def _long(matrix, groups):
    n_ann, n_items = matrix.shape
    return (
        matrix.T.ravel(),
        np.tile(groups, n_items),
        np.repeat(np.arange(n_items), n_ann),
    )


def method_apunim(matrix, groups, seed):
    ann, fac, com = _long(matrix, groups)

    try:
        res = apunim.aposteriori_unimodality(
            ann,
            fac,
            com,
            num_bins=N_LEVELS,
            iterations=100,
            seed=seed,
        )
    except ValueError:
        return 0.0, 1.0

    if not res:
        return 0.0, 1.0

    best = min(res.items(), key=lambda kv: kv[1].pvalue)
    return (
        best[1].apunim,
        min(1.0, best[1].pvalue * len(res)),
    )


def _alpha(matrix):
    return (
        np.nan
        if matrix.shape[0] < 2
        else krippendorff.alpha(
            reliability_data=matrix,
            level_of_measurement="ordinal",
        )
    )


def _delta_alpha(matrix, groups):
    overall = _alpha(matrix)
    within = [
        alpha
        for alpha in (
            _alpha(matrix[groups == group]) for group in np.unique(groups)
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


def _run_methods(matrix, groups, rep, apunim_label):
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
