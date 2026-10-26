"""
Data-generating models for the synthetic experiments.

simulate() is the two-group model used by metric_comparison.py;
simulate_multigroup() is the k-group model used by
metric_comparison_multiple.py.
"""

import numpy as np


# Rating scale of the generated annotations: integers 1..N_LEVELS.
N_LEVELS = 5
SIGMA = 1.8

DELTAS = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
]

# Simulation labels, stored in the result caches.
SIMULATION_UNIDIRECTIONAL = "unidirectional"
SIMULATION_BIDIRECTIONAL = "bidirectional"


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
            Annotator-specific group loading, drawn once per annotator:
            g_i ~ Uniform(-1/2, 0) for group A,
            g_i ~ Uniform(0, 1/2) for group B.
            Groups always pull in opposite directions, but annotators
            within a group differ in how strongly.

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

    group_code = _group_codes(groups, rng)
    shift = group_code[:, None] * polarization[None, :]
    vals = rng.normal(base[None, :] + shift, sigma)
    vals = np.clip(np.round(vals), 1, N_LEVELS)

    return vals, groups


def _group_codes(groups, rng):
    """g_i ~ Uniform(-1/2, 0) for group A and Uniform(0, 1/2) for group B."""
    n_ann = len(groups)
    return np.where(
        groups == "B",
        rng.uniform(0, 0.5, size=n_ann),
        rng.uniform(-0.5, 0, size=n_ann),
    )


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

        y_ij = b_j + h_i * mu_{g_i, j} + epsilon_ij

    where:

        b_j:
            Baseline rating for item j, drawn from Uniform(2.5, 3.5).

        h_i:
            Annotator-specific strength, drawn once per annotator from
            Uniform(0, 1). Annotators in the polarizing group all shift in
            the same direction, but by differing amounts (mirrors g_i in
            simulate). Irrelevant for non-polarizing groups (mu = 0).

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

        strength = rng.uniform(0, 1, size=pol_mask.sum())
        shift[pol_mask] = strength[:, None] * pol_effect[None, :]

    vals = rng.normal(base[None, :] + shift, sigma)
    vals = np.clip(np.round(vals), 1, N_LEVELS)

    return vals, groups
