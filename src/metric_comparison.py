"""
Comparing apunim and agreement/statistical attribution methods.

Goal: To test whether different methods can correctly detect a polarization
signal (a known group effect) embedded in synthetic data.

The simulation uses a comment-specific group effect:

    y_ij = b_j + g_i * p_j + epsilon_ij

where:
    b_j        = baseline position/difficulty of comment j
    g_i        = group membership (-1/2 or +1/2)
    p_j        = comment-specific polarization effect
    epsilon_ij = annotation noise

The sign of p_j determines which group rates a comment higher, while
|p_j| determines the strength of the disagreement for that comment.

The maximum magnitude of p_j is controlled by delta.

All methods receive identical synthetic annotations where the strength
of polarization is controlled by the simulation parameters.
"""

import argparse
import csv
import math
import zlib
import functools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import apunim
import krippendorff
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

import tasks.graphs


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

# (annotators/item, minority share)
CONDITIONS = [
    (80, 0.5),
    (80, 0.2),
    (20, 0.5),
    (20, 0.2),
    (6, 0.5),
]

PLOT_NUM_COLS = 3


# --------------------------------------------------------------- generators


def _groups(n_ann, minority_frac):
    n_min = max(1, int(round(minority_frac * n_ann)))

    return np.array(["B"] * n_min + ["A"] * (n_ann - n_min))


def simulate_simple(n_items, n_ann, delta, minority_frac, rng, sigma=SIGMA):
    """Group-driven polarization: the two groups sit `delta` apart."""
    groups = _groups(n_ann, minority_frac)

    base = rng.uniform(2.5, 3.5, size=n_items)

    shift = np.where(
        groups == "B",
        delta / 2,
        -delta / 2,
    )

    vals = rng.normal(
        base[None, :] + shift[:, None],
        sigma,
    )

    return (
        np.clip(
            np.round(vals),
            1,
            N_LEVELS,
        ),
        groups,
    )


def simulate(
    n_items,
    n_ann,
    delta,
    minority_frac,
    rng,
    sigma=SIGMA,
):
    """
    Generate annotations with a comment-specific group effect.

    Model:

        y_ij = b_j + g_i * p_j + epsilon_ij

    where:

        b_j:
            Baseline position/difficulty of comment j.

        g_i:
            Group membership:
                -1/2 for group A
                +1/2 for group B.

        p_j:
            Comment-specific polarization effect sampled uniformly
            from [-delta, delta].

        epsilon_ij:
            Independent annotation noise.

    Because the group codes are +/- 1/2, p_j represents approximately
    the difference between the two group means for comment j.

    Importantly, the direction of polarization can change from comment
    to comment. A comment may be favored by group A while another is
    favored by group B.
    """

    groups = _groups(
        n_ann,
        minority_frac,
    )

    # -----------------------------------------------------------
    # Baseline comment position
    # -----------------------------------------------------------

    base = rng.uniform(
        2.5,
        3.5,
        size=n_items,
    )

    # -----------------------------------------------------------
    # Comment-specific polarization
    # -----------------------------------------------------------
    #
    # delta = 0:
    #     no polarization anywhere.
    #
    # delta > 0:
    #     each comment receives its own polarization effect.
    #
    # The sign determines which group rates the comment higher.
    # The magnitude determines how strongly they disagree.
    #

    if delta == 0:
        polarization = np.zeros(n_items)
    else:
        polarization = rng.uniform(
            -delta,
            delta,
            size=n_items,
        )

    # -----------------------------------------------------------
    # Group membership
    # -----------------------------------------------------------

    group_code = np.where(
        groups == "B",
        0.5,
        -0.5,
    )

    # -----------------------------------------------------------
    # Group × comment interaction
    # -----------------------------------------------------------

    shift = group_code[:, None] * polarization[None, :]

    # -----------------------------------------------------------
    # Generate noisy annotations
    # -----------------------------------------------------------

    vals = rng.normal(
        base[None, :] + shift,
        sigma,
    )

    # Convert continuous ratings to the observed ordinal scale.
    vals = np.clip(
        np.round(vals),
        1,
        N_LEVELS,
    )

    return vals, groups


# ------------------------------------------------------------------ methods


def _long(matrix, groups):
    n_ann, n_items = matrix.shape

    return (
        matrix.T.ravel(),
        np.tile(groups, n_items),
        np.repeat(
            np.arange(n_items),
            n_ann,
        ),
    )


def method_apunim(matrix, groups, seed):
    ann, fac, com = _long(
        matrix,
        groups,
    )

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
        # No comment passes the eligibility filter.
        return 0.0, 1.0

    if not res:
        return 0.0, 1.0

    # Bonferroni correction over levels.
    best = min(
        res.items(),
        key=lambda kv: kv[1].pvalue,
    )

    return (
        best[1].apunim,
        min(
            1.0,
            best[1].pvalue * len(res),
        ),
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
        (
            float(np.mean(within)) - overall,
            overall,
        )
        if within
        else (np.nan, np.nan)
    )


def method_delta_alpha(
    matrix,
    groups,
    seed,
    n_perm=200,
):
    obs, overall = _delta_alpha(
        matrix,
        groups,
    )

    if np.isnan(obs):
        return np.nan, 1.0, overall

    rng = np.random.default_rng(seed)

    null = np.array(
        [
            _delta_alpha(
                matrix,
                rng.permutation(groups),
            )[0]
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

        if (
            apunim.dfu(
                col,
                bins=N_LEVELS,
            )
            <= 0
        ):
            continue

        eligible += 1

        if all(
            apunim.dfu(
                col[groups == group],
                bins=N_LEVELS,
            )
            <= 0
            for group in levels
        ):
            hits += 1

    return hits / eligible if eligible else np.nan


def method_original_au(
    matrix,
    groups,
    seed,
    n_perm=100,
):
    obs = _frac_explained(
        matrix,
        groups,
    )

    if np.isnan(obs):
        return np.nan, 1.0

    rng = np.random.default_rng(seed)

    null = np.array(
        [
            _frac_explained(
                matrix,
                rng.permutation(groups),
            )
            for _ in range(n_perm)
        ]
    )

    null = null[~np.isnan(null)]

    return (
        obs,
        (1 + np.sum(null >= obs)) / (1 + len(null)),
    )


def method_chi2_variance(
    matrix,
    groups,
    seed,
):
    """
    Chi-squared test of independence between group membership and
    pooled annotation level.

    Note that this tests a global association between group and
    annotation level. Under the new comment-specific polarization
    simulation, positive and negative comment effects can cancel
    each other out in the pooled distribution.
    """

    levels = np.arange(
        1,
        N_LEVELS + 1,
    )

    unique_groups = np.unique(groups)

    table = np.array(
        [
            [(matrix[groups == g] == lvl).sum() for lvl in levels]
            for g in unique_groups
        ]
    )

    if table.shape[0] < 2 or np.any(table.sum(axis=1) == 0):
        return np.nan, 1.0

    # Drop levels with no observations.
    table = table[
        :,
        table.sum(axis=0) > 0,
    ]

    if table.shape[1] < 2:
        return np.nan, 1.0

    try:
        chi2_stat, pvalue, _, _ = chi2_contingency(table)

    except ValueError:
        return np.nan, 1.0

    return chi2_stat, pvalue


def method_mixture_clustering(
    matrix,
    groups,
    seed,
    n_perm=200,
):
    """
    Fit a 2-component Gaussian mixture to annotator mean ratings.

    This is an unsupervised clustering baseline. Under the new
    comment-specific polarization model, the direction of the group
    effect can vary across comments, so annotator means may no longer
    cleanly separate into two groups.
    """

    unique_groups = np.unique(groups)

    if len(unique_groups) < 2 or matrix.shape[0] < 4:
        return np.nan, 1.0

    ann_means = matrix.mean(axis=1).reshape(-1, 1)

    try:
        gmm = GaussianMixture(
            n_components=2,
            random_state=seed,
            n_init=3,
        )

        cluster_labels = gmm.fit_predict(ann_means)

    except ValueError:
        return np.nan, 1.0

    obs_ari = adjusted_rand_score(
        groups,
        cluster_labels,
    )

    rng = np.random.default_rng(seed)

    null = np.array(
        [
            adjusted_rand_score(
                rng.permutation(groups),
                cluster_labels,
            )
            for _ in range(n_perm)
        ]
    )

    return (
        obs_ari,
        (1 + np.sum(null >= obs_ari)) / (1 + n_perm),
    )


# ------------------------------------------------------------------- runner


def _seed(
    n_ann,
    minority,
    delta,
    rep,
):
    """
    Deterministic across processes and runs.
    hash() is salted per process, so use crc32 instead.
    """

    key = (f"{n_ann}|{minority}|" f"{delta}|{rep}").encode()

    return zlib.crc32(key)


def _one(job, n_items, label, use_simple_simulate: bool):
    n_ann, minority, delta, rep = job

    rng = np.random.default_rng(
        _seed(
            n_ann,
            minority,
            delta,
            rep,
        )
    )

    simulation_func = simulate_simple if use_simple_simulate else simulate

    matrix, groups = simulation_func(
        n_items,
        n_ann,
        delta,
        minority,
        rng,
    )

    rows = [
        (
            label,
            *method_apunim(
                matrix,
                groups,
                rep,
            ),
        )
    ]

    stat, pvalue, overall_alpha = method_delta_alpha(
        matrix,
        groups,
        rep,
    )

    rows += [
        (
            "Krippendorff delta-alpha",
            stat,
            pvalue,
        ),
        (
            "aposteriori unimodality (2024)",
            *method_original_au(
                matrix,
                groups,
                rep,
            ),
        ),
        (
            "chi-squared (Akhtar et al. 2019)",
            *method_chi2_variance(
                matrix,
                groups,
                rep,
            ),
        ),
        (
            "GMM clustering (Checco/Mignemi)",
            *method_mixture_clustering(
                matrix,
                groups,
                rep,
            ),
        ),
    ]

    return [
        (
            n_ann,
            minority,
            delta,
            rep,
            method,
            stat,
            pvalue,
            int(pvalue < ALPHA) if pvalue == pvalue else 0,
        )
        for method, stat, pvalue in rows
    ]


def run(
    out_csv: Path,
    n_items: int,
    n_reps: int,
    workers: int,
    use_simple_simulate: bool,
) -> None:

    jobs = [
        (
            n_ann,
            minority,
            delta,
            rep,
        )
        for n_ann, minority in CONDITIONS
        for delta in DELTAS
        for rep in range(n_reps)
    ]

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        out_csv,
        "w",
        newline="",
    ) as fh:

        writer = csv.writer(fh)

        writer.writerow(
            [
                "n_ann",
                "minority",
                "delta",
                "rep",
                "method",
                "stat",
                "pvalue",
                "detected",
            ]
        )

        worker = functools.partial(
            _one,
            n_items=n_items,
            label="apunim",
            use_simple_simulate=use_simple_simulate,
        )

        with ProcessPoolExecutor(max_workers=workers) as executor:

            for result in executor.map(
                worker,
                jobs,
                chunksize=4,
            ):
                writer.writerows(result)

    print(f"wrote {out_csv} " f"({len(jobs)} runs)")


# ------------------------------------------------------------------ figure


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


def _condition_title(
    n_ann,
    minority
):
    return f"{n_ann} ann/item, " f"{int(round(minority * 100))}\\% minority"


def plot(
    rows,
    methods,
    out_path
):
    methods = [m for m in METHOD_ORDER if m in methods] + [
        m for m in methods if m not in METHOD_ORDER
    ]

    df = pd.DataFrame(rows)

    colors = tasks.graphs.COLORBLIND_PALETTE

    n_rows = math.ceil(len(CONDITIONS) / PLOT_NUM_COLS)

    fig, axes = plt.subplots(
        n_rows,
        PLOT_NUM_COLS,
        figsize=(
            PLOT_NUM_COLS * 5.0,
            n_rows * 2.8,
        ),
        sharey=True,
        squeeze=False,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.86,
        bottom=0.18,
        wspace=0.25,
        hspace=0.35,
    )

    axes = axes.ravel()

    for ax, (
        n_ann,
        minority,
    ) in zip(
        axes,
        CONDITIONS,
    ):

        condition_df = df[
            (df["n_ann"] == n_ann)
            & (df["minority"] == minority)
            & (df["method"].isin(methods))
        ]

        for i, method in enumerate(methods):

            method_df = condition_df[condition_df["method"] == method]

            if method_df.empty:
                continue

            sns.lineplot(
                data=method_df,
                x="delta",
                y="detected",
                estimator="mean",
                errorbar="se",
                marker=tasks.graphs.MARKERS[i % len(tasks.graphs.MARKERS)],
                color=colors[i % len(colors)],
                label=LEGEND_LABEL.get(
                    method,
                    method,
                ),
                lw=1.4,
                ms=3.8,
                err_style="bars",
                err_kws={"capsize": 3},
                ax=ax,
                legend=False,
            )

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(
            _condition_title(
                n_ann,
                minority,
            )
        )
        ax.set_ylim(
            -0.04,
            1.08,
        )

    for ax in axes[len(CONDITIONS) :]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower right",
        title="Attribution Methodology",
    )

    fig.suptitle(
        "Apunim vs. prior approaches on polarization subgroup attribution"
    )

    fig.supylabel("Detection rate")

    fig.supxlabel(r"Maximum group effect size $\delta$")

    tasks.graphs.save_plot(out_path)


# --------------------------------------------------------------------- main


def main(
    cache_path: Path,
    graph_output_path: Path,
    n_items: int,
    n_reps: int,
    workers: int,
    use_simple_simulate: bool,
) -> None:
    if use_simple_simulate:
        print("Using simple simulation.")
    else:
        print("Using standard simulation.")

    tasks.graphs.graph_setup()

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if cache_path.exists():
        print(f"loading cached results " f"from {cache_path}")

    else:
        run(
            cache_path,
            n_items,
            n_reps,
            workers,
            use_simple_simulate=use_simple_simulate,
        )

    with open(
        cache_path,
        newline="",
    ) as fh:

        rows = list(csv.DictReader(fh))

    for row in rows:
        row["delta"] = float(row["delta"])

        row["n_ann"] = int(row["n_ann"])

        row["minority"] = float(row["minority"])

        row["detected"] = int(row["detected"])

        row["stat"] = float(row["stat"])

    methods = [
        method for method in dict.fromkeys(row["method"] for row in rows)
    ]

    plot(
        rows,
        methods,
        graph_output_path,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Compare apunim against "
            "agreement-based and other "
            "statistical attribution "
            "baselines on synthetic data "
            "with comment-specific "
            "group polarization."
        )
    )

    parser.add_argument(
        "--cache-path",
        required=True,
        help=("Directory used to cache " "calculation results as CSV."),
    )

    parser.add_argument(
        "--graph-output-path",
        required=True,
        help=("Directory for graphs."),
    )

    parser.add_argument(
        "--n-items",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--n-reps",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--simple-simulation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    args = parser.parse_args()

    main(
        Path(args.cache_path),
        Path(args.graph_output_path),
        args.n_items,
        args.n_reps,
        args.workers,
        args.simple_simulation,
    )
