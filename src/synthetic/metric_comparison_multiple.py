"""
Multi-group ablation for apunim and comparison methods.

Goal: Test whether methods can detect polarization when only one of k groups
is responsible for it, and the rest behave identically.

Generative model:

    y_ij = b_j + mu_{g_i, j} + epsilon_ij

where:

    b_j:
        Baseline rating for item j, drawn from Uniform(2.5, 3.5).

    mu_{g, j}:
        Group- and item-specific offset (Option B).
        For the single polarizing group G0:
            unidirectional:  mu_{G0, j} ~ Uniform(0, delta)
            bidirectional:   mu_{G0, j} ~ Uniform(-delta, delta)
        For all non-polarizing groups G1..Gk-1:
            mu_{g, j} = 0

    epsilon_ij:
        Independent annotation noise ~ Normal(0, sigma).

All k group labels are passed to each method unchanged; no binary collapse
is performed. The minimum of 3 annotators per group (required by apunim)
sets a hard lower bound of n >= 3k annotators per item.

Conditions: two per k value, one near the minimum n, one with a spread
minority share (polarizing group is a smaller fraction than 1/k).

k=3  (min n=9):   (9,  1/3),  (30, 1/5)
k=5  (min n=15):  (15, 1/5),  (50, 1/8)
k=10 (min n=30):  (30, 1/10), (100, 1/15)
"""

import argparse
import csv
import math
import functools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from ..lib import graphs
from ..lib.util import skip_if_exists
from .shared import (
    ALPHA,
    DELTAS,
    LEGEND_LABEL,
    METHOD_ORDER,
    SIMULATION_BIDIRECTIONAL,
    SIMULATION_LINESTYLE,
    SIMULATION_UNIDIRECTIONAL,
    _run_methods,
    _seed,
    simulate_multigroup,
)


# ---------------------------------------------------------------- conditions

# CONDITIONS_MULTIGROUP: dict mapping k -> list of (n_ann, minority_frac)
#
# minority_frac is the share of annotators belonging to the single
# polarizing group G0.  The remaining k-1 groups share the rest equally.
#
# Constraint: n_ann * minority_frac >= 3  (apunim needs >= 3 per group)
#             n_ann * (1 - minority_frac) / (k-1) >= 3

CONDITIONS_MULTIGROUP = {
    3: [(9, 1 / 3), (30, 1 / 5)],
    5: [(15, 1 / 5), (50, 1 / 8)],
    10: [(30, 1 / 10), (100, 1 / 15)],
}

PLOT_NUM_COLS = 2


# --------------------------------------------------------------------- main


def main(
    cache_path: Path,
    graph_output_path: Path,
    n_items: int,
    n_reps: int,
    workers: int,
) -> None:
    graphs.graph_setup()

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if skip_if_exists(cache_path):
        print(f"loading cached results from {cache_path}")
    else:
        run(cache_path, n_items, n_reps, workers)

    with open(cache_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    for row in rows:
        row["delta"] = float(row["delta"])
        row["n_ann"] = int(row["n_ann"])
        row["k"] = int(row["k"])
        row["minority"] = float(row["minority"])
        row["detected"] = int(row["detected"])
        row["stat"] = float(row["stat"])
        # `simulation` and `method` are already strings.

    methods = [
        method for method in dict.fromkeys(row["method"] for row in rows)
    ]

    plot(rows, methods, graph_output_path)


# ------------------------------------------------------------------- runner


def _one_multigroup(job, n_items):
    """
    Run both simulations for a single (k, n_ann, minority, delta, rep) job.

    Each row carries `k` and `simulation` columns so the plotting code
    can facet by number of groups and separate simulation types.
    """
    k, n_ann, minority, delta, rep = job

    seed = _seed(k, n_ann, minority, delta, rep)

    rng_uni = np.random.default_rng(seed ^ 0xABCD1234)
    rng_bi = np.random.default_rng(seed ^ 0xDCBA4321)

    matrix_uni, groups_uni = simulate_multigroup(
        n_items, n_ann, k, delta, minority, rng_uni, unidirectional=True
    )
    matrix_bi, groups_bi = simulate_multigroup(
        n_items, n_ann, k, delta, minority, rng_bi, unidirectional=False
    )

    method_rows = []

    for matrix, groups, sim_label in [
        (matrix_uni, groups_uni, SIMULATION_UNIDIRECTIONAL),
        (matrix_bi, groups_bi, SIMULATION_BIDIRECTIONAL),
    ]:
        for method, stat, pvalue in _run_methods(
            matrix, groups, rep, "apunim"
        ):
            method_rows.append(
                (
                    k,
                    n_ann,
                    minority,
                    delta,
                    rep,
                    sim_label,
                    method,
                    stat,
                    pvalue,
                    int(pvalue < ALPHA) if pvalue == pvalue else 0,
                )
            )

    return method_rows


def run(out_csv: Path, n_items: int, n_reps: int, workers: int) -> None:
    jobs = [
        (k, n_ann, minority, delta, rep)
        for k, conditions in CONDITIONS_MULTIGROUP.items()
        for n_ann, minority in conditions
        for delta in DELTAS
        for rep in range(n_reps)
    ]

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "k",
                "n_ann",
                "minority",
                "delta",
                "rep",
                "simulation",  # "unidirectional" or "bidirectional"
                "method",
                "stat",
                "pvalue",
                "detected",
            ]
        )

        worker = functools.partial(_one_multigroup, n_items=n_items)

        with ProcessPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(worker, jobs, chunksize=4):
                writer.writerows(result)

    print(f"wrote {out_csv} ({len(jobs)} jobs × 2 simulations)")


# ------------------------------------------------------------------ figure


def _condition_title(k, n_ann, minority):
    return (
        f"k={k} groups, {n_ann} ann/item, "
        f"{int(round(minority * 100))}\\% polarizing"
    )


def plot(rows, methods, out_path):
    """
    Plot detection rates faceted by k, one panel per (k, condition).
    """
    methods = [m for m in METHOD_ORDER if m in methods] + [
        m for m in methods if m not in METHOD_ORDER
    ]

    df = pd.DataFrame(rows)
    colors = graphs.COLORBLIND_PALETTE

    # Build ordered list of (k, n_ann, minority) panels
    panels = [
        (k, n_ann, minority)
        for k, conditions in CONDITIONS_MULTIGROUP.items()
        for n_ann, minority in conditions
    ]

    n_rows = math.ceil(len(panels) / PLOT_NUM_COLS)

    fig, axes = plt.subplots(
        n_rows,
        PLOT_NUM_COLS,
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, (k, n_ann, minority) in zip(axes, panels):
        condition_df = df[
            (df["k"] == k)
            & (df["n_ann"] == n_ann)
            & (df["minority"] == minority)
            & (df["method"].isin(methods))
        ]

        for i, method in enumerate(methods):
            method_df = condition_df[condition_df["method"] == method]

            if len(method_df) == 0:
                continue

            for simulation in [
                SIMULATION_UNIDIRECTIONAL,
                SIMULATION_BIDIRECTIONAL,
            ]:
                sim_df = method_df[method_df["simulation"] == simulation]
                opacity = (
                    0.3 if simulation == SIMULATION_UNIDIRECTIONAL else 1
                )

                if len(sim_df) == 0:
                    continue

                sns.lineplot(
                    data=sim_df,
                    x="delta",
                    y="detected",
                    estimator="mean",
                    errorbar="se",
                    marker=graphs.MARKERS[i % len(graphs.MARKERS)],
                    color=colors[i % len(colors)],
                    linestyle=SIMULATION_LINESTYLE[simulation],
                    lw=1.4,
                    ms=3.8,
                    err_style="bars",
                    err_kws={"capsize": 3},
                    ax=ax,
                    legend=False,
                    alpha=opacity,
                )

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(_condition_title(k, n_ann, minority))
        ax.set_ylim(-0.04, 1.08)

    for ax in axes[len(panels) :]:
        ax.set_visible(False)

    method_handles = [
        plt.Line2D(
            [0],
            [0],
            color=colors[i % len(colors)],
            marker=graphs.MARKERS[i % len(graphs.MARKERS)],
            linestyle="-",
            lw=1.4,
            ms=4.5,
            label=LEGEND_LABEL.get(m, m),
        )
        for i, m in enumerate(methods)
    ]

    fig.legend(
        handles=method_handles,
        loc="lower center",
        title="Method",
        ncols=len(method_handles)
    )

    fig.suptitle(
        "Multi-group ablation: detection when one of k groups drives polarization"
    )
    fig.supylabel("Detection rate")
    fig.supxlabel(r"Maximum group effect size $\delta$")

    graphs.save_plot(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Multi-group ablation: compare apunim and other methods when "
            "only one of k groups causes polarization. Runs unidirectional "
            "and bidirectional simulations for k in {3, 5, 10}."
        )
    )

    parser.add_argument(
        "--cache-path",
        required=True,
        help="Path for the output CSV cache.",
    )

    parser.add_argument(
        "--graph-output-path",
        required=True,
        help="Path for the output graph.",
    )

    parser.add_argument("--n-items", type=int, default=200)
    parser.add_argument("--n-reps", type=int, default=40)
    parser.add_argument("--workers", type=int, default=7)

    args = parser.parse_args()

    main(
        Path(args.cache_path),
        Path(args.graph_output_path),
        args.n_items,
        args.n_reps,
        args.workers,
    )
