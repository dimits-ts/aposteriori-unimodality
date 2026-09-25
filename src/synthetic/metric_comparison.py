"""
Comparing apunim and agreement/statistical attribution methods.

Goal: To test whether different methods can correctly detect a polarization
signal (a known group effect) embedded in synthetic data.

Both simulations share the same generative model:

    y_ij = b_j + g_i * p_j + epsilon_ij

and differ only in how p_j is sampled:

    Unidirectional (solid lines):
        p_j ~ Uniform(0, delta)

        The group effect is always positive: Group B always rates higher
        than Group A, though by a varying amount across items.

    Bidirectional (dashed lines):
        p_j ~ Uniform(-delta, delta)

        The direction of the group effect varies across items, so groups
        can systematically disagree in opposite directions on different
        items. Positive and negative effects can cancel each other out
        in the pooled distribution.

All methods receive identical synthetic annotations for a given simulation,
where the strength of polarization is controlled by delta.
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
    simulate,
)


# (annotators/item, minority share)
CONDITIONS = [
    (80, 0.5),
    (80, 0.2),
    (20, 0.5),
    (20, 0.2),
    (6, 0.5),
]

PLOT_NUM_COLS = 3


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
        row["minority"] = float(row["minority"])
        row["detected"] = int(row["detected"])
        row["stat"] = float(row["stat"])
        # `simulation` and `method` are already strings.

    methods = [
        method for method in dict.fromkeys(row["method"] for row in rows)
    ]

    plot(rows, methods, graph_output_path)


# ------------------------------------------------------------------- runner


def _one(job, n_items):
    """
    Run both simulations for a single job and emit rows for each.

    Each row carries a `simulation` column so that the plotting
    code can separate the two experiments visually.
    """
    n_ann, minority, delta, rep = job

    seed = _seed(n_ann, minority, delta, rep)

    # Use child RNGs derived from the same seed so that the two
    # simulations are independent but still deterministic.
    rng_unidirectional = np.random.default_rng(seed ^ 0xABCD1234)
    rng_bidirectional = np.random.default_rng(seed ^ 0xDCBA4321)

    matrix_unidirectional, groups_unidirectional = simulate(
        n_items,
        n_ann,
        delta,
        minority,
        rng_unidirectional,
        unidirectional=True,
    )
    matrix_bidirectional, groups_bidirectional = simulate(
        n_items,
        n_ann,
        delta,
        minority,
        rng_bidirectional,
        unidirectional=False,
    )

    method_rows = []

    for matrix, groups, sim_label, apunim_label in [
        (
            matrix_unidirectional,
            groups_unidirectional,
            SIMULATION_UNIDIRECTIONAL,
            "apunim",
        ),
        (
            matrix_bidirectional,
            groups_bidirectional,
            SIMULATION_BIDIRECTIONAL,
            "apunim",
        ),
    ]:
        for method, stat, pvalue in _run_methods(
            matrix, groups, rep, apunim_label
        ):
            method_rows.append(
                (
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
        (n_ann, minority, delta, rep)
        for n_ann, minority in CONDITIONS
        for delta in DELTAS
        for rep in range(n_reps)
    ]

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
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

        worker = functools.partial(_one, n_items=n_items)

        with ProcessPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(worker, jobs, chunksize=4):
                writer.writerows(result)

    print(f"wrote {out_csv} ({len(jobs)} jobs × 2 simulations)")


# ------------------------------------------------------------------ figure


def _condition_title(n_ann, minority):
    return f"{n_ann} ann/item, {int(round(minority * 100))}\\% minority"


def plot(rows, methods, out_path):
    """
    Plot detection rates for both simulations in a shared panel grid.

    Visual encoding
    ---------------
    Color  + marker  ->  method
    Line style       ->  simulation type (solid = unidirectional, dashed = bidirectional)
    """
    methods = [m for m in METHOD_ORDER if m in methods] + [
        m for m in methods if m not in METHOD_ORDER
    ]

    df = pd.DataFrame(rows)
    colors = graphs.COLORBLIND_PALETTE
    n_rows = math.ceil(len(CONDITIONS) / PLOT_NUM_COLS)

    fig, axes = plt.subplots(
        n_rows,
        PLOT_NUM_COLS,
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, (n_ann, minority) in zip(axes, CONDITIONS):
        condition_df = df[
            (df["n_ann"] == n_ann)
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
        ax.set_title(_condition_title(n_ann, minority))
        ax.set_ylim(-0.04, 1.08)

    for ax in axes[len(CONDITIONS) :]:
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
        loc="lower right",
        title="Method",
    )

    fig.suptitle(
        "Apunim vs. prior approaches on polarization subgroup attribution"
    )
    fig.supylabel("Detection rate")
    fig.supxlabel(r"Maximum group effect size $\delta$")

    graphs.save_plot(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compare apunim against agreement-based and other statistical "
            "attribution baselines on synthetic data with comment-specific "
            "group polarization. Both unidirectional and bidirectional "
            "simulation models are run and displayed in a single plot, "
            "distinguished by line style."
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
