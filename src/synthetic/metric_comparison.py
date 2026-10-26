"""
Comparing apunim and agreement/statistical attribution methods.

Goal: To test whether different methods can correctly detect a polarization
signal (a known group effect) embedded in synthetic data.

Both simulations share the same generative model:

    y_ij = b_j + g_i * p_j + epsilon_ij

where g_i ~ Uniform(-1/2, 0) for Group A and g_i ~ Uniform(0, 1/2) for
Group B, drawn once per annotator. The simulations differ only in how p_j
is sampled:

    Unidirectional (dashed, faded lines):
        p_j ~ Uniform(0, delta)

        The group effect is always positive: Group B always rates higher
        than Group A, though by a varying amount across items.

    Bidirectional (solid lines):
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
import functools
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

from ..lib import graphs
from ..lib.util import skip_if_exists, stable_seed
from .plots import (
    DETECTION_RATE_LABEL,
    EFFECT_SIZE_LABEL,
    method_legend_handles,
    order_methods,
    panel_rows,
    plot_detection_panel,
)
from .methods import ALPHA, run_methods
from .simulation import (
    DELTAS,
    SIMULATION_BIDIRECTIONAL,
    SIMULATION_UNIDIRECTIONAL,
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

# Grid layout is derived from the conditions: one row per distinct
# annotators/item value, one column per distinct minority share
# (in order of first appearance). Cells with no condition (here:
# 6 ann/item, 20% minority) are left empty and host the legend.
ROW_VALUES = list(dict.fromkeys(n_ann for n_ann, _ in CONDITIONS))
COL_VALUES = list(dict.fromkeys(minority for _, minority in CONDITIONS))

# Gap (in points) between the column labels and the top row's subplot
# titles. The labels are anchored to the tallest top-row title, so they
# move up automatically for multi-line titles and stay level.
COL_LABEL_PAD = 6

# The shared rcParams default (14, 6) is sized for a single row of panels;
# a 3-row grid needs more height so titles, tick labels and margin labels
# have room at the global font sizes set in graphs.graph_setup().
FIGSIZE = (14, 9)

# Horizontal padding (inches) the constrained layout engine leaves around
# each axes and figure-level label.
LAYOUT_W_PAD = 0.15

# Direction cues drawn in the figure margins. Each arrow lives in its own
# thin gridspec track (left of the rows / above the columns) so the
# constrained layout engine reserves space for it like any other element.
# ARROW_TRACK is that track's size relative to one panel.
ROW_ARROW_LABELS = ("Utopic", "Realistic")  # top -> bottom
COL_ARROW_LABELS = ("Balanced", "Minority")  # left -> right
ARROW_TRACK = 0.04
ARROW_COLOR = "0.35"
ARROW_LABEL_GAP = 4  # points between an arrow end and its label


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

    plot(
        rows,
        methods,
        graph_output_path,
        panel_titles=[
            "Exceptional",
            "Competitive",
            "Competitive",
            "Struggle",
            "Collapse",
        ],
    )


# ------------------------------------------------------------------- runner


def _one(job, n_items):
    """
    Run both simulations for a single job and emit rows for each.

    Each row carries a `simulation` column so that the plotting
    code can separate the two experiments visually.
    """
    n_ann, minority, delta, rep = job

    seed = stable_seed(n_ann, minority, delta, rep)

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
        for method, stat, pvalue in run_methods(
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


def _bold(text):
    return "\n".join(rf"\textbf{{{line}}}" for line in text.split("\n"))


def _row_label(n_ann):
    return _bold(f"{n_ann} annotators\nper item")


def _col_label(minority):
    return _bold(
        f"{int(round(minority * 100))}\\%-"
        f"{int(round((1-minority) * 100))}\\% group sizes"
    )


def _direction_arrow(ax, start_label, end_label, vertical):
    """
    Draw `start_label ---> end_label` across the full length of `ax`.

    Horizontal arrows run left to right; vertical ones run top to bottom
    with labels rotated to read in the same direction as the arrow.
    """
    ax.axis("off")
    text_kws = dict(
        transform=ax.transAxes,
        color=ARROW_COLOR,
        fontsize=plt.rcParams["axes.labelsize"],
    )

    if vertical:
        start = ax.text(
            0.5,
            1,
            start_label,
            ha="center",
            va="top",
            rotation=90,
            **text_kws,
        )
        end = ax.text(
            0.5,
            0,
            end_label,
            ha="center",
            va="bottom",
            rotation=90,
            **text_kws,
        )
        tail, head = (0.5, 0.0), (0.5, 1.0)  # bottom of start, top of end
    else:
        start = ax.text(
            0, 0.5, start_label, ha="left", va="center", **text_kws
        )
        end = ax.text(1, 0.5, end_label, ha="right", va="center", **text_kws)
        tail, head = (1.0, 0.5), (0.0, 0.5)  # right of start, left of end

    # Arrow from the start label's edge to the end label's edge, so it
    # always fills the space between them whatever the figure size.
    ax.annotate(
        "",
        xy=head,
        xycoords=end,
        xytext=tail,
        textcoords=start,
        arrowprops=dict(
            arrowstyle="-|>",
            color=ARROW_COLOR,
            lw=1.4,
            mutation_scale=18,
            shrinkA=ARROW_LABEL_GAP,
            shrinkB=ARROW_LABEL_GAP,
        ),
    )


def plot(rows, methods, out_path, panel_titles=None):
    """
    Plot detection rates for both simulations in a factorial panel grid.

    Layout
    ------
    Rows     ->  annotators per item (labelled on the left margin)
    Columns  ->  minority share (labelled above the top row)
    Titles   ->  free-form, supplied via `panel_titles` (one per entry in
                 CONDITIONS, in the same order)
    Arrows   ->  ROW_ARROW_LABELS left of the rows,
                 COL_ARROW_LABELS above the columns

    Visual encoding
    ---------------
    Color  + marker  ->  method
    Line style       ->  simulation type (dashed and faded = unidirectional,
                         solid = bidirectional)
    """
    methods = order_methods(methods)
    df = pd.DataFrame(rows)

    n_rows, n_cols = len(ROW_VALUES), len(COL_VALUES)

    fig = plt.figure(figsize=FIGSIZE, layout="constrained")
    # Extra horizontal padding so the figure-level "Detection rate" label
    # is visibly separate from the elements next to it.
    fig.get_layout_engine().set(w_pad=LAYOUT_W_PAD)

    # Panel grid plus one thin track on the left (row arrow) and one on
    # top (column arrow); the top-left corner cell stays empty.
    gs = fig.add_gridspec(
        n_rows + 1,
        n_cols + 1,
        width_ratios=[ARROW_TRACK] + [1] * n_cols,
        height_ratios=[ARROW_TRACK] + [1] * n_rows,
    )

    axes = np.empty((n_rows, n_cols), dtype=object)
    for r, c in product(range(n_rows), range(n_cols)):
        axes[r, c] = fig.add_subplot(
            gs[r + 1, c + 1],
            sharey=axes[0, 0] if (r, c) != (0, 0) else None,
        )
        if c > 0:
            axes[r, c].tick_params(labelleft=False)

    _direction_arrow(
        fig.add_subplot(gs[1:, 0]), *ROW_ARROW_LABELS, vertical=True
    )
    _direction_arrow(
        fig.add_subplot(gs[0, 1:]), *COL_ARROW_LABELS, vertical=False
    )

    occupied = set()

    for k, (n_ann, minority) in enumerate(CONDITIONS):
        r = ROW_VALUES.index(n_ann)
        c = COL_VALUES.index(minority)
        occupied.add((r, c))

        plot_detection_panel(
            axes[r, c],
            panel_rows(df, methods, n_ann=n_ann, minority=minority),
            methods,
            title=panel_titles[k] if panel_titles is not None else "",
        )

    # Margin labels: rows on the left (first column), columns on top.
    # Row labels use the y-label slot and column labels are anchored to the
    # top of each top-row title, so the constrained layout engine reserves
    # space for both and they never overlap the (repurposed) titles.
    for r, n_ann in enumerate(ROW_VALUES):
        axes[r, 0].set_ylabel(_row_label(n_ann))

    # All column labels share one baseline: the top of the tallest title in
    # the first row (x follows each column's axes, y follows that union).
    top_titles = [ax.title for ax in axes[0]]

    def _top_titles_bbox(renderer):
        return Bbox.union(
            [title.get_window_extent(renderer) for title in top_titles]
        )

    for c, minority in enumerate(COL_VALUES):
        axes[0, c].annotate(
            _col_label(minority),
            xy=(0.5, 1.0),
            xycoords=("axes fraction", _top_titles_bbox),
            xytext=(0, COL_LABEL_PAD),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=plt.rcParams["axes.labelsize"],
        )

    empty_axes = [
        axes[r, c]
        for r, c in product(range(n_rows), range(n_cols))
        if (r, c) not in occupied
    ]
    for ax in empty_axes:
        ax.axis("off")

    _legends(fig, empty_axes, methods)

    fig.suptitle(
        "Efficacy of IAA and Other Prior Methods on "
        "Systematic Disagreement Attribution"
    )
    fig.supylabel(DETECTION_RATE_LABEL)
    fig.supxlabel(EFFECT_SIZE_LABEL)

    graphs.save_plot(out_path)


def _legends(fig, empty_axes, methods):
    """Method legend in the first empty grid cell (figure-level if the grid
    happens to be full).

    There is no simulation (line style) legend: at the graphs default font
    sizes it does not fit in the cell next to the method legend, so line
    styles are explained in the caption instead."""
    method_handles = method_legend_handles(methods)
    if empty_axes:
        empty_axes[0].legend(
            handles=method_handles, loc="center", title="Method", ncol=2
        )
    else:
        fig.legend(handles=method_handles, loc="lower right", title="Method")


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
