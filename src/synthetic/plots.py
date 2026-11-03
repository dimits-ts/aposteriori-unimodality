"""
Plotting helpers shared by metric_comparison.py and
metric_comparison_multiple.py.

Each script keeps its own figure layout; this module draws the panels and
builds the legend entries so both figures encode methods and simulations
identically. Kept separate from simulation.py and methods.py so the
simulation workers do not import matplotlib / seaborn.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..lib import graphs
from .methods import LEGEND_LABEL, METHOD_ORDER
from .simulation import SIMULATION_BIDIRECTIONAL, SIMULATION_UNIDIRECTIONAL


SIMULATIONS = [SIMULATION_UNIDIRECTIONAL, SIMULATION_BIDIRECTIONAL]

# Line styles keyed by simulation label.
SIMULATION_LINESTYLE = {
    SIMULATION_UNIDIRECTIONAL: graphs.LINESTYLES[1],  # dashed
    SIMULATION_BIDIRECTIONAL: graphs.LINESTYLES[0],  # solid
}

# Human-readable legend labels for simulation types.
SIMULATION_LEGEND_LABEL = {
    SIMULATION_UNIDIRECTIONAL: "Unidirectional",
    SIMULATION_BIDIRECTIONAL: "Bidirectional",
}
# Unidirectional lines are faded so the bidirectional ones stand out.
SIMULATION_ALPHA = {
    SIMULATION_UNIDIRECTIONAL: 0.3,
    SIMULATION_BIDIRECTIONAL: 1,
}

LINE_WIDTH = 1.4
MARKER_SIZE = 7
LEGEND_MARKER_SIZE = 9
Y_LIM = (-0.04, 1.08)
# Simulation legend entries are neutral so they read as line styles only.
SIMULATION_LEGEND_COLOR = "black"

# Figure-level axis labels, identical across both figures.
DETECTION_RATE_LABEL = "Detection Rate (higher=better)"
EFFECT_SIZE_LABEL = (
    r"Mean Rating Difference $\delta$"
    " Between Groups (higher is easier to detect)"
)


def order_methods(methods: list[str]) -> list[str]:
    """METHOD_ORDER first, then any unknown methods in their given order."""
    return [m for m in METHOD_ORDER if m in methods] + [
        m for m in methods if m not in METHOD_ORDER
    ]


def panel_rows(
    df: pd.DataFrame, methods: list[str], **conditions
) -> pd.DataFrame:
    """Rows of one panel, e.g.
    panel_rows(df, methods, n_ann=20, minority=0.2)."""
    mask = df["method"].isin(methods)
    for column, value in conditions.items():
        mask &= df[column] == value
    return df[mask]


def plot_detection_panel(
    ax, panel_df: pd.DataFrame, methods: list[str], title: str
) -> None:
    """Detection rate against delta, one line per (method, simulation)."""
    for i, method in enumerate(methods):
        method_df = panel_df[panel_df["method"] == method]
        for simulation in SIMULATIONS:
            sim_df = method_df[method_df["simulation"] == simulation]
            if len(sim_df) > 0:
                _detection_line(ax, sim_df, i, simulation)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title)
    ax.set_ylim(*Y_LIM)


def _detection_line(
    ax, sim_df: pd.DataFrame, method_idx: int, simulation: str
) -> None:
    sns.lineplot(
        data=sim_df,
        x="delta",
        y="detected",
        estimator="mean",
        errorbar="se",
        marker=_method_marker(method_idx),
        color=_method_color(method_idx),
        linestyle=SIMULATION_LINESTYLE[simulation],
        lw=LINE_WIDTH,
        ms=MARKER_SIZE,
        err_style="bars",
        err_kws={"capsize": 3},
        ax=ax,
        legend=False,
        alpha=SIMULATION_ALPHA[simulation],
    )


def method_legend_handles(methods: list[str]) -> list[plt.Line2D]:
    return [
        plt.Line2D(
            [0],
            [0],
            color=_method_color(i),
            marker=_method_marker(i),
            linestyle=graphs.LINESTYLES[0],
            lw=LINE_WIDTH,
            ms=LEGEND_MARKER_SIZE,
            label=LEGEND_LABEL.get(m, m),
        )
        for i, m in enumerate(methods)
    ]


def simulation_legend_handles() -> list[plt.Line2D]:
    """Line style (and fading) per simulation, as drawn in the panels."""
    return [
        plt.Line2D(
            [0],
            [0],
            color=SIMULATION_LEGEND_COLOR,
            linestyle=SIMULATION_LINESTYLE[simulation],
            lw=LINE_WIDTH,
            alpha=SIMULATION_ALPHA[simulation],
            label=SIMULATION_LEGEND_LABEL[simulation],
        )
        for simulation in SIMULATIONS
    ]


def add_simulation_legend(fig, below=None) -> None:
    """Simulation legend centred below the figure's bottom edge (under the
    shared x-label), or below the artist `below` if given.

    Legends outside the figure need an explicit figure-coordinate anchor:
    without one they re-anchor to the grown saved area when save_plot's
    bbox_inches="tight" includes them, and land on top of each other."""
    fig.legend(
        handles=simulation_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, _bottom_of(fig, below) if below else 0),
        title="Simulation",
        ncols=2,
    )


def _bottom_of(fig, artist) -> float:
    """Bottom edge of an artist, in figure coordinates."""
    fig.draw_without_rendering()
    extent = artist.get_window_extent()
    return extent.transformed(fig.transFigure.inverted()).y0


def _method_color(method_idx: int):
    palette = graphs.COLORBLIND_PALETTE
    return palette[method_idx % len(palette)]


def _method_marker(method_idx: int):
    return graphs.MARKERS[method_idx % len(graphs.MARKERS)]
