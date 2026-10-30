from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns
import numpy as np
import apunim

from . import labels, preprocessing

# Okabe–Ito palette
COLORBLIND_PALETTE = [
    # --- Core ---
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    # --- Neutral / light extensions ---
    "#999999",
    "#8DD3C7",
    "#FDB462",
    "#B3DE69",
    "#80B1D3",
    "#FB8072",
    "#CAB2D6",
    "#BC80BD",
    # --- Additional medium/dark colors ---
    "#4D4D4D",
    "#1B9E77",
    "#7570B3",
    "#A6761D",
    "#66A61E",
    "#E7298A",
    "#A6CEE3",
    "#FFB000",
]

MARKERS = ["o", "s", "D", "^", "v", "P", "X"]
HATCHES = ["//", "\\\\", "++", "oo", "..", "xx", "**", "--"]

# Visually distinct line styles, as (offset, (on, off, ...)) dash tuples so
# every pattern is explicit. Patterns scale with linewidth (matplotlib's
# lines.scale_dashes), so they stay distinct at any line weight.
# Index meaning is fixed; callers pick by index or cycle in order:
#   0 solid, 1 dashed, 2 dotted, 3 dash-dot, 4 dash-dot-dot,
#   5 long dash, 6 loosely dotted
LINESTYLES = [
    "solid",
    (0, (5, 2)),
    (0, (1, 1)),
    (0, (5, 2, 1, 2)),
    (0, (5, 2, 1, 2, 1, 2)),
    (0, (10, 3)),
    (0, (1, 3)),
]

DEFAULT_FIGSIZE = (14, 6)


def polarization_plot(
    ds: preprocessing.Dataset, output_path: Path, dpi: int = 300
) -> None:
    df = ds.get_dataset()
    annotation_col = ds.get_annotation_column()
    sdb_columns = ds.get_sdb_columns()

    # Data Preparation
    all_annotations = []
    for annotations_list in df[annotation_col].to_list():
        if isinstance(annotations_list, (list, np.ndarray)):
            all_annotations.extend(annotations_list)

    if not all_annotations:
        print(
            "Warning: No valid annotations found. "
            "Cannot calculate polarization."
        )
        return

    bins = len(np.unique(all_annotations))

    records = []

    # ITERATE THROUGH ALL COMMENTS
    for _, row in df.iterrows():
        annotations = row[annotation_col]

        # Check if the comment has any annotations
        if (
            not isinstance(annotations, (list, np.ndarray))
            or len(annotations) == 0
        ):
            continue

        # Calculate NDFU for the entire comment (row)
        try:
            ndfu_value = apunim.dfu(annotations, bins=bins, normalized=True)
        except Exception as e:
            # Handle cases where the apunim function might fail
            print(f"Error calculating NDFU for an item: {e}")
            continue

        for sdb_col in sdb_columns:
            sdb_values = row[sdb_col]

            for value in sdb_values:
                combined_category = f"{sdb_col}: {value}"
                records.append(
                    {"PC Dimension": combined_category, "nDFU": ndfu_value}
                )

    plot_df = pd.DataFrame(records)

    # Note: Since the categories are now complex strings (e.g., "Race: Asian"),
    # setting a strict, predefined order is usually necessary for
    # clean visualization.
    plot_df["PC Dimension"] = pd.Categorical(
        plot_df["PC Dimension"], ordered=True
    )

    fig, ax = plt.subplots()

    sns.boxplot(
        x="PC Dimension",
        y="nDFU",
        data=plot_df,
        ax=ax,
        # skip black color
        palette=COLORBLIND_PALETTE[1:],
    )

    ax.set_xlabel("Group")
    ax.set_ylabel(labels.NDFU.short)
    ax.set_title(labels.dataset_label(ds.get_name()))

    ax.set_ylim(-0.05, 1.05)

    plt.xticks(rotation=90, ha="right")
    plt.grid(axis="y", alpha=0.5)

    save_plot(output_path, dpi)
    plt.close()


def read_inherent_csv(path: Path) -> pd.Series:
    """Inherent-polarization values of a "<name>-inherent*.csv" file (as
    written by the human, LLM and synthetic scripts), NaNs dropped."""
    values = pd.read_csv(path).inherent_polarization
    return values[~np.isnan(values)]


def _draw_density_bars(
    ax, values: pd.Series, edges: np.ndarray, sign: int, color, hatch
) -> None:
    """Horizontal density histogram on shared `edges`, drawn to the right
    of x=0 (sign=1) or mirrored to the left (sign=-1)."""
    counts, _ = np.histogram(values, bins=edges)
    widths = np.diff(edges)
    density = counts / (counts.sum() * widths) if counts.sum() else counts
    ax.barh(
        y=edges[:-1],
        width=sign * density,
        height=widths,
        align="edge",
        color=color,
        alpha=0.7,
        hatch=hatch,
        edgecolor="black",
        linewidth=0.3,
    )


def plot_mirrored_inherent_histograms(
    right: dict[str, pd.Series],
    left: dict[str, pd.Series],
    output_path: Path,
    title: str,
    right_label: str,
    left_label: str,
    order: list[str] | None = None,
    legend_title: str | None = None,
    bins: int = 30,
    figsize: tuple[float, float] = (10, 4),
) -> None:
    """
    Mirrored horizontal density histograms of inherent polarization
    (USD), one colour/hatch per series: `right` is drawn right of the
    centre line, `left` mirrored to its left. A series present on both
    sides keeps the same colour and hatch on both, so the two histograms
    of one series mirror each other.

    Used for the human (one series per dataset), LLM (per model) and
    synthetic (per number of groups) figures.

    `order` fixes the colour/hatch assignment and the legend order; by
    default the series are sorted by name. Names in `order` without values
    are not drawn but keep their colour/hatch slot, so figures sharing an
    `order` style a series identically. All histograms share one set of
    bins on [0, 1], so the two sides line up.
    """
    style_order = list(order) if order is not None else sorted(right)
    style_order += sorted((set(right) | set(left)) - set(style_order))
    # Styles are assigned over the whole `order`, before dropping names
    # without values, so a series keeps its colour/hatch across figures
    # that show different subsets of the same order.
    styles = {
        name: (
            COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)],
            HATCHES[i % len(HATCHES)],
        )
        for i, name in enumerate(style_order)
    }
    names = [n for n in style_order if n in right or n in left]
    if not names:
        print(f"No inherent-polarization values; skipping {output_path}.")
        return
    edges = np.linspace(0, 1, bins + 1)

    fig, ax = plt.subplots(figsize=figsize)

    for sign, series in ((1, right), (-1, left)):
        for name in sorted(series):
            _draw_density_bars(ax, series[name], edges, sign, *styles[name])

    # symmetric x-range so the centre line sits in the middle
    xmax = max(abs(x) for x in ax.get_xlim())
    ax.set_xlim(-xmax, xmax)
    ax.axvline(0, color="black", linewidth=0.8)
    # densities are positive on both sides
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{abs(x):g}"))

    for x, text, ha in (
        (-0.02, left_label, "right"),
        (0.02, right_label, "left"),
    ):
        ax.text(
            0.5 + x,
            0.97,
            text,
            transform=ax.transAxes,
            ha=ha,
            va="top",
            fontsize="small",
        )

    legend_handles = [
        mpatches.Patch(
            facecolor=styles[name][0],
            edgecolor="black",
            hatch=styles[name][1],
            label=name,
            alpha=0.4,
        )
        for name in names
    ]

    ax.set_title(title)
    ax.set_xlabel("Density")
    ax.set_ylabel(labels.USD.short)
    ax.set_ylim((0, 1))

    ax.legend(
        handles=legend_handles,
        title=legend_title,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    save_plot(output_path)
    plt.close(fig)


def save_plot(path: Path, dpi: int = 300) -> None:
    """
    Saves a plot to the specified filepath.

    :param path:
        The full path (including filename) where the plot will be saved.
    :type path: pathlib.Path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=dpi)
    print(f"Figure saved to {path.resolve()}")


def graph_setup() -> None:
    sns.set_theme(
        context="paper",
        style="ticks",
        font="serif",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        },
    )

    plt.rcParams.update(
        {
            "text.usetex": True,
            # Figure
            "figure.figsize": DEFAULT_FIGSIZE,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            # Fonts
            "font.family": "serif",
            "font.serif": ["Liberation Serif", "Nimbus Roman"],
            "font.size": 18,
            "axes.titlesize": 24,
            "axes.labelsize": 22,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 22,
            "legend.title_fontsize": 22,
            "figure.titlesize": 26,
            "figure.labelsize": 22,
            # Subplots
            "figure.subplot.wspace": 0.3,
            "figure.subplot.hspace": 0.7,
            "figure.subplot.bottom": 0.2,
            # Axes
            "axes.linewidth": 0.8,
            "axes.edgecolor": "black",
            "axes.grid": False,
            # Ticks
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            # Lines
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            # Legend
            "legend.frameon": False,
            "legend.loc": "best",
            # Math text
            "mathtext.fontset": "cm",
            # PDF/PS output (important for LaTeX + journals)
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    sns.set_palette(COLORBLIND_PALETTE)
