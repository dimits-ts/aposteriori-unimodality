from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import apunim

from . import preprocessing

COLORBLIND_PALETTE = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#999999",  # mid gray (neutral, high legibility)
    "#8DD3C7",  # light teal (tone-safe extension)
    "#FDB462",  # light orange (paired but distinct in luminance)
    "#B3DE69",  # light yellow-green (safe vs. green due to brightness)
    "#80B1D3",  # soft blue (lightened blue variant)
    "#FB8072",  # soft coral (distinct from vermillion)
    "#CAB2D6",  # lavender (low-saturation purple)
    "#BC80BD",  # plum (dark purple contrast partner)
]

MARKERS = ["o", "s", "D", "^", "v", "P", "X"]
HATCHES = ["//", "\\\\", "xx", "oo", "..", "**", "++", "--"]


def polarization_plot(ds: preprocessing.Dataset, output_path: Path) -> None:
    df = ds.get_dataset()
    annotation_col = ds.get_annotation_column()
    sdb_columns = ds.get_sdb_columns()
    bins = len(
        np.unique(np.concatenate(df[ds.get_annotation_column()].to_list()))
    )

    records = []
    for sdb_col in sdb_columns:
        for _, row in df.iterrows():
            annotations = row[annotation_col]

            if (
                not isinstance(annotations, (list, np.ndarray))
                or len(annotations) == 0
            ):
                continue

            ndfu_value = apunim.dfu(annotations, bins=bins, normalized=True)
            records.append({"SDB Feature": sdb_col, "nDFU": ndfu_value})

    plot_df = pd.DataFrame(records)
    plot_df = plot_df.rename(columns={"SDB Feature": "PC Dimension"})

    # important for proper legend handling
    plot_df["PC Dimension"] = pd.Categorical(
        plot_df["PC Dimension"], categories=sdb_columns
    )

    ax = sns.histplot(
        data=plot_df,
        x="nDFU",
        hue="PC Dimension",
        multiple="stack",
        stat="count",
        palette=COLORBLIND_PALETTE,
        edgecolor="black",
        bins=10,
    )
    ax.get_legend().set_title(None)
    ax.set_xlabel("nDFU (Polarization)")
    ax.set_ylabel("Number of Comments")
    ax.set_title(ds.get_name())
    ax.set_xlim(0, 1)

    save_plot(output_path)
    plt.close()


def save_plot(path: Path) -> None:
    """
    Saves a plot to the specified filepath.

    :param path:
        The full path (including filename) where the plot will be saved.
    :type path: pathlib.Path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=300)
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
            "figure.figsize": (12, 8),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            # Fonts
            "font.family": "serif",
            "font.serif": ["Liberation Serif", "Nimbus Roman"],
            "font.size": 18,
            "axes.titlesize": 18,
            "axes.labelsize": 22,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 22,
            "figure.titlesize": 22,
            "figure.labelsize": 22,
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
