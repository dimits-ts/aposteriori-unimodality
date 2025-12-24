from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import apunim

from . import preprocessing, graphs


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

            ndfu_value = apunim.dfu(
                annotations, bins=bins, normalized=True
            )
            records.append({"SDB Feature": sdb_col, "nDFU": ndfu_value})

    plot_df = pd.DataFrame(records)

    # important for proper legend handling
    plot_df["SDB Feature"] = pd.Categorical(
        plot_df["SDB Feature"], categories=sdb_columns
    )

    # --- Plot ---
    plt.figure(figsize=(12, 7))
    sns.set(style="whitegrid", font_scale=1.2)

    ax = sns.histplot(
        data=plot_df,
        x="nDFU",
        hue="SDB Feature",
        multiple="stack",
        stat="count",
        palette="tab10",
        edgecolor="black",
    )

    ax.set_xlabel("nDFU (Polarization)")
    ax.set_ylabel("Number of Comments")
    ax.set_title(ds.get_name())
    ax.set_xlim(0, 1)

    # Force legend redraw (ensures it appears even if some bins are empty)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        title="SDB Feature",
        bbox_to_anchor=(0.75, 1),  # put legend inside plot
        loc="upper left",
    )

    plt.tight_layout()
    graphs.save_plot(output_path)
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

    plt.rcParams.update({
        "text.usetex": True,
        # Figure
        "figure.figsize": (12, 8),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,

        # Fonts
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,

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
    })

    COLORBLIND_PALETTE = [
        "#000000",  # black
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#009E73",  # bluish green
        "#F0E442",  # yellow
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#CC79A7",  # reddish purple
    ]

    sns.set_palette(COLORBLIND_PALETTE)
