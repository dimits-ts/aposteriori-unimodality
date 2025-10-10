from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from . import preprocessing, graphs
from ..apunim import aposteriori


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

            ndfu_value = aposteriori.dfu(
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
    if not handles:  # if seaborn didn't generate any
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=c)
            for c in sns.color_palette("tab10", n_colors=len(sdb_columns))
        ]
        labels = sdb_columns
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
