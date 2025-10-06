from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from . import preprocessing
from ..apunim import aposteriori


def polarization_plot(
    ds: preprocessing.Dataset, output_path: Path, bins: int = 5
) -> None:
    df = ds.get_dataset()
    annotation_col = ds.get_annotation_column()
    sdb_columns = ds.get_sdb_columns()

    # Prepare a list to store nDFU values per SDB group
    records = []

    for sdb_col in sdb_columns:
        # Iterate over each comment row
        for _, row in df.iterrows():
            annotations = row[annotation_col]
            sdb_values = row[sdb_col]

            # Ensure we have list-like annotations
            if (
                not isinstance(annotations, (list, np.ndarray))
                or len(annotations) == 0
            ):
                continue

            # Compute global nDFU for this comment
            ndfu_value = aposteriori.dfu(
                annotations, bins=bins, normalized=True
            )

            # Record one row per SDB group value
            # (You could also aggregate by unique SDB group if you prefer)
            for val in np.atleast_1d(sdb_values):
                records.append(
                    {"SDB Feature": sdb_col, "Group": val, "nDFU": ndfu_value}
                )

    plot_df = pd.DataFrame(records)

    # --- Plot ---
    sns.set(style="whitegrid", font_scale=1.1)
    g = sns.FacetGrid(
        plot_df,
        col="SDB Feature",
        col_wrap=3,
        sharex=True,
        sharey=False,
        height=3.5,
    )
    g.map_dataframe(
        sns.histplot,
        x="nDFU",
        hue="Group",
        stat="density",
        common_norm=False,
        multiple="stack",
        palette="tab10",
        bins=20,
    )

    g.set_axis_labels("nDFU (Polarization)", "Density")
    g.set_titles(col_template="{col_name}")
    g.add_legend(title="Group")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_plot(path: Path) -> None:
    """
    Saves a plot to the specified filepath.

    :param path:
        The full path (including filename) where the plot will be saved.
    :type path: pathlib.Path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    print(f"Figure saved to {path.resolve()}")
