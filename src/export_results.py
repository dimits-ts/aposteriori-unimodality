import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .tasks import run_helper
from .tasks import graphs


def csv_to_latex(results_dir: Path, latex_output_dir: Path) -> None:
    for result_file in results_dir.rglob("*.csv"):
        dataset_name = result_file.stem
        df = pd.read_csv(result_file)
        df = df.loc[df.pvalue.notna()]
        run_helper.results_to_latex(
            res_df=df,
            output_path=latex_output_dir / f"{dataset_name}.tex",
            dataset_name=dataset_name,
            table_label=f"tab:{dataset_name}",
        )


def ordinal_graphs(results_dir: Path, graph_output_dir: Path) -> None:
    """
    For each CSV in results_dir:
    - Identify ordinal-valued rows grouped by the 'SDB Feature' column.
    - Keep only groups where at least one ordinal factor has pvalue <= 0.5.
    - Extract ordinals and build a stretched x-axis per feature.
    - Plot ordinal vs apunim across all datasets.
    """
    records = []

    # --- Collect all data first ---
    for file in results_dir.rglob("*.csv"):
        df = pd.read_csv(file)
        dataset = file.stem

        if "SDB Feature" not in df.columns:
            continue

        ordinal_col = next(
            (
                c
                for c in df.columns
                if df[c].astype(str).str.match(r"^\d+\)").any()
            ),
            None,
        )
        if ordinal_col is None:
            continue

        for feature_name, df_group in df.groupby("SDB Feature"):

            g = df_group[
                df_group[ordinal_col].astype(str).str.match(r"^\d+\)")
            ].copy()
            g = g[g.pvalue.notna()]

            if g.empty:
                continue

            if (g.pvalue > 0.05).all():
                continue

            g["ordinal"] = (
                g[ordinal_col].astype(str).str.extract(r"^(\d+)").astype(int)
            )

            for _, row in g.iterrows():
                records.append(
                    {
                        "dataset": dataset,
                        "feature": f"{dataset}-{feature_name}",
                        "ordinal": row["ordinal"],
                        "apunim": row["apunim"],
                    }
                )

    if not records:
        print("No usable ordinal data found.")
        return

    data = pd.DataFrame(records)

    # --- Stretch each feature's ordinal series ---
    max_points = data.groupby("feature")["ordinal"].max().max()
    stretched_records = []

    for feature, df_feat in data.groupby("feature"):
        df_feat = df_feat.sort_values("ordinal").reset_index(drop=True)
        n_rows = len(df_feat)
        df_feat["stretched_ordinal"] = np.linspace(1, max_points, n_rows)
        stretched_records.append(df_feat)

    data_stretched = pd.concat(stretched_records, ignore_index=True)

    # --- Color configuration ---
    highlight_group_1 = {
        "kumar-Religion Important",
        "dices-990-Age",
    }

    highlight_group_2 = {
        "kumar-Education",
        "kumar-Toxicity Problem",
    }

    COLOR_GROUP_1 = "#0072B2"  # blue
    COLOR_GROUP_2 = "#D55E00"  # vermillion
    COLOR_OTHER = "#B0B0B0"  # light gray

    palette = {}
    for f in data_stretched["feature"].unique():
        if f in highlight_group_1:
            palette[f] = COLOR_GROUP_1
        elif f in highlight_group_2:
            palette[f] = COLOR_GROUP_2
        else:
            palette[f] = COLOR_OTHER

    # --- Plot ---
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=data_stretched,
        x="stretched_ordinal",
        y="apunim",
        hue="feature",
        style="feature",
        markers=True,
        dashes=True,
        markersize=10,
        legend=True,
        errorbar=None,
        palette=palette,
    )

    # De-emphasize non-highlighted lines
    for line in ax.lines:
        if line.get_color() == COLOR_OTHER:
            line.set_alpha(0.6)

    add_grouped_legend(
        ax,
        group_1=highlight_group_1,
        group_1_title="Escalating",
        group_2=highlight_group_2,
        group_2_title="Diverging",
        others_title="Neither",
        legend_title="Behavior",
        loc="lower center"
    )

    plt.title("Apunim trends in ordinal variables")
    plt.xlabel("Order (normalized)")
    plt.ylabel("Apunim value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    graphs.save_plot(graph_output_dir / "apunim_ordinal.png")


def add_grouped_legend(
    ax,
    group_1: Iterable[str],
    group_2: Iterable[str],
    group_1_title: str = "Highlighted: Group 1",
    group_2_title: str = "Highlighted: Group 2",
    others_title: str = "Other features",
    legend_title: str = "Feature groups",
    loc: str = "best",
):
    """
    Create a grouped legend on an existing axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis containing the plotted lines.
    group_1, group_2 : iterable of str
        Feature names belonging to the two highlighted groups.
    """
    handles, labels = ax.get_legend_handles_labels()
    handle_map = dict(zip(labels, handles))

    group_1 = list(group_1)
    group_2 = list(group_2)

    highlighted = set(group_1) | set(group_2)

    legend_handles = []
    legend_labels = []

    def add_group(title, features):
        # Section header (dummy handle)
        legend_handles.append(Line2D([], [], linestyle="none"))
        legend_labels.append(title)

        for f in features:
            if f in handle_map:
                legend_handles.append(handle_map[f])
                legend_labels.append(f)

    add_group(group_1_title, group_1)
    add_group(group_2_title, group_2)

    other_features = [f for f in labels if f not in highlighted]
    if other_features:
        add_group(others_title, other_features)

    legend = ax.legend(
        legend_handles,
        legend_labels,
        title=legend_title,
        frameon=True,
        loc=loc,
    )

    # Make section headers bold
    for text in legend.get_texts():
        if text.get_text() in {group_1_title, group_2_title, others_title}:
            text.set_weight("bold")

    return legend


def main(results_dir: Path, latex_output_dir: Path, graph_output_dir: Path):
    graphs.graph_setup()
    csv_to_latex(results_dir=results_dir, latex_output_dir=latex_output_dir)
    ordinal_graphs(results_dir=results_dir, graph_output_dir=graph_output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Create graphs and latex tables from results.")
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Results CSV directory.",
    )
    parser.add_argument(
        "--latex-output-dir",
        required=True,
        help="Directory for the latex tables.",
    )
    parser.add_argument(
        "--graph-output-dir",
        required=True,
        help="Directory for graphs.",
    )
    args = parser.parse_args()
    main(
        results_dir=Path(args.results_dir),
        latex_output_dir=Path(args.latex_output_dir),
        graph_output_dir=Path(args.graph_output_dir),
    )
