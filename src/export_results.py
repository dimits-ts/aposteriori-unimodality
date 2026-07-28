import argparse
from pathlib import Path
from collections.abc import Iterable

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import tasks.graphs
import tasks.run_helper

MIN_SUPPORT = 50
SIG_ALPHA = 0.05  # p-value threshold for "statistically significant"

# Explicit marker/linestyle cycles so we can reuse the *same* marker per
# feature when overlaying filled vs. hollow points. (sns.lineplot's
# auto-assigned style-index isn't something you can reliably recover
# after the fact, so we do the cycling ourselves.)
MARKER_CYCLE = ["o", "X", "P", "^", "D", "s", "v", "*", "d", "p"]
LINESTYLE_CYCLE = [
    (0, (1, 1)),  # dotted
    (0, (5, 2)),  # dashed
    (0, (5, 1, 1, 1)),  # dash-dot
    (0, (3, 1, 1, 1, 1, 1)),  # dash-dot-dot
    "solid",
]


def main(results_dir: Path, latex_output_dir: Path, graph_output_dir: Path):
    tasks.graphs.graph_setup()
    csv_to_latex(
        result_paths=list(results_dir.rglob("*-results.csv")),
        latex_output_dir=latex_output_dir,
    )
    ordinal_graph(results_dir=results_dir, graph_output_dir=graph_output_dir)
    ordinal_graph_per_feature(
        results_dir=results_dir, graph_output_dir=graph_output_dir
    )
    plot_dfu_histograms(
        file_paths=list(results_dir.rglob("*-inherent.csv")),
        graph_output_dir=graph_output_dir,
    )
    plot_sample_size_polarization(
        csv_path=results_dir / "sample_size_polarization.csv",
        output_path=graph_output_dir / "sample_size_polarization.png",
    )


def plot_dfu_histograms(
    file_paths: list[Path],
    graph_output_dir: Path,
    bins: int = 30,
):
    """
    Plot histogram distributions per dataset with colorblind palette
    and hatch patterns.
    """
    all_data = []

    for path in file_paths:
        path = Path(path)
        label = " ".join(path.stem.split("-")[:-1]).capitalize()

        arr = pd.read_csv(path).inherent_polarization
        arr = arr[~np.isnan(arr)]

        all_data.append(pd.DataFrame({"value": arr, "dataset": label}))

    full_df = pd.concat(all_data, ignore_index=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    datasets = sorted(full_df["dataset"].unique())
    legend_handles = []
    for i, (dataset, color, hatch) in enumerate(
        zip(datasets, tasks.graphs.COLORBLIND_PALETTE, tasks.graphs.HATCHES)
    ):
        before = len(ax.patches)

        sns.histplot(
            data=full_df[full_df["dataset"] == dataset],
            x="value",
            bins=bins,
            stat="density",
            common_norm=False,
            alpha=0.7,
            color=color,
            ax=ax,
        )

        # Only newly created bars
        new_patches = ax.patches[before:]

        for patch in new_patches:
            patch.set_hatch(hatch)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.3)

        legend_handles.append(
            plt.matplotlib.patches.Patch(
                facecolor=color,
                edgecolor="black",
                hatch=hatch,
                label=dataset,
                alpha=0.4,
            )
        )

    ax.set_xlabel("Inherent polarization")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)

    ax.legend(handles=legend_handles)

    tasks.graphs.save_plot(graph_output_dir / "apriori.png")
    plt.close()


def csv_to_latex(result_paths: list[Path], latex_output_dir: Path) -> None:
    for result_file in result_paths:
        if "sample_size" not in result_file.stem:
            dataset_name = result_file.stem
            df = pd.read_csv(result_file)
            df = df.loc[df.pvalue.notna()]
            tasks.run_helper.results_to_latex(
                res_df=df,
                output_path=latex_output_dir / f"{dataset_name}.tex",
                dataset_name=dataset_name,
                table_label=f"tab:{dataset_name}",
            )


def ordinal_graph_per_feature(
    results_dir: Path, graph_output_dir: Path
) -> None:
    for file in results_dir.rglob("*.csv"):
        df = pd.read_csv(file)
        dataset = file.stem.replace("-results", "")

        if "SDB Feature" not in df.columns or "Unnamed: 1" not in df.columns:
            continue

        for feature_name, g in df.groupby("SDB Feature"):

            g = g[g["Unnamed: 1"].astype(str).str.match(r"^\d+\)")]
            g = g[g.apunim.notna()]
            g = g[g["support"] >= MIN_SUPPORT]
            if g.empty:
                continue

            g["ordinal_num"] = (
                g["Unnamed: 1"].astype(str).str.extract(r"^(\d+)").astype(int)
            )
            g["ordinal_label"] = g["Unnamed: 1"]

            # Drop duplicates so each label appears once
            g_unique = g.drop_duplicates(subset="ordinal_label")

            plt.figure(figsize=(8, 5))
            ax = sns.lineplot(
                data=g,
                x="ordinal_num",  # use numeric x-axis
                y="apunim",
                marker="o",
                errorbar=None,
            )

            plt.title(f"{dataset} — {feature_name}")
            plt.xlabel("")
            plt.ylabel("Apunim value")
            plt.grid(True, alpha=0.3)

            # One tick per unique label
            ax.set_xticks(g_unique["ordinal_num"])
            ax.set_xticklabels(
                g_unique["ordinal_label"], rotation=45, ha="right"
            )

            plt.tight_layout()

            safe_feature = (
                str(feature_name).replace(" ", "_").replace("/", "-")
            )
            out_path = (
                graph_output_dir
                / f"apunim_ordinal_{dataset}_{safe_feature}.png"
            )

            tasks.graphs.save_plot(out_path)
            plt.close()


def plot_sample_size_polarization(csv_path: Path, output_path: Path):
    df = pd.read_csv(csv_path)

    _, ax = plt.subplots(figsize=(8, 5))

    for dataset, group in df.groupby("dataset"):
        color = sns.color_palette()[
            list(df["dataset"].unique()).index(dataset)
        ]
        ax.plot(
            group["sample_size"], group["mean"], label=dataset, color=color
        )
        ax.fill_between(
            group["sample_size"],
            group["mean"] - group["std"],
            group["mean"] + group["std"],
            alpha=0.2,
            color=color,
        )

    ax.set_xlabel("Number of annotators")
    ax.set_ylabel("Mean polarization")
    ax.legend(title="Dataset")
    tasks.graphs.save_plot(output_path)
    plt.close()


def ordinal_graph(results_dir: Path, graph_output_dir: Path) -> None:
    """
    For each CSV in results_dir:
    - Identify ordinal-valued rows grouped by the 'SDB Feature' column.
    - Extract ordinals and build a stretched x-axis per feature.
    - Plot ordinal vs apunim across all datasets.
    - Mark non-statistically-significant points with hollow markers.
    """
    records = []

    # --- Collect all data first ---
    for file in results_dir.rglob("*.csv"):
        df = pd.read_csv(file)
        dataset = file.stem.replace("-results", "")

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
            g = g[g["support"] >= MIN_SUPPORT]

            if g.empty:
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
                        "pvalue": row["pvalue"],
                    }
                )

    if not records:
        print("No usable ordinal data found.")
        return

    data = pd.DataFrame(records)
    data["significant"] = data["pvalue"] < SIG_ALPHA

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
        "sap-Age",
    }

    highlight_group_2 = {}

    COLOR_GROUP_1 = tasks.graphs.COLORBLIND_PALETTE[0]
    COLOR_GROUP_2 = tasks.graphs.COLORBLIND_PALETTE[1]
    COLOR_OTHER = tasks.graphs.COLORBLIND_PALETTE[2]

    all_features = list(data_stretched["feature"].unique())

    # Order features so the two highlight groups come first (matches the
    # legend grouping), keeping cycling deterministic run-to-run.
    ordered_features = (
        [f for f in all_features if f in highlight_group_1]
        + [f for f in all_features if f in highlight_group_2]
        + [
            f
            for f in all_features
            if f not in highlight_group_1 | set(highlight_group_2)
        ]
    )

    palette = {}
    marker_map = {}
    linestyle_map = {}
    for i, f in enumerate(ordered_features):
        if f in highlight_group_1:
            palette[f] = COLOR_GROUP_1
        elif f in highlight_group_2:
            palette[f] = COLOR_GROUP_2
        else:
            palette[f] = COLOR_OTHER
        marker_map[f] = MARKER_CYCLE[i % len(MARKER_CYCLE)]
        linestyle_map[f] = LINESTYLE_CYCLE[i % len(LINESTYLE_CYCLE)]

    # --- Plot (manual matplotlib instead of sns.lineplot so we control
    # marker fill per-point for the significance encoding) ---
    fig, ax = plt.subplots(figsize=(16, 8))

    for f in ordered_features:
        df_feat = data_stretched[data_stretched["feature"] == f].sort_values(
            "stretched_ordinal"
        )
        color = palette[f]
        marker = marker_map[f]
        linestyle = linestyle_map[f]
        line_alpha = 0.6 if color == COLOR_OTHER else 1.0

        x = df_feat["stretched_ordinal"].to_numpy()
        y = df_feat["apunim"].to_numpy()
        sig = df_feat["significant"].to_numpy()

        # Line only (no markers here — markers drawn separately below so
        # we can vary fill per point).
        (line,) = ax.plot(
            x,
            y,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            alpha=line_alpha,
            label=f,
            zorder=2,
        )

        # Filled markers: significant points. Full opacity regardless of
        # line alpha, and a thin black rim so the marker reads crisply
        # even for pale colors (e.g. the light-blue "Other" series).
        if sig.any():
            ax.plot(
                x[sig],
                y[sig],
                linestyle="none",
                marker=marker,
                markersize=11,
                markerfacecolor=color,
                markeredgecolor="grey",
                markeredgewidth=0.8,
                zorder=3,
            )

        # Hollow markers: non-significant points. Edge is always black
        # (not the series color) so fill/no-fill contrast doesn't depend
        # on how light or desaturated that series' color is.
        if (~sig).any():
            ax.plot(
                x[~sig],
                y[~sig],
                linestyle="none",
                marker=marker,
                markersize=9,
                markerfacecolor="white",
                markeredgecolor="grey",
                markeredgewidth=1.6,
                zorder=3,
            )

        # Keep a handle with the *feature's* marker for the legend
        # (matplotlib's Line2D used below in add_grouped_legend).
        line.set_marker(marker)
        line.set_markerfacecolor(color)
        line.set_markeredgecolor(color)
        line.set_markersize(9)

    add_grouped_legend(
        ax,
        group_1=highlight_group_1,
        group_1_title="Directional",
        group_2=highlight_group_2,
        group_2_title="Diverging",
        others_title="Other",
        loc="lower center",
    )

    ax.set_xlabel(r"Order (low $\rightarrow$ high)")
    ax.set_ylabel("Apunim")
    ax.grid(True, alpha=0.3)
    ax.set_xticks([])  # Remove x-axis ticks
    fig.tight_layout()

    tasks.graphs.save_plot(graph_output_dir / "apunim_ordinal.png")


def add_grouped_legend(
    ax,
    group_1: Iterable[str],
    group_2: Iterable[str],
    group_1_title: str = "Highlighted: Group 1",
    group_2_title: str = "Highlighted: Group 2",
    others_title: str = "Other features",
    loc: str = "best",
):
    """
    Create a grouped legend on an existing axis, plus a small
    significance key (filled vs. hollow marker) at the end.

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

    if len(group_2) > 0:
        add_group(group_2_title, group_2)

    other_features = [f for f in labels if f not in highlighted]
    if other_features:
        add_group(others_title, other_features)

    # --- Significance key ---
    legend_handles.append(Line2D([], [], linestyle="none"))
    legend_labels.append("Significance")

    legend_handles.append(
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=9,
            markerfacecolor="black",
            markeredgecolor="black",
        )
    )
    legend_labels.append(r"significant ($p<0.05$)")

    legend_handles.append(
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=9,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.6,
        )
    )
    legend_labels.append("not significant")

    legend = ax.legend(
        legend_handles,
        legend_labels,
        frameon=True,
        loc=loc,
        bbox_to_anchor=(1.3, 0),
    )

    # Make section headers bold
    for text in legend.get_texts():
        if text.get_text() in {
            group_1_title,
            group_2_title,
            others_title,
            "Significance",
        }:
            text.set_weight("bold")

    return legend


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
