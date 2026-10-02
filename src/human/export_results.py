import argparse
from os import path
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.sparse import data
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ..lib import graphs
from ..lib.util import center_table_latex, significance_superscript
from ..lib.preprocessing import (
    Dataset,
    DicesDataset,
    PopquornDataset,
    SapDataset,
    KumarDataset,
)

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


def main(
    results_dir: Path,
    dices_small_path: Path,
    dices_large_path: Path,
    popquorn_offensiveness_path: Path,
    kumar_path: Path,
    sap_path: Path,
    latex_output_dir: Path,
    graph_output_dir: Path,
):
    graphs.graph_setup()
    csv_to_latex(
        result_paths=list(results_dir.rglob("*-results.csv")),
        latex_output_dir=latex_output_dir,
    )
    plot_dfu_histograms(
        file_paths=list(results_dir.rglob("*-inherent.csv")),
        graph_output_dir=graph_output_dir,
    )
    plot_sample_size_polarization(
        csv_path=results_dir / "sample_size_polarization.csv",
        output_path=graph_output_dir / "sample_size_polarization.png",
    )

    dices350_ds = DicesDataset(dataset_path=dices_small_path, variant="350")
    dices990_ds = DicesDataset(dataset_path=dices_large_path, variant="990")
    popquorn_ds = PopquornDataset(dataset_path=popquorn_offensiveness_path)
    kumar_ds = KumarDataset(dataset_path=kumar_path)
    sap_ds = SapDataset(dataset_path=sap_path)
    datasets: list[Dataset] = [
        dices350_ds,
        dices990_ds,
        popquorn_ds,
        kumar_ds,
        sap_ds,
    ]

    Dataset.print_descriptive_statistics(datasets)
    Dataset.print_annotation_count_table(datasets)

    ann_size_df = get_annotator_counts_df(datasets)
    stats_df = get_statistics_df(ann_size_df)
    stats_df.to_latex(
        latex_output_dir / "ann_stats.tex",
        caption=(
            "Descriptive statistics for the number of annotations per dataset."
        ),
        label="tab:num-annot",
        position="ht",
        index=True,
        float_format="%.4f",
        escape=True,
    )

    plot_annotator_count_histogram_from_datasets(
        datasets=datasets,
        graph_path=graph_output_dir / "annotator_count_histogram.png",
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

    fig, ax = plt.subplots(figsize=(6, 7))

    datasets = sorted(full_df["dataset"].unique())
    legend_handles = []
    for i, (dataset, color, hatch) in enumerate(
        zip(datasets, graphs.COLORBLIND_PALETTE, graphs.HATCHES)
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
            mpatches.Patch(
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

    graphs.save_plot(graph_output_dir / "apriori.png")
    plt.close()


def plot_annotator_count_histogram_from_datasets(
    datasets: list[Dataset],
    graph_path: Path,
):
    """
    Plot a histogram of annotator counts per comment across multiple datasets,
    showing the percentage of comments for each bin.

    Parameters
    ----------
    datasets : list
        List of dataset objects.
    graph_path : Path
        If provided, saves the figure to this path.
    """
    N_BINS = 100
    all_df = get_annotator_counts_df(datasets)

    dataset_names = all_df["dataset"].unique().tolist()

    # Determine bin boundaries
    min_val = all_df["n_annotators"].min()
    max_val = all_df["n_annotators"].max()
    bins_edges = np.linspace(min_val, max_val, N_BINS + 1)
    _, ax = plt.subplots()

    for i, dataset_name in enumerate(dataset_names):
        data_subset = all_df[all_df["dataset"] == dataset_name]["n_annotators"]

        raw_counts, edges = np.histogram(data_subset, bins=bins_edges)

        total_comments_for_dataset = len(data_subset)

        if total_comments_for_dataset > 0:
            percentage_counts = raw_counts / total_comments_for_dataset
        else:
            percentage_counts = np.zeros_like(raw_counts, dtype=float)

        selected_color = graphs.COLORBLIND_PALETTE[
            i % len(graphs.COLORBLIND_PALETTE)
        ]
        selected_hatch = graphs.HATCHES[i % len(graphs.HATCHES)]

        ax.bar(
            x=edges[:-1],
            height=percentage_counts * 100,
            width=(edges[1] - edges[0]),
            label=dataset_name,
            color=selected_color,
            alpha=0.6,
            hatch=selected_hatch,
            edgecolor="black",
        )

    ax.legend(title=None, loc="center")
    ax.set_xlabel(r"\# Annotators")
    ax.set_ylabel(r"Comments (\%)")
    ax.set_title(r"\# Annotators per comment for each dataset")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    graphs.save_plot(graph_path)
    plt.close()


def get_annotator_counts_df(
    datasets: list[Dataset],
) -> pd.DataFrame:
    rows = []

    for ds in datasets:
        df = ds.get_dataset().reset_index(drop=True)
        ann_col = ds.get_annotation_column()
        ds_name = ds.get_name()

        tmp = pd.DataFrame(
            {
                "dataset": ds_name,
                "n_annotators": df[ann_col].apply(len),
            }
        )
        rows.append(tmp)

    all_df = pd.concat(rows, ignore_index=True).dropna(subset=["n_annotators"])
    return all_df


def get_statistics_df(all_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a dataframe where each row corresponds to a dataset and each column
    is a statistic from `describe()` applied to annotator counts.
    """
    return (
        all_df.groupby("dataset")["n_annotators"]
        .describe()  # computes count, mean, std, min, 25%, 50%, 75%, max
        .rename_axis(index=None)  # optional: cleaner row index name
    )


def csv_to_latex(result_paths: list[Path], latex_output_dir: Path) -> None:
    for result_file in result_paths:
        if "sample_size" not in result_file.stem:
            dataset_name = result_file.stem.split("_")[0]
            df = pd.read_csv(result_file)
            df = df.loc[df.pvalue.notna()]
            _results_to_latex(
                res_df=df,
                output_path=latex_output_dir / f"{dataset_name}.tex",
                dataset_name=dataset_name,
                table_label=f"tab:{dataset_name}",
            )


def _results_to_latex(
    res_df: pd.DataFrame,
    output_path: Path,
    dataset_name: str,
    table_label: str,
    columns: list[str] | None = None,
) -> None:
    """
    Export results to a single LaTeX table where apunim values include
    significance stars (as superscripts), and the pvalue column is removed.
    """
    res_df = (
        res_df.replace("_", r"\_", regex=True)
        .rename(columns={"Unnamed: 1": "Value", "SDB Feature": r"\ac{pc}"})
        .set_index([r"\ac{pc}", "Value"])
    )

    if "pvalue" in res_df.columns and "apunim" in res_df.columns:
        res_df["apunim"] = res_df.apply(
            lambda r: (
                f"{r['apunim']:.4f}{significance_superscript(r['pvalue'])}"
                if not pd.isna(r["pvalue"])
                else "---"
            ),
            axis=1,
        )
        res_df = res_df.drop(columns=["pvalue"])

    if columns is None:
        columns = list(res_df.columns)

    latex_str = res_df.to_latex(
        caption=(
            "Aposteriori unimodality results for the "
            f"{dataset_name.capitalize()} dataset. "
            "Stars indicate statistical significance: "
            "*: p<0.1, **: p<0.05, ***: p<0.01."
        ),
        label=table_label,
        escape=False,  # allow LaTeX math ($^{*}$)
        columns=columns,
        position="t",
        index=True,
        float_format="%.4f",
        multirow=False,
        longtable=dataset_name == "kumar",
    )

    latex_str = center_table_latex(latex_str=latex_str)

    # Write to file
    output_path.write_text(latex_str)
    print(f"Table exported to {output_path.resolve()}")


def plot_sample_size_polarization(csv_path: Path, output_path: Path):
    df = pd.read_csv(csv_path)

    _, ax = plt.subplots()

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
    graphs.save_plot(output_path)
    plt.close()


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
        "--dices-small-path",
        required=True,
        help="Path to the DICES 350 annotator CSV file.",
    )
    parser.add_argument(
        "--dices-large-path",
        required=True,
        help="Path to the DICES 990 annotator CSV file.",
    )
    parser.add_argument(
        "--popquorn-path",
        required=True,
        help=("Path to the POPQUORN offensiveness dataset."),
    )
    parser.add_argument(
        "--sap-path",
        required=True,
        help=("Path to the Sap dataset."),
    )
    parser.add_argument(
        "--kumar-path",
        required=True,
        help=("Path to the Kumar dataset."),
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
        sap_path=Path(args.sap_path),
        dices_small_path=Path(args.dices_small_path),
        dices_large_path=Path(args.dices_large_path),
        kumar_path=Path(args.kumar_path),
        popquorn_offensiveness_path=Path(args.popquorn_path),
        latex_output_dir=Path(args.latex_output_dir),
        graph_output_dir=Path(args.graph_output_dir),
    )
