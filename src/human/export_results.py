import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ..lib import graphs
from ..lib.util import center_table_latex, significance_superscript

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
