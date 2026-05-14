import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from .tasks import run_helper
from .tasks import graphs


def main(results_dir: Path, latex_output_dir: Path, graph_output_dir: Path):
    graphs.graph_setup()
    csv_to_latex(results_dir=results_dir, latex_output_dir=latex_output_dir)
    ordinal_graph(results_dir=results_dir, graph_output_dir=graph_output_dir)
    ordinal_graph_per_feature(
        results_dir=results_dir, graph_output_dir=graph_output_dir
    )
    plot_dfu_histograms(
        file_paths=list(results_dir.rglob("*.npy")),
        graph_output_dir=graph_output_dir,
    )


def plot_dfu_histograms(
    file_paths: list[str],
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

        arr = np.load(path)
        arr = np.asarray(arr, dtype=float)
        arr = arr[~np.isnan(arr)]

        all_data.append(pd.DataFrame({"value": arr, "dataset": label}))

    full_df = pd.concat(all_data, ignore_index=True)

    fig, ax = plt.subplots(figsize=(8, 5))

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
            plt.matplotlib.patches.Patch(
                facecolor=color,
                edgecolor="black",
                hatch=hatch,
                label=dataset,
                alpha=0.4,
            )
        )

    ax.set_xlabel("Apriori polarization")
    ax.set_ylabel("Density")

    ax.legend(handles=legend_handles)

    graphs.save_plot(graph_output_dir / "apriori.png")
    plt.close()


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


def ordinal_graph_per_feature(
    results_dir: Path, graph_output_dir: Path
) -> None:
    graphs.graph_setup()

    for file in results_dir.rglob("*.csv"):
        df = pd.read_csv(file)
        dataset = file.stem

        if "SDB Feature" not in df.columns or "Unnamed: 1" not in df.columns:
            continue

        for feature_name, g in df.groupby("SDB Feature"):

            g = g[g["Unnamed: 1"].astype(str).str.match(r"^\d+\)")]
            g = g[g.apunim.notna()]
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

            graphs.save_plot(out_path)
            plt.close()


def ordinal_graph(results_dir: Path, graph_output_dir: Path) -> None:
    """
    For each CSV in results_dir:
    - Identify ordinal-valued rows grouped by the 'SDB Feature' column.
    - Keep only groups where at least two ordinal factors have pvalue <= 0.05.
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

            g["ordinal"] = (
                g[ordinal_col].astype(str).str.extract(r"^(\d+)").astype(int)
            )

            for _, row in g.iterrows():
                records.append(
                    {
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

        df_feat["stretched_ordinal"] = np.linspace(
            1,
            max_points,
            n_rows,
        )

        stretched_records.append(df_feat)

    data_stretched = pd.concat(
        stretched_records,
        ignore_index=True,
    )

    plt.figure(figsize=(12, 7))

    sns.lineplot(
        data=data_stretched,
        x="stretched_ordinal",
        y="apunim",
        hue="feature",
        style="feature",
        markers=graphs.MARKERS,
        dashes=True,
        palette=graphs.COLORBLIND_PALETTE,
        linewidth=1.8,
        markersize=7,
        estimator=None,
    )

    plt.title("Pol. attribution is generally monotonic")
    plt.xlabel(
        r"$\{Age | Opinion | Education\}: Low \rightarrow High$"
    )
    plt.ylabel("Apunim value")

    plt.legend(loc="center")

    plt.tight_layout()

    graphs.save_plot(graph_output_dir / "apunim_ordinal.png")


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
