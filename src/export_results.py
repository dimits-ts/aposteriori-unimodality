import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

        # Must have SDB Feature column
        if "SDB Feature" not in df.columns:
            continue

        # Detect the ordinal column (values like "1)", "2)", ...)
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

        # Work groupwise over SDB Feature
        for feature_name, df_group in df.groupby("SDB Feature"):

            # Keep only ordinal rows
            g = df_group[
                df_group[ordinal_col].astype(str).str.match(r"^\d+\)")
            ].copy()
            g = g[g.pvalue.notna()]

            if g.empty:
                continue

            # Exclude features where *all* pvalues > 0.5
            if (g.pvalue > 0.05).all():
                continue

            # Extract ordinal number
            g["ordinal"] = (
                g[ordinal_col].astype(str).str.extract(r"^(\d+)").astype(int)
            )

            # Add all valid rows
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

        # normalize ordinal sequence to 1...max_points
        stretched_x = np.linspace(1, max_points, n_rows)
        df_feat["stretched_ordinal"] = stretched_x
        stretched_records.append(df_feat)

    data_stretched = pd.concat(stretched_records, ignore_index=True)

    # --- Plot ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(
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
    )

    plt.title(
        "Ordinal trends (only features with p ≤ 0.5 in at least one factor)"
    )
    plt.xlabel("Order (normalized)")
    plt.ylabel("Apunim value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    graphs.save_plot(graph_output_dir / "apunim_ordinal.png")


def main(results_dir: Path, latex_output_dir: Path, graph_output_dir: Path):
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
