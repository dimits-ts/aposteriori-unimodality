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
        run_helper.results_to_latex(
            res_df=df,
            output_path=latex_output_dir / f"{dataset_name}.tex",
            dataset_name=dataset_name,
            table_label=f"tab:{dataset_name}",
        )


def ordinal_graphs(results_dir: Path, graph_output_dir: Path) -> None:
    """
    For each CSV in results_dir, extract rows whose feature value starts with
    an ordinal like '1)', '2)', ... and has non-null pvalue.
    Then plot ordinal vs apunim for each feature, stretching all ordinal
    series to occupy the same x-axis length.
    """
    records = []

    # --- Collect all data first ---
    for file in results_dir.rglob("*.csv"):
        df = pd.read_csv(file)
        dataset = file.stem

        # find ordinal column
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

        # filter ordinal + pvalue
        d = df[df[ordinal_col].astype(str).str.match(r"^\d+\)")].copy()
        d = d[d.pvalue.notna()]
        if d.empty:
            continue

        # extract ordinal number
        d["ordinal"] = (
            d[ordinal_col].astype(str).str.extract(r"^(\d+)").astype(int)
        )

        feature_col = df.columns[0]

        # append rows to global table
        for _, row in d.iterrows():
            records.append(
                {
                    "dataset": dataset,
                    "feature": f"{dataset}-{row[feature_col]}",
                    "ordinal": row["ordinal"],
                    "apunim": row["apunim"],
                }
            )

    if not records:
        print("No usable ordinal data found.")
        return

    data = pd.DataFrame(records)

    # --- Stretch each feature's ordinal to full x-axis ---
    max_points = (
        data.groupby("feature")["ordinal"].max().max()
    )  # maximum ordinal count
    stretched_records = []
    for feature, df_feat in data.groupby("feature"):
        df_feat = df_feat.sort_values("ordinal").reset_index(drop=True)
        n_rows = len(df_feat)
        stretched_x = np.linspace(1, max_points, n_rows)
        df_feat["stretched_ordinal"] = stretched_x
        stretched_records.append(df_feat)

    data_stretched = pd.concat(stretched_records, ignore_index=True)

    # --- Plot with Seaborn ---
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

    plt.title("Ordinal trends in all datasets")
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
