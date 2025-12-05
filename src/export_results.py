import argparse
from pathlib import Path

import pandas as pd

from .tasks import run_helper


def main(results_dir: Path, latex_output_dir: Path, graph_output_dir: Path):
    for result_file in results_dir.rglob("*.csv"):
        dataset_name = result_file.stem
        df = pd.read_csv(result_file)
        run_helper.results_to_latex(
            res_df=df,
            output_path=latex_output_dir / f"{dataset_name}.tex",
            dataset_name=dataset_name,
            table_label=f"tab::{dataset_name}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Create graphs and latex tables from results."
        )
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
