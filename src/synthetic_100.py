import argparse
from pathlib import Path
import ast

import pandas as pd

from .tasks import preprocessing
from .tasks import run_helper


class HundredDataset(preprocessing.Dataset):
    def __init__(self, dataset_path: Path):
        self.df = HundredDataset._base_df(dataset_path)

    def get_name(self) -> str:
        return "100 Annotator Synthetic"

    def get_dataset(self) -> pd.DataFrame:
        return self.df

    def get_sdb_columns(self) -> list[str]:
        return [
            "age",
            "gender",
            "sexual_orientation",
            "education_level",
            "political_affiliation",
        ]

    def get_comment_key_column(self) -> str:
        return "text"

    def get_annotation_column(self) -> str:
        return "hate_speech"

    @staticmethod
    def _base_df(dataset_path: Path):
        df = pd.read_csv(dataset_path)
        df = df.rename(columns={"sex": "gender", "annotation": "hate_speech"})
        grouped_df = df.groupby("text", as_index=False).agg(list)
        grouped_df.age = grouped_df.age.apply(
            lambda ls: [int(x) for x in ls]
        ).apply(lambda x: pd.cut(x, bins=4))
        return grouped_df


def main(dataset_path: Path, latex_output_dir: Path, graph_output_dir: Path):
    ds = HundredDataset(dataset_path=dataset_path)
    run_helper.run_experiments_on_dataset(
        ds,
        latex_output_dir=latex_output_dir,
        graph_path=graph_output_dir / "100.png",
        table_label="tab:synthetic_100"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Classify forum comments using taxonomy categories and an LLM."
        )
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to the full dataset CSV file.",
    )
    parser.add_argument(
        "--latex-output-dir",
        required=True,
        help="Directory for the latex result files.",
    )
    parser.add_argument(
        "--graph-output-dir",
        required=True,
        help="Directory for graphs.",
    )
    args = parser.parse_args()
    main(
        dataset_path=Path(args.dataset_path),
        latex_output_dir=Path(args.latex_output_dir),
        graph_output_dir=Path(args.graph_output_dir),
    )
