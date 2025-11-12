import argparse
from pathlib import Path

import pandas as pd

from .tasks import preprocessing
from .tasks import run_helper


class DicesDataset(preprocessing.Dataset):
    def __init__(self, dataset_path: Path, variant: str):
        self.df = DicesDataset._base_df(dataset_path)
        self.variant = variant

    def get_name(self) -> str:
        return "DICES-" + self.variant

    def get_dataset(self) -> pd.DataFrame:
        return self.df

    def get_sdb_columns(self) -> list[str]:
        return ["Gender", "Race", "Age", "Education"]

    def get_comment_key_column(self) -> str:
        return "item_id"

    def get_annotation_column(self) -> str:
        return "is_harmful"

    @staticmethod
    def _base_df(dataset_path: Path) -> pd.DataFrame:
        df = pd.read_csv(dataset_path)
        df = df.loc[
            :,
            [
                "rater_gender",
                "rater_age",
                "rater_race",
                "rater_education",
                "Q_overall",
                "item_id",
            ],
        ]
        df.Q_overall = df.Q_overall.map(
            {"No": -1, "Unsure": "0", "Yes": 1}
        ).astype(int)
        df = df.groupby("item_id").agg(list).reset_index()
        df = df.rename(
            columns={
                "rater_gender": "Gender",
                "rater_age": "Age",
                "rater_race": "Race",
                "rater_education": "Education",
                "Q_overall": "is_harmful",
            }
        )
        return df


def main(
    dataset_path_small: Path,
    dataset_path_large: Path,
    latex_output_dir: Path,
    graph_output_dir: Path,
):
    ds = DicesDataset(dataset_path=dataset_path_small, variant="350")
    run_helper.run_experiments_on_dataset(
        ds,
        latex_output_dir=latex_output_dir,
        graph_path=graph_output_dir / "dices-350.png",
        table_label="tab:kumar",
    )

    ds = DicesDataset(dataset_path=dataset_path_large, variant="990")
    run_helper.run_experiments_on_dataset(
        ds,
        latex_output_dir=latex_output_dir,
        graph_path=graph_output_dir / "dices-990.png",
        table_label="tab:kumar",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Classify forum comments using taxonomy categories and an LLM."
        )
    )
    parser.add_argument(
        "--dataset-path-small",
        required=True,
        help="Path to the DICES-350 CSV file.",
    )
    parser.add_argument(
        "--dataset-path-large",
        required=True,
        help="Path to the DICES-990 CSV file.",
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
        dataset_path_small=Path(args.dataset_path_small),
        dataset_path_large=Path(args.dataset_path_large),
        latex_output_dir=Path(args.latex_output_dir),
        graph_output_dir=Path(args.graph_output_dir),
    )
