import argparse
from pathlib import Path

import pandas as pd

from .tasks import preprocessing
from .tasks import run_helper
from .tasks import graphs


class SapDataset(preprocessing.Dataset):
    def __init__(self, dataset_path: Path):
        self.df = SapDataset._base_df(dataset_path)

    def get_name(self) -> str:
        return "Sap et al. 2022"

    def get_dataset(self) -> pd.DataFrame:
        return self.df

    def get_sdb_columns(self) -> list[str]:
        return [
            "Age",
            "Ethnicity",
            "Gender",
        ]

    def get_comment_key_column(self) -> str:
        return "tweet"

    def get_annotation_column(self) -> str:
        return "Racism"

    @staticmethod
    def _base_df(dataset_path: Path) -> pd.DataFrame:
        df = pd.read_pickle(dataset_path)
        df = df.loc[
            :,
            [
                "tweet",
                "racism",
                "annotatorAge",
                "annotatorRace",
                "annotatorGender",
            ],
        ]
        all_ages = [
            age
            for sublist in df["annotatorAge"]
            if isinstance(sublist, (list, tuple))
            for age in sublist
            if pd.notna(age)
        ]
        all_ages = list(map(int, all_ages))

        df["annotatorAge"] = df["annotatorAge"].apply(
            lambda x: preprocessing.process_age_list(x)
        )
        df.annotatorRace = df.annotatorRace.apply(
            lambda x: None if ("na" in x) else x
        )

        df.annotatorGender = df.annotatorGender.apply(
            lambda x: None if ("na" in x) else x
        )
        df = df.dropna()

        df = df.rename(
            columns={
                "racism": "Racism",
                "annotatorAge": "Age",
                "annotatorRace": "Ethnicity",
                "annotatorGender": "Gender",
            }
        )
        return df


def main(dataset_path: Path, output_dir: Path, graph_output_dir: Path):
    ds = SapDataset(dataset_path=dataset_path)

    graphs.polarization_plot(
        ds=ds, output_path=graph_output_dir / "sap.png"
    )

    res = run_helper.run_all_results(
        df=ds.get_dataset(),
        sdb_columns=ds.get_sdb_columns(),
        value_col=ds.get_annotation_column(),
        comment_key_col=ds.get_comment_key_column(),
    )
    res.to_csv(output_dir / "sap.csv")


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
        "--output-dir",
        required=True,
        help="Directory for the CSV result files.",
    )
    parser.add_argument(
        "--graph-output-dir",
        required=True,
        help="Directory for graphs.",
    )
    args = parser.parse_args()
    main(
        dataset_path=Path(args.dataset_path),
        output_dir=Path(args.output_dir),
        graph_output_dir=Path(args.graph_output_dir),
    )
