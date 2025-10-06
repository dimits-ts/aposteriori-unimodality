import argparse
from pathlib import Path

import pandas as pd

from .tasks import preprocessing
from .tasks import run_helper


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
    def _base_df(dataset_path: Path, num_samples: int) -> pd.DataFrame:
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
        bin_edges = [0, 20, 40, 60, 80]

        df["annotatorAge"] = df["annotatorAge"].apply(
            lambda x: SapDataset._process_age_list(x, bin_edges)
        )
        df.annotatorRace = df.annotatorRace.apply(
            lambda x: None if ("na" in x) else x
        )

        df.annotatorGender = df.annotatorGender.apply(
            lambda x: None if ("na" in x) else x
        )
        df = df.dropna()
        df["random"] = preprocessing.get_rand_col(df, "annotatorAge")

        df = df.rename(
            columns={
                "racism": "Racism",
                "annotatorAge": "Age",
                "annotatorRace": "Ethnicity",
                "annotatorGender": "Gender",
            }
        )
        return df

    @staticmethod
    def _process_age_list(x, bins):
        if not isinstance(x, (list, tuple)):
            return None
        if any(pd.isna(age) for age in x):
            return None
        try:
            int_ages = [int(age) for age in x]
            return pd.cut(int_ages, bins=bins, include_lowest=True)
        except Exception:
            return None


def main(dataset_path: Path, output_dir: Path):
    ds = SapDataset(dataset_path=dataset_path)

    run_helper.run_experiments_on_dataset(
        ds=ds,
        full_latex_path=output_dir / "res_sap.tex",
        random_latex_path=output_dir / "random_res_sap.tex",
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
    args = parser.parse_args()
    main(
        dataset_path=Path(args.dataset_path),
        output_dir=Path(args.latex_output_dir),
    )
