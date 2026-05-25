import argparse
from pathlib import Path

import pandas as pd
import numpy as np

import tasks.graphs
import tasks.preprocessing
import tasks.run_helper


class SapDataset(tasks.preprocessing.Dataset):
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
        df.annotatorAge = df.annotatorAge.apply(SapDataset._map_generation)
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

    @staticmethod
    def _map_generation(age_list):
        if age_list is None or not isinstance(age_list, (list, tuple)):
            return None

        gens = []
        for a in age_list:
            if pd.isna(a):
                continue
            age = int(a)

            # reference year: 2022
            if age < 26:
                gens.append("3) Gen. Z")
            elif age < 41:
                gens.append("2) Millennial")
            else:
                gens.append("1) Gen. X+")

        return gens if len(gens) > 0 else None


def main(dataset_path: Path, output_dir: Path, graph_output_dir: Path):
    tasks.graphs.graph_setup()
    ds = SapDataset(dataset_path=dataset_path)

    tasks.graphs.polarization_plot(
        ds=ds, output_path=graph_output_dir / "sap.png"
    )

    res = tasks.run_helper.compute_inherent_polarization_exhaustive(dataset=ds)
    np.save(output_dir / "sap-apriori.npy", res)

    res = tasks.run_helper.run_all_results(ds)
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
