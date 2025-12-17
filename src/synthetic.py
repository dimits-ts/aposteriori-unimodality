import argparse
from pathlib import Path

import pandas as pd

from .tasks import preprocessing
from .tasks import run_helper
from .tasks import graphs


class HundredDataset(preprocessing.Dataset):
    def __init__(self, dataset_path: Path):
        self.df = HundredDataset._base_df(dataset_path)

    def get_name(self) -> str:
        return "100 Annotator Synthetic"

    def get_dataset(self) -> pd.DataFrame:
        return self.df

    def get_sdb_columns(self) -> list[str]:
        return [
            "Age",
            "Gender",
            "Sexual Orientation",
            "Education",
            "Politics",
        ]

    def get_comment_key_column(self) -> str:
        return "text"

    def get_annotation_column(self) -> str:
        return "Toxicity"

    @staticmethod
    def _base_df(dataset_path: Path) -> pd.DataFrame:
        df = pd.read_csv(dataset_path)
        df.age = df.age.apply(HundredDataset._map_generation)

        df = df.rename(
            columns={
                "age": "Age",
                "sex": "Gender",
                "annotation": "Toxicity",
                "education_level": "Education",
                "political_affiliation": "Politics",
                "sexual_orientation": "Sexual Orientation",
            }
        )
        grouped_df = df.groupby("text", as_index=False).agg(list)

        return pd.DataFrame(grouped_df)

    # dry violation ;)))))))))))))))))))))))
    @staticmethod
    def _map_generation(age):
        # reference year: 2025
        if age < 29:  # Gen Z
            return "3) Gen. Z"
        elif age < 45:  # Millennial
            return "2) Millennial"
        else:  # Gen X / Boomers
            return "1) Gen. X+"


def main(dataset_path: Path, output_dir: Path, graph_output_dir: Path):
    ds = HundredDataset(dataset_path=dataset_path)
    graphs.polarization_plot(
        ds=ds, output_path=graph_output_dir / "hundred.png"
    )
    res = run_helper.run_all_results(ds)
    res.to_csv(output_dir / "hundred.csv")


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
