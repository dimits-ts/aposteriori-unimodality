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
        return (
            [
                "Age",
                "Gender",
                "Sexual Orientation",
                "Ethnicity",
                "Employment",
                "annot_education_level",
                "Political Affiliation",
            ],
        )

    def get_comment_key_column(self) -> str:
        return "comment_key"

    def get_annotation_column(self) -> str:
        return "Toxicity"

    @staticmethod
    def _base_df(dataset_path: Path):
        df = pd.read_csv(
            dataset_path,
            converters={"annot_personality_characteristics": ast.literal_eval},
        )
        df["toxicity"] = df.annotation.apply(lambda x: x[-1]).astype(int)
        df["annot_politics"] = df.annot_personality_characteristics.apply(
            lambda x: x[0]
        )
        df.annot_age = pd.cut(df.annot_age, bins=4)
        df.message_id = df.message_id.astype(str)
        df["comment_key"] = df.conv_id + df.message_id

        df = df.loc[
            :,
            [
                "conv_id",
                "message_id",
                "comment_key",
                "message",
                "toxicity",
                "annot_age",
                "annot_sex",
                "annot_sexual_orientation",
                "annot_demographic_group",
                "annot_current_employment",
                "annot_education_level",
                "annot_politics",
            ],
        ]
        df = df.groupby(
            ["conv_id", "message_id", "comment_key", "message"]
        ).apply(
            lambda x: pd.Series(
                {
                    col: x[col].tolist()
                    for col in df.columns
                    if col
                    not in ["conv_id", "message_id", "comment_key", "message"]
                }
            ),
            include_groups=False,
        )
        df["random"] = preprocessing.get_rand_col(df, "annot_sex")
        df = df.reset_index()

        df = df.rename(
            columns={
                "annot_age": "Age",
                "annot_sex": "Gender",
                "annot_sexual_orientation": "Sexual Orientation",
                "annot_demographic_group": "Ethnicity",
                "annot_current_employment": "Employment",
                "annot_education_level": "Education",
                "annot_politics": "Political Affiliation",
                "toxicity": "Toxicity"
            }
        )
        return df


def main(dataset_path: Path, latex_output_dir: Path, graph_output_dir: Path):
    ds = HundredDataset(dataset_path=dataset_path)
    run_helper.run_experiments_on_dataset(
        ds,
        full_latex_path=latex_output_dir / "res_synthetic_100.tex",
        random_latex_path=latex_output_dir / "random_res_synthetic_100.tex",
        graph_path=graph_output_dir / "synthetic_100.png"
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
        output_dir=Path(args.latex_output_dir),
        graph_path=Path(args.graph_output_dir)
    )
