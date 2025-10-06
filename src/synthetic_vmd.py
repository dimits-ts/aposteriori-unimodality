import argparse
import ast
from pathlib import Path

import pandas as pd

from .tasks import preprocessing
from .tasks import run_helper


class VMDDataset(preprocessing.Dataset):
    def __init__(self, dataset_path: Path):
        self.df = VMDDataset._base_df(dataset_path)

    def get_name(self) -> str:
        return "Virtual Moderation Dataset"

    def get_dataset(self) -> pd.DataFrame:
        return self.df

    def get_sdb_columns(self) -> list[str]:
        return (
            [
                "Age",
                "Gender",
                "Sexual Orientation",
                "Employment",
                "Education",
            ],
        )

    def get_comment_key_column(self) -> str:
        return "comment_key"

    def get_annotation_column(self) -> str:
        return "Toxicity"

    @staticmethod
    def base_df(dataset_path: Path) -> pd.DataFrame:
        syn_df = pd.read_csv(
            dataset_path,
            converters={
                "annot_personality_characteristics": ast.literal_eval,
                "Toxicity": ast.literal_eval,
                "Argument Quality": ast.literal_eval,
                "age_annot": ast.literal_eval,
                "sex_annot": ast.literal_eval,
                "sexual_orientation_annot": ast.literal_eval,
                "current_employment_annot": ast.literal_eval,
                "education_level_annot": ast.literal_eval,
            },
        )
        syn_df["comment_key"] = syn_df.message + syn_df.conv_id

        syn_df.Toxicity = syn_df.Toxicity.apply(
            lambda x: [int(tox) for tox in x]
        )

        syn_df.age_annot = syn_df.age_annot.apply(
            lambda ls: [int(x) for x in ls]
        ).apply(lambda x: pd.cut(x, bins=4))
        syn_df["random"] = preprocessing.get_rand_col(syn_df, "sex_annot")

        syn_df = syn_df.rename(
            columns={
                "age_annot": "Age",
                "sex_annot": "Gender",
                "sexual_orientation_annot": "Sexual Orientation",
                "current_employment_annot": "Employment",
                "education_level_annot": "Education"
            }
        )
        return syn_df


def main(dataset_path: Path, latex_output_dir: Path, graph_output_dir: Path):
    ds = VMDDataset(dataset_path=dataset_path)
    run_helper.run_experiments_on_dataset(
        ds=ds,
        full_latex_path=latex_output_dir / "res_synthetic_vmd.tex",
        random_latex_path=latex_output_dir / "random_res_synthetic_vmd.tex",
        graph_path=graph_output_dir / "synthetic_vmd.png"
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
