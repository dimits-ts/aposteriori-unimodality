import argparse
from pathlib import Path

import pandas as pd

from .tasks import preprocessing
from .tasks import run_helper

NUM_COMMENTS = 500


class KumarDataset(preprocessing.Dataset):
    def __init__(self, dataset_path: Path, num_samples: int):
        self.df = KumarDataset._base_df(dataset_path, num_samples)

    def get_name(self) -> str:
        return "Kumar et al. 2021"

    def get_dataset(self) -> pd.DataFrame:
        return self.df

    def get_sdb_columns(self) -> list[str]:
        return [
            "Seen Toxicity",
            "Has Been Targeted",
            "Is Transgender",
            "Thinks Toxicity Is Problem",
            "Education",
            "Age",
            "Sexual Orientation",
            "Political Affiliation",
            "Is Parent",
            "Thinks Religion Is Important",
        ]

    def get_comment_key_column(self) -> str:
        return "comment"

    def get_annotation_column(self) -> str:
        return "Toxicity"

    @staticmethod
    def _base_df(dataset_path: Path, num_samples: int) -> pd.DataFrame:
        df = pd.read_json(dataset_path, lines=True)
        df = df.explode(column="ratings")

        ratings_df = pd.json_normalize(df.ratings)
        df = pd.concat([df.reset_index(), ratings_df.reset_index()], axis=1)
        df = df.drop(columns=["ratings", "index"])
        # shorten names
        df = df.replace(
            {
                (
                    "High school graduate (high school diploma or equivalent "
                    "including GED)"
                ): "High School graduate",
                "Associate degree in college (2-year)": "Associate degree",
                "Bachelor's degree in college (4-year)": "Bachelor's degree",
                "Less than high school degree": "No high school",
                "Professional degree (JD, MD)": "Professional degree",
                "Some college but no degree": "College, no degree",
            }
        )

        df = df.loc[
            :,
            [
                "comment",
                "toxic_score",
                "personally_seen_toxic_content",
                "personally_been_target",
                "identify_as_transgender",
                "toxic_comments_problem",
                "education",
                "age_range",
                "lgbtq_status",
                "political_affilation",
                "is_parent",
                "religion_important",
            ],
        ]
        df = df.groupby("comment").agg(list)
        print(f"Selecting {num_samples} out of {len(df)} total comments.")
        df = df.sample(num_samples)
        df = df.reset_index()
        df["random"] = preprocessing.get_rand_col(df, "education")

        df = df.rename(
            columns={
                "personally_seen_toxic_content": "Seen Toxicity",
                "personally_been_target": "Has Been Targeted",
                "identify_as_transgender": "Is Transgender",
                "toxic_comments_problem": "Thinks Toxicity Is Problem",
                "education": "Education",
                "age_range": "Age",
                "lgbtq_status": "Sexual Orientation",
                "political_affilation": "Political Affiliation",
                "is_parent": "Is Parent",
                "religion_important": "Thinks Religion Is Important",
                "toxic_score": "Toxicity",
            }
        )
        return df


def main(dataset_path: Path, latex_output_dir: Path, graph_output_dir: Path):
    ds = KumarDataset(dataset_path=dataset_path, num_samples=NUM_COMMENTS)

    run_helper.run_experiments_on_dataset(
        ds=ds,
        full_latex_path=latex_output_dir / "res_kumar.tex",
        random_latex_path=latex_output_dir / "random_res_kumar.tex",
        graph_path=graph_output_dir / "kumar.png"
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
        graph_output_dir=Path(args.graph_output_dir)
    )
