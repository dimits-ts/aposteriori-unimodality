import argparse
from os import remove
from pathlib import Path

import pandas as pd

from .tasks import preprocessing, run_helper, graphs

NUM_COMMENTS = 20_000


class KumarDataset(preprocessing.Dataset):
    def __init__(
        self,
        dataset_path: Path,
        num_samples: int | None = None
    ):
        self.df = KumarDataset._base_df(
            dataset_path, num_samples
        )

    def get_name(self) -> str:
        return "Kumar et al. 2021"

    def get_dataset(self) -> pd.DataFrame:
        return self.df

    def get_sdb_columns(self) -> list[str]:
        return [
            "Gender",
            "Ethnicity",
            "Age",
            "Education",
            "Sexual Orientation",
            "Is Transgender",
            "Political Affiliation",
            "Is Parent",
            "Technology Impact",
            "Toxicity Problem",
            "Religion Important",
            "Seen Toxicity",
            "Has Been Targeted",
        ]

    def get_comment_key_column(self) -> str:
        return "comment"

    def get_annotation_column(self) -> str:
        return "Toxicity"

    @staticmethod
    def _base_df(
        dataset_path: Path,
        num_samples: int | None
    ) -> pd.DataFrame:
        df = pd.read_json(dataset_path, lines=True)
        df = df.explode(column="ratings")
        df = df.dropna()

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
        # define ranking from most to least qualified
        ranking = [
            "Doctoral degree",
            "Professional degree",
            "Master's degree",
            "Bachelor's degree",
            "Associate degree",
            "College, no degree",
            "High School graduate",
            "No high school",
        ]

        # create a mapping with ordinal prefix: 1), 2), 3)...
        ordinal_map = {
            name: f"{i+1}) {name}" for i, name in enumerate(ranking)
        }

        # apply the new labels
        df["education"] = df["education"].replace(ordinal_map)

        df = df.replace(
            {
                "Very important": "4) Very",
                "Somewhat important": "3) Somewhat",
                "Not too important": "2) Not very",
                "Not important": "1) No",
            }
        )
        df = df.replace(
            {
                "Very frequently a problem": "5) Very Frequently",
                "Frequently a problem": "4) Frequently",
                "Occasionally a problem": "3) Occasionally",
                "Rarely a problem": "2) Rarely",
                "Not a problem": "1) Never",
            }
        )
        df = df.replace(
            {
                "Very positive": "5) Very positive",
                "Somewhat positive": "4) Somewhat positive",
                # wtf?
                "Neutral \u00e2\u0080\u0093 neither positive nor negative": "3) Neutral",
                "Somewhat negative": "2) Somewhat negative",
                "Very negative": "1) Very negative",
            }
        )

        age_ranking = [
            "18 - 24",
            "25 - 34",
            "35 - 44",
            "45 - 54",
            "55 - 64",
            "65 or older",
        ]
        age_ordinal_map = {
            name: f"{i+1}) {name}" for i, name in enumerate(age_ranking)
        }
        df.age_range = df.age_range.replace(age_ordinal_map)

        df = df.loc[
            :,
            [
                "comment",
                "toxic_score",
                "gender",
                "race",
                "personally_seen_toxic_content",
                "personally_been_target",
                "identify_as_transgender",
                "toxic_comments_problem",
                "education",
                "age_range",
                "lgbtq_status",
                "political_affilation",  # sic
                "is_parent",
                "religion_important",
                "technology_impact",
            ],
        ]
        df.race = df.race.apply(KumarDataset._simplify_ethnicity)
        df = df.groupby("comment").agg(list)

        if num_samples is not None:
            print(f"Selecting {num_samples} out of {len(df)} total comments.")
            df = df.sample(num_samples, random_state=42)

        df = df.reset_index()

        df = df.rename(
            columns={
                "personally_seen_toxic_content": "Seen Toxicity",
                "personally_been_target": "Has Been Targeted",
                "identify_as_transgender": "Is Transgender",
                "toxic_comments_problem": "Toxicity Problem",
                "education": "Education",
                "age_range": "Age",
                "lgbtq_status": "Sexual Orientation",
                "political_affilation": "Political Affiliation",
                "is_parent": "Is Parent",
                "religion_important": "Religion Important",
                "toxic_score": "Toxicity",
                "gender": "Gender",
                "race": "Ethnicity",
                "technology_impact": "Technology Impact",
            }
        )
        return df

    @staticmethod
    def _simplify_ethnicity(x):
        if isinstance(x, list):
            # If your field is a list (after aggregation)
            x = x[0]

        if pd.isna(x):
            return "Unknown"

        if "," in x:
            return "Multiracial"

        mapping = {
            "Asian": "Asian",
            "Black or African American": "Black",
            "Hispanic": "Hispanic",
            "White": "White",
            "Other": "Other",
            "Prefer not to say": "Unknown",
        }
        return mapping.get(x, "Other")


def ordinal_to_yn_neutral(lst):
    new_lst = []
    for x in lst:
        # extract the numeric prefix
        try:
            num = int(x.split(")")[0])
        except:
            num = 3  # fallback
        if num == 3:
            new_lst.append("Neutral")
        elif num > 3:
            new_lst.append("Yes")
        else:
            new_lst.append("No")
    return new_lst


def map_age_list(age_map, lst):
    return [age_map[x] for x in lst]


def main(dataset_path: Path, output_dir: Path, graph_output_dir: Path):
    graphs.graph_setup()

    print("Generating sample polarization plot...")
    ds = KumarDataset(
        dataset_path=dataset_path,
        num_samples=NUM_COMMENTS
    )
    graphs.polarization_plot(
        ds=ds, output_path=graph_output_dir / "kumar_sample.png"
    )
    print("Running experiment...")
    res = run_helper.run_all_results(ds)
    res.to_csv(output_dir / "kumar.csv")


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
