import argparse
from pathlib import Path

import pandas as pd

from .tasks import preprocessing
from .tasks import run_helper


def base_df(dataset_path: Path):
    df = pd.read_json(dataset_path, lines=True)
    df = df.explode(column="ratings")

    ratings_df = pd.json_normalize(df.ratings)
    df = pd.concat([df.reset_index(), ratings_df.reset_index()], axis=1)
    df = df.drop(columns=["ratings", "index"])

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
    df = df.head(5000)
    df = df.reset_index()
    return df


def main(dataset_path: Path, output_dir: Path):
    df = base_df(dataset_path)
    df["random"] = preprocessing.get_rand_col(df, "education")
    sdb_columns = [
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
    ]

    res = run_helper.run_all_results(
        df=df,
        sdb_columns=sdb_columns,
        value_col="toxic_score",
        comment_key_col="comment",
    )
    print(res)

    rand_res = run_helper.run_result(
        df,
        sdb_column="random",
        value_col="toxic_score",
        comment_key_col="comment",
    )
    print(rand_res)

    run_helper.results_to_latex(
        rand_res,
        output_path=output_dir / "res_kumar.tex",
        dataset_name="Kumar et al 2222",
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
