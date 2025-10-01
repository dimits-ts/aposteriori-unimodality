import argparse
from pathlib import Path
import ast

import pandas as pd

from .tasks import preprocessing
from .tasks import run_helper


def base_df(dataset_path: Path):
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
    df = df.groupby(["conv_id", "message_id", "comment_key", "message"]).apply(
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
    df = df.reset_index()
    return df


def main(dataset_path: Path, output_dir: Path):
    df = base_df(dataset_path)
    df["random"] = preprocessing.get_rand_col(df, "annot_sex")

    sdb_columns = [
        "annot_age",
        "annot_sex",
        "annot_demographic_group",
        "annot_sexual_orientation",
        "annot_current_employment",
        "annot_education_level",
        "annot_politics",
    ]
    res = run_helper.run_all_results(
        df=df,
        sdb_columns=sdb_columns,
        value_col="toxicity",
        comment_key_col="comment_key",
    )
    print(res)

    rand_res = run_helper.run_result(
        df,
        sdb_column="random",
        value_col="toxicity",
        comment_key_col="comment_key",
    )
    print(rand_res)

    run_helper.results_to_latex(
        rand_res,
        output_path=output_dir / "res_synthetic_100.tex",
        dataset_name="100 Annotator Synthetic",
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
