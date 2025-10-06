import argparse
import ast
from pathlib import Path

import pandas as pd

from .tasks import preprocessing
from .tasks import run_helper

DATASET_NAME = "Virtual Moderation Dataset"


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
    syn_df["fake_index"] = 1

    syn_df.Toxicity = syn_df.Toxicity.apply(lambda x: [int(tox) for tox in x])

    syn_df.age_annot = syn_df.age_annot.apply(
        lambda ls: [int(x) for x in ls]
    ).apply(lambda x: pd.cut(x, bins=4))
    return syn_df


def main(dataset_path: Path, output_dir: Path):
    df = base_df(dataset_path)
    df["random"] = preprocessing.get_rand_col(df, "sex_annot")

    sdb_columns = [
        "age_annot",
        "sexual_orientation_annot",
        "education_level_annot",
        "current_employment_annot",
        "sex_annot",
    ]
    res = run_helper.run_all_results(
        df=df,
        sdb_columns=sdb_columns,
        value_col="Toxicity",
        comment_key_col="comment_key",
    )
    print(res)
    run_helper.results_to_latex(
        res,
        output_path=output_dir / "res_synthetic_vmd.tex",
        dataset_name=DATASET_NAME,
    )

    rand_res = run_helper.run_result(
        df,
        sdb_column="random",
        value_col="Toxicity",
        comment_key_col="comment_key",
    )
    print(rand_res)
    run_helper.results_to_latex(
        rand_res,
        output_path=output_dir / "random_res_synthetic_vmd.tex",
        dataset_name=f"random_{DATASET_NAME}",
    )
    print(f"Finished {DATASET_NAME} dataset.")


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
