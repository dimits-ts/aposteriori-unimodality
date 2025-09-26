import ast

import pandas as pd

from .tasks import preprocessing
from .tasks import run_helper


def base_df():
    syn_df = pd.read_csv(
        "../data/synthetic.csv",
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


def main():
    df = base_df()
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

    rand_res = run_helper.run_result(
        df,
        sdb_column="random",
        value_col="Toxicity",
        comment_key_col="comment_key",
    )
    print(rand_res)


if __name__ == "__main__":
    main()
