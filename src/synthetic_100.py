import pandas as pd
import ast

import tasks.run_helper
import tasks.preprocessing


def base_df():
    df = pd.read_csv(
        "../data/100_annotators.csv",
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


def main():
    df = base_df()
    df["random"] = tasks.preprocessing.get_rand_col(df, "annot_sex")

    sdb_columns = [
        "annot_age",
        "annot_sex",
        "annot_demographic_group",
        "annot_sexual_orientation",
        "annot_current_employment",
        "annot_education_level",
        "annot_politics",
    ]
    res = tasks.run_helper.run_all_results(
        df=df,
        sdb_columns=sdb_columns,
        value_col="toxicity",
        comment_key_col="comment_key",
    )
    print(res)

    rand_res = tasks.run_helper.run_result(
        df,
        sdb_column="random",
        value_col="toxicity",
        comment_key_col="comment_key",
    )
    print(rand_res)


if __name__ == "__main__":
    main()
