import pandas as pd

from .tasks import preprocessing
from .tasks import run_helper


def base_df():
    df = pd.read_pickle("data/attitudes_embedded.csv")
    df = df.loc[
        :,
        [
            "tweet",
            "racism",
            "annotatorAge",
            "annotatorRace",
            "annotatorGender",
        ],
    ]
    all_ages = [
        age
        for sublist in df["annotatorAge"]
        if isinstance(sublist, (list, tuple))
        for age in sublist
        if pd.notna(age)
    ]
    all_ages = list(map(int, all_ages))
    bin_edges = [0, 20, 40, 60, 80]

    df["annotatorAge"] = df["annotatorAge"].apply(
        lambda x: _process_age_list(x, bin_edges)
    )
    df.annotatorRace = df.annotatorRace.apply(
        lambda x: None if ("na" in x) else x
    )

    df.annotatorGender = df.annotatorGender.apply(
        lambda x: None if ("na" in x) else x
    )
    df = df.dropna()
    return df


def _process_age_list(x, bins):
    if not isinstance(x, (list, tuple)):
        return None
    if any(pd.isna(age) for age in x):
        return None
    try:
        int_ages = [int(age) for age in x]
        return pd.cut(int_ages, bins=bins, include_lowest=True)
    except Exception:
        return None


def main():
    df = base_df()
    df["random"] = preprocessing.get_rand_col(df, "annotatorAge")

    sdb_columns = ["annotatorAge", "annotatorRace", "annotatorGender"]
    res = run_helper.run_all_results(
        df=df,
        sdb_columns=sdb_columns,
        value_col="racism",
        comment_key_col="tweet",
    )
    print(res)

    rand_res = run_helper.run_result(
        df,
        sdb_column="random",
        value_col="racism",
        comment_key_col="tweet",
    )
    print(rand_res)


if __name__ == "__main__":
    main()
