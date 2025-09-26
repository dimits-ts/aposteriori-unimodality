import pandas as pd
import numpy as np


def find_inconsistent_rows(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    inconsistent_rows = []

    for index, row in df.iterrows():
        list_lengths = [len(row[col]) for col in columns]
        # check if any list has different length
        inconsistent_rows.append(len(set(list_lengths)) > 1)

    return np.array(inconsistent_rows)


def get_rand_col(
    df: pd.DataFrame, sample_annot_col: str, num_bins: int = 4
) -> pd.Series:
    return df[sample_annot_col].apply(
        lambda x: (
            [np.random.randint(1, num_bins + 1) for _ in range(len(x))]
            if x is not None
            else []
        )
    )
