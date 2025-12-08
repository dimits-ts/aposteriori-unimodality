import abc

import pandas as pd
import numpy as np


class Dataset(abc.ABC):

    def get_dataset(self) -> pd.DataFrame:
        raise NotImplementedError()

    def get_name(self) -> str:
        raise NotImplementedError()

    def get_sdb_columns(self) -> list[str]:
        raise NotImplementedError()

    def get_annotation_column(self) -> str:
        raise NotImplementedError()

    def get_comment_key_column(self) -> str:
        raise NotImplementedError()


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


def process_age_list(
    x: list[int], bins: list[int] = [0, 20, 40, 60, 80]
) -> list[int] | None:
    if not isinstance(x, (list, tuple)):
        return None
    if any(pd.isna(age) for age in x):
        return None
    int_ages = [int(age) for age in x]

    # "0–20", "20–40", etc.
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
    binned = pd.cut(int_ages, bins=bins, labels=labels, include_lowest=True)
    return list(binned)
