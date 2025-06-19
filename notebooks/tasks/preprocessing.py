from collections.abc import Sequence

import pandas as pd
import numpy as np


def find_inconsistent_rows(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    inconsistent_rows = []

    for index, row in df.iterrows():
        list_lengths = [len(row[col]) for col in columns]
        # check if any list has different length
        inconsistent_rows.append(len(set(list_lengths)) > 1)

    return np.array(inconsistent_rows)
