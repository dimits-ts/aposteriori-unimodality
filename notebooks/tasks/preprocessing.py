import pandas as pd


def find_inconsistent_rows(df: pd.DataFrame) -> pd.DataFrame:
    inconsistent_rows = []

    for index, row in df.iterrows():
        list_lengths = [
            len(row[col]) for col in df.columns if isinstance(row[col], list)
        ]

        if len(set(list_lengths)) > 1:  # Check if lengths are inconsistent
            inconsistent_rows.append(index)

    return df.loc[inconsistent_rows]
