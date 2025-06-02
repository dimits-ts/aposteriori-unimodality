import pandas as pd
import numpy as np

from src import aposteriori


def extract_annotations_and_attributes(
    df: pd.DataFrame, value_col: str, feature_col: str, comment_key_col: str
) -> tuple[list, list]:
    all_annotations = []
    all_attributes = []
    all_keys = []

    for _, row in df.iterrows():
        values = row[value_col]
        factors = row[feature_col]
        key = row[comment_key_col]
        assert len(values) == len(
            factors
        ), f"values {values} (length {len(values)}) \n"
        f"have different length than factors{factors} (length {len(factors)})"
        all_annotations.extend(values)
        all_attributes.extend(factors)
        # extend the key for each value in the above extracted list
        all_keys.extend([key] * len(factors))

    return all_annotations, all_attributes, all_keys


def run_aposteriori(
    df: pd.DataFrame,
    value_col: str,
    feature_col: str,
    comment_key_col: str,
    bins: int = -1
) -> pd.Series:
    if bins == -1:
        bins = len(np.unique(df[value_col]))

    annotations, attributes, keys = extract_annotations_and_attributes(
        df=df,
        value_col=value_col,
        feature_col=feature_col,
        comment_key_col=comment_key_col,
    )

    stat = aposteriori.aposteriori_unimodality(
        annotations=annotations,
        factor_group=attributes,
        comment_group=keys,
        bins=bins,
    )
    return pd.Series(stat)
