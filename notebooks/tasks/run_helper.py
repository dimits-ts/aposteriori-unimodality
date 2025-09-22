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

        if len(values) != len(factors):
            raise ValueError(
                f"Values {values} (length {len(values)}) \n"
                f"have different length than factors{factors} "
                f"(length {len(factors)})"
            )

        all_annotations.extend(values)
        all_attributes.extend(factors)
        # extend the key for each value in the above extracted list
        all_keys.extend([key] * len(factors))

    return all_annotations, all_attributes, all_keys


def results(
    df,
    discussion_id_col: str,
    sdb_column: str,
    value_col: str,
    comment_key_col: str,
) -> pd.DataFrame:
    res_ls = []

    # get results for each discussion
    discussion_ids = df.reset_index()[discussion_id_col].unique()
    for discussion_id in discussion_ids:
        discussion_df = df.dropna(subset=[sdb_column])
        discussion_df = discussion_df.reset_index()
        discussion_df = discussion_df[
            discussion_df[discussion_id_col] == discussion_id
        ]

        res = _run_aposteriori(
            discussion_df,
            feature_col=sdb_column,
            value_col=value_col,
            comment_key_col=comment_key_col,
        )
        res_ls.append(res)

    # each element in res_ls is a dict[FactorType, ApunimResult]
    # convert to DataFrame: MultiIndex (discussion_id, factor)
    # -> columns: value, pvalue
    all_dfs = []
    for discussion_id, res_dict in zip(discussion_ids, res_ls):
        df_disc = pd.DataFrame(
            {
                factor: {"value": r.value, "pvalue": r.pvalue}
                for factor, r in res_dict.items()
            }
        ).T
        df_disc.columns = pd.MultiIndex.from_product(
            [[discussion_id], df_disc.columns]
        )
        all_dfs.append(df_disc)

    return pd.concat(all_dfs, axis=1)


def _run_aposteriori(
    df: pd.DataFrame,
    value_col: str,
    feature_col: str,
    comment_key_col: str,
    bins: int = -1,
    iterations: int = 5000,
    alpha: float = -1
) -> dict:
    if bins == -1:
        bins = len(np.unique(df[value_col]))

    annotations, attributes, keys = extract_annotations_and_attributes(
        df=df,
        value_col=value_col,
        feature_col=feature_col,
        comment_key_col=comment_key_col,
    )

    # aposteriori_unimodality now returns dict[FactorType, ApunimResult]
    result_dict = aposteriori.aposteriori_unimodality(
        annotations=annotations,
        factor_group=attributes,
        comment_group=keys,
        bins=bins,
        iterations=iterations,
        alpha=alpha
    )

    return result_dict
