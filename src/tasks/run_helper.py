from pathlib import Path

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from ..apunim import aposteriori
from . import preprocessing, graphs


def run_experiments_on_dataset(
    ds: preprocessing.Dataset,
    full_latex_path: Path,
    random_latex_path: Path,
    graph_path: Path,
) -> None:
    res = run_all_results(
        df=ds.get_dataset(),
        sdb_columns=ds.get_sdb_columns(),
        value_col=ds.get_annotation_column(),
        comment_key_col=ds.get_comment_key_column(),
    )
    print(res)
    results_to_latex(
        res,
        output_path=full_latex_path,
        dataset_name=ds.get_name(),
    )

    rand_res = run_result(
        df=ds.get_dataset(),
        sdb_column="random",
        value_col=ds.get_annotation_column(),
        comment_key_col=ds.get_comment_key_column(),
    )
    print(rand_res)
    results_to_latex(
        rand_res,
        output_path=random_latex_path,
        dataset_name=f"random_{ds.get_name()}",
    )

    graphs.polarization_plot(ds=ds, output_path=graph_path)
    print(f"Finished {ds.get_name()} dataset.")


def run_all_results(
    df: pd.DataFrame,
    sdb_columns: list[str],
    value_col: str,
    comment_key_col: str,
) -> pd.DataFrame:
    """
    Runs tasks.run_helper.results for each sdb_column and combines the results
    into a single MultiIndex DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    sdb_columns : list
        List of sdb_column names to analyze.
    discussion_id_col : str
        Column name representing the discussion ID.
    value_col : str
        Column name with the value (e.g., toxic_score).
    comment_key_col : str
        Column name with the comment key.

    Returns
    -------
    pd.DataFrame
        A hierarchical DataFrame where the first index is sdb_column,
        the second index are the factors within that column,
        and the columns are `kappa` and `pvalue`.
    """
    results = []
    for sdb_column in tqdm(sdb_columns, desc="Evaluating SDB dimensions"):
        res_df = run_result(
            df,
            sdb_column=sdb_column,
            value_col=value_col,
            comment_key_col=comment_key_col,
        )
        # Ensure index is named (factor values)
        res_df.index.name = sdb_column
        # Add a column to store which sdb_column this came from
        res_df["SDB Feature"] = sdb_column
        results.append(res_df)

    # Concatenate all results and build a MultiIndex
    combined_df = pd.concat(results)
    combined_df.set_index("SDB Feature", append=True, inplace=True)
    combined_df = combined_df.reorder_levels(
        ["SDB Feature", combined_df.index.names[0]]
    )
    combined_df.sort_index(inplace=True)

    return combined_df


def results_to_latex(
    res_df: pd.DataFrame, output_path: Path, dataset_name: str
) -> None:
    # this should be done automatically but pandas is having a stroke
    res_df = res_df.replace("_", r"\_")

    export_name = dataset_name.split()[0].lower()
    table_name = f"tab:results_{export_name}"
    res_df.to_latex(
        buf=output_path,
        longtable=True,
        caption=(
            "Aposteriori Unimodality kappa and pvalue results "
            f"for the {dataset_name} dataset"
        ),
        label=table_name,
        escape=True,
    )
    print(f"Table {table_name} exported to {output_path.resolve()}")


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


def run_result(
    df: pd.DataFrame,
    sdb_column: str,
    value_col: str,
    comment_key_col: str,
) -> pd.DataFrame:
    res = _run_aposteriori(
        df,
        feature_col=sdb_column,
        value_col=value_col,
        comment_key_col=comment_key_col,
    )

    res_df = pd.DataFrame(res).T
    res_df = res_df.rename(columns={0: "kappa", 1: "pvalue"})
    return res_df


def _run_aposteriori(
    df: pd.DataFrame,
    value_col: str,
    feature_col: str,
    comment_key_col: str,
    bins: int = -1,
    iterations: int = 100,
    alpha: float = 0.1,
) -> dict:
    if bins == -1:
        bins = len(np.unique(np.concatenate(df[value_col])))

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
        alpha=alpha,
    )

    return result_dict
