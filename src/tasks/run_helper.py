from pathlib import Path
import re

import pandas as pd
from tqdm.auto import tqdm
import apunim

from . import preprocessing


def run_all_results(ds: preprocessing.Dataset) -> pd.DataFrame:
    """
    Runs tasks.run_helper.results for each sdb_column and combines the results
    into a single MultiIndex DataFrame.

    Parameters
    ----------
    ds: The dataset

    Returns
    -------
    pd.DataFrame
        A hierarchical DataFrame where the first index is sdb_column,
        the second index are the factors within that column,
        and the columns are `kappa` and `pvalue`.
    """
    results = []
    for sdb_column in tqdm(
        ds.get_sdb_columns(), desc="Evaluating SDB dimensions"
    ):
        res = _run_aposteriori(
            ds.get_dataset(),
            feature_col=sdb_column,
            value_col=ds.get_annotation_column(),
            comment_key_col=ds.get_comment_key_column(),
        )
        res_df = pd.DataFrame.from_dict(
            {k: v._asdict() for k, v in res.items()},
            orient="index",
        )
        res_df.index.name = sdb_column
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
    res_df: pd.DataFrame,
    output_path: Path,
    dataset_name: str,
    table_label: str,
    columns: list[str] | None = None,
    two_column: bool = False,
    small_fontsize: bool = True,
) -> None:
    """
    Export results to a single LaTeX table where apunim values include
    significance stars (as superscripts), and the pvalue column is removed.
    """
    res_df = (
        res_df.replace("_", r"\_", regex=True)
        .rename(columns={"Unnamed: 1": "Value"})
        .set_index(["SDB Feature", "Value"])
    )

    if "pvalue" in res_df.columns and "apunim" in res_df.columns:
        res_df["apunim"] = res_df.apply(
            lambda r: (
                f"{r['apunim']:.4f}{significance_superscript(r['pvalue'])}"
                if not pd.isna(r["pvalue"])
                else "---"
            ),
            axis=1,
        )
        res_df = res_df.drop(columns=["pvalue"])

    if columns is None:
        columns = list(res_df.columns)

    latex_str = res_df.to_latex(
        longtable=False,
        caption=(
            f"Aposteriori unimodality results for the {dataset_name} "
            "dataset."
        ),
        label=table_label,
        escape=False,  # allow LaTeX math ($^{*}$)
        columns=columns,
        position="ht",
        index=True,
        float_format="%.4f",
        multirow=False
    )

    # Small font
    if small_fontsize:
        latex_str = latex_str.replace(
            r"\begin{table}[ht]",
            r"\begin{table}[ht]\centering",
        )

    # Two-column layout support
    if two_column:
        latex_str = latex_str.replace(r"\begin{table}", r"\begin{table*}")
        latex_str = latex_str.replace(r"\end{table}", r"\end{table*}")
        latex_str = re.sub(
            r"\\begin\{tabular\}\{([^}]+)\}",
            r"\\centering\\begin{tabular*}{\\textwidth}"
            r"{@{\\extracolsep{\\fill}}\1}",
            latex_str,
        )
        latex_str = latex_str.replace(r"\end{tabular}", r"\end{tabular*}")

    # Write to file
    output_path.write_text(latex_str)
    print(f"Table exported to {output_path.resolve()}")


def _extract_annotations_and_attributes(
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


def _run_aposteriori(
    df: pd.DataFrame,
    value_col: str,
    feature_col: str,
    comment_key_col: str,
    iterations: int = 100,
    alpha: float = 0.05,
) -> dict[str, apunim.ApunimResult]:
    annotations, attributes, keys = _extract_annotations_and_attributes(
        df=df,
        value_col=value_col,
        feature_col=feature_col,
        comment_key_col=comment_key_col,
    )

    results = apunim.aposteriori_unimodality(
        annotations=annotations,
        factor_group=attributes,
        comment_group=keys,
        iterations=iterations,
        alpha=alpha,
        seed=42,
    )

    return results


def significance_superscript(p):
    if pd.isna(p):
        return ""
    elif p < 0.001:
        return r"$^{***}$"
    elif p < 0.01:
        return r"$^{**}$"
    elif p < 0.05:
        return r"$^{*}$"
    else:
        return ""
