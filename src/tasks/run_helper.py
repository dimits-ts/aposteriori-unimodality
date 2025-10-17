from pathlib import Path
import re

import pandas as pd
from tqdm.auto import tqdm

from ..apunim import aposteriori
from . import preprocessing, graphs


def run_experiments_on_dataset(
    ds: preprocessing.Dataset,
    table_label: str,
    latex_output_dir: Path,
    graph_path: Path,
) -> None:
    dataset_first_name = ds.get_name().split()[0].lower()

    # full table
    res = run_all_results(
        df=ds.get_dataset(),
        sdb_columns=ds.get_sdb_columns(),
        value_col=ds.get_annotation_column(),
        comment_key_col=ds.get_comment_key_column(),
    )
    print(res)
    results_to_latex(
        res,
        output_path=latex_output_dir / f"{dataset_first_name}.tex",
        dataset_name=dataset_first_name,
        table_label=table_label,
        two_column=True,
    )

    # apunim-only table
    results_to_latex(
        res,
        output_path=Path(
            latex_output_dir / f"{dataset_first_name}_apunim_only.tex"
        ),
        dataset_name=ds.get_name(),
        table_label=table_label + r"_apunim_only",
        columns=["apunim"],
        two_column=False,
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

    # Validate expected structure
    if not (
        isinstance(res, dict)
        and set(res.keys()) == {"apunim", "p_param", "p_nonparam"}
        and all(isinstance(v, dict) for v in res.values())
    ):
        raise ValueError(
            "Unexpected result format from _run_aposteriori. "
            "Expected a dict with keys {'apunim', 'p_param', "
            "'p_nonparam'}, "
            "each mapping to a dict[FactorType, float]."
        )

    # Convert to DataFrame
    res_df = pd.DataFrame(
        {
            "apunim": pd.Series(res["apunim"]),
            "p_param": pd.Series(res["p_param"]),
            "p_nonparam": pd.Series(res["p_nonparam"]),
        }
    )

    return res_df


def results_to_latex(
    res_df: pd.DataFrame,
    output_path: Path,
    dataset_name: str,
    table_label: str,
    columns: list[str] | None = None,
    two_column: bool = False,
) -> None:
    # Replace underscores for LaTeX compatibility
    res_df = res_df.replace("_", r"\_")

    # Generate LaTeX string (don't write directly)
    latex_str = res_df.to_latex(
        longtable=False,
        caption=f"Apunim results for the {dataset_name} dataset",
        label=table_label,
        escape=True,
        columns=columns,
        position="t",
        float_format="%.4f",
    )

    if two_column:
        # 1) Turn table -> table* and add centering on begin
        latex_str = latex_str.replace(r"\begin{table}", r"\begin{table*}")
        # Ensure end is also table*
        latex_str = latex_str.replace(r"\end{table}", r"\end{table*}")
        # fill in columns to page width
        latex_str = re.sub(
            r"\\begin\{tabular\}\{([^}]+)\}",
            r"\\centering\\begin{tabular*}{\\textwidth}"
            r"{@{\\extracolsep{\\fill}}\1}",
            latex_str,
        )

        # 3) Replace \end{tabular} -> \end{tabular*}
        latex_str = latex_str.replace(r"\end{tabular}", r"\end{tabular*}")

    # Write to file
    output_path.write_text(latex_str)
    print(f"Table {table_label} exported to {output_path.resolve()}")


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
    alpha: float = 0.1,
) -> dict:
    annotations, attributes, keys = _extract_annotations_and_attributes(
        df=df,
        value_col=value_col,
        feature_col=feature_col,
        comment_key_col=comment_key_col,
    )

    result_dict = aposteriori.aposteriori_unimodality(
        annotations=annotations,
        factor_group=attributes,
        comment_group=keys,
        iterations=iterations,
        alpha=alpha,
        seed=42,
    )

    return result_dict
