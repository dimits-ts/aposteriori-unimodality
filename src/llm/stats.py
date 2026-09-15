"""
Statistical analysis of LLM (and, for the consistency tables, cross-model)
annotation agreement and significance:

1. Cross-model / cross-variant / repeat consistency tables built on
   Krippendorff's alpha (ordinal) -- how consistent are different LLMs
   with each other, and with themselves, across prompt paraphrases and
   repeated runs.

2. Aposteriori-unimodality (apunim) results per (dataset, SDB group,
   model), pivoted into one column per instruction prompt
   (default/stereotype/persona/...) so a group's apunim value can be
   compared across prompts directly.

3. A one-way ANOVA, per (dataset, model, SDB group), testing whether that
   group's per-comment nDFU distribution shifts across instruction
   prompts -- with p-values corrected for multiple hypothesis testing
   across the whole batch (see compute_ndfu_anova_by_prompt).

All tests operate purely on the LLM annotation CSVs; none of the tables or
tests in this module compare against human annotations (see
polarization.py for the Human-vs-LLM comparison).
"""

from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ..lib.preprocessing import LazyDatasetLoader
from ..lib import run_helper
from .common import (
    MAIN_PROMPT_NAMES,
    LLMAnnotationDataset,
    _available_dataset_keys,
    _key_columns,
    _keyed_series,
    _order_models,
    find_annotation_files,
    find_repeat_files,
    load_llm_df,
    PROMPT_COMPARISON_DATASET_KEYS,
    VARIANT_NAMES,
    _compute_ndfu_records
)


# ---------------------------------------------------------------------------
# Krippendorff's-alpha consistency tables
# ---------------------------------------------------------------------------


def _build_reliability_matrix(
    dfs: dict[str, pd.DataFrame], key_cols: list[str]
) -> tuple[np.ndarray, int]:
    """
    dfs: {rater_label: dataframe}, each with `key_cols` plus
    'annotation_clean'.

    Returns
    -------
    (matrix, n_items)
        matrix has shape (n_raters, n_items) -- the layout krippendorff.alpha
        expects -- aligned on the union of keys across raters, with NaN
        where a given rater has no annotation for that item.
    """
    series_list = [
        _keyed_series(df, key_cols, label) for label, df in dfs.items()
    ]
    if not series_list:
        return np.empty((0, 0)), 0

    wide = pd.concat(series_list, axis=1)
    return wide.to_numpy(dtype=float).T, wide.shape[0]


def krippendorff_alpha_safe(matrix: np.ndarray) -> float:
    if matrix.shape[0] < 2 or matrix.shape[1] == 0:
        return np.nan
    try:
        return krippendorff.alpha(
            reliability_data=matrix, level_of_measurement="ordinal"
        )
    except (ValueError, ZeroDivisionError):
        return np.nan


def _consistency_row(
    dfs: dict[str, pd.DataFrame],
    key_cols: list[str],
    count_label: str,
    extra_fields: dict,
) -> dict:
    matrix, n_items = _build_reliability_matrix(dfs, key_cols)
    alpha = krippendorff_alpha_safe(matrix)
    return {
        **extra_fields,
        count_label: matrix.shape[0],
        "Items": n_items,
        "Krippendorff's alpha": alpha,
    }


def _cross_model_row(
    human_datasets: LazyDatasetLoader,
    annotations_dir: Path,
    key: str,
    prompt_name: str,
    exclude_models: set[str],
) -> dict | None:
    files = {
        pseudo: path
        for pseudo, path in find_annotation_files(
            annotations_dir, key, prompt_name
        ).items()
        if pseudo not in exclude_models
    }
    if len(files) < 2:
        return None

    dfs = {pseudo: load_llm_df(path) for pseudo, path in files.items()}
    key_cols = _key_columns(dfs)
    return _consistency_row(
        dfs,
        key_cols,
        "Models",
        {"Dataset": human_datasets[key].get_name(), "Prompt": prompt_name},
    )


def cross_model_consistency_table(
    human_datasets: LazyDatasetLoader,
    annotations_dir: Path,
    prompt_name: str = "default",
    exclude_models: list[str] | None = None,
) -> pd.DataFrame:
    """
    For each dataset: how consistent are the different LLMs with each
    other, when all of them are given the same (default) prompt?

    `exclude_models`, if given, drops those model pseudos (e.g. "olmo7b")
    from the comparison entirely.
    """
    exclude_models = set(exclude_models or [])
    rows = []
    for key in _available_dataset_keys(human_datasets):
        row = _cross_model_row(
            human_datasets, annotations_dir, key, prompt_name, exclude_models
        )
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def _variant_dfs_for_model(
    files_by_variant: dict[str, dict[str, Path]],
    pseudo: str,
    variant_names: list[str],
) -> dict[str, pd.DataFrame]:
    dfs = {}
    for v in variant_names:
        path = files_by_variant[v].get(pseudo)
        if path is not None:
            dfs[v] = load_llm_df(path)
    return dfs


def _variant_rows_for_dataset(
    human_datasets: LazyDatasetLoader,
    key: str,
    files_by_variant: dict[str, dict[str, Path]],
    variant_names: list[str],
) -> list[dict]:
    models = sorted(set.union(*(set(f) for f in files_by_variant.values())))
    rows = []
    for pseudo in models:
        dfs = _variant_dfs_for_model(files_by_variant, pseudo, variant_names)
        if len(dfs) < 2:
            continue
        key_cols = _key_columns(dfs)
        rows.append(
            _consistency_row(
                dfs,
                key_cols,
                "Variants",
                {"Dataset": human_datasets[key].get_name(), "Model": pseudo},
            )
        )
    return rows


def per_model_variant_consistency_table(
    human_datasets: LazyDatasetLoader,
    paraphrase_dir: Path,
    variant_names: list[str] = None,
) -> pd.DataFrame:
    """
    For each (dataset, model): how consistent is that model with itself
    across the paraphrased prompt variants (variant1/variant2/variant3)?
    """
    variant_names = variant_names or VARIANT_NAMES
    rows = []
    for key in _available_dataset_keys(human_datasets):
        files_by_variant = {
            v: find_annotation_files(paraphrase_dir, key, v)
            for v in variant_names
        }
        if not any(files_by_variant.values()):
            continue
        rows.extend(
            _variant_rows_for_dataset(
                human_datasets, key, files_by_variant, variant_names
            )
        )
    return pd.DataFrame(rows)


def _repeat_row(
    human_datasets: LazyDatasetLoader,
    key: str,
    pseudo: str,
    run_files: dict[str, Path],
) -> dict | None:
    dfs = {
        run_label: load_llm_df(path)
        for run_label, path in sorted(run_files.items())
    }
    if len(dfs) < 2:
        return None
    key_cols = _key_columns(dfs)
    return _consistency_row(
        dfs,
        key_cols,
        "Runs",
        {"Dataset": human_datasets[key].get_name(), "Model": pseudo},
    )


def _repeat_rows_for_dataset(
    human_datasets: LazyDatasetLoader,
    key: str,
    files_by_model: dict[str, dict[str, Path]],
) -> list[dict]:
    rows = []
    for pseudo, run_files in sorted(files_by_model.items()):
        row = _repeat_row(human_datasets, key, pseudo, run_files)
        if row is not None:
            rows.append(row)
    return rows


def per_model_repeat_consistency_table(
    human_datasets: LazyDatasetLoader,
    repeat_dir: Path,
    prompt_name: str = "default",
) -> pd.DataFrame:
    """
    For each (dataset, model): how consistent is that model with itself
    across repeated runs of the *same* prompt (the "-run0" .. "-runN"
    repeat ablation in output/ablations/repeat)?
    """

    rows = []
    for key in _available_dataset_keys(human_datasets):
        files_by_model = find_repeat_files(repeat_dir, key, prompt_name)
        if not files_by_model:
            continue
        rows.extend(
            _repeat_rows_for_dataset(human_datasets, key, files_by_model)
        )
    return pd.DataFrame(rows)


def export_latex_table(
    df: pd.DataFrame, output_path: Path, caption: str, label: str
) -> None:
    df = df.copy()
    if "Krippendorff's alpha" in df.columns:
        df["Krippendorff's alpha"] = df["Krippendorff's alpha"].map(
            lambda x: "---" if pd.isna(x) else f"{x:.4f}"
        )

    latex_str = df.to_latex(
        index=False,
        caption=caption,
        label=label,
        position="ht",
        escape=True,
    )
    latex_str = latex_str.replace(
        r"\begin{table}[ht]", r"\begin{table}[ht]\centering"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_str)
    print(f"Table exported to {output_path.resolve()}")


# ---------------------------------------------------------------------------
# Apunim-by-prompt LaTeX table (default / stereotype / persona columns)
# ---------------------------------------------------------------------------


def _apunim_row_for_model(
    dataset_key: str, prompt_name: str, pseudo: str, path: Path
) -> pd.DataFrame | None:
    df = load_llm_df(path)
    ds = LLMAnnotationDataset(df, dataset_key, pseudo, prompt_name)
    try:
        res_df = run_helper.run_all_results(ds).reset_index()
    except ValueError as e:
        # E.g. "No polarized comments found." -- can happen for a small/
        # sparse (dataset, prompt, model) sample. Skip just this
        # combination rather than failing the whole table.
        print(f"Skipping apunim for {dataset_key}/{prompt_name}/{pseudo}: {e}")
        return None

    # The 2nd column is the (unnamed) per-SDB-column factor level, e.g.
    # "Age" -> "1) Gen. X+"; rename positionally since its actual column
    # label depends on pandas' index-naming, not on anything we control
    # here (see run_all_results).
    res_df = res_df.rename(columns={res_df.columns[1]: "Value"})
    res_df["Model"] = pseudo
    res_df["Prompt"] = prompt_name
    return res_df


def _apunim_rows_for_prompt(
    annotations_dir: Path,
    dataset_key: str,
    prompt_name: str,
    exclude_models: set[str],
) -> list[pd.DataFrame]:
    files = find_annotation_files(annotations_dir, dataset_key, prompt_name)
    models = _order_models(set(files) - exclude_models)
    rows = []
    for pseudo in models:
        res_df = _apunim_row_for_model(
            dataset_key, prompt_name, pseudo, files[pseudo]
        )
        if res_df is not None:
            rows.append(res_df)
    return rows


def compute_llm_apunim_by_prompt(
    annotations_dir: Path,
    dataset_key: str,
    prompt_names: list[str] = MAIN_PROMPT_NAMES,
    exclude_models: set[str] | None = None,
) -> pd.DataFrame:
    """
    Runs the same apunim analysis as lib.run_helper.run_all_results (the
    pipeline sap.py/dices.py/kumar.py use to produce their "-results.csv"
    files), but over the LLM annotation CSVs, once per (prompt, model)
    available for `dataset_key`. Returns a long-format DataFrame with one
    row per (SDB Feature, Value, Model, Prompt) combination and columns
    'apunim', 'pvalue', 'support', ready to be pivoted into a table.
    """
    exclude_models = set(exclude_models or ())
    rows = []
    for prompt_name in prompt_names:
        rows.extend(
            _apunim_rows_for_prompt(
                annotations_dir, dataset_key, prompt_name, exclude_models
            )
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "SDB Feature",
                "Value",
                "Model",
                "Prompt",
                "apunim",
                "pvalue",
                "support",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def _trim_numeric_col(col):
    return pd.to_numeric(
        col,
        errors="coerce",
    ).map(lambda x: "---" if pd.isna(x) else f"{x:.3f}")


def export_llm_apunim_prompt_table(
    df: pd.DataFrame, output_path: Path, dataset_name: str, label: str
) -> None:
    if df.empty:
        print(
            f"No LLM apunim-by-prompt results for {dataset_name}; "
            f"skipping {output_path}."
        )
        return
    df = df.rename(columns={"SDB Feature": r"\ac{pc}"})

    for number_col in MAIN_PROMPT_NAMES:
        number_col = number_col.capitalize()
        df[number_col] = _trim_numeric_col(df[number_col])

    df = df.replace("_", r"\_", regex=True).set_index(
        [r"\ac{pc}", "Value", "Model"]
    )

    latex_str = df.to_latex(
        caption=(
            "Aposteriori unimodality results for the LLM annotations of "
            f"the {dataset_name} dataset, across instruction prompts."
        ),
        label=label,
        escape=False,  # allow LaTeX math ($^{*}$) already in the cells
        position="ht",
        index=True,
        multirow=True,
        longtable=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_str)
    print(f"Table exported to {output_path.resolve()}")


def _apunim_prompt_cell(row: pd.Series) -> str:
    if pd.isna(row["apunim"]):
        return "---"
    stars = run_helper.significance_superscript(row["pvalue"])
    return f"{row['apunim']:.4f}{stars}"


def build_llm_apunim_prompt_table(
    long_df: pd.DataFrame, prompt_names: list[str] = MAIN_PROMPT_NAMES
) -> pd.DataFrame:
    """
    Pivots `compute_llm_apunim_by_prompt`'s long-format output into one row
    per (SDB Feature, Value, Model) and one column per prompt, so a given
    (dataset, model, SDB group)'s apunim value can be compared across the
    default/stereotype/persona prompts directly. Cells combine the apunim
    value with its significance stars (matching
    lib.run_helper.results_to_latex); missing (prompt, model) results
    show as "---".
    """
    if long_df.empty:
        return pd.DataFrame()

    long_df = long_df.copy()
    long_df["cell"] = long_df.apply(_apunim_prompt_cell, axis=1)

    wide = long_df.pivot_table(
        index=["SDB Feature", "Value", "Model"],
        columns="Prompt",
        values="cell",
        aggfunc="first",
    )
    prompt_cols = [p for p in prompt_names if p in wide.columns]
    wide = wide.reindex(columns=prompt_cols)
    wide.columns = [c.capitalize() for c in wide.columns]
    wide = wide.fillna("---").reset_index()

    model_order = _order_models(set(wide["Model"]))
    wide["Model"] = pd.Categorical(
        wide["Model"], categories=model_order, ordered=True
    )
    return wide.sort_values(["SDB Feature", "Value", "Model"]).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# Prompt-sensitivity ANOVA on per-comment nDFU
# ---------------------------------------------------------------------------


def compute_ndfu_anova_by_prompt(
    human_datasets: LazyDatasetLoader,
    annotations_dir: Path,
    dataset_keys: list[str] = PROMPT_COMPARISON_DATASET_KEYS,
    prompt_names: list[str] = MAIN_PROMPT_NAMES,
    exclude_models: set[str] | None = None,
    correction_method: str = "fdr_bh",
) -> pd.DataFrame:
    """
    For each (dataset, model, SDB group), runs a one-way ANOVA
    (scipy.stats.f_oneway) comparing that group's per-COMMENT nDFU values
    across instruction prompts (default vs. each adversarial prompt).
    LLM-only: human annotations are never used here (see polarization.py
    for the Human-vs-LLM comparison).

    Each comment contributes AT MOST ONE observation per (prompt, PC
    Dimension) -- via common._compute_ndfu_records(..., deduped=True) --
    unlike the apunim grid's plotting path, which broadcasts a comment's
    single nDFU value once per annotator persona matching that group and
    so double(sextuple-)counts it. With N comments and up to
    MAX_ANNOTATORS_PER_ITEM annotators/comment, that broadcast would
    inflate a group's observation count from <=N to close to
    N * MAX_ANNOTATORS_PER_ITEM, both misreporting the true sample size
    and violating the ANOVA independence assumption (the same value
    appearing multiple times), inflating F-statistics and deflating
    p-values.

    Raw p-values are corrected for multiple hypothesis testing across the
    entire batch of tests (all datasets/models/groups together) using
    `correction_method` (default: Benjamini-Hochberg FDR). Set
    `correction_method=None` to skip correction and keep only the raw
    p-value.

    Returns a long-format DataFrame with one row per (Dataset, Model,
    PC Dimension) test, columns: 'Dataset', 'Model', 'PC Dimension',
    'F', 'p_raw', 'n_prompts_compared', 'n_total_obs', and -- if
    `correction_method` is set -- 'p_corrected' and 'reject_null'
    (at alpha=0.05).
    """

    exclude_models = set(exclude_models or ())
    records = []

    for dataset_key in dataset_keys:
        if dataset_key not in human_datasets:
            continue

        files_by_prompt = {
            p: find_annotation_files(annotations_dir, dataset_key, p)
            for p in prompt_names
        }
        models = _order_models(
            set.union(*(set(f) for f in files_by_prompt.values()))
            - exclude_models
        )
        dataset_name = human_datasets[dataset_key].get_name()

        for model in models:
            ndfu_by_prompt = {}
            for prompt_name in prompt_names:
                path = files_by_prompt[prompt_name].get(model)
                if path is None:
                    continue
                df = load_llm_df(path)
                ds = LLMAnnotationDataset(df, dataset_key, model, prompt_name)
                group_records = _compute_ndfu_records(ds, deduped=True)
                if not group_records.empty:
                    ndfu_by_prompt[prompt_name] = group_records.groupby(
                        "PC Dimension"
                    )["nDFU"].apply(list)

            if len(ndfu_by_prompt) < 2:
                continue  # need >=2 prompts to compare

            all_groups = sorted(
                set.union(*(set(s.index) for s in ndfu_by_prompt.values()))
            )

            for group in all_groups:
                samples = [
                    s[group]
                    for s in ndfu_by_prompt.values()
                    if group in s.index and len(s[group]) >= 2
                ]
                if len(samples) < 2:
                    continue
                f_stat, p_value = stats.f_oneway(*samples)
                records.append(
                    {
                        "Dataset": dataset_name,
                        "Model": model,
                        "PC Dimension": group,
                        "F": f_stat,
                        "p_raw": p_value,
                        "n_prompts_compared": len(samples),
                        "n_total_obs": sum(len(s) for s in samples),
                    }
                )

    result_df = pd.DataFrame(records)
    if result_df.empty or correction_method is None:
        return result_df

    reject, p_corrected, _, _ = multipletests(
        result_df["p_raw"], alpha=0.05, method=correction_method
    )
    result_df["p_corrected"] = p_corrected
    result_df["reject_null"] = reject
    return result_df


def export_ndfu_anova_by_prompt(
    human_datasets: LazyDatasetLoader,
    annotations_dir: Path,
    output_path: Path,
    dataset_keys: list[str] = PROMPT_COMPARISON_DATASET_KEYS,
    prompt_names: list[str] = MAIN_PROMPT_NAMES,
    exclude_models: set[str] | None = None,
    correction_method: str = "fdr_bh",
) -> pd.DataFrame:
    """
    Computes `compute_ndfu_anova_by_prompt` and writes it to `output_path`
    as a CSV. Returns the DataFrame as well, for scripts that want to
    chain further processing (e.g. filtering to significant rows).
    """
    result_df = compute_ndfu_anova_by_prompt(
        human_datasets=human_datasets,
        annotations_dir=annotations_dir,
        dataset_keys=dataset_keys,
        prompt_names=prompt_names,
        exclude_models=exclude_models,
        correction_method=correction_method,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"ANOVA results exported to {output_path.resolve()}")
    return result_df
