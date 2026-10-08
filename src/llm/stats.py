"""
Statistical analysis of LLM (and, for the consistency tables, cross-model)
annotation agreement and significance.
"""

from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ..lib import run_helper
from ..lib.util import (
    center_table_latex,
    small_table_latex,
    trim_numeric_col_latex,
    significance_superscript,
)
from .shared import (
    APUNIM_TABLE_EXCLUDE_MODELS,
    DATASET_KEYS,
    _apunim_dfu,
    _human_sample_dataset,
    _ndfu_bin_count,
    HumanDatasets,
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
    _compute_ndfu_records,
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
    human_datasets: HumanDatasets,
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
    human_datasets: HumanDatasets,
    annotations_dir: Path,
    prompt_name: str = "default",
    exclude_models: list[str] | None = None,
    dataset_keys: list[str] = DATASET_KEYS,
) -> pd.DataFrame:
    """
    For each dataset in `dataset_keys`: how consistent are the different
    LLMs with each other, when all of them are given the same (default)
    prompt?
    """
    exclude_models = set(exclude_models or [])
    rows = []
    for key in _available_dataset_keys(human_datasets, dataset_keys):
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
    human_datasets: HumanDatasets,
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
    human_datasets: HumanDatasets,
    paraphrase_dir: Path,
    variant_names: list[str] = None,
    dataset_keys: list[str] = DATASET_KEYS,
) -> pd.DataFrame:
    """
    For each (dataset in `dataset_keys`, model): how consistent is that
    model with itself across the paraphrased prompt variants
    (variant1/variant2/variant3)?
    """
    variant_names = variant_names or VARIANT_NAMES
    rows = []
    for key in _available_dataset_keys(human_datasets, dataset_keys):
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
    human_datasets: HumanDatasets,
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
    human_datasets: HumanDatasets,
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
    human_datasets: HumanDatasets,
    repeat_dir: Path,
    prompt_name: str = "default",
    dataset_keys: list[str] = DATASET_KEYS,
) -> pd.DataFrame:
    """
    For each (dataset in `dataset_keys`, model): how consistent is that
    model with itself across repeated runs of the *same* prompt?
    """
    rows = []
    for key in _available_dataset_keys(human_datasets, dataset_keys):
        files_by_model = find_repeat_files(repeat_dir, key, prompt_name)
        if not files_by_model:
            raise ValueError(
                f"No files for repeat ablation found in {repeat_dir} for "
                f"dataset {key} and prompt {prompt_name}."
            )
        rows.extend(
            _repeat_rows_for_dataset(human_datasets, key, files_by_model)
        )
    return pd.DataFrame(rows)


def export_latex_table(
    df: pd.DataFrame, output_path: Path, caption: str, label: str
) -> None:
    df = df.copy()
    # very common column in all exported tables in this module
    if "Krippendorff's alpha" in df.columns:
        df["Krippendorff's alpha"] = trim_numeric_col_latex(
            df["Krippendorff's alpha"], float_format=".4f"  # type: ignore
        )

    latex_str = df.to_latex(
        index=False,
        caption=caption,
        label=label,
        position="t",
        escape=True,
    )
    latex_str = center_table_latex(latex_str=latex_str)
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
        print(f"Skipping apunim for {dataset_key}/{prompt_name}/{pseudo}: {e}")
        return None

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
    Runs apunim over LLM annotation CSVs, once per (prompt, model)
    available for `dataset_key`. Returns long-format DataFrame with one
    row per (SDB Feature, Value, Model, Prompt).
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
        df[number_col] = trim_numeric_col_latex(
            df[number_col], float_format=".3f"
        )

    df = df.replace("_", r"\_", regex=True).set_index(
        [r"\ac{pc}", "Value", "Model"]
    )

    latex_str = df.to_latex(
        caption=(
            "Aposteriori unimodality results for the LLM annotations of "
            f"the {dataset_name} dataset, across instruction prompts."
        ),
        label=label,
        escape=False,
        position="t",
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
    stars = significance_superscript(row["pvalue"])
    return f"{row['apunim']:.4f}{stars}"


def build_llm_apunim_prompt_table(
    long_df: pd.DataFrame, prompt_names: list[str] = MAIN_PROMPT_NAMES
) -> pd.DataFrame:
    """
    Pivots long-format apunim output into one row per (SDB Feature, Value,
    Model) and one column per prompt.
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


def _cohens_d(a, b) -> float:
    """Pooled-SD standardized mean difference between two samples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1, s2 = a.var(ddof=1), b.var(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return np.nan
    return (a.mean() - b.mean()) / pooled_sd


def compute_ndfu_anova_by_prompt(
    human_datasets: HumanDatasets,
    annotations_dir: Path,
    dataset_keys: list[str] = PROMPT_COMPARISON_DATASET_KEYS,
    prompt_names: list[str] = MAIN_PROMPT_NAMES,
    exclude_models: set[str] | None = None,
    correction_method: str = "fdr_bh",
    baseline_prompt: str = "default",
) -> pd.DataFrame:
    """
    One-way ANOVA per (dataset, model, SDB group) testing whether that
    group's per-comment nDFU distribution shifts across instruction prompts,
    with FDR-corrected p-values and Cohen's d effect sizes vs. the baseline.
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
                continue

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

                baseline_values = None
                if (
                    baseline_prompt in ndfu_by_prompt
                    and group in ndfu_by_prompt[baseline_prompt].index
                ):
                    baseline_values = ndfu_by_prompt[baseline_prompt][group]

                record = {
                    "Dataset": dataset_name,
                    "Model": model,
                    "PC Dimension": group,
                    "F": f_stat,
                    "p_raw": p_value,
                    "n_prompts_compared": len(samples),
                    "n_total_obs": sum(len(s) for s in samples),
                }

                for prompt_name in prompt_names:
                    if prompt_name == baseline_prompt:
                        continue
                    col = f"cohens_d_{prompt_name}_vs_{baseline_prompt}"
                    prompt_values = None
                    if (
                        prompt_name in ndfu_by_prompt
                        and group in ndfu_by_prompt[prompt_name].index
                    ):
                        prompt_values = ndfu_by_prompt[prompt_name][group]
                    record[col] = (
                        _cohens_d(prompt_values, baseline_values)
                        if prompt_values is not None
                        and baseline_values is not None
                        else np.nan
                    )

                records.append(record)

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
    result_df: pd.DataFrame, output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"ANOVA results exported to {output_path.resolve()}")


# ---------------------------------------------------------------------------
# Cohen's d summary table
# ---------------------------------------------------------------------------


def compute_cohens_d_compact_table(
    result_df: pd.DataFrame,
    prompt_names: list[str] = MAIN_PROMPT_NAMES,
    baseline_prompt: str = "default",
    pooled_label: str = "All",
) -> pd.DataFrame:
    """
    Compact Cohen's d summary: one row per (Model, Dataset) plus a pooled
    `pooled_label` row per model (all datasets together), one column per
    non-baseline prompt, and a leading `n` column with the number of
    groups the statistics are computed over. Each cell is "mean (SD)" of
    Cohen's d (prompt vs. `baseline_prompt`) across groups.
    """
    cols = {
        p: f"cohens_d_{p}_vs_{baseline_prompt}"
        for p in prompt_names
        if p != baseline_prompt
        and f"cohens_d_{p}_vs_{baseline_prompt}" in result_df.columns
    }

    def _row(sub: pd.DataFrame) -> dict:
        row = {"n": str(len(sub))}
        for prompt_name, col in cols.items():
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            if vals.empty:
                row[prompt_name] = "---"
            elif len(vals) < 2:
                row[prompt_name] = f"{vals.iloc[0]:.2f}"
            else:
                row[prompt_name] = f"{vals.mean():.2f} ({vals.std():.2f})"
        return row

    rows, index = [], []
    for model in _order_models(set(result_df["Model"])):
        model_df = result_df.loc[result_df["Model"] == model]
        rows.append(_row(model_df))
        index.append((model, pooled_label))
        # Preserve dataset order of appearance in the results.
        for dataset in model_df["Dataset"].drop_duplicates():
            rows.append(_row(model_df.loc[model_df["Dataset"] == dataset]))
            index.append((model, dataset))

    out = pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(index, names=["Model", "Dataset"]),
    )
    out.columns = ["n"] + [str(c).capitalize() for c in out.columns[1:]]
    return out


def _merge_span_cells(latex_str: str, n_total_cols: int) -> str:
    """
    Rewrites rows containing a `_SPAN_MARK` value cell so that the value
    spans all remaining columns of the row (the rest of the row is empty).
    """
    out = []
    for line in latex_str.splitlines():
        if _SPAN_MARK not in line:
            out.append(line)
            continue
        prefix, rest = line.split(_SPAN_MARK, 1)
        value = rest.split(" & ", 1)[0].removesuffix("\\\\").strip()
        span = n_total_cols - prefix.count(" & ")
        out.append(f"{prefix}\\multicolumn{{{span}}}{{c}}{{{value}}} \\\\")
    return "\n".join(out) + "\n"


def export_compact_prompt_table_latex(
    compact_df: pd.DataFrame,
    output_path: Path,
    caption: str,
    label: str,
) -> None:
    latex_str = compact_df.to_latex(
        caption=caption,
        label=label,
        position="t",
        escape=True,
        index=True,
        multirow=True,
        column_format="ll" + "r" * len(compact_df.columns),
    )
    latex_str = _merge_span_cells(
        latex_str,
        n_total_cols=compact_df.index.nlevels + len(compact_df.columns),
    )
    latex_str = center_table_latex(latex_str)
    latex_str = small_table_latex(latex_str)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_str)
    print(f"Table exported to {output_path.resolve()}")


def export_default_mean_ndfu_latex(
    table_df: pd.DataFrame, output_path: Path, caption: str, label: str
) -> None:
    n_groups = len(table_df.columns) // 2
    latex_str = table_df.to_latex(
        caption=caption,
        label=label,
        position="t",
        escape=True,
        index=True,
        multicolumn=True,
        multicolumn_format="c",
        column_format="l" + "rr" * n_groups,
    )
    latex_str = center_table_latex(latex_str)
    latex_str = small_table_latex(latex_str)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_str)
    print(f"Table exported to {output_path.resolve()}")


# ---------------------------------------------------------------------------
# Mean nDFU: human annotations vs. LLM annotations (per instruction prompt)
# ---------------------------------------------------------------------------

HUMAN_LABEL = "Human"
_SPAN_MARK = "@@SPAN@@"


def _per_comment_ndfu(ds) -> pd.Series:
    """nDFU of each comment's annotation list, indexed by comment id."""
    df = ds.get_dataset()
    annotation_col = ds.get_annotation_column()
    bins = _ndfu_bin_count(df[annotation_col].to_list())
    if bins == 0:
        return pd.Series(dtype=float)

    key_col = ds.get_comment_key_column()
    values = {}
    for _, row in df.iterrows():
        annotations = row[annotation_col]
        if not isinstance(annotations, (list, np.ndarray)) or (
            len(annotations) == 0
        ):
            continue
        try:
            values[row[key_col]] = _apunim_dfu(annotations, bins)
        except Exception as e:
            print(f"Error calculating NDFU for an item: {e}")
    return pd.Series(values, dtype=float)


def compute_mean_ndfu_records(
    human_datasets: HumanDatasets,
    annotations_dir: Path,
    dataset_keys: list[str] = PROMPT_COMPARISON_DATASET_KEYS,
    prompt_names: list[str] = MAIN_PROMPT_NAMES,
    exclude_models: set[str] | None = APUNIM_TABLE_EXCLUDE_MODELS,
    sample_prompt: str = "default",
) -> pd.DataFrame:
    """
    Long-format per-comment nDFU, one row per (Dataset, Model, Prompt,
    text_id). Human rows have Model == HUMAN_LABEL and Prompt == "human";
    they are computed on the comments actually sampled for the LLM runs
    (see `_human_sample_dataset`), so human and LLM nDFU are comparable
    comment by comment.
    """
    exclude_models = set(exclude_models or ())
    records = []

    def _emit(dataset_name, model, prompt, ndfu: pd.Series):
        return [
            {
                "Dataset": dataset_name,
                "Model": model,
                "Prompt": prompt,
                "text_id": text_id,
                "ndfu": value,
            }
            for text_id, value in ndfu.dropna().items()
        ]

    for dataset_key in dataset_keys:
        if dataset_key not in human_datasets:
            continue
        human_ds = _human_sample_dataset(
            human_datasets[dataset_key],
            annotations_dir,
            dataset_key,
            sample_prompt,
        )
        if human_ds is None:
            continue
        dataset_name = human_datasets[dataset_key].get_name()
        records.extend(
            _emit(
                dataset_name,
                HUMAN_LABEL,
                "human",
                _per_comment_ndfu(human_ds),
            )
        )

        for prompt_name in prompt_names:
            files = find_annotation_files(
                annotations_dir, dataset_key, prompt_name
            )
            for model, path in files.items():
                if model in exclude_models:
                    continue
                llm_ds = LLMAnnotationDataset(
                    load_llm_df(path), dataset_key, model, prompt_name
                )
                records.extend(
                    _emit(
                        dataset_name,
                        model,
                        prompt_name,
                        _per_comment_ndfu(llm_ds),
                    )
                )

    return pd.DataFrame(records)


def _mean_se_str(values: pd.Series) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return "---"
    se = (
        values.std(ddof=1) / np.sqrt(len(values))
        if len(values) > 1
        else np.nan
    )
    return f"{values.mean():.2f} ({se:.2f})"


def _datasets_in_order(df: pd.DataFrame) -> list[str]:
    return list(df["Dataset"].drop_duplicates())


def build_mean_ndfu_table(
    ndfu_df: pd.DataFrame,
    prompt_names: list[str] = MAIN_PROMPT_NAMES,
    pooled_label: str = "All",
) -> pd.DataFrame:
    """
    Compact table: a leading Human block, then one block per model; each
    block has a pooled `pooled_label` row and one row per dataset. Columns
    are `n` (comments) and one per instruction prompt, with the mean (SE)
    per-comment nDFU. Human annotations do not depend on the prompt, so
    their value is placed in a single cell spanning all prompt columns
    (see `export_compact_prompt_table_latex`). Within a model row only
    comments that the human sample and every available prompt share are
    used, so all cells in a row use the same comments.
    """
    human_df = ndfu_df.loc[ndfu_df["Model"] == HUMAN_LABEL]
    llm_df = ndfu_df.loc[ndfu_df["Model"] != HUMAN_LABEL]
    prompts = [p for p in prompt_names if p in set(llm_df["Prompt"])]
    datasets = _datasets_in_order(human_df)

    def _human_row(sub: pd.DataFrame) -> dict:
        row = {"n": str(len(sub))}
        for i, p in enumerate(prompts):
            row[p] = _SPAN_MARK + _mean_se_str(sub["ndfu"]) if i == 0 else ""
        return row

    rows, index = [], []

    # Human block
    rows.append(_human_row(human_df))
    index.append((HUMAN_LABEL, pooled_label))
    for d in datasets:
        rows.append(_human_row(human_df.loc[human_df["Dataset"] == d]))
        index.append((HUMAN_LABEL, d))

    # Model blocks
    for model in _order_models(set(llm_df["Model"])):
        model_df = llm_df.loc[llm_df["Model"] == model]
        matched = {}
        for d in datasets:
            sub = model_df.loc[model_df["Dataset"] == d]
            present = [p for p in prompts if p in set(sub["Prompt"])]
            if not present:
                continue
            ids = set(human_df.loc[human_df["Dataset"] == d, "text_id"])
            for p in present:
                ids &= set(sub.loc[sub["Prompt"] == p, "text_id"])
            matched[d] = sub[sub["text_id"].isin(ids)]

        def _row(sub: pd.DataFrame) -> dict:
            first = sub.loc[sub["Prompt"] == sub["Prompt"].iloc[0]]
            row = {"n": str(first["text_id"].nunique())}
            for p in prompts:
                row[p] = _mean_se_str(sub.loc[sub["Prompt"] == p, "ndfu"])
            return row

        if not matched:
            continue
        pooled_n = sum(
            m.loc[m["Prompt"] == m["Prompt"].iloc[0], "text_id"].nunique()
            for m in matched.values()
            if not m.empty
        )
        pooled = _row(pd.concat(matched.values()))
        pooled["n"] = str(pooled_n)
        rows.append(pooled)
        index.append((model, pooled_label))
        for d, sub in matched.items():
            if sub.empty:
                continue
            rows.append(_row(sub))
            index.append((model, d))

    out = pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(index, names=["Model", "Dataset"]),
    )
    out.columns = ["n"] + [str(c).capitalize() for c in out.columns[1:]]
    return out


def build_default_mean_ndfu_table(
    ndfu_df: pd.DataFrame,
    prompt_name: str = "default",
    pooled_label: str = "All",
) -> pd.DataFrame:
    """
    Human row first, then one row per model. For each dataset (plus a
    pooled `pooled_label` group) an `n` column (comments) and a
    "Mean (SE)" column with the mean per-comment nDFU. Model rows only use
    comments that also appear in the human sample.
    """
    human_df = ndfu_df.loc[ndfu_df["Model"] == HUMAN_LABEL]
    llm_df = ndfu_df.loc[
        (ndfu_df["Model"] != HUMAN_LABEL) & (ndfu_df["Prompt"] == prompt_name)
    ]
    datasets = _datasets_in_order(human_df)

    def _fill(row: dict, label: str, sub: pd.DataFrame) -> None:
        row[(label, "n")] = str(len(sub)) if len(sub) else "---"
        row[(label, "Mean (SE)")] = _mean_se_str(sub["ndfu"])

    def _make_row(source: pd.DataFrame, restrict: bool) -> dict:
        row, pooled_parts = {}, []
        for d in datasets:
            sub = source.loc[source["Dataset"] == d]
            if restrict:
                ids = set(human_df.loc[human_df["Dataset"] == d, "text_id"])
                sub = sub[sub["text_id"].isin(ids)]
            _fill(row, d, sub)
            pooled_parts.append(sub)
        _fill(row, pooled_label, pd.concat(pooled_parts))
        return row

    rows = {HUMAN_LABEL: _make_row(human_df, restrict=False)}
    for model in _order_models(set(llm_df["Model"])):
        rows[model] = _make_row(
            llm_df.loc[llm_df["Model"] == model], restrict=True
        )

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index.name = "Model"
    return out


def _significance_ratio(res_df: pd.DataFrame, col: str) -> pd.Series:
    total = res_df[col].value_counts()
    significant = res_df.loc[res_df.reject_null, col].value_counts()
    significant = significant.reindex(total.index, fill_value=0)
    return significant.astype(str) + " / " + total.astype(str)


def run_exploratory_stats(res_df: pd.DataFrame) -> None:
    print("Statistically significant / total distinct groups by model:")
    print(_significance_ratio(res_df, "Model"))
    print("Statistically significant / total distinct groups by dataset:")
    print(_significance_ratio(res_df, "Dataset"))
