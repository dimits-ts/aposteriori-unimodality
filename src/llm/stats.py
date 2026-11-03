"""
Statistical analysis of LLM (and, for the consistency tables, cross-model)
annotation agreement and significance.
"""

from collections.abc import Callable, Collection, Iterable
from enum import Enum
from pathlib import Path
from typing import NamedTuple

import krippendorff
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ..lib import run_helper
from ..lib.preprocessing import DATASET_CLASSES, Dataset
from .shared import (
    HUMAN_LABEL,
    HUMAN_PROMPT_NAME,
    POOLED_LABEL,
    HumanDatasets,
    LLMAnnotationDataset,
    available_dataset_keys,
    clean_annotation,
    find_annotation_files,
    find_repeat_files,
    human_sample_dataset,
    key_columns,
    keyed_series,
    load_llm_df,
    order_models,
    out_of_range,
)

# Maps a dataset key to its (human, LLM) apunim results.
ApunimInputs = dict[str, tuple[pd.DataFrame, pd.DataFrame]]

# (model, dataset) -> number of annotations.
AnnotationTotals = dict[tuple[str, str], int]


# ---------------------------------------------------------------------------
# Non-standard answers
# ---------------------------------------------------------------------------


def _annotation_status(
    annotation: pd.Series, valid: Collection[int]
) -> pd.Series:
    """
    Per-reply status, judged with the parser the analysis uses
    (shared.clean_annotation): "unparsed" (no integer found) or
    "out_of_range" (integer not valid for the dataset), both NaN in
    shared.load_llm_df and so dropped downstream, or NA for a reply that
    parses to a valid value (including replies that only parse after
    cleaning).
    """
    cleaned = clean_annotation(annotation)

    status = pd.Series(pd.NA, index=annotation.index, dtype="object")
    status[cleaned.isna()] = "unparsed"
    status[out_of_range(cleaned, valid)] = "out_of_range"
    return status


def _nonstandard_counts_for_file(
    path: Path, model: str, key: str, valid: Collection[int]
) -> tuple[pd.DataFrame, int]:
    """(status/answer value counts of one annotation file, its row count)."""
    df = pd.read_csv(path)
    raw = df["annotation"].fillna("<empty>").astype(str)
    status = _annotation_status(df["annotation"], valid)

    flagged = status.notna()
    counts = (
        pd.DataFrame({"status": status[flagged], "answer": raw[flagged]})
        .value_counts()
        .reset_index(name="count")
    )
    counts["model"] = model
    counts["dataset"] = key
    return counts, len(df)


def _nonstandard_counts_for_dataset(
    annotations_dir: Path,
    key: str,
    prompt_names: Iterable[str],
    totals: AnnotationTotals,
) -> list[pd.DataFrame]:
    """Counts for every prompt and model of one dataset. Adds each file's
    row count to `totals[(model, key)]` (in place)."""
    valid = DATASET_CLASSES[key].VALID_ANNOTATION_VALUES
    frames = []
    for prompt_name in prompt_names:
        files = find_annotation_files(annotations_dir, key, prompt_name)
        for model, path in sorted(files.items()):
            counts, n_rows = _nonstandard_counts_for_file(
                path, model, key, valid
            )
            totals[(model, key)] = totals.get((model, key), 0) + n_rows
            frames.append(counts)
    return frames


def collect_nonstandard_counts(
    annotations_dir: Path,
    dataset_keys: Iterable[str],
    prompt_names: Collection[str],
) -> tuple[pd.DataFrame, AnnotationTotals]:
    """
    (per-file counts of non-standard replies with columns
    model/dataset/status/answer/count, total number of annotations per
    (model, dataset)), over `dataset_keys` and `prompt_names`. Statuses:
    "unparsed" and "out_of_range" (see _annotation_status).
    """
    frames: list[pd.DataFrame] = []
    totals: AnnotationTotals = {}
    for key in dataset_keys:
        frames.extend(
            _nonstandard_counts_for_dataset(
                annotations_dir, key, prompt_names, totals
            )
        )

    if not frames:
        empty = pd.DataFrame(
            columns=["model", "dataset", "status", "answer", "count"]
        )
        return empty, totals
    return pd.concat(frames, ignore_index=True), totals


def nonstandard_by_model(
    detailed: pd.DataFrame, totals: AnnotationTotals
) -> pd.DataFrame:
    """One row per (model, status, answer), with the model's total number
    of annotations (n_total_model) and the share each row represents."""
    per_model: dict[str, int] = {}
    for (model, _), n in totals.items():
        per_model[model] = per_model.get(model, 0) + n

    result = detailed.groupby(["model", "status", "answer"], as_index=False)[
        "count"
    ].sum()
    result["n_total_model"] = result["model"].map(per_model)
    result["rate"] = result["count"] / result["n_total_model"]
    return result.sort_values(
        ["model", "status", "count"], ascending=[True, True, False]
    ).reset_index(drop=True)


def nonstandard_summary(
    detailed: pd.DataFrame, totals: AnnotationTotals
) -> pd.DataFrame:
    """One row per (model, dataset, status), summed over answers and
    prompts. `incidence` is the count as a percentage of all of the
    model's annotations on that dataset (over all prompts)."""
    summary = detailed.groupby(["model", "dataset", "status"], as_index=False)[
        "count"
    ].sum()
    keys = pd.Series(
        list(zip(summary["model"], summary["dataset"])), index=summary.index
    )
    summary["incidence"] = 100 * summary["count"] / keys.map(totals)
    return summary.sort_values(["model", "dataset", "status"]).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# Krippendorff's-alpha consistency
# ---------------------------------------------------------------------------


def _build_reliability_matrix(
    dfs: dict[str, pd.DataFrame], key_cols: list[str]
) -> tuple[np.ndarray, int]:
    """
    dfs: {rater_label: dataframe}, each with `key_cols` plus
    'annotation_clean'.

    Returns (matrix, n_items): matrix has shape (n_raters, n_items) -- the
    layout krippendorff.alpha expects -- aligned on the union of keys
    across raters, with NaN where a rater has no annotation for an item.
    """
    series_list = [
        keyed_series(df, key_cols, label) for label, df in dfs.items()
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
    count_label: str,
    extra_fields: dict[str, str],
) -> dict[str, object]:
    matrix, n_items = _build_reliability_matrix(dfs, key_columns(dfs))
    return {
        **extra_fields,
        count_label: matrix.shape[0],
        "Items": n_items,
        "Krippendorff's alpha": krippendorff_alpha_safe(matrix),
    }


def cross_model_consistency_table(
    human_datasets: HumanDatasets,
    annotations_dir: Path,
    prompt_name: str,
    dataset_keys: Iterable[str],
    exclude_models: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    For each dataset in `dataset_keys`: how consistent are the different
    LLMs with each other, when all of them are given the same prompt?
    """
    excluded = set(exclude_models or ())
    rows = []
    for key in available_dataset_keys(human_datasets, dataset_keys):
        files = {
            pseudo: path
            for pseudo, path in find_annotation_files(
                annotations_dir, key, prompt_name
            ).items()
            if pseudo not in excluded
        }
        if len(files) < 2:
            continue
        dfs = {pseudo: load_llm_df(path) for pseudo, path in files.items()}
        rows.append(
            _consistency_row(
                dfs,
                "Models",
                {
                    "Dataset": human_datasets[key].get_name(),
                    "Prompt": prompt_name,
                },
            )
        )
    return pd.DataFrame(rows)


def _variant_rows_for_dataset(
    dataset_name: str,
    files_by_variant: dict[str, dict[str, Path]],
    variant_names: list[str],
) -> list[dict[str, object]]:
    models = sorted(set.union(*(set(f) for f in files_by_variant.values())))
    rows = []
    for pseudo in models:
        dfs = {
            v: load_llm_df(files_by_variant[v][pseudo])
            for v in variant_names
            if pseudo in files_by_variant[v]
        }
        if len(dfs) < 2:
            continue
        rows.append(
            _consistency_row(
                dfs, "Variants", {"Dataset": dataset_name, "Model": pseudo}
            )
        )
    return rows


def per_model_variant_consistency_table(
    human_datasets: HumanDatasets,
    paraphrase_dir: Path,
    variant_names: list[str],
    dataset_keys: Iterable[str],
) -> pd.DataFrame:
    """
    For each (dataset in `dataset_keys`, model): how consistent is that
    model with itself across the paraphrased prompt variants?
    """
    rows = []
    for key in available_dataset_keys(human_datasets, dataset_keys):
        files_by_variant = {
            v: find_annotation_files(paraphrase_dir, key, v)
            for v in variant_names
        }
        if not any(files_by_variant.values()):
            continue
        rows.extend(
            _variant_rows_for_dataset(
                human_datasets[key].get_name(), files_by_variant, variant_names
            )
        )
    return pd.DataFrame(rows)


def per_model_repeat_consistency_table(
    human_datasets: HumanDatasets,
    repeat_dir: Path,
    prompt_name: str,
    dataset_keys: Iterable[str],
) -> pd.DataFrame:
    """
    For each (dataset in `dataset_keys`, model): how consistent is that
    model with itself across repeated runs of the *same* prompt?
    """
    rows = []
    for key in available_dataset_keys(human_datasets, dataset_keys):
        files_by_model = find_repeat_files(repeat_dir, key, prompt_name)
        if not files_by_model:
            raise ValueError(
                f"No files for repeat ablation found in {repeat_dir} for "
                f"dataset {key} and prompt {prompt_name}."
            )
        for pseudo, run_files in sorted(files_by_model.items()):
            dfs = {
                run_label: load_llm_df(path)
                for run_label, path in sorted(run_files.items())
            }
            if len(dfs) < 2:
                continue
            rows.append(
                _consistency_row(
                    dfs,
                    "Runs",
                    {
                        "Dataset": human_datasets[key].get_name(),
                        "Model": pseudo,
                    },
                )
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# apunim on the LLM annotations
# ---------------------------------------------------------------------------

# Columns of an apunim results frame (compute_llm_apunim_by_prompt).
APUNIM_COLUMNS = [
    "SDB Feature",
    "Value",
    "Model",
    "Prompt",
    "apunim",
    "pvalue",
    "support",
]


def _apunim_for_model(
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


def compute_llm_apunim_by_prompt(
    annotations_dir: Path,
    dataset_key: str,
    prompt_names: Iterable[str],
    exclude_models: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Runs apunim over LLM annotation CSVs, once per (prompt, model)
    available for `dataset_key`. Returns a long-format DataFrame with one
    row per (SDB Feature, Value, Model, Prompt); see APUNIM_COLUMNS.
    """
    excluded = set(exclude_models or ())
    frames = []
    for prompt_name in prompt_names:
        files = find_annotation_files(
            annotations_dir, dataset_key, prompt_name
        )
        for pseudo in order_models(set(files) - excluded):
            res_df = _apunim_for_model(
                dataset_key, prompt_name, pseudo, files[pseudo]
            )
            if res_df is not None:
                frames.append(res_df)

    if not frames:
        return pd.DataFrame(columns=APUNIM_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def load_human_apunim_results(
    human_results_dir: Path, dataset_key: str
) -> pd.DataFrame | None:
    """
    Human apunim results (`<dataset_key>-results.csv`, written by the human
    analysis scripts) with columns SDB Feature, Value, apunim, pvalue.
    Returns None if the file does not exist.
    """
    path = human_results_dir / f"{dataset_key}-results.csv"
    if not path.exists():
        matches = sorted(human_results_dir.rglob(f"{dataset_key}-results.csv"))
        if not matches:
            print(
                f"No human apunim results for {dataset_key} in "
                f"{human_results_dir}; skipping."
            )
            return None
        path = matches[0]
    df = pd.read_csv(path)
    df = df.rename(
        columns={df.columns[0]: "SDB Feature", df.columns[1]: "Value"}
    )
    return df[["SDB Feature", "Value", "apunim", "pvalue"]]


# ---------------------------------------------------------------------------
# Significant PC dimensions: human vs. LLM annotations, per prompt
# ---------------------------------------------------------------------------

# apunim's p-values already account for testing several PC dimensions, so a
# dimension counts as significant when its reported p-value is below this.
SIGNIFICANCE_ALPHA = 0.05


class SignificanceOutcome(Enum):
    """How an LLM result for a PC dimension compares with the human one."""

    AGREE_SIGNIFICANT = "agree_significant"  # both significant, same sign
    AGREE_NEITHER = "agree_neither"  # neither significant
    SIGN_DIFFERS = "sign_differs"  # both significant, sign differs
    DISAGREE = "disagree"  # significant for only one of the two


def _dimension_significance(df: pd.DataFrame, alpha: float) -> dict[str, bool]:
    """
    PC dimension (SDB feature, e.g. age) -> significant?

    apunim reports one p-value per group (feature value). A dimension counts
    as significant if at least one of its groups has a p-value below
    `alpha`. Dimensions without any p-value are left out.
    """
    p_values = pd.to_numeric(df["pvalue"], errors="coerce")
    features = df["SDB Feature"].astype(str)
    significant = (p_values < alpha).groupby(features).any()
    has_p_value = p_values.notna().groupby(features).any()
    return {
        feature: bool(significant[feature])
        for feature in has_p_value.index[has_p_value]
    }


def _significant_group_signs(
    df: pd.DataFrame, alpha: float
) -> dict[tuple[str, str], int]:
    """
    (SDB feature, value) -> sign (+1 or -1) of the group's apunim value, for
    the groups that are significant (p below `alpha`) and have a non-zero
    value. Other groups are left out: the sign of a non-significant value
    is noise.
    """
    p_values = pd.to_numeric(df["pvalue"], errors="coerce")
    apunim = pd.to_numeric(df["apunim"], errors="coerce")
    keep = (p_values < alpha) & apunim.notna() & (apunim != 0)
    groups = zip(
        df.loc[keep, "SDB Feature"].astype(str),
        df.loc[keep, "Value"].astype(str),
    )
    return {g: int(np.sign(a)) for g, a in zip(groups, apunim[keep])}


def _sign_disagreement_dimensions(
    human_signs: dict[tuple[str, str], int],
    llm_signs: dict[tuple[str, str], int],
) -> set[str]:
    """PC dimensions with a group that is significant for both the human and
    the LLM results but with opposite signs."""
    return {
        feature
        for (feature, value), sign in llm_signs.items()
        if human_signs.get((feature, value), sign) != sign
    }


def _comparison_outcome(
    llm_significant: bool, human_significant: bool, sign_differs: bool
) -> SignificanceOutcome:
    if llm_significant != human_significant:
        return SignificanceOutcome.DISAGREE
    if llm_significant and sign_differs:
        return SignificanceOutcome.SIGN_DIFFERS
    if llm_significant:
        return SignificanceOutcome.AGREE_SIGNIFICANT
    return SignificanceOutcome.AGREE_NEITHER


def _llm_pair_results(
    llm_df: pd.DataFrame,
    models: list[str],
    prompts: list[str],
    alpha: float,
    human_signs: dict[tuple[str, str], int],
) -> dict[tuple[str, str], tuple[dict[str, bool], set[str]]]:
    """(model, prompt) -> (dimension significance, dimensions whose sign
    differs from the human one for some group significant in both)."""
    results = {}
    for m in models:
        for p in prompts:
            pair_df = llm_df.loc[
                (llm_df["Model"] == m) & (llm_df["Prompt"] == p)
            ]
            differs = _sign_disagreement_dimensions(
                human_signs, _significant_group_signs(pair_df, alpha)
            )
            results[(m, p)] = (
                _dimension_significance(pair_df, alpha),
                differs,
            )
    return results


def _report_unmatched(
    dataset_name: str,
    human_dims: set[str],
    llm_dims: set[str],
    human_df: pd.DataFrame,
    llm_df: pd.DataFrame,
) -> None:
    """Prints label mismatches between the human and the LLM results: whole
    PC dimensions, and groups of the dimensions both have (the sign
    comparison cannot use groups without a counterpart)."""
    only_human, only_llm = human_dims - llm_dims, llm_dims - human_dims
    if only_human or only_llm:
        print(
            f"{dataset_name}: {len(only_human)} PC dimensions only in the "
            f"human results and {len(only_llm)} only in the LLM results; if "
            "these should match, check that the feature labels agree."
        )

    def groups(df: pd.DataFrame) -> set[tuple[str, str]]:
        return set(zip(df["SDB Feature"].astype(str), df["Value"].astype(str)))

    shared_dims = human_dims & llm_dims
    unmatched = {
        g for g in groups(llm_df) - groups(human_df) if g[0] in shared_dims
    }
    if unmatched:
        print(
            f"{dataset_name}: {len(unmatched)} LLM groups have no group with "
            "the same label in the human results; their signs cannot be "
            "compared."
        )


def _dataset_significance_rows(
    dataset_name: str,
    human_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    models: list[str],
    prompts: list[str],
    alpha: float,
) -> list[dict[str, object]]:
    """Rows (one per PC dimension) of one dataset."""
    human = _dimension_significance(human_df, alpha)
    pairs = _llm_pair_results(
        llm_df,
        models,
        prompts,
        alpha,
        _significant_group_signs(human_df, alpha),
    )

    llm_dims: set[str] = set()
    for significance, _ in pairs.values():
        llm_dims |= set(significance)
    _report_unmatched(dataset_name, set(human), llm_dims, human_df, llm_df)

    rows: list[dict[str, object]] = []
    for dim in sorted(llm_dims):
        human_significant = bool(human.get(dim))
        cells = {
            m: tuple(
                # A dimension without a result counts as not significant.
                _comparison_outcome(
                    bool(pairs[(m, p)][0].get(dim)),
                    human_significant,
                    dim in pairs[(m, p)][1],
                )
                for p in prompts
            )
            for m in models
        }
        rows.append({"Dataset": dataset_name, "SDB Feature": dim, **cells})
    return rows


def compute_apunim_significance(
    inputs: ApunimInputs,
    prompt_names: list[str],
    alpha: float = SIGNIFICANCE_ALPHA,
) -> pd.DataFrame:
    """
    One row per (dataset, PC dimension) for which the LLMs have results, and
    one column per model. `inputs` maps a dataset name to its (human, LLM)
    apunim results.

    A model cell is a tuple with one SignificanceOutcome per prompt (in
    `prompt_names` order, restricted to the prompts with results), saying
    how the model's result for the dimension compares with the human one.
    A missing result counts as not significant (for the LLM, this happens
    when apunim found no polarized comments with enough annotations per
    group). The prompts used are stored in `.attrs["prompts"]`.
    """
    llm_dfs = [llm_df for _, llm_df in inputs.values()]
    models = order_models({m for df in llm_dfs for m in df["Model"]})
    prompts = [
        p
        for p in prompt_names
        if any(p in set(df["Prompt"]) for df in llm_dfs)
    ]

    rows = []
    for dataset_name, (human_df, llm_df) in inputs.items():
        rows.extend(
            _dataset_significance_rows(
                dataset_name, human_df, llm_df, models, prompts, alpha
            )
        )
    out = pd.DataFrame(rows)
    out.attrs["prompts"] = prompts
    return out


def _dimension_count(
    dataset_name: str, model: str, prompt: str, df: pd.DataFrame, alpha: float
) -> dict[str, object]:
    significance = _dimension_significance(df, alpha)
    return {
        "Dataset": dataset_name,
        "Model": model,
        "Prompt": prompt,
        "significant": sum(significance.values()),
        "tested": len(significance),
    }


def count_significant_dimensions(
    inputs: ApunimInputs,
    prompt_names: Collection[str],
    alpha: float = SIGNIFICANCE_ALPHA,
) -> pd.DataFrame:
    """
    One row per (Dataset, Model, Prompt): how many PC dimensions are
    significant (`significant`) out of those with results (`tested`), with
    significance decided as in compute_apunim_significance. `inputs` is the
    same mapping as there; the human results are included as Model
    HUMAN_LABEL, Prompt HUMAN_PROMPT_NAME.
    """
    rows = []
    for dataset_name, (human_df, llm_df) in inputs.items():
        rows.append(
            _dimension_count(
                dataset_name, HUMAN_LABEL, HUMAN_PROMPT_NAME, human_df, alpha
            )
        )
        for (model, prompt), pair_df in llm_df.groupby(
            ["Model", "Prompt"], sort=False
        ):
            if prompt in prompt_names:
                rows.append(
                    _dimension_count(
                        dataset_name, str(model), str(prompt), pair_df, alpha
                    )
                )
    return pd.DataFrame(
        rows,
        columns=["Dataset", "Model", "Prompt", "significant", "tested"],
    )


# ---------------------------------------------------------------------------
# Prompt-sensitivity ANOVA on per-comment nDFU, and Cohen's d
# ---------------------------------------------------------------------------

# Per-comment nDFU (shared with the human analysis scripts).
_per_comment_ndfu = run_helper.per_comment_ndfu


def _cohens_d(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Pooled-SD standardized mean difference between two samples; NaN if
    either is missing or too small."""
    if a is None or b is None:
        return np.nan
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


def _cohens_d_column(prompt_name: str, baseline_prompt: str) -> str:
    return f"cohens_d_{prompt_name}_vs_{baseline_prompt}"


def _ndfu_by_prompt(
    files_by_prompt: dict[str, dict[str, Path]],
    dataset_key: str,
    model: str,
) -> dict[str, np.ndarray]:
    """prompt -> per-comment nDFU values of `model` (prompts with fewer
    than two values are left out)."""
    out = {}
    for prompt_name, files in files_by_prompt.items():
        path = files.get(model)
        if path is None:
            continue
        ds = LLMAnnotationDataset(
            load_llm_df(path), dataset_key, model, prompt_name
        )
        values = _per_comment_ndfu(ds).dropna().to_numpy()
        if len(values) >= 2:
            out[prompt_name] = values
    return out


def compute_ndfu_anova_by_prompt(
    human_datasets: HumanDatasets,
    annotations_dir: Path,
    dataset_keys: Iterable[str],
    prompt_names: list[str],
    baseline_prompt: str,
    exclude_models: Iterable[str] | None = None,
    correction_method: str | None = "fdr_bh",
) -> pd.DataFrame:
    """
    One-way ANOVA per (dataset, model) testing whether the per-comment nDFU
    distribution shifts across instruction prompts, with corrected p-values
    and Cohen's d effect sizes vs. `baseline_prompt`.

    Each comment contributes exactly one nDFU value per prompt (computed on
    all of its annotations). The test and the effect size are therefore
    computed once per (dataset, model) on the comment-level values, instead
    of once per demographic group: per-group samples are near-copies of the
    same comments, so per-group tests would not be independent evidence.
    """
    excluded = set(exclude_models or ())
    records = []

    for dataset_key in available_dataset_keys(human_datasets, dataset_keys):
        files_by_prompt = {
            p: find_annotation_files(annotations_dir, dataset_key, p)
            for p in prompt_names
        }
        models = order_models(
            set.union(*(set(f) for f in files_by_prompt.values())) - excluded
        )
        dataset_name = human_datasets[dataset_key].get_name()

        for model in models:
            ndfu_by_prompt = _ndfu_by_prompt(
                files_by_prompt, dataset_key, model
            )
            samples = list(ndfu_by_prompt.values())
            if len(samples) < 2:
                continue
            f_stat, p_value = stats.f_oneway(*samples)

            baseline_values = ndfu_by_prompt.get(baseline_prompt)
            record: dict[str, object] = {
                "Dataset": dataset_name,
                "Model": model,
                "F": f_stat,
                "p_raw": p_value,
                "n_prompts_compared": len(samples),
                "n_total_obs": sum(len(v) for v in samples),
                "n_comments": (
                    len(baseline_values)
                    if baseline_values is not None
                    else np.nan
                ),
            }
            for prompt_name in prompt_names:
                if prompt_name != baseline_prompt:
                    record[_cohens_d_column(prompt_name, baseline_prompt)] = (
                        _cohens_d(
                            ndfu_by_prompt.get(prompt_name), baseline_values
                        )
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


def write_ndfu_anova_by_prompt(
    result_df: pd.DataFrame, output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"ANOVA results exported to {output_path.resolve()}")


def compute_cohens_d_compact_table(
    result_df: pd.DataFrame,
    prompt_names: Iterable[str],
    baseline_prompt: str,
) -> pd.DataFrame:
    """
    Compact Cohen's d summary of compute_ndfu_anova_by_prompt's results:
    one row per (Model, Dataset), a leading `n` column with the number of
    comments the effect size is computed over, and one column per
    non-baseline prompt holding Cohen's d (prompt vs. `baseline_prompt`) on
    per-comment nDFU, NaN if unavailable. A pooled POOLED_LABEL row per
    model gives the unweighted mean of the per-dataset values (and the
    total comment count).
    """
    cols = {
        p: _cohens_d_column(p, baseline_prompt)
        for p in prompt_names
        if p != baseline_prompt
        and _cohens_d_column(p, baseline_prompt) in result_df.columns
    }

    def _row(sub: pd.DataFrame) -> dict[str, float]:
        n = pd.to_numeric(sub["n_comments"], errors="coerce").sum()
        row: dict[str, float] = {"n": int(n)}
        for prompt_name, col in cols.items():
            row[prompt_name] = pd.to_numeric(sub[col], errors="coerce").mean()
        return row

    rows, index = [], []
    for model in order_models(set(result_df["Model"])):
        model_df = result_df.loc[result_df["Model"] == model]
        rows.append(_row(model_df))
        index.append((model, POOLED_LABEL))
        # Preserve dataset order of appearance in the results.
        for dataset in model_df["Dataset"].drop_duplicates():
            rows.append(_row(model_df.loc[model_df["Dataset"] == dataset]))
            index.append((model, dataset))

    return pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(index, names=["Model", "Dataset"]),
        columns=["n", *cols],
    )


# ---------------------------------------------------------------------------
# Per-comment records: human annotations vs. LLM annotations
# ---------------------------------------------------------------------------


def per_comment_mean_annotation(ds: Dataset) -> pd.Series:
    """Mean annotation of each comment's annotation list (non-numeric and
    missing values ignored), indexed by the comment key of the dataset."""
    df = ds.get_dataset()
    annotation_col = ds.get_annotation_column()
    key_col = ds.get_comment_key_column()
    values = {}
    for key, annotations in zip(df[key_col], df[annotation_col]):
        if not isinstance(annotations, (list, np.ndarray)):
            continue
        numeric = pd.to_numeric(pd.Series(annotations), errors="coerce")
        if numeric.notna().any():
            values[key] = float(numeric.mean())
    return pd.Series(values, dtype=float)


# Signature shared by compute_mean_ndfu_records and
# compute_mean_annotation_records.
RecordsFunction = Callable[..., pd.DataFrame]


def _compute_per_comment_records(
    human_datasets: HumanDatasets,
    annotations_dir: Path,
    per_comment: Callable[[Dataset], pd.Series],
    value_col: str,
    dataset_keys: Iterable[str],
    prompt_names: Iterable[str],
    exclude_models: Iterable[str] | None,
    sample_prompt: str,
    include_human: bool,
) -> pd.DataFrame:
    """
    Long-format per-comment values (`per_comment` of each dataset, in
    column `value_col`), one row per (Dataset, Model, Prompt, text_id).
    Human rows have Model == HUMAN_LABEL and Prompt == HUMAN_PROMPT_NAME;
    they are computed on the comments actually sampled for the LLM runs
    (see shared.human_sample_dataset), so human and LLM values are
    comparable comment by comment. With `include_human=False` only LLM
    rows are returned (e.g. to add further prompts to an existing set of
    records).
    """
    excluded = set(exclude_models or ())
    records: list[dict[str, object]] = []

    def _emit(
        dataset_name: str, model: str, prompt: str, values: pd.Series
    ) -> None:
        records.extend(
            {
                "Dataset": dataset_name,
                "Model": model,
                "Prompt": prompt,
                "text_id": text_id,
                value_col: value,
            }
            for text_id, value in values.dropna().items()
        )

    for dataset_key in available_dataset_keys(human_datasets, dataset_keys):
        human_ds = human_sample_dataset(
            human_datasets[dataset_key],
            annotations_dir,
            dataset_key,
            sample_prompt,
        )
        if human_ds is None:
            continue
        dataset_name = human_datasets[dataset_key].get_name()
        if include_human:
            _emit(
                dataset_name,
                HUMAN_LABEL,
                HUMAN_PROMPT_NAME,
                per_comment(human_ds),
            )

        for prompt_name in prompt_names:
            files = find_annotation_files(
                annotations_dir, dataset_key, prompt_name
            )
            for model, path in files.items():
                if model in excluded:
                    continue
                llm_ds = LLMAnnotationDataset(
                    load_llm_df(path), dataset_key, model, prompt_name
                )
                _emit(dataset_name, model, prompt_name, per_comment(llm_ds))

    # Explicit columns so an empty result still round-trips through a CSV.
    return pd.DataFrame(
        records, columns=["Dataset", "Model", "Prompt", "text_id", value_col]
    )


def compute_mean_ndfu_records(
    human_datasets: HumanDatasets,
    annotations_dir: Path,
    dataset_keys: Iterable[str],
    prompt_names: Iterable[str],
    exclude_models: Iterable[str] | None = None,
    sample_prompt: str = "default",
    include_human: bool = True,
) -> pd.DataFrame:
    """Per-comment nDFU records (column "ndfu"), see
    _compute_per_comment_records."""
    return _compute_per_comment_records(
        human_datasets,
        annotations_dir,
        per_comment=_per_comment_ndfu,
        value_col="ndfu",
        dataset_keys=dataset_keys,
        prompt_names=prompt_names,
        exclude_models=exclude_models,
        sample_prompt=sample_prompt,
        include_human=include_human,
    )


def compute_mean_annotation_records(
    human_datasets: HumanDatasets,
    annotations_dir: Path,
    dataset_keys: Iterable[str],
    prompt_names: Iterable[str],
    exclude_models: Iterable[str] | None = None,
    sample_prompt: str = "default",
    include_human: bool = True,
) -> pd.DataFrame:
    """Per-comment mean annotation records (column "annotation"), see
    _compute_per_comment_records and per_comment_mean_annotation."""
    return _compute_per_comment_records(
        human_datasets,
        annotations_dir,
        per_comment=per_comment_mean_annotation,
        value_col="annotation",
        dataset_keys=dataset_keys,
        prompt_names=prompt_names,
        exclude_models=exclude_models,
        sample_prompt=sample_prompt,
        include_human=include_human,
    )


# ---------------------------------------------------------------------------
# Mean nDFU tables
# ---------------------------------------------------------------------------


class MeanSE(NamedTuple):
    """Mean and standard error of a sample: mean is NaN for an empty sample,
    se is NaN for fewer than two values."""

    mean: float
    se: float


def mean_se(values: pd.Series) -> MeanSE:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return MeanSE(np.nan, np.nan)
    se = (
        values.std(ddof=1) / np.sqrt(len(values))
        if len(values) > 1
        else np.nan
    )
    return MeanSE(values.mean(), se)


def _datasets_in_order(df: pd.DataFrame) -> list[str]:
    return list(df["Dataset"].drop_duplicates())


def _matched_model_rows(
    model_df: pd.DataFrame,
    human_df: pd.DataFrame,
    datasets: list[str],
    prompts: list[str],
) -> dict[str, pd.DataFrame]:
    """dataset -> rows of one model restricted to the comments that the
    human sample and every prompt the model has for that dataset share
    (datasets without any of `prompts` are left out)."""
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
    return matched


def _n_comments(sub: pd.DataFrame) -> int:
    """Number of comments of `sub` (counted under its first prompt; all of
    its prompts share the same comments)."""
    first = sub.loc[sub["Prompt"] == sub["Prompt"].iloc[0]]
    return first["text_id"].nunique()


def compute_mean_ndfu_table(
    ndfu_df: pd.DataFrame, prompt_names: Iterable[str]
) -> pd.DataFrame:
    """
    Mean (SE) per-comment nDFU of the human annotations and of each model
    under each prompt. Index (Model, Dataset): a leading HUMAN_LABEL block,
    then one block per model, each with a pooled POOLED_LABEL row and one
    row per dataset. Columns: `n` (comments), then one MeanSE per prompt
    (the prompts in `prompt_names` that have results). Human annotations do
    not depend on the prompt, so the human rows hold the same value in
    every prompt column. Within a model row only comments that the human
    sample and every available prompt share are used, so all cells in a
    row use the same comments.
    """
    human_df = ndfu_df.loc[ndfu_df["Model"] == HUMAN_LABEL]
    llm_df = ndfu_df.loc[ndfu_df["Model"] != HUMAN_LABEL]
    prompts = [p for p in prompt_names if p in set(llm_df["Prompt"])]
    datasets = _datasets_in_order(human_df)

    def _human_row(sub: pd.DataFrame) -> dict[str, object]:
        value = mean_se(sub["ndfu"])
        return {"n": len(sub), **{p: value for p in prompts}}

    def _model_row(sub: pd.DataFrame) -> dict[str, object]:
        return {
            "n": _n_comments(sub),
            **{
                p: mean_se(sub.loc[sub["Prompt"] == p, "ndfu"])
                for p in prompts
            },
        }

    rows = [_human_row(human_df)]
    index = [(HUMAN_LABEL, POOLED_LABEL)]
    for d in datasets:
        rows.append(_human_row(human_df.loc[human_df["Dataset"] == d]))
        index.append((HUMAN_LABEL, d))

    for model in order_models(set(llm_df["Model"])):
        matched = _matched_model_rows(
            llm_df.loc[llm_df["Model"] == model], human_df, datasets, prompts
        )
        if not matched:
            continue
        pooled = _model_row(pd.concat(matched.values()))
        pooled["n"] = sum(
            _n_comments(m) for m in matched.values() if not m.empty
        )
        rows.append(pooled)
        index.append((model, POOLED_LABEL))
        for d, sub in matched.items():
            if sub.empty:
                continue
            rows.append(_model_row(sub))
            index.append((model, d))

    return pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(index, names=["Model", "Dataset"]),
        columns=["n", *prompts],
    )


def compute_default_mean_ndfu_table(
    ndfu_df: pd.DataFrame,
    prompt_name: str,
    no_persona_prompt_name: str,
) -> pd.DataFrame:
    """
    Mean (SE) per-comment nDFU under `no_persona_prompt_name` and under
    `prompt_name`: a HUMAN_LABEL row first, then one row per model; columns
    are a (Dataset, Prompt) MultiIndex with the no-persona prompt first,
    cells are MeanSE. Model rows only use comments that also appear in the
    human sample. Human annotations do not depend on the prompt, so the
    human row holds the same value under both prompts of a dataset.
    """
    human_df = ndfu_df.loc[ndfu_df["Model"] == HUMAN_LABEL]
    prompts = [no_persona_prompt_name, prompt_name]
    llm_df = ndfu_df.loc[
        (ndfu_df["Model"] != HUMAN_LABEL) & ndfu_df["Prompt"].isin(prompts)
    ]
    datasets = _datasets_in_order(human_df)

    human_row = {}
    for d in datasets:
        value = mean_se(human_df.loc[human_df["Dataset"] == d, "ndfu"])
        for prompt in prompts:
            human_row[(d, prompt)] = value

    def _model_row(model_df: pd.DataFrame) -> dict[tuple[str, str], MeanSE]:
        row = {}
        for d in datasets:
            ids = set(human_df.loc[human_df["Dataset"] == d, "text_id"])
            for prompt in prompts:
                sub = model_df.loc[
                    (model_df["Dataset"] == d)
                    & (model_df["Prompt"] == prompt)
                    & model_df["text_id"].isin(ids)
                ]
                row[(d, prompt)] = mean_se(sub["ndfu"])
        return row

    rows = {HUMAN_LABEL: human_row}
    for model in order_models(set(llm_df["Model"])):
        rows[model] = _model_row(llm_df.loc[llm_df["Model"] == model])

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.columns = pd.MultiIndex.from_tuples(
        list(out.columns), names=["Dataset", "Prompt"]
    )
    out.index.name = "Model"
    return out


# ---------------------------------------------------------------------------
# Spearman correlation of per-comment values: human vs. LLM, per prompt
# ---------------------------------------------------------------------------


def _spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """(rho, p) of two aligned series; NaN when undefined (fewer than three
    points or a constant series)."""
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan, np.nan
    res = stats.spearmanr(x, y)
    return float(res.statistic), float(res.pvalue)


def compute_human_llm_correlations(
    records_df: pd.DataFrame,
    prompt_names: list[str],
    value_col: str = "annotation",
    correction_method: str | None = "holm",
) -> pd.DataFrame:
    """
    Spearman correlation, across the comments of a dataset, between a
    per-comment value of the human annotations and of each model's
    annotations, per (model, dataset, prompt); by default the value is the
    mean annotation of each comment.

    `records_df` holds long records with a `value_col` column, e.g. from
    compute_mean_annotation_records (human rows have Model ==
    HUMAN_LABEL); include the no-persona records to get the no-persona
    column. Comments are matched by text_id. For each (model, dataset),
    only the comments that the human sample and every prompt the model has
    for that dataset share are used, so all of its prompts are compared
    on the same `n` comments.

    Returns one row per (Model, Dataset, Prompt) with `n`, `spearman_rho`,
    `p_raw` and, unless `correction_method` is None, `p_corrected`
    (corrected over every correlation in the result).
    """
    is_human = records_df["Model"] == HUMAN_LABEL
    human_df = records_df.loc[is_human, ["Dataset", "text_id", value_col]]
    human_df = human_df.drop_duplicates(["Dataset", "text_id"])
    llm_df = records_df.loc[
        ~is_human & records_df["Prompt"].isin(set(prompt_names))
    ]

    records = []
    for model in order_models(set(llm_df["Model"])):
        for dataset in _datasets_in_order(human_df):
            sub = llm_df.loc[
                (llm_df["Model"] == model) & (llm_df["Dataset"] == dataset)
            ]
            if sub.empty:
                continue
            human = human_df.loc[human_df["Dataset"] == dataset].set_index(
                "text_id"
            )[value_col]
            by_prompt = {
                prompt: g.set_index("text_id")[value_col]
                for prompt, g in sub.groupby("Prompt")
            }
            ids = sorted(
                set(human.index).intersection(
                    *(set(v.index) for v in by_prompt.values())
                )
            )
            for prompt in prompt_names:
                if prompt not in by_prompt:
                    continue
                rho, p = _spearman(human.loc[ids], by_prompt[prompt].loc[ids])
                records.append(
                    {
                        "Model": model,
                        "Dataset": dataset,
                        "Prompt": prompt,
                        "n": len(ids),
                        "spearman_rho": rho,
                        "p_raw": p,
                    }
                )

    out = pd.DataFrame(
        records,
        columns=["Model", "Dataset", "Prompt", "n", "spearman_rho", "p_raw"],
    )
    if correction_method is None or out.empty:
        return out
    out["p_corrected"] = np.nan
    tested = out["p_raw"].notna()
    if tested.any():
        out.loc[tested, "p_corrected"] = multipletests(
            out.loc[tested, "p_raw"], alpha=0.05, method=correction_method
        )[1]
    return out
