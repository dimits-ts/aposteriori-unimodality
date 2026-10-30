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

from ..lib import labels, run_helper
from ..lib.util import (
    center_table_latex,
    small_table_latex,
    trim_numeric_col_latex,
    significance_superscript,
)
from .shared import (
    APUNIM_TABLE_EXCLUDE_MODELS,
    DATASET_KEYS,
    _human_sample_dataset,
    HumanDatasets,
    MAIN_PROMPT_NAMES,
    LLMAnnotationDataset,
    _available_dataset_keys,
    _key_columns,
    _keyed_series,
    _order_models,
    find_annotation_files,
    load_llm_df,
    PROMPT_COMPARISON_DATASET_KEYS,
)

# Prompt-variant names used by the paraphrase ablation
# (instructions/ablation/<dataset>/variant{1,2,3}.txt).
VARIANT_NAMES = ["variant1", "variant2", "variant3"]


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


def find_repeat_files(
    directory: Path, dataset_key: str, prompt_name: str
) -> dict[str, dict[str, Path]]:
    """
    Returns {model_pseudo: {run_label: path}} for every file in `directory`
    matching f"{dataset_key}-{prompt_name}-<pseudo>-run<N>.csv" (e.g. as
    written by the repeat ablation in annotate_experiments.sh: the same
    prompt run N times over the same 10% sub-sample).
    """
    import re

    pattern = re.compile(
        rf"^{re.escape(dataset_key)}-{re.escape(prompt_name)}-"
        rf"([^-.]+)-(run\d+)\.csv$"
    )
    out: dict[str, dict[str, Path]] = {}
    if not directory.exists():
        return out
    for path in sorted(
        directory.glob(f"{dataset_key}-{prompt_name}-*-run*.csv")
    ):
        m = pattern.match(path.name)
        if m:
            pseudo, run_label = m.group(1), m.group(2)
            out.setdefault(pseudo, {})[run_label] = path
    return out


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
    df = labels.relabel_frame(df)
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
    df = labels.relabel_frame(df).rename(
        columns={"SDB Feature": labels.SDD.tex}
    )

    # The prompt columns already hold formatted "value$^{***}$" strings (see
    # _apunim_prompt_cell); they must not be passed through
    # trim_numeric_col_latex, which would turn the starred cells into "---".
    df = df.replace("_", r"\_", regex=True).set_index(
        [labels.SDD.tex, "Value", "Model"]
    )

    latex_str = df.to_latex(
        caption=(
            "Aposteriori unimodality results for the LLM annotations of "
            f"the {labels.dataset_label(dataset_name)} dataset, across "
            "instruction prompts. "
            r"$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$."
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


# ---------------------------------------------------------------------------
# Significant PC dimensions: human vs. LLM annotations, per prompt
# ---------------------------------------------------------------------------

# apunim's p-values already account for testing several PC dimensions, so a
# dimension counts as significant when its reported p-value is below this.
SIGNIFICANCE_ALPHA = 0.05

# Symbols of the significance table (LaTeX; \sigyes and \signo are macros
# defined in the paper preamble). The legend is written into the caption.
_SYM_AGREE = r"\sigyes"  # agree on significance (and on sign, if significant)
_SYM_DISAGREE = r"\signo"  # significant for only one of the two
_SYM_SIGN_DIFFERS = r"$\pm$"  # both significant, but the sign differs

# Colour macros for the symbols, defined in the paper preamble as
# \newcommand{\sigcol...}[1]{\textcolor[HTML]{...}{#1}} with colours from
# graphs.COLORBLIND_PALETTE. Colour only separates the categories; every
# category is also distinguished by its symbol (or, for the two kinds of
# agreement, explained in the caption).
_COL_AGREE_SIGNIFICANT = r"\sigcolagree"  # both significant, same sign
_COL_AGREE_NEITHER = r"\sigcolneither"  # neither significant
_COL_SIGN_DIFFERS = r"\sigcolsign"  # both significant, sign differs
_COL_DISAGREE = r"\sigcoldisagree"  # significant for only one of the two


def _colored(color_macro: str, symbol: str) -> str:
    return rf"{color_macro}{{{symbol}}}"


# Coloured symbol per comparison outcome (used in cells and caption alike).
_CELL_AGREE_SIGNIFICANT = _colored(_COL_AGREE_SIGNIFICANT, _SYM_AGREE)
_CELL_AGREE_NEITHER = _colored(_COL_AGREE_NEITHER, _SYM_AGREE)
_CELL_SIGN_DIFFERS = _colored(_COL_SIGN_DIFFERS, _SYM_SIGN_DIFFERS)
_CELL_DISAGREE = _colored(_COL_DISAGREE, _SYM_DISAGREE)

# Each prompt gets a fixed-width slot (in em), so that the symbols of
# different rows line up and a prompt without LLM results is simply blank.
# It can be set per table (`slot_width_em`); the symbols are roughly an em
# wide, so much narrower slots make them touch.
DEFAULT_SLOT_WIDTH_EM = 1.2


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


def _slot(symbol: str, width_em: float) -> str:
    return rf"\makebox[{width_em:g}em][c]{{{symbol}}}"


def _comparison_symbol(
    llm_significant: bool, human_significant: bool, sign_differs: bool
) -> str:
    """
    How the LLM result for a dimension compares with the human one, as a
    coloured symbol: disagreement on significance, both significant but
    with a different sign for some group, or agreement (both significant,
    or neither, which get different colours).
    """
    if llm_significant != human_significant:
        return _CELL_DISAGREE
    if llm_significant and sign_differs:
        return _CELL_SIGN_DIFFERS
    if llm_significant:
        return _CELL_AGREE_SIGNIFICANT
    return _CELL_AGREE_NEITHER


def _llm_cell(
    llm_results: list[bool | None],
    human_significant: bool | None,
    sign_differs: list[bool],
    slot_width_em: float = DEFAULT_SLOT_WIDTH_EM,
) -> str:
    """
    One slot per prompt (in prompt order); a prompt without an LLM result is
    a blank slot, and a model without any result gives an empty cell. A
    dimension missing from the human results counts as not significant.
    `sign_differs` says, per prompt, whether some group's sign differs from
    the human one; `slot_width_em` is the width of each prompt slot.
    """
    if all(llm is None for llm in llm_results):
        return ""
    return "".join(
        _slot(
            (
                ""
                if llm is None
                else _comparison_symbol(llm, bool(human_significant), differs)
            ),
            slot_width_em,
        )
        for llm, differs in zip(llm_results, sign_differs)
    )


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

    def groups(df):
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
    slot_width_em: float,
) -> list[dict]:
    """Table rows (one per PC dimension) of one dataset."""
    human = _dimension_significance(human_df, alpha)
    pairs = _llm_pair_results(
        llm_df,
        models,
        prompts,
        alpha,
        _significant_group_signs(human_df, alpha),
    )

    llm_dims = set()
    for significance, _ in pairs.values():
        llm_dims |= set(significance)
    _report_unmatched(dataset_name, set(human), llm_dims, human_df, llm_df)

    rows = []
    for dim in sorted(llm_dims):
        cells = {}
        for m in models:
            results = [pairs[(m, p)][0].get(dim) for p in prompts]
            differs = [dim in pairs[(m, p)][1] for p in prompts]
            cells[m] = _llm_cell(
                results, human.get(dim), differs, slot_width_em
            )
        rows.append({"Dataset": dataset_name, "SDB Feature": dim, **cells})
    return rows


def build_apunim_significance_table(
    inputs: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    prompt_names: list[str] = MAIN_PROMPT_NAMES,
    alpha: float = SIGNIFICANCE_ALPHA,
    slot_width_em: float = DEFAULT_SLOT_WIDTH_EM,
) -> pd.DataFrame:
    """
    One row per (dataset, PC dimension) for which the LLMs have results, and
    one column per model. `inputs` maps a dataset name to its (human, LLM)
    apunim results.

    A model cell holds one symbol per prompt (in `prompt_names` order)
    saying how the model's result for the dimension compares with the human
    one (see _comparison_symbol); a prompt without results is blank.
    `slot_width_em` sets the width of each prompt slot, i.e. the cell size.
    The prompts used are stored in `.attrs["prompts"]`.
    """
    llm_dfs = [llm_df for _, llm_df in inputs.values()]
    models = _order_models({m for df in llm_dfs for m in df["Model"]})
    prompts = [
        p
        for p in prompt_names
        if any(p in set(df["Prompt"]) for df in llm_dfs)
    ]

    rows = []
    for dataset_name, (human_df, llm_df) in inputs.items():
        rows.extend(
            _dataset_significance_rows(
                dataset_name,
                human_df,
                llm_df,
                models,
                prompts,
                alpha,
                slot_width_em,
            )
        )
    out = pd.DataFrame(rows)
    out.attrs["prompts"] = prompts
    return out


def export_apunim_significance_table(
    table_df: pd.DataFrame,
    output_path: Path,
    label: str,
    alpha: float = SIGNIFICANCE_ALPHA,
    note: str = "",
    table_size: str | None = None,
) -> None:
    if table_df.empty:
        print(f"No apunim significance results; skipping {output_path}.")
        return
    prompts = table_df.attrs.get("prompts", [])
    caption = (
        rf"Statistically significant ($p<{alpha}$) {labels.SDD.tex_plural} of "
        "each LLM compared with the human annotations, per dataset. A "
        "dimension counts as significant if at least one of its groups is. "
        "Each cell shows one symbol per prompt, in the order "
        f"{', '.join(prompts)}: "
        rf"{_CELL_AGREE_SIGNIFICANT}~both significant, with the same sign "
        "for every group that is significant in both; "
        rf"{_CELL_AGREE_NEITHER}~neither significant; "
        rf"{_CELL_SIGN_DIFFERS}~both significant, but the sign of at least "
        "one group's effect differs; "
        rf"{_CELL_DISAGREE}~significant for only one of the two; "
        "blank: no LLM result." + note
    )

    # Model columns and short dataset names (narrow table).
    df = labels.relabel_frame(table_df, short=True)
    df = df.rename(
        columns={
            **{c: labels.model_label(c) for c in df.columns},
            "SDB Feature": labels.SDD.tex,
            "Dataset": "Ds",
        }
    )
    df = df.replace("_", r"\_", regex=True).set_index(["Ds", labels.SDD.tex])
    latex_str = df.to_latex(
        caption=caption,
        label=label,
        escape=False,
        position="t",
        index=True,
        multirow=True,
        longtable=False,
        column_format="ll" + "c" * len(df.columns),
    )
    latex_str = "\\setlength{\\tabcolsep}{3pt}\n" + latex_str + "\n"
    if table_size is not None:
        latex_str = small_table_latex(latex_str, size=table_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_str)
    print(f"Table exported to {output_path.resolve()}")


# ---------------------------------------------------------------------------
# Number of significant PC dimensions per model and dataset (and prompt)
# ---------------------------------------------------------------------------

# Prompt label of the human rows in count_significant_dimensions.
_HUMAN_PROMPT = "human"
_COUNT_MISSING = "---"
# Marks a ratio over fewer dimensions than the dataset has.
_PARTIAL_MARK = r"$^\dagger$"


def count_significant_dimensions(
    inputs: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    prompt_names: list[str] = MAIN_PROMPT_NAMES,
    alpha: float = SIGNIFICANCE_ALPHA,
) -> pd.DataFrame:
    """
    One row per (Dataset, Model, Prompt): how many PC dimensions are
    significant (`significant`) out of those with results (`tested`), with
    significance decided as in the significance table
    (_dimension_significance). `inputs` is the same mapping as for
    build_apunim_significance_table; the human results are included as
    Model HUMAN_LABEL, Prompt "human".
    """
    rows = []
    for dataset_name, (human_df, llm_df) in inputs.items():
        rows.append(
            _dimension_count(
                dataset_name, HUMAN_LABEL, _HUMAN_PROMPT, human_df, alpha
            )
        )
        for (model, prompt), pair_df in llm_df.groupby(
            ["Model", "Prompt"], sort=False
        ):
            if prompt in prompt_names:
                rows.append(
                    _dimension_count(
                        dataset_name, model, prompt, pair_df, alpha
                    )
                )
    return pd.DataFrame(
        rows,
        columns=["Dataset", "Model", "Prompt", "significant", "tested"],
    )


def _dimension_count(
    dataset_name: str, model: str, prompt: str, df: pd.DataFrame, alpha: float
) -> dict:
    significance = _dimension_significance(df, alpha)
    return {
        "Dataset": dataset_name,
        "Model": model,
        "Prompt": prompt,
        "significant": sum(significance.values()),
        "tested": len(significance),
    }


def _count_layout(
    counts: pd.DataFrame,
) -> tuple[list[str], list[tuple[list[str], str]], dict[str, int]]:
    """(LLM models in order; row groups, one per dataset, as
    (datasets, label); label -> number of PC dimensions)."""
    llm_models = _order_models(set(counts["Model"]) - {HUMAN_LABEL})
    datasets = list(dict.fromkeys(counts["Dataset"]))
    n_dims = counts.groupby("Dataset")["tested"].max().to_dict()
    groups = [([d], d) for d in datasets]
    return llm_models, groups, n_dims


def _ratio_cell(
    counts: pd.DataFrame,
    model: str,
    prompt: str,
    datasets: list[str],
    n_dimensions: int,
) -> str:
    """Share of the tested PC dimensions that `model` finds significant
    under `prompt`, over `datasets`.
    Marked with _PARTIAL_MARK if the model has results for fewer than
    `n_dimensions` dimensions."""
    if model == HUMAN_LABEL:
        prompt = _HUMAN_PROMPT
    rows = counts.loc[
        (counts["Model"] == model)
        & (counts["Prompt"] == prompt)
        & counts["Dataset"].isin(datasets)
    ]
    tested = rows["tested"].sum()
    if tested == 0:
        return _COUNT_MISSING
    ratio = rf"{int(rows['significant'].sum() / tested*100)}\%"
    return ratio if tested == n_dimensions else ratio + _PARTIAL_MARK


def _count_row_label(name: str) -> str:
    return labels.dataset_label(name, short=True).replace("_", r"\_")


def _model_header(
    leading: list[str], llm_models: list[str], rotate: bool = False
) -> str:
    """Header row: the leading columns, then the model names (rotated by
    90 degrees if `rotate`)."""
    names = [
        labels.model_label(m).replace("_", r"\_")
        for m in [HUMAN_LABEL] + llm_models
    ]
    if rotate:
        names = [rf"\rotatebox{{90}}{{{name}}}" for name in names]
    return " & ".join(leading + names) + r" \\"


def _model_col_spec(leading: str, n_llm_models: int) -> str:
    """Leading columns, the Human column, a gap, then one per LLM."""
    return leading + r"c@{\hspace{10pt}}" + "c" * n_llm_models


def _count_caption_note(counts: pd.DataFrame) -> str:
    """Explains _PARTIAL_MARK, if the table uses it."""
    n_dims = counts.groupby("Dataset")["tested"].transform("max")
    if not (counts["tested"] < n_dims).any():
        return ""
    return (
        f" {_PARTIAL_MARK}: ratio over fewer dimensions, as the model has "
        "results for only some of them."
    )


def _count_table_latex(
    caption: str, label: str, col_spec: str, body: list[str]
) -> str:
    lines = [
        r"\begin{table}[t]",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\centering",
        r"\normalsize",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    table = "\n".join(lines) + "\n"
    return table


def _write_count_table(latex_str: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_str)
    print(f"Table exported to {output_path.resolve()}")


def export_significant_dimension_summary(
    counts: pd.DataFrame,
    output_path: Path,
    label: str,
    prompt_name: str = "default",
    alpha: float = SIGNIFICANCE_ALPHA,
    note: str = "",
) -> None:
    """
    Concise table: one row per dataset, a column with the number
    of PC dimensions (n), then one column per model (Human first) with the
    share of dimensions found significant under `prompt_name`.
    """
    if counts.empty:
        print(f"No significance counts; skipping {output_path}.")
        return
    llm_models, groups, n_dims = _count_layout(counts)

    body = [_model_header(["Dataset", "$n$"], llm_models), r"\midrule"]
    for datasets, name in groups:
        cells = [
            _ratio_cell(counts, m, prompt_name, datasets, n_dims[name])
            for m in [HUMAN_LABEL] + llm_models
        ]
        row = [_count_row_label(name), str(n_dims[name]), *cells]
        body.append(" & ".join(row) + r" \\")

    caption = (
        rf"Share of \acp{{sdd}} that are statistically significant "
        rf"($p<{alpha}$) in the human annotations and in each LLM's "
        f"annotations under the {prompt_name} prompt, per dataset; $n$ is "
        "the number of dimensions. A dimension counts as significant if at "
        "least one of its groups is." + _count_caption_note(counts) + note
    )
    _write_count_table(
        _count_table_latex(
            caption, label, _model_col_spec("lr", len(llm_models)), body
        ),
        output_path,
    )


def export_significant_dimension_by_prompt(
    counts: pd.DataFrame,
    output_path: Path,
    label: str,
    prompt_names: list[str] = MAIN_PROMPT_NAMES,
    alpha: float = SIGNIFICANCE_ALPHA,
    note: str = "",
) -> None:
    """
    Full table: for each dataset, one row per prompt, with the
    number of PC dimensions (n) and one column per model (Human first)
    holding the share of dimensions found significant. Human annotations
    do not depend on the prompt, so their value spans the dataset's rows.
    """
    if counts.empty:
        print(f"No significance counts; skipping {output_path}.")
        return
    llm_models, groups, n_dims = _count_layout(counts)
    prompts = [p for p in prompt_names if p in set(counts["Prompt"])]

    body = [
        _model_header(["Dataset", "$n$", "Prompt"], llm_models, rotate=True)
    ]
    for datasets, name in groups:
        body.append(r"\midrule")
        body += _by_prompt_rows(
            counts, datasets, name, prompts, llm_models, n_dims[name]
        )

    caption = (
        rf"Share of \acp{{sdd}} that are statistically significant "
        rf"($p<{alpha}$) in the human annotations and in each LLM's "
        "annotations under each instruction prompt, per dataset; $n$ is the "
        "number of dimensions. A dimension counts as significant if at least "
        "one of its groups is; human annotations do not depend on the prompt."
        + _count_caption_note(counts)
        + note
    )
    _write_count_table(
        _count_table_latex(
            caption, label, _model_col_spec("lrl", len(llm_models)), body
        ),
        output_path,
    )


def _by_prompt_rows(
    counts: pd.DataFrame,
    datasets: list[str],
    name: str,
    prompts: list[str],
    llm_models: list[str],
    n_dimensions: int,
) -> list[str]:
    """One row per prompt for one dataset group; the dataset name, n and
    the human value span the group's rows."""

    def spanning(text: str) -> str:
        return rf"\multirow{{{len(prompts)}}}{{*}}{{{text}}}"

    human = _ratio_cell(counts, HUMAN_LABEL, "", datasets, n_dimensions)
    rows = []
    for i, prompt in enumerate(prompts):
        first = i == 0
        lead = [
            spanning(_count_row_label(name)) if first else "",
            spanning(str(n_dimensions)) if first else "",
            prompt.capitalize().replace("_", r"\_"),
            spanning(human) if first else "",
        ]
        cells = [
            _ratio_cell(counts, m, prompt, datasets, n_dimensions)
            for m in llm_models
        ]
        rows.append(" & ".join(lead + cells) + r" \\")
    return rows


def _apunim_prompt_cell(row: pd.Series) -> str:
    if pd.isna(row["apunim"]):
        return "---"
    stars = significance_superscript(row["pvalue"])
    return f"{row['apunim']:.3f}{stars}"


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
    One-way ANOVA per (dataset, model) testing whether the per-comment nDFU
    distribution shifts across instruction prompts, with corrected p-values
    and Cohen's d effect sizes vs. the baseline prompt.

    Each comment contributes exactly one nDFU value per prompt (computed on
    all of its annotations). The test and the effect size are therefore
    computed once per (dataset, model) on the comment-level values, instead
    of once per demographic group: per-group samples are near-copies of the
    same comments, so per-group tests would not be independent evidence.
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
                ds = LLMAnnotationDataset(
                    load_llm_df(path), dataset_key, model, prompt_name
                )
                values = _per_comment_ndfu(ds).dropna().to_numpy()
                if len(values) >= 2:
                    ndfu_by_prompt[prompt_name] = values

            samples = list(ndfu_by_prompt.values())
            if len(samples) < 2:
                continue
            f_stat, p_value = stats.f_oneway(*samples)

            baseline_values = ndfu_by_prompt.get(baseline_prompt)
            record = {
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
                if prompt_name == baseline_prompt:
                    continue
                prompt_values = ndfu_by_prompt.get(prompt_name)
                record[f"cohens_d_{prompt_name}_vs_{baseline_prompt}"] = (
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
    Compact Cohen's d summary: one row per (Model, Dataset), one column per
    non-baseline prompt, and a leading `n` column with the number of
    comments the effect size is computed over. Each cell is a single
    Cohen's d (prompt vs. `baseline_prompt`) on per-comment nDFU. A pooled
    `pooled_label` row per model gives the unweighted mean of the
    per-dataset values (and the total comment count).
    """
    cols = {
        p: f"cohens_d_{p}_vs_{baseline_prompt}"
        for p in prompt_names
        if p != baseline_prompt
        and f"cohens_d_{p}_vs_{baseline_prompt}" in result_df.columns
    }

    def _row(sub: pd.DataFrame) -> dict:
        n = pd.to_numeric(sub["n_comments"], errors="coerce").sum()
        row = {"n": str(int(n))}
        for prompt_name, col in cols.items():
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            row[prompt_name] = "---" if vals.empty else f"{vals.mean():.2f}"
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
    compact_df = labels.relabel_frame(compact_df)
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


# ---------------------------------------------------------------------------
# Mean nDFU: human annotations vs. LLM annotations (per instruction prompt)
# ---------------------------------------------------------------------------

HUMAN_LABEL = "Human"
_SPAN_MARK = "@@SPAN@@"

# Prompt name of the no-persona baseline runs (llm_annotate.py
# --no-persona), i.e. the files "<dataset>-nopersona-<model>.csv" in the
# annotations directory, next to the persona runs.
NO_PERSONA_PROMPT_NAME = "nopersona"

# Column labels of the default-prompt mean nDFU table.
NO_PERSONA_COLUMN = "No Persona"
WITH_PERSONA_COLUMN = "With Persona"


# Per-comment nDFU (shared with the human analysis scripts).
_per_comment_ndfu = run_helper.per_comment_ndfu


def compute_mean_ndfu_records(
    human_datasets: HumanDatasets,
    annotations_dir: Path,
    dataset_keys: list[str] = PROMPT_COMPARISON_DATASET_KEYS,
    prompt_names: list[str] = MAIN_PROMPT_NAMES,
    exclude_models: set[str] | None = APUNIM_TABLE_EXCLUDE_MODELS,
    sample_prompt: str = "default",
    include_human: bool = True,
) -> pd.DataFrame:
    """
    Long-format per-comment nDFU, one row per (Dataset, Model, Prompt,
    text_id). Human rows have Model == HUMAN_LABEL and Prompt == "human";
    they are computed on the comments actually sampled for the LLM runs
    (see `_human_sample_dataset`), so human and LLM nDFU are comparable
    comment by comment. With `include_human=False` only LLM rows are
    returned (e.g. to add further prompts to an existing set of records).
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
        if include_human:
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

    # Explicit columns so an empty result still round-trips through a CSV.
    return pd.DataFrame(
        records, columns=["Dataset", "Model", "Prompt", "text_id", "ndfu"]
    )


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
    no_persona_prompt_name: str = NO_PERSONA_PROMPT_NAME,
) -> pd.DataFrame:
    """
    Human row first, then one row per model. For each dataset a
    NO_PERSONA_COLUMN column with the mean (SE) per-comment nDFU under
    `no_persona_prompt_name` and a WITH_PERSONA_COLUMN column with the same
    under `prompt_name` (columns are a (Dataset, column) MultiIndex). Model
    rows only use comments that also appear in the human sample. Human
    annotations do not depend on the prompt, so the Human row holds the
    same value in both columns of a dataset (the exporter merges them).
    """
    human_df = ndfu_df.loc[ndfu_df["Model"] == HUMAN_LABEL]
    columns = {
        NO_PERSONA_COLUMN: no_persona_prompt_name,
        WITH_PERSONA_COLUMN: prompt_name,
    }
    llm_df = ndfu_df.loc[
        (ndfu_df["Model"] != HUMAN_LABEL)
        & ndfu_df["Prompt"].isin(set(columns.values()))
    ]
    datasets = _datasets_in_order(human_df)

    human_row = {}
    for d in datasets:
        value = _mean_se_str(human_df.loc[human_df["Dataset"] == d, "ndfu"])
        for col in columns:
            human_row[(d, col)] = value

    def _model_row(model_df: pd.DataFrame) -> dict:
        row = {}
        for d in datasets:
            ids = set(human_df.loc[human_df["Dataset"] == d, "text_id"])
            for col, prompt in columns.items():
                sub = model_df.loc[
                    (model_df["Dataset"] == d)
                    & (model_df["Prompt"] == prompt)
                    & model_df["text_id"].isin(ids)
                ]
                row[(d, col)] = _mean_se_str(sub["ndfu"])
        return row

    rows = {HUMAN_LABEL: human_row}
    for model in _order_models(set(llm_df["Model"])):
        rows[model] = _model_row(llm_df.loc[llm_df["Model"] == model])

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.columns = pd.MultiIndex.from_tuples(
        out.columns, names=["Dataset", None]
    )
    out.index.name = "Model"
    return out