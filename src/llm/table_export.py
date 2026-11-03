"""
LaTeX export of every table of the LLM analysis.
"""

import math
from collections.abc import Collection, Iterable
from pathlib import Path
from typing import cast

import pandas as pd

from ..lib import labels
from ..lib.util import (
    SIGNIFICANCE_LEGEND,
    SIGNIFICANCE_LEVELS,
    center_table_latex,
    significance_superscript,
    small_table_latex,
    trim_numeric_col_latex,
)
from .shared import (
    HUMAN_LABEL,
    HUMAN_PROMPT_NAME,
    NO_PERSONA_PROMPT_NAME,
    POOLED_LABEL,
    order_models,
)
from .stats import SIGNIFICANCE_ALPHA, MeanSE, SignificanceOutcome

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Missing cell. The Cohen's d and default-prompt mean nDFU tables use the
# shorter MISSING_NARROW to save width.
MISSING = "---"
MISSING_NARROW = "--"

# Font size of the tables with one column (group) per model: the version
# with every model has more columns than the "excluding" version, so it is
# set smaller.
TABLE_SIZE_ALL_MODELS = r"\footnotesize"
TABLE_SIZE_EXCLUDING = r"\normalsize"

# Gap between the Human column and the LLM columns, and between the
# no-persona column and the prompt columns.
COLUMN_GROUP_GAP = r"@{\hspace{10pt}}"

# Display labels of the no-persona and persona (default prompt) columns.
NO_PERSONA_COLUMN = "No Persona"
WITH_PERSONA_COLUMN = "With Persona"

# Krippendorff's alpha in the consistency tables.
CONSISTENCY_ALPHA_FORMAT = ".4f"

# Display names of the statuses in the non-standard answers table.
NONSTANDARD_STATUS_LABELS = {
    "unparsed": "Unparsed",
    "out_of_range": "Out of range",
}

# Cohen's d table: column padding, and the gap between dataset groups.
COHENS_D_TABCOLSEP_PT = 1.5
COHENS_D_GROUP_GAP_PT = 4

# Default-prompt mean nDFU table: a tabularx spanning the full line width,
# so the value columns share the spare width evenly and adjacent
# "mean (SE)" cells never touch. Needs \usepackage{tabularx}.
MEAN_NDFU_FONT_SIZE = r"\normalsize"
MEAN_NDFU_TABLE_WIDTH = r"\linewidth"
# Centered paragraph column that stretches to fill the table width.
MEAN_NDFU_VALUE_COL = r">{\centering\arraybackslash}X"

# Human-LLM correlation table.
CORRELATION_TABLE_SIZE = r"\small"

# Significance table symbols (LaTeX; \sigyes and \signo are macros defined
# in the paper preamble). The legend is written into the caption.
_SYM_AGREE = r"\sigyes"  # agree on significance (and on sign, if significant)
_SYM_DISAGREE = r"\signo"  # significant for only one of the two
_SYM_SIGN_DIFFERS = r"$\pm$"  # both significant, but the sign differs

# Colour macros for the symbols, defined in the paper preamble as
# \newcommand{\sigcol...}[1]{\textcolor[HTML]{...}{#1}} with colours from
# graphs.COLORBLIND_PALETTE. Colour only separates the categories; every
# category is also distinguished by its symbol (or, for the two kinds of
# agreement, explained in the caption).
SIGNIFICANCE_CELLS = {
    SignificanceOutcome.AGREE_SIGNIFICANT: rf"\sigcolagree{{{_SYM_AGREE}}}",
    SignificanceOutcome.AGREE_NEITHER: rf"\sigcolneither{{{_SYM_AGREE}}}",
    SignificanceOutcome.SIGN_DIFFERS: rf"\sigcolsign{{{_SYM_SIGN_DIFFERS}}}",
    SignificanceOutcome.DISAGREE: rf"\sigcoldisagree{{{_SYM_DISAGREE}}}",
}

# Each prompt gets a fixed-width slot (in em) in a significance-table cell,
# so that the symbols of different rows line up. The symbols are roughly
# an em wide, so much narrower slots make them touch; the all-models
# version has more columns, so its slots are narrower.
SIGNIFICANCE_SLOT_WIDTH_ALL_MODELS_EM = 0.9
SIGNIFICANCE_SLOT_WIDTH_EXCLUDING_EM = 1.2
SIGNIFICANCE_TABCOLSEP_PT = 3

# Marks a share over fewer dimensions than the dataset has, in the
# significance count tables.
PARTIAL_MARK = r"$^\dagger$"

# Placeholder for a value cell that should span the rest of its row; see
# _merge_span_cells.
_SPAN_MARK = "@@SPAN@@"

_TEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _model_table_size(all_models: bool) -> str:
    return TABLE_SIZE_ALL_MODELS if all_models else TABLE_SIZE_EXCLUDING


def _tex_escape(text: object) -> str:
    return "".join(_TEX_SPECIALS.get(char, char) for char in str(text))


def _escape_underscores(text: object) -> str:
    return str(text).replace("_", r"\_")


def _write_table(latex_str: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_str)
    print(f"Table exported to {output_path.resolve()}")


def _table_environment(
    caption: str,
    label: str,
    size: str,
    tabular: str,
    col_spec: str,
    body: list[str],
    preamble: Iterable[str] = (),
    tabular_args: str = "",
) -> str:
    """A [t] table with a booktabs tabular (or tabularx, with its width in
    `tabular_args`) holding `body` between \\toprule and \\bottomrule."""
    lines = [
        r"\begin{table}[t]",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\centering",
        size,
        *preamble,
        rf"\begin{{{tabular}}}{tabular_args}{{{col_spec}}}",
        r"\toprule",
        *body,
        r"\bottomrule",
        rf"\end{{{tabular}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def _grouped_header(
    groups: list[str], sub_header: str, n_sub: int
) -> list[str]:
    """
    Header of a table with a leading Model column and one column group per
    entry of `groups` (already display labels), each of `n_sub` columns:
    the group row, the rules under it, and the sub-column row (the same
    `sub_header` cells repeated for every group).
    """
    cells, rules = [], []
    for i, group in enumerate(groups):
        first = 2 + i * n_sub
        last = first + n_sub - 1
        cells.append(rf"\multicolumn{{{n_sub}}}{{c}}{{{_tex_escape(group)}}}")
        rules.append(rf"\cmidrule(lr){{{first}-{last}}}")
    return [
        " & " + " & ".join(cells) + r" \\",
        "".join(rules),
        "Model & " + " & ".join([sub_header] * len(groups)) + r" \\",
    ]


def _format_mean_se(value: object, missing: str) -> str:
    if not isinstance(value, MeanSE) or math.isnan(value.mean):
        return missing
    return f"{value.mean:.2f} ({value.se:.2f})"


def _prompt_label(prompt_name: str) -> str:
    return prompt_name.capitalize()


# ---------------------------------------------------------------------------
# Non-standard answers
# ---------------------------------------------------------------------------


def export_nonstandard_summary(
    summary: pd.DataFrame, output_path: Path, label: str
) -> None:
    """`summary` is stats.nonstandard_summary's output."""
    caption = (
        "Number of non-standard LLM annotations per model and dataset, "
        "summed over all instruction prompts. Unparsed: no integer found "
        "in the reply; Out of range: an integer outside the dataset's "
        "annotation scale. Both are dropped from the analysis. Incidence: "
        "percentage of the model's annotations on that dataset."
    )
    table = summary.assign(
        model=summary["model"].map(labels.model_label),
        dataset=summary["dataset"].map(labels.dataset_label),
        status=summary["status"].map(NONSTANDARD_STATUS_LABELS),
        incidence=trim_numeric_col_latex(summary["incidence"], ".2f"),
    ).rename(
        columns={
            "model": "Model",
            "dataset": "Dataset",
            "status": "Status",
            "count": "Count",
            "incidence": "Incidence (%)",
        }
    )
    latex = table.to_latex(
        index=False,
        escape=True,
        caption=caption,
        label=label,
        position="htbp",
    )
    _write_table(small_table_latex(center_table_latex(latex)), output_path)


# ---------------------------------------------------------------------------
# Consistency (Krippendorff's alpha)
# ---------------------------------------------------------------------------

_CONSISTENCY_CAPTION = "Consistency (Krippendorff's $\\alpha$, ordinal)"


def _export_consistency_table(
    df: pd.DataFrame, output_path: Path, caption: str, label: str
) -> None:
    df = labels.relabel_frame(df)
    if "Krippendorff's alpha" in df.columns:
        df["Krippendorff's alpha"] = trim_numeric_col_latex(
            df["Krippendorff's alpha"], float_format=CONSISTENCY_ALPHA_FORMAT
        )
    latex_str = df.to_latex(
        index=False,
        caption=caption,
        label=label,
        position="t",
        escape=True,
    )
    _write_table(center_table_latex(latex_str), output_path)


def export_cross_model_consistency(
    df: pd.DataFrame,
    output_path: Path,
    label: str,
    prompt_name: str,
    excluded_models: Collection[str] = (),
) -> None:
    """`df` is stats.cross_model_consistency_table's output, computed
    without `excluded_models` (named in the caption)."""
    caption = (
        f"{_CONSISTENCY_CAPTION} between LLMs given the same "
        f"({prompt_name}) prompt, per dataset"
    )
    if excluded_models:
        excluded = ", ".join(
            labels.model_label(m) for m in sorted(excluded_models)
        )
        caption += f", excluding the following models: {excluded}"
    _export_consistency_table(df, output_path, caption + ".", label)


def export_variant_consistency(
    df: pd.DataFrame, output_path: Path, label: str
) -> None:
    """`df` is stats.per_model_variant_consistency_table's output."""
    caption = (
        f"{_CONSISTENCY_CAPTION} of each model with itself across the "
        "three paraphrased prompt variants."
    )
    _export_consistency_table(df, output_path, caption, label)


def export_repeat_consistency(
    df: pd.DataFrame, output_path: Path, label: str, prompt_name: str
) -> None:
    """`df` is stats.per_model_repeat_consistency_table's output."""
    caption = (
        f"{_CONSISTENCY_CAPTION} of each model with itself across repeated "
        f"runs of the same ({prompt_name}) prompt."
    )
    _export_consistency_table(df, output_path, caption, label)


# ---------------------------------------------------------------------------
# Cohen's d
# ---------------------------------------------------------------------------


def _cohens_d_cell(value: float) -> str:
    """Two decimals, in math mode when negative (for the minus sign)."""
    if pd.isna(value):
        return MISSING_NARROW
    text = f"{value:.2f}"
    return f"${text}$" if text.startswith("-") else text


def _cohens_d_n_note(flat: pd.DataFrame) -> str:
    """Caption sentence with the number of comments per dataset, given as
    a range if it differs between models."""
    per_dataset = flat.loc[flat["Dataset"] != POOLED_LABEL]
    counts = pd.to_numeric(per_dataset["n"]).groupby(
        per_dataset["Dataset"], sort=False
    )
    parts = []
    for dataset, n in counts:
        low, high = int(n.min()), int(n.max())
        n_str = str(low) if low == high else f"{low}--{high}"
        parts.append(f"{_tex_escape(dataset)}: {n_str}")
    if not parts:
        return ""
    return "Number of comments: " + "; ".join(parts) + "."


def export_cohens_d_table(
    compact: pd.DataFrame, output_path: Path, label: str, all_models: bool
) -> None:
    """
    Writes the Cohen's d table with models as rows and one column group
    per dataset (one column per prompt), so it fits the text width without
    \\resizebox. `compact` is stats.compute_cohens_d_compact_table's
    output; the comment counts of its `n` column go into the caption.
    `all_models` says whether this is the version with every model.
    """
    flat = labels.relabel_frame(compact.reset_index())
    prompt_cols = [
        c for c in flat.columns if c not in ("Model", "Dataset", "n")
    ]
    prompts = [_prompt_label(c) for c in prompt_cols]
    models = list(dict.fromkeys(flat["Model"]))
    datasets = list(dict.fromkeys(flat["Dataset"]))

    caption = (
        "Cohen's d between the default and each adversarial instruction "
        f"prompt, computed on the per-comment {labels.NDFU.tex}. Results "
        "are given per model, for each dataset separately and as the "
        f"unweighted mean over datasets ({POOLED_LABEL})."
    )
    caption = " ".join(
        part for part in (caption, _cohens_d_n_note(flat)) if part
    )

    values = flat.set_index(["Model", "Dataset"])[prompt_cols]
    rows = []
    for model in models:
        cells = []
        for dataset in datasets:
            if (model, dataset) in values.index:
                cells += [
                    _cohens_d_cell(v) for v in values.loc[(model, dataset)]
                ]
            else:
                cells += [MISSING_NARROW] * len(prompts)
        rows.append(f"{_tex_escape(model)} & " + " & ".join(cells) + r" \\")

    rotated = " & ".join(
        rf"\rotatebox{{90}}{{{_tex_escape(prompt)}}}" for prompt in prompts
    )
    group_gap = rf"@{{\hspace{{{COHENS_D_GROUP_GAP_PT}pt}}}}"
    col_spec = "l " + group_gap.join(["r" * len(prompts)] * len(datasets))
    _write_table(
        _table_environment(
            caption,
            label,
            size=_model_table_size(all_models),
            tabular="tabular",
            col_spec=col_spec,
            preamble=[
                rf"\setlength{{\tabcolsep}}{{{COHENS_D_TABCOLSEP_PT}pt}}"
            ],
            body=[
                *_grouped_header(datasets, rotated, len(prompts)),
                r"\midrule",
                *rows,
            ],
        ),
        output_path,
    )


# ---------------------------------------------------------------------------
# Mean nDFU
# ---------------------------------------------------------------------------


def _merge_span_cells(latex_str: str, n_total_cols: int) -> str:
    """
    Rewrites rows containing a _SPAN_MARK value cell so that the value
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


def export_mean_ndfu_by_prompt(
    table: pd.DataFrame, output_path: Path, label: str
) -> None:
    """
    Writes stats.compute_mean_ndfu_table's output: a Human block, then one
    block per model, with a pooled row and one row per dataset; `n` and
    one "mean (SE)" column per prompt. The Human value does not depend on
    the prompt and spans all prompt columns.
    """
    prompts = [c for c in table.columns if c != "n"]
    display = pd.DataFrame(index=table.index)
    display["n"] = table["n"].astype(str)
    is_human = table.index.get_level_values("Model") == HUMAN_LABEL
    for i, prompt in enumerate(prompts):
        cells = table[prompt].map(lambda v: _format_mean_se(v, MISSING))
        if i == 0:
            cells = cells.where(~is_human, _SPAN_MARK + cells)
        else:
            cells = cells.where(~is_human, "")
        display[_prompt_label(prompt)] = cells

    caption = (
        f"Mean (SE) per-comment {labels.NDFU.tex} of the human annotations "
        "and of the LLM annotations under each instruction prompt. Human "
        "annotations do not depend on the prompt. Rows give results per "
        f"model, pooled over all datasets ({POOLED_LABEL}) and for each "
        "dataset separately; $n$ is the number of comments."
    )
    display = labels.relabel_frame(display)
    latex_str = display.to_latex(
        caption=caption,
        label=label,
        position="t",
        escape=True,
        index=True,
        multirow=True,
        column_format="ll" + "r" * len(display.columns),
    )
    latex_str = _merge_span_cells(
        latex_str,
        n_total_cols=display.index.nlevels + len(display.columns),
    )
    _write_table(small_table_latex(center_table_latex(latex_str)), output_path)


def export_mean_ndfu_default_table(
    table: pd.DataFrame, output_path: Path, label: str
) -> None:
    """
    Writes stats.compute_default_mean_ndfu_table's output as a full-width
    tabularx: models as rows, one column group per dataset with a
    no-persona and a with-persona column each. The Human value does not
    depend on the prompt and spans both columns of its dataset.
    """
    datasets = list(dict.fromkeys(table.columns.get_level_values(0)))
    prompts = list(dict.fromkeys(table.columns.get_level_values(1)))
    sub_cols = [
        NO_PERSONA_COLUMN
        if p == NO_PERSONA_PROMPT_NAME
        else WITH_PERSONA_COLUMN
        for p in prompts
    ]

    def _cell(value: object) -> str:
        return _format_mean_se(value, MISSING_NARROW)

    rows = []
    for model, row in table.iterrows():
        if model == HUMAN_LABEL:
            cells = [
                rf"\multicolumn{{{len(prompts)}}}{{c}}"
                rf"{{{_cell(row[(d, prompts[0])])}}}"
                for d in datasets
            ]
        else:
            cells = [_cell(row[(d, p)]) for d in datasets for p in prompts]
        rows.append(
            f"{_tex_escape(labels.model_label(str(model)))} & "
            + " & ".join(cells)
            + r" \\"
        )

    caption = (
        f"Mean (SE) per-comment {labels.NDFU.tex} of the human annotations "
        "and of the LLM annotations under the default prompt, without a "
        f"persona ({NO_PERSONA_COLUMN}) and with a sociodemographic persona "
        f"({WITH_PERSONA_COLUMN}) with default instruction prompting (no "
        "adversarial prompts), per model and dataset."
    )
    n_value_cols = len(prompts) * len(datasets)
    _write_table(
        _table_environment(
            caption,
            label,
            size=MEAN_NDFU_FONT_SIZE,
            tabular="tabularx",
            tabular_args=f"{{{MEAN_NDFU_TABLE_WIDTH}}}",
            col_spec=rf"l*{{{n_value_cols}}}{{{MEAN_NDFU_VALUE_COL}}}",
            body=[
                *_grouped_header(
                    [labels.dataset_label(d) for d in datasets],
                    " & ".join(_tex_escape(col) for col in sub_cols),
                    len(sub_cols),
                ),
                r"\midrule",
                *rows,
            ],
        ),
        output_path,
    )


# ---------------------------------------------------------------------------
# apunim by prompt
# ---------------------------------------------------------------------------


def _apunim_prompt_cell(row: pd.Series) -> str:
    if pd.isna(row["apunim"]):
        return MISSING
    return f"{row['apunim']:.3f}{significance_superscript(row['pvalue'])}"


def _apunim_prompt_wide(
    long_df: pd.DataFrame, prompt_names: Iterable[str]
) -> pd.DataFrame:
    """One row per (SDB Feature, Value, Model) and one column per prompt,
    holding the formatted apunim value with significance stars."""
    long_df = long_df.assign(cell=long_df.apply(_apunim_prompt_cell, axis=1))
    wide = long_df.pivot_table(
        index=["SDB Feature", "Value", "Model"],
        columns="Prompt",
        values="cell",
        aggfunc="first",
    )
    wide = wide.reindex(columns=[p for p in prompt_names if p in wide.columns])
    wide.columns = [_prompt_label(c) for c in wide.columns]
    wide = wide.fillna(MISSING).reset_index()

    wide["Model"] = pd.Categorical(
        wide["Model"],
        categories=order_models(set(wide["Model"])),
        ordered=True,
    )
    return wide.sort_values(["SDB Feature", "Value", "Model"]).reset_index(
        drop=True
    )


def export_apunim_prompt_table(
    long_df: pd.DataFrame,
    output_path: Path,
    label: str,
    dataset_name: str,
    prompt_names: Iterable[str],
) -> None:
    """`long_df` holds the apunim results of one dataset (every model and
    prompt), as from stats.compute_llm_apunim_by_prompt."""
    if long_df.empty:
        print(
            f"No LLM apunim-by-prompt results for {dataset_name}; "
            f"skipping {output_path}."
        )
        return
    df = labels.relabel_frame(_apunim_prompt_wide(long_df, prompt_names))
    df = df.rename(columns={"SDB Feature": labels.SDD.tex})
    # The prompt cells already hold LaTeX ("value$^{***}$"), so the frame
    # is exported with escape=False and only underscores are escaped.
    df = df.replace("_", r"\_", regex=True).set_index(
        [labels.SDD.tex, "Value", "Model"]
    )
    latex_str = df.to_latex(
        caption=(
            "Aposteriori unimodality results for the LLM annotations of "
            f"the {labels.dataset_label(dataset_name)} dataset, across "
            f"instruction prompts. {SIGNIFICANCE_LEGEND}"
        ),
        label=label,
        escape=False,
        position="t",
        index=True,
        multirow=True,
        longtable=True,
    )
    _write_table(latex_str, output_path)


# ---------------------------------------------------------------------------
# Significant PC dimensions: human vs. LLM annotations
# ---------------------------------------------------------------------------


def _significance_cell(
    outcomes: tuple[SignificanceOutcome, ...], slot_width_em: float
) -> str:
    """One fixed-width slot per prompt holding its coloured symbol."""
    return "".join(
        rf"\makebox[{slot_width_em:g}em][c]{{{SIGNIFICANCE_CELLS[o]}}}"
        for o in outcomes
    )


def export_apunim_significance_table(
    table_df: pd.DataFrame,
    output_path: Path,
    label: str,
    all_models: bool,
    alpha: float = SIGNIFICANCE_ALPHA,
) -> None:
    """
    `table_df` is stats.compute_apunim_significance's output: one row per
    (dataset, PC dimension), one column of per-prompt outcomes per model.
    Each cell is drawn as one symbol per prompt. `all_models` says whether
    this is the version with every model (narrower slots, smaller font).
    """
    if table_df.empty:
        print(f"No apunim significance results; skipping {output_path}.")
        return
    prompts = table_df.attrs.get("prompts", [])
    cells = SIGNIFICANCE_CELLS
    caption = (
        rf"Statistically significant ($p<{alpha}$) {labels.SDD.tex_plural} of "
        "each LLM compared with the human annotations, per dataset. A "
        "dimension counts as significant if at least one of its groups is. "
        "Each cell shows one symbol per prompt, in the order "
        f"{', '.join(prompts)}: "
        rf"{cells[SignificanceOutcome.AGREE_SIGNIFICANT]}~both significant, "
        "with the same sign for every group that is significant in both; "
        rf"{cells[SignificanceOutcome.AGREE_NEITHER]}~neither significant; "
        rf"{cells[SignificanceOutcome.SIGN_DIFFERS]}~both significant, but "
        "the sign of at least one group's effect differs; "
        rf"{cells[SignificanceOutcome.DISAGREE]}~significant for only one "
        "of the two. A dimension without an LLM result (no polarized "
        "comments with enough annotations per group) counts as not "
        "significant."
    )

    slot_width_em = (
        SIGNIFICANCE_SLOT_WIDTH_ALL_MODELS_EM
        if all_models
        else SIGNIFICANCE_SLOT_WIDTH_EXCLUDING_EM
    )
    model_cols = [
        c for c in table_df.columns if c not in ("Dataset", "SDB Feature")
    ]
    df = table_df.copy()
    for col in model_cols:
        df[col] = df[col].map(lambda o: _significance_cell(o, slot_width_em))

    # Model columns and short dataset names (narrow table).
    df = labels.relabel_frame(df, short=True)
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
    latex_str = (
        rf"\setlength{{\tabcolsep}}{{{SIGNIFICANCE_TABCOLSEP_PT}pt}}"
        + "\n"
        + latex_str
        + "\n"
    )
    # Only the all-models version is set in a smaller font.
    if all_models:
        latex_str = small_table_latex(latex_str, size=TABLE_SIZE_ALL_MODELS)
    _write_table(latex_str, output_path)


# Number of significant PC dimensions per model and dataset (and prompt).


def _count_layout(
    counts: pd.DataFrame,
) -> tuple[list[str], list[str], dict[str, int]]:
    """(LLM models in order, datasets in order, dataset -> number of PC
    dimensions)."""
    llm_models = order_models(set(counts["Model"]) - {HUMAN_LABEL})
    datasets = [str(d) for d in dict.fromkeys(counts["Dataset"])]
    n_dims = {
        str(d): int(n)
        for d, n in counts.groupby("Dataset")["tested"].max().items()
    }
    return llm_models, datasets, n_dims


def _ratio_cell(
    counts: pd.DataFrame,
    model: str,
    prompt: str,
    dataset: str,
    n_dimensions: int,
) -> str:
    """Share of the tested PC dimensions of `dataset` that `model` finds
    significant under `prompt` (any prompt for the human results). Marked
    with PARTIAL_MARK if the model has results for fewer than
    `n_dimensions` dimensions."""
    if model == HUMAN_LABEL:
        prompt = HUMAN_PROMPT_NAME
    rows = counts.loc[
        (counts["Model"] == model)
        & (counts["Prompt"] == prompt)
        & (counts["Dataset"] == dataset)
    ]
    tested = rows["tested"].sum()
    if tested == 0:
        return MISSING
    ratio = rf"{int(rows['significant'].sum() / tested * 100)}\%"
    return ratio if tested == n_dimensions else ratio + PARTIAL_MARK


def _count_row_label(name: str) -> str:
    return _escape_underscores(labels.dataset_label(name, short=True))


def _model_header(
    leading: list[str], llm_models: list[str], rotate: bool = False
) -> str:
    """Header row: the leading columns, then the model names, Human first
    (rotated by 90 degrees if `rotate`)."""
    names = [
        _escape_underscores(labels.model_label(m))
        for m in [HUMAN_LABEL] + llm_models
    ]
    if rotate:
        names = [rf"\rotatebox{{90}}{{{name}}}" for name in names]
    return " & ".join(leading + names) + r" \\"


def _model_col_spec(leading: str, n_llm_models: int) -> str:
    """Leading columns, the Human column, a gap, then one per LLM."""
    return leading + "c" + COLUMN_GROUP_GAP + "c" * n_llm_models


def _count_caption_note(counts: pd.DataFrame) -> str:
    """Explains PARTIAL_MARK, if the table uses it."""
    n_dims = counts.groupby("Dataset")["tested"].transform("max")
    if not (counts["tested"] < n_dims).any():
        return ""
    return (
        f" {PARTIAL_MARK}: ratio over fewer dimensions, as the model has "
        "results for only some of them."
    )


def export_significance_counts_summary(
    counts: pd.DataFrame,
    output_path: Path,
    label: str,
    prompt_name: str,
    all_models: bool,
    alpha: float = SIGNIFICANCE_ALPHA,
) -> None:
    """
    Concise table of stats.count_significant_dimensions' output: one row
    per dataset, a column with the number of PC dimensions (n), then one
    column per model (Human first) with the share of dimensions found
    significant under `prompt_name`.
    """
    if counts.empty:
        print(f"No significance counts; skipping {output_path}.")
        return
    llm_models, datasets, n_dims = _count_layout(counts)

    body = [_model_header(["Dataset", "$n$"], llm_models), r"\midrule"]
    for name in datasets:
        cells = [
            _ratio_cell(counts, m, prompt_name, name, n_dims[name])
            for m in [HUMAN_LABEL] + llm_models
        ]
        row = [_count_row_label(name), str(n_dims[name]), *cells]
        body.append(" & ".join(row) + r" \\")

    caption = (
        rf"Share of {labels.SDD.tex_plural} that are statistically "
        rf"significant ($p<{alpha}$) in the human annotations and in each "
        f"LLM's annotations under the {prompt_name} prompt, per dataset; "
        "$n$ is the number of dimensions. A dimension counts as significant "
        "if at least one of its groups is." + _count_caption_note(counts)
    )
    _write_table(
        _table_environment(
            caption,
            label,
            size=_model_table_size(all_models),
            tabular="tabular",
            col_spec=_model_col_spec("lr", len(llm_models)),
            body=body,
        ),
        output_path,
    )


def export_significance_counts_by_prompt(
    counts: pd.DataFrame,
    output_path: Path,
    label: str,
    prompt_names: Iterable[str],
    all_models: bool,
    alpha: float = SIGNIFICANCE_ALPHA,
) -> None:
    """
    Full table of stats.count_significant_dimensions' output: for each
    dataset, one row per prompt, with the number of PC dimensions (n) and
    one column per model (Human first) holding the share of dimensions
    found significant. Human annotations do not depend on the prompt, so
    their value spans the dataset's rows.
    """
    if counts.empty:
        print(f"No significance counts; skipping {output_path}.")
        return
    llm_models, datasets, n_dims = _count_layout(counts)
    prompts = [p for p in prompt_names if p in set(counts["Prompt"])]

    def spanning(text: str) -> str:
        return rf"\multirow{{{len(prompts)}}}{{*}}{{{text}}}"

    body = [
        _model_header(["Dataset", "$n$", "Prompt"], llm_models, rotate=True)
    ]
    for name in datasets:
        n_dimensions = n_dims[name]
        human = _ratio_cell(counts, HUMAN_LABEL, "", name, n_dimensions)
        body.append(r"\midrule")
        for i, prompt in enumerate(prompts):
            first = i == 0
            lead = [
                spanning(_count_row_label(name)) if first else "",
                spanning(str(n_dimensions)) if first else "",
                _escape_underscores(_prompt_label(prompt)),
                spanning(human) if first else "",
            ]
            cells = [
                _ratio_cell(counts, m, prompt, name, n_dimensions)
                for m in llm_models
            ]
            body.append(" & ".join(lead + cells) + r" \\")

    caption = (
        rf"Share of {labels.SDD.tex_plural} that are statistically "
        rf"significant ($p<{alpha}$) in the human annotations and in each "
        "LLM's annotations under each instruction prompt, per dataset; $n$ "
        "is the number of dimensions. A dimension counts as significant if "
        "at least one of its groups is; human annotations do not depend on "
        "the prompt." + _count_caption_note(counts)
    )
    _write_table(
        _table_environment(
            caption,
            label,
            size=_model_table_size(all_models),
            tabular="tabular",
            col_spec=_model_col_spec("lrl", len(llm_models)),
            body=body,
        ),
        output_path,
    )


# ---------------------------------------------------------------------------
# Human-LLM correlation
# ---------------------------------------------------------------------------


def _correlation_column_label(prompt_name: str) -> str:
    if prompt_name == NO_PERSONA_PROMPT_NAME:
        return NO_PERSONA_COLUMN
    return _escape_underscores(_prompt_label(prompt_name))


def _correlation_cell(row: pd.Series, p_col: str) -> str:
    """rho as a math expression, with significance stars as superscript."""
    if pd.isna(row["spearman_rho"]):
        return MISSING
    stars = next(
        (s for threshold, s in SIGNIFICANCE_LEVELS if row[p_col] < threshold),
        "",
    )
    value = f"{row['spearman_rho']:.2f}"
    return f"${value}^{{{stars}}}$" if stars else f"${value}$"


def _correlation_wide(
    corr_df: pd.DataFrame, prompts: list[str], p_col: str
) -> pd.DataFrame:
    """One row per (Model, Dataset), in the order of `corr_df`, with its
    number of comments (`n`) and one formatted cell per prompt."""
    df = corr_df.assign(
        cell=corr_df.apply(_correlation_cell, axis=1, p_col=p_col)
    )
    rows = pd.MultiIndex.from_frame(df[["Model", "Dataset"]].drop_duplicates())
    wide = df.set_index(["Model", "Dataset", "Prompt"])["cell"].unstack(
        "Prompt"
    )
    wide = wide.reindex(index=rows, columns=prompts).fillna(MISSING)
    n = df.groupby(["Model", "Dataset"])["n"].first().reindex(rows)
    wide.insert(0, "n", n.astype(int).astype(str))
    return wide


def export_human_llm_correlation_table(
    corr_df: pd.DataFrame,
    output_path: Path,
    label: str,
    prompt_names: Iterable[str],
) -> None:
    """
    Writes stats.compute_human_llm_correlations' output: one block of rows
    per model (one row per dataset; the model named on the block's first
    row, blocks separated by a rule); a dataset and an n column, then one
    column per prompt of `prompt_names` that has results (the no-persona
    baseline, if first, separated by a small gap) under a common
    "Spearman's rho" header. Stars come from the corrected p-values if
    present, else from the raw ones.
    """
    if corr_df.empty:
        print(f"No human-LLM correlations; skipping {output_path}.")
        return
    corrected = "p_corrected" in corr_df.columns
    p_col = "p_corrected" if corrected else "p_raw"
    prompts = [p for p in prompt_names if p in set(corr_df["Prompt"])]
    n_prompts = len(prompts)
    table_df = _correlation_wide(corr_df, prompts, p_col)

    first_gap = (
        COLUMN_GROUP_GAP
        if prompts and prompts[0] == NO_PERSONA_PROMPT_NAME and n_prompts > 1
        else ""
    )
    col_spec = "llr" + (
        "r" + first_gap + "r" * (n_prompts - 1) if n_prompts else ""
    )

    body = [
        rf" & & & \multicolumn{{{n_prompts}}}{{c}}{{Spearman's $\rho$}} \\",
        rf"\cmidrule(lr){{4-{3 + n_prompts}}}",
        "Model & Dataset & $n$ & "
        + " & ".join(_correlation_column_label(p) for p in prompts)
        + r" \\",
        r"\midrule",
    ]
    previous_model = None
    for key, row in table_df.iterrows():
        model, dataset = cast(tuple[str, str], key)
        new_model = model != previous_model
        if new_model and previous_model is not None:
            body.append(r"\midrule")
        previous_model = model
        cells = [
            _escape_underscores(labels.model_label(model))
            if new_model
            else "",
            _escape_underscores(labels.dataset_label(dataset)),
            row["n"],
            *(row[p] for p in prompts),
        ]
        body.append(" & ".join(cells) + r" \\")

    caption = (
        "Spearman correlation, across the comments of each dataset, between "
        "the mean human annotation of a comment and each LLM's mean "
        f"annotation of it, without a persona ({NO_PERSONA_COLUMN}) and with "
        "a sociodemographic persona under each instruction prompt. $n$ is "
        "the number of comments, shared by all prompts of a row. "
        + ("Holm-corrected " if corrected else "")
        + "significance: "
        + SIGNIFICANCE_LEGEND
    )
    _write_table(
        _table_environment(
            caption,
            label,
            size=CORRELATION_TABLE_SIZE,
            tabular="tabular",
            col_spec=col_spec,
            body=body,
        ),
        output_path,
    )
