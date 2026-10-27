"""
Compares human annotations against LLM-generated annotations produced by
llm_annotate.py / annotate_experiments.sh.

The following are produced:

1. A grid of normalized histograms (one subplot per dataset) overlaying the
   human annotation distribution with each LLM's annotation distribution,
   for the "default" prompt.

1b. A grid of bar charts (one subplot per dataset) showing, per model, the
    mean difference (+/- standard error) between that model's annotations
    under the "stereotype"/"persona" prompts and under the "default"
    prompt, computed item-by-item on matched (comment, persona) pairs
    (see plot_prompt_mean_diff).

2. Three LaTeX tables built on Krippendorff's alpha (ordinal):
     - Cross-model consistency: for each dataset, how consistent the six
       LLMs are with each other when given the *same* (default) prompt.
     - Cross-variant consistency: for each (dataset, model), how consistent
       that model is with itself across the three paraphrased prompt
       variants used in the paraphrase ablation
       (instructions/ablation/<dataset>/variant{1,2,3}.txt).
     - Repeat consistency: for each (dataset, model), how consistent that
       model is with itself across repeated runs of the *same* prompt (the
       "-run0".."-runN" repeat ablation in output/ablations/repeat).

3. Aposteriori-unimodality (apunim) results for the LLM annotations, using
   the same underlying analysis as sap.py / dices.py / kumar.py
   (run_helper.run_all_results), but exported as a single LaTeX
   table per dataset -- one row per (SDB Feature, Value, Model), one
   column per prompt (default/stereotype/persona) -- rather than
   per-(dataset, model) "-results.csv"/"-inherent.csv" files. Restricted
   to the datasets the adversarial prompts were actually run on
   (shared.PROMPT_COMPARISON_DATASET_KEYS); all models are included.

4. Two compact LaTeX tables of the mean per-comment nDFU of the human
   annotations and of the LLM annotations (mean_ndfu_by_prompt.tex: every
   instruction prompt; mean_ndfu_default_vs_human.tex: default prompt
   only), per model and dataset (see stats.build_mean_ndfu_table). The
   same tables, and the Cohen's d table (cohens_d.tex), are also exported
   without the models in EXCLUDED_MODELS (files with an "_excluding"
   suffix).

5. A single LaTeX table, over all datasets, of which PC dimensions (SDB
   features such as age or gender; a dimension counts as significant if any
   of its groups is) each model finds statistically significant under each
   prompt, compared with the human findings (llm-apunim-significance.tex; also
   without the models in EXCLUDED_MODELS, with an "_excluding" suffix).

6. A CSV of the non-standard answers in the raw LLM annotation files
   (nonstandard_answers_by_model.csv), and an abbreviated LaTeX summary
   per model, dataset and status (nonstandard_answers_summary.tex):
   replies that cannot be parsed or are out of range for the dataset.

Rows are matched across files (models, or prompt variants) using the
comment id ("text_id") together with the sampled persona's characteristics,
since llm_annotate.py is seeded so that the same comments/personas are
drawn for every model and every prompt variant of a given dataset.
"""

import argparse
from pathlib import Path

import pandas as pd

from ..lib.preprocessing import (
    LazyDatasetLoader,
    DicesDataset,
    SapDataset,
    KumarDataset,
    PopquornDataset,
)

from ..lib import graphs
from ..lib.util import (
    skip_if_exists,
    center_table_latex,
    small_table_latex,
    trim_numeric_col_latex,
)
from . import polarization, stats, shared, plots

# ---------------------------------------------------------------------------
# Run configuration (only used by this script)
# ---------------------------------------------------------------------------

# Datasets used for the main annotation runs (default + adversarial prompts).
MAIN_DATASET_KEYS = shared.DATASET_KEYS

# Datasets used for the ablations (paraphrase variants, repeated runs).
ABLATION_DATASET_KEYS = shared.DATASET_KEYS

# Inherent-polarization subsampling ablation: every comment (human and LLM
# alike) is repeatedly subsampled down to this many annotators and the
# results averaged over this many repeats.
INHERENT_SUBSAMPLE_SIZE = 6
INHERENT_SUBSAMPLE_REPEATS = 10

# Grid of the subsampled inherent-polarization histogram figure: datasets
# are the columns, prompts the rows (all main prompts, as in the table).
INHERENT_HIST_DATASET_KEYS = ["sap", "kumar", "popquorn"]
INHERENT_HIST_PROMPT_NAMES = shared.MAIN_PROMPT_NAMES

# Models left out of the "excluding" variants of the tables/plots.
EXCLUDED_MODELS = ["llama8b", "qwen7b", "olmo7b"]

# Width (em) of one prompt slot in the significance table, i.e. the cell
# size. The all-models version has more columns, so its cells are narrower.
SIGNIFICANCE_SLOT_WIDTH_ALL_MODELS_EM = 0.9
SIGNIFICANCE_SLOT_WIDTH_EXCLUDING_EM = stats.DEFAULT_SLOT_WIDTH_EM

# Valid annotation values per dataset key.
VALID_ANNOTATION_VALUES = {
    "dices-350": {-1, 0, 1},
    "dices-990": {-1, 0, 1},
    "sap": {1, 2, 3, 4, 5},
    "kumar": {1, 2, 3, 4, 5},
    "popquorn": {1, 2, 3, 4, 5},
}


def main(
    dices_small_path: Path,
    dices_large_path: Path,
    sap_path: Path,
    kumar_path: Path,
    popquorn_path: Path,
    annotations_dir: Path,
    paraphrase_dir: Path,
    repeat_dir: Path,
    graph_output_dir: Path,
    latex_output_dir: Path,
    cache_dir: Path,
    human_results_dir: Path,
    exclude_models: list[str],
    prompt_name: str = "default",
):
    graphs.graph_setup()
    graph_output_dir.mkdir(parents=True, exist_ok=True)
    latex_output_dir.mkdir(parents=True, exist_ok=True)

    run_nonstandard_answers_step(annotations_dir, cache_dir, latex_output_dir)

    human_datasets = shared.HumanDatasets(
        {
            "dices-350": LazyDatasetLoader(
                lambda p=dices_small_path: DicesDataset(
                    dataset_path=p, variant="350"
                )
            ),
            "dices-990": LazyDatasetLoader(
                lambda p=dices_large_path: DicesDataset(
                    dataset_path=p, variant="990"
                )
            ),
            "sap": LazyDatasetLoader(
                lambda p=sap_path: SapDataset(dataset_path=p)
            ),
            "kumar": LazyDatasetLoader(
                lambda p=kumar_path: KumarDataset(
                    dataset_path=p, num_samples=1_000
                )
            ),
            "popquorn": LazyDatasetLoader(
                lambda p=popquorn_path: PopquornDataset(dataset_path=p)
            ),
        }
    )

    # apunim results for every comparison dataset, computed (or loaded from
    # the cache) once and shared by the apunim table steps below.
    apunim_results = load_apunim_results(
        human_datasets, annotations_dir, cache_dir
    )

    run_histogram_step(
        human_datasets, annotations_dir, graph_output_dir, prompt_name
    )
    run_prompt_diff_step(
        human_datasets, annotations_dir, graph_output_dir, exclude_models
    )

    run_cross_model_consistency_step(
        human_datasets, annotations_dir, latex_output_dir, prompt_name
    )
    _run_cross_model_consistency_excluding_step(
        human_datasets,
        annotations_dir,
        latex_output_dir,
        prompt_name,
        exclude_models,
    )

    run_variant_consistency_step(
        human_datasets, paraphrase_dir, latex_output_dir
    )
    run_repeat_consistency_step(
        human_datasets, repeat_dir, latex_output_dir, prompt_name
    )
    run_apunim_prompt_table_step(
        human_datasets, apunim_results, latex_output_dir
    )
    run_apunim_significance_step(
        human_datasets,
        apunim_results,
        latex_output_dir,
        human_results_dir,
        exclude_models,
    )
    run_mean_ndfu_step(
        human_datasets,
        annotations_dir,
        latex_output_dir,
        cache_dir,
        exclude_models,
    )

    run_inherent_polarization_steps(
        human_datasets,
        annotations_dir,
        latex_output_dir,
        exclude_models,
        human_results_dir=human_results_dir,
        cache_dir=cache_dir,
        graph_output_dir=graph_output_dir,
    )

    stats_path = (
        cache_dir / "polarization_by_instruction_anova_by_model_dataset.csv"
    )
    if skip_if_exists(stats_path):
        res_df = pd.read_csv(stats_path)
    else:
        res_df = stats.compute_ndfu_anova_by_prompt(
            human_datasets=human_datasets,
            annotations_dir=annotations_dir,
            correction_method="holm",
        )

    stats.export_ndfu_anova_by_prompt(result_df=res_df, output_path=stats_path)
    run_cohens_d_step(res_df, latex_output_dir, exclude_models)
    stats.run_exploratory_stats(res_df)


# ---------------------------------------------------------------------------
# Non-standard answers
# ---------------------------------------------------------------------------

# Display names of the statuses in the LaTeX summary table.
_STATUS_LABELS = {"unparsed": "Unparsed", "out_of_range": "Out of range"}


def _annotation_status(annotation: pd.Series, valid: set) -> pd.Series:
    """
    Per-reply status, judged with the parser the analysis uses
    (shared._clean_annotation): "unparsed" (no integer found, dropped
    downstream), "out_of_range" (integer not valid for the dataset; NOT
    dropped by the pipeline), or NA for a reply that parses to a valid
    value (including replies that only parse after cleaning).
    """
    cleaned = shared._clean_annotation(annotation)

    status = pd.Series(pd.NA, index=annotation.index, dtype="object")
    status[cleaned.isna()] = "unparsed"
    status[cleaned.notna() & ~cleaned.isin(valid)] = "out_of_range"
    return status


def _nonstandard_counts_for_file(
    path: Path, model: str, key: str, valid: set
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


def _nonstandard_counts_for_prompt(
    files: dict[str, Path],
    key: str,
    valid: set,
    totals: dict[tuple[str, str], int],
) -> list[pd.DataFrame]:
    """Counts for every model file of one (dataset, prompt). Adds each
    file's row count to `totals[(model, dataset)]` (in place)."""
    frames = []
    for model, path in sorted(files.items()):
        counts, n_rows = _nonstandard_counts_for_file(path, model, key, valid)
        totals[(model, key)] = totals.get((model, key), 0) + n_rows
        frames.append(counts)
    return frames


def _nonstandard_counts_for_dataset(
    annotations_dir: Path, key: str, totals: dict[tuple[str, str], int]
) -> list[pd.DataFrame]:
    """Counts for every prompt and model of one dataset."""
    valid = VALID_ANNOTATION_VALUES[key]
    frames = []
    for prompt_name in shared.MAIN_PROMPT_NAMES:
        files = shared.find_annotation_files(annotations_dir, key, prompt_name)
        frames.extend(
            _nonstandard_counts_for_prompt(files, key, valid, totals)
        )
    return frames


def _collect_nonstandard_counts(
    annotations_dir: Path,
) -> tuple[pd.DataFrame, dict[tuple[str, str], int]]:
    """
    (per-file counts with columns model/dataset/status/answer/count,
    total number of annotations per (model, dataset)), over all main
    datasets and prompts.
    """
    frames: list[pd.DataFrame] = []
    totals: dict[tuple[str, str], int] = {}
    for key in MAIN_DATASET_KEYS:
        frames.extend(
            _nonstandard_counts_for_dataset(annotations_dir, key, totals)
        )

    if not frames:
        empty = pd.DataFrame(
            columns=["model", "dataset", "status", "answer", "count"]
        )
        return empty, totals
    return pd.concat(frames, ignore_index=True), totals


def _totals_per_model(totals: dict[tuple[str, str], int]) -> dict[str, int]:
    """Sums (model, dataset) annotation totals over datasets."""
    per_model: dict[str, int] = {}
    for (model, _), n in totals.items():
        per_model[model] = per_model.get(model, 0) + n
    return per_model


def _nonstandard_by_model(
    detailed: pd.DataFrame, totals: dict[tuple[str, str], int]
) -> pd.DataFrame:
    """One row per (model, status, answer), with the model's total number
    of annotations (n_total_model) and the share each row represents."""
    result = detailed.groupby(["model", "status", "answer"], as_index=False)[
        "count"
    ].sum()
    result["n_total_model"] = result["model"].map(_totals_per_model(totals))
    result["rate"] = result["count"] / result["n_total_model"]
    return result.sort_values(
        ["model", "status", "count"], ascending=[True, True, False]
    ).reset_index(drop=True)


def get_nonstandard_answer_counts(annotations_dir: Path) -> pd.DataFrame:
    """
    Value counts of non-standard answers in the raw LLM annotation files,
    aggregated over all datasets and prompts, one row per
    (model, status, answer).

    Statuses (see _annotation_status): "unparsed", "out_of_range".
    Also returns each model's total number of annotations
    (n_total_model) and the share of those that each row represents (rate).
    """
    detailed, totals = _collect_nonstandard_counts(annotations_dir)
    return _nonstandard_by_model(detailed, totals)


def get_nonstandard_summary(
    detailed: pd.DataFrame, totals: dict[tuple[str, str], int]
) -> pd.DataFrame:
    """Abbreviated summary: one row per (model, dataset, status), summed
    over answers and prompts. `incidence` is the count as a percentage of
    all of the model's annotations on that dataset (over all prompts)."""
    summary = detailed.groupby(["model", "dataset", "status"], as_index=False)[
        "count"
    ].sum()
    summary["status"] = summary["status"].map(_STATUS_LABELS)

    keys = pd.Series(
        list(zip(summary["model"], summary["dataset"])), index=summary.index
    )
    summary["incidence"] = 100 * summary["count"] / keys.map(totals)
    return summary.sort_values(["model", "dataset", "status"]).reset_index(
        drop=True
    )


def export_nonstandard_summary_latex(
    summary: pd.DataFrame, output_path: Path
) -> None:
    caption = (
        "Number of non-standard LLM annotations per model and dataset, "
        "summed over all instruction prompts. Unparsed: no integer found "
        "in the reply (dropped from the analysis); Out of range: an "
        "integer outside the dataset's annotation scale. Incidence: "
        "percentage of the model's annotations on that dataset."
    )
    table = summary.assign(
        incidence=trim_numeric_col_latex(summary["incidence"], ".2f")
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
        label="tab:nonstandard-answers",
        position="htbp",
    )
    latex = small_table_latex(center_table_latex(latex))
    output_path.write_text(latex)


def run_nonstandard_answers_step(
    annotations_dir, csv_output_dir, latex_output_dir
):
    detailed, totals = _collect_nonstandard_counts(annotations_dir)
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    _nonstandard_by_model(detailed, totals).to_csv(
        csv_output_dir / "nonstandard_answers_by_model.csv", index=False
    )

    summary = get_nonstandard_summary(detailed, totals)
    if summary.empty:
        print("No non-standard answers found; skipping summary table.")
        return
    export_nonstandard_summary_latex(
        summary, latex_output_dir / "nonstandard_answers_summary.tex"
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _model_variants(exclude_models):
    """
    Variants of a per-model table: all models, and (if any models are
    excluded) all models but the excluded ones. Yields
    (file suffix, label suffix, caption note, models to drop).
    """
    yield "", "", "", set()
    if exclude_models:
        excluded_str = ", ".join(sorted(exclude_models))
        yield (
            "_excluding",
            "-excluding",
            f" Excluding the following models: {excluded_str}.",
            set(exclude_models),
        )


def _drop_models(df: pd.DataFrame, models: set[str]) -> pd.DataFrame:
    """Rows of a long/results frame whose `Model` is not in `models`."""
    return df.loc[~df["Model"].isin(models)] if models else df


def _comparison_keys(human_datasets) -> list[str]:
    """PROMPT_COMPARISON_DATASET_KEYS that are loaded in human_datasets."""
    return [
        k for k in shared.PROMPT_COMPARISON_DATASET_KEYS if k in human_datasets
    ]


# ---------------------------------------------------------------------------
# Cohen's d
# ---------------------------------------------------------------------------

# Dataset label of the unweighted mean over datasets in the compact table.
COHENS_D_ALL_LABEL = "All"
# Layout of the exported Cohen's d table: models as rows, one column group
# per dataset, one column per prompt. Sized to fit a 455pt text width
# without \resizebox.
COHENS_D_FONT_SIZE = r"\footnotesize"
COHENS_D_TABCOLSEP_PT = 1.5
COHENS_D_GROUP_GAP_PT = 4
# Shown for a (model, dataset) pair without results.
COHENS_D_MISSING = "--"

_TEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
}


def _tex_escape(text) -> str:
    return "".join(_TEX_SPECIALS.get(char, char) for char in str(text))


def _cohens_d_flat(compact: pd.DataFrame) -> pd.DataFrame:
    """The compact table with Model and Dataset as columns, whether they
    are its index or already columns."""
    if {"Model", "Dataset"} <= set(compact.columns):
        return compact.reset_index(drop=True)
    return compact.reset_index()


def _cohens_d_cell(value) -> str:
    """Two decimals, a math minus for negative values, and
    COHENS_D_MISSING for missing ones. Accepts numbers or strings."""
    if not isinstance(value, str) and pd.isna(value):
        return COHENS_D_MISSING
    text = value.strip() if isinstance(value, str) else f"{value:.2f}"
    if text in ("", "nan", "NaN"):
        return COHENS_D_MISSING
    return f"${text}$" if text.startswith("-") else text


def _cohens_d_n_note(flat: pd.DataFrame) -> str:
    """Caption sentence with the number of comments per dataset; models
    whose count differs from the usual one are listed in parentheses."""
    per_dataset = flat.loc[flat["Dataset"] != COHENS_D_ALL_LABEL]
    parts = []
    for dataset, group in per_dataset.groupby("Dataset", sort=False):
        counts = pd.to_numeric(group["n"])
        usual = int(counts.mode().iloc[0])
        part = f"{_tex_escape(dataset)}: {usual}"
        odd = group.loc[counts != usual]
        if not odd.empty:
            exceptions = ", ".join(
                f"{_tex_escape(model)}: {int(n)}"
                for model, n in zip(odd["Model"], odd["n"])
            )
            part += f" ({exceptions})"
        parts.append(part)
    if not parts:
        return ""
    return "Comments per model: " + "; ".join(parts) + "."


def _cohens_d_missing_note(
    flat: pd.DataFrame, models: list, datasets: list
) -> str:
    """Caption sentences for models without results on some datasets."""
    present = set(zip(flat["Model"], flat["Dataset"]))
    notes = []
    for model in models:
        missing = [
            str(dataset)
            for dataset in datasets
            if dataset != COHENS_D_ALL_LABEL
            and (model, dataset) not in present
        ]
        if missing:
            notes.append(
                f"{_tex_escape(model)} has no results for "
                f"{_tex_escape(', '.join(missing))} ({COHENS_D_MISSING}), "
                f"so its {COHENS_D_ALL_LABEL} is the mean over its "
                "remaining datasets."
            )
    return " ".join(notes)


def _cohens_d_header(datasets: list, prompts: list) -> list[str]:
    """Dataset group row, the rules under it, and the rotated prompt row."""
    groups, rules = [], []
    for i, dataset in enumerate(datasets):
        first = 2 + i * len(prompts)
        last = first + len(prompts) - 1
        groups.append(
            rf"\multicolumn{{{len(prompts)}}}{{c}}{{{_tex_escape(dataset)}}}"
        )
        rules.append(rf"\cmidrule(lr){{{first}-{last}}}")
    rotated = " & ".join(
        rf"\rotatebox{{90}}{{{_tex_escape(prompt)}}}" for prompt in prompts
    )
    return [
        " & " + " & ".join(groups) + r" \\",
        "".join(rules),
        "Model & " + " & ".join([rotated] * len(datasets)) + r" \\",
    ]


def _cohens_d_rows(
    flat: pd.DataFrame, models: list, datasets: list, prompts: list
) -> list[str]:
    """One row per model; a dataset the model has no results for is
    shown as COHENS_D_MISSING."""
    values = flat.set_index(["Model", "Dataset"])[prompts]
    rows = []
    for model in models:
        cells = []
        for dataset in datasets:
            if (model, dataset) in values.index:
                cells += [
                    _cohens_d_cell(v) for v in values.loc[(model, dataset)]
                ]
            else:
                cells += [COHENS_D_MISSING] * len(prompts)
        rows.append(f"{_tex_escape(model)} & " + " & ".join(cells) + r" \\")
    return rows


def export_cohens_d_table_latex(
    compact: pd.DataFrame,
    output_path: Path,
    caption: str,
    label: str,
    note: str = "",
) -> None:
    """
    Writes the Cohen's d table with models as rows and one column group
    per dataset (one column per prompt), so it fits the text width without
    \\resizebox. `compact` is stats.compute_cohens_d_compact_table's
    output: (Model, Dataset) rows, an `n` column and one column per
    prompt. The comment counts move from the `n` column to the caption.
    """
    flat = _cohens_d_flat(compact)
    prompts = [c for c in flat.columns if c not in ("Model", "Dataset", "n")]
    models = list(dict.fromkeys(flat["Model"]))
    datasets = list(dict.fromkeys(flat["Dataset"]))

    caption_parts = (
        caption.strip(),
        _cohens_d_n_note(flat),
        _cohens_d_missing_note(flat, models, datasets),
    )
    full_caption = " ".join(part for part in caption_parts if part) + note

    group_gap = rf"@{{\hspace{{{COHENS_D_GROUP_GAP_PT}pt}}}}"
    col_spec = "l " + group_gap.join(["r" * len(prompts)] * len(datasets))

    lines = [
        r"\begin{table}[t]",
        rf"\caption{{{full_caption}}}",
        rf"\label{{{label}}}",
        r"\centering",
        COHENS_D_FONT_SIZE,
        rf"\setlength{{\tabcolsep}}{{{COHENS_D_TABCOLSEP_PT}pt}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        *_cohens_d_header(datasets, prompts),
        r"\midrule",
        *_cohens_d_rows(flat, models, datasets, prompts),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def _export_cohens_d_variant(
    res_df,
    latex_output_dir,
    caption,
    file_suffix,
    label_suffix,
    note,
    models,
):
    variant_df = _drop_models(res_df, models)
    if variant_df.empty:
        print(f"No Cohen's d results left{note}; skipping table.")
        return
    export_cohens_d_table_latex(
        stats.compute_cohens_d_compact_table(variant_df),
        output_path=latex_output_dir / f"cohens_d{file_suffix}.tex",
        caption=caption,
        label=f"tab:cohens-d{label_suffix}",
        note=note,
    )


def run_cohens_d_step(res_df, latex_output_dir, exclude_models):
    # The effect sizes are computed independently per (model, dataset), so
    # dropping models from the results gives exactly the table one would
    # get by never running them.
    caption = r"""Cohen's d between the default and each adversarial
        instruction prompt, computed on the per-comment \ac{ndfu}. Results
        are given per model, for each dataset separately and as the
        unweighted mean over datasets (All)."""
    for file_suffix, label_suffix, note, models in _model_variants(
        exclude_models
    ):
        _export_cohens_d_variant(
            res_df,
            latex_output_dir,
            caption,
            file_suffix,
            label_suffix,
            note,
            models,
        )


# ---------------------------------------------------------------------------
# Histograms, prompt diffs, consistency tables
# ---------------------------------------------------------------------------


def run_histogram_step(
    human_datasets, annotations_dir, graph_output_dir, prompt_name
):
    output_path = graph_output_dir / "human_vs_llm_histograms.png"
    plots.plot_annotation_histograms(
        human_datasets=human_datasets,
        annotations_dir=annotations_dir,
        output_path=output_path,
        prompt_name=prompt_name,
        dataset_keys=MAIN_DATASET_KEYS,
    )


def run_prompt_diff_step(
    human_datasets, annotations_dir, graph_output_dir, exclude_models
):
    output_path = graph_output_dir / "llm_prompt_mean_diff.png"
    plots.plot_prompt_mean_diff(
        human_datasets=human_datasets,
        annotations_dir=annotations_dir,
        output_path=output_path,
        prompt_names=shared.MAIN_PROMPT_NAMES,
        exclude_models=exclude_models,
        dataset_keys=MAIN_DATASET_KEYS,
    )


def run_cross_model_consistency_step(
    human_datasets, annotations_dir, latex_output_dir, prompt_name
):
    output_path = latex_output_dir / "llm-consistency-cross-model.tex"
    stats.export_latex_table(
        stats.cross_model_consistency_table(
            human_datasets=human_datasets,
            annotations_dir=annotations_dir,
            prompt_name=prompt_name,
            dataset_keys=MAIN_DATASET_KEYS,
        ),
        output_path=output_path,
        caption=(
            "Consistency (Krippendorff's $\\alpha$, ordinal) between "
            f"LLMs given the same ({prompt_name}) prompt, per dataset."
        ),
        label="tab:llm-consistency-cross-model",
    )


def _run_cross_model_consistency_excluding_step(
    human_datasets,
    annotations_dir,
    latex_output_dir,
    prompt_name,
    exclude_models,
):
    if not exclude_models:
        return
    output_path = (
        latex_output_dir / "llm-consistency-cross-model-excluding.tex"
    )
    excluded_str = ", ".join(sorted(exclude_models))
    stats.export_latex_table(
        stats.cross_model_consistency_table(
            human_datasets=human_datasets,
            annotations_dir=annotations_dir,
            prompt_name=prompt_name,
            exclude_models=exclude_models,
            dataset_keys=MAIN_DATASET_KEYS,
        ),
        output_path=output_path,
        caption=(
            "Consistency (Krippendorff's $\\alpha$, ordinal) between LLMs "
            f"given the same ({prompt_name}) prompt, per dataset, excluding "
            f"the following models: {excluded_str}."
        ),
        label="tab:llm-consistency-cross-model-excluding",
    )


def run_variant_consistency_step(
    human_datasets, paraphrase_dir, latex_output_dir
):
    output_path = latex_output_dir / "llm-consistency-variants.tex"
    stats.export_latex_table(
        stats.per_model_variant_consistency_table(
            human_datasets=human_datasets,
            paraphrase_dir=paraphrase_dir,
            dataset_keys=ABLATION_DATASET_KEYS,
        ),
        output_path=output_path,
        caption=(
            "Consistency (Krippendorff's $\\alpha$, ordinal) of each model "
            "with itself across the three paraphrased prompt variants."
        ),
        label="tab:llm-consistency-variants",
    )


def run_repeat_consistency_step(
    human_datasets, repeat_dir, latex_output_dir, prompt_name
):
    output_path = latex_output_dir / "llm-consistency-repeats.tex"
    stats.export_latex_table(
        stats.per_model_repeat_consistency_table(
            human_datasets=human_datasets,
            repeat_dir=repeat_dir,
            prompt_name=prompt_name,
            dataset_keys=ABLATION_DATASET_KEYS,
        ),
        output_path=output_path,
        caption=(
            "Consistency (Krippendorff's $\\alpha$, ordinal) of each model "
            f"with itself across repeated runs of the same ({prompt_name}) "
            "prompt."
        ),
        label="tab:llm-consistency-repeats",
    )


# ---------------------------------------------------------------------------
# apunim cache
# ---------------------------------------------------------------------------


def _read_apunim_cache(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        return pd.DataFrame()
    long_df = pd.read_csv(cache_path)
    # Older caches were written with the DataFrame index as a column.
    return long_df.loc[:, ~long_df.columns.str.startswith("Unnamed")]


def _cached_apunim_pairs(long_df: pd.DataFrame) -> set:
    if long_df.empty:
        return set()
    return set(zip(long_df["Model"], long_df["Prompt"]))


def _write_apunim_cache(long_df: pd.DataFrame, cache_path: Path) -> None:
    # Write to a temp file and rename, so an interrupt during the write
    # cannot leave a truncated cache behind.
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".csv.tmp")
    long_df.to_csv(tmp_path, index=False)
    tmp_path.replace(cache_path)


def _update_apunim_cache(
    long_df: pd.DataFrame, new_df: pd.DataFrame, cache_path: Path
) -> pd.DataFrame:
    """Appends `new_df` to the cache (on disk and in the returned frame)."""
    if new_df.empty:
        return long_df
    if not long_df.empty:
        new_df = pd.concat([long_df, new_df], ignore_index=True)
    _write_apunim_cache(new_df, cache_path)
    return new_df


def _pending_apunim_pairs(
    annotations_dir: Path, key: str, cached_pairs: set
) -> list[tuple[str, str, set[str]]]:
    """(prompt, model, all models of that prompt) for every (model, prompt)
    pair that has an annotation file but is not cached yet."""
    pending = []
    for prompt_name in shared.MAIN_PROMPT_NAMES:
        files = shared.find_annotation_files(annotations_dir, key, prompt_name)
        pending.extend(
            (prompt_name, model, set(files))
            for model in sorted(files)
            if (model, prompt_name) not in cached_pairs
        )
    return pending


def _compute_apunim_pair(
    annotations_dir: Path,
    key: str,
    prompt_name: str,
    model: str,
    all_models: set[str],
) -> pd.DataFrame:
    print(f"Computing apunim: {key} / {prompt_name} / {model}")
    return stats.compute_llm_apunim_by_prompt(
        annotations_dir=annotations_dir,
        dataset_key=key,
        prompt_names=[prompt_name],
        exclude_models=all_models - {model},
    )


def _load_or_compute_apunim(
    annotations_dir: Path, cache_dir: Path, key: str
) -> pd.DataFrame:
    """
    apunim results for every model and prompt of one dataset, cached in
    cache_dir/apunim_<key>.csv.

    The cache is checked per (model, prompt) pair -- the unit apunim is run
    on -- so annotation runs that are still in progress are handled: pairs
    already in the cache are never recomputed, pairs whose annotation file
    appears later are added on the next run, and the cache is rewritten
    after every pair, so an interrupted run keeps what it finished.
    """
    cache_path = cache_dir / f"apunim_{key}.csv"
    long_df = _read_apunim_cache(cache_path)
    n_rows_before = len(long_df)

    pending = _pending_apunim_pairs(
        annotations_dir, key, _cached_apunim_pairs(long_df)
    )
    for prompt_name, model, all_models in pending:
        new_df = _compute_apunim_pair(
            annotations_dir, key, prompt_name, model, all_models
        )
        long_df = _update_apunim_cache(long_df, new_df, cache_path)

    if len(long_df) == n_rows_before:
        print(f"Skipping (nothing new to compute): {cache_path}")
    return long_df


def load_apunim_results(
    human_datasets, annotations_dir: Path, cache_dir: Path
) -> dict[str, pd.DataFrame]:
    """dataset key -> apunim results (every model and prompt), for every
    comparison dataset. Computed once and shared by the apunim-by-prompt
    tables and the significant-PC-dimension tables."""
    return {
        key: _load_or_compute_apunim(annotations_dir, cache_dir, key)
        for key in _comparison_keys(human_datasets)
    }


# ---------------------------------------------------------------------------
# apunim tables
# ---------------------------------------------------------------------------


def _run_apunim_prompt_table_for_dataset(
    human_datasets,
    long_df: pd.DataFrame,
    latex_output_dir: Path,
    key: str,
):
    wide_df = stats.build_llm_apunim_prompt_table(
        long_df, prompt_names=shared.MAIN_PROMPT_NAMES
    )

    output_path = latex_output_dir / f"llm-apunim-by-prompt-{key}.tex"
    stats.export_llm_apunim_prompt_table(
        wide_df,
        output_path=output_path,
        dataset_name=human_datasets[key].get_name(),
        label=f"tab:llm-apunim-by-prompt-{key}",
    )


def run_apunim_prompt_table_step(
    human_datasets,
    apunim_results: dict[str, pd.DataFrame],
    latex_output_dir: Path,
):
    # apunim_results only holds the datasets the adversarial prompts were
    # run on (shared.PROMPT_COMPARISON_DATASET_KEYS).
    for key, long_df in apunim_results.items():
        _run_apunim_prompt_table_for_dataset(
            human_datasets=human_datasets,
            long_df=long_df,
            latex_output_dir=latex_output_dir,
            key=key,
        )


def _collect_apunim_significance_inputs(
    apunim_results: dict[str, pd.DataFrame], human_results_dir: Path
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """dataset key -> (human apunim results, LLM apunim results), for every
    comparison dataset that has human results."""
    inputs = {}
    for key, llm_df in apunim_results.items():
        # ignore kumar since no human results can be significant
        if key == "kumar":
            continue
        human_df = stats.load_human_apunim_results(human_results_dir, key)
        if human_df is None:
            continue
        inputs[key] = (human_df, llm_df)
    return inputs


def _significance_inputs_by_name(
    human_datasets, inputs, models
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """`inputs` keyed by dataset display name and without the given models;
    datasets left without LLM results are dropped."""
    by_name = {}
    for key, (human_df, llm_df) in inputs.items():
        variant_df = _drop_models(llm_df, models)
        if not variant_df.empty:
            by_name[human_datasets[key].get_name()] = (human_df, variant_df)
    return by_name


def run_apunim_significance_step(
    human_datasets,
    apunim_results: dict[str, pd.DataFrame],
    latex_output_dir,
    human_results_dir,
    exclude_models,
):
    """
    One table over all datasets: which PC dimensions (SDB features, e.g.
    age) each model finds statistically significant under each prompt,
    compared with the human results.
    """
    inputs = _collect_apunim_significance_inputs(
        apunim_results, human_results_dir
    )
    for file_suffix, label_suffix, note, models in _model_variants(
        exclude_models
    ):
        by_name = _significance_inputs_by_name(human_datasets, inputs, models)
        if not by_name:
            continue
        slot_width_em = (
            SIGNIFICANCE_SLOT_WIDTH_EXCLUDING_EM
            if models
            else SIGNIFICANCE_SLOT_WIDTH_ALL_MODELS_EM
        )
        stats.export_apunim_significance_table(
            stats.build_apunim_significance_table(
                by_name,
                prompt_names=shared.MAIN_PROMPT_NAMES,
                slot_width_em=slot_width_em,
            ),
            output_path=latex_output_dir
            / f"llm-apunim-significance{file_suffix}.tex",
            label=f"tab:llm-apunim-significance{label_suffix}",
            note=note,
        )


# ---------------------------------------------------------------------------
# Mean nDFU
# ---------------------------------------------------------------------------


def run_mean_ndfu_step(
    human_datasets,
    annotations_dir,
    latex_output_dir,
    cache_dir,
    exclude_models,
):
    # Computed for every model; the "excluding" variants are filtered from
    # these records. (New cache name: the old mean_ndfu_records.csv was
    # written without olmo7b/llama8b and would be silently reused.)
    cache_path = cache_dir / "mean_ndfu_records_all_models.csv"
    if skip_if_exists(cache_path):
        ndfu_df = pd.read_csv(cache_path)
    else:
        ndfu_df = stats.compute_mean_ndfu_records(
            human_datasets=human_datasets,
            annotations_dir=annotations_dir,
            dataset_keys=shared.PROMPT_COMPARISON_DATASET_KEYS,
            prompt_names=shared.MAIN_PROMPT_NAMES,
            exclude_models=None,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ndfu_df.to_csv(cache_path, index=False)

    if ndfu_df.empty:
        print("No matched human/LLM annotations; skipping mean nDFU tables.")
        return

    by_prompt_caption = r"""Mean (SE) per-comment \ac{ndfu} of the human annotations
        and of the LLM annotations under each instruction prompt. Human
        annotations do not depend on the prompt. Rows give results per
        model, pooled over all datasets (All) and for each dataset
        separately; $n$ is the number of comments."""
    default_caption = r"""Mean (SE) per-comment \ac{ndfu} of the human annotations
        and of the LLM annotations under the default prompt, per dataset
        and pooled over all datasets (All). $n$ is the number of
        comments."""

    for file_suffix, label_suffix, note, models in _model_variants(
        exclude_models
    ):
        variant_df = _drop_models(ndfu_df, models)

        stats.export_compact_prompt_table_latex(
            stats.build_mean_ndfu_table(
                variant_df, prompt_names=shared.MAIN_PROMPT_NAMES
            ),
            output_path=latex_output_dir
            / f"mean_ndfu_by_prompt{file_suffix}.tex",
            caption=by_prompt_caption + note,
            label=f"tab:mean-ndfu-by-prompt{label_suffix}",
        )

        stats.export_default_mean_ndfu_latex(
            stats.build_default_mean_ndfu_table(
                variant_df, prompt_name="default"
            ),
            output_path=latex_output_dir
            / f"mean_ndfu_default_vs_human{file_suffix}.tex",
            caption=default_caption + note,
            label=f"tab:mean-ndfu-default-vs-human{label_suffix}",
        )


# ---------------------------------------------------------------------------
# Inherent polarization
# ---------------------------------------------------------------------------


def run_inherent_polarization_step(
    human_datasets,
    annotations_dir,
    latex_output_dir,
    exclude_models,
    human_results_dir: Path,
    cache_dir: Path,
    graph_output_dir: Path,
    subsample: polarization.SubsampleSpec | None = None,
):
    # human_results_dir holds the human scripts' own "<dataset>-inherent.csv"
    # outputs (e.g. output/human/main), which are reused as-is for the main
    # run. The per-(dataset, prompt, model) values are cached separately
    # under cache_dir/inherent (the subsampling ablation uses its own
    # folder, since its values differ). Delete the folder to force a
    # recompute.
    suffix = "" if subsample is None else f"-{subsample.tag}"
    caption = polarization.DEFAULT_INHERENT_CAPTION
    if subsample is not None:
        caption = (
            caption.removesuffix(".")
            + f", after repeatedly subsampling every comment down to at "
            f"most {subsample.size} annotators ({subsample.n_repeats} "
            "repeats)."
        )

    inherent_df = polarization.compute_inherent_polarization_comparison(
        human_datasets=human_datasets,
        annotations_dir=annotations_dir,
        apunim_output_dir=cache_dir / f"inherent{suffix}",
        human_results_dir=human_results_dir,
        dataset_keys=MAIN_DATASET_KEYS,
        prompt_names=shared.MAIN_PROMPT_NAMES,
        exclude_models=set(exclude_models),
        use_monte_carlo=True,
        subsample=subsample,
    )

    # Normalized histogram grid of the same per-comment values (datasets as
    # columns, prompts as rows), exported next to the table. Only produced
    # for the annotator-subsampled variant.
    if subsample is not None:
        plots.plot_inherent_polarization_histogram_grid(
            long_df=inherent_df,
            output_path=(
                graph_output_dir
                / f"inherent-polarization{suffix}-histogram.png"
            ),
            dataset_keys=INHERENT_HIST_DATASET_KEYS,
            prompt_names=INHERENT_HIST_PROMPT_NAMES,
            common_norm=False,
        )


def run_inherent_polarization_steps(
    human_datasets,
    annotations_dir,
    latex_output_dir,
    exclude_models,
    human_results_dir: Path,
    cache_dir: Path,
    graph_output_dir: Path,
):
    """Annotator-subsampling ablation (same pipeline, `subsample` set)."""
    run_inherent_polarization_step(
        human_datasets,
        annotations_dir,
        latex_output_dir,
        exclude_models,
        human_results_dir=human_results_dir,
        cache_dir=cache_dir,
        graph_output_dir=graph_output_dir,
        subsample=polarization.SubsampleSpec(
            size=INHERENT_SUBSAMPLE_SIZE,
            n_repeats=INHERENT_SUBSAMPLE_REPEATS,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compare human vs. LLM annotations: normalized histograms, "
            "cross-model / cross-variant / repeat consistency tables, and "
            "apunim results for the LLM annotations."
        )
    )
    parser.add_argument(
        "--dices-small-path",
        required=True,
        help="Path to the DICES-350 CSV file.",
    )
    parser.add_argument(
        "--dices-large-path",
        required=True,
        help="Path to the DICES-990 CSV file.",
    )
    parser.add_argument(
        "--sap-path", required=True, help="Path to the Sap et al. CSV file."
    )
    parser.add_argument(
        "--kumar-path",
        required=True,
        help="Path to the Kumar et al. JSON file.",
    )
    parser.add_argument(
        "--popquorn-path",
        required=True,
        help="Path to the POPQUORN offensiveness CSV.",
    )
    parser.add_argument(
        "--annotations-dir",
        required=True,
        help=(
            "Directory containing the main (non-ablation) llm_annotate.py "
            "outputs, e.g. output/annotations."
        ),
    )
    parser.add_argument(
        "--paraphrase-dir",
        required=True,
        help=(
            "Directory containing the paraphrase-ablation llm_annotate.py "
            "outputs (variant1/variant2/variant3), e.g. "
            "output/ablations/paraphrase."
        ),
    )
    parser.add_argument(
        "--repeat-dir",
        required=True,
        help=(
            "Directory containing the repeat-ablation llm_annotate.py "
            "outputs (same prompt, run N times: '-run0', '-run1', ...), "
            "e.g. output/ablations/repeat."
        ),
    )
    parser.add_argument(
        "--graph-output-dir",
        required=True,
        help="Directory for the histogram and apunim polarization plots.",
    )
    parser.add_argument(
        "--latex-output-dir",
        required=True,
        help="Directory for the consistency LaTeX tables.",
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="Directory for cached apunim computations.",
    )
    parser.add_argument(
        "--human-results-dir",
        required=True,
        help=(
            "Directory with the human <dataset>-inherent.csv files "
            "written by the human analysis scripts, e.g. output/human/main."
        ),
    )
    args = parser.parse_args()

    main(
        dices_small_path=Path(args.dices_small_path),
        dices_large_path=Path(args.dices_large_path),
        sap_path=Path(args.sap_path),
        kumar_path=Path(args.kumar_path),
        popquorn_path=Path(args.popquorn_path),
        annotations_dir=Path(args.annotations_dir),
        paraphrase_dir=Path(args.paraphrase_dir),
        repeat_dir=Path(args.repeat_dir),
        graph_output_dir=Path(args.graph_output_dir),
        latex_output_dir=Path(args.latex_output_dir),
        prompt_name="default",
        cache_dir=Path(args.cache_dir),
        human_results_dir=Path(args.human_results_dir),
        exclude_models=EXCLUDED_MODELS,
    )
