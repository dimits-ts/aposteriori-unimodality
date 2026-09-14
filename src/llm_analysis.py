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
   (tasks.run_helper.run_all_results), but exported as a single LaTeX
   table per dataset -- one row per (SDB Feature, Value, Model), one
   column per prompt (default/stereotype/persona) -- rather than
   per-(dataset, model) "-results.csv"/"-inherent.csv" files. Restricted
   to the datasets the stereotype/persona prompts were actually run on
   (kumar, sap) and to the models run on all three prompts (see
   PROMPT_COMPARISON_DATASET_KEYS / APUNIM_TABLE_EXCLUDE_MODELS).

4. A single composite figure (llm_apunim_grid.png) with one subplot per
   (dataset, model) -- an nDFU-by-SDB-group boxplot for the "default"
   prompt -- assembled into one grid instead of many separate images, and
   sized/fonted so it stays readable once placed in a paper (see
   plot_apunim_grid's docstring for how that sizing works). The same grid
   is also produced separately for each adversarial prompt (currently
   "stereotype" and "persona", see ADVERSARIAL_PROMPT_NAMES), as
   llm_apunim_grid_<prompt>.png, with a title that just names the prompt
   instead of the main figure's title.

Rows are matched across files (models, or prompt variants) using the
comment id ("text_id") together with the sampled persona's characteristics,
since llm_annotate.py is seeded so that the same comments/personas are
drawn for every model and every prompt variant of a given dataset.
"""

import argparse
import re
from pathlib import Path

import krippendorff
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tasks.graphs
import tasks.preprocessing
import tasks.run_helper
from dices import DicesDataset
from kumar import KumarDataset
from sap import SapDataset


DATASET_KEYS = ["dices-350", "dices-990", "sap", "kumar"]

DATASET_LOADERS = {
    "dices-350": lambda p: DicesDataset(dataset_path=p, variant="350"),
    "dices-990": lambda p: DicesDataset(dataset_path=p, variant="990"),
    "kumar": lambda p: KumarDataset(dataset_path=p, num_samples=3_000),
    "sap": lambda p: SapDataset(dataset_path=p),
}

# Columns in an llm_annotate.py output CSV (plus the "annotation_clean"
# column we add in load_llm_df) that are *not* persona/SDB attributes.
NON_PERSONA_COLS = {
    "model",
    "instruction_prompt",
    "text_id",
    "text",
    "annotation",
    "annotation_clean",
}

VARIANT_NAMES = ["variant1", "variant2", "variant3"]

# N_PERSONAS_PER_COMMENT in llm_annotate.py: number of distinct annotator
# personas sampled per comment, i.e. the max number of "annotators" any
# single comment has in the LLM-annotation CSVs.
MAX_ANNOTATORS_PER_ITEM = 6

# Preferred left-to-right column order for the composite apunim grid (models
# not in this list are appended alphabetically after it).
MODEL_DISPLAY_ORDER = [
    "llama70b",
    "llama8b",
    "olmo32b",
    "olmo7b",
    "qwen32b",
    "qwen7b",
]

# The three main instruction prompts compared throughout this module (mean-
# diff plots, apunim prompt table). "default" is treated as the baseline
# that "stereotype"/"persona" are compared against.
MAIN_PROMPT_NAMES = ["default", "stereotype", "persona"]

# Datasets for which all three MAIN_PROMPT_NAMES were actually run (the
# DICES datasets only have the "default" prompt) -- used for both the
# prompt mean-diff plot and the apunim-by-prompt LaTeX table.
PROMPT_COMPARISON_DATASET_KEYS = ["kumar", "sap"]

# The "adversarial" instruction prompts (instructions/adversarial/<dataset>/,
# run by annotate_adversarial.sh) -- every MAIN_PROMPT_NAMES entry besides
# the "default" baseline. Each gets its own composite apunim grid (see
# plot_apunim_grid / main()).
ADVERSARIAL_PROMPT_NAMES = [p for p in MAIN_PROMPT_NAMES if p != "default"]

# Models excluded from the apunim-by-prompt LaTeX table (they were never
# run on the stereotype/persona prompts to begin with; listed explicitly
# so the table is correct even if that changes).
# Also used for LLM polarization grid.
APUNIM_TABLE_EXCLUDE_MODELS = {"olmo7b", "llama8b"}


class LLMAnnotationDataset(tasks.preprocessing.Dataset):
    """
    Adapts a single (dataset, prompt, model) llm_annotate.py output CSV --
    one row per (comment, persona) -- into the per-comment,
    list-of-annotators shape that tasks.run_helper / tasks.graphs expect,
    treating the sampled persona attributes as the SDB columns.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        dataset_key: str,
        model_pseudo: str,
        prompt_name: str,
    ):
        self._name = f"{dataset_key}-{prompt_name}-{model_pseudo}"
        persona_cols = _persona_columns(df)

        df = df.dropna(subset=["annotation_clean"]).copy()
        agg = {col: list for col in persona_cols}
        agg["annotation_clean"] = list
        self.df = df.groupby("text_id").agg(agg).reset_index()
        self.sdb_columns = persona_cols

    def get_name(self) -> str:
        return self._name

    def get_dataset(self) -> pd.DataFrame:
        return self.df

    def get_sdb_columns(self) -> list[str]:
        return self.sdb_columns

    def get_comment_key_column(self) -> str:
        return "text_id"

    def get_annotation_column(self) -> str:
        return "annotation_clean"

    def get_text_column(self) -> str:
        return "text_id"


def _compute_ndfu_records(
    ds: tasks.preprocessing.Dataset, sdb_columns: list[str] | None = None
) -> pd.DataFrame:
    """
    Per-comment nDFU (apunim.dfu over that comment's annotator list),
    broadcast onto every SDB group any of its annotators belonged to --
    the same computation tasks.graphs.polarization_plot does internally,
    factored out here so it can be drawn onto an arbitrary subplot axis
    instead of always producing its own standalone figure.

    `sdb_columns`, if given, overrides `ds.get_sdb_columns()` (e.g. to cap
    how many SDB dimensions are included) without needing to mutate `ds`.
    """
    import apunim  # local import: heavy-ish, only needed here and in graphs.py

    df = ds.get_dataset()
    annotation_col = ds.get_annotation_column()
    if sdb_columns is None:
        sdb_columns = ds.get_sdb_columns()

    all_annotations = []
    for annotations_list in df[annotation_col].to_list():
        if isinstance(annotations_list, (list, np.ndarray)):
            all_annotations.extend(annotations_list)

    if not all_annotations:
        return pd.DataFrame(columns=["PC Dimension", "nDFU"])

    bins = len(np.unique(all_annotations))

    records = []
    for _, row in df.iterrows():
        annotations = row[annotation_col]
        if (
            not isinstance(annotations, (list, np.ndarray))
            or len(annotations) == 0
        ):
            continue
        try:
            ndfu_value = apunim.dfu(annotations, bins=bins, normalized=True)
        except Exception as e:
            print(f"Error calculating NDFU for an item: {e}")
            continue

        for sdb_col in sdb_columns:
            for value in row[sdb_col]:
                records.append(
                    {"PC Dimension": f"{sdb_col}: {value}", "nDFU": ndfu_value}
                )

    return pd.DataFrame(records)


def _order_models(models: set[str] | list[str]) -> list[str]:
    known = [m for m in MODEL_DISPLAY_ORDER if m in models]
    unknown = sorted(m for m in models if m not in MODEL_DISPLAY_ORDER)
    return known + unknown


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _persona_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_PERSONA_COLS]


def _clean_annotation(series: pd.Series) -> pd.Series:
    """
    LLMs were asked to "reply with a single number only", but generations
    can still contain stray characters (whitespace, punctuation, a partial
    second token, ...). Extract the first signed integer found in each
    reply; anything that can't be parsed becomes NaN and is dropped
    downstream.
    """
    extracted = series.astype(str).str.extract(r"(-?\d+)")[0]
    return pd.to_numeric(extracted, errors="coerce")


def find_annotation_files(
    directory: Path, dataset_key: str, prompt_name: str
) -> dict[str, Path]:
    """
    Returns {model_pseudo: path} for every file in `directory` matching
    f"{dataset_key}-{prompt_name}-<pseudo>.csv" (e.g. as written by
    llm_annotate.py / annotate_experiments.sh). Run-suffixed files (e.g.
    the "-run0" repeat ablation) are intentionally excluded.
    """
    pattern = re.compile(
        rf"^{re.escape(dataset_key)}-{re.escape(prompt_name)}-([^-.]+)\.csv$"
    )
    out = {}
    if not directory.exists():
        return out
    for path in sorted(directory.glob(f"{dataset_key}-{prompt_name}-*.csv")):
        m = pattern.match(path.name)
        if m:
            out[m.group(1)] = path
    return out


def find_repeat_files(
    directory: Path, dataset_key: str, prompt_name: str
) -> dict[str, dict[str, Path]]:
    """
    Returns {model_pseudo: {run_label: path}} for every file in `directory`
    matching f"{dataset_key}-{prompt_name}-<pseudo>-run<N>.csv" (e.g. as
    written by the repeat ablation in annotate_experiments.sh: the same
    prompt run N times over the same 10% sub-sample).
    """
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


def load_llm_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["annotation_clean"] = _clean_annotation(df["annotation"])
    return df


def _skip_if_exists(path: Path) -> bool:
    """
    Returns True (and prints a message) if `path` already exists, so the
    caller can skip recomputing it. Mirrors the same helper in dices.py.
    """
    if path.exists():
        print(f"Skipping (already exists): {path}")
        return True
    return False


# ---------------------------------------------------------------------------
# 1. Histograms: human vs. LLM annotation frequencies, one subplot/dataset
# ---------------------------------------------------------------------------


def collect_human_annotations(ds: tasks.preprocessing.Dataset) -> np.ndarray:
    col = ds.get_annotation_column()
    values = []
    for entry in ds.get_dataset()[col]:
        if isinstance(entry, (list, np.ndarray)):
            values.extend(entry)
        else:
            values.append(entry)
    return (
        pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    )


def collect_llm_annotations(
    annotations_dir: Path, dataset_key: str, prompt_name: str
) -> dict[str, np.ndarray]:
    out = {}
    for pseudo, path in find_annotation_files(
        annotations_dir, dataset_key, prompt_name
    ).items():
        df = load_llm_df(path)
        out[pseudo] = df["annotation_clean"].dropna().to_numpy()
    return out


def plot_annotation_histograms(
    human_datasets: tasks.preprocessing.LazyDatasetLoader,
    annotations_dir: Path,
    output_path: Path,
    prompt_name: str = "default",
    ncols: int = 2,
) -> None:
    dataset_keys = [k for k in DATASET_KEYS if k in human_datasets] or list(
        human_datasets.keys()
    )
    n = len(dataset_keys)
    nrows = -(-n // ncols)  # ceil division

    fig, axes = plt.subplots(nrows, ncols, squeeze=False)

    for i, key in enumerate(dataset_keys):
        ax = axes[i // ncols][i % ncols]
        ds = human_datasets[key]

        human_vals = collect_human_annotations(ds)
        llm_vals = collect_llm_annotations(annotations_dir, key, prompt_name)

        records = [{"value": v, "source": "Human"} for v in human_vals]
        for pseudo, vals in sorted(llm_vals.items()):
            records.extend({"value": v, "source": pseudo} for v in vals)

        if not records:
            ax.set_visible(False)
            print(f"No annotations found for {key}; skipping subplot.")
            continue

        plot_df = pd.DataFrame(records)
        sources = ["Human"] + sorted(llm_vals.keys())
        sources = [s for s in sources if s in set(plot_df["source"])]
        palette = dict(zip(sources, tasks.graphs.COLORBLIND_PALETTE))

        # Step-line histograms (rather than dodged bars) so up to 7
        # overlapping distributions (human + 6 models) stay legible.
        sns.histplot(
            data=plot_df,
            x="value",
            hue="source",
            hue_order=sources,
            discrete=True,
            stat="probability",
            common_norm=False,
            palette=palette,
            ax=ax,
        )

        ax.set_title(ds.get_name())
        ax.set_xlabel(f"{ds.get_annotation_column()} value")
        ax.set_ylabel("Proportion")

        legend = ax.get_legend()
        if legend is not None:
            legend.set_title(None)
            for text in legend.get_texts():
                text.set_fontsize(10)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(
        f"Human vs. LLM annotation distributions ({prompt_name} prompt)",
        y=1.02,
    )
    fig.tight_layout()
    tasks.graphs.save_plot(output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1b. Mean-difference-between-prompts bar charts (per model, per dataset)
# ---------------------------------------------------------------------------


def _paired_prompt_annotations(
    annotations_dir: Path,
    dataset_key: str,
    model: str,
    prompt_names: list[str],
) -> pd.DataFrame | None:
    """
    For a single (dataset, model), loads the annotation CSV for each of
    `prompt_names` and aligns them on (text_id + persona attributes) -- the
    key llm_annotate.py's seeding guarantees is shared across every prompt
    variant of a given dataset (see module docstring). Returns None if the
    model is missing any of the requested prompts, or if no items survive
    the alignment; otherwise returns one row per matched (comment, persona)
    item, with one column per prompt holding that prompt's cleaned
    annotation value.
    """
    dfs = {}
    for prompt_name in prompt_names:
        path = find_annotation_files(
            annotations_dir, dataset_key, prompt_name
        ).get(model)
        if path is None:
            return None
        dfs[prompt_name] = load_llm_df(path)

    key_cols = ["text_id"] + _persona_columns(next(iter(dfs.values())))
    series_list = []
    for prompt_name, df in dfs.items():
        d = df.dropna(subset=["annotation_clean"]).copy()
        d["_key"] = list(zip(*[d[c] for c in key_cols]))
        d = d.drop_duplicates(subset="_key")
        s = d.set_index("_key")["annotation_clean"]
        s.name = prompt_name
        series_list.append(s)

    wide = pd.concat(series_list, axis=1).dropna()
    return wide if not wide.empty else None


def plot_prompt_mean_diff(
    human_datasets: tasks.preprocessing.LazyDatasetLoader,
    annotations_dir: Path,
    output_path: Path,
    exclude_models: list[str],
    prompt_names: list[str] = MAIN_PROMPT_NAMES,
    baseline_prompt: str = "default",
) -> None:
    """
    One subplot per dataset (skipping any dataset for which fewer than two
    of `prompt_names` were run): for each model, the mean difference --
    with standard-error bars -- between that model's annotations under
    each non-baseline prompt (e.g. "stereotype", "persona") and its
    annotations under `baseline_prompt` ("default"), computed item-by-item
    on the *same* (comment, persona) pairs via `_paired_prompt_annotations`
    so the comparison is apples-to-apples rather than comparing marginal
    distributions.
    """
    dataset_keys: list[str] | None = None
    ncols: int = 2

    exclude_models = set(exclude_models)
    other_prompts = [p for p in prompt_names if p != baseline_prompt]
    if dataset_keys is None:
        dataset_keys = DATASET_KEYS
    dataset_keys = [k for k in dataset_keys if k in human_datasets]

    records_by_dataset: dict[str, pd.DataFrame] = {}
    for key in dataset_keys:
        models = _order_models(
            set(find_annotation_files(annotations_dir, key, baseline_prompt))
            - exclude_models
        )
        records = []
        for model in models:
            wide = _paired_prompt_annotations(
                annotations_dir, key, model, prompt_names
            )
            if wide is None:
                continue
            for other in other_prompts:
                diff_label = f"{other} - {baseline_prompt}"
                for value in wide[other] - wide[baseline_prompt]:
                    records.append(
                        {"Model": model, "Diff": diff_label, "value": value}
                    )
        if records:
            records_by_dataset[key] = pd.DataFrame(records)

    if not records_by_dataset:
        print(
            "No datasets with >=2 matched prompts found; skipping prompt "
            "mean-diff plot."
        )
        return

    keys_with_data = [k for k in dataset_keys if k in records_by_dataset]
    n = len(keys_with_data)
    nrows = -(-n // ncols)  # ceil division

    fig, axes = plt.subplots(nrows, ncols, squeeze=False)

    for i, key in enumerate(keys_with_data):
        ax = axes[i // ncols][i % ncols]
        plot_df = records_by_dataset[key]

        diff_order = [
            f"{other} - {baseline_prompt}"
            for other in other_prompts
            if f"{other} - {baseline_prompt}" in set(plot_df["Diff"])
        ]
        model_order = _order_models(set(plot_df["Model"]))
        palette = dict(zip(diff_order, tasks.graphs.COLORBLIND_PALETTE[1:]))

        sns.barplot(
            data=plot_df,
            x="Model",
            y="value",
            hue="Diff",
            hue_order=diff_order,
            order=model_order,
            errorbar="se",
            capsize=0.15,
            palette=palette,
            ax=ax,
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(human_datasets[key].get_name())
        ax.set_xlabel("")
        ax.set_ylabel(f"Mean annotation diff vs. {baseline_prompt}")
        ax.tick_params(axis="x", rotation=30)

        legend = ax.get_legend()
        if legend is not None:
            legend.set_title(None)
            for text in legend.get_texts():
                text.set_fontsize(10)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(
        "Mean difference (\u00b1 SE) in LLM annotations across prompt "
        "variants",
        y=1.02,
    )
    fig.tight_layout()
    tasks.graphs.save_plot(output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Consistency tables (Krippendorff's alpha)
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
    series_list = []
    for label, df in dfs.items():
        d = df.dropna(subset=["annotation_clean"]).copy()
        d["_key"] = list(zip(*[d[c] for c in key_cols]))
        d = d.drop_duplicates(subset="_key")
        s = d.set_index("_key")["annotation_clean"]
        s.name = label
        series_list.append(s)

    if not series_list:
        return np.empty((0, 0)), 0

    wide = pd.concat(series_list, axis=1)
    matrix = wide.to_numpy(dtype=float).T
    return matrix, wide.shape[0]


def krippendorff_alpha_safe(matrix: np.ndarray) -> float:
    if matrix.shape[0] < 2 or matrix.shape[1] == 0:
        return np.nan
    try:
        return krippendorff.alpha(
            reliability_data=matrix, level_of_measurement="ordinal"
        )
    except (ValueError, ZeroDivisionError):
        return np.nan


def cross_model_consistency_table(
    human_datasets: tasks.preprocessing.LazyDatasetLoader,
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
    for key in DATASET_KEYS:
        if key not in human_datasets:
            continue
        files = find_annotation_files(annotations_dir, key, prompt_name)
        files = {
            pseudo: path
            for pseudo, path in files.items()
            if pseudo not in exclude_models
        }
        if len(files) < 2:
            continue

        dfs = {pseudo: load_llm_df(path) for pseudo, path in files.items()}
        key_cols = ["text_id"] + _persona_columns(next(iter(dfs.values())))
        matrix, n_items = _build_reliability_matrix(dfs, key_cols)
        alpha = krippendorff_alpha_safe(matrix)

        rows.append(
            {
                "Dataset": human_datasets[key].get_name(),
                "Prompt": prompt_name,
                "Models": matrix.shape[0],
                "Items": n_items,
                "Krippendorff's alpha": alpha,
            }
        )
    return pd.DataFrame(rows)


def per_model_variant_consistency_table(
    human_datasets: tasks.preprocessing.LazyDatasetLoader,
    paraphrase_dir: Path,
    variant_names: list[str] = VARIANT_NAMES,
) -> pd.DataFrame:
    """
    For each (dataset, model): how consistent is that model with itself
    across the paraphrased prompt variants (variant1/variant2/variant3)?
    """
    rows = []
    for key in DATASET_KEYS:
        if key not in human_datasets:
            continue

        files_by_variant = {
            v: find_annotation_files(paraphrase_dir, key, v)
            for v in variant_names
        }
        non_empty = [f for f in files_by_variant.values() if f]
        if not non_empty:
            continue

        models = sorted(
            set.union(*(set(f) for f in files_by_variant.values()))
        )

        for pseudo in models:
            dfs = {}
            for v in variant_names:
                path = files_by_variant[v].get(pseudo)
                if path is None:
                    continue
                dfs[v] = load_llm_df(path)

            if len(dfs) < 2:
                continue

            key_cols = ["text_id"] + _persona_columns(next(iter(dfs.values())))
            matrix, n_items = _build_reliability_matrix(dfs, key_cols)
            alpha = krippendorff_alpha_safe(matrix)

            rows.append(
                {
                    "Dataset": human_datasets[key].get_name(),
                    "Model": pseudo,
                    "Variants": matrix.shape[0],
                    "Items": n_items,
                    "Krippendorff's alpha": alpha,
                }
            )
    return pd.DataFrame(rows)


def per_model_repeat_consistency_table(
    human_datasets: tasks.preprocessing.LazyDatasetLoader,
    repeat_dir: Path,
    prompt_name: str = "default",
) -> pd.DataFrame:
    """
    For each (dataset, model): how consistent is that model with itself
    across repeated runs of the *same* prompt (the "-run0" .. "-runN"
    repeat ablation in output/ablations/repeat)?
    """
    rows = []
    for key in DATASET_KEYS:
        if key not in human_datasets:
            continue

        files_by_model = find_repeat_files(repeat_dir, key, prompt_name)
        if not files_by_model:
            continue

        for pseudo, run_files in sorted(files_by_model.items()):
            dfs = {
                run_label: load_llm_df(path)
                for run_label, path in sorted(run_files.items())
            }
            if len(dfs) < 2:
                continue

            key_cols = ["text_id"] + _persona_columns(next(iter(dfs.values())))
            matrix, n_items = _build_reliability_matrix(dfs, key_cols)
            alpha = krippendorff_alpha_safe(matrix)

            rows.append(
                {
                    "Dataset": human_datasets[key].get_name(),
                    "Model": pseudo,
                    "Runs": matrix.shape[0],
                    "Items": n_items,
                    "Krippendorff's alpha": alpha,
                }
            )
    return pd.DataFrame(rows)


def export_latex_table(
    df: pd.DataFrame, output_path: Path, caption: str, label: str
) -> None:
    if _skip_if_exists(output_path):
        return

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


def _sample_text_ids(
    annotations_dir: Path, dataset_key: str, prompt_name: str
) -> set:
    """
    Every text_id appearing in any model's annotation CSV for
    (dataset_key, prompt_name) -- i.e. the comments llm_annotate.py
    actually sampled for that (dataset, prompt). Shared by
    _human_sample_dataset and the inherent-polarization comparison so both
    restrict to exactly the same comment set the same way.
    """
    ids: set = set()
    for path in find_annotation_files(
        annotations_dir, dataset_key, prompt_name
    ).values():
        ids.update(pd.read_csv(path, usecols=["text_id"])["text_id"])
    return ids


def _human_sample_dataset(
    ds_human: tasks.preprocessing.Dataset,
    annotations_dir: Path,
    dataset_key: str,
    prompt_name: str,
) -> tasks.preprocessing.Dataset | None:
    """
    Restricts `ds_human` down to just the comments actually sampled by
    llm_annotate.py for (dataset_key, prompt_name) -- i.e. the same
    "text_id"s that appear in the LLM annotation CSVs, since
    llm_annotate.py's text_id *is* the human dataset's own comment-key
    column value (see sample_texts() in llm_annotate.py). This is what
    lets a "Human" column show the human annotations for the exact same
    sample the LLM columns use, rather than the full dataset the way
    sap.png/kumar.png/etc. do. Reuses tasks.preprocessing.SubsampledView
    (rather than a new Dataset subclass) since it already does exactly
    this -- wrap a filtered DataFrame while delegating every other
    Dataset method to the original.

    Returns None if no LLM annotation files exist for this (dataset,
    prompt), since there's then no sample to restrict to.
    """
    sample_ids = _sample_text_ids(annotations_dir, dataset_key, prompt_name)
    if not sample_ids:
        return None

    comment_col = ds_human.get_comment_key_column()
    df = ds_human.get_dataset()
    restricted = df[df[comment_col].isin(sample_ids)]
    return tasks.preprocessing.SubsampledView(ds_human, restricted)


def _limited_sdb_columns(
    ds: tasks.preprocessing.Dataset, limit: int | None
) -> list[str]:
    """Caps ds.get_sdb_columns() to the first `limit` entries (or returns
    them unchanged if `limit` is None), without needing to mutate `ds`."""
    cols = ds.get_sdb_columns()
    return cols if limit is None else cols[:limit]


def plot_apunim_grid(
    human_datasets: tasks.preprocessing.LazyDatasetLoader,
    annotations_dir: Path,
    output_path: Path,
    prompt_name: str = "default",
    models: list[str] | None = None,
    sdb_columns_limit: int | None = 6,
    title: str = "LLM Polarization is Disconnected From Humans",
) -> None:
    """Plot all datasets and models in a single 2x5 grid using subfigures.

    Each dataset occupies one subfigure (row), with one subplot per
    Human/model column. A single title is placed above each dataset row.

    `title` is the figure-level suptitle. It defaults to the main
    ("default" prompt) plot's title; callers producing a separate grid per
    adversarial prompt (see main()) pass something that just names the
    instruction instead, e.g. "Stereotype Prompt".
    """
    dataset_keys = [
        k for k in PROMPT_COMPARISON_DATASET_KEYS if k in human_datasets
    ]

    if not dataset_keys:
        print(
            "No non-DICES datasets available; skipping composite apunim grid."
        )
        return

    if models is None:
        found = set()
        for key in dataset_keys:
            found |= set(
                find_annotation_files(annotations_dir, key, prompt_name)
            )
        models = _order_models(found)

    if not models:
        print("No LLM annotation files found; skipping composite apunim grid.")
        return

    # One column for Human + one for each model.
    columns = ["Human"] + models

    fig = plt.figure(constrained_layout=True)

    subfigures = fig.subfigures(
        nrows=len(dataset_keys),
        ncols=1,
        height_ratios=[1] * len(dataset_keys),
    )

    # When there is only one dataset, matplotlib returns a single SubFigure
    # rather than an array.
    if len(dataset_keys) == 1:
        subfigures = [subfigures]

    for r, dataset_key in enumerate(dataset_keys):
        subfig = subfigures[r]

        # Single title for the entire dataset row.
        subfig.suptitle(human_datasets[dataset_key].get_name())

        # 1 row x N columns within this subfigure.
        axes = subfig.subplots(
            nrows=1,
            ncols=len(columns),
            squeeze=False,
        )[0]

        ds_human = human_datasets[dataset_key]
        human_ds = _human_sample_dataset(
            ds_human,
            annotations_dir,
            dataset_key,
            prompt_name,
        )
        files = find_annotation_files(
            annotations_dir,
            dataset_key,
            prompt_name,
        )

        for c, column in enumerate(columns):
            if c == 0:
                color = tasks.graphs.COLORBLIND_PALETTE[1]
            else:
                color = tasks.graphs.COLORBLIND_PALETTE[2]

            ax = axes[c]

            if column == "Human":
                if human_ds is None:
                    ax.axis("off")
                    continue
                ds = human_ds
            else:
                path = files.get(column)
                if path is None:
                    ax.axis("off")
                    continue

                df = load_llm_df(path)
                ds = LLMAnnotationDataset(
                    df,
                    dataset_key,
                    column,
                    prompt_name,
                )

            plot_df = _compute_ndfu_records(
                ds,
                sdb_columns=_limited_sdb_columns(
                    ds,
                    sdb_columns_limit,
                ),
            )

            if plot_df.empty:
                ax.axis("off")
                continue

            sns.boxplot(
                x="PC Dimension",
                y="nDFU",
                data=plot_df,
                ax=ax,
                fliersize=1,
                linewidth=0.6,
                color=color,
            )

            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.grid(axis="y", alpha=0.3, linewidth=0.4)

            # Column titles only on the first dataset row.
            if r == 0:
                ax.set_title(column)

            # Only the first column gets the y-axis labels.
            if c != 0:
                ax.set_yticklabels([])

    # One shared y-axis label for the whole figure.
    fig.supylabel("nDFU")
    fig.suptitle(title)

    tasks.graphs.save_plot(output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Apunim-by-prompt LaTeX table (default / stereotype / persona columns)
# ---------------------------------------------------------------------------


def compute_llm_apunim_by_prompt(
    annotations_dir: Path,
    dataset_key: str,
    prompt_names: list[str] = MAIN_PROMPT_NAMES,
    exclude_models: set[str] | None = None,
) -> pd.DataFrame:
    """
    Runs the same apunim analysis as tasks.run_helper.run_all_results (the
    pipeline sap.py/dices.py/kumar.py use to produce their "-results.csv"
    files), but over the LLM annotation CSVs, once per (prompt, model)
    available for `dataset_key`. Returns a long-format DataFrame with one
    row per (SDB Feature, Value, Model, Prompt) combination and columns
    'apunim', 'pvalue', 'support', ready to be pivoted into a table.
    """
    exclude_models = set(exclude_models or ())
    rows = []
    for prompt_name in prompt_names:
        files = find_annotation_files(
            annotations_dir, dataset_key, prompt_name
        )
        models = _order_models(set(files) - exclude_models)
        for pseudo in models:
            df = load_llm_df(files[pseudo])
            ds = LLMAnnotationDataset(df, dataset_key, pseudo, prompt_name)
            try:
                res_df = tasks.run_helper.run_all_results(ds).reset_index()
            except ValueError as e:
                # E.g. "No polarized comments found." -- can happen for a
                # small/sparse (dataset, prompt, model) sample. Skip just
                # this combination rather than failing the whole table.
                print(
                    f"Skipping apunim for {dataset_key}/{prompt_name}/"
                    f"{pseudo}: {e}"
                )
                continue
            # The 2nd column is the (unnamed) per-SDB-column factor level,
            # e.g. "Age" -> "1) Gen. X+"; rename positionally since its
            # actual column label depends on pandas' index-naming, not on
            # anything we control here (see run_all_results).
            res_df = res_df.rename(columns={res_df.columns[1]: "Value"})
            res_df["Model"] = pseudo
            res_df["Prompt"] = prompt_name
            rows.append(res_df)

    if not rows:
        return pd.DataFrame(
            columns=np.array(
                [
                    "SDB Feature",
                    "Value",
                    "Model",
                    "Prompt",
                    "apunim",
                    "pvalue",
                    "support",
                ]
            )
        )
    return pd.concat(rows, ignore_index=True)


def export_llm_apunim_prompt_table(
    df: pd.DataFrame, output_path: Path, dataset_name: str, label: str
) -> None:
    if _skip_if_exists(output_path):
        return
    if df.empty:
        print(
            f"No LLM apunim-by-prompt results for {dataset_name}; "
            f"skipping {output_path}."
        )
        return

    df = df.replace("_", r"\_", regex=True).set_index(
        ["SDB Feature", "Value", "Model"]
    )

    latex_str = df.to_latex(
        caption=(
            "Aposteriori unimodality results for the LLM annotations of "
            f"the {dataset_name} dataset, across the default, stereotype "
            "and persona prompts."
        ),
        label=label,
        escape=False,  # allow LaTeX math ($^{*}$) already in the cells
        position="ht",
        index=True,
        multirow=True,
        longtable=True,
    )
    latex_str = latex_str.replace(
        r"\begin{table}[ht]", r"\begin{table}[ht]\centering"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_str)
    print(f"Table exported to {output_path.resolve()}")


def build_llm_apunim_prompt_table(
    long_df: pd.DataFrame, prompt_names: list[str] = MAIN_PROMPT_NAMES
) -> pd.DataFrame:
    """
    Pivots `compute_llm_apunim_by_prompt`'s long-format output into one row
    per (SDB Feature, Value, Model) and one column per prompt, so a given
    (dataset, model, SDB group)'s apunim value can be compared across the
    default/stereotype/persona prompts directly. Cells combine the apunim
    value with its significance stars (matching
    tasks.run_helper.results_to_latex); missing (prompt, model) results
    show as "---".
    """
    if long_df.empty:
        return pd.DataFrame()

    long_df = long_df.copy()
    long_df["cell"] = long_df.apply(
        lambda r: (
            "---"
            if pd.isna(r["apunim"])
            else (
                f"{r['apunim']:.4f}"
                f"{tasks.run_helper.significance_superscript(r['pvalue'])}"
            )
        ),
        axis=1,
    )

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
    wide = wide.sort_values(["SDB Feature", "Value", "Model"]).reset_index(
        drop=True
    )
    return wide


# ---------------------------------------------------------------------------
# 5. Inherent-polarization comparison (Human vs. LLM), default + adversarial
# ---------------------------------------------------------------------------


def _human_inherent_polarization(
    dataset_key: str,
    human_ds: tasks.preprocessing.Dataset,
    sample_ids: set,
    human_results_dir: Path,
) -> pd.Series:
    """
    Human inherent-polarization values, restricted to `sample_ids` (the
    same comments the LLMs were run on -- see _human_sample_dataset).

    Reuses the precomputed output/main/<dataset>-inherent.csv written by
    sap.py/kumar.py/dices.py when present: subsetting an *already
    computed* per-comment Series to a smaller comment set is exact, not
    an approximation, since inherent polarization is computed
    independently per comment. Only recomputes -- mirroring each
    dataset's own choice of exhaustive (sap/kumar) vs. random (dices, see
    compute_inherent_polarization_random/_exhaustive) -- when no cached
    file is found.
    """
    cached_path = human_results_dir / f"{dataset_key}-inherent.csv"
    if cached_path.exists():
        series = pd.read_csv(cached_path, index_col="comment")[
            "inherent_polarization"
        ]
        return series[series.index.isin(sample_ids)].dropna()

    fn = (
        tasks.run_helper.compute_inherent_polarization_random
        if dataset_key.startswith("dices")
        else tasks.run_helper.compute_inherent_polarization_exhaustive
    )
    return fn(human_ds).dropna()


def _llm_inherent_polarization(
    dataset_key: str,
    prompt_name: str,
    pseudo: str,
    path: Path,
    apunim_output_dir: Path,
) -> pd.Series:
    """
    LLM inherent-polarization values for a single (dataset, prompt,
    model). Reuses the precomputed
    apunim_output_dir/<dataset>-<prompt>-<model>-inherent.csv when
    present (currently only written for the "default" prompt); otherwise
    computes it directly on an LLMAnnotationDataset built from `path`.
    Always exhaustive -- LLMs have at most MAX_ANNOTATORS_PER_ITEM (6)
    annotators per comment, so the exhaustive search sap.py/kumar.py use
    is trivially cheap here regardless of dataset.
    """
    cached_path = (
        apunim_output_dir
        / f"{dataset_key}-{prompt_name}-{pseudo}-inherent.csv"
    )
    if cached_path.exists():
        series = pd.read_csv(cached_path, index_col="comment")[
            "inherent_polarization"
        ]
        return series.dropna()

    df = load_llm_df(path)
    ds = LLMAnnotationDataset(df, dataset_key, pseudo, prompt_name)
    return tasks.run_helper.compute_inherent_polarization_exhaustive(
        ds
    ).dropna()


def compute_inherent_polarization_comparison(
    human_datasets: tasks.preprocessing.LazyDatasetLoader,
    annotations_dir: Path,
    apunim_output_dir: Path,
    human_results_dir: Path,
    dataset_keys: list[str] = DATASET_KEYS,
    prompt_names: list[str] = MAIN_PROMPT_NAMES,
    exclude_models: set[str] | None = None,
) -> pd.DataFrame:
    """
    Long-format DataFrame with one row per comment giving that comment's
    inherent polarization, for every (Dataset, Prompt, Source) where
    Source is "Human" or a model pseudo -- restricted, for both Human and
    every model, to the sample of comments the LLMs were actually run on
    for that (dataset, prompt) (see _human_sample_dataset). (dataset,
    prompt) combinations with no LLM annotation files (e.g. DICES'
    stereotype/persona columns, which were never run) are skipped.
    Columns: 'Dataset', 'Prompt', 'Source', 'TextID', 'value'.
    """
    exclude_models = set(exclude_models or ())
    records = []
    for key in dataset_keys:
        if key not in human_datasets:
            continue
        ds_human = human_datasets[key]

        for prompt_name in prompt_names:
            files = find_annotation_files(annotations_dir, key, prompt_name)
            models = _order_models(set(files) - exclude_models)
            if not models:
                continue

            sample_ids = _sample_text_ids(annotations_dir, key, prompt_name)
            human_ds = _human_sample_dataset(
                ds_human, annotations_dir, key, prompt_name
            )
            if human_ds is not None:
                human_series = _human_inherent_polarization(
                    key, human_ds, sample_ids, human_results_dir
                )
                for text_id, value in human_series.items():
                    records.append(
                        {
                            "Dataset": key,
                            "Prompt": prompt_name,
                            "Source": "Human",
                            "TextID": text_id,
                            "value": value,
                        }
                    )

            for pseudo in models:
                llm_series = _llm_inherent_polarization(
                    key, prompt_name, pseudo, files[pseudo], apunim_output_dir
                )
                for text_id, value in llm_series.items():
                    records.append(
                        {
                            "Dataset": key,
                            "Prompt": prompt_name,
                            "Source": pseudo,
                            "TextID": text_id,
                            "value": value,
                        }
                    )

    return pd.DataFrame(records)


def build_inherent_polarization_table(
    long_df: pd.DataFrame,
    human_datasets: dict[str, pd.DataFrame],
    dataset_keys: list[str],
    prompt_names: list[str],
) -> pd.DataFrame:
    """
    Build the inherent-polarization table as mean ± 2 SD.

    Rows are (Dataset, Source), columns are prompts.
    """
    table = (
        long_df.groupby(["Dataset", "Source", "Prompt"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )

    table["formatted"] = table.apply(
        lambda row: f"{row['mean']:.3f} $\\pm$ {2 * row['std']:.3f}",
        axis=1,
    )

    table = table.pivot(
        index=["Dataset", "Source"],
        columns="Prompt",
        values="formatted",
    )

    # Ensure prompts appear in the requested order.
    table = table.reindex(columns=prompt_names)

    # Replace missing combinations with "---".
    table = table.fillna("---")

    return table


def export_inherent_polarization_table(
    df: pd.DataFrame,
    output_path: Path,
    label: str = "tab:inherent-polarization",
    longtable: bool = True,
) -> None:
    if _skip_if_exists(output_path):
        return

    if df.empty:
        print(f"No inherent-polarization results; skipping {output_path}.")
        return

    # Dataset and Source are already the index from
    # build_inherent_polarization_table().
    df = df.copy()

    # Escape underscores in the index and column names/cells.
    df.index = pd.MultiIndex.from_tuples(
        [
            (str(dataset).replace("_", r"\_"),
             str(source).replace("_", r"\_"))
            for dataset, source in df.index
        ],
        names=df.index.names,
    )

    latex_str = df.to_latex(
        caption=(
            "Mean inherent polarization (mean $\\pm$ 2 SD) per "
            "(dataset, prompt), for Human and each LLM."
        ),
        label=label,
        escape=False,
        position="ht",
        index=True,
        multirow=True,
        longtable=longtable,
    )
    

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_str)

    print(f"Table exported to {output_path.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_human_datasets(
    dices_small_path: Path,
    dices_large_path: Path,
    sap_path: Path,
    kumar_path: Path,
) -> tasks.preprocessing.LazyDatasetLoader:
    return tasks.preprocessing.LazyDatasetLoader(
        {
            "dices-350": dices_small_path,
            "dices-990": dices_large_path,
            "sap": sap_path,
            "kumar": kumar_path,
        },
        dataset_keys=DATASET_KEYS,
        dataset_loaders=DATASET_LOADERS,
    )


def main(
    dices_small_path: Path,
    dices_large_path: Path,
    sap_path: Path,
    kumar_path: Path,
    annotations_dir: Path,
    paraphrase_dir: Path,
    repeat_dir: Path,
    graph_output_dir: Path,
    latex_output_dir: Path,
    exclude_models: list[str],
    prompt_name: str = "default",
):
    tasks.graphs.graph_setup()
    graph_output_dir.mkdir(parents=True, exist_ok=True)
    latex_output_dir.mkdir(parents=True, exist_ok=True)

    human_datasets = load_human_datasets(
        dices_small_path=dices_small_path,
        dices_large_path=dices_large_path,
        sap_path=sap_path,
        kumar_path=kumar_path,
    )

    # 1. Histograms: human vs. LLM annotation frequencies, per dataset.
    histogram_path = graph_output_dir / "human_vs_llm_histograms.png"
    if _skip_if_exists(histogram_path):
        pass
    else:
        plot_annotation_histograms(
            human_datasets=human_datasets,
            annotations_dir=annotations_dir,
            output_path=histogram_path,
            prompt_name=prompt_name,
        )

    # 1b. Mean difference (+/- SE) between the default/stereotype/persona
    #     prompts, per model, per dataset.
    prompt_diff_path = graph_output_dir / "llm_prompt_mean_diff.png"
    if _skip_if_exists(prompt_diff_path):
        pass
    else:
        plot_prompt_mean_diff(
            human_datasets=human_datasets,
            annotations_dir=annotations_dir,
            output_path=prompt_diff_path,
            prompt_names=MAIN_PROMPT_NAMES,
            exclude_models=exclude_models,
        )

    # 2. Consistency tables.
    cross_model_path = latex_output_dir / "llm-consistency-cross-model.tex"
    if _skip_if_exists(cross_model_path):
        pass
    else:
        cross_model_df = cross_model_consistency_table(
            human_datasets=human_datasets,
            annotations_dir=annotations_dir,
            prompt_name=prompt_name,
        )
        export_latex_table(
            cross_model_df,
            output_path=cross_model_path,
            caption=(
                "Consistency (Krippendorff's $\\alpha$, ordinal) between "
                f"LLMs given the same ({prompt_name}) prompt, per dataset."
            ),
            label="tab:llm-consistency-cross-model",
        )

    if exclude_models:
        cross_model_excl_path = (
            latex_output_dir / "llm-consistency-cross-model-excluding.tex"
        )
        if _skip_if_exists(cross_model_excl_path):
            pass
        else:
            cross_model_excl_df = cross_model_consistency_table(
                human_datasets=human_datasets,
                annotations_dir=annotations_dir,
                prompt_name=prompt_name,
                exclude_models=exclude_models,
            )
            excluded_str = ", ".join(sorted(exclude_models))
            export_latex_table(
                cross_model_excl_df,
                output_path=cross_model_excl_path,
                caption=(
                    "Consistency (Krippendorff's $\\alpha$, ordinal) "
                    f"between LLMs given the same ({prompt_name}) prompt, "
                    "per dataset, excluding the following models: "
                    f"{excluded_str}."
                ),
                label="tab:llm-consistency-cross-model-excluding",
            )

    variant_path = latex_output_dir / "llm-consistency-variants.tex"
    if _skip_if_exists(variant_path):
        pass
    else:
        variant_df = per_model_variant_consistency_table(
            human_datasets=human_datasets,
            paraphrase_dir=paraphrase_dir,
        )
        export_latex_table(
            variant_df,
            output_path=variant_path,
            caption=(
                "Consistency (Krippendorff's $\\alpha$, ordinal) of each "
                "model with itself across the three paraphrased prompt "
                "variants."
            ),
            label="tab:llm-consistency-variants",
        )

    repeat_path = latex_output_dir / "llm-consistency-repeats.tex"
    if _skip_if_exists(repeat_path):
        pass
    else:
        repeat_df = per_model_repeat_consistency_table(
            human_datasets=human_datasets,
            repeat_dir=repeat_dir,
            prompt_name=prompt_name,
        )
        export_latex_table(
            repeat_df,
            output_path=repeat_path,
            caption=(
                "Consistency (Krippendorff's $\\alpha$, ordinal) of each "
                f"model with itself across repeated runs of the same "
                f"({prompt_name}) prompt."
            ),
            label="tab:llm-consistency-repeats",
        )

    # 3. Apunim results for the LLM annotations, as a LaTeX table (one per
    #    dataset) with a column per prompt (default/stereotype/persona)
    #    instead of per-(dataset, model) "-results.csv" files. Restricted
    #    to the datasets the paraphrase/stereotype/persona prompts were
    #    actually run on (kumar, sap), excluding models that were only
    #    ever run on the default prompt.
    for key in PROMPT_COMPARISON_DATASET_KEYS:
        if key not in human_datasets:
            continue
        apunim_table_path = (
            latex_output_dir / f"llm-apunim-by-prompt-{key}.tex"
        )
        if _skip_if_exists(apunim_table_path):
            continue
        long_df = compute_llm_apunim_by_prompt(
            annotations_dir=annotations_dir,
            dataset_key=key,
            prompt_names=MAIN_PROMPT_NAMES,
            exclude_models=APUNIM_TABLE_EXCLUDE_MODELS,
        )
        wide_df = build_llm_apunim_prompt_table(
            long_df, prompt_names=MAIN_PROMPT_NAMES
        )
        export_llm_apunim_prompt_table(
            wide_df,
            output_path=apunim_table_path,
            dataset_name=human_datasets[key].get_name(),
            label=f"tab:llm-apunim-by-prompt-{key}",
        )

    # Single composite apunim grid figure (datasets x models).
    apunim_grid_path = graph_output_dir / "llm_apunim_grid.png"
    if _skip_if_exists(apunim_grid_path):
        pass
    else:
        plot_apunim_grid(
            human_datasets=human_datasets,
            annotations_dir=annotations_dir,
            output_path=apunim_grid_path,
            prompt_name=prompt_name,
            models=list(
                set(MODEL_DISPLAY_ORDER) - APUNIM_TABLE_EXCLUDE_MODELS
            ),
        )

    # The same composite apunim grid, but one per adversarial prompt
    # (stereotype/persona) instead of the default prompt, with a title
    # that just names the instruction rather than the main plot's title.
    for adv_prompt_name in ADVERSARIAL_PROMPT_NAMES:
        adv_grid_path = (
            graph_output_dir / f"llm_apunim_grid_{adv_prompt_name}.png"
        )
        if _skip_if_exists(adv_grid_path):
            continue
        plot_apunim_grid(
            human_datasets=human_datasets,
            annotations_dir=annotations_dir,
            output_path=adv_grid_path,
            prompt_name=adv_prompt_name,
            models=list(
                set(MODEL_DISPLAY_ORDER) - APUNIM_TABLE_EXCLUDE_MODELS
            ),
            title=f"{adv_prompt_name.capitalize()} Prompt",
        )

    # -----------------------------------------------------------------------
    # 5. Inherent polarization: Human vs. LLM annotations.
    # -----------------------------------------------------------------------

    # Human inherent-polarization cache:
    #   output/main/<dataset>-inherent.csv
    #
    # LLM inherent-polarization cache:
    #   output/apunim/<dataset>-<prompt>-<model>-inherent.csv
    #
    # These are deliberately NOT latex_output_dir.
    human_results_dir = Path("output/main")
    apunim_output_dir = Path("output/apunim")

    inherent_df = compute_inherent_polarization_comparison(
        human_datasets=human_datasets,
        annotations_dir=annotations_dir,
        apunim_output_dir=apunim_output_dir,
        human_results_dir=human_results_dir,
        dataset_keys=DATASET_KEYS[2:],
        prompt_names=MAIN_PROMPT_NAMES,
        exclude_models=set(exclude_models),
    )

    # Inherent-polarization LaTeX table.
    inherent_table_path = latex_output_dir / "inherent-polarization.tex"

    if not _skip_if_exists(inherent_table_path):
        inherent_table_df = build_inherent_polarization_table(
            long_df=inherent_df,
            human_datasets=human_datasets,  # type: ignore
            dataset_keys=DATASET_KEYS[2:],
            prompt_names=MAIN_PROMPT_NAMES,
        )

        export_inherent_polarization_table(
            df=inherent_table_df,
            output_path=inherent_table_path,
            label="tab:inherent-polarization",
            longtable=False
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
        default=None,
        help="Path to the DICES-350 CSV file.",
    )
    parser.add_argument(
        "--dices-large-path",
        default=None,
        help="Path to the DICES-990 CSV file.",
    )
    parser.add_argument(
        "--sap-path", default=None, help="Path to the Sap et al. CSV file."
    )
    parser.add_argument(
        "--kumar-path",
        default=None,
        help="Path to the Kumar et al. JSON file.",
    )
    parser.add_argument(
        "--annotations-dir",
        default="output/annotations",
        help=(
            "Directory containing the main (non-ablation) llm_annotate.py "
            "outputs, e.g. output/annotations."
        ),
    )
    parser.add_argument(
        "--paraphrase-dir",
        default="output/ablations/paraphrase",
        help=(
            "Directory containing the paraphrase-ablation llm_annotate.py "
            "outputs (variant1/variant2/variant3), e.g. "
            "output/ablations/paraphrase."
        ),
    )
    parser.add_argument(
        "--repeat-dir",
        default="output/ablations/repeat",
        help=(
            "Directory containing the repeat-ablation llm_annotate.py "
            "outputs (same prompt, run N times: '-run0', '-run1', ...), "
            "e.g. output/ablations/repeat."
        ),
    )
    parser.add_argument(
        "--graph-output-dir",
        default="graphs",
        help="Directory for the histogram and apunim polarization plots.",
    )
    parser.add_argument(
        "--latex-output-dir",
        default="manuscript/generated",
        help="Directory for the consistency LaTeX tables.",
    )
    parser.add_argument(
        "--prompt-name",
        default="default",
        help=(
            "Instruction-prompt stem (matches the instructions/*/<name>.txt "
            "file) whose LLM annotations are used for the histograms and "
            "cross-model consistency table."
        ),
    )
    parser.add_argument(
        "--exclude-models",
        nargs="+",
        default=[],
        help=(
            "Model pseudo names (e.g. olmo7b llama8b) to leave out of an "
            "additional cross-model consistency table, exported alongside "
            "the normal one."
        ),
    )
    args = parser.parse_args()

    main(
        dices_small_path=Path(args.dices_small_path),
        dices_large_path=Path(args.dices_large_path),
        sap_path=Path(args.sap_path),
        kumar_path=Path(args.kumar_path),
        annotations_dir=Path(args.annotations_dir),
        paraphrase_dir=Path(args.paraphrase_dir),
        repeat_dir=Path(args.repeat_dir),
        graph_output_dir=Path(args.graph_output_dir),
        latex_output_dir=Path(args.latex_output_dir),
        prompt_name=args.prompt_name,
        exclude_models=args.exclude_models,
    )
