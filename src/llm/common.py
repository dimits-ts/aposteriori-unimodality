"""
Shared constants, dataset/file loading, and nDFU-record helpers used by
stats.py, plots.py, and polarization.py.

Nothing in this module produces a plot, a LaTeX table, or a statistical
test on its own -- it's the common substrate the other modules build on:
locating and loading llm_annotate.py output CSVs, adapting them into the
Dataset shape lib.run_helper/lib.graphs expect (LLMAnnotationDataset),
and computing the per-comment nDFU values that both the apunim grid
(plots.py) and the prompt-sensitivity ANOVA (stats.py) are built from.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ..lib.preprocessing import (
    DicesDataset,
    KumarDataset,
    SapDataset,
    Dataset,
    LazyDatasetLoader,
    SubsampledView
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# TODO: lazy loading doesnt work currently because of Dataset.get_name() calls
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
MAIN_PROMPT_NAMES = ["default", "stereotype", "persona", "single", "direct"]

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


class LLMAnnotationDataset(Dataset):
    """
    Adapts a single (dataset, prompt, model) llm_annotate.py output CSV --
    one row per (comment, persona) -- into the per-comment,
    list-of-annotators shape that lib.run_helper / lib.graphs expect,
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


# ---------------------------------------------------------------------------
# Generic helpers: dataset/model bookkeeping, file discovery, loading
# ---------------------------------------------------------------------------


def _persona_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_PERSONA_COLS]


def _order_models(models: set[str] | list[str]) -> list[str]:
    known = [m for m in MODEL_DISPLAY_ORDER if m in models]
    unknown = sorted(m for m in models if m not in MODEL_DISPLAY_ORDER)
    return known + unknown


def _available_dataset_keys(
    human_datasets: LazyDatasetLoader,
    dataset_keys: list[str] = DATASET_KEYS,
) -> list[str]:
    return [k for k in dataset_keys if k in human_datasets]


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
    import re

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


def load_llm_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["annotation_clean"] = _clean_annotation(df["annotation"])
    return df


def _key_columns(dfs: dict[str, pd.DataFrame]) -> list[str]:
    """
    The columns that jointly identify a sampled (comment, persona) item --
    shared across models/prompts/variants because llm_annotate.py is seeded
    to draw the same items for all of them (see module docstring).
    """
    return ["text_id"] + _persona_columns(next(iter(dfs.values())))


def _keyed_series(
    df: pd.DataFrame, key_cols: list[str], label: str
) -> pd.Series:
    d = df.dropna(subset=["annotation_clean"]).copy()
    d["_key"] = list(zip(*[d[c] for c in key_cols]))
    d = d.drop_duplicates(subset="_key")
    s = d.set_index("_key")["annotation_clean"]
    s.name = label
    return s


# ---------------------------------------------------------------------------
# Sampled-comment helpers, shared by the apunim grid (plots.py) and the
# inherent-polarization comparison (polarization.py) -- both need "the
# same items the LLMs saw".
# ---------------------------------------------------------------------------


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
    ds_human: Dataset,
    annotations_dir: Path,
    dataset_key: str,
    prompt_name: str,
) -> Dataset | None:
    """
    Restricts `ds_human` down to just the comments actually sampled by
    llm_annotate.py for (dataset_key, prompt_name) -- i.e. the same
    "text_id"s that appear in the LLM annotation CSVs, since
    llm_annotate.py's text_id *is* the human dataset's own comment-key
    column value (see sample_texts() in llm_annotate.py). This is what
    lets a "Human" column show the human annotations for the exact same
    sample the LLM columns use, rather than the full dataset the way
    sap.png/kumar.png/etc. do. Reuses lib.preprocessing.SubsampledView
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
    return SubsampledView(ds_human, restricted)


def _limited_sdb_columns(ds: Dataset, limit: int | None) -> list[str]:
    """Caps ds.get_sdb_columns() to the first `limit` entries (or returns
    them unchanged if `limit` is None), without needing to mutate `ds`."""
    cols = ds.get_sdb_columns()
    return cols if limit is None else cols[:limit]


# ---------------------------------------------------------------------------
# nDFU-by-SDB-group records, shared by the composite apunim grid (plots.py)
# and the prompt-sensitivity ANOVA (stats.py)
# ---------------------------------------------------------------------------


def _apunim_dfu(annotations, bins: int) -> float:
    import apunim  # local import: heavy-ish, only needed for this call

    return apunim.dfu(annotations, bins=bins, normalized=True)


def _ndfu_bin_count(annotation_lists) -> int:
    all_annotations = [
        v
        for lst in annotation_lists
        if isinstance(lst, (list, np.ndarray))
        for v in lst
    ]
    return len(np.unique(all_annotations)) if all_annotations else 0


def _ndfu_records_for_row(
    row: pd.Series, annotation_col: str, sdb_columns: list[str], bins: int
) -> list[dict]:
    """
    One record per (annotator persona value, SDB column) for this comment
    -- i.e. a comment's single nDFU value is broadcast once per annotator
    whose persona matches a given group. This is what plot_apunim_grid's
    boxplots are built from (see plots.py); it is NOT deduplicated per
    comment, so a comment with several annotators sharing the same SDB
    value contributes that many copies of its nDFU to that group. For a
    per-comment (non-pseudoreplicated) version, see
    _ndfu_records_for_row_deduped below, used by the ANOVA in stats.py.
    """
    annotations = row[annotation_col]
    if (
        not isinstance(annotations, (list, np.ndarray))
        or len(annotations) == 0
    ):
        return []
    try:
        ndfu_value = _apunim_dfu(annotations, bins)
    except Exception as e:
        print(f"Error calculating NDFU for an item: {e}")
        return []

    return [
        {"PC Dimension": f"{sdb_col}: {value}", "nDFU": ndfu_value}
        for sdb_col in sdb_columns
        for value in row[sdb_col]
    ]


def _ndfu_records_for_row_deduped(
    row: pd.Series, annotation_col: str, sdb_columns: list[str], bins: int
) -> list[dict]:
    """
    Same as `_ndfu_records_for_row`, but emits at most ONE record per
    (comment, PC Dimension) instead of one per matching annotator. A
    comment has a single nDFU value; if it should count as evidence for
    e.g. "Gender: Female" at all, it should count once, not once per
    annotator who happened to be sampled as Female. Avoids the
    pseudoreplication in `_ndfu_records_for_row`, which inflates a group's
    observation count from <=N comments to close to N * annotators/comment,
    both misreporting the true sample size and violating the independence
    assumption of downstream significance tests (see stats.py's
    compute_ndfu_anova_by_prompt).
    """
    annotations = row[annotation_col]
    if (
        not isinstance(annotations, (list, np.ndarray))
        or len(annotations) == 0
    ):
        return []
    try:
        ndfu_value = _apunim_dfu(annotations, bins)
    except Exception as e:
        print(f"Error calculating NDFU for an item: {e}")
        return []

    return [
        {"PC Dimension": f"{sdb_col}: {value}", "nDFU": ndfu_value}
        for sdb_col in sdb_columns
        for value in set(row[sdb_col])
    ]


def _compute_ndfu_records(
    ds: Dataset,
    sdb_columns: list[str] | None = None,
    deduped: bool = False,
) -> pd.DataFrame:
    """
    Per-comment nDFU (apunim.dfu over that comment's annotator list),
    broadcast onto every SDB group any of its annotators belonged to --
    the same computation lib.graphs.polarization_plot does internally,
    factored out here so it can be drawn onto an arbitrary subplot axis
    (plots.py) or fed into a significance test (stats.py) instead of
    always producing its own standalone figure.

    `sdb_columns`, if given, overrides `ds.get_sdb_columns()` (e.g. to cap
    how many SDB dimensions are included) without needing to mutate `ds`.

    `deduped`, if True, uses `_ndfu_records_for_row_deduped` so each
    comment contributes at most one row per (comment, PC Dimension)
    instead of one row per matching annotator persona. Plotting code
    (plots.py) defaults to False to match the original apunim-grid
    behavior; statistical tests (stats.py) should pass True to avoid
    pseudoreplication.
    """
    df = ds.get_dataset()
    annotation_col = ds.get_annotation_column()
    if sdb_columns is None:
        sdb_columns = ds.get_sdb_columns()

    bins = _ndfu_bin_count(df[annotation_col].to_list())
    if bins == 0:
        return pd.DataFrame(columns=["PC Dimension", "nDFU"])

    row_fn = (
        _ndfu_records_for_row_deduped if deduped else _ndfu_records_for_row
    )
    records = []
    for _, row in df.iterrows():
        records.extend(row_fn(row, annotation_col, sdb_columns, bins))
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Human dataset loading
# ---------------------------------------------------------------------------


def load_human_datasets(
    dices_small_path: Path,
    dices_large_path: Path,
    sap_path: Path,
    kumar_path: Path,
) -> LazyDatasetLoader:
    return LazyDatasetLoader(
        {
            "dices-350": dices_small_path,
            "dices-990": dices_large_path,
            "sap": sap_path,
            "kumar": kumar_path,
        },
        dataset_keys=DATASET_KEYS,
        dataset_loaders=DATASET_LOADERS,
    )
