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
    Dataset,
    SubsampledView,
    LazyDatasetLoader,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_KEYS = ["dices-350", "dices-990", "sap", "kumar", "popquorn"]

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

# Inherent-polarization subsampling ablation: every comment (human and LLM
# alike) is repeatedly subsampled down to this many annotators and the
# results averaged over this many repeats.
INHERENT_SUBSAMPLE_SIZE = 6
INHERENT_SUBSAMPLE_REPEATS = 10

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

# Datasets used for the main annotation runs (default + adversarial
# prompts). The DICES datasets are deliberately excluded: they were only
# run for the ablations below.
MAIN_DATASET_KEYS = DATASET_KEYS

# Grid of the subsampled inherent-polarization histogram figure: datasets
# are the columns, prompts the rows (all main prompts, as in the table).
INHERENT_HIST_DATASET_KEYS = ["dices-990", "sap", "kumar", "popquorn"]
INHERENT_HIST_PROMPT_NAMES = MAIN_PROMPT_NAMES

# Datasets used for the ablations (paraphrase variants, repeated runs).
ABLATION_DATASET_KEYS = DATASET_KEYS

# Datasets for which all MAIN_PROMPT_NAMES were run -- used for the prompt
# mean-diff plot, the apunim-by-prompt LaTeX table and grids, and the
# prompt-sensitivity ANOVA.
PROMPT_COMPARISON_DATASET_KEYS = MAIN_DATASET_KEYS

# Models excluded from the apunim-by-prompt LaTeX table (they were never
# run on the stereotype/persona prompts to begin with; listed explicitly
# so the table is correct even if that changes).
# Also used for LLM polarization grid.
APUNIM_TABLE_EXCLUDE_MODELS = {"olmo7b", "llama8b"}


# ---------------------------------------------------------------------------
# HumanDatasets: dict-like container of per-key LazyDatasetLoaders
# ---------------------------------------------------------------------------


class HumanDatasets:
    """
    Dict-like container that holds one :class:`LazyDatasetLoader` per
    dataset key and exposes the same ``key in ds``, ``ds[key]``,
    ``ds.keys()`` interface that the rest of the codebase uses, so no
    call sites need to change.

    Each dataset is constructed at most once: the first access to
    ``ds[key]`` calls that key's loader factory; subsequent accesses
    return the cached instance.
    """

    def __init__(
        self,
        loaders: dict[str, LazyDatasetLoader],
    ) -> None:
        self._loaders = loaders

    def __contains__(self, key: object) -> bool:
        return key in self._loaders

    def __getitem__(self, key: str) -> Dataset:
        return self._loaders[key].get()

    def keys(self) -> list[str]:
        return list(self._loaders)


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
    human_datasets: HumanDatasets,
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
    llm_annotate.py for (dataset_key, prompt_name). Returns None if no
    LLM annotation files exist for this (dataset, prompt).
    """
    sample_ids = _sample_text_ids(annotations_dir, dataset_key, prompt_name)
    if not sample_ids:
        return None

    comment_col = ds_human.get_comment_key_column()
    df = ds_human.get_dataset()
    restricted = df[df[comment_col].isin(sample_ids)]
    return SubsampledView(ds_human, restricted)


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
    One record per (annotator persona value, SDB column) for this comment.
    Not deduplicated per comment; see _ndfu_records_for_row_deduped for
    the ANOVA-safe version.
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
    (comment, PC Dimension) to avoid pseudoreplication in significance
    tests (see stats.py's compute_ndfu_anova_by_prompt).
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
    Per-comment nDFU broadcast onto every SDB group any of its annotators
    belonged to. `sdb_columns` overrides `ds.get_sdb_columns()` without
    mutating `ds`. `deduped=True` avoids pseudoreplication (use for
    statistical tests; plotting defaults to False).
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
