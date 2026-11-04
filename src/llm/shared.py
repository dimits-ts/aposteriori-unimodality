"""
Shared constants and dataset/file-loading helpers used by stats.py,
plots.py, and polarization.py.

Nothing in this module produces a plot, a LaTeX table, or a statistical
test on its own -- it's the common substrate the other modules build on:
locating and loading llm_annotate.py output CSVs, and adapting them into the
Dataset shape lib.run_helper/lib.graphs expect (LLMAnnotationDataset).

Only things used by more than one module live here; anything used by a
single module is defined in that module.
"""

import re
from collections.abc import Collection, Iterable
from pathlib import Path

import pandas as pd

from ..lib.preprocessing import (
    DATASET_CLASSES,
    Dataset,
    LazyDatasetLoader,
    SubsampledView,
)

# ---------------------------------------------------------------------------
# Data vocabulary
# ---------------------------------------------------------------------------

# `Model` value of the rows holding human (rather than LLM) results, and the
# `Prompt` value they carry (human annotations do not depend on the prompt).
HUMAN_LABEL = "Human"
HUMAN_PROMPT_NAME = "human"

# `Dataset` value of rows pooled over all datasets.
POOLED_LABEL = "All"

# Prompt name of the no-persona baseline runs (llm_annotate.py
# --no-persona), i.e. the files "<dataset>-nopersona-<model>.csv" in the
# annotations directory, next to the persona runs.
NO_PERSONA_PROMPT_NAME = "nopersona"

# Columns of an llm_annotate.py output CSV (plus the "annotation_clean"
# column added by load_llm_df) that are *not* persona/SDB attributes.
_NON_PERSONA_COLS = frozenset(
    {
        "model",
        "instruction_prompt",
        "text_id",
        "text",
        "annotation",
        "annotation_clean",
    }
)

# Datasets whose LLM annotations are on the reverse of the human scale: a
# parsed reply v is mapped to min + max - v of the dataset's valid values
# (for Kumar et al., 1 <-> 5), so it matches the human annotations.
_REVERSED_SCALE_DATASETS = frozenset({"kumar"})

# Display order of the models in every table and figure (models not listed
# are appended alphabetically).
_MODEL_DISPLAY_ORDER = (
    "llama8b",
    "olmo32b",
    "olmo7b",
    "qwen32b",
    "qwen7b",
    "mistral7b",
    "mistral24b",
    "gemma7b",
    "gemma31b",
)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class HumanDatasets:
    """
    Dict-like container of one :class:`LazyDatasetLoader` per dataset key,
    supporting ``key in ds``, ``ds[key]`` and ``ds.keys()``. Each dataset is
    constructed at most once, on first access.
    """

    def __init__(self, loaders: dict[str, LazyDatasetLoader]) -> None:
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
    ) -> None:
        self._name = f"{dataset_key}-{prompt_name}-{model_pseudo}"
        persona_cols = _persona_columns(df)

        df = df.dropna(subset=["annotation_clean"]).copy()
        agg: dict[str, type[list]] = {col: list for col in persona_cols}
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


def available_dataset_keys(
    human_datasets: HumanDatasets, dataset_keys: Iterable[str]
) -> list[str]:
    """The keys of `dataset_keys` that are loaded in `human_datasets`."""
    return [k for k in dataset_keys if k in human_datasets]


def order_models(models: Collection[str]) -> list[str]:
    """`models` in display order (see _MODEL_DISPLAY_ORDER)."""
    known = [m for m in _MODEL_DISPLAY_ORDER if m in models]
    unknown = sorted(m for m in models if m not in _MODEL_DISPLAY_ORDER)
    return known + unknown


# ---------------------------------------------------------------------------
# Annotation files
# ---------------------------------------------------------------------------


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
    out: dict[str, Path] = {}
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


def dataset_key_of_file(path: Path) -> str:
    """
    Dataset key of an annotation file named "<key>-<prompt>-<model>
    [-run<N>].csv". Longest keys first, so that a key that is a prefix of
    another cannot match the wrong one.
    """
    for key in sorted(DATASET_CLASSES, key=len, reverse=True):
        if path.name.startswith(f"{key}-"):
            return key
    raise ValueError(
        f"Cannot tell the dataset of {path}: its name does not start with "
        f"any of {sorted(DATASET_CLASSES)}."
    )


def clean_annotation(series: pd.Series) -> pd.Series:
    """
    LLMs were asked to "reply with a single number only", but generations
    can still contain stray characters (whitespace, punctuation, a partial
    second token, ...). Extract the first signed integer found in each
    reply; anything that can't be parsed becomes NaN and is dropped
    downstream.
    """
    extracted = series.astype(str).str.extract(r"(-?\d+)")[0]
    return pd.to_numeric(extracted, errors="coerce")


def out_of_range(cleaned: pd.Series, valid: Collection[int]) -> pd.Series:
    """Parsed annotations (see clean_annotation) that are not NaN but not
    in `valid` either."""
    return cleaned.notna() & ~cleaned.isin(valid)


def load_llm_df(path: Path) -> pd.DataFrame:
    """
    An llm_annotate.py output CSV, with an added "annotation_clean" column:
    the reply parsed to an integer (see clean_annotation), NaN if it
    cannot be parsed or is outside the dataset's valid range
    (VALID_ANNOTATION_VALUES of the lib.preprocessing class of the dataset
    key in the file name), and reversed for the datasets in
    _REVERSED_SCALE_DATASETS. The raw "annotation" column is kept as is.
    """
    df = pd.read_csv(path)
    cleaned = clean_annotation(df["annotation"])
    dataset_key = dataset_key_of_file(path)
    valid = DATASET_CLASSES[dataset_key].VALID_ANNOTATION_VALUES
    cleaned = cleaned.mask(out_of_range(cleaned, valid))
    if dataset_key in _REVERSED_SCALE_DATASETS:
        cleaned = min(valid) + max(valid) - cleaned
    df["annotation_clean"] = cleaned
    return df


def _persona_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _NON_PERSONA_COLS]


def key_columns(dfs: dict[str, pd.DataFrame]) -> list[str]:
    """
    The columns that jointly identify a sampled (comment, persona) item --
    shared across models/prompts/variants because llm_annotate.py is seeded
    to draw the same items for all of them.
    """
    return ["text_id"] + _persona_columns(next(iter(dfs.values())))


def keyed_series(
    df: pd.DataFrame, key_cols: list[str], label: str
) -> pd.Series:
    """The parsed annotations of `df`, indexed by the tuple of `key_cols`
    (duplicates dropped) and named `label`."""
    d = df.dropna(subset=["annotation_clean"]).copy()
    d["_key"] = list(zip(*[d[c] for c in key_cols]))
    d = d.drop_duplicates(subset="_key")
    s = d.set_index("_key")["annotation_clean"]
    s.name = label
    return s


# ---------------------------------------------------------------------------
# The comments the LLMs saw, shared by the human/LLM comparisons
# ---------------------------------------------------------------------------


def sample_text_ids(
    annotations_dir: Path, dataset_key: str, prompt_name: str
) -> set[object]:
    """
    Every text_id appearing in any model's annotation CSV for
    (dataset_key, prompt_name) -- i.e. the comments llm_annotate.py
    actually sampled for that (dataset, prompt).
    """
    ids: set[object] = set()
    for path in find_annotation_files(
        annotations_dir, dataset_key, prompt_name
    ).values():
        ids.update(pd.read_csv(path, usecols=["text_id"])["text_id"])
    return ids


def human_sample_dataset(
    ds_human: Dataset,
    annotations_dir: Path,
    dataset_key: str,
    prompt_name: str,
) -> Dataset | None:
    """
    `ds_human` restricted to the comments sampled by llm_annotate.py for
    (dataset_key, prompt_name). None if no LLM annotation files exist for
    this (dataset, prompt).
    """
    sample_ids = sample_text_ids(annotations_dir, dataset_key, prompt_name)
    if not sample_ids:
        return None

    comment_col = ds_human.get_comment_key_column()
    df = ds_human.get_dataset()
    restricted = df[df[comment_col].isin(sample_ids)]
    # SubsampledView delegates everything else to the wrapped Dataset.
    return SubsampledView(ds_human, restricted)  # type: ignore[return-value]
