from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from ..lib import run_helper
from ..lib.preprocessing import Dataset
from ..lib.util import (
    skip_if_exists,
    center_table_latex,
    small_table_latex,
    trim_numeric_col_latex,
)
from .shared import (
    HumanDatasets,
    load_llm_df,
    LLMAnnotationDataset,
    find_annotation_files,
    _sample_text_ids,
    _human_sample_dataset,
    _order_models,
    _available_dataset_keys,
    DATASET_KEYS,
    MAIN_PROMPT_NAMES,
)


# ---------------------------------------------------------------------------
# 5. Inherent-polarization comparison (Human vs. LLM), default + adversarial
# ---------------------------------------------------------------------------
#
# Progress is reported with three nested tqdm bars:
#   position 0: datasets
#   position 1: prompts within the current dataset
#   position 2: sources (Human, then each model) within the current prompt
#   position 3: comments within the current source (only shown when the
#               values are actually computed, not read from a cache)

COMMENT_BAR_POSITION = 3


@dataclass(frozen=True)
class SubsampleSpec:
    """Repeatedly subsample every comment down to `size` annotators."""

    size: int
    n_repeats: int
    seed: int = 42

    @property
    def tag(self) -> str:
        return f"subsampled-n{self.size}"


def _compute_inherent(
    ds: Dataset, subsample: SubsampleSpec | None, default_fn, **progress
) -> pd.Series:
    """Inherent polarization of `ds`: `default_fn` normally, the repeated
    annotator-subsampling variant when `subsample` is given."""
    if subsample is None:
        return default_fn(ds, **progress)
    return run_helper.compute_inherent_polarization_subsampled(
        ds,
        size=subsample.size,
        n_repeats=subsample.n_repeats,
        seed=subsample.seed,
        **progress,
    )


def _cached_series(cached_path: Path, compute_fn) -> pd.Series:
    """Read `cached_path` if it exists, else compute and write it."""
    if skip_if_exists(cached_path):
        series = pd.read_csv(cached_path, index_col="comment")[
            "inherent_polarization"
        ]
        return series.dropna()

    series = compute_fn()
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    series.rename("inherent_polarization").rename_axis("comment").to_csv(
        cached_path
    )
    return series.dropna()


def _human_inherent_polarization(
    dataset_key: str,
    prompt_name: str,
    human_ds: Dataset,
    sample_ids: set,
    human_results_dir: Path,
    apunim_output_dir: Path,
    use_monte_carlo: bool = False,
    subsample: SubsampleSpec | None = None,
) -> pd.Series:
    """
    Human inherent-polarization values, restricted to `sample_ids`.
    Normally reuses the precomputed <dataset>-inherent.csv when present
    (only recomputing when no cached file is found). With `subsample` the
    precomputed full-annotator file does not apply, so the values are
    computed on the subsampled data and cached in `apunim_output_dir`.
    """
    progress = dict(
        show_progress=True,
        progress_position=COMMENT_BAR_POSITION,
        progress_desc=f"    {dataset_key} Human: comments",
    )
    if subsample is not None:
        return _cached_series(
            apunim_output_dir
            / f"{dataset_key}-{prompt_name}-Human-inherent.csv",
            lambda: _compute_inherent(human_ds, subsample, None, **progress),
        )

    cached_path = human_results_dir / f"{dataset_key}-inherent.csv"
    if skip_if_exists(cached_path):
        series = pd.read_csv(cached_path, index_col="comment")[
            "inherent_polarization"
        ]
        return series[series.index.isin(sample_ids)].dropna()

    fn = (
        run_helper.compute_inherent_polarization_random
        if use_monte_carlo
        else run_helper.compute_inherent_polarization_exhaustive
    )
    return fn(human_ds, **progress).dropna()


def _llm_inherent_polarization(
    dataset_key: str,
    prompt_name: str,
    pseudo: str,
    path: Path,
    apunim_output_dir: Path,
    subsample: SubsampleSpec | None = None,
) -> pd.Series:
    """
    LLM inherent-polarization values for a single (dataset, prompt, model).
    Reuses a cached CSV in `apunim_output_dir` when present; otherwise
    computes exhaustively (or on repeatedly subsampled annotators, see
    `subsample`) and writes the result there.
    """

    def _compute() -> pd.Series:
        ds = LLMAnnotationDataset(
            load_llm_df(path), dataset_key, pseudo, prompt_name
        )
        return _compute_inherent(
            ds,
            subsample,
            run_helper.compute_inherent_polarization_exhaustive,
            show_progress=True,
            progress_position=COMMENT_BAR_POSITION,
            progress_desc=f"    {dataset_key}/{prompt_name}/{pseudo}: comments",
        )

    return _cached_series(
        apunim_output_dir
        / f"{dataset_key}-{prompt_name}-{pseudo}-inherent.csv",
        _compute,
    )


def _records_from_series(
    series: pd.Series, dataset_key: str, prompt_name: str, source: str
) -> list[dict]:
    return [
        {
            "Dataset": dataset_key,
            "Prompt": prompt_name,
            "Source": source,
            "TextID": text_id,
            "value": value,
        }
        for text_id, value in series.items()
    ]


def _inherent_records_for_models(
    dataset_key: str,
    prompt_name: str,
    models: list[str],
    files: dict[str, Path],
    apunim_output_dir: Path,
    progress: tqdm | None = None,
    subsample: SubsampleSpec | None = None,
) -> list[dict]:
    records = []
    for pseudo in models:
        if progress is not None:
            progress.set_postfix_str(pseudo)
        series = _llm_inherent_polarization(
            dataset_key,
            prompt_name,
            pseudo,
            files[pseudo],
            apunim_output_dir,
            subsample,
        )
        records.extend(
            _records_from_series(series, dataset_key, prompt_name, pseudo)
        )
        if progress is not None:
            progress.update(1)
    return records


def _inherent_records_for_prompt(
    key: str,
    ds_human: Dataset,
    prompt_name: str,
    annotations_dir: Path,
    apunim_output_dir: Path,
    human_results_dir: Path,
    exclude_models: set[str],
    use_monte_carlo: bool,
    subsample: SubsampleSpec | None = None,
) -> list[dict]:
    files = find_annotation_files(annotations_dir, key, prompt_name)
    models = _order_models(set(files) - exclude_models)
    if not models:
        return []

    sample_ids = _sample_text_ids(annotations_dir, key, prompt_name)
    human_ds = _human_sample_dataset(
        ds_human, annotations_dir, key, prompt_name
    )

    n_sources = len(models) + (1 if human_ds is not None else 0)
    records = []
    with tqdm(
        total=n_sources,
        desc=f"  {prompt_name}: sources",
        position=2,
        leave=False,
    ) as progress:
        if human_ds is not None:
            progress.set_postfix_str("Human")
            human_series = _human_inherent_polarization(
                key,
                prompt_name,
                human_ds,
                sample_ids,
                human_results_dir,
                apunim_output_dir,
                use_monte_carlo,
                subsample,
            )
            records.extend(
                _records_from_series(human_series, key, prompt_name, "Human")
            )
            progress.update(1)
        records.extend(
            _inherent_records_for_models(
                key,
                prompt_name,
                models,
                files,
                apunim_output_dir,
                progress=progress,
                subsample=subsample,
            )
        )
    return records


def _inherent_records_for_dataset(
    human_datasets: HumanDatasets,
    key: str,
    prompt_names: list[str],
    annotations_dir: Path,
    apunim_output_dir: Path,
    human_results_dir: Path,
    exclude_models: set[str],
    use_monte_carlo: bool,
    subsample: SubsampleSpec | None = None,
) -> list[dict]:
    ds_human = human_datasets[key]
    records = []
    for prompt_name in tqdm(
        prompt_names,
        desc=f" {key}: prompts",
        position=1,
        leave=False,
    ):
        records.extend(
            _inherent_records_for_prompt(
                key,
                ds_human,
                prompt_name,
                annotations_dir,
                apunim_output_dir,
                human_results_dir,
                exclude_models,
                use_monte_carlo,
                subsample,
            )
        )
    return records


def compute_inherent_polarization_comparison(
    human_datasets: HumanDatasets,
    annotations_dir: Path,
    apunim_output_dir: Path,
    human_results_dir: Path,
    use_monte_carlo: bool,
    dataset_keys: list[str] = DATASET_KEYS,
    prompt_names: list[str] = MAIN_PROMPT_NAMES,
    exclude_models: set[str] | None = None,
    subsample: SubsampleSpec | None = None,
) -> pd.DataFrame:
    """
    Long-format DataFrame with one row per comment giving its inherent
    polarization, for every (Dataset, Prompt, Source) restricted to the
    sample of comments the LLMs were actually run on. With `subsample`,
    comments are repeatedly subsampled down to `subsample.size`
    annotators first (use a separate `apunim_output_dir` for this so the
    cached values of the full-annotator run are not mixed up).
    """
    exclude_models = set(exclude_models or ())
    records = []
    available_keys = _available_dataset_keys(human_datasets, dataset_keys)
    for key in tqdm(
        available_keys,
        desc="Inherent polarization: datasets",
        position=0,
        leave=True,
    ):
        records.extend(
            _inherent_records_for_dataset(
                human_datasets,
                key,
                prompt_names,
                annotations_dir,
                apunim_output_dir,
                human_results_dir,
                exclude_models,
                use_monte_carlo,
                subsample,
            )
        )
    return pd.DataFrame(records)


def build_inherent_polarization_table(
    long_df: pd.DataFrame,
    prompt_names: list[str],
) -> pd.DataFrame:
    """Build the inherent-polarization table with raw mean and 2*std values."""
    table = (
        long_df.groupby(["Dataset", "Source", "Prompt"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    table["std"] = 2 * table["std"]  # store as 2*SD from here on

    mean_pivot = table.pivot(
        index=["Dataset", "Source"], columns="Prompt", values="mean"
    ).reindex(columns=prompt_names)
    std_pivot = table.pivot(
        index=["Dataset", "Source"], columns="Prompt", values="std"
    ).reindex(columns=prompt_names)

    # MultiIndex columns: (stat, prompt)
    mean_pivot.columns = pd.MultiIndex.from_tuples(
        [("mean", c) for c in mean_pivot.columns]
    )
    std_pivot.columns = pd.MultiIndex.from_tuples(
        [("std", c) for c in std_pivot.columns]
    )
    return pd.concat([mean_pivot, std_pivot], axis=1)


def _escape_underscores(index: pd.MultiIndex) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(
        [
            tuple(str(level).replace("_", r"\_") for level in row)
            for row in index
        ],
        names=index.names,
    )


DEFAULT_INHERENT_CAPTION = (
    "Mean inherent polarization (mean $\\pm$ 2 SD) per "
    "(dataset, prompt), for Human and each LLM."
)


def export_inherent_polarization_table(
    df: pd.DataFrame,
    output_path: Path,
    label: str,
    float_format: str,
    caption: str = DEFAULT_INHERENT_CAPTION,
) -> None:
    if df.empty:
        print(f"No inherent-polarization results; skipping {output_path}.")
        return

    df = df.copy()
    prompt_names = df["mean"].columns.tolist()

    # Build display DataFrame with formatted "mean ± 2SD" strings per prompt
    display = pd.DataFrame(index=df.index)
    for prompt in prompt_names:
        mean_col = trim_numeric_col_latex(
            df["mean"][prompt], float_format=float_format
        )
        std_col = trim_numeric_col_latex(df["std"][prompt], float_format=".3f")
        # trim_numeric_col_latex returns a Series of strings; combine them
        display[prompt.capitalize()] = [
            f"{m} $\\pm$ {s}" if m != "---" else "---"
            for m, s in zip(mean_col, std_col)
        ]

    display.index = _escape_underscores(display.index)
    col_count = len(display.columns)

    latex_str = display.to_latex(
        caption=caption,
        label=label,
        escape=False,
        position="t",
        index=True,
        multirow=True,
        column_format="ll" + "r" * col_count,
    )

    latex_str = center_table_latex(latex_str)
    latex_str = small_table_latex(latex_str)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_str)
    print(f"Table exported to {output_path.resolve()}")
