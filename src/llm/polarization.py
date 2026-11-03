"""
Inherent polarization (USD) of the human and the LLM annotations, on the
comments the LLMs were run on.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from ..lib import run_helper
from ..lib.preprocessing import Dataset
from ..lib.util import skip_if_exists
from .shared import (
    HUMAN_LABEL,
    HumanDatasets,
    LLMAnnotationDataset,
    available_dataset_keys,
    find_annotation_files,
    human_sample_dataset,
    load_llm_df,
    order_models,
    sample_text_ids,
)


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
    ds: Dataset, subsample: SubsampleSpec | None, **progress: Any
) -> pd.Series:
    """Inherent polarization of `ds` (exhaustive): on all annotators
    normally, on repeatedly subsampled annotators when `subsample` is
    given."""
    if subsample is None:
        return run_helper.compute_inherent_polarization_exhaustive(
            ds, **progress
        )
    return run_helper.compute_inherent_polarization_subsampled(
        ds,
        size=subsample.size,
        n_repeats=subsample.n_repeats,
        seed=subsample.seed,
        **progress,
    )


def _cached_series(
    cached_path: Path, compute_fn: Callable[[], pd.Series]
) -> pd.Series:
    """Read `cached_path` if it exists, else compute and write it."""
    if skip_if_exists(cached_path, quiet=True):
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
    sample_ids: set[object],
    human_results_dir: Path,
    apunim_output_dir: Path,
    subsample: SubsampleSpec | None = None,
) -> pd.Series:
    """
    Human inherent-polarization values, restricted to `sample_ids`.
    Normally reuses the precomputed <dataset>-inherent.csv when present
    (only recomputing when no cached file is found). With `subsample` the
    precomputed full-annotator file does not apply, so the values are
    computed on the subsampled data and cached in `apunim_output_dir`.
    """
    progress: dict[str, Any] = dict(
        show_progress=True,
        progress_position=COMMENT_BAR_POSITION,
        progress_desc=f"    {dataset_key} Human: comments",
    )
    if subsample is not None:
        return _cached_series(
            apunim_output_dir
            / f"{dataset_key}-{prompt_name}-Human-inherent.csv",
            lambda: _compute_inherent(human_ds, subsample, **progress),
        )

    cached_path = human_results_dir / f"{dataset_key}-inherent.csv"
    if skip_if_exists(cached_path, quiet=True):
        series = pd.read_csv(cached_path, index_col="comment")[
            "inherent_polarization"
        ]
        return series[series.index.isin(sample_ids)].dropna()

    return _compute_inherent(human_ds, None, **progress).dropna()


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
) -> list[dict[str, object]]:
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
) -> list[dict[str, object]]:
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
    subsample: SubsampleSpec | None = None,
) -> list[dict[str, object]]:
    files = find_annotation_files(annotations_dir, key, prompt_name)
    models = order_models(set(files) - exclude_models)
    if not models:
        return []

    sample_ids = sample_text_ids(annotations_dir, key, prompt_name)
    human_ds = human_sample_dataset(
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
            progress.set_postfix_str(HUMAN_LABEL)
            human_series = _human_inherent_polarization(
                key,
                prompt_name,
                human_ds,
                sample_ids,
                human_results_dir,
                apunim_output_dir,
                subsample,
            )
            records.extend(
                _records_from_series(
                    human_series, key, prompt_name, HUMAN_LABEL
                )
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
    subsample: SubsampleSpec | None = None,
) -> list[dict[str, object]]:
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
                subsample,
            )
        )
    return records


def compute_inherent_polarization_comparison(
    human_datasets: HumanDatasets,
    annotations_dir: Path,
    apunim_output_dir: Path,
    human_results_dir: Path,
    dataset_keys: Iterable[str],
    prompt_names: list[str],
    exclude_models: Iterable[str] | None = None,
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
    excluded = set(exclude_models or ())
    records = []
    available_keys = available_dataset_keys(human_datasets, dataset_keys)
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
                excluded,
                subsample,
            )
        )
    return pd.DataFrame(records)
