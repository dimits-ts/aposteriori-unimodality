from pathlib import Path

import pandas as pd

from ..lib import run_helper
from ..lib.preprocessing import Dataset
from ..lib.util import skip_if_exists
from .common import (
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


def _human_inherent_polarization(
    dataset_key: str,
    human_ds: Dataset,
    sample_ids: set,
    human_results_dir: Path,
) -> pd.Series:
    """
    Human inherent-polarization values, restricted to `sample_ids`.
    Reuses the precomputed <dataset>-inherent.csv when present; only
    recomputes when no cached file is found.
    """
    cached_path = human_results_dir / f"{dataset_key}-inherent.csv"
    if skip_if_exists(cached_path):
        series = pd.read_csv(cached_path, index_col="comment")[
            "inherent_polarization"
        ]
        return series[series.index.isin(sample_ids)].dropna()

    fn = (
        run_helper.compute_inherent_polarization_random
        if dataset_key.startswith("dices")
        else run_helper.compute_inherent_polarization_exhaustive
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
    LLM inherent-polarization values for a single (dataset, prompt, model).
    Reuses a precomputed CSV when present; otherwise computes exhaustively.
    """
    cached_path = (
        apunim_output_dir
        / f"{dataset_key}-{prompt_name}-{pseudo}-inherent.csv"
    )
    if skip_if_exists(cached_path):
        series = pd.read_csv(cached_path, index_col="comment")[
            "inherent_polarization"
        ]
        return series.dropna()

    df = load_llm_df(path)
    ds = LLMAnnotationDataset(df, dataset_key, pseudo, prompt_name)
    return run_helper.compute_inherent_polarization_exhaustive(ds).dropna()


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
) -> list[dict]:
    records = []
    for pseudo in models:
        series = _llm_inherent_polarization(
            dataset_key, prompt_name, pseudo, files[pseudo], apunim_output_dir
        )
        records.extend(
            _records_from_series(series, dataset_key, prompt_name, pseudo)
        )
    return records


def _inherent_records_for_prompt(
    key: str,
    ds_human: Dataset,
    prompt_name: str,
    annotations_dir: Path,
    apunim_output_dir: Path,
    human_results_dir: Path,
    exclude_models: set[str],
) -> list[dict]:
    files = find_annotation_files(annotations_dir, key, prompt_name)
    models = _order_models(set(files) - exclude_models)
    if not models:
        return []

    sample_ids = _sample_text_ids(annotations_dir, key, prompt_name)
    human_ds = _human_sample_dataset(
        ds_human, annotations_dir, key, prompt_name
    )

    records = []
    if human_ds is not None:
        human_series = _human_inherent_polarization(
            key, human_ds, sample_ids, human_results_dir
        )
        records.extend(
            _records_from_series(human_series, key, prompt_name, "Human")
        )
    records.extend(
        _inherent_records_for_models(
            key, prompt_name, models, files, apunim_output_dir
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
) -> list[dict]:
    ds_human = human_datasets[key]
    records = []
    for prompt_name in prompt_names:
        records.extend(
            _inherent_records_for_prompt(
                key,
                ds_human,
                prompt_name,
                annotations_dir,
                apunim_output_dir,
                human_results_dir,
                exclude_models,
            )
        )
    return records


def compute_inherent_polarization_comparison(
    human_datasets: HumanDatasets,
    annotations_dir: Path,
    apunim_output_dir: Path,
    human_results_dir: Path,
    dataset_keys: list[str] = DATASET_KEYS,
    prompt_names: list[str] = MAIN_PROMPT_NAMES,
    exclude_models: set[str] | None = None,
) -> pd.DataFrame:
    """
    Long-format DataFrame with one row per comment giving its inherent
    polarization, for every (Dataset, Prompt, Source) restricted to the
    sample of comments the LLMs were actually run on.
    """
    exclude_models = set(exclude_models or ())
    records = []
    for key in _available_dataset_keys(human_datasets, dataset_keys):
        records.extend(
            _inherent_records_for_dataset(
                human_datasets,
                key,
                prompt_names,
                annotations_dir,
                apunim_output_dir,
                human_results_dir,
                exclude_models,
            )
        )
    return pd.DataFrame(records)


def build_inherent_polarization_table(
    long_df: pd.DataFrame,
    prompt_names: list[str],
) -> pd.DataFrame:
    """Build the inherent-polarization table as mean ± 2 SD."""
    table = (
        long_df.groupby(["Dataset", "Source", "Prompt"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    table["formatted"] = table.apply(
        lambda row: f"{row['mean']:.3f} $\\pm$ {2 * row['std']:.3f}", axis=1
    )
    table = table.pivot(
        index=["Dataset", "Source"], columns="Prompt", values="formatted"
    )
    table = table.reindex(columns=prompt_names)
    return table.fillna("---")


def _escape_underscores(index: pd.MultiIndex) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(
        [
            tuple(str(level).replace("_", r"\_") for level in row)
            for row in index
        ],
        names=index.names,
    )


def export_inherent_polarization_table(
    df: pd.DataFrame,
    output_path: Path,
    label: str = "tab:inherent-polarization",
    longtable: bool = True,
) -> None:
    if df.empty:
        print(f"No inherent-polarization results; skipping {output_path}.")
        return

    df = df.copy()
    df.index = _escape_underscores(df.index)

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
        float_format="%.3f",
    )
    latex_str = latex_str.replace(
        r"\begin{table}[ht]", r"\begin{table}[ht]\centering\scriptsize"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_str)
    print(f"Table exported to {output_path.resolve()}")
