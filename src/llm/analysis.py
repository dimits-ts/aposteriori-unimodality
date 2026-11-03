"""
Compares the human annotations with the LLM annotations of
llm_annotate.py / annotate_experiments.sh.

Figures (graph_output_dir):
  human_vs_llm_histograms.png
  llm_prompt_mean_diff.png
  inherent-polarization-subsampled-n6-histogram.png
      Cache: inherent-subsampled-n6/<dataset>-<prompt>-<source>-inherent.csv
  inherent-polarization-by-model[_excluding].png
  inherent-polarization-by-dataset[_excluding].png
      Cache: inherent/<dataset>-<prompt>-<model>-inherent.csv and the
      subsampled cache above; the human values of the full run are read
      from human_results_dir/<dataset>-inherent.csv.

LaTeX tables (latex_output_dir):
  nonstandard_answers_summary.tex
  llm-consistency-cross-model.tex
  llm-consistency-cross-model-excluding.tex
  llm-consistency-variants.tex
  llm-consistency-repeats.tex
  llm-apunim-by-prompt-<dataset>.tex
      Cache: apunim_<dataset>.csv
  llm-apunim-significance[_excluding].tex
  llm-apunim-significance-counts[_excluding].tex
  llm-apunim-significance-counts-by-prompt[_excluding].tex
      Cache: apunim_<dataset>.csv
  mean_ndfu_by_prompt[_excluding].tex
  mean_ndfu_default_vs_human[_excluding].tex
      Cache: mean_ndfu_records_all_models.csv,
      mean_ndfu_records_no_persona.csv
  human_llm_annotation_spearman.tex
      Cache: mean_annotation_records_inherent_datasets_all_models.csv,
      mean_annotation_records_inherent_datasets_no_persona.csv
  cohens_d[_excluding].tex
      Cache: polarization_by_instruction_anova_by_model_dataset.csv
"""

import argparse
import logging
from collections.abc import Collection, Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from ..lib import graphs
from ..lib.preprocessing import (
    DicesDataset,
    KumarDataset,
    LazyDatasetLoader,
    PopquornDataset,
    SapDataset,
)
from ..lib.util import skip_if_exists
from . import plots, polarization, stats, table_export
from .shared import (
    NO_PERSONA_PROMPT_NAME,
    HumanDatasets,
    available_dataset_keys,
    find_annotation_files,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------

DATASET_KEYS = ["dices-350", "dices-990", "sap", "kumar", "popquorn"]

# Datasets of the main annotation runs (default + adversarial prompts).
MAIN_DATASET_KEYS = DATASET_KEYS

# Datasets of the ablations (paraphrase variants, repeated runs).
ABLATION_DATASET_KEYS = DATASET_KEYS

# Number of Kumar et al. comments loaded (as in llm_annotate.py).
KUMAR_NUM_SAMPLES = 1_000

# The instruction prompts compared throughout the analysis. The baseline is
# the prompt the others are compared against.
MAIN_PROMPT_NAMES = ["default", "stereotype", "persona", "single", "direct"]
BASELINE_PROMPT_NAME = "default"

# Prompt variants of the paraphrase ablation
# (instructions/ablation/<dataset>/variant{1,2,3}.txt).
PARAPHRASE_VARIANT_NAMES = ["variant1", "variant2", "variant3"]

# Models left out of the "excluding" variants of the tables/plots.
EXCLUDED_MODELS = ["llama8b", "qwen7b", "olmo7b"]

# Datasets left out of the significance (symbol) table: no human result of
# kumar can be significant. They are kept in the significance counts.
SIGNIFICANCE_TABLE_SKIP_KEYS = {"kumar"}

# Inherent-polarization subsampling ablation: every comment (human and LLM
# alike) is repeatedly subsampled down to this many annotators and the
# results averaged over this many repeats.
INHERENT_SUBSAMPLE_SIZE = 6
INHERENT_SUBSAMPLE_REPEATS = 10

# Grid of the subsampled inherent-polarization histogram figure: datasets
# are the rows, prompts the columns. Also the datasets of the annotation
# histograms.
INHERENT_HIST_DATASET_KEYS = ["sap", "kumar", "popquorn"]
INHERENT_HIST_PROMPT_NAMES = MAIN_PROMPT_NAMES

# Prompt of the per-model (values pooled over all datasets) and per-dataset
# (values pooled over all models) mirrored inherent-polarization histograms.
INHERENT_BY_MODEL_PROMPT_NAME = BASELINE_PROMPT_NAME

# Datasets (no DICES) and prompts, in column order, of the human-LLM
# annotation correlation table: the no-persona baseline first, then the
# persona runs.
CORRELATION_DATASET_KEYS = INHERENT_HIST_DATASET_KEYS
CORRELATION_PROMPT_NAMES = [NO_PERSONA_PROMPT_NAME, *MAIN_PROMPT_NAMES]


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
    exclude_models: Collection[str],
    prompt_name: str = BASELINE_PROMPT_NAME,
    n_workers: int = 1,
) -> None:
    graphs.graph_setup()
    graph_output_dir.mkdir(parents=True, exist_ok=True)
    latex_output_dir.mkdir(parents=True, exist_ok=True)

    run_nonstandard_answers_step(annotations_dir, cache_dir, latex_output_dir)

    human_datasets = build_human_datasets(
        dices_small_path, dices_large_path, sap_path, kumar_path, popquorn_path
    )

    # apunim results for every comparison dataset, computed (or loaded from
    # the cache) once and shared by the apunim table steps below.
    apunim_results = load_apunim_results(
        human_datasets, annotations_dir, cache_dir, n_workers=n_workers
    )

    run_histogram_step(
        human_datasets, annotations_dir, graph_output_dir, prompt_name
    )
    run_prompt_diff_step(
        human_datasets, annotations_dir, graph_output_dir, exclude_models
    )
    run_cross_model_consistency_step(
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
    ndfu_df, no_persona_df = load_mean_ndfu_records(
        human_datasets, annotations_dir, cache_dir
    )
    run_mean_ndfu_step(
        ndfu_df, no_persona_df, latex_output_dir, exclude_models
    )
    run_annotation_correlation_step(
        human_datasets, annotations_dir, cache_dir, latex_output_dir
    )
    run_inherent_polarization_steps(
        human_datasets,
        annotations_dir,
        exclude_models,
        human_results_dir=human_results_dir,
        cache_dir=cache_dir,
        graph_output_dir=graph_output_dir,
    )
    anova_df = load_ndfu_anova(human_datasets, annotations_dir, cache_dir)
    run_cohens_d_step(anova_df, latex_output_dir, exclude_models)


def build_human_datasets(
    dices_small_path: Path,
    dices_large_path: Path,
    sap_path: Path,
    kumar_path: Path,
    popquorn_path: Path,
) -> HumanDatasets:
    """The human datasets, each loaded on first use."""
    return HumanDatasets(
        {
            "dices-350": LazyDatasetLoader(
                lambda: DicesDataset(
                    dataset_path=dices_small_path, variant="350"
                )
            ),
            "dices-990": LazyDatasetLoader(
                lambda: DicesDataset(
                    dataset_path=dices_large_path, variant="990"
                )
            ),
            "sap": LazyDatasetLoader(
                lambda: SapDataset(dataset_path=sap_path)
            ),
            "kumar": LazyDatasetLoader(
                lambda: KumarDataset(
                    dataset_path=kumar_path, num_samples=KUMAR_NUM_SAMPLES
                )
            ),
            "popquorn": LazyDatasetLoader(
                lambda: PopquornDataset(dataset_path=popquorn_path)
            ),
        }
    )


# ---------------------------------------------------------------------------
# Model variants of the per-model figures and tables
# ---------------------------------------------------------------------------


class ModelVariant(NamedTuple):
    """One version of a per-model figure or table."""

    file_suffix: str  # appended to the output file name
    label_suffix: str  # appended to the LaTeX \label{}
    dropped_models: frozenset[str]  # models left out of this version

    @property
    def all_models(self) -> bool:
        return not self.dropped_models


def _model_variants(exclude_models: Iterable[str]) -> list[ModelVariant]:
    """
    The versions to export of every per-model figure or table: one with
    all models and, if `exclude_models` is not empty, a second one without
    them ("_excluding" file, "-excluding" label).
    """
    variants = [ModelVariant("", "", frozenset())]
    excluded = frozenset(exclude_models)
    if excluded:
        variants.append(ModelVariant("_excluding", "-excluding", excluded))
    return variants


def _drop_models(df: pd.DataFrame, models: Collection[str]) -> pd.DataFrame:
    """Rows of a long/results frame whose `Model` is not in `models`. A
    frame without a `Model` column (no results at all) is returned as is."""
    if not models or "Model" not in df.columns:
        return df
    return df.loc[~df["Model"].isin(models)]


# ---------------------------------------------------------------------------
# Non-standard answers
# ---------------------------------------------------------------------------


def run_nonstandard_answers_step(
    annotations_dir: Path, csv_output_dir: Path, latex_output_dir: Path
) -> None:
    detailed, totals = stats.collect_nonstandard_counts(
        annotations_dir, MAIN_DATASET_KEYS, MAIN_PROMPT_NAMES
    )
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    stats.nonstandard_by_model(detailed, totals).to_csv(
        csv_output_dir / "nonstandard_answers_by_model.csv", index=False
    )

    summary = stats.nonstandard_summary(detailed, totals)
    if summary.empty:
        print("No non-standard answers found; skipping summary table.")
        return
    table_export.export_nonstandard_summary(
        summary,
        latex_output_dir / "nonstandard_answers_summary.tex",
        label="tab:nonstandard-answers",
    )


# ---------------------------------------------------------------------------
# Histograms, prompt diffs
# ---------------------------------------------------------------------------


def run_histogram_step(
    human_datasets: HumanDatasets,
    annotations_dir: Path,
    graph_output_dir: Path,
    prompt_name: str,
) -> None:
    plots.plot_annotation_histograms(
        human_datasets=human_datasets,
        annotations_dir=annotations_dir,
        output_path=graph_output_dir / "human_vs_llm_histograms.png",
        prompt_name=prompt_name,
        dataset_keys=INHERENT_HIST_DATASET_KEYS,
    )


def run_prompt_diff_step(
    human_datasets: HumanDatasets,
    annotations_dir: Path,
    graph_output_dir: Path,
    exclude_models: Collection[str],
) -> None:
    plots.plot_prompt_mean_diff(
        human_datasets=human_datasets,
        annotations_dir=annotations_dir,
        output_path=graph_output_dir / "llm_prompt_mean_diff.png",
        prompt_names=MAIN_PROMPT_NAMES,
        baseline_prompt=BASELINE_PROMPT_NAME,
        exclude_models=exclude_models,
        dataset_keys=MAIN_DATASET_KEYS,
    )


# ---------------------------------------------------------------------------
# Consistency tables
# ---------------------------------------------------------------------------


def run_cross_model_consistency_step(
    human_datasets: HumanDatasets,
    annotations_dir: Path,
    latex_output_dir: Path,
    prompt_name: str,
    exclude_models: Collection[str],
) -> None:
    """With every model, and (if `exclude_models` is not empty) without
    `exclude_models`."""
    versions: list[tuple[str, Collection[str]]] = [("", ())]
    if exclude_models:
        versions.append(("-excluding", exclude_models))
    for suffix, excluded in versions:
        table_export.export_cross_model_consistency(
            stats.cross_model_consistency_table(
                human_datasets=human_datasets,
                annotations_dir=annotations_dir,
                prompt_name=prompt_name,
                dataset_keys=MAIN_DATASET_KEYS,
                exclude_models=excluded,
            ),
            output_path=(
                latex_output_dir / f"llm-consistency-cross-model{suffix}.tex"
            ),
            label=f"tab:llm-consistency-cross-model{suffix}",
            prompt_name=prompt_name,
            excluded_models=excluded,
        )


def run_variant_consistency_step(
    human_datasets: HumanDatasets,
    paraphrase_dir: Path,
    latex_output_dir: Path,
) -> None:
    table_export.export_variant_consistency(
        stats.per_model_variant_consistency_table(
            human_datasets=human_datasets,
            paraphrase_dir=paraphrase_dir,
            variant_names=PARAPHRASE_VARIANT_NAMES,
            dataset_keys=ABLATION_DATASET_KEYS,
        ),
        output_path=latex_output_dir / "llm-consistency-variants.tex",
        label="tab:llm-consistency-variants",
    )


def run_repeat_consistency_step(
    human_datasets: HumanDatasets,
    repeat_dir: Path,
    latex_output_dir: Path,
    prompt_name: str,
) -> None:
    table_export.export_repeat_consistency(
        stats.per_model_repeat_consistency_table(
            human_datasets=human_datasets,
            repeat_dir=repeat_dir,
            prompt_name=prompt_name,
            dataset_keys=ABLATION_DATASET_KEYS,
        ),
        output_path=latex_output_dir / "llm-consistency-repeats.tex",
        label="tab:llm-consistency-repeats",
        prompt_name=prompt_name,
    )


# ---------------------------------------------------------------------------
# apunim cache
# ---------------------------------------------------------------------------


class ApunimPair(NamedTuple):
    """One (dataset, prompt, model) apunim run; `all_models` are every
    model with annotations for that (dataset, prompt)."""

    key: str
    prompt_name: str
    model: str
    all_models: frozenset[str]


def _read_apunim_cache(cache_path: Path) -> pd.DataFrame:
    # Without a cache, start from an empty frame that still has the result
    # columns: a dataset none of whose pairs produce results stays empty,
    # and the steps downstream select columns such as "Model".
    if not cache_path.exists():
        return pd.DataFrame(columns=stats.APUNIM_COLUMNS)
    long_df = pd.read_csv(cache_path)
    # Older caches were written with the DataFrame index as a column.
    return long_df.loc[:, ~long_df.columns.str.startswith("Unnamed")]


def _cached_apunim_pairs(long_df: pd.DataFrame) -> set[tuple[str, str]]:
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
    annotations_dir: Path, key: str, cached_pairs: set[tuple[str, str]]
) -> list[ApunimPair]:
    """Every (model, prompt) pair of `key` that has an annotation file but
    is not cached yet."""
    pending: list[ApunimPair] = []
    for prompt_name in MAIN_PROMPT_NAMES:
        files = find_annotation_files(annotations_dir, key, prompt_name)
        pending.extend(
            ApunimPair(key, prompt_name, model, frozenset(files))
            for model in sorted(files)
            if (model, prompt_name) not in cached_pairs
        )
    return pending


def _apunim_pair_kwargs(
    annotations_dir: Path, pair: ApunimPair
) -> dict[str, object]:
    """Arguments of stats.compute_llm_apunim_by_prompt for one (model,
    prompt) pair: every other model of the prompt is excluded."""
    return dict(
        annotations_dir=annotations_dir,
        dataset_key=pair.key,
        prompt_names=[pair.prompt_name],
        exclude_models=pair.all_models - {pair.model},
    )


def _pair_name(pair: ApunimPair) -> str:
    return f"{pair.key} / {pair.prompt_name} / {pair.model}"


def _iter_apunim_pairs_serial(
    annotations_dir: Path, pending: list[ApunimPair]
) -> Iterator[tuple[str, pd.DataFrame]]:
    for pair in pending:
        print(f"Computing apunim: {_pair_name(pair)}")
        yield (
            pair.key,
            stats.compute_llm_apunim_by_prompt(
                **_apunim_pair_kwargs(annotations_dir, pair)  # type: ignore[arg-type]
            ),
        )


def _iter_apunim_pairs_parallel(
    annotations_dir: Path, pending: list[ApunimPair], n_workers: int
) -> Iterator[tuple[str, pd.DataFrame]]:
    """
    Runs the pairs in a pool of `n_workers` processes and yields
    (dataset key, results) in order of completion. The worker is
    stats.compute_llm_apunim_by_prompt itself (a function of an importable
    module), so it pickles under every start method, including spawn and
    forkserver. If a pair fails, the pairs not started yet are cancelled,
    the ones already running are still yielded (and so cached) as they
    finish, and the first error is then re-raised.
    """
    executor = ProcessPoolExecutor(max_workers=n_workers)
    try:
        futures: dict[Future[pd.DataFrame], ApunimPair] = {}
        for pair in pending:
            print(f"Queued apunim: {_pair_name(pair)}")
            future = executor.submit(
                stats.compute_llm_apunim_by_prompt,
                **_apunim_pair_kwargs(annotations_dir, pair),  # type: ignore[arg-type]
            )
            futures[future] = pair

        error: BaseException | None = None
        for future in as_completed(futures):
            if future.cancelled():
                continue
            pair = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"Failed apunim: {_pair_name(pair)}")
                if error is None:
                    error = exc
                    for other in futures:
                        other.cancel()
                continue
            print(f"Finished apunim: {_pair_name(pair)}")
            yield pair.key, result
        if error is not None:
            raise error
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _iter_apunim_pairs(
    annotations_dir: Path, pending: list[ApunimPair], n_workers: int
) -> Iterator[tuple[str, pd.DataFrame]]:
    """(dataset key, results) of every pending pair; in a process pool if
    n_workers > 1."""
    n_workers = min(n_workers, len(pending))
    if n_workers <= 1:
        return _iter_apunim_pairs_serial(annotations_dir, pending)
    print(f"Computing {len(pending)} apunim pairs with {n_workers} processes.")
    return _iter_apunim_pairs_parallel(annotations_dir, pending, n_workers)


def _apunim_cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"apunim_{key}.csv"


def load_apunim_results(
    human_datasets: HumanDatasets,
    annotations_dir: Path,
    cache_dir: Path,
    n_workers: int = 1,
) -> dict[str, pd.DataFrame]:
    """
    dataset key -> apunim results (every model and prompt), for every
    MAIN_DATASET_KEYS dataset, cached in cache_dir/apunim_<key>.csv.
    Computed once and shared by the apunim-by-prompt tables and the
    significant-PC-dimension tables.

    The cache is checked per (model, prompt) pair -- the unit apunim is run
    on -- so annotation runs that are still in progress are handled: pairs
    already in the cache are never recomputed, pairs whose annotation file
    appears later are added on the next run, and the cache is rewritten
    after every pair, so an interrupted run keeps what it finished.

    The pending pairs of all datasets are computed together, in a pool of
    `n_workers` processes when n_workers > 1. Results are written to the
    cache by this (parent) process only, as each pair finishes.
    """
    keys = available_dataset_keys(human_datasets, MAIN_DATASET_KEYS)
    results = {
        key: _read_apunim_cache(_apunim_cache_path(cache_dir, key))
        for key in keys
    }
    n_rows_before = {key: len(df) for key, df in results.items()}

    pending = [
        pair
        for key in keys
        for pair in _pending_apunim_pairs(
            annotations_dir, key, _cached_apunim_pairs(results[key])
        )
    ]
    for key, new_df in _iter_apunim_pairs(annotations_dir, pending, n_workers):
        results[key] = _update_apunim_cache(
            results[key], new_df, _apunim_cache_path(cache_dir, key)
        )

    for key in keys:
        if results[key].empty:
            print(f"Warning: no apunim results for {key}.")
        elif len(results[key]) == n_rows_before[key]:
            print(
                "Skipping (nothing new to compute): "
                f"{_apunim_cache_path(cache_dir, key)}"
            )
    return results


# ---------------------------------------------------------------------------
# apunim tables
# ---------------------------------------------------------------------------


def run_apunim_prompt_table_step(
    human_datasets: HumanDatasets,
    apunim_results: dict[str, pd.DataFrame],
    latex_output_dir: Path,
) -> None:
    for key, long_df in apunim_results.items():
        table_export.export_apunim_prompt_table(
            long_df,
            output_path=latex_output_dir / f"llm-apunim-by-prompt-{key}.tex",
            label=f"tab:llm-apunim-by-prompt-{key}",
            dataset_name=human_datasets[key].get_name(),
            prompt_names=MAIN_PROMPT_NAMES,
        )


def _collect_apunim_significance_inputs(
    apunim_results: dict[str, pd.DataFrame], human_results_dir: Path
) -> stats.ApunimInputs:
    """dataset key -> (human apunim results, LLM apunim results), for every
    dataset that has human results."""
    inputs = {}
    for key, llm_df in apunim_results.items():
        human_df = stats.load_human_apunim_results(human_results_dir, key)
        if human_df is not None:
            inputs[key] = (human_df, llm_df)
    return inputs


def _significance_inputs_by_name(
    human_datasets: HumanDatasets,
    inputs: stats.ApunimInputs,
    models: Collection[str],
) -> stats.ApunimInputs:
    """`inputs` keyed by dataset display name and without the given models;
    datasets left without LLM results are dropped."""
    by_name = {}
    for key, (human_df, llm_df) in inputs.items():
        variant_df = _drop_models(llm_df, models)
        if not variant_df.empty:
            by_name[human_datasets[key].get_name()] = (human_df, variant_df)
    return by_name


def run_apunim_significance_step(
    human_datasets: HumanDatasets,
    apunim_results: dict[str, pd.DataFrame],
    latex_output_dir: Path,
    human_results_dir: Path,
    exclude_models: Collection[str],
) -> None:
    """
    Which PC dimensions (SDB features, e.g. age) each model finds
    statistically significant under each prompt, compared with the human
    results: one symbol table over all datasets (except
    SIGNIFICANCE_TABLE_SKIP_KEYS) and the shares of significant dimensions
    per dataset (for the baseline prompt, and per prompt).
    """
    inputs = _collect_apunim_significance_inputs(
        apunim_results, human_results_dir
    )
    table_inputs = {
        key: value
        for key, value in inputs.items()
        if key not in SIGNIFICANCE_TABLE_SKIP_KEYS
    }
    for variant in _model_variants(exclude_models):
        file_suffix, label_suffix, models = variant

        counts = stats.count_significant_dimensions(
            _significance_inputs_by_name(human_datasets, inputs, models),
            prompt_names=MAIN_PROMPT_NAMES,
        )
        table_export.export_significance_counts_summary(
            counts,
            output_path=latex_output_dir
            / f"llm-apunim-significance-counts{file_suffix}.tex",
            label=f"tab:llm-apunim-significance-counts{label_suffix}",
            prompt_name=BASELINE_PROMPT_NAME,
            all_models=variant.all_models,
        )
        table_export.export_significance_counts_by_prompt(
            counts,
            output_path=latex_output_dir
            / f"llm-apunim-significance-counts-by-prompt{file_suffix}.tex",
            label=f"tab:llm-apunim-significance-counts-by-prompt{label_suffix}",
            prompt_names=MAIN_PROMPT_NAMES,
            all_models=variant.all_models,
        )

        by_name = _significance_inputs_by_name(
            human_datasets, table_inputs, models
        )
        if not by_name:
            continue
        table_export.export_apunim_significance_table(
            stats.compute_apunim_significance(
                by_name, prompt_names=MAIN_PROMPT_NAMES
            ),
            output_path=latex_output_dir
            / f"llm-apunim-significance{file_suffix}.tex",
            label=f"tab:llm-apunim-significance{label_suffix}",
            all_models=variant.all_models,
        )


# ---------------------------------------------------------------------------
# Per-comment records: mean nDFU, human-LLM correlation
# ---------------------------------------------------------------------------


def _missing_no_persona_annotations(
    annotations_dir: Path, dataset_keys: Iterable[str]
) -> list[Path]:
    """
    The expected no-persona annotation files ("<dataset>-nopersona-<model>
    .csv") that do not exist, for every model that has baseline-prompt
    annotations for a dataset. Each missing file (or a missing annotations
    directory) is logged as a warning; the matching table cells are then
    left empty.
    """
    if not annotations_dir.is_dir():
        logger.warning("Annotations directory not found: %s", annotations_dir)
        return [annotations_dir]
    missing = []
    for dataset_key in dataset_keys:
        baseline_files = find_annotation_files(
            annotations_dir, dataset_key, BASELINE_PROMPT_NAME
        )
        for model in baseline_files:
            path = (
                annotations_dir
                / f"{dataset_key}-{NO_PERSONA_PROMPT_NAME}-{model}.csv"
            )
            if not path.is_file():
                logger.warning("Missing no-persona annotations: %s", path)
                missing.append(path)
    return missing


def _read_records(path: Path) -> pd.DataFrame:
    """Cached per-comment records, with text_id always read as a string (it
    mixes numeric and non-numeric ids across datasets)."""
    return pd.read_csv(path, dtype={"text_id": str})


def _load_per_comment_records(
    human_datasets: HumanDatasets,
    annotations_dir: Path,
    cache_dir: Path,
    compute_records: stats.RecordsFunction,
    cache_stem: str,
    dataset_keys: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    (records of the human annotations and of every model under every main
    prompt, the same for the no-persona runs; LLM rows only), computed by
    `compute_records` (stats.compute_mean_ndfu_records or
    stats.compute_mean_annotation_records) for `dataset_keys`, and cached
    in cache_dir as <cache_stem>_all_models.csv and
    <cache_stem>_no_persona.csv. The cache is not keyed by the datasets,
    so a different set of datasets needs a different `cache_stem`.
    """
    # Computed for every model; the "excluding" variants are filtered from
    # these records.
    cache_path = cache_dir / f"{cache_stem}_all_models.csv"
    if skip_if_exists(cache_path):
        records_df = _read_records(cache_path)
    else:
        records_df = compute_records(
            human_datasets=human_datasets,
            annotations_dir=annotations_dir,
            dataset_keys=dataset_keys,
            prompt_names=MAIN_PROMPT_NAMES,
            sample_prompt=BASELINE_PROMPT_NAME,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        records_df.to_csv(cache_path, index=False)

    # The no-persona baseline runs (LLM rows only), cached separately so
    # the records above need not be recomputed. Every model run on the
    # baseline prompt should have a no-persona run too; missing files are
    # logged and their cells left empty. Results are only cached once no
    # file is missing, so a cache written before all files existed is
    # never reused.
    missing = _missing_no_persona_annotations(
        annotations_dir, available_dataset_keys(human_datasets, dataset_keys)
    )
    no_persona_cache_path = cache_dir / f"{cache_stem}_no_persona.csv"
    if not missing and skip_if_exists(no_persona_cache_path):
        no_persona_df = _read_records(no_persona_cache_path)
    else:
        no_persona_df = compute_records(
            human_datasets=human_datasets,
            annotations_dir=annotations_dir,
            dataset_keys=dataset_keys,
            prompt_names=[NO_PERSONA_PROMPT_NAME],
            sample_prompt=BASELINE_PROMPT_NAME,
            include_human=False,
        )
        if not missing:
            no_persona_cache_path.parent.mkdir(parents=True, exist_ok=True)
            no_persona_df.to_csv(no_persona_cache_path, index=False)

    # Comments are matched across these frames by text_id, so its type must
    # agree: a cached CSV gives strings where freshly computed records may
    # give integers, and "123" != 123 would silently drop the rows.
    records_df["text_id"] = records_df["text_id"].astype(str)
    no_persona_df["text_id"] = no_persona_df["text_id"].astype(str)
    return records_df, no_persona_df


def load_mean_ndfu_records(
    human_datasets: HumanDatasets, annotations_dir: Path, cache_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-comment nDFU records (main prompts, no-persona runs), for the
    mean nDFU tables. (Cache name: the old mean_ndfu_records.csv was
    written without olmo7b/llama8b and would be silently reused.)"""
    return _load_per_comment_records(
        human_datasets,
        annotations_dir,
        cache_dir,
        stats.compute_mean_ndfu_records,
        cache_stem="mean_ndfu_records",
        dataset_keys=MAIN_DATASET_KEYS,
    )


def load_mean_annotation_records(
    human_datasets: HumanDatasets, annotations_dir: Path, cache_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-comment mean annotation records (main prompts, no-persona
    runs) of the CORRELATION_DATASET_KEYS datasets, for the human-LLM
    correlation table."""
    return _load_per_comment_records(
        human_datasets,
        annotations_dir,
        cache_dir,
        stats.compute_mean_annotation_records,
        # Named after the datasets: an earlier cache with all datasets
        # ("mean_annotation_records_*") must not be reused.
        cache_stem="mean_annotation_records_inherent_datasets",
        dataset_keys=CORRELATION_DATASET_KEYS,
    )


def run_mean_ndfu_step(
    ndfu_df: pd.DataFrame,
    no_persona_df: pd.DataFrame,
    latex_output_dir: Path,
    exclude_models: Collection[str],
) -> None:
    if ndfu_df.empty:
        print("No matched human/LLM annotations; skipping mean nDFU tables.")
        return

    for file_suffix, label_suffix, models in _model_variants(exclude_models):
        variant_df = _drop_models(ndfu_df, models)
        table_export.export_mean_ndfu_by_prompt(
            stats.compute_mean_ndfu_table(
                variant_df, prompt_names=MAIN_PROMPT_NAMES
            ),
            output_path=latex_output_dir
            / f"mean_ndfu_by_prompt{file_suffix}.tex",
            label=f"tab:mean-ndfu-by-prompt{label_suffix}",
        )
        table_export.export_mean_ndfu_default_table(
            stats.compute_default_mean_ndfu_table(
                pd.concat(
                    [variant_df, _drop_models(no_persona_df, models)],
                    ignore_index=True,
                ),
                prompt_name=BASELINE_PROMPT_NAME,
                no_persona_prompt_name=NO_PERSONA_PROMPT_NAME,
            ),
            output_path=latex_output_dir
            / f"mean_ndfu_default_vs_human{file_suffix}.tex",
            label=f"tab:mean-ndfu-default-vs-human{label_suffix}",
        )


def run_annotation_correlation_step(
    human_datasets: HumanDatasets,
    annotations_dir: Path,
    cache_dir: Path,
    latex_output_dir: Path,
) -> None:
    """
    Spearman correlation, across the comments of each
    CORRELATION_DATASET_KEYS dataset, between the mean human annotation
    and each model's mean annotation of a comment, for every model under
    the no-persona baseline and every main prompt
    (human_llm_annotation_spearman.tex; see
    stats.compute_human_llm_correlations). The Holm correction is applied
    over all of the table's cells.
    """
    records_df, no_persona_df = load_mean_annotation_records(
        human_datasets, annotations_dir, cache_dir
    )
    if records_df.empty:
        print("No matched human/LLM annotations; skipping correlation table.")
        return
    corr_df = stats.compute_human_llm_correlations(
        pd.concat([records_df, no_persona_df], ignore_index=True),
        prompt_names=CORRELATION_PROMPT_NAMES,
        value_col="annotation",
        correction_method="holm",
    )
    table_export.export_human_llm_correlation_table(
        corr_df,
        output_path=latex_output_dir / "human_llm_annotation_spearman.tex",
        label="tab:human-llm-annotation-spearman",
        prompt_names=CORRELATION_PROMPT_NAMES,
    )


# ---------------------------------------------------------------------------
# Inherent polarization
# ---------------------------------------------------------------------------


def run_inherent_polarization_step(
    human_datasets: HumanDatasets,
    annotations_dir: Path,
    exclude_models: Collection[str],
    human_results_dir: Path,
    cache_dir: Path,
    graph_output_dir: Path,
    subsample: polarization.SubsampleSpec | None = None,
) -> pd.DataFrame:
    """
    Per-comment inherent polarization of Human and of *every* model (the
    long frame of polarization.compute_inherent_polarization_comparison),
    returned so the per-model figures can choose which models to show.
    For the annotator-subsampled variant the histogram grid is exported
    too, without the models in `exclude_models`.
    """
    # human_results_dir holds the human scripts' own "<dataset>-inherent.csv"
    # outputs (e.g. output/human/main), which are reused as-is for the main
    # run. The per-(dataset, prompt, model) values are cached separately
    # under cache_dir/inherent (the subsampling ablation uses its own
    # folder, since its values differ). Delete the folder to force a
    # recompute. Values are cached per model, so computing all models here
    # and filtering afterwards costs nothing extra on later runs.
    suffix = "" if subsample is None else f"-{subsample.tag}"

    inherent_df = polarization.compute_inherent_polarization_comparison(
        human_datasets=human_datasets,
        annotations_dir=annotations_dir,
        apunim_output_dir=cache_dir / f"inherent{suffix}",
        human_results_dir=human_results_dir,
        dataset_keys=MAIN_DATASET_KEYS,
        prompt_names=MAIN_PROMPT_NAMES,
        subsample=subsample,
    )

    # Normalized histogram grid of the same per-comment values (datasets as
    # rows, prompts as columns). Only produced for the annotator-subsampled
    # variant.
    if subsample is not None and not inherent_df.empty:
        plots.plot_inherent_polarization_histogram_grid(
            long_df=inherent_df.loc[
                ~inherent_df["Source"].isin(set(exclude_models))
            ],
            output_path=(
                graph_output_dir
                / f"inherent-polarization{suffix}-histogram.png"
            ),
            dataset_keys=INHERENT_HIST_DATASET_KEYS,
            prompt_names=INHERENT_HIST_PROMPT_NAMES,
            common_norm=False,
        )

    return inherent_df


def run_inherent_polarization_by_model_step(
    full_df: pd.DataFrame,
    subsampled_df: pd.DataFrame,
    graph_output_dir: Path,
    exclude_models: Collection[str],
    subsample_size: int = INHERENT_SUBSAMPLE_SIZE,
) -> None:
    """
    Mirrored USD histograms per model (all personas vs. at most
    `subsample_size` personas per comment), in the style of the human
    apriori.png: inherent-polarization-by-model.png with every model, and
    inherent-polarization-by-model_excluding.png without `exclude_models`.
    """
    for file_suffix, _, dropped in _model_variants(exclude_models):
        plots.plot_inherent_polarization_by_model(
            full_df=full_df,
            subsampled_df=subsampled_df,
            output_path=(
                graph_output_dir
                / f"inherent-polarization-by-model{file_suffix}.png"
            ),
            subsample_size=subsample_size,
            exclude_models=dropped,
            prompt_name=INHERENT_BY_MODEL_PROMPT_NAME,
        )


def run_inherent_polarization_by_dataset_step(
    full_df: pd.DataFrame,
    subsampled_df: pd.DataFrame,
    graph_output_dir: Path,
    exclude_models: Collection[str],
    subsample_size: int = INHERENT_SUBSAMPLE_SIZE,
) -> None:
    """
    Mirrored USD histograms per dataset, models pooled (all personas vs.
    at most `subsample_size` personas per comment), in the style of the
    human apriori.png: inherent-polarization-by-dataset.png pooling every
    model, and inherent-polarization-by-dataset_excluding.png without
    `exclude_models`.
    """
    for file_suffix, _, dropped in _model_variants(exclude_models):
        plots.plot_inherent_polarization_by_dataset(
            full_df=full_df,
            subsampled_df=subsampled_df,
            output_path=(
                graph_output_dir
                / f"inherent-polarization-by-dataset{file_suffix}.png"
            ),
            subsample_size=subsample_size,
            exclude_models=dropped,
            prompt_name=INHERENT_BY_MODEL_PROMPT_NAME,
            dataset_keys=MAIN_DATASET_KEYS,
        )


def run_inherent_polarization_steps(
    human_datasets: HumanDatasets,
    annotations_dir: Path,
    exclude_models: Collection[str],
    human_results_dir: Path,
    cache_dir: Path,
    graph_output_dir: Path,
) -> None:
    """
    Inherent polarization with every comment subsampled down to
    INHERENT_SUBSAMPLE_SIZE annotators (histogram grid) and with all
    annotators; both feed the per-model and per-dataset mirrored
    histograms.
    """
    subsampled_df = run_inherent_polarization_step(
        human_datasets,
        annotations_dir,
        exclude_models,
        human_results_dir=human_results_dir,
        cache_dir=cache_dir,
        graph_output_dir=graph_output_dir,
        subsample=polarization.SubsampleSpec(
            size=INHERENT_SUBSAMPLE_SIZE,
            n_repeats=INHERENT_SUBSAMPLE_REPEATS,
        ),
    )
    full_df = run_inherent_polarization_step(
        human_datasets,
        annotations_dir,
        exclude_models,
        human_results_dir=human_results_dir,
        cache_dir=cache_dir,
        graph_output_dir=graph_output_dir,
    )
    run_inherent_polarization_by_model_step(
        full_df=full_df,
        subsampled_df=subsampled_df,
        graph_output_dir=graph_output_dir,
        exclude_models=exclude_models,
    )
    run_inherent_polarization_by_dataset_step(
        full_df=full_df,
        subsampled_df=subsampled_df,
        graph_output_dir=graph_output_dir,
        exclude_models=exclude_models,
    )


# ---------------------------------------------------------------------------
# Prompt-sensitivity ANOVA and Cohen's d
# ---------------------------------------------------------------------------


def load_ndfu_anova(
    human_datasets: HumanDatasets, annotations_dir: Path, cache_dir: Path
) -> pd.DataFrame:
    """stats.compute_ndfu_anova_by_prompt's results (Holm-corrected),
    cached in cache_dir."""
    cache_path = (
        cache_dir / "polarization_by_instruction_anova_by_model_dataset.csv"
    )
    if skip_if_exists(cache_path):
        return pd.read_csv(cache_path)
    anova_df = stats.compute_ndfu_anova_by_prompt(
        human_datasets=human_datasets,
        annotations_dir=annotations_dir,
        dataset_keys=MAIN_DATASET_KEYS,
        prompt_names=MAIN_PROMPT_NAMES,
        baseline_prompt=BASELINE_PROMPT_NAME,
        correction_method="holm",
    )
    stats.write_ndfu_anova_by_prompt(anova_df, cache_path)
    return anova_df


def run_cohens_d_step(
    anova_df: pd.DataFrame,
    latex_output_dir: Path,
    exclude_models: Collection[str],
) -> None:
    # The effect sizes are computed independently per (model, dataset), so
    # dropping models from the results gives exactly the table one would
    # get by never running them.
    for variant in _model_variants(exclude_models):
        output_path = latex_output_dir / f"cohens_d{variant.file_suffix}.tex"
        variant_df = _drop_models(anova_df, variant.dropped_models)
        if variant_df.empty:
            print(f"No Cohen's d results left; skipping {output_path}.")
            continue
        table_export.export_cohens_d_table(
            stats.compute_cohens_d_compact_table(
                variant_df,
                prompt_names=MAIN_PROMPT_NAMES,
                baseline_prompt=BASELINE_PROMPT_NAME,
            ),
            output_path=output_path,
            label=f"tab:cohens-d{variant.label_suffix}",
            all_models=variant.all_models,
        )


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return n


if __name__ == "__main__":
    # Warnings from every library, but INFO only from this project, so
    # third-party INFO messages (e.g. matplotlib's font lookup) stay hidden.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("src").setLevel(logging.INFO)
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
    parser.add_argument(
        "--n-workers",
        type=_positive_int,
        default=1,
        help=(
            "Number of worker processes for the apunim computation (one "
            "task per dataset/prompt/model pair not yet cached). Only that "
            "step is parallelized. Default: 1 (no multiprocessing)."
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
        cache_dir=Path(args.cache_dir),
        human_results_dir=Path(args.human_results_dir),
        exclude_models=EXCLUDED_MODELS,
        n_workers=args.n_workers,
    )
