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

5. For each dataset, a LaTeX table of which PC dimensions each model finds
   statistically significant under each prompt, next to the human findings
   (llm-apunim-significance-<dataset>.tex; also without the models in
   EXCLUDED_MODELS, with an "_excluding" suffix).

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
from ..lib.util import skip_if_exists
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
        human_datasets, annotations_dir, latex_output_dir, cache_dir=cache_dir
    )
    run_apunim_significance_step(
        human_datasets,
        annotations_dir,
        latex_output_dir,
        cache_dir,
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


def run_cohens_d_step(res_df, latex_output_dir, exclude_models):
    # The effect sizes are computed independently per (model, dataset), so
    # dropping models from the results gives exactly the table one would
    # get by never running them.
    caption = r"""Cohen's d between the default and each adversarial
        instruction prompt, computed on the per-comment \ac{ndfu}. Rows give
        results per model, for each dataset separately and as the unweighted
        mean over datasets (All); $n$ is the number of comments."""
    for file_suffix, label_suffix, note, models in _model_variants(
        exclude_models
    ):
        variant_df = _drop_models(res_df, models)
        if variant_df.empty:
            print(f"No Cohen's d results left{note}; skipping table.")
            continue
        stats.export_compact_prompt_table_latex(
            stats.compute_cohens_d_compact_table(variant_df),
            output_path=latex_output_dir / f"cohens_d{file_suffix}.tex",
            caption=caption + note,
            label=f"tab:cohens-d{label_suffix}",
        )


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


def _load_or_compute_apunim(
    annotations_dir: Path, cache_dir: Path, key: str
) -> pd.DataFrame:
    """
    apunim results for every model and prompt of one dataset, cached in
    cache_dir/apunim_<key>.csv and shared by the apunim-by-prompt tables and
    the significant-PC-dimension tables.

    The cache is checked per (model, prompt) pair -- the unit apunim is run
    on -- so annotation runs that are still in progress are handled: pairs
    already in the cache are never recomputed, pairs whose annotation file
    appears later are added on the next run, and the cache is rewritten
    after every pair, so an interrupted run keeps what it finished.
    """
    cache_path = cache_dir / f"apunim_{key}.csv"
    long_df = pd.DataFrame()
    if cache_path.exists():
        long_df = pd.read_csv(cache_path)
        # Older caches were written with the DataFrame index as a column.
        long_df = long_df.loc[:, ~long_df.columns.str.startswith("Unnamed")]
    cached_pairs = (
        set(zip(long_df["Model"], long_df["Prompt"]))
        if not long_df.empty
        else set()
    )

    n_computed = 0
    for prompt_name in shared.MAIN_PROMPT_NAMES:
        files = shared.find_annotation_files(annotations_dir, key, prompt_name)
        for model in sorted(files):
            if (model, prompt_name) in cached_pairs:
                continue
            print(f"Computing apunim: {key} / {prompt_name} / {model}")
            new_df = stats.compute_llm_apunim_by_prompt(
                annotations_dir=annotations_dir,
                dataset_key=key,
                prompt_names=[prompt_name],
                exclude_models=set(files) - {model},
            )
            if new_df.empty:
                continue
            long_df = (
                new_df
                if long_df.empty
                else pd.concat([long_df, new_df], ignore_index=True)
            )
            n_computed += 1

            # Write to a temp file and rename, so an interrupt during the
            # write cannot leave a truncated cache behind.
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(".csv.tmp")
            long_df.to_csv(tmp_path, index=False)
            tmp_path.replace(cache_path)

    if not n_computed:
        print(f"Skipping (nothing new to compute): {cache_path}")
    return long_df


def _run_apunim_prompt_table_for_dataset(
    human_datasets: list[str],
    annotations_dir: Path,
    latex_output_dir: Path,
    cache_dir: Path,
    key: str,
):
    long_df = _load_or_compute_apunim(annotations_dir, cache_dir, key)

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
    human_datasets: list[str],
    annotations_dir: Path,
    latex_output_dir: Path,
    cache_dir: Path,
):
    # Restricted to the datasets the adversarial prompts were run on
    # (shared.PROMPT_COMPARISON_DATASET_KEYS).
    for key in shared.PROMPT_COMPARISON_DATASET_KEYS:
        if key in human_datasets:
            _run_apunim_prompt_table_for_dataset(
                human_datasets=human_datasets,
                annotations_dir=annotations_dir,
                latex_output_dir=latex_output_dir,
                key=key,
                cache_dir=cache_dir,
            )


def run_apunim_significance_step(
    human_datasets,
    annotations_dir,
    latex_output_dir,
    cache_dir,
    human_results_dir,
    exclude_models,
):
    """
    Per dataset: which PC dimensions each model finds statistically
    significant under each prompt, compared with the human results.
    """
    for key in shared.PROMPT_COMPARISON_DATASET_KEYS:
        if key not in human_datasets:
            continue
        human_df = stats.load_human_apunim_results(human_results_dir, key)
        if human_df is None:
            continue

        llm_df = _load_or_compute_apunim(annotations_dir, cache_dir, key)

        for file_suffix, label_suffix, note, models in _model_variants(
            exclude_models
        ):
            variant_df = _drop_models(llm_df, models)
            if variant_df.empty:
                continue
            stats.export_apunim_significance_table(
                stats.build_apunim_significance_table(
                    human_df, variant_df, prompt_names=shared.MAIN_PROMPT_NAMES
                ),
                output_path=latex_output_dir
                / f"llm-apunim-significance-{key}{file_suffix}.tex",
                dataset_name=human_datasets[key].get_name(),
                label=f"tab:llm-apunim-significance-{key}{label_suffix}",
                note=note,
            )


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
