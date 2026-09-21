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
from pathlib import Path

import pandas as pd

from ..lib import graphs
from ..lib.util import skip_if_exists
from . import polarization, stats, common, plots

# TODO: Separate table export and io from common?


def run_histogram_step(
    human_datasets, annotations_dir, graph_output_dir, prompt_name
):
    output_path = graph_output_dir / "human_vs_llm_histograms.png"
    plots.plot_annotation_histograms(
        human_datasets=human_datasets,
        annotations_dir=annotations_dir,
        output_path=output_path,
        prompt_name=prompt_name,
    )


def run_prompt_diff_step(
    human_datasets, annotations_dir, graph_output_dir, exclude_models
):
    output_path = graph_output_dir / "llm_prompt_mean_diff.png"
    plots.plot_prompt_mean_diff(
        human_datasets=human_datasets,
        annotations_dir=annotations_dir,
        output_path=output_path,
        prompt_names=common.MAIN_PROMPT_NAMES,
        exclude_models=exclude_models,
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
            human_datasets=human_datasets, paraphrase_dir=paraphrase_dir
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
        ),
        output_path=output_path,
        caption=(
            "Consistency (Krippendorff's $\\alpha$, ordinal) of each model "
            f"with itself across repeated runs of the same ({prompt_name}) "
            "prompt."
        ),
        label="tab:llm-consistency-repeats",
    )


def _run_apunim_prompt_table_for_dataset(
    human_datasets: list[str],
    annotations_dir: Path,
    latex_output_dir: Path,
    cache_dir: Path,
    key: str,
):
    cache_path = cache_dir / f"apunim_{key}.csv"
    if skip_if_exists(cache_path):
        long_df = pd.read_csv(cache_path)
    else:
        long_df = stats.compute_llm_apunim_by_prompt(
            annotations_dir=annotations_dir,
            dataset_key=key,
            prompt_names=common.MAIN_PROMPT_NAMES,
            exclude_models=common.APUNIM_TABLE_EXCLUDE_MODELS,
        )
        long_df.to_csv(cache_path)

    wide_df = stats.build_llm_apunim_prompt_table(
        long_df, prompt_names=common.MAIN_PROMPT_NAMES
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
    # Restricted to the datasets the paraphrase/stereotype/persona prompts
    # were actually run on (kumar, sap).
    for key in common.PROMPT_COMPARISON_DATASET_KEYS:
        if key in human_datasets:
            _run_apunim_prompt_table_for_dataset(
                human_datasets=human_datasets,
                annotations_dir=annotations_dir,
                latex_output_dir=latex_output_dir,
                key=key,
                cache_dir=cache_dir,
            )


def _run_apunim_grid_for_prompt(
    human_datasets, annotations_dir, graph_output_dir, prompt_name
):
    output_path = graph_output_dir / f"llm_apunim_grid_{prompt_name}.png"
    grid_models = list(
        set(common.MODEL_DISPLAY_ORDER) - common.APUNIM_TABLE_EXCLUDE_MODELS
    )
    plots.plot_apunim_grid(
        human_datasets=human_datasets,
        annotations_dir=annotations_dir,
        output_path=output_path,
        prompt_name=prompt_name,
        models=grid_models,
        title=f"{prompt_name.capitalize()} Prompt",
    )


def run_apunim_grid_steps(human_datasets, annotations_dir, graph_output_dir):
    for prompt_name in common.MAIN_PROMPT_NAMES:
        _run_apunim_grid_for_prompt(
            human_datasets, annotations_dir, graph_output_dir, prompt_name
        )


def run_inherent_polarization_step(
    human_datasets, annotations_dir, latex_output_dir, exclude_models
):
    # Deliberately separate from latex_output_dir: these caches are shared
    # with sap.py/kumar.py/dices.py's own "-inherent.csv" outputs.
    human_results_dir = Path("output/main")
    apunim_output_dir = Path("output/apunim")

    output_path = latex_output_dir / "inherent-polarization.tex"
    inherent_df = polarization.compute_inherent_polarization_comparison(
        human_datasets=human_datasets,
        annotations_dir=annotations_dir,
        apunim_output_dir=apunim_output_dir,
        human_results_dir=human_results_dir,
        dataset_keys=common.DATASET_KEYS[2:],
        prompt_names=common.MAIN_PROMPT_NAMES,
        exclude_models=set(exclude_models),
    )
    inherent_table_df = polarization.build_inherent_polarization_table(
        long_df=inherent_df,
        prompt_names=common.MAIN_PROMPT_NAMES,
    )
    polarization.export_inherent_polarization_table(
        df=inherent_table_df,
        output_path=output_path,
        label="tab:inherent-polarization",
        float_format=".2f"
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
    cache_dir: Path,
    exclude_models: list[str],
    prompt_name: str = "default",
):
    graphs.graph_setup()
    graph_output_dir.mkdir(parents=True, exist_ok=True)
    latex_output_dir.mkdir(parents=True, exist_ok=True)

    human_datasets = common.load_human_datasets(
        dices_small_path=dices_small_path,
        dices_large_path=dices_large_path,
        sap_path=sap_path,
        kumar_path=kumar_path,
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
    run_apunim_grid_steps(human_datasets, annotations_dir, graph_output_dir)

    run_inherent_polarization_step(
        human_datasets, annotations_dir, latex_output_dir, exclude_models
    )

    stats_path = cache_dir / "polarization_by_instruction_anova.csv"
    if skip_if_exists(stats_path):
        res_df = pd.read_csv(stats_path)
    else:
        res_df = stats.compute_ndfu_anova_by_prompt(
            human_datasets=human_datasets,
            annotations_dir=annotations_dir,
            correction_method="holm",
        )
    stats.export_ndfu_anova_by_prompt(
        result_df=res_df,
        output_path=stats_path
    )
    summary_df = stats.compute_cohens_d_summary_table(res_df)
    stats.export_cohens_d_summary_latex(
        summary_df,
        output_path=latex_output_dir / "cohens_d.tex",
        caption=r"""Cohen's d statistics showing the quantitative difference
        in \ac{ndfu} between the default and each adversarial instruction
        prompt for each of the groups of both datasets and across all models.""",
        label="tab:cohens-d"
    )
    stats.run_exploratory_stats(res_df)


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
        "--cache-dir",
        required=True,
        help="Directory for cached apunim computations.",
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
        cache_dir=Path(args.cache_dir),
        exclude_models=args.exclude_models,
    )
