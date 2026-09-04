"""
Compares human annotations against LLM-generated annotations produced by
llm_annotate.py / annotate_experiments.sh.

Three things are produced:

1. A grid of normalized histograms (one subplot per dataset) overlaying the
   human annotation distribution with each LLM's annotation distribution,
   for the "default" prompt.

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

3. Aposteriori-unimodality (apunim) results for every (dataset, model) pair,
   using the same pipeline as sap.py / dices.py / kumar.py (a polarization
   boxplot, an "-inherent.csv" and a "-results.csv"). The "-results.csv"
   files are named so that export_results.py's `*-results.csv` glob picks
   them up directly if pointed at the output directory used here.

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
    human_datasets: dict[str, tasks.preprocessing.Dataset],
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

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(8 * ncols, 5 * nrows), squeeze=False
    )

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
            element="step",
            fill=False,
            linewidth=2,
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
    human_datasets: dict[str, tasks.preprocessing.Dataset],
    annotations_dir: Path,
    prompt_name: str = "default",
) -> pd.DataFrame:
    """
    For each dataset: how consistent are the different LLMs with each
    other, when all of them are given the same (default) prompt?
    """
    rows = []
    for key in DATASET_KEYS:
        if key not in human_datasets:
            continue
        files = find_annotation_files(annotations_dir, key, prompt_name)
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
    human_datasets: dict[str, tasks.preprocessing.Dataset],
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
    human_datasets: dict[str, tasks.preprocessing.Dataset],
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_str)
    print(f"Table exported to {output_path.resolve()}")


# ---------------------------------------------------------------------------
# 3. Apunim on LLM annotations (mirrors sap.py / dices.py / kumar.py)
# ---------------------------------------------------------------------------


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


def run_llm_apunim(
    human_datasets: dict[str, tasks.preprocessing.Dataset],
    annotations_dir: Path,
    output_dir: Path,
    graph_output_dir: Path,
    prompt_name: str = "default",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_key in DATASET_KEYS:
        if dataset_key not in human_datasets:
            continue

        files = find_annotation_files(
            annotations_dir, dataset_key, prompt_name
        )
        for pseudo, path in files.items():
            tag = f"{dataset_key}-{prompt_name}-{pseudo}"

            results_path = output_dir / f"{tag}-results.csv"
            inherent_path = output_dir / f"{tag}-inherent.csv"
            graph_path = graph_output_dir / f"llm_apunim_{tag}.png"

            df = load_llm_df(path)
            ds = LLMAnnotationDataset(df, dataset_key, pseudo, prompt_name)

            if results_path.exists():
                print(f"Skipping (already exists): {results_path}")
            else:
                print(f"Running apunim for {tag}...")
                try:
                    res = tasks.run_helper.run_all_results(ds)
                    res.to_csv(results_path)
                except ValueError as e:
                    # apunim raises when a whole SDB dimension has no
                    # eligible/polarized comments to test (e.g. a model
                    # whose outputs are too degenerate/uniform). Skip that
                    # (dataset, model) pair rather than aborting the run.
                    print(f"  Skipping apunim results for {tag}: {e}")

            if inherent_path.exists():
                print(f"Skipping (already exists): {inherent_path}")
            else:
                try:
                    inherent = tasks.run_helper.compute_inherent_polarization_exhaustive(
                        dataset=ds, max_annotators=MAX_ANNOTATORS_PER_ITEM
                    )
                    inherent.to_csv(
                        inherent_path, header=True, index_label="comment"
                    )
                except ValueError as e:
                    print(f"  Skipping inherent polarization for {tag}: {e}")

            if graph_path.exists():
                print(f"Skipping (already exists): {graph_path}")
            else:
                try:
                    tasks.graphs.polarization_plot(
                        ds=ds, output_path=graph_path
                    )
                except ValueError as e:
                    print(f"  Skipping polarization plot for {tag}: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_human_datasets(
    dices_small_path: Path,
    dices_large_path: Path,
    sap_path: Path,
    kumar_path: Path,
) -> dict[str, tasks.preprocessing.Dataset]:
    paths = {
        "dices-350": dices_small_path,
        "dices-990": dices_large_path,
        "sap": sap_path,
        "kumar": kumar_path,
    }
    datasets = {}
    for key, path in paths.items():
        if path is None:
            continue
        datasets[key] = DATASET_LOADERS[key](path)
    return datasets


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
    apunim_output_dir: Path,
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

    # 3. Apunim on LLM annotations.
    run_llm_apunim(
        human_datasets=human_datasets,
        annotations_dir=annotations_dir,
        output_dir=apunim_output_dir,
        graph_output_dir=graph_output_dir,
        prompt_name=prompt_name,
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
        "--apunim-output-dir",
        default="output/llm",
        help=(
            "Directory for the per-(dataset, model) apunim '-results.csv' "
            "and '-inherent.csv' files. Named so export_results.py's "
            "'*-results.csv' glob picks them up if pointed here."
        ),
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
    args = parser.parse_args()

    main(
        dices_small_path=(
            Path(args.dices_small_path) if args.dices_small_path else None
        ),
        dices_large_path=(
            Path(args.dices_large_path) if args.dices_large_path else None
        ),
        sap_path=Path(args.sap_path) if args.sap_path else None,
        kumar_path=Path(args.kumar_path) if args.kumar_path else None,
        annotations_dir=Path(args.annotations_dir),
        paraphrase_dir=Path(args.paraphrase_dir),
        repeat_dir=Path(args.repeat_dir),
        graph_output_dir=Path(args.graph_output_dir),
        latex_output_dir=Path(args.latex_output_dir),
        apunim_output_dir=Path(args.apunim_output_dir),
        prompt_name=args.prompt_name,
    )
