import argparse
import re
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from scipy import stats

from ..lib import graphs, run_helper
from ..lib.util import (
    SIGNIFICANCE_LEGEND,
    center_table_latex,
    significance_superscript,
    small_table_latex,
    trim_numeric_col_latex,
)
from ..lib.preprocessing import (
    Dataset,
    DicesDataset,
    PopquornDataset,
    SapDataset,
    KumarDataset,
)

MIN_SUPPORT = 50
SIG_ALPHA = 0.05  # p-value threshold for "statistically significant"

# Explicit marker cycle so we can reuse the *same* marker per feature when
# overlaying filled vs. hollow points. (sns.lineplot's auto-assigned
# style-index isn't something you can reliably recover after the fact, so
# we do the cycling ourselves.) Line styles live in graphs.LINESTYLES.
MARKER_CYCLE = ["o", "X", "P", "^", "D", "s", "v", "*", "d", "p"]


def main(
    results_dir: Path,
    dices_small_path: Path,
    dices_large_path: Path,
    popquorn_offensiveness_path: Path,
    kumar_path: Path,
    sap_path: Path,
    latex_output_dir: Path,
    graph_output_dir: Path,
    cache_dir: Path | None = None,
    ratings_path: Path | None = None,
):
    graphs.graph_setup()
    csv_to_latex(
        result_paths=list(results_dir.rglob("*-results.csv")),
        latex_output_dir=latex_output_dir,
    )
    plot_dfu_histograms(
        file_paths=list(results_dir.rglob("*-inherent.csv")),
        subsampled_paths=list(
            results_dir.rglob(
                f"*{run_helper.INHERENT_SUBSAMPLED_STEM_SUFFIX}.csv"
            )
        ),
        graph_output_dir=graph_output_dir,
    )
    plot_sample_size_polarization(
        csv_path=results_dir / "sample_size_polarization.csv",
        output_path=graph_output_dir / "sample_size_polarization.png",
    )
    if ratings_path is not None:
        if cache_dir is None:
            raise ValueError("--cache-dir is required with --ratings-path.")
        ndfu_rating_correlation_to_latex(
            cache_dir=cache_dir,
            ratings_path=ratings_path,
            output_path=latex_output_dir / "ndfu_rating_correlations.tex",
            label="tab:human-ndfu-rating-correlations",
        )

    dices350_ds = DicesDataset(dataset_path=dices_small_path, variant="350")
    dices990_ds = DicesDataset(dataset_path=dices_large_path, variant="990")
    popquorn_ds = PopquornDataset(dataset_path=popquorn_offensiveness_path)
    kumar_ds = KumarDataset(dataset_path=kumar_path)
    sap_ds = SapDataset(dataset_path=sap_path)
    datasets: list[Dataset] = [
        dices350_ds,
        dices990_ds,
        popquorn_ds,
        kumar_ds,
        sap_ds,
    ]

    ann_size_df = get_annotator_counts_df(datasets)
    stats_df = get_statistics_df(ann_size_df)
    annotation_count_table_to_latex(
        stats_df=stats_df,
        output_path=latex_output_dir / "annotation_count_table.tex",
    )
    subgroup_counts_to_latex(
        datasets=datasets,
        latex_output_dir=latex_output_dir,
    )

    ann_size_df = get_annotator_counts_df(datasets)
    stats_df = get_statistics_df(ann_size_df)
    stats_df.to_latex(
        latex_output_dir / "ann_stats.tex",
        caption=(
            "Descriptive statistics for the number of annotations per dataset."
        ),
        label="tab:num-annot",
        position="ht",
        index=True,
        float_format="%.4f",
        escape=True,
    )

    plot_annotator_count_histogram_from_datasets(
        datasets=datasets,
        graph_path=graph_output_dir / "annotator_count_histogram.png",
    )


def _inherent_label(path: Path, stem_suffix: str = "-inherent") -> str:
    """Dataset label from an inherent-polarization CSV name, e.g.
    "dices-350-inherent.csv" -> "Dices 350"."""
    return " ".join(
        path.stem.removesuffix(stem_suffix).split("-")
    ).capitalize()


def _read_inherent(path: Path) -> pd.Series:
    arr = pd.read_csv(path).inherent_polarization
    return arr[~np.isnan(arr)]


def _draw_density_bars(
    ax, values: pd.Series, edges: np.ndarray, sign: int, color, hatch
) -> None:
    """Horizontal density histogram on shared `edges`, drawn to the right
    of x=0 (sign=1) or mirrored to the left (sign=-1)."""
    counts, _ = np.histogram(values, bins=edges)
    widths = np.diff(edges)
    density = counts / (counts.sum() * widths) if counts.sum() else counts
    ax.barh(
        y=edges[:-1],
        width=sign * density,
        height=widths,
        align="edge",
        color=color,
        alpha=0.7,
        hatch=hatch,
        edgecolor="black",
        linewidth=0.3,
    )


def plot_dfu_histograms(
    file_paths: list[Path],
    graph_output_dir: Path,
    subsampled_paths: list[Path] | None = None,
    bins: int = 30,
):
    """
    Mirrored histograms of unattributable polarization per dataset, with
    colorblind palette and hatch patterns.

    Right of the centre line: the full datasets ("<dataset>-inherent.csv").
    Left of it: the same datasets with at most 6 annotators per comment
    (every comment subsampled down to 6; "<dataset>-inherent-subsampled-n6.csv"),
    in the same colour and hatch, so each dataset's two histograms mirror
    each other. A dataset whose subsampled file is missing is only drawn on
    the right, with a printed warning.

    All histograms share one set of bins on [0, 1], so the two sides line up.
    """
    full_series = {
        _inherent_label(Path(path)): _read_inherent(path)
        for path in file_paths
    }
    subsampled_series = {
        _inherent_label(
            Path(path), run_helper.INHERENT_SUBSAMPLED_STEM_SUFFIX
        ): _read_inherent(path)
        for path in (subsampled_paths or [])
    }

    # Colours/hatches follow the full datasets, as in the original figure.
    datasets = sorted(full_series)
    extra = sorted(set(subsampled_series) - set(datasets))
    styles = {
        name: (color, hatch)
        for name, color, hatch in zip(
            datasets + extra, graphs.COLORBLIND_PALETTE, graphs.HATCHES
        )
    }
    edges = np.linspace(0, 1, bins + 1)

    for name in sorted(set(datasets) - set(subsampled_series)):
        print(
            f"Warning: no max-{run_helper.INHERENT_SUBSAMPLE_SIZE} file for "
            f"{name}; it is only shown with all annotators. Rerun its script."
        )

    fig, ax = plt.subplots(figsize=(10, 4))

    for sign, series in ((1, full_series), (-1, subsampled_series)):
        for name in sorted(series):
            _draw_density_bars(ax, series[name], edges, sign, *styles[name])

    # symmetric x-range so the centre line sits in the middle
    xmax = max(abs(x) for x in ax.get_xlim())
    ax.set_xlim(-xmax, xmax)
    ax.axvline(0, color="black", linewidth=0.8)
    # densities are positive on both sides
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{abs(x):g}"))

    size = run_helper.INHERENT_SUBSAMPLE_SIZE
    for x, text, ha in (
        (-0.02, rf"Max {size} annotators", "right"),
        (0.02, "All annotators", "left"),
    ):
        ax.text(
            0.5 + x,
            0.97,
            text,
            transform=ax.transAxes,
            ha=ha,
            va="top",
            fontsize="small",
        )

    legend_handles = [
        mpatches.Patch(
            facecolor=styles[name][0],
            edgecolor="black",
            hatch=styles[name][1],
            label=name,
            alpha=0.4,
        )
        for name in datasets + extra
    ]

    ax.set_title(
        "UnPol Frequently Exists in Datasets with Low Annotation Density"
    )
    ax.set_xlabel("Density")
    ax.set_ylabel("UnPol")
    ax.set_ylim((0, 1))

    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    graphs.save_plot(graph_output_dir / "apriori.png")
    plt.close()


def plot_annotator_count_histogram_from_datasets(
    datasets: list[Dataset],
    graph_path: Path,
):
    """
    Plot a histogram of annotator counts per comment across multiple datasets,
    showing the percentage of comments for each bin.

    Parameters
    ----------
    datasets : list
        List of dataset objects.
    graph_path : Path
        If provided, saves the figure to this path.
    """
    N_BINS = 100
    all_df = get_annotator_counts_df(datasets)

    dataset_names = all_df["dataset"].unique().tolist()

    # Determine bin boundaries
    min_val = all_df["n_annotators"].min()
    max_val = all_df["n_annotators"].max()
    bins_edges = np.linspace(min_val, max_val, N_BINS + 1)
    _, ax = plt.subplots()

    for i, dataset_name in enumerate(dataset_names):
        data_subset = all_df[all_df["dataset"] == dataset_name]["n_annotators"]

        raw_counts, edges = np.histogram(data_subset, bins=bins_edges)

        total_comments_for_dataset = len(data_subset)

        if total_comments_for_dataset > 0:
            percentage_counts = raw_counts / total_comments_for_dataset
        else:
            percentage_counts = np.zeros_like(raw_counts, dtype=float)

        selected_color = graphs.COLORBLIND_PALETTE[
            i % len(graphs.COLORBLIND_PALETTE)
        ]
        selected_hatch = graphs.HATCHES[i % len(graphs.HATCHES)]

        ax.bar(
            x=edges[:-1],
            height=percentage_counts * 100,
            width=(edges[1] - edges[0]),
            label=dataset_name,
            color=selected_color,
            alpha=0.6,
            hatch=selected_hatch,
            edgecolor="black",
        )

    ax.legend(title=None, loc="center")
    ax.set_xlabel(r"\# Annotators")
    ax.set_ylabel(r"Comments (\%)")
    ax.set_title(r"\# Annotators per comment for each dataset")
    ax.grid(True, linestyle=graphs.LINESTYLES[1], alpha=0.3)
    plt.tight_layout()

    graphs.save_plot(graph_path)
    plt.close()


def get_annotator_counts_df(
    datasets: list[Dataset],
) -> pd.DataFrame:
    rows = []

    for ds in datasets:
        df = ds.get_dataset().reset_index(drop=True)
        ann_col = ds.get_annotation_column()
        ds_name = ds.get_name()

        tmp = pd.DataFrame(
            {
                "dataset": ds_name,
                "n_annotators": df[ann_col].apply(len),
            }
        )
        rows.append(tmp)

    all_df = pd.concat(rows, ignore_index=True).dropna(subset=["n_annotators"])
    return all_df


def get_statistics_df(all_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a dataframe where each row corresponds to a dataset and each column
    is a statistic from `describe()` applied to annotator counts.
    """
    return (
        all_df.groupby("dataset", sort=False)["n_annotators"]
        .describe()  # computes count, mean, std, min, 25%, 50%, 75%, max
        .rename_axis(index=None)  # optional: cleaner row index name
    )


def csv_to_latex(result_paths: list[Path], latex_output_dir: Path) -> None:
    for result_file in result_paths:
        if "sample_size" not in result_file.stem:
            dataset_name = result_file.stem.split("_")[0]
            df = pd.read_csv(result_file)
            df = df.loc[df.pvalue.notna()]
            _results_to_latex(
                res_df=df,
                output_path=latex_output_dir / f"{dataset_name}.tex",
                dataset_name=dataset_name,
                table_label=f"tab:{dataset_name}",
            )


def _fmt_num(x: float) -> str:
    """Formats a float to 4 decimal places, stripping trailing zeros
    (and a trailing decimal point) so whole numbers print cleanly."""
    return f"{x:.4f}".rstrip("0").rstrip(".")


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def annotation_count_table_to_latex(
    stats_df: pd.DataFrame, output_path: Path
) -> None:
    """
    Write a LaTeX table* summarizing the distribution of the number of
    annotations per comment (one row per dataset), from the output of
    `get_statistics_df`.
    """
    lines = [
        "\\begin{table*}[t]",
        "\t\\centering",
        "\t\\caption{Descriptive statistics for the number of "
        "annotations per comment, grouped by dataset.}",
        "\t\\label{tab:num-annot}",
        "\t\\begin{tabular}{lrrrrrrrr}",
        "\t\t\\toprule",
        "\t\t& count & mean & std & min & 25\\% & 50\\% & 75\\% & max \\\\",
        "\t\t\\midrule",
    ]
    for name, row in stats_df.iterrows():
        lines.append(
            f"\t\t{name} & {int(row['count'])} & "
            f"{_fmt_num(row['mean'])} & "
            f"{_fmt_num(row['std'])} & "
            f"{_fmt_num(row['min'])} & "
            f"{_fmt_num(row['25%'])} & "
            f"{_fmt_num(row['50%'])} & "
            f"{_fmt_num(row['75%'])} & "
            f"{_fmt_num(row['max'])} \\\\"
        )
    lines += [
        "\t\t\\bottomrule",
        "\t\\end{tabular}",
        "\\end{table*}",
    ]
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Table exported to {output_path.resolve()}")


def get_subgroup_counts(ds: Dataset) -> dict[str, pd.Series]:
    """Kept for backwards compatibility; see Dataset.get_subgroup_counts."""
    return ds.get_subgroup_counts()


def subgroup_counts_to_latex(
    datasets: list[Dataset], latex_output_dir: Path
) -> None:
    """
    Write one LaTeX table per dataset with the number of annotations in each
    subgroup of every personal characteristic (SDB) dimension.
    """
    for ds in datasets:
        counts = pd.concat(
            get_subgroup_counts(ds), names=[r"\ac{sdd}", "Value"]
        ).to_frame("Count")
        latex_str = counts.to_latex(
            caption=(
                f"Number of annotations per subgroup for the {ds.get_name()} "
                "dataset."
            ),
            label=f"tab:subgroups-{_slug(ds.get_name())}",
            escape=False,
            position="t",
            index=True,
            multirow=False,
            longtable=len(counts) > 40,
        )
        latex_str = center_table_latex(latex_str=latex_str)

        output_path = (
            latex_output_dir / f"{_slug(ds.get_name())}_subgroup_counts.tex"
        )
        output_path.write_text(latex_str)
        print(f"Table exported to {output_path.resolve()}")


def _results_to_latex(
    res_df: pd.DataFrame,
    output_path: Path,
    dataset_name: str,
    table_label: str,
    columns: list[str] | None = None,
) -> None:
    """
    Export results to a single LaTeX table where apunim values include
    significance stars (as superscripts), and the pvalue column is removed.
    """
    res_df = (
        res_df.replace("_", r"\_", regex=True)
        .rename(columns={"Unnamed: 1": "Value", "SDB Feature": r"\ac{sdd}"})
        .set_index([r"\ac{sdd}", "Value"])
    )

    if "pvalue" in res_df.columns and "apunim" in res_df.columns:
        res_df["apunim"] = res_df.apply(
            lambda r: (
                f"{r['apunim']:.4f}{significance_superscript(r['pvalue'])}"
                if not pd.isna(r["pvalue"])
                else "---"
            ),
            axis=1,
        )
        res_df = res_df.drop(columns=["pvalue"])

    if columns is None:
        columns = list(res_df.columns)

    latex_str = res_df.to_latex(
        caption=(
            "Aposteriori unimodality results for the "
            f"{dataset_name.capitalize()} dataset. "
            "Stars indicate statistical significance: "
            f"{SIGNIFICANCE_LEGEND}"
        ),
        label=table_label,
        escape=False,  # allow LaTeX math ($^{*}$)
        columns=columns,
        position="t",
        index=True,
        float_format="%.4f",
        multirow=False,
        longtable=dataset_name == "kumar",
    )

    latex_str = center_table_latex(latex_str=latex_str)

    # Write to file
    output_path.write_text(latex_str)
    print(f"Table exported to {output_path.resolve()}")


# ---------------------------------------------------------------------------
# Human nDFU vs. external polarization ratings
# ---------------------------------------------------------------------------

# Columns of the ratings CSV: one row per rated comment, with the dataset it
# comes from, its id in that dataset and its 1-5 polarization rating.
RATING_DATASET_COLUMN = "dataset"
RATING_KEY_COLUMN = "item_id"
RATING_TEXT_COLUMN = "text"
RATING_COLUMN = "polarization_rating"

# Dataset label in the ratings CSV -> (name of its human nDFU cache file, see
# run_helper.human_ndfu_path; name shown in the table). Also the row order.
RATING_DATASETS = {
    "swx": ("popquorn", "POPQUORN"),
    "sap": ("sap", "Sap et al. 2022"),
    "dices_990": ("dices-990", "DICES-990"),
    "dices_350": ("dices-350", "DICES-350"),
}


def load_polarization_ratings(path: Path) -> pd.DataFrame:
    """
    Reads a ratings CSV (columns `dataset`, `item_id`, `text` and
    `polarization_rating`) with one numeric rating per (dataset, comment).
    Duplicate comments keep their first rating.
    """
    df = pd.read_csv(path)
    required = [
        RATING_DATASET_COLUMN,
        RATING_KEY_COLUMN,
        RATING_TEXT_COLUMN,
        RATING_COLUMN,
    ]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required column(s): {sorted(missing)}"
        )
    df = df[required].copy()
    df[RATING_COLUMN] = pd.to_numeric(df[RATING_COLUMN], errors="coerce")
    df = df.dropna(subset=[RATING_DATASET_COLUMN, RATING_COLUMN])
    return df.drop_duplicates(
        [RATING_DATASET_COLUMN, RATING_KEY_COLUMN]
    ).reset_index(drop=True)


def _normalize_text(values: pd.Series) -> pd.Series:
    """Text with whitespace runs collapsed, for matching comments by text."""
    return values.astype(str).str.split().str.join(" ")


def _match_ndfu_to_ratings(
    ndfu_df: pd.DataFrame, ratings_df: pd.DataFrame
) -> pd.DataFrame:
    """
    The per-comment nDFU of one dataset (columns `comment`, `ndfu`) joined
    with its ratings. Comments are matched either on the id (comment ==
    item_id) or on the comment text (comment == text, for datasets such as
    Sap et al. that identify their comments by their text); whichever
    matches more comments is used.
    """
    by_id = ndfu_df.assign(
        _key=pd.to_numeric(ndfu_df["comment"], errors="coerce")
    ).merge(
        ratings_df.assign(
            _key=pd.to_numeric(ratings_df[RATING_KEY_COLUMN], errors="coerce")
        )
        .dropna(subset=["_key"])
        .drop_duplicates("_key")[["_key", RATING_COLUMN]],
        on="_key",
    )
    by_text = ndfu_df.assign(_key=_normalize_text(ndfu_df["comment"])).merge(
        ratings_df.assign(
            _key=_normalize_text(ratings_df[RATING_TEXT_COLUMN])
        ).drop_duplicates("_key")[["_key", RATING_COLUMN]],
        on="_key",
    )
    return by_id if len(by_id) >= len(by_text) else by_text


def _correlation_stats(x: pd.Series, y: pd.Series) -> dict:
    """Spearman and Kendall correlation coefficients of two aligned series
    (NaN when undefined, e.g. too few points or a constant series)."""
    out = {"n": len(x), "spearman_rho": np.nan, "kendall_tau": np.nan}
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return out
    out["spearman_rho"] = stats.spearmanr(x, y)[0]
    out["kendall_tau"] = stats.kendalltau(x, y, variant="b")[0]
    return out


def compute_ndfu_rating_correlations(
    cache_dir: Path, ratings_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Correlates, per dataset, the per-comment nDFU of the human annotations
    (read from the cache files written by the human scripts, one per
    dataset) with the polarization rating of the comment.

    One row per dataset of RATING_DATASETS that has ratings and a cache
    file: `n_rated` (rated comments), `n` (comments with both an nDFU and a
    rating), Spearman's rho (the main statistic, since the ratings are
    ordinal) and Kendall's tau-b (robust to the many ties of a 5-point
    scale). No significance tests are computed.
    """
    unknown = sorted(
        set(ratings_df[RATING_DATASET_COLUMN]) - set(RATING_DATASETS)
    )
    if unknown:
        print(f"Ratings for unknown dataset(s) {unknown} are ignored.")

    records = []
    for label, (stem, name) in RATING_DATASETS.items():
        ratings = ratings_df.loc[ratings_df[RATING_DATASET_COLUMN] == label]
        if ratings.empty:
            continue
        ndfu_path = run_helper.human_ndfu_path(cache_dir, stem)
        if not ndfu_path.exists():
            print(
                f"{name}: {ndfu_path} not found; run the dataset's human "
                "script first. Skipping."
            )
            continue
        ndfu_df = pd.read_csv(ndfu_path)
        pairs = _match_ndfu_to_ratings(ndfu_df, ratings)
        if pairs.empty:
            print(
                f"{name}: {len(ndfu_df)} nDFU values and {len(ratings)} "
                "ratings, but none could be matched; check that the ids / "
                f"texts agree (nDFU comment examples: "
                f"{ndfu_df['comment'].head(3).tolist()}, rating item_id "
                f"examples: {ratings[RATING_KEY_COLUMN].head(3).tolist()})."
            )
        records.append(
            {
                "Dataset": name,
                "n_rated": len(ratings),
                **_correlation_stats(pairs["ndfu"], pairs[RATING_COLUMN]),
            }
        )
    return pd.DataFrame(
        records,
        columns=["Dataset", "n_rated", "n", "spearman_rho", "kendall_tau"],
    )


def ndfu_rating_correlation_to_latex(
    cache_dir: Path, ratings_path: Path, output_path: Path, label: str
) -> None:
    """
    LaTeX table of the correlation between the human per-comment nDFU and
    the polarization ratings: one row per dataset with the number of
    comments, Spearman's rho and Kendall's tau-b (two decimals).
    """
    corr_df = compute_ndfu_rating_correlations(
        cache_dir, load_polarization_ratings(ratings_path)
    )
    if corr_df.empty:
        print(
            "No human nDFU values could be matched to the polarization "
            f"ratings; skipping {output_path}."
        )
        return

    table = pd.DataFrame(
        {
            "Dataset": corr_df["Dataset"],
            "$n$": corr_df["n"].astype(str),
            r"Spearman $\rho$": trim_numeric_col_latex(
                corr_df["spearman_rho"], ".2f"
            ),
            r"Kendall $\tau_b$": trim_numeric_col_latex(
                corr_df["kendall_tau"], ".2f"
            ),
        }
    ).replace("_", r"\_", regex=True)

    caption = (
        r"Correlation between the per-comment \ac{ndfu} of the human "
        "annotations and the polarization ratings, per dataset. $n$ is the "
        r"number of comments with both an \ac{ndfu} and a rating."
    )
    latex_str = table.to_latex(
        caption=caption,
        label=label,
        position="t",
        escape=False,
        index=False,
        column_format="lrrr",
    )
    latex_str = small_table_latex(center_table_latex(latex_str))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_str)
    print(f"Table exported to {output_path.resolve()}")


def plot_sample_size_polarization(csv_path: Path, output_path: Path):
    df = pd.read_csv(csv_path)

    _, ax = plt.subplots()

    for dataset, group in df.groupby("dataset"):
        color = sns.color_palette()[
            list(df["dataset"].unique()).index(dataset)
        ]
        ax.plot(
            group["sample_size"], group["mean"], label=dataset, color=color
        )
        ax.fill_between(
            group["sample_size"],
            group["mean"] - group["std"],
            group["mean"] + group["std"],
            alpha=0.2,
            color=color,
        )

    ax.set_xlabel("Number of annotators")
    ax.set_ylabel("Mean polarization")
    ax.legend(title="Dataset")
    graphs.save_plot(output_path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Create graphs and latex tables from results.")
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Results CSV directory.",
    )
    parser.add_argument(
        "--dices-small-path",
        required=True,
        help="Path to the DICES 350 annotator CSV file.",
    )
    parser.add_argument(
        "--dices-large-path",
        required=True,
        help="Path to the DICES 990 annotator CSV file.",
    )
    parser.add_argument(
        "--popquorn-path",
        required=True,
        help=("Path to the POPQUORN offensiveness dataset."),
    )
    parser.add_argument(
        "--sap-path",
        required=True,
        help=("Path to the Sap dataset."),
    )
    parser.add_argument(
        "--kumar-path",
        required=True,
        help=("Path to the Kumar dataset."),
    )
    parser.add_argument(
        "--cache-dir",
        help=(
            "Directory with the human nDFU files written by the human "
            "scripts (required with --ratings-path)."
        ),
    )
    parser.add_argument(
        "--ratings-path",
        help=(
            "Optional CSV with columns dataset, item_id, text and "
            "polarization_rating. If given, a table of the correlation "
            "between the human per-comment nDFU and these ratings is "
            "exported."
        ),
    )
    parser.add_argument(
        "--latex-output-dir",
        required=True,
        help="Directory for the latex tables.",
    )
    parser.add_argument(
        "--graph-output-dir",
        required=True,
        help="Directory for graphs.",
    )
    args = parser.parse_args()
    main(
        results_dir=Path(args.results_dir),
        sap_path=Path(args.sap_path),
        dices_small_path=Path(args.dices_small_path),
        dices_large_path=Path(args.dices_large_path),
        kumar_path=Path(args.kumar_path),
        popquorn_offensiveness_path=Path(args.popquorn_path),
        latex_output_dir=Path(args.latex_output_dir),
        graph_output_dir=Path(args.graph_output_dir),
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        ratings_path=Path(args.ratings_path) if args.ratings_path else None,
    )
