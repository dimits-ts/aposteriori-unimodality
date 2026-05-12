# Revised script with syntactically-correct caching and dynamic sample sizes.

import typing
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import apunim

import tasks.graphs
import tasks.preprocessing
import sap
import kumar
import dices


MARKERS = {
    "DICES-350": "o",
    "DICES-990": "s",
    "Kumar et al. 2021": "^",
    "Sap et al. 2022": "*",
}


def main(
    dices_small_path: Path,
    dices_large_path: Path,
    latex_output_dir: Path,
    sap_path: Path,
    kumar_path: Path,
    graph_dir: Path,
    cache_dir: Path,
    min_comment_annotators: int = 3,
):
    tasks.graphs.graph_setup()
    dices350_ds = dices.DicesDataset(
        dataset_path=dices_small_path, variant="350"
    )
    dices990_ds = dices.DicesDataset(
        dataset_path=dices_large_path, variant="990"
    )
    sap_ds = sap.SapDataset(dataset_path=sap_path)
    kumar_ds = kumar.KumarDataset(
        dataset_path=kumar_path, num_samples=kumar.NUM_COMMENTS
    )
    datasets = [dices350_ds, dices990_ds, sap_ds, kumar_ds]

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

    kumar_ds = cull_kumar_ds(kumar_ds)
    plot_annotator_count_histogram_from_datasets(
        datasets=datasets,
        graph_path=graph_dir / "annotator_count_histogram.png",
    )

    variance_df_ls = []
    for dataset in datasets:
        res_df = get_dataset_variance(
            dataset, cache_dir, min_comment_annotators=min_comment_annotators
        )
        res_df["dataset"] = dataset.get_name()
        variance_df_ls.append(res_df)

    variance_df = pd.concat(variance_df_ls, ignore_index=True)

    plot_variance_curve(
        variance_df,
        graph_path=graph_dir / "ndfu_std_error_sample_size.png",
    )


def sample_se_vs_sample_size_unimodality(
    df: pd.DataFrame,
    annotation_col: str,
    group_col: str,
    bins: int = 5,
    min_size: int = 2,
    max_size: typing.Optional[int] = None,
    step: int = 1,
    iters: int = 30,
    min_comment_annotators: int = 3,
) -> pd.DataFrame:
    """Sample decreasing subsets of annotations and compute the standard error
    of Aposteriori Unimodality statistics at each sample size.

    This version:
    - If max_size is None, uses the maximum number of annotators found across
      comments in `df[annotation_col]`.
    - Skips comments that have fewer than `min_comment_annotators`.
    """

    # determine max_size dynamically if not provided
    if max_size is None:
        # ensure we handle empty lists gracefully
        max_size = 0
        for a in df[annotation_col]:
            try:
                max_size = max(max_size, len(a))
            except Exception:
                # if entries are not list-like, attempt to coerce
                max_size = max(max_size, int(a))
        max_size = int(max_size)

    results: list[dict[str, typing.Any]] = []

    for size in tqdm(range(min_size, max_size + 1, step), desc="#Annotators"):
        iter_sds: list[float] = []

        for _ in tqdm(range(iters), desc="#Iterations", leave=False):
            sample_stats: list[float] = []

            for _, row in df.iterrows():
                anns = row[annotation_col]
                grps = row[group_col]

                if anns is None:
                    continue
                try:
                    n_ann = len(anns)
                except Exception:
                    continue

                if n_ann < min_comment_annotators or n_ann < size:
                    continue

                annotations = np.array(anns)
                groups = np.array(grps)

                idx = np.random.choice(n_ann, size=size, replace=False)
                sub_ann = annotations[idx]
                sub_grp = groups[idx]

                stats_dict = apunim._factor_dfu_stat(
                    sub_ann, sub_grp, bins=bins
                )

                sample_stats.extend(
                    [float(v) for v in stats_dict.values() if not np.isnan(v)]
                )

            if len(sample_stats) > 1:
                sd = float(np.std(sample_stats, ddof=1))
                iter_sds.append(sd)

        if len(iter_sds) > 0:
            results.append(
                {
                    "sample_size": size,
                    "standard_deviation": float(np.mean(iter_sds)),
                }
            )

    return pd.DataFrame(results)


def plot_variance_curve(results_df, graph_path: Path):
    # ensure proper ordering
    if "dataset" in results_df.columns:
        results_df = results_df.sort_values(["dataset", "sample_size"])
    else:
        results_df = results_df.sort_values(["sample_size"])

    plt.figure(figsize=(10, 6))

    if "dataset" in results_df.columns:
        # plot each dataset separately to control markers and colors
        for ds_name, subdf in results_df.groupby("dataset"):
            marker = MARKERS[ds_name]

            # lineplot for dataset
            sns.lineplot(
                data=subdf,
                x="sample_size",
                y="standard_deviation",
                marker=marker,
                label=ds_name,
            )
    else:
        sns.lineplot(
            data=results_df,
            x="sample_size",
            y="standard_deviation",
            marker="o",
        )

    plt.xlabel(r"\# Annotators")
    plt.ylabel("Std deviation of $pol_{obs.}$")
    plt.title("Robustness of $pol_{obs.}$ depends on the number of annotators")
    plt.grid(True)
    plt.tight_layout()

    tasks.graphs.save_plot(graph_path)
    plt.close()


def get_dataset_variance(
    dataset: tasks.preprocessing.Dataset,
    cache_dir: Path,
    min_comment_annotators: int,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{dataset.get_name()}_variance.csv"

    if cache_file.exists():
        print(
            f"Loading cached variance results for {dataset.get_name()}"
            f"from {cache_file}"
        )
        return pd.read_csv(cache_file)

    print(f"Computing variance results for {dataset.get_name()}...")
    res_df = sample_se_vs_sample_size_unimodality(
        df=dataset.get_dataset().reset_index(),
        annotation_col=dataset.get_annotation_column(),
        group_col="Gender",
        bins=5,
        min_size=3,
        max_size=None,
        step=1,
        iters=1000,
        min_comment_annotators=min_comment_annotators,
    )

    res_df.to_csv(cache_file, index=False)
    return res_df


def plot_annotator_count_histogram_from_datasets(
    datasets: list[tasks.preprocessing.Dataset],
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

    fig, ax = plt.subplots(figsize=(10, 6))

    # Determine bin boundaries
    min_val = all_df["n_annotators"].min()
    max_val = all_df["n_annotators"].max()
    bins_edges = np.linspace(min_val, max_val, N_BINS + 1)

    for i, dataset_name in enumerate(dataset_names):
        data_subset = all_df[all_df["dataset"] == dataset_name]["n_annotators"]

        raw_counts, edges = np.histogram(data_subset, bins=bins_edges)

        total_comments_for_dataset = len(data_subset)

        if total_comments_for_dataset > 0:
            percentage_counts = raw_counts / total_comments_for_dataset
        else:
            percentage_counts = np.zeros_like(raw_counts, dtype=float)

        selected_color = tasks.graphs.COLORBLIND_PALETTE[
            i % len(tasks.graphs.COLORBLIND_PALETTE)
        ]
        selected_hatch = tasks.graphs.HATCHES[i % len(tasks.graphs.HATCHES)]

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
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    tasks.graphs.save_plot(graph_path)
    plt.close()


def get_annotator_counts_df(
    datasets: list[tasks.preprocessing.Dataset],
) -> pd.DataFrame:
    rows = []

    for ds in datasets:
        df = ds.get_dataset().reset_index(drop=True)
        ann_col = ds.get_annotation_column()
        ds_name = ds.get_name()

        tmp = pd.DataFrame(
            {
                "dataset": ds_name,
                "n_annotators": df[ann_col].apply(_safe_len),
            }
        )
        rows.append(tmp)

    all_df = pd.concat(rows, ignore_index=True).dropna(subset=["n_annotators"])
    return all_df


def cull_kumar_ds(
    kumar_ds: tasks.preprocessing.Dataset,
) -> tasks.preprocessing.Dataset:
    # --- There is a single comment with 650 annotators ---
    df = kumar_ds.get_dataset()
    df["annotator_count"] = df["Toxicity"].apply(_safe_len)

    over_10_mask = df["annotator_count"] > 10

    if over_10_mask.any():
        over_10_df = pd.DataFrame(
            {
                "comment": df.index[over_10_mask],
                "annotator_count": df.loc[over_10_mask, "annotator_count"],
            }
        ).sort_values("annotator_count", ascending=False)
        print(f"#Comments with >10 annotators:{len(over_10_df)}")

    df = df.loc[~over_10_mask].drop(columns=["annotator_count"])
    kumar_ds.df = df
    return kumar_ds


def get_statistics_df(all_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a dataframe where each row corresponds to a dataset and each column
    is a statistic from `describe()` applied to annotator counts.
    """
    return (
        all_df.groupby("dataset")["n_annotators"]
        .describe()  # computes count, mean, std, min, 25%, 50%, 75%, max
        .rename_axis(index=None)  # optional: cleaner row index name
    )


def _safe_len(x):
    try:
        return len(x)
    except Exception:
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Create plots analyzing effect of #annotators.")
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
        "--sap-path",
        required=True,
        help="Path to the Sap annotator CSV file.",
    )
    parser.add_argument(
        "--kumar-path",
        required=True,
        help="Path to the Kumar annotator CSV file.",
    )
    parser.add_argument(
        "--graph-output-dir", required=True, help="Directory for the graphs."
    )
    parser.add_argument(
        "--latex-output-dir",
        required=True,
        help="Directory for the latex tables.",
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="Directory for cached variance computations.",
    )
    parser.add_argument(
        "--min-comment-annotators",
        type=int,
        default=3,
        help="Minimum annotators per comment to include in sampling.",
    )

    args = parser.parse_args()

    main(
        dices_small_path=Path(args.dices_small_path),
        dices_large_path=Path(args.dices_large_path),
        latex_output_dir=Path(args.latex_output_dir),
        sap_path=Path(args.sap_path),
        kumar_path=Path(args.kumar_path),
        graph_dir=Path(args.graph_output_dir),
        cache_dir=Path(args.cache_dir),
        min_comment_annotators=args.min_comment_annotators,
    )
