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

from .tasks import graphs
from .tasks import preprocessing
from . import dices


MARKERS = {
    "DICES-350": "o",
    "DICES-990": "s",
}


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
        iter_ses: list[float] = []

        for _ in tqdm(range(iters), desc="#Iterations", leave=False):
            sample_stats: list[float] = []

            # loop over comments
            for _, row in df.iterrows():
                anns = row[annotation_col]
                grps = row[group_col]

                # skip comments with too few annotators
                if anns is None:
                    continue
                try:
                    n_ann = len(anns)
                except Exception:
                    # if annotations are stored differently, skip
                    continue

                if n_ann < min_comment_annotators:
                    continue

                # skip if not enough annotators for this sample size
                if n_ann < size:
                    continue

                annotations = np.array(anns)
                groups = np.array(grps)

                # subsample
                idx = np.random.choice(n_ann, size=size, replace=False)
                sub_ann = annotations[idx]
                sub_grp = groups[idx]

                # compute factor DFU stats
                stats_dict = apunim._factor_dfu_stat(
                    sub_ann, sub_grp, bins=bins
                )

                # collect values (drop NaNs)
                sample_stats.extend(
                    [float(v) for v in stats_dict.values() if not np.isnan(v)]
                )

            # compute SE for this iteration
            if len(sample_stats) > 1:
                se = float(
                    np.std(sample_stats, ddof=1) / np.sqrt(len(sample_stats))
                )
                iter_ses.append(se)

        # average SE across iterations
        if len(iter_ses) > 0:
            results.append(
                {
                    "sample_size": size,
                    "standard_error": float(np.mean(iter_ses)),
                }
            )

    return pd.DataFrame(results)


def plot_variance_curve(results_df, graph_path: Path):
    # Ensure proper ordering
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
            ax = sns.lineplot(
                data=subdf,
                x="sample_size",
                y="standard_error",
                marker=marker,
                label=ds_name,
            )

            # regression trend line (same color but no scatter)
            sns.regplot(
                data=subdf,
                x="sample_size",
                y="standard_error",
                scatter=False,
                ci=None,
                color=ax.lines[-1].get_color(),  # match line color
                label=None,
            )
    else:
        sns.lineplot(
            data=results_df, x="sample_size", y="standard_error", marker="o"
        )

    plt.xlabel(r"\# Annotators")
    plt.ylabel("Std Error of Polarization Statistic")
    plt.title("Sample Size Effects on Polarization Statistic Estimation")
    plt.grid(True)
    plt.tight_layout()

    graphs.save_plot(graph_path)
    plt.close()


def get_dataset_variance(
    dataset: preprocessing.Dataset,
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
        max_size=None,  # let the function determine the proper max
        step=1,
        iters=1000,
        min_comment_annotators=min_comment_annotators,
    )

    res_df.to_csv(cache_file, index=False)
    return res_df


def main(
    dices_small_path: Path,
    dices_large_path: Path,
    graph_dir: Path,
    cache_dir: Path,
    min_comment_annotators: int = 3,
):
    graphs.graph_setup()
    dices350 = dices.DicesDataset(dataset_path=dices_small_path, variant="350")
    dices990 = dices.DicesDataset(dataset_path=dices_large_path, variant="990")

    variance_df_ls = []

    for dataset in [dices350, dices990]:
        res_df = get_dataset_variance(
            dataset, cache_dir, min_comment_annotators=min_comment_annotators
        )
        res_df["dataset"] = dataset.get_name()
        variance_df_ls.append(res_df)

    variance_df = pd.concat(variance_df_ls, ignore_index=True)

    plot_variance_curve(
        variance_df,
        graph_path=graph_dir / "ndfu_std_error_sample_size_llm.png",
    )


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
        "--graph-output-dir", required=True, help="Directory for the graphs."
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
        graph_dir=Path(args.graph_output_dir),
        cache_dir=Path(args.cache_dir),
        min_comment_annotators=args.min_comment_annotators,
    )
