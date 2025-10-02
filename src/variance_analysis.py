import typing
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm

from .tasks import graphs
from . import synthetic_100
from .apunim import aposteriori


def sample_se_vs_sample_size_unimodality(
    df: pd.DataFrame,
    annotation_col: str,
    group_col: str,
    comment_col: str,
    bins: int = 5,
    min_size: int = 2,
    max_size: int = 100,
    step: int = 10,
    iters: int = 30,
) -> pd.DataFrame:
    """
    Sample decreasing subsets of annotations and compute the standard error
    of Aposteriori Unimodality statistics at each sample size.

    :param df: DataFrame with annotations, factor groups, and comment ids.
    :param annotation_col: Column name for list of annotations.
    :param group_col: Column name for list of factor groups (e.g., annot_sex).
    :param comment_col: Column name for list of comment ids.
    :param bins: Number of bins for DFU.
    :param min_size: Minimum sample size to consider.
    :param max_size: Maximum sample size to consider.
    :param step: Step size to decrease sample size.
    :param iters: Repetitions per sample size for averaging.
    :return: DataFrame with columns [sample_size, standard_error]
    """
    results: list[dict[str, typing.Any]] = []
    for size in tqdm(range(min_size, max_size + 1, step), desc="#Annotators"):
        iter_ses = []

        for _ in tqdm(range(iters), desc="#Comments", leave=False):
            sample_stats = []

            # loop over comments
            for _, row in df.iterrows():
                annotations = np.array(row[annotation_col])
                groups = np.array(row[group_col])

                # skip if not enough annotators
                if len(annotations) < size:
                    continue

                # subsample
                idx = np.random.choice(
                    len(annotations), size=size, replace=False
                )
                sub_ann = annotations[idx]
                sub_grp = groups[idx]

                # compute factor DFU stats
                stats_dict = aposteriori._factor_dfu_stat(
                    sub_ann, sub_grp, bins=bins
                )

                # collect values (drop NaNs)
                sample_stats.extend(
                    [v for v in stats_dict.values() if not np.isnan(v)]
                )

            # compute SE for this iteration
            if len(sample_stats) > 1:
                se = np.std(sample_stats, ddof=1) / np.sqrt(len(sample_stats))
                iter_ses.append(se)

        # average SE across iterations
        if len(iter_ses) > 0:
            results.append(
                {"sample_size": size, "standard_error": np.mean(iter_ses)}
            )

    return pd.DataFrame(results)


def plot_variance_curve(results_df: pd.DataFrame, graph_path: Path) -> None:
    results_df = results_df.sort_values("sample_size")

    x = results_df["sample_size"].values.reshape(-1, 1)
    y = results_df["standard_error"].values

    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)

    plt.figure(figsize=(8, 5))
    plt.plot(
        x,
        y_pred,
        label=f"Linear Fit: y={model.coef_[0]:.4f}x + {model.intercept_:.4f}",
        color="tab:orange",
    )
    plt.plot(x, y, color="tab:blue", marker="o")
    plt.xlabel("#Annotators")
    plt.ylabel("Std error of Polarization Statistic")
    plt.title("Sample size affects pol-statistic estimation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    graphs.save_plot(graph_path)
    plt.close()


def main(dataset_path: Path, graph_dir: Path):
    df = synthetic_100.base_df(dataset_path=dataset_path)
    res_df = sample_se_vs_sample_size_unimodality(
        df=df.reset_index(),
        annotation_col="toxicity",
        group_col="annot_sex",
        comment_col="message_id",
        bins=5,
        min_size=3,
        max_size=100,
        step=1,
        iters=1000,
    )
    plot_variance_curve(
        res_df, graph_path=graph_dir / "ndfu_std_error_sample_size.png"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Create plots analyzing effect of #annotators.")
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to the 100 annotator CSV file.",
    )
    parser.add_argument(
        "--graph-dir",
        required=True,
        help="Directory for the graphs.",
    )
    args = parser.parse_args()
    main(dataset_path=Path(args.dataset_path), graph_dir=Path(args.graph_dir))
