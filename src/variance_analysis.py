# Below is a revised version of your script that adds caching for
# sample_se_vs_sample_size_unimodality results.
#
# New functionality:
# - A --cache-dir argument.
# - For each dataset, cached results are saved as CSV files inside cache_dir.
# - If a cache file exists, results are loaded instead of recomputed.

import typing
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from .tasks import graphs
from .tasks import preprocessing
from . import synthetic_100, dices
from .apunim import aposteriori


def sample_se_vs_sample_size_unimodality(
    df: pd.DataFrame,
    annotation_col: str,
    group_col: str,
    bins: int = 5,
    min_size: int = 2,
    max_size: int = 100,
    step: int = 10,
    iters: int = 30,
) -> pd.DataFrame:
    results: list[dict[str, typing.Any]] = []
    for size in tqdm(range(min_size, max_size + 1, step), desc="#Annotators"):
        iter_ses = []

        for _ in tqdm(range(iters), desc="#Comments", leave=False):
            sample_stats = []

            for _, row in df.iterrows():
                annotations = np.array(row[annotation_col])
                groups = np.array(row[group_col])
                if len(annotations) < size:
                    continue

                idx = np.random.choice(
                    len(annotations), size=size, replace=False
                )
                sub_ann = annotations[idx]
                sub_grp = groups[idx]

                stats_dict = aposteriori._factor_dfu_stat(
                    sub_ann, sub_grp, bins=bins
                )
                sample_stats.extend(
                    [v for v in stats_dict.values() if not np.isnan(v)]
                )

            if len(sample_stats) > 1:
                se = np.std(sample_stats, ddof=1) / np.sqrt(len(sample_stats))
                iter_ses.append(se)

        if len(iter_ses) > 0:
            results.append(
                {"sample_size": size, "standard_error": np.mean(iter_ses)}
            )

    return pd.DataFrame(results)


def plot_variance_curve(results_df: pd.DataFrame, graph_path: Path) -> None:
    results_df = results_df.sort_values(["dataset", "sample_size"])
    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=results_df,
        x="sample_size",
        y="standard_error",
        hue="dataset",
        marker="o",
    )

    for ds_name, subdf in results_df.groupby("dataset"):
        sns.regplot(
            data=subdf,
            x="sample_size",
            y="standard_error",
            scatter=False,
            label=f"{ds_name} trend",
            ci=None,
        )

    plt.xlabel("# Annotators")
    plt.ylabel("Std Error of Polarization Statistic")
    plt.title("Sample Size Effects on Polarization Statistic Estimation")
    plt.grid(True)
    plt.tight_layout()

    graphs.save_plot(graph_path)
    plt.close()


def get_dataset_variance(
    dataset: preprocessing.Dataset, cache_dir: Path
) -> pd.DataFrame:
    """Load results from cache if available, otherwise compute and store."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{dataset.get_name()}_variance.csv"

    if cache_file.exists():
        print(
            f"Loading cached variance results for {dataset.get_name()} "
            f"from {cache_file}"
        )
        return pd.read_csv(cache_file)
    else:
        print(f"Computing variance results for {dataset.get_name()}...")
        res_df = sample_se_vs_sample_size_unimodality(
            df=dataset.get_dataset().reset_index(),
            annotation_col=dataset.get_annotation_column(),
            group_col="Gender",
            bins=5,
            min_size=3,
            max_size=100,
            step=1,
            iters=1000,
        )

        res_df.to_csv(cache_file, index=False)
        return res_df


def main(
    hundred_dataset_path: Path,
    dices_small_path: Path,
    dices_large_path: Path,
    graph_dir: Path,
    cache_dir: Path,
):

    ds_hundred = synthetic_100.HundredDataset(
        dataset_path=hundred_dataset_path
    )
    dices350 = dices.DicesDataset(
        dataset_path=dices_small_path, variant="dices-350"
    )
    dices990 = dices.DicesDataset(
        dataset_path=dices_large_path, variant="dices-990"
    )

    variance_df_ls = []

    for dataset in [ds_hundred, dices350, dices990]:
        res_df = get_dataset_variance(dataset, cache_dir)
        res_df["dataset"] = dataset.get_name()
        variance_df_ls.append(res_df)

    variance_df = pd.concat(variance_df_ls, ignore_index=True)

    plot_variance_curve(
        variance_df.loc[
            variance_df.dataset.isin(
                ["dices-350", "dices-990"],
            )
        ],
        graph_path=graph_dir / "ndfu_std_error_sample_size.png",
    )

    plot_variance_curve(
        variance_df,
        graph_path=graph_dir / "ndfu_std_error_sample_size_llm.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Create plots analyzing effect of #annotators.")
    )
    parser.add_argument(
        "--hundred-dataset-path",
        required=True,
        help="Path to the 100 annotator CSV file.",
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

    args = parser.parse_args()

    main(
        hundred_dataset_path=Path(args.hundred_dataset_path),
        dices_small_path=Path(args.dices_small_path),
        dices_large_path=Path(args.dices_large_path),
        graph_dir=Path(args.graph_output_dir),
        cache_dir=Path(args.cache_dir),
    )
