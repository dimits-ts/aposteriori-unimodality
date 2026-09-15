import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ..lib.preprocessing import KumarDataset
from ..lib.util import skip_if_exists
from ..lib import run_helper
from ..lib import graphs


# Seeds used to repeat the 3k-comment sample experiment with different
# random subsamples, to see how sensitive results are to which 3k
# comments happen to get selected.
KUMAR_3K_SEED_ABLATION_SEEDS = list(range(10))


def run_experiment(
    dataset_path: Path,
    output_path: Path,
    num_samples: int,
    seed: int = 42,
) -> None:
    if skip_if_exists(output_path):
        print(f"{output_path} exists, skipping...")
        return

    print(f"Running experiment {output_path}...")
    ds = KumarDataset(
        dataset_path=dataset_path, num_samples=num_samples, seed=seed
    )
    res = run_helper.run_all_results(ds)
    res.to_csv(output_path)


def run_seed_ablation_experiment(
    dataset_path: Path,
    ablation_dir: Path,
    num_samples: int = 3_000,
    seeds: list[int] = KUMAR_3K_SEED_ABLATION_SEEDS,
) -> None:
    """
    Repeats the num_samples-comment experiment `len(seeds)` times, each
    with a different random seed for the comment subsample, to gauge how
    sensitive results are to which comments get sampled. Each run is
    written to its own CSV under `ablation_dir`.
    """
    for seed in seeds:
        output_path = (
            ablation_dir / f"kumar{num_samples//1000}k-seed{seed}-results.csv"
        )
        run_experiment(
            dataset_path=dataset_path,
            output_path=output_path,
            num_samples=num_samples,
            seed=seed,
        )


def seed_ablation(
    ablation_dir: Path,
    dataset_prefix: str,
    seeds: list[int],
    output_path: Path,
) -> None:
    """
    Reads the per-seed ablation result CSVs written by
    run_seed_ablation_experiment and plots the mean Apunim value across
    seeds, with standard deviation error bars, for every individual
    subgroup, faceted by SDB dimension. Uses points rather than bars so
    the error bars (seed-to-seed variance) are the visual focus.
    """
    dfs = []
    for seed in seeds:
        df = pd.read_csv(
            ablation_dir / f"{dataset_prefix}-seed{seed}-results.csv",
            index_col=0,
        )
        df = df.rename_axis("dimension").reset_index()
        subgroup_col = df.columns[1]
        df = df.rename(columns={subgroup_col: "subgroup"})  # type: ignore
        dfs.append(df[["dimension", "subgroup", "apunim"]])

    combined = pd.concat(dfs, ignore_index=True)

    g = sns.catplot(
        data=combined,
        kind="point",
        x="subgroup",
        y="apunim",
        col="dimension",
        col_wrap=3,
        errorbar="sd",
        capsize=0.3,
        linestyle="none",
        color="C0",
        markers="o",
        sharex=False,
        height=4,
        aspect=1.5,
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("", "Apunim")
    g.set_xticklabels(rotation=90)
    g.refline(y=0, color="gray", linestyle="--", linewidth=1)
    g.figure.suptitle(
        f"Mean Apunim across {len(seeds)} seeds, by subgroup", y=1.02
    )
    g.figure.tight_layout()

    graphs.save_plot(output_path)
    plt.close(g.figure)


def main(
    dataset_path: Path,
    output_dir: Path,
    graph_output_dir: Path,
    ablations_dir: Path,
):
    graph_output_dir.mkdir(parents=True, exist_ok=True)
    graphs.graph_setup()

    print("Generating sample polarization plot...")
    graphs.polarization_plot(
        ds=KumarDataset(dataset_path=dataset_path, num_samples=3_000),
        output_path=graph_output_dir / "kumar_sample.png",
    )
    print("Calculating inherent polarization...")
    res = run_helper.compute_inherent_polarization_exhaustive(
        dataset=KumarDataset(dataset_path=dataset_path, num_samples=3_000),
        max_annotators=6,
    )
    res.to_csv(
        output_dir / "kumar-inherent.csv", header=True, index_label="comment"
    )

    run_experiment(
        dataset_path=dataset_path,
        output_path=output_dir / "kumar-results.csv",
        num_samples=3_000,
    )

    for sample_size in [30_000, 10_000, 1_000]:
        run_experiment(
            dataset_path=dataset_path,
            output_path=ablations_dir
            / f"kumar{sample_size // 1000}k-results.csv",
            num_samples=sample_size,
        )

    # New ablation: repeat the 3k-comment experiment across 10 different
    # seeds to measure sensitivity to which comments get sampled.
    run_seed_ablation_experiment(
        dataset_path=dataset_path,
        ablation_dir=ablations_dir,
        num_samples=3_000,
        seeds=KUMAR_3K_SEED_ABLATION_SEEDS,
    )

    # Figure: boxplots of the 10 seed runs, per subgroup.
    boxplot_path = graph_output_dir / "kumar3k_seed_ablation.png"
    seed_ablation(
        ablation_dir=ablations_dir,
        dataset_prefix="kumar3k",
        seeds=KUMAR_3K_SEED_ABLATION_SEEDS,
        output_path=boxplot_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Classify forum comments using taxonomy categories and an LLM."
        )
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to the full dataset CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for the CSV result files.",
    )
    parser.add_argument(
        "--graph-output-dir",
        required=True,
        help="Directory for graphs.",
    )
    parser.add_argument(
        "--ablation-dir",
        required=True,
        help=(
            "Directory for results derived from subsampled ('ablated') "
            "datasets: the sample-size ablations and the 10-seed repeated "
            "3k-comment ablation."
        ),
    )
    args = parser.parse_args()
    main(
        dataset_path=Path(args.dataset_path),
        output_dir=Path(args.output_dir),
        graph_output_dir=Path(args.graph_output_dir),
        ablations_dir=Path(args.ablation_dir),
    )
