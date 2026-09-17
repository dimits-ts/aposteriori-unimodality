import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ..lib.preprocessing import KumarDataset, LazyDatasetLoader
from ..lib.util import skip_if_exists
from ..lib import run_helper
from ..lib import graphs


def main(
    dataset_path: Path,
    output_dir: Path,
    graph_output_dir: Path,
    ablations_dir: Path,
    latex_output_dir: Path,
):
    KUMAR_SEED_ABLATION_SEEDS = list(range(10))
    SEED = 42
    graph_output_dir.mkdir(parents=True, exist_ok=True)
    graphs.graph_setup()

    main_analysis(
        dataset_path=dataset_path,
        graph_output_dir=graph_output_dir,
        res_output_dir=output_dir,
        num_samples=1_000,
        seed=SEED,
    )

    for sample_size in [30_000, 10_000, 3_000, 1_000]:
        _run_experiment(
            dataset_path=dataset_path,
            output_path=ablations_dir
            / f"kumar{sample_size // 1000}k-results.csv",
            num_samples=sample_size,
            seed=SEED,
        )

    run_seed_ablation_experiment(
        dataset_path=dataset_path,
        ablation_dir=ablations_dir,
        num_samples=1_000,
        seeds=KUMAR_SEED_ABLATION_SEEDS,
    )

    _seed_ablation(
        ablation_dir=ablations_dir,
        dataset_prefix="kumar1k",
        seeds=KUMAR_SEED_ABLATION_SEEDS,
        output_path=graph_output_dir / "kumar1k_seed_ablation.png",
    )

    export_latex_sample_size_table(
        ablations_dir=ablations_dir,
        output_path=latex_output_dir / "kumar_sample_size_comparison.tex",
        caption=(
            "Comparison of Apunim values across different sample sizes "
            "(1k, 3k, 10k, 30k) for the Kumar dataset."
        ),
        label=r"tab:num-comments-apunim",
        sample_sizes=[1_000, 3_000, 10_000, 30_000],
    )


def main_analysis(
    dataset_path: Path,
    graph_output_dir: Path,
    res_output_dir: Path,
    num_samples: int,
    seed: int,
):
    # Use a lazy loader so KumarDataset is only constructed if at least one
    # of the three sub-steps below actually needs it.  Previously the
    # dataset was loaded unconditionally before any skip_if_exists check.
    loader = LazyDatasetLoader(
        lambda: KumarDataset(
            dataset_path=dataset_path, num_samples=num_samples, seed=seed
        )
    )

    _polarization_plot(
        loader=loader,
        output_path=graph_output_dir / "kumar_sample.png",
    )
    _inherent_experiment(
        loader=loader,
        res_output_dir=res_output_dir,
    )
    _main_experiment(
        dataset_path=dataset_path,
        res_output_dir=res_output_dir,
        num_samples=num_samples,
        seed=seed,
    )


def run_seed_ablation_experiment(
    dataset_path: Path,
    ablation_dir: Path,
    seeds: list[int],
    num_samples: int = 1_000,
) -> None:
    """
    Repeats the num_samples-comment experiment ``len(seeds)`` times, each
    with a different random seed for the comment subsample, to gauge how
    sensitive results are to which comments get sampled. Each run is
    written to its own CSV under ``ablation_dir``.
    """
    for seed in seeds:
        output_path = (
            ablation_dir
            / f"kumar{num_samples // 1000}k-seed{seed}-results.csv"
        )
        _run_experiment(
            dataset_path=dataset_path,
            output_path=output_path,
            num_samples=num_samples,
            seed=seed,
        )


def export_latex_sample_size_table(
    ablations_dir: Path,
    output_path: Path,
    caption: str,
    label: str,
    sample_sizes: list[int],
    decimals: int = 3,
) -> None:
    frames: dict[int, pd.DataFrame] = {}
    for n in sample_sizes:
        p = ablations_dir / f"kumar{n // 1000}k-results.csv"
        if not p.exists():
            raise FileNotFoundError(
                f"Expected results file not found: {p}\n"
                f"Run the experiment for sample size {n} first."
            )
        frames[n] = _load_results(p).set_index(["dimension", "subgroup"])

    # Pivot: rows = (dimension, subgroup), columns = sample sizes.
    combined = pd.DataFrame({n: df["apunim"] for n, df in frames.items()})
    combined.columns = [r"\textbf{" + str(n) + "}" for n in combined.columns]
    combined.index.names = [None, r"\textbf{\ac{pc}}"]

    latex = combined.to_latex(
        longtable=True,
        multirow=True,
        na_rep="---",
        float_format=f"{{:.{decimals}f}}".format,
        caption=caption,
        label=label,
        escape=False,
        column_format="ll" + "r" * len(sample_sizes),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex, encoding="utf-8")
    print(f"Ablations table written to: {output_path}")


def _polarization_plot(
    loader: LazyDatasetLoader,
    output_path: Path,
) -> None:
    if skip_if_exists(output_path):
        return

    print("Generating sample polarization plot...")
    graphs.polarization_plot(
        ds=loader.get(),
        output_path=output_path,
    )


def _inherent_experiment(
    loader: LazyDatasetLoader,
    res_output_dir: Path,
) -> None:
    inherent_path = res_output_dir / "kumar-inherent.csv"
    if skip_if_exists(inherent_path):
        return

    print("Calculating inherent polarization...")
    res = run_helper.compute_inherent_polarization_exhaustive(
        dataset=loader.get(),
        max_annotators=6,
    )
    res.to_csv(
        inherent_path,
        header=True,
        index_label="comment",
    )


def _main_experiment(
    dataset_path: Path, res_output_dir: Path, num_samples: int, seed: int
) -> None:
    main_res_path = res_output_dir / "kumar-results.csv"
    if skip_if_exists(main_res_path):
        return

    _run_experiment(
        dataset_path=dataset_path,
        output_path=main_res_path,
        num_samples=num_samples,
        seed=seed,
    )


def _run_experiment(
    dataset_path: Path,
    output_path: Path,
    num_samples: int,
    seed: int,
) -> None:
    if skip_if_exists(output_path):
        return

    print(f"Running experiment {output_path}...")
    ds = KumarDataset(
        dataset_path=dataset_path, num_samples=num_samples, seed=seed
    )
    res = run_helper.run_all_results(ds)
    res.to_csv(output_path)


def _seed_ablation(
    ablation_dir: Path,
    dataset_prefix: str,
    seeds: list[int],
    output_path: Path,
) -> None:
    """
    Reads the per-seed ablation result CSVs written by
    run_seed_ablation_experiment and plots the mean Apunim value across
    seeds, with standard deviation error bars, for every individual
    subgroup, faceted by SDB dimension.
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


def _load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df = df.rename_axis("dimension").reset_index()
    subgroup_col = df.columns[1]
    df = df.rename(columns={subgroup_col: "subgroup"})
    return df[["dimension", "subgroup", "apunim"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Classify forum comments using taxonomy categories and an LLM."
        )
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--graph-output-dir", required=True)
    parser.add_argument("--ablation-dir", required=True)
    parser.add_argument("--latex-output-dir", required=True)
    args = parser.parse_args()
    main(
        dataset_path=Path(args.dataset_path),
        output_dir=Path(args.output_dir),
        graph_output_dir=Path(args.graph_output_dir),
        ablations_dir=Path(args.ablation_dir),
        latex_output_dir=Path(args.latex_output_dir),
    )
