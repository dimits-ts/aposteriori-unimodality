import argparse
from pathlib import Path

from ..lib import graphs, run_helper
from ..lib.preprocessing import DicesDataset
from ..lib.util import skip_if_exists

# Fixed-size resampled experiment (see run_resampled_experiment).
RESAMPLED_SIZE = 5
RESAMPLED_RUNS = 10

# Repeats of the inherent-polarization experiment with every comment
# subsampled down to run_helper.INHERENT_SUBSAMPLE_SIZE (6) annotators.
INHERENT_SUBSAMPLE_REPEATS = 100


def run_resampled_experiment(
    ds: DicesDataset,
    ablation_dir: Path,
    sample_size: int = RESAMPLED_SIZE,
    n_runs: int = RESAMPLED_RUNS,
) -> None:
    """
    For each SDB factor, resample `sample_size` annotators per comment (with
    replacement), `n_runs` times, and save the mean apunim value with its
    standard deviation (in place of kappa/pvalue). Output is written under
    `ablation_dir` since this operates on a subsampled ("ablated") view of
    the dataset.
    """
    output_path = (
        ablation_dir
        / f"{ds.get_name().lower()}-results-resampled-n{sample_size}.csv"
    )
    if skip_if_exists(output_path):
        return

    res = run_helper.run_all_results_resampled(
        ds=ds, sample_size=sample_size, n_runs=n_runs
    )
    res.to_csv(output_path)


def run_for_dataset(
    ds: DicesDataset,
    name: str,
    output_dir: Path,
    graph_output_dir: Path,
    ablation_dir: Path,
    cache_dir: Path,
) -> None:
    """Every result of one DICES variant; each output is skipped if it
    already exists."""
    graph_path = graph_output_dir / f"{name}.png"
    if not skip_if_exists(graph_path):
        graphs.polarization_plot(ds=ds, output_path=graph_path)

    # Inherent polarization (exhaustive) on all annotators, and with every
    # comment subsampled down to 6 annotators.
    inherent_path = output_dir / f"{name}-inherent.csv"
    if not skip_if_exists(inherent_path):
        res = run_helper.compute_inherent_polarization_exhaustive(ds)
        res.to_csv(inherent_path, header=True, index_label="comment")
    run_helper.write_inherent_polarization_subsampled(
        ds, output_dir, name=name, n_repeats=INHERENT_SUBSAMPLE_REPEATS
    )

    ndfu_path = run_helper.human_ndfu_path(cache_dir, name)
    if not skip_if_exists(ndfu_path):
        run_helper.write_human_ndfu(ds, ndfu_path)

    results_path = output_dir / f"{name}-results.csv"
    if not skip_if_exists(results_path):
        res = run_helper.run_all_results(ds=ds)
        res.to_csv(results_path)

    run_resampled_experiment(ds, ablation_dir)


def main(
    dataset_path_small: Path,
    dataset_path_large: Path,
    output_dir: Path,
    graph_output_dir: Path,
    ablation_dir: Path,
    cache_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_output_dir.mkdir(parents=True, exist_ok=True)
    ablation_dir.mkdir(parents=True, exist_ok=True)

    graphs.graph_setup()
    for path, variant in [
        (dataset_path_small, "350"),
        (dataset_path_large, "990"),
    ]:
        run_for_dataset(
            DicesDataset(dataset_path=path, variant=variant),
            name=f"dices-{variant}",
            output_dir=output_dir,
            graph_output_dir=graph_output_dir,
            ablation_dir=ablation_dir,
            cache_dir=cache_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Classify forum comments using taxonomy categories and an LLM."
        )
    )
    parser.add_argument(
        "--dataset-small-path",
        required=True,
        help="Path to the DICES-350 CSV file.",
    )
    parser.add_argument(
        "--dataset-large-path",
        required=True,
        help="Path to the DICES-990 CSV file.",
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
            "Directory for the fixed-size (N=5, 10-run) resampled "
            "experiment, which operates on a subsampled dataset."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="Directory for cached computations (e.g. human nDFU).",
    )
    args = parser.parse_args()
    main(
        dataset_path_small=Path(args.dataset_small_path),
        dataset_path_large=Path(args.dataset_large_path),
        output_dir=Path(args.output_dir),
        graph_output_dir=Path(args.graph_output_dir),
        ablation_dir=Path(args.ablation_dir),
        cache_dir=Path(args.cache_dir),
    )
