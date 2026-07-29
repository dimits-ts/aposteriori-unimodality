import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import tasks.graphs
import tasks.preprocessing
import tasks.run_helper

SAMPLE_SIZES = range(5, 51, 2)
N_RUNS = 10

# Config for the new fixed-size resampled experiment
RESAMPLED_SIZE = 5
RESAMPLED_RUNS = 10


class DicesDataset(tasks.preprocessing.Dataset):
    def __init__(self, dataset_path: Path, variant: str):
        self.df = DicesDataset._base_df(dataset_path)
        self.variant = variant

    def get_name(self) -> str:
        return "DICES-" + self.variant

    def get_dataset(self) -> pd.DataFrame:
        return self.df

    def get_sdb_columns(self) -> list[str]:
        return ["Gender", "Race", "Age", "Education"]

    def get_comment_key_column(self) -> str:
        return "item_id"

    def get_annotation_column(self) -> str:
        return "is_harmful"

    @staticmethod
    def _base_df(dataset_path: Path) -> pd.DataFrame:
        df = pd.read_csv(dataset_path)

        if "Q3_bias_overall" not in df.columns:
            df = df.rename(
                {"Q3_unfair_bias_overall": "Q3_bias_overall"}, axis=1
            )

        df = df.loc[
            :,
            [
                "rater_gender",
                "rater_age",
                "rater_race",
                "rater_education",
                "Q3_bias_overall",
                "item_id",
            ],
        ]
        df.Q3_bias_overall = df.Q3_bias_overall.map(
            {"No": -1, "Unsure": 0, "Yes": 1}
        ).astype(int)

        df = df.replace(
            {
                "College degree or higher": "College +",
                "High school or below": "High school -",
            }
        )
        df = df.replace(
            {
                "Asian/Asian subcontinent": "Asian",
                "Black/African American": "African Am.",
                "LatinX, Latino, Hispanic or Spanish Origin": "Latino",
                "Self-describe (below)": "Other",
            }
        )
        # add numbers for proper ordering during export
        df = df.replace(
            {
                "gen x+": "3) Gen. X+",
                "millenial": "2) Millennial",
                "gen z": "1) Gen. Z",
            }
        )

        df = df.groupby("item_id").agg(list).reset_index()
        df = df.rename(
            columns={
                "rater_gender": "Gender",
                "rater_age": "Age",
                "rater_race": "Race",
                "rater_education": "Education",
                "Q3_bias_overall": "is_harmful",
            }
        )
        return df


def _skip_if_exists(path: Path) -> bool:
    """
    Returns True (and prints a message) if `path` already exists, so the
    caller can skip recomputing it. Centralized here so every experiment
    step uses the same check/logging behavior.
    """
    if path.exists():
        print(f"Skipping (already exists): {path}")
        return True
    return False


def run_for_dataset(
    ds: DicesDataset, sample_sizes: range, seed: int = 42
) -> pd.DataFrame:
    """
    Original sample-size sweep experiment: for each size in `sample_sizes`,
    runs N_RUNS resamples and records the mean/std of inherent polarization
    across runs.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for size in tqdm(sample_sizes, desc=f"Sample sizes for {ds.get_name()}"):
        run_means = []
        for _ in range(N_RUNS):
            subsampled_ds = tasks.run_helper.subsample_dataset(ds, size, rng)
            result = tasks.run_helper.compute_inherent_polarization_random(
                subsampled_ds
            )
            run_means.append(np.mean(result))
        rows.append(
            {
                "dataset": ds.get_name(),
                "sample_size": size,
                "mean": np.mean(run_means),
                "std": np.std(run_means),
            }
        )
    return pd.DataFrame(rows)


def run_resampled_experiment(
    ds: DicesDataset,
    ablation_dir: Path,
    sample_size: int = RESAMPLED_SIZE,
    n_runs: int = RESAMPLED_RUNS,
) -> None:
    """
    New experiment: for each SDB factor, resample `sample_size` annotators
    per comment (with replacement), `n_runs` times, and save the mean apunim
    value with its standard deviation (in place of kappa/pvalue). Output is
    written under `ablation_dir` since this operates on a subsampled
    ("ablated") view of the dataset.
    """
    output_path = (
        ablation_dir
        / f"{ds.get_name().lower()}-results-resampled-n{sample_size}.csv"
    )
    if _skip_if_exists(output_path):
        return

    res = tasks.run_helper.run_all_results_resampled(
        ds=ds, sample_size=sample_size, n_runs=n_runs
    )
    res.to_csv(output_path)


def main(
    dataset_path_small: Path,
    dataset_path_large: Path,
    output_dir: Path,
    graph_output_dir: Path,
    ablation_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_output_dir.mkdir(parents=True, exist_ok=True)
    ablation_dir.mkdir(parents=True, exist_ok=True)

    tasks.graphs.graph_setup()
    ds_350 = DicesDataset(dataset_path=dataset_path_small, variant="350")

    graph_path = graph_output_dir / "dices-350.png"
    if not _skip_if_exists(graph_path):
        tasks.graphs.polarization_plot(ds=ds_350, output_path=graph_path)

    inherent_path = output_dir / "dices-350-inherent.csv"
    if not _skip_if_exists(inherent_path):
        res = tasks.run_helper.compute_inherent_polarization_random(ds_350)
        res.to_csv(inherent_path, header=True, index_label="comment")

    results_path = output_dir / "dices-350-results.csv"
    if not _skip_if_exists(results_path):
        res = tasks.run_helper.run_all_results(ds=ds_350)
        res.to_csv(results_path)

    ds_990 = DicesDataset(dataset_path=dataset_path_large, variant="990")

    graph_path = graph_output_dir / "dices-990.png"
    if not _skip_if_exists(graph_path):
        tasks.graphs.polarization_plot(ds=ds_990, output_path=graph_path)

    inherent_path = output_dir / "dices-990-inherent.csv"
    if not _skip_if_exists(inherent_path):
        res = tasks.run_helper.compute_inherent_polarization_random(ds_990)
        res.to_csv(inherent_path, header=True, index_label="comment")

    results_path = output_dir / "dices-990-results.csv"
    if not _skip_if_exists(results_path):
        res = tasks.run_helper.run_all_results(ds=ds_990)
        res.to_csv(results_path)

    # Sample-size sweep -> back to original behavior: computed together and
    # written once to output_dir, skipped entirely if it already exists.
    combined_path = output_dir / "sample_size_polarization.csv"
    if not _skip_if_exists(combined_path):
        df_350 = run_for_dataset(ds_350, SAMPLE_SIZES)
        df_990 = run_for_dataset(ds_990, SAMPLE_SIZES)
        combined = pd.concat([df_350, df_990], ignore_index=True)
        combined.to_csv(combined_path, index=False)

    # New, separate experiment: fixed-size (5 annotators) x 10 runs,
    # reporting mean apunim +/- std instead of kappa/pvalue. Operates on a
    # subsampled ("ablated") dataset, so it's written to ablation_dir.
    run_resampled_experiment(ds_350, ablation_dir)
    run_resampled_experiment(ds_990, ablation_dir)


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
    args = parser.parse_args()
    main(
        dataset_path_small=Path(args.dataset_small_path),
        dataset_path_large=Path(args.dataset_large_path),
        output_dir=Path(args.output_dir),
        graph_output_dir=Path(args.graph_output_dir),
        ablation_dir=Path(args.ablation_dir),
    )
