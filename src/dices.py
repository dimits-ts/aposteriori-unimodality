import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import tasks.graphs
import tasks.preprocessing
import tasks.run_helper

SAMPLE_SIZES = range(5, 51, 2)
N_RUNS = 10


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
                "gen x+": "1) Gen. X+",
                "millenial": "2) Millennial",
                "gen z": "3) Gen. Z",
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


def subsample_dataset(
    ds: DicesDataset, size: int, rng: np.random.Generator
) -> DicesDataset:
    """
    Return a copy of the dataset with each comment subsampled to `size`
    annotators (with replacement). All columns are sampled with the same
    indices to preserve per-annotator alignment across columns.
    """
    df = ds.get_dataset().copy()
    annotation_col = ds.get_annotation_column()
    cols = ds.get_sdb_columns() + [annotation_col]

    # Sample indices once per row so all columns stay aligned
    row_indices = [
        rng.choice(len(values), size=size, replace=True)
        for values in df[annotation_col]
    ]

    for col in cols:
        df[col] = [
            [row[i] for i in indices]
            for row, indices in zip(df[col], row_indices)
        ]

    subsampled = object.__new__(DicesDataset)
    subsampled.df = df
    subsampled.variant = ds.variant
    return subsampled


def run_for_dataset(
    ds: DicesDataset, sample_sizes: range, seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for size in tqdm(sample_sizes, desc=f"Sample sizes for {ds.get_name()}"):
        run_means = []
        for _ in range(N_RUNS):
            subsampled_ds = subsample_dataset(ds, size, rng)
            result = tasks.run_helper.compute_inherent_polarization_random(
                subsampled_ds, seed=None  # do not produce identical runs
            )
            print(f"#Annotators: {size}, result: {result}")
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


def plot_sample_size_polarization(csv_path: Path, output_path: Path):
    df = pd.read_csv(csv_path)

    _, ax = plt.subplots(figsize=(8, 5))

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
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main(
    dataset_path_small: Path,
    dataset_path_large: Path,
    output_dir: Path,
    graph_output_dir: Path,
):
    tasks.graphs.graph_setup()
    ds_350 = DicesDataset(dataset_path=dataset_path_small, variant="350")
    tasks.graphs.polarization_plot(
        ds=ds_350, output_path=graph_output_dir / "dices-350.png"
    )

    res = tasks.run_helper.compute_inherent_polarization_random(ds_350)
    np.save(output_dir / "dices-350-apriori.npy", res)

    #res = tasks.run_helper.run_all_results(ds=ds_350)
    #res.to_csv(output_dir / "dices-350.csv")

    ds_990 = DicesDataset(dataset_path=dataset_path_large, variant="990")
    tasks.graphs.polarization_plot(
        ds=ds_990, output_path=graph_output_dir / "dices-990.png"
    )

    res = tasks.run_helper.compute_inherent_polarization_random(ds_990)
    np.save(output_dir / "dices-990-apriori.npy", res)

    #res = tasks.run_helper.run_all_results(ds=ds_990)
    #res.to_csv(output_dir / "dices-990.csv")

    df_350 = run_for_dataset(ds_350, SAMPLE_SIZES)
    df_990 = run_for_dataset(ds_990, SAMPLE_SIZES)

    combined = pd.concat([df_350, df_990], ignore_index=True)
    csv_path = output_dir / "sample_size_polarization.csv"
    combined.to_csv(csv_path, index=False)

    plot_sample_size_polarization(
        csv_path=csv_path,
        output_path=graph_output_dir / "sample_size_polarization.png",
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
    args = parser.parse_args()
    main(
        dataset_path_small=Path(args.dataset_small_path),
        dataset_path_large=Path(args.dataset_large_path),
        output_dir=Path(args.output_dir),
        graph_output_dir=Path(args.graph_output_dir),
    )
