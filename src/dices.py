import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .tasks import preprocessing
from .tasks import run_helper
from .tasks import graphs


class DicesDataset(preprocessing.Dataset):
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
        df = df.loc[
            :,
            [
                "rater_gender",
                "rater_age",
                "rater_race",
                "rater_education",
                "Q_overall",
                "item_id",
            ],
        ]
        df.Q_overall = df.Q_overall.map(
            {"No": -1, "Unsure": "0", "Yes": 1}
        ).astype(int)

        df = df.replace(
            {
                "Asian/Asian subcontinent": "Asian",
                "Black/African American": "African American",
                "LatinX, Latino, Hispanic or Spanish Origin": "Latino",
                "Self-describe (below)": "Other",
            }
        )
        # add numbers for proper ordering during export
        df = df.replace(
            {
                "gen x+": "1) Gen. X+",
                "millenial": "2) Millenial",
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
                "Q_overall": "is_harmful",
            }
        )
        return df


def plot_annotation_histograms(
    dataset_path_small: Path, dataset_path_large: Path, output_path: Path
):
    """
    Create a plot with two seaborn histograms showing the distribution
    of the number of annotators per comment for DICES-350 and DICES-990.
    """

    # Load datasets
    df_small = pd.read_csv(dataset_path_small)
    df_large = pd.read_csv(dataset_path_large)

    # Compute annotator counts per item_id
    counts_small = df_small.groupby("item_id").size().reset_index(name="count")
    counts_large = df_large.groupby("item_id").size().reset_index(name="count")

    counts_small["dataset"] = "DICES-350"
    counts_large["dataset"] = "DICES-990"

    # Combine for seaborn
    combined = pd.concat([counts_small, counts_large], ignore_index=True)

    # Plot
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=combined,
        x="count",
        hue="dataset",
        element="step",
        stat="count",
        common_bins=True,
        alpha=0.5,
        bins=120
    )

    plt.xlabel("Number of annotators per comment")
    plt.ylabel("Count of comments")
    plt.title("Annotator Count Distribution")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main(
    dataset_path_small: Path,
    dataset_path_large: Path,
    output_dir: Path,
    graph_output_dir: Path,
):
    ds_350 = DicesDataset(dataset_path=dataset_path_small, variant="350")
    graphs.polarization_plot(
        ds=ds_350, output_path=graph_output_dir / "dices-350.png"
    )
    res = run_helper.run_all_results(
        df=ds_350.get_dataset(),
        sdb_columns=ds_350.get_sdb_columns(),
        value_col=ds_350.get_annotation_column(),
        comment_key_col=ds_350.get_comment_key_column(),
    )
    res.to_csv(output_dir / "dices-350.csv")

    ds_990 = DicesDataset(dataset_path=dataset_path_large, variant="990")
    graphs.polarization_plot(
        ds=ds_350, output_path=graph_output_dir / "dices-990.png"
    )
    res = run_helper.run_all_results(
        df=ds_990.get_dataset(),
        sdb_columns=ds_990.get_sdb_columns(),
        value_col=ds_990.get_annotation_column(),
        comment_key_col=ds_990.get_comment_key_column(),
    )
    res.to_csv(output_dir / "dices-990.csv")

    plot_annotation_histograms(
        dataset_path_small=dataset_path_small,
        dataset_path_large=dataset_path_large,
        output_path=graph_output_dir / "annotator_histograms.png",
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
