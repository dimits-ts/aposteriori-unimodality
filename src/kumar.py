import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tasks.graphs
import tasks.preprocessing
import tasks.run_helper

# Seeds used to repeat the 3k-comment sample experiment with different
# random subsamples, to see how sensitive results are to which 3k
# comments happen to get selected.
KUMAR_3K_SEED_ABLATION_SEEDS = list(range(10))


class KumarDataset(tasks.preprocessing.Dataset):
    def __init__(
        self,
        dataset_path: Path,
        num_samples: int | None = None,
        seed: int = 42,
    ):
        self.df = self._remove_invalid_ann_counts(
            KumarDataset._base_df(dataset_path, num_samples, seed)
        )

    def get_name(self) -> str:
        return "Kumar et al. 2021"

    def get_dataset(self) -> pd.DataFrame:
        return self.df

    def get_sdb_columns(self) -> list[str]:
        return [
            "Gender",
            "Ethnicity",
            "Age",
            "Education",
            "Sexual Orientation",
            "Is Transgender",
            "Political Affiliation",
            "Is Parent",
            "Technology Impact",
            "Toxicity Problem",
            "Religion Important",
            "Seen Toxicity",
            "Has Been Targeted",
        ]

    def get_comment_key_column(self) -> str:
        return "comment"

    def get_annotation_column(self) -> str:
        return "Toxicity"
    
    def get_text_column(self) -> str:
        return "comment"

    @staticmethod
    def _base_df(
        dataset_path: Path, num_samples: int | None, seed: int = 42
    ) -> pd.DataFrame:
        df = pd.read_json(dataset_path, lines=True)
        df = df.explode(column="ratings")
        df = df.dropna()

        ratings_df = pd.json_normalize(df.ratings)
        df = pd.concat([df.reset_index(), ratings_df.reset_index()], axis=1)
        df = df.drop(columns=["ratings", "index"])
        # shorten names
        df = df.replace(
            {
                (
                    "High school graduate (high school diploma or equivalent "
                    "including GED)"
                ): "High School graduate",
                "Associate degree in college (2-year)": "Associate degree",
                "Bachelor's degree in college (4-year)": "Bachelor's degree",
                "Less than high school degree": "No high school",
                "Professional degree (JD, MD)": "Professional degree",
                "Some college but no degree": "College, no degree",
            }
        )
        # define ranking from most to least qualified
        ranking = [
            "Doctoral degree",
            "Professional degree",
            "Master's degree",
            "Bachelor's degree",
            "Associate degree",
            "College, no degree",
            "High School graduate",
            "No high school",
        ]
        ranking.reverse()

        # create a mapping with ordinal prefix: 1), 2), 3)...
        ordinal_map = {
            name: f"{i+1}) {name}" for i, name in enumerate(ranking)
        }

        # apply the new labels
        df["education"] = df["education"].replace(ordinal_map)

        df = df.replace(
            {
                "Very important": "4) Very",
                "Somewhat important": "3) Somewhat",
                "Not too important": "2) Not very",
                "Not important": "1) No",
            }
        )
        df = df.replace(
            {
                "Very frequently a problem": "5) Very Frequently",
                "Frequently a problem": "4) Frequently",
                "Occasionally a problem": "3) Occasionally",
                "Rarely a problem": "2) Rarely",
                "Not a problem": "1) Never",
            }
        )
        df = df.replace(
            {
                "Very positive": "5) Very positive",
                "Somewhat positive": "4) Somewhat positive",
                # wtf?
                "Neutral \u00e2\u0080\u0093 neither positive nor negative": "3) Neutral",
                "Somewhat negative": "2) Somewhat negative",
                "Very negative": "1) Very negative",
            }
        )

        age_ranking = [
            "18 - 24",
            "25 - 34",
            "35 - 44",
            "45 - 54",
            "55 - 64",
            "65 or older",
        ]
        age_ordinal_map = {
            name: f"{i+1}) {name}" for i, name in enumerate(age_ranking)
        }
        df.age_range = df.age_range.replace(age_ordinal_map)

        df = df.loc[
            :,
            [
                "comment",
                "toxic_score",
                "gender",
                "race",
                "personally_seen_toxic_content",
                "personally_been_target",
                "identify_as_transgender",
                "toxic_comments_problem",
                "education",
                "age_range",
                "lgbtq_status",
                "political_affilation",  # sic
                "is_parent",
                "religion_important",
                "technology_impact"
            ],
        ]
        df.race = df.race.apply(KumarDataset._simplify_ethnicity)
        df = df.groupby("comment").agg(list)

        if num_samples is not None:
            print(
                f"Selecting {num_samples} out of {len(df)} total comments "
                f"(seed={seed})."
            )
            df = df.sample(num_samples, random_state=seed)

        df = df.reset_index()

        df = df.rename(
            columns={
                "personally_seen_toxic_content": "Seen Toxicity",
                "personally_been_target": "Has Been Targeted",
                "identify_as_transgender": "Is Transgender",
                "toxic_comments_problem": "Toxicity Problem",
                "education": "Education",
                "age_range": "Age",
                "lgbtq_status": "Sexual Orientation",
                "political_affilation": "Political Affiliation",
                "is_parent": "Is Parent",
                "religion_important": "Religion Important",
                "toxic_score": "Toxicity",
                "gender": "Gender",
                "race": "Ethnicity",
                "technology_impact": "Technology Impact",
            }
        )
        return df

    @staticmethod
    def _simplify_ethnicity(x):
        if isinstance(x, list):
            # If your field is a list (after aggregation)
            x = x[0]

        if pd.isna(x):
            return "Unknown"

        if "," in x:
            return "Multiracial"

        mapping = {
            "Asian": "Asian",
            "Black or African American": "Black",
            "Hispanic": "Hispanic",
            "White": "White",
            "Other": "Other",
            "Prefer not to say": "Unknown",
        }
        return mapping.get(x, "Other")

    @staticmethod
    def _remove_invalid_ann_counts(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        # --- There is a single comment with 650 annotators ---
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
        return df


def _ordinal_to_yn_neutral(lst):
    new_lst = []
    for x in lst:
        # extract the numeric prefix
        try:
            num = int(x.split(")")[0])
        except:
            num = 3  # fallback
        if num == 3:
            new_lst.append("Neutral")
        elif num > 3:
            new_lst.append("Yes")
        else:
            new_lst.append("No")
    return new_lst


def _safe_len(x):
    try:
        return len(x)
    except Exception:
        return 0


def run_experiment(
    dataset_path: Path,
    output_path: Path,
    num_samples: int,
    seed: int = 42,
) -> None:
    if output_path.exists():
        print(f"{output_path} exists, skipping...")
        return

    print(f"Running experiment {output_path}...")
    ds = KumarDataset(
        dataset_path=dataset_path, num_samples=num_samples, seed=seed
    )
    res = tasks.run_helper.run_all_results(ds)
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
        df = df.rename(columns={subgroup_col: "subgroup"})
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

    tasks.graphs.save_plot(output_path)
    plt.close(g.figure)


def main(
    dataset_path: Path,
    output_dir: Path,
    graph_output_dir: Path,
    ablations_dir: Path,
):
    graph_output_dir.mkdir(parents=True, exist_ok=True)
    tasks.graphs.graph_setup()

    print("Generating sample polarization plot...")
    tasks.graphs.polarization_plot(
        ds=KumarDataset(dataset_path=dataset_path, num_samples=3_000),
        output_path=graph_output_dir / "kumar_sample.png",
    )
    print("Calculating inherent polarization...")
    res = tasks.run_helper.compute_inherent_polarization_exhaustive(
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
