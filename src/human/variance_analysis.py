import re
import typing
import argparse
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm.auto import tqdm
import apunim

from ..lib import graphs, run_helper
from ..lib.preprocessing import (
    DicesDataset,
    PopquornDataset,
    Dataset,
)
from ..lib.util import skip_if_exists

# Every resample is a full apunim computation over all comments. Its cost
# is dominated by per-comment overhead inside apunim (roughly quadratic in
# the number of comments), not by the permutations, so the permutations match
# the main analysis (100) while the number of resamples is kept small.
RESAMPLE_ITERS = 10  # resamples per annotator sample size
APUNIM_ITERATIONS = 100  # random partitions per comment inside apunim

# Shared ethnicity groups. The order also fixes the colours in the plots.
ETHNICITY_GROUPS = [
    "Asian",
    "Black",
    "Hispanic",
    "White",
    "Multiracial",
    "Other",
]

# Shared gender groups, in plotting order.
GENDER_GROUPS = ["Man", "Woman", "Non-binary", "Other"]

# Lowercase fragments identifying a group inside a dataset's own label.
_ETHNICITY_KEYWORDS = {
    "Asian": ("asian",),
    "Black": ("black", "african am"),
    "Hispanic": ("hisp", "latin"),
    "White": ("white", "caucasian"),
}
_UNDISCLOSED = {
    "",
    "na",
    "n/a",
    "nan",
    "none",
    "unknown",
    "undisclosed",
    "prefer not to say",
    "prefer not to answer",
}


# Kumar can not be subsampled down from 5
def main(
    dices_small_path: Path,
    dices_large_path: Path,
    popquorn_offensiveness_path: Path,
    graph_dir: Path,
    cache_dir: Path,
    min_comment_annotators: int = 3,
):
    graphs.graph_setup()
    dices350_ds = DicesDataset(dataset_path=dices_small_path, variant="350")
    dices990_ds = DicesDataset(dataset_path=dices_large_path, variant="990")
    popquorn_ds = PopquornDataset(dataset_path=popquorn_offensiveness_path)
    datasets: list[Dataset] = [dices350_ds, dices990_ds, popquorn_ds]

    label_audit(datasets, cache_dir / "label_mapping.csv")

    variance_dfs = {}
    for feature in FEATURES:
        variance_df_ls = []
        for dataset in datasets:
            res_df = get_dataset_variance(
                dataset,
                feature,
                cache_dir,
                min_comment_annotators=min_comment_annotators,
            )
            res_df["dataset"] = dataset.get_name()
            variance_df_ls.append(res_df)

        variance_df = pd.concat(variance_df_ls, ignore_index=True)

        if variance_df.empty:
            print(f"No apunim values could be computed for {feature}.")
            continue

        variance_dfs[feature] = variance_df

    plot_variance_curve(
        variance_dfs,
        graph_path=graph_dir / "apunim_subsampling_robustness.png",
    )


def harmonize_ethnicity(label) -> typing.Optional[str]:
    """Map a dataset-specific ethnicity label to one of ETHNICITY_GROUPS.

    Datasets phrase ethnicity differently ("African Am.", "Black or African
    American", "black"; "Latino", "Hispanic or Latino", "hisp"). Labels are
    matched by keyword on the whole string rather than split on commas, since
    some contain commas themselves ("LatinX, Latino, Hispanic or Spanish
    Origin"). A label naming several groups, or "multiracial"/"mixed", is
    Multiracial; anything unrecognised (Native American, Pacific Islander,
    self-described, ...) is Other. Returns None if the annotator did not
    disclose their ethnicity.
    """
    if label is None or pd.isna(label):
        return None

    # "Non-Hispanic White" names one group, not two
    text = re.sub(r"non[- ]?hispanic", "", str(label).lower()).strip()

    if text in _UNDISCLOSED:
        return None
    if any(w in text for w in ("multi", "mixed", "two or more")):
        return "Multiracial"

    matched = [
        group
        for group, keywords in _ETHNICITY_KEYWORDS.items()
        if any(k in text for k in keywords)
    ]
    if len(matched) > 1:
        return "Multiracial"

    return matched[0] if matched else "Other"


def harmonize_gender(label) -> typing.Optional[str]:
    """Map a dataset-specific gender label to one of GENDER_GROUPS.

    Datasets phrase gender differently ("Man", "Male", "man"; "nonBinary",
    "Non-binary"). Returns None if the annotator did not disclose it.
    """
    if label is None or pd.isna(label):
        return None

    text = str(label).lower().strip()

    if text in _UNDISCLOSED:
        return None
    if "binary" in text:
        return "Non-binary"
    # whole words only: "woman" and "female" contain "man" and "male"
    if re.search(r"\b(woman|women|female)\b", text):
        return "Woman"
    if re.search(r"\b(man|men|male)\b", text):
        return "Man"

    return "Other"


# feature -> (function mapping a label to a shared group, shared groups)
FEATURES = {
    "gender": (harmonize_gender, GENDER_GROUPS),
    "ethnicity": (harmonize_ethnicity, ETHNICITY_GROUPS),
}


def _group_column(dataset: Dataset, feature: str) -> str:
    """Name of the annotator-group column for a feature in this dataset."""
    if feature == "gender":
        return "Gender"

    # the datasets disagree on the name: DICES/POPQUORN "Race", others
    # "Ethnicity"
    for column in ("Race", "Ethnicity"):
        if column in dataset.get_sdb_columns():
            return column

    raise ValueError(f"{dataset.get_name()} has no ethnicity column.")


def label_audit(datasets: list[Dataset], out_path: Path) -> None:
    """Save and print how every gender/ethnicity label is mapped.

    `label_in_dataset` is the label as produced by the dataset loader, which
    already relabels some values. Check this table whenever a dataset changes.
    """
    rows = []
    for feature, (harmonize, _) in FEATURES.items():
        for ds in datasets:
            labels = ds.get_dataset()[_group_column(ds, feature)].explode()
            for label, n in labels.value_counts(dropna=False).items():
                rows.append(
                    {
                        "feature": feature,
                        "dataset": ds.get_name(),
                        "label_in_dataset": label,
                        "harmonized": harmonize(label) or "(dropped)",
                        "annotations": n,
                    }
                )

    audit = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out_path, index=False)
    print(audit.to_string(index=False))


def apunim_vs_sample_size(
    df: pd.DataFrame,
    annotation_col: str,
    group_col: str,
    bins: typing.Optional[int] = None,
    min_size: int = 2,
    max_size: typing.Optional[int] = None,
    step: int = 1,
    iters: int = RESAMPLE_ITERS,
    min_comment_annotators: int = 3,
    relabel: typing.Optional[typing.Callable] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Resample annotators per comment and compute apunim on the result.

    For each sample size, `iters` times: draw `size` annotators (with
    replacement) from every comment, then compute apunim for `group_col`
    over all resampled comments. One row is returned per (sample_size,
    iteration, level) for every level of `group_col` that apunim could score,
    along with the level's support (number of annotations it is based on).

    This version:
    - If bins is None, uses the number of distinct annotation values in the
      whole dataset, so that every resample shares the same bin grid.
    - If relabel is given, it maps each group label to a shared label (or
      None). Annotations whose label maps to None are dropped.
    - If max_size is None, uses the maximum number of annotators found across
      comments after that filtering.
    - Skips comments that have fewer than `min_comment_annotators` or fewer
      than `size` annotators.
    - Produces no row for a level in an iteration in which apunim is
      undefined for it (e.g. no comment has enough annotators of that level).
    """
    columns = ["sample_size", "iteration", "level", "apunim", "support"]
    rng = np.random.default_rng(seed)

    if bins is None:
        bins = run_helper._compute_bins(df[annotation_col].to_numpy(), None)

    # (comment id, annotations, groups) for the comments that can be sampled
    comments = []
    for comment_id, (anns, grps) in enumerate(
        zip(df[annotation_col], df[group_col])
    ):
        if anns is None:
            continue
        try:
            len(anns)
        except Exception:
            continue

        anns, grps = np.array(anns), np.array(grps)
        if relabel is not None:
            grps = np.array([relabel(g) for g in grps], dtype=object)
            known = np.array([g is not None for g in grps], dtype=bool)
            anns, grps = anns[known], grps[known]

        if len(anns) >= min_comment_annotators:
            comments.append((comment_id, anns, grps))

    if not comments:
        return pd.DataFrame(columns=columns)

    if max_size is None:
        max_size = max(len(anns) for _, anns, _ in comments)

    results: list[dict[str, typing.Any]] = []

    for size in tqdm(range(min_size, max_size + 1, step), desc="#Annotators"):
        for iteration in tqdm(range(iters), desc="#Iterations", leave=False):
            annotations, groups, comment_ids = [], [], []

            for comment_id, anns, grps in comments:
                if len(anns) < size:
                    continue

                idx = rng.choice(len(anns), size=size)
                annotations.extend(anns[idx])
                groups.extend(grps[idx])
                comment_ids.extend([comment_id] * size)

            for level, res in _apunim_by_level(
                annotations, groups, comment_ids, bins, rng
            ).items():
                results.append(
                    {
                        "sample_size": size,
                        "iteration": iteration,
                        "level": str(level),
                        "apunim": res.apunim,
                        "support": res.support,
                    }
                )

    # explicit columns keep the cached CSV readable even if `results` is empty
    return pd.DataFrame(results, columns=columns)


def _apunim_by_level(
    annotations: list,
    groups: list,
    comment_ids: list,
    bins: int,
    rng: np.random.Generator,
) -> dict:
    """apunim result per group level, empty if undefined."""
    try:
        with warnings.catch_warnings():
            # apunim warns on degenerate small resamples; they are expected
            warnings.simplefilter("ignore")
            res = apunim.aposteriori_unimodality(
                annotations=annotations,
                factor_group=groups,
                comment_group=comment_ids,
                num_bins=bins,
                iterations=APUNIM_ITERATIONS,
                alpha=None,
                seed=int(rng.integers(2**31)),
            )
    except ValueError:
        return {}

    return {k: r for k, r in res.items() if not np.isnan(r.apunim)}


def _tex(text: str) -> str:
    """Escape characters that LaTeX would interpret in a plot label."""
    return re.sub(r"([&%$#_])", r"\\\1", text)


def plot_variance_curve(
    results_by_feature: dict[str, pd.DataFrame], graph_path: Path
):
    """One row per feature, one column per dataset, one line per group level
    (mean, +-2 SD)."""
    features = list(results_by_feature)
    datasets = list(
        dict.fromkeys(
            ds
            for df in results_by_feature.values()
            for ds in df["dataset"].unique()
        )
    )

    nrows, ncols = len(features), len(datasets)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        sharex="col",
        sharey="row",  # gender and ethnicity can have different ranges
        squeeze=False,
        figsize=(5 * ncols, 3.5 * nrows),
    )

    for row, feature in enumerate(features):
        results_df = results_by_feature[feature]
        present = results_df["level"].unique()
        levels = [g for g in FEATURES[feature][1] if g in present]

        colors = {
            level: graphs.COLORBLIND_PALETTE[
                i % len(graphs.COLORBLIND_PALETTE)
            ]
            for i, level in enumerate(levels)
        }
        markers = {
            level: graphs.MARKERS[i % len(graphs.MARKERS)]
            for i, level in enumerate(levels)
        }

        for col, ds_name in enumerate(datasets):
            ax = axes[row, col]
            ds_df = results_df[results_df["dataset"] == ds_name]

            for level in levels:
                level_df = ds_df[ds_df["level"] == level].sort_values(
                    "sample_size"
                )
                if level_df.empty:
                    continue

                sns.lineplot(
                    data=level_df,
                    x="sample_size",
                    y="apunim",
                    errorbar=("sd", 2),
                    err_kws={"alpha": 0.12},
                    color=colors[level],
                    marker=markers[level],
                    ax=ax,
                )

            # apunim = 0: polarization is explained by chance
            ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")

            # titles on the top row, x labels on the bottom row only
            ax.set_title(ds_name if row == 0 else "")
            ax.set_xlabel(
                r"\# Annotators sampled per comment"
                if row == nrows - 1
                else ""
            )
            # row label on the first column only
            ax.set_ylabel(
                f"{feature.capitalize()}\napunim" if col == 0 else ""
            )

        # each feature has its own levels, so each row gets its own legend
        handles = [
            Line2D(
                [0],
                [0],
                color=colors[level],
                marker=markers[level],
                label=_tex(level),
            )
            for level in levels
        ]
        axes[row, -1].legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            title=feature.capitalize(),
        )

    fig.suptitle(
        "Effect of annotator sample size on apunim "
        r"(mean $\pm$ 2 SD across resamples)"
    )
    fig.tight_layout()

    graphs.save_plot(graph_path)
    plt.close(fig)


def get_dataset_variance(
    dataset: Dataset,
    feature: str,
    cache_dir: Path,
    min_comment_annotators: int,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    # One file per dataset and feature. Not "_variance.csv": caches from the
    # old pol_obs analysis have a different schema and must not be picked up.
    cache_file = (
        cache_dir / f"{dataset.get_name()}_{feature}_apunim_variance.csv"
    )

    if skip_if_exists(cache_file):
        print(
            f"Loading cached {feature} variance results for "
            f"{dataset.get_name()} from {cache_file}"
        )
        return pd.read_csv(cache_file)

    print(f"Computing {feature} variance results for {dataset.get_name()}...")
    res_df = apunim_vs_sample_size(
        df=dataset.get_dataset().reset_index(),
        annotation_col=dataset.get_annotation_column(),
        group_col=_group_column(dataset, feature),
        bins=None,
        min_size=6,
        max_size=None,
        step=1,
        iters=RESAMPLE_ITERS,
        min_comment_annotators=min_comment_annotators,
        relabel=FEATURES[feature][0],
    )

    res_df.to_csv(cache_file, index=False)
    return res_df


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
        "--popquorn-path",
        required=True,
        help=("Path to the POPQUORN offensiveness dataset."),
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
        popquorn_offensiveness_path=Path(args.popquorn_path),
    )
