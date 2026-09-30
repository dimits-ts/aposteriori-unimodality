"""
Running the synthetic-experiment methods on the five real datasets.

Goal: To see how the methods compared in metric_comparison.py behave on real
annotation data, where the group structure is not known in advance.

For every dataset and every SDB (sociodemographic background) column, each
method is run once and produces a single (stat, pvalue) pair. The results are
cached in one CSV with one row per (dataset, SDB feature, method).

Methods (labels match shared.METHOD_ORDER):

    apunim, Krippendorff delta-alpha, aposteriori unimodality (2024),
    chi-squared (Akhtar et al. 2019)

GMM clustering is not included: it clusters per-annotator mean ratings, which
requires annotators to be identifiable across comments. The real datasets only
store annotations per comment.

The methods in shared.py expect a dense annotators x items matrix. Real data
is ragged (each comment has its own annotators), so this module contains
long-format versions of the methods:

    - Krippendorff's alpha only depends on the values within each comment,
      so comments are padded with NaN into a matrix of annotation "slots".
    - Group-label permutations are done within each comment, which keeps
      each comment's group composition fixed.
    - The number of rating levels is read from the data instead of N_LEVELS.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import apunim
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from ..lib import run_helper
from ..lib.preprocessing import (
    Dataset,
    DicesDataset,
    KumarDataset,
    PopquornDataset,
    SapDataset,
)
from ..lib.util import skip_if_exists
from .shared import ALPHA, _alpha, _seed

KUMAR_NUM_SAMPLES = 3_000

METHOD_APUNIM = "apunim"
METHOD_DELTA_ALPHA = "Krippendorff delta-alpha"
METHOD_ORIGINAL_AU = "aposteriori unimodality (2024)"
METHOD_CHI2 = "chi-squared (Akhtar et al. 2019)"

N_PERM_DELTA_ALPHA = 200
N_PERM_ORIGINAL_AU = 100

COLUMNS = ["dataset", "sdb_feature", "method", "stat", "pvalue", "detected"]


# --------------------------------------------------------------------- main


def main(
    cache_path: Path,
    dices_small_path: Path,
    dices_large_path: Path,
    sap_path: Path,
    kumar_path: Path,
    popquorn_path: Path,
    workers: int,
) -> pd.DataFrame:
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if skip_if_exists(cache_path):
        print(f"loading cached results from {cache_path}")
    else:
        run(
            cache_path,
            dices_small_path,
            dices_large_path,
            sap_path,
            kumar_path,
            popquorn_path,
            workers,
        )

    return pd.read_csv(cache_path)


# ------------------------------------------------------------------- runner


def _flatten(
    ds: Dataset, feature: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Flatten one SDB column of a dataset to parallel arrays of annotations,
    group labels and integer comment codes, sorted by comment code.
    """
    annotations, groups, keys = run_helper._extract_annotations_and_attributes(
        df=ds.get_dataset(),
        value_col=ds.get_annotation_column(),
        feature_col=feature,
        comment_key_col=ds.get_comment_key_column(),
    )
    _, codes = np.unique(np.asarray(keys, dtype=str), return_inverse=True)
    order = np.argsort(codes, kind="stable")

    return (
        np.asarray(annotations, dtype=float)[order],
        np.asarray(groups, dtype=str)[order],
        codes[order],
    )


def _load_datasets(
    dices_small_path: Path,
    dices_large_path: Path,
    sap_path: Path,
    kumar_path: Path,
    popquorn_path: Path,
) -> list[Dataset]:
    return [
        DicesDataset(dataset_path=dices_small_path, variant="350"),
        DicesDataset(dataset_path=dices_large_path, variant="990"),
        SapDataset(dataset_path=sap_path),
        KumarDataset(dataset_path=kumar_path, num_samples=KUMAR_NUM_SAMPLES),
        PopquornDataset(dataset_path=popquorn_path),
    ]


def run(
    out_csv: Path,
    dices_small_path: Path,
    dices_large_path: Path,
    sap_path: Path,
    kumar_path: Path,
    popquorn_path: Path,
    workers: int,
) -> None:
    datasets = _load_datasets(
        dices_small_path,
        dices_large_path,
        sap_path,
        kumar_path,
        popquorn_path,
    )

    jobs = []
    for ds in datasets:
        columns = set(ds.get_sdb_columns()).intersection(
            ds.get_dataset().columns
        )
        for feature in sorted(columns):
            jobs.append((ds.get_name(), feature, *_flatten(ds, feature)))

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(_one, jobs):
            rows.extend(result)

    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(out_csv, index=False)

    print(f"wrote {out_csv} ({len(jobs)} dataset/SDB feature pairs)")


def _one(job) -> list[tuple]:
    """Run all methods for one (dataset, SDB feature) pair."""
    name, feature, values, groups, codes = job

    results = [
        (METHOD_APUNIM, *method_apunim(values, groups, codes)),
        (
            METHOD_DELTA_ALPHA,
            *method_delta_alpha(
                values, groups, codes, _seed(name, feature, "delta-alpha")
            ),
        ),
        (
            METHOD_ORIGINAL_AU,
            *method_original_au(
                values, groups, codes, _seed(name, feature, "original-au")
            ),
        ),
        (METHOD_CHI2, *method_chi2_variance(values, groups)),
    ]

    return [
        (
            name,
            feature,
            method,
            stat,
            pvalue,
            int(pvalue < ALPHA) if pvalue == pvalue else 0,
        )
        for method, stat, pvalue in results
    ]


# ------------------------------------------------------------------ helpers


def _num_bins(values):
    """Number of distinct rating levels in the data (at least 2)."""
    return max(len(np.unique(values)), 2)


def _comment_slices(codes):
    """Index arrays, one per comment. Assumes `codes` is sorted."""
    bounds = np.flatnonzero(np.diff(codes)) + 1
    return np.split(np.arange(len(codes)), bounds)


def _permute_within_comments(groups, codes, rng):
    """
    Shuffle group labels within each comment. Assumes `codes` is sorted, so
    that sorting by (code, random key) keeps every comment in its own block.
    """
    return groups[np.lexsort((rng.random(len(groups)), codes))]


def _pad(values, codes, n_comments):
    """Matrix (max annotations per comment x n_comments), NaN padded."""
    counts = np.bincount(codes, minlength=n_comments)
    starts = np.cumsum(counts) - counts
    slot = np.arange(len(values)) - starts[codes]

    matrix = np.full((max(counts.max(), 1), n_comments), np.nan)
    matrix[slot, codes] = values

    return matrix


# ------------------------------------------------------------------ methods


def method_apunim(values, groups, codes):
    """
    Same aggregation as shared.method_apunim: the factor with the smallest
    p-value, Bonferroni-corrected over the factors that could be tested.
    """
    try:
        res = apunim.aposteriori_unimodality(
            values,
            groups,
            codes,
            num_bins=_num_bins(values),
            iterations=100,
            seed=42,
        )
    except ValueError:
        return 0.0, 1.0

    res = {k: v for k, v in res.items() if not np.isnan(v.pvalue)}

    if not res:
        return 0.0, 1.0

    best = min(res.values(), key=lambda r: r.pvalue)
    return best.apunim, min(1.0, best.pvalue * len(res))


def _safe_alpha(matrix):
    try:
        return _alpha(matrix)
    except ValueError:
        return np.nan


def _delta_alpha(values, groups, codes, n_comments):
    overall = _safe_alpha(_pad(values, codes, n_comments))
    within = [
        alpha
        for alpha in (
            _safe_alpha(
                _pad(values[groups == g], codes[groups == g], n_comments)
            )
            for g in np.unique(groups)
        )
        if not np.isnan(alpha)
    ]

    return (float(np.mean(within)) - overall) if within else np.nan


def method_delta_alpha(values, groups, codes, seed, n_perm=N_PERM_DELTA_ALPHA):
    n_comments = codes.max() + 1
    obs = _delta_alpha(values, groups, codes, n_comments)

    if np.isnan(obs):
        return np.nan, 1.0

    rng = np.random.default_rng(seed)
    null = np.array(
        [
            _delta_alpha(
                values,
                _permute_within_comments(groups, codes, rng),
                codes,
                n_comments,
            )
            for _ in range(n_perm)
        ]
    )
    null = null[~np.isnan(null)]

    return obs, (1 + np.sum(null >= obs)) / (1 + len(null))


def _frac_explained(values, groups, slices, bins):
    """
    Fraction of the (non-unimodal) comments in which every group's
    sub-distribution is unimodal.
    """
    hits = 0

    for s in slices:
        col, col_groups = values[s], groups[s]

        if all(
            apunim.dfu(col[col_groups == g], bins=bins) <= 0
            for g in np.unique(col_groups)
        ):
            hits += 1

    return hits / len(slices) if slices else np.nan


def method_original_au(values, groups, codes, seed, n_perm=N_PERM_ORIGINAL_AU):
    bins = _num_bins(values)

    # Which comments are non-unimodal does not depend on the group labels,
    # so find them once instead of in every permutation.
    slices = [
        s
        for s in _comment_slices(codes)
        if apunim.dfu(values[s], bins=bins) > 0
    ]

    obs = _frac_explained(values, groups, slices, bins)

    if np.isnan(obs):
        return np.nan, 1.0

    rng = np.random.default_rng(seed)
    null = np.array(
        [
            _frac_explained(
                values,
                _permute_within_comments(groups, codes, rng),
                slices,
                bins,
            )
            for _ in range(n_perm)
        ]
    )

    return obs, (1 + np.sum(null >= obs)) / (1 + len(null))


def method_chi2_variance(values, groups):
    """
    Chi-squared test of independence between group membership and pooled
    annotation level.
    """
    levels = np.unique(values)
    table = np.array(
        [
            [((groups == g) & (values == lvl)).sum() for lvl in levels]
            for g in np.unique(groups)
        ]
    )

    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan, 1.0

    try:
        chi2_stat, pvalue, _, _ = chi2_contingency(table)
    except ValueError:
        return np.nan, 1.0

    return chi2_stat, pvalue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run the methods from the synthetic experiments (apunim, "
            "Krippendorff delta-alpha, original aposteriori unimodality, "
            "chi-squared) on the five real datasets, for each SDB feature. "
            "Results are cached in a CSV and reused if it exists."
        )
    )

    parser.add_argument(
        "--cache-path",
        required=True,
        help="Path for the output CSV cache.",
    )
    parser.add_argument(
        "--dices-small-path",
        required=True,
        help="Path to the DICES-350 CSV file.",
    )
    parser.add_argument(
        "--dices-large-path",
        required=True,
        help="Path to the DICES-990 CSV file.",
    )
    parser.add_argument(
        "--sap-path",
        required=True,
        help="Path to the Sap et al. dataset.",
    )
    parser.add_argument(
        "--kumar-path",
        required=True,
        help="Path to the Kumar et al. dataset.",
    )
    parser.add_argument(
        "--popquorn-path",
        required=True,
        help="Path to the POPQUORN offensiveness dataset.",
    )
    parser.add_argument("--workers", type=int, default=7)

    args = parser.parse_args()

    main(
        Path(args.cache_path),
        Path(args.dices_small_path),
        Path(args.dices_large_path),
        Path(args.sap_path),
        Path(args.kumar_path),
        Path(args.popquorn_path),
        args.workers,
    )
