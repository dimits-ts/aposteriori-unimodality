"""
Statistical significance testing of synthetic metrics on human datasets.

Applies all synthetic comparison methods (apunim, Krippendorff delta-alpha,
aposteriori unimodality, chi-squared, GMM clustering) to each SDB dimension
of each human dataset (DICES-350, DICES-990, Kumar, Sap), and exports a
single LaTeX table showing the statistically significant results.

Each cell reports the method's test statistic with a p-value superscript:
    *   p < 0.05
    **  p < 0.01
    *** p < 0.001

Results are cached to a CSV file.  If the cache file already exists the
entire computation step is skipped and the table is exported directly from
the cache.  Pass ``--force`` to recompute even when the cache exists.

Usage (from the project root, with the src package on PYTHONPATH):

    python -m src.synthetic.human_dataset_significance \\
        --sap-dataset-path      data/sap.pkl \\
        --dices-350-path        data/dices-350.csv \\
        --dices-990-path        data/dices-990.csv \\
        --kumar-dataset-path    data/kumar.jsonl \\
        --cache-dir             cache/ \\
        --latex-output-dir      latex/ \\
        [--workers 4] [--force]
"""

import argparse
import csv
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Generator, Iterator

import apunim as _apunim
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..lib.preprocessing import DicesDataset, KumarDataset, SapDataset
from ..lib.util import significance_superscript, skip_if_exists
from .shared import (
    ALPHA,
    METHOD_ORDER,
    N_LEVELS,
    method_chi2_variance,
    method_delta_alpha,
    method_mixture_clustering,
    method_original_au,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KUMAR_NUM_SAMPLES = 1_000
KUMAR_SEED = 42

# Minimum number of annotations in a subgroup to include in the analysis.
MIN_SUPPORT = 6

LATEX_METHOD_LABELS: dict[str, str] = {
    "apunim": r"\textsc{Apunim}",
    "Krippendorff delta-alpha": r"Krippendorff $\Delta\alpha$",
    "aposteriori unimodality (2024)": r"Apost.\ unim.",
    "chi-squared (Akhtar et al. 2019)": r"$\chi^2$",
    "GMM clustering (Checco/Mignemi)": r"GMM",
}

CSV_FIELDNAMES: list[str] = [
    "dataset",
    "sdb_column",
    "factor",
    "method",
    "stat",
    "pvalue",
    "n_annotations",
]

# Type aliases
LongFormat = tuple[np.ndarray, np.ndarray, np.ndarray]
MethodResult = tuple[str, float, float]
Job = tuple[str, str, str, str, str, list[dict]]
CacheKey = tuple[str, str, str, str]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(
    sap_path: Path,
    dices_350_path: Path,
    dices_990_path: Path,
    kumar_path: Path,
    cache_dir: Path,
    latex_output_dir: Path,
    workers: int,
    force: bool,
) -> None:
    datasets = _load_datasets(
        sap_path, dices_350_path, dices_990_path, kumar_path
    )

    if not datasets:
        raise RuntimeError(
            "No datasets could be loaded. "
            "Check that at least one --*-path argument "
            "points to an existing file."
        )

    print(
        f"Loaded {len(datasets)} dataset(s): "
        f"{[ds.get_name() for ds in datasets]}"
    )

    rows = run(
        datasets,
        cache_dir / "human_synthetic.csv",
        workers,
        force=force,
    )
    export_latex(rows, latex_output_dir / "human_synthetic.tex")


# ---------------------------------------------------------------------------
# Long-format extraction
# ---------------------------------------------------------------------------


def _flatten_rows(
    df: pd.DataFrame,
    annotation_col: str,
    sdb_col: str,
    comment_key_col: str,
    factor: str,
) -> Generator[tuple[object, str, object], None, None]:
    """
    Iterate over the grouped DataFrame and yield ``(ann, group, item)``
    triples, assigning ``factor`` or ``"other"`` as the group label.
    """
    for _, row in df.iterrows():
        key = row[comment_key_col]
        for ann, grp in zip(row[annotation_col], row[sdb_col]):
            yield ann, (factor if grp == factor else "other"), key


def _drop_nan_annotations(
    annotations: np.ndarray,
    groups: np.ndarray,
    items: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.array([a == a and a is not None for a in annotations])
    return annotations[valid], groups[valid], items[valid]


def _items_with_both_groups(
    items: np.ndarray,
    groups: np.ndarray,
    factor: str,
) -> set:
    """Return the set of item ids that have annotations from both groups."""
    item_group_sets: dict = {}
    for item, grp in zip(items, groups):
        item_group_sets.setdefault(item, set()).add(grp)
    return {
        item
        for item, gs in item_group_sets.items()
        if "other" in gs and factor in gs
    }


def _has_sufficient_support(groups: np.ndarray, factor: str) -> bool:
    for grp_label in (factor, "other"):
        if np.sum(groups == grp_label) < MIN_SUPPORT:
            return False
    return True


def _build_long_format(
    df: pd.DataFrame,
    annotation_col: str,
    sdb_col: str,
    comment_key_col: str,
    factor: str,
) -> LongFormat:
    """
    Return ``(annotations, groups, item_ids)`` long-format arrays for a
    single ``(sdb_column, factor)`` combination, using **all** available
    annotations without any sampling or truncation.

    Returns None if the data is too sparse to be useful.
    """
    triples = list(
        _flatten_rows(df, annotation_col, sdb_col, comment_key_col, factor)
    )
    if not triples:
        return None

    ann_arr = np.array([t[0] for t in triples], dtype=object)
    grp_arr = np.array([t[1] for t in triples])
    item_arr = np.array([t[2] for t in triples])

    ann_arr, grp_arr, item_arr = _drop_nan_annotations(
        ann_arr, grp_arr, item_arr
    )

    both = _items_with_both_groups(item_arr, grp_arr, factor)
    if len(both) < 2:
        return None

    mask = np.isin(item_arr, list(both))
    ann_arr = ann_arr[mask]
    grp_arr = grp_arr[mask]
    item_arr = item_arr[mask]

    if not _has_sufficient_support(grp_arr, factor):
        return None

    return ann_arr.astype(float), grp_arr, item_arr


# ---------------------------------------------------------------------------
# Matrix conversion (for methods that require a 2-D matrix)
# ---------------------------------------------------------------------------


def _max_per_group_per_item(
    item_ids: np.ndarray,
    groups: np.ndarray,
    factor_label: str,
    unique_items: list,
) -> tuple[int, int]:
    """Return (max_factor_count, max_other_count) across all items."""
    max_factor = max(
        np.sum((item_ids == item) & (groups == factor_label))
        for item in unique_items
    )
    max_other = max(
        np.sum((item_ids == item) & (groups == "other"))
        for item in unique_items
    )
    return int(max_factor), int(max_other)


def _fill_matrix_column(
    matrix: np.ndarray,
    col: int,
    annotations: np.ndarray,
    groups: np.ndarray,
    factor_label: str,
    max_factor: int,
) -> None:
    """Write one item's annotations into the correct rows of the matrix."""
    factor_anns = annotations[groups == factor_label]
    other_anns = annotations[groups == "other"]
    matrix[: len(factor_anns), col] = factor_anns
    matrix[max_factor : max_factor + len(other_anns), col] = other_anns


def _long_to_matrix(
    annotations: np.ndarray,
    groups: np.ndarray,
    item_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert long-format arrays into a NaN-padded ``(n_ann_max, n_items)``
    matrix plus a ``groups`` array of length ``n_ann_max``.

    Factor-group annotations occupy the first rows; "other" occupies the
    rest.  NaN pads items that have fewer annotations than the maximum.

    krippendorff.alpha treats NaN as missing natively.  The chi-squared
    and GMM methods mask NaN out before computing.  method_original_au
    iterates column-wise and passes each column through apunim.dfu, which
    ignores NaN.
    """
    unique_items = list(dict.fromkeys(item_ids))
    item_index = {item: i for i, item in enumerate(unique_items)}
    factor_label = str(groups[groups != "other"][0])

    max_factor, max_other = _max_per_group_per_item(
        item_ids, groups, factor_label, unique_items
    )
    n_ann = max_factor + max_other
    n_items = len(unique_items)

    matrix = np.full((n_ann, n_items), np.nan)
    mat_groups = np.array([factor_label] * max_factor + ["other"] * max_other)

    for item in unique_items:
        mask = item_ids == item
        _fill_matrix_column(
            matrix,
            item_index[item],
            annotations[mask],
            groups[mask],
            factor_label,
            max_factor,
        )

    return matrix, mat_groups


# ---------------------------------------------------------------------------
# Method dispatch
# ---------------------------------------------------------------------------


def _run_apunim(
    annotations: np.ndarray,
    groups: np.ndarray,
    item_ids: np.ndarray,
) -> tuple[float, float]:
    """Run apunim on long-format arrays; return (stat, pvalue)."""
    try:
        res = _apunim.aposteriori_unimodality(
            annotations,
            groups,
            item_ids,
            num_bins=N_LEVELS,
            iterations=100,
            seed=42,
        )
    except Exception:
        return float("nan"), float("nan")

    if not res:
        return 0.0, 1.0

    best = min(res.items(), key=lambda kv: kv[1].pvalue)
    return best[1].apunim, min(1.0, best[1].pvalue * len(res))


def _all_method_results(
    annotations: np.ndarray,
    groups: np.ndarray,
    item_ids: np.ndarray,
) -> list[MethodResult]:
    """
    Run every comparison method and return a list of
    ``(method_label, stat, pvalue)`` tuples.
    """
    matrix, mat_groups = _long_to_matrix(annotations, groups, item_ids)
    stat_da, pvalue_da, _ = method_delta_alpha(matrix, mat_groups, 42)

    return [
        ("apunim", *_run_apunim(annotations, groups, item_ids)),
        ("Krippendorff delta-alpha", stat_da, pvalue_da),
        (
            "aposteriori unimodality (2024)",
            *method_original_au(matrix, mat_groups, 42),
        ),
        (
            "chi-squared (Akhtar et al. 2019)",
            *method_chi2_variance(matrix, mat_groups, 42),
        ),
        (
            "GMM clustering (Checco/Mignemi)",
            *method_mixture_clustering(matrix, mat_groups, 42),
        ),
    ]


# ---------------------------------------------------------------------------
# Worker (runs inside ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _make_result_row(
    dataset_name: str,
    sdb_col: str,
    factor: str,
    n_annotations: int,
    method: str,
    stat: float,
    pvalue: float,
) -> dict:
    return {
        "dataset": dataset_name,
        "sdb_column": sdb_col,
        "factor": factor,
        "method": method,
        "stat": float(stat) if stat == stat else float("nan"),
        "pvalue": float(pvalue) if pvalue == pvalue else float("nan"),
        "n_annotations": n_annotations,
    }


def _run_one_factor(
    dataset_name: str,
    sdb_col: str,
    factor: str,
    annotation_col: str,
    comment_key_col: str,
    df_records: list[dict],
) -> list[dict]:
    """
    Serialisation-friendly worker: build long-format data, run all
    methods, and return result rows for one
    ``(dataset, sdb_column, factor)`` triple.
    """
    df = pd.DataFrame(df_records)
    result = _build_long_format(
        df, annotation_col, sdb_col, comment_key_col, factor
    )
    if result is None:
        return []

    annotations, groups, item_ids = result
    method_results = _all_method_results(annotations, groups, item_ids)

    return [
        _make_result_row(dataset_name, sdb_col, factor, len(annotations), *mr)
        for mr in method_results
    ]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_datasets(
    sap_path: Path,
    dices_350_path: Path,
    dices_990_path: Path,
    kumar_path: Path,
) -> list:
    """Return Dataset instances for every path that exists."""
    candidates = [
        (sap_path, lambda p: SapDataset(dataset_path=p)),
        (
            dices_350_path,
            lambda p: DicesDataset(dataset_path=p, variant="350"),
        ),
        (
            dices_990_path,
            lambda p: DicesDataset(dataset_path=p, variant="990"),
        ),
        (
            kumar_path,
            lambda p: KumarDataset(
                dataset_path=p,
                num_samples=KUMAR_NUM_SAMPLES,
                seed=KUMAR_SEED,
            ),
        ),
    ]
    return [
        factory(path) for path, factory in candidates if path and path.exists()
    ]


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


def _parse_cache_row(row: dict) -> dict:
    return {
        "stat": float(row["stat"]) if row["stat"] else float("nan"),
        "pvalue": (float(row["pvalue"]) if row["pvalue"] else float("nan")),
        "n_annotations": (
            int(row["n_annotations"]) if row["n_annotations"] else 0
        ),
    }


def _load_cache_rows(cache_path: Path) -> list[dict]:
    """Return all rows from the cache CSV as a list of dicts."""
    if not cache_path.exists():
        return []
    with open(cache_path, newline="") as fh:
        return [
            {
                "dataset": r["dataset"],
                "sdb_column": r["sdb_column"],
                "factor": r["factor"],
                "method": r["method"],
                **_parse_cache_row(r),
            }
            for r in csv.DictReader(fh)
        ]


def _append_to_cache(cache_path: Path, rows: list[dict]) -> None:
    """Append rows to the cache CSV, writing the header if new."""
    write_header = not cache_path.exists()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Job construction
# ---------------------------------------------------------------------------


def _is_fully_cached(
    dataset_name: str,
    sdb_col: str,
    factor: str,
    cached_keys: set[CacheKey],
) -> bool:
    needed = set(LATEX_METHOD_LABELS.keys())
    cached = {
        m
        for m in needed
        if (dataset_name, sdb_col, str(factor), m) in cached_keys
    }
    return cached == needed


def _jobs_for_dataset(
    ds,
    cached_keys: set[CacheKey],
) -> Iterator[Job]:
    """Yield jobs for any (sdb_column, factor) not yet fully cached."""
    df = ds.get_dataset()
    annotation_col = ds.get_annotation_column()
    comment_key_col = ds.get_comment_key_column()
    sdb_columns = [c for c in ds.get_sdb_columns() if c in df.columns]

    for sdb_col in sdb_columns:
        factors = df[sdb_col].explode().dropna().unique()
        df_records = df[[annotation_col, sdb_col, comment_key_col]].to_dict(
            orient="records"
        )

        for factor in factors:
            if _is_fully_cached(ds.get_name(), sdb_col, factor, cached_keys):
                continue
            yield (
                ds.get_name(),
                sdb_col,
                str(factor),
                annotation_col,
                comment_key_col,
                df_records,
            )


def _collect_jobs(datasets: list, cached_keys: set[CacheKey]) -> list[Job]:
    return [
        job for ds in datasets for job in _jobs_for_dataset(ds, cached_keys)
    ]


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------


def _process_future(
    future,
    cache_path: Path,
    all_rows: list[dict],
) -> None:
    """Persist and collect the rows produced by one completed future."""
    new_rows = future.result()
    if not new_rows:
        return
    _append_to_cache(cache_path, new_rows)
    all_rows.extend(new_rows)


def _run_jobs(
    jobs: list[Job],
    cache_path: Path,
    workers: int,
) -> list[dict]:
    """Run jobs in parallel, streaming results to cache as they finish."""
    print(
        f"Running {len(jobs)} (dataset, SDB, factor) combinations"
        f" ({workers} workers)\u2026"
    )
    all_new_rows: list[dict] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_one_factor, *job): job for job in jobs}
        for future in tqdm(futures, total=len(futures), desc="Computing"):
            _process_future(future, cache_path, all_new_rows)

    return all_new_rows


def run(
    datasets: list,
    cache_path: Path,
    workers: int,
    force: bool = False,
) -> list[dict]:
    """
    Compute results for all (dataset, sdb_column, factor) triples.

    If ``cache_path`` already exists and ``force`` is False, skip all
    computation and load directly from the cache.  Otherwise use the
    per-tuple cache to skip only the triples already present.

    Returns the full list of result dicts.
    """
    if skip_if_exists(cache_path) and not force:
        print(f"Loading cached results from {cache_path}")
        return _load_cache_rows(cache_path)

    cached_rows = _load_cache_rows(cache_path)
    cached_keys: set[CacheKey] = {
        (r["dataset"], r["sdb_column"], r["factor"], r["method"])
        for r in cached_rows
    }

    jobs = _collect_jobs(datasets, cached_keys)
    if not jobs:
        print("All results already cached.")
        return cached_rows

    new_rows = _run_jobs(jobs, cache_path, workers)
    return cached_rows + new_rows


# ---------------------------------------------------------------------------
# LaTeX export
# ---------------------------------------------------------------------------


def _fmt_cell(stat: float, pvalue: float) -> str:
    """Format ``(stat, pvalue)`` as ``value^{stars}`` or ``---``."""
    if math.isnan(stat) or math.isnan(pvalue):
        return "---"
    return f"{stat:.3f}{significance_superscript(pvalue)}"


def _filter_significant(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only (dataset, sdb_column, factor) triples with at least one
    p < ALPHA.
    """
    sig_index = (
        df[df["pvalue"] < ALPHA]
        .groupby(["dataset", "sdb_column", "factor"])
        .size()
        .index
    )
    idx = pd.MultiIndex.from_frame(df[["dataset", "sdb_column", "factor"]])
    return df[idx.isin(sig_index)]


def _ordered_methods(df: pd.DataFrame) -> list[str]:
    present = df["method"].unique().tolist()
    return [m for m in METHOD_ORDER if m in present] + [
        m for m in present if m not in METHOD_ORDER
    ]


def _build_display_row(
    dataset: str,
    sdb_col: str,
    factor: str,
    group_df: pd.DataFrame,
    ordered_methods: list[str],
) -> dict:
    row: dict = {
        "Dataset": dataset,
        r"\ac{pc}": sdb_col,
        "Factor": str(factor),
    }
    for method in ordered_methods:
        label = LATEX_METHOD_LABELS.get(method, method)
        match = group_df[group_df["method"] == method]
        if match.empty:
            row[label] = "---"
        else:
            row[label] = _fmt_cell(
                match["stat"].iloc[0], match["pvalue"].iloc[0]
            )
    return row


def _build_display_df(
    df_sig: pd.DataFrame,
    ordered_methods: list[str],
) -> pd.DataFrame:
    groups = df_sig.groupby(["dataset", "sdb_column", "factor"])
    display_rows = [
        _build_display_row(dataset, sdb_col, factor, group, ordered_methods)
        for (dataset, sdb_col, factor), group in groups
    ]
    display_df = pd.DataFrame(display_rows)

    for col in ["Dataset", r"\ac{pc}", "Factor"]:
        display_df[col] = display_df[col].str.replace("_", r"\_", regex=False)

    return display_df.set_index(["Dataset", r"\ac{pc}", "Factor"])


def export_latex(rows: list[dict], output_path: Path) -> None:
    """
    Export a single longtable.  Rows are (Dataset, SDB dimension, Factor);
    columns are one per method.  Only triples with at least one significant
    result are included.
    """
    if not rows:
        print("No results to export.")
        return

    df = pd.DataFrame(rows)
    df["pvalue"] = pd.to_numeric(df["pvalue"], errors="coerce")
    df["stat"] = pd.to_numeric(df["stat"], errors="coerce")

    df_sig = _filter_significant(df)
    if df_sig.empty:
        print("No statistically significant results found.")
        return

    ordered_methods = _ordered_methods(df_sig)
    display_df = _build_display_df(df_sig, ordered_methods)

    latex_str = display_df.to_latex(
        longtable=True,
        multirow=True,
        escape=False,
        na_rep="---",
        caption=(
            "Statistically significant results ($p < 0.05$) for all "
            "synthetic comparison metrics applied to each SDB dimension "
            "of the human annotation datasets. "
            "Stars denote significance level: "
            r"${}^{*}$\,$p{<}0.05$, "
            r"${}^{**}$\,$p{<}0.01$, "
            r"${}^{***}$\,$p{<}0.001$. "
            "Only rows with at least one significant result are shown."
        ),
        label="tab:human-dataset-significance",
        column_format="lll" + "r" * len(ordered_methods),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_str, encoding="utf-8")
    print(f"LaTeX table written to: {output_path}")

    n_sig = len(display_df)
    n_total = df.groupby(["dataset", "sdb_column", "factor"]).ngroups
    print(
        f"{n_sig}/{n_total} triples have at least one "
        f"significant result (\u03b1={ALPHA})."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Apply synthetic comparison metrics to human annotation "
            "datasets and export a single LaTeX table of statistically "
            "significant results."
        )
    )
    parser.add_argument("--sap-path", type=Path, required=True)
    parser.add_argument("--dices-350-path", type=Path, required=True)
    parser.add_argument("--dices-990-path", type=Path, required=True)
    parser.add_argument("--kumar-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--latex-output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if the cache file already exists.",
    )

    args = parser.parse_args()
    main(
        sap_path=args.sap_path,
        dices_350_path=args.dices_350_path,
        dices_990_path=args.dices_990_path,
        kumar_path=args.kumar_path,
        cache_dir=args.cache_dir,
        latex_output_dir=args.latex_output_dir,
        workers=args.workers,
        force=args.force,
    )
