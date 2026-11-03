"""
Inherent (unattributable) polarization of synthetic annotations, per number
of annotator groups, in the style of the human apriori.png figure.

For every number of groups k in GROUP_COUNTS, items are simulated with
N_ANNOTATORS annotators each (k = 2 with simulation.simulate, k >= 3 with
simulation.simulate_multigroup; bidirectional, annotator-specific group
effects). The polarizing group holds a 1/k share of the annotators, so all
k groups have the same size and only the number of groups varies between
the series. The same number of items is drawn for every delta in
simulation.DELTAS and the items are pooled, so each series covers
everything from unpolarized to strongly polarized items.

Inherent polarization is the minimum nDFU over all annotator groups of an
item (run_helper.compute_inherent_polarization_exhaustive); it ignores the
group labels. The figure shows, per k, its distribution with all
annotators (right of the centre line) and with every item repeatedly
subsampled down to at most run_helper.INHERENT_SUBSAMPLE_SIZE annotators
(left), as for the human datasets.

Per-k values are cached as "synthetic-k<k>-inherent.csv" and
"synthetic-k<k>-inherent-subsampled-n6.csv" in --cache-dir; delete them to
recompute.
"""

import argparse
import functools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import numpy as np

from ..lib import graphs, labels, run_helper
from ..lib.preprocessing import Dataset
from ..lib.util import stable_seed
from .simulation import DELTAS, simulate, simulate_multigroup


# Numbers of annotator groups, one series (colour/hatch) each.
GROUP_COUNTS = (2, 3, 5, 10)
# Annotators per item. 30 is the smallest size of the multi-group
# conditions that gives every group of the k = 10 case 3 annotators.
N_ANNOTATORS = 30
UNIDIRECTIONAL = False  # bidirectional simulation
FIXED_GROUP_EFFECT = False  # annotator-specific group effects

FILE_PREFIX = "synthetic"
FIGURE_TITLE = f"{labels.USD.short} in Synthetic Data per Number of Groups"


class SyntheticDataset(Dataset):
    """
    Simulated annotation matrix (annotators x items) in the per-item,
    list-of-annotators shape that lib.run_helper expects. Every item is
    annotated by the same annotators, whose group labels are the only SDB
    column.
    """

    def __init__(self, matrix: np.ndarray, groups: np.ndarray, item_ids):
        n_items = matrix.shape[1]
        self.df = pd.DataFrame(
            {
                "item_id": list(item_ids),
                "annotations": [list(matrix[:, j]) for j in range(n_items)],
                "group": [list(groups)] * n_items,
            }
        )

    def get_name(self) -> str:
        return "synthetic"

    def get_dataset(self) -> pd.DataFrame:
        return self.df

    def get_sdb_columns(self) -> list[str]:
        return ["group"]

    def get_annotation_column(self) -> str:
        return "annotations"

    def get_comment_key_column(self) -> str:
        return "item_id"

    def get_text_column(self) -> str:
        return "item_id"


def main(
    cache_dir: Path,
    graph_output_path: Path,
    n_items: int,
    workers: int,
) -> None:
    graphs.graph_setup()
    full, subsampled = load_or_compute(cache_dir, n_items, workers)
    plot(full, subsampled, graph_output_path)


def group_label(k: int) -> str:
    return f"{k} groups"


def full_path(cache_dir: Path, k: int) -> Path:
    return cache_dir / f"{FILE_PREFIX}-k{k}-inherent.csv"


def subsampled_path(cache_dir: Path, k: int) -> Path:
    return (
        cache_dir
        / f"{FILE_PREFIX}-k{k}{run_helper.INHERENT_SUBSAMPLED_STEM_SUFFIX}.csv"
    )


def load_or_compute(
    cache_dir: Path, n_items: int, workers: int
) -> tuple[dict[int, pd.Series], dict[int, pd.Series]]:
    """
    Per-k inherent polarization with all annotators and subsampled, read
    from `cache_dir` where both files of a k exist, computed (and cached)
    otherwise. `n_items` is the number of items per delta.
    """
    missing = [
        k
        for k in GROUP_COUNTS
        if not (
            full_path(cache_dir, k).exists()
            and subsampled_path(cache_dir, k).exists()
        )
    ]
    if missing:
        _compute(missing, cache_dir, n_items, workers)
    else:
        print(f"Loading cached synthetic inherent polarization: {cache_dir}")

    full = {
        k: graphs.read_inherent_csv(full_path(cache_dir, k))
        for k in GROUP_COUNTS
    }
    subsampled = {
        k: graphs.read_inherent_csv(subsampled_path(cache_dir, k))
        for k in GROUP_COUNTS
    }
    return full, subsampled


def _compute(
    group_counts: list[int], cache_dir: Path, n_items: int, workers: int
) -> None:
    jobs = [(k, delta) for k in group_counts for delta in DELTAS]
    worker = functools.partial(_one, n_items=n_items)

    results: dict[int, list[tuple[pd.Series, pd.Series]]] = {
        k: [] for k in group_counts
    }
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for (k, _), result in zip(jobs, executor.map(worker, jobs)):
            results[k].append(result)

    cache_dir.mkdir(parents=True, exist_ok=True)
    for k, parts in results.items():
        for path, series in (
            (full_path(cache_dir, k), pd.concat([p[0] for p in parts])),
            (subsampled_path(cache_dir, k), pd.concat([p[1] for p in parts])),
        ):
            series.rename("inherent_polarization").to_csv(
                path, header=True, index_label="comment"
            )
            print(f"wrote {path}")


def simulate_items(
    k: int, delta: float, n_items: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Annotations (N_ANNOTATORS x n_items) and group labels for k equally
    sized groups, one of which polarizes."""
    if k == 2:
        return simulate(
            n_items,
            N_ANNOTATORS,
            delta,
            1 / k,
            rng,
            unidirectional=UNIDIRECTIONAL,
            fixed_group_effect=FIXED_GROUP_EFFECT,
        )
    return simulate_multigroup(
        n_items,
        N_ANNOTATORS,
        k,
        delta,
        1 / k,
        rng,
        unidirectional=UNIDIRECTIONAL,
        fixed_group_effect=FIXED_GROUP_EFFECT,
    )


def _one(job, n_items: int) -> tuple[pd.Series, pd.Series]:
    """Inherent polarization (all annotators, subsampled) of the items of
    one (k, delta) job."""
    k, delta = job
    seed = stable_seed("inherent", k, delta)
    matrix, groups = simulate_items(
        k, delta, n_items, np.random.default_rng(seed)
    )
    ds = SyntheticDataset(
        matrix, groups, item_ids=[f"d{delta}-{j}" for j in range(n_items)]
    )
    full = run_helper.compute_inherent_polarization_exhaustive(ds)
    subsampled = run_helper.compute_inherent_polarization_subsampled(
        ds,
        size=run_helper.INHERENT_SUBSAMPLE_SIZE,
        n_repeats=run_helper.INHERENT_SUBSAMPLE_REPEATS,
        seed=seed,
    )
    return full, subsampled


def plot(
    full: dict[int, pd.Series],
    subsampled: dict[int, pd.Series],
    output_path: Path,
) -> None:
    """Mirrored USD histograms, one colour/hatch per number of groups."""
    graphs.plot_mirrored_inherent_histograms(
        right={group_label(k): v for k, v in full.items()},
        left={group_label(k): v for k, v in subsampled.items()},
        output_path=output_path,
        title=FIGURE_TITLE,
        right_label=rf"All {N_ANNOTATORS} annotators",
        left_label=rf"Max {run_helper.INHERENT_SUBSAMPLE_SIZE} annotators",
        order=[group_label(k) for k in sorted(GROUP_COUNTS)],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Inherent polarization of synthetic annotations per number of "
            "annotator groups, with all annotators and subsampled down to "
            f"{run_helper.INHERENT_SUBSAMPLE_SIZE} (mirrored histograms)."
        )
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="Directory for the per-k inherent-polarization CSVs.",
    )
    parser.add_argument(
        "--graph-output-path",
        required=True,
        help="Path for the output graph.",
    )
    parser.add_argument(
        "--n-items",
        type=int,
        default=30,
        help=(
            "Items per delta and number of groups "
            f"({len(DELTAS)} deltas are pooled)."
        ),
    )
    parser.add_argument("--workers", type=int, default=7)

    args = parser.parse_args()

    main(
        Path(args.cache_dir),
        Path(args.graph_output_path),
        args.n_items,
        args.workers,
    )
