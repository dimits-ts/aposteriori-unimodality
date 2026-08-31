"""
Comparing apunim and agreement-based attribution.

Goal: To test whether different methods (like apunim and agreement metrics)
can correctly detect a polarization signal (a known 'group effect') embedded
in synthetic data.

Methodology:
1. All methods receive identical synthetic annotations where the size of the
   group effect (polarization) is set by design (ground truth).
2. Each method is asked the same question: Does this grouping explain the
   polarization?
3. We compare the detection rate (how often the method correctly flags the
   effect) against this known ground truth.

Context and Baselines:
- The agreement baseline (using Krippendorff's alpha) is adjusted to measure
  the *within-group* gain it implies, allowing for a fair comparison with
  apunim.
- The simulation uses a full crossed design (every annotator rates every item)
  to maximize the potential for agreement detection.
- This study compares apunim (the specific apunim arm used, documented by
  --label) against agreement-based methods and the original unimodality test
  (Pavlopoulos & Likas, 2024).
"""

import argparse
import csv
import math
import zlib
import functools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import apunim
import krippendorff
import matplotlib.pyplot as plt
import numpy as np

import tasks.graphs


N_LEVELS = 5
SIGMA = 1.2
ALPHA = 0.05
DELTAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# (annotators/item, minority share)
CONDITIONS = [
    (80, 0.5),
    (80, 0.2),
    (20, 0.5),
    (20, 0.2),
    (6, 0.5)
]

PLOT_NUM_COLS = 3


# --------------------------------------------------------------- generators


def _groups(n_ann, minority_frac):
    n_min = max(1, int(round(minority_frac * n_ann)))
    return np.array(["B"] * n_min + ["A"] * (n_ann - n_min))


def simulate(n_items, n_ann, delta, minority_frac, rng, sigma=SIGMA):
    """Group-driven polarization: the two groups sit `delta` apart."""
    groups = _groups(n_ann, minority_frac)
    base = rng.uniform(2.5, 3.5, size=n_items)
    shift = np.where(groups == "B", delta / 2, -delta / 2)

    vals = rng.normal(base[None, :] + shift[:, None], sigma)

    return np.clip(np.round(vals), 1, N_LEVELS), groups


# ------------------------------------------------------------------ methods


def _long(matrix, groups):
    n_ann, n_items = matrix.shape

    return (
        matrix.T.ravel(),
        np.tile(groups, n_items),
        np.repeat(np.arange(n_items), n_ann),
    )


def method_apunim(matrix, groups, seed):
    ann, fac, com = _long(matrix, groups)

    try:
        res = apunim.aposteriori_unimodality(
            ann,
            fac,
            com,
            num_bins=N_LEVELS,
            iterations=100,
            seed=seed,
        )
    except ValueError:
        # No comment passes the eligibility filter, which is the expected
        # outcome when there is no polarization to attribute (delta = 0).
        # That is a non-detection, not a failure.
        return 0.0, 1.0

    if not res:
        return 0.0, 1.0

    # Bonferroni correction over levels.
    best = min(res.items(), key=lambda kv: kv[1].pvalue)

    return best[1].apunim, min(1.0, best[1].pvalue * len(res))


def _alpha(matrix):
    return (
        np.nan
        if matrix.shape[0] < 2
        else krippendorff.alpha(
            reliability_data=matrix,
            level_of_measurement="ordinal",
        )
    )


def _delta_alpha(matrix, groups):
    overall = _alpha(matrix)

    within = [
        alpha
        for alpha in (
            _alpha(matrix[groups == group]) for group in np.unique(groups)
        )
        if not np.isnan(alpha)
    ]

    return (
        (float(np.mean(within)) - overall, overall)
        if within
        else (np.nan, np.nan)
    )


def method_delta_alpha(matrix, groups, seed, n_perm=200):
    obs, overall = _delta_alpha(matrix, groups)

    if np.isnan(obs):
        return np.nan, 1.0, overall

    rng = np.random.default_rng(seed)

    null = np.array(
        [
            _delta_alpha(matrix, rng.permutation(groups))[0]
            for _ in range(n_perm)
        ]
    )

    return (
        obs,
        (1 + np.sum(null >= obs)) / (1 + n_perm),
        overall,
    )


def _frac_explained(matrix, groups):
    levels = np.unique(groups)

    hits = 0
    eligible = 0

    for c in range(matrix.shape[1]):
        col = matrix[:, c]

        if apunim.dfu(col, bins=N_LEVELS) <= 0:
            continue

        eligible += 1

        if all(
            apunim.dfu(col[groups == group], bins=N_LEVELS) <= 0
            for group in levels
        ):
            hits += 1

    return (hits / eligible) if eligible else np.nan


def method_original_au(matrix, groups, seed, n_perm=100):
    obs = _frac_explained(matrix, groups)

    if np.isnan(obs):
        return np.nan, 1.0

    rng = np.random.default_rng(seed)

    null = np.array(
        [
            _frac_explained(matrix, rng.permutation(groups))
            for _ in range(n_perm)
        ]
    )

    null = null[~np.isnan(null)]

    return (
        obs,
        (1 + np.sum(null >= obs)) / (1 + len(null)),
    )


# ------------------------------------------------------------------- runner


def _seed(n_ann, minority, delta, rep):
    """
    Deterministic across processes and runs; hash() is salted per process.
    """
    key = f"{n_ann}|{minority}|{delta}|{rep}".encode()

    return zlib.crc32(key)


def _one(job, n_items, label):
    n_ann, minority, delta, rep = job

    rng = np.random.default_rng(_seed(n_ann, minority, delta, rep))

    matrix, groups = simulate(
        n_items,
        n_ann,
        delta,
        minority,
        rng,
    )

    rows = [(label, *method_apunim(matrix, groups, rep))]

    stat, pvalue, overall_alpha = method_delta_alpha(
        matrix,
        groups,
        rep,
    )

    rows += [
        (
            "Krippendorff delta-alpha",
            stat,
            pvalue,
        ),
        (
            "Krippendorff alpha (pooled)",
            overall_alpha,
            float("nan"),
        ),
        (
            "aposteriori unimodality (2024)",
            *method_original_au(matrix, groups, rep),
        ),
    ]

    return [
        (
            n_ann,
            minority,
            delta,
            rep,
            method,
            stat,
            pvalue,
            int(pvalue < ALPHA) if pvalue == pvalue else 0,
        )
        for method, stat, pvalue in rows
    ]


def run(out_csv: Path, n_items: int, n_reps: int, workers: int) -> None:
    jobs = [
        (n_ann, minority, delta, rep)
        for n_ann, minority in CONDITIONS
        for delta in DELTAS
        for rep in range(n_reps)
    ]

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)

        writer.writerow(
            [
                "n_ann",
                "minority",
                "delta",
                "rep",
                "method",
                "stat",
                "pvalue",
                "detected",
            ]
        )

        worker = functools.partial(_one, n_items=n_items, label="apunim")

        with ProcessPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(worker, jobs, chunksize=4):
                writer.writerows(result)

    print(f"wrote {out_csv} ({len(jobs)} runs)")


# ------------------------------------------------------------------ figure


# Drawn in this order, so the two apunim arms sit next to each other.
METHOD_ORDER = [
    "apunim",
    "Krippendorff delta-alpha",
    "aposteriori unimodality (2024)",
]

LEGEND_LABEL = {
    "Krippendorff delta-alpha": r"Krippendorff $\Delta\alpha$",
    "aposteriori unimodality (2024)": "aposteriori unim. (2024)",
}


def _condition_title(n_ann, minority):
    return (
        f"{n_ann} ann/item, " f"{int(round(minority * 100))}\\% minority"
    )


def plot(rows, methods, out_path):
    methods = [m for m in METHOD_ORDER if m in methods] + [
        m for m in methods if m not in METHOD_ORDER
    ]

    colors = tasks.graphs.COLORBLIND_PALETTE

    n_rows = math.ceil(len(CONDITIONS) / PLOT_NUM_COLS)

    fig, axes = plt.subplots(
        n_rows,
        PLOT_NUM_COLS,
        figsize=(PLOT_NUM_COLS * 5.0, n_rows * 2.8),
        sharey=True,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.86,
        bottom=0.18,
        wspace=0.25,
        hspace=0.35,
    )

    # Always work with a flat array, regardless of the grid shape.
    axes = axes.ravel()

    for ax, (n_ann, minority) in zip(axes, CONDITIONS):
        for i, method in enumerate(methods):
            y = [
                np.mean(
                    [
                        row["detected"]
                        for row in rows
                        if row["method"] == method
                        and row["n_ann"] == n_ann
                        and row["minority"] == minority
                        and row["delta"] == delta
                    ]
                    or [np.nan]
                )
                for delta in DELTAS
            ]

            ax.plot(
                DELTAS,
                y,
                marker=tasks.graphs.MARKERS[i % len(tasks.graphs.MARKERS)],
                color=colors[i % len(colors)],
                label=LEGEND_LABEL.get(method, method),
                lw=1.4,
                ms=3.8,
            )

        ax.set_title(
            _condition_title(n_ann, minority)
        )
        ax.set_ylim(-0.04, 1.08)

    # Hide unused axes when len(CONDITIONS) is not divisible by
    # PLOT_NUM_COLS.
    for ax in axes[len(CONDITIONS):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower right",
        title="Attribution Methodology"
    )
    fig.suptitle("Apunim beats baselines on group polarization attribution")
    fig.supylabel("Detection rate")
    fig.supxlabel(r"Group effect size $\delta$")

    tasks.graphs.save_plot(out_path)


# --------------------------------------------------------------------- main


def main(
    cache_dir: Path,
    graph_output_dir: Path,
    n_items: int,
    n_reps: int,
    workers: int,
) -> None:
    tasks.graphs.graph_setup()

    # graph_setup sizes type for full-width figures; this one is a
    # multi-panel strip.
    """
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
        }
    )
    """

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_csv = cache_dir / "metric-comparison.csv"

    if cache_csv.exists():
        print(f"loading cached results from {cache_csv}")
    else:
        run(
            cache_csv,
            n_items,
            n_reps,
            workers
        )

    with open(cache_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))

    for row in rows:
        row["delta"] = float(row["delta"])
        row["n_ann"] = int(row["n_ann"])
        row["minority"] = float(row["minority"])
        row["detected"] = int(row["detected"])
        row["stat"] = float(row["stat"])

    methods = [
        method
        for method in dict.fromkeys(row["method"] for row in rows)
        if method != "Krippendorff alpha (pooled)"
    ]

    plot(
        rows,
        methods,
        Path(graph_output_dir) / "metric_comparison.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compare apunim against agreement-based attribution "
            "on synthetic data with a known group effect."
        )
    )

    parser.add_argument(
        "--cache-dir",
        required=True,
        help="Directory used to cache calculation results as CSV.",
    )

    parser.add_argument(
        "--graph-output-dir",
        required=True,
        help="Directory for graphs.",
    )

    parser.add_argument(
        "--n-items",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--n-reps",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=7,
    )

    args = parser.parse_args()

    main(
        Path(args.cache_dir),
        Path(args.graph_output_dir),
        args.n_items,
        args.n_reps,
        args.workers
    )
