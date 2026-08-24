"""Recovery of a known group effect: apunim vs agreement-based attribution.

Every method is handed identical synthetic annotations, in which the size of
the group effect is set by construction, and is asked the same question: does
this grouping explain the polarization? Reporting detection rate against a
known ground truth is what lets polarization and agreement metrics be compared
at all, given that they rest on different assumptions and are not otherwise
commensurable.

Baselines
---------
Krippendorff's alpha on its own measures agreement, not attribution, so the
agreement baseline is the *within-group gain* it implies: split the annotators
on the grouping and ask whether alpha rises. That difference is calibrated with
a label permutation test, so it gets a p-value on the same footing as apunim.

The original aposteriori unimodality test (Pavlopoulos & Likas, 2024) flags a
comment when it is polarized overall while every group is internally unimodal.

Annotations use a full crossed design (every annotator rates every item), which
is the best case for agreement metrics: no missing cells, nothing for them to
be handicapped by.

The apunim arm reflects whichever apunim version is installed; --label records
it, so running this twice against two pinned versions produces both arms of the
bin-grid comparison.
"""

import argparse
import csv
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import krippendorff
import apunim

import tasks.graphs

N_LEVELS = 5
SIGMA = 1.2       # wide enough that samples reach both ends of the 1-5 scale
ALPHA = 0.05
DELTAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
CONDITIONS = [(80, 0.5), (20, 0.5), (80, 0.2)]   # (annotators/item, minority share)


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


def simulate_group_variance(n_items, n_ann, minority_frac, rng,
                            sigma_lo=0.8, sigma_hi=2.5):
    """Group causes disagreement but NOT polarization: shared mean, group B
    simply noisier. Splitting on it raises within-group agreement, so an
    agreement-based test should fire even though there is no bimodality."""
    groups = _groups(n_ann, minority_frac)
    base = rng.uniform(2.5, 3.5, size=n_items)
    sd = np.where(groups == "B", sigma_hi, sigma_lo)
    return np.clip(np.round(rng.normal(base[None, :], sd[:, None])), 1, N_LEVELS), groups


def simulate_inherent(n_items, n_ann, minority_frac, rng, sigma=SIGMA):
    """Inherent polarization: every item is bimodal, but the split is
    independent of the grouping, so nothing is attributable to it."""
    groups = _groups(n_ann, minority_frac)
    pole = rng.random((n_ann, n_items)) < 0.5
    return np.clip(np.round(rng.normal(np.where(pole, 2.0, 4.0), sigma)), 1, N_LEVELS), groups


# ------------------------------------------------------------------ methods

def _long(matrix, groups):
    n_ann, n_items = matrix.shape
    return (matrix.T.ravel(), np.tile(groups, n_items),
            np.repeat(np.arange(n_items), n_ann))


def method_apunim(matrix, groups, seed):
    ann, fac, com = _long(matrix, groups)
    try:
        res = apunim.aposteriori_unimodality(
            ann, fac, com, num_bins=N_LEVELS, iterations=100, seed=seed)
    except ValueError:
        # No comment passes the eligibility filter, which is the expected
        # outcome when there is no polarization to attribute (delta = 0).
        # That is a non-detection, not a failure.
        return 0.0, 1.0
    if not res:
        return 0.0, 1.0
    best = min(res.items(), key=lambda kv: kv[1].pvalue)   # Bonferroni over levels
    return best[1].apunim, min(1.0, best[1].pvalue * len(res))


def _alpha(m):
    return np.nan if m.shape[0] < 2 else krippendorff.alpha(
        reliability_data=m, level_of_measurement="ordinal")


def _delta_alpha(matrix, groups):
    overall = _alpha(matrix)
    within = [a for a in (_alpha(matrix[groups == g]) for g in np.unique(groups))
              if not np.isnan(a)]
    return (float(np.mean(within)) - overall, overall) if within else (np.nan, np.nan)


def method_delta_alpha(matrix, groups, seed, n_perm=200):
    obs, overall = _delta_alpha(matrix, groups)
    if np.isnan(obs):
        return np.nan, 1.0, overall
    rng = np.random.default_rng(seed)
    null = np.array([_delta_alpha(matrix, rng.permutation(groups))[0]
                     for _ in range(n_perm)])
    return obs, (1 + np.sum(null >= obs)) / (1 + n_perm), overall


def _frac_explained(matrix, groups):
    levels = np.unique(groups)
    hits = eligible = 0
    for c in range(matrix.shape[1]):
        col = matrix[:, c]
        if apunim.dfu(col, bins=N_LEVELS) <= 0:
            continue
        eligible += 1
        hits += all(apunim.dfu(col[groups == g], bins=N_LEVELS) <= 0 for g in levels)
    return (hits / eligible) if eligible else np.nan


def method_original_au(matrix, groups, seed, n_perm=100):
    obs = _frac_explained(matrix, groups)
    if np.isnan(obs):
        return np.nan, 1.0
    rng = np.random.default_rng(seed)
    null = np.array([_frac_explained(matrix, rng.permutation(groups))
                     for _ in range(n_perm)])
    null = null[~np.isnan(null)]
    return obs, (1 + np.sum(null >= obs)) / (1 + len(null))


# ------------------------------------------------------------------- runner

_CFG = {}


def _seed(kind, n_ann, minority, delta, rep):
    """Deterministic across processes and runs; hash() is salted per process."""
    key = f"{kind}|{n_ann}|{minority}|{delta}|{rep}".encode()
    return zlib.crc32(key)


def _one(job):
    kind, n_ann, minority, delta, rep = job
    rng = np.random.default_rng(_seed(*job))
    n_items = _CFG["n_items"]
    if kind == "polarization":
        m, g = simulate(n_items, n_ann, delta, minority, rng)
    elif kind == "group-variance":
        m, g = simulate_group_variance(n_items, n_ann, minority, rng)
    elif kind == "inherent":
        m, g = simulate_inherent(n_items, n_ann, minority, rng)
    elif kind == "placebo":            # real polarization, grouping shuffled
        m, g = simulate(n_items, n_ann, delta, minority, rng)
        g = rng.permutation(g)

    rows = [(_CFG["label"], *method_apunim(m, g, rep))]
    s, p, a = method_delta_alpha(m, g, rep)
    rows += [("Krippendorff delta-alpha", s, p),
             ("Krippendorff alpha (pooled)", a, float("nan")),
             ("aposteriori unimodality (2024)", *method_original_au(m, g, rep))]
    return [(kind, n_ann, minority, delta, rep, meth, st, pv,
             int(pv < ALPHA) if pv == pv else 0) for meth, st, pv in rows]


def _init(cfg):
    _CFG.update(cfg)


def run(out_csv, n_items, n_reps, workers):
    jobs = [("polarization", n, mi, d, r)
            for n, mi in CONDITIONS for d in DELTAS for r in range(n_reps)]
    jobs += [(k, 80, 0.5, 2.5, r)
             for k in ("group-variance", "inherent", "placebo") for r in range(n_reps)]
    cfg = dict(_CFG)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "n_ann", "minority", "delta", "rep",
                    "method", "stat", "pvalue", "detected"])
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=_init, initargs=(cfg,)) as ex:
            for res in ex.map(_one, jobs, chunksize=4):
                w.writerows(res)
    print(f"wrote {out_csv} ({len(jobs)} runs)")


# ------------------------------------------------------------------ figure

#: drawn in this order, so the two apunim arms sit next to each other
METHOD_ORDER = ["apunim (fixed binning)", "apunim (as shipped, 1.0.5)",
                "Krippendorff delta-alpha", "aposteriori unimodality (2024)"]
LEGEND_LABEL = {"Krippendorff delta-alpha": r"Krippendorff $\Delta\alpha$",
                "aposteriori unimodality (2024)": "aposteriori unim.\ (2024)"}


def plot(rows, methods, out_path):
    tasks.graphs.graph_setup()
    import matplotlib.pyplot as plt
    # graph_setup sizes type for full-width figures; this one is a 3-panel strip.
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8,
                         "axes.labelsize": 9, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "legend.fontsize": 7})
    methods = ([m for m in METHOD_ORDER if m in methods]
               + [m for m in methods if m not in METHOD_ORDER])
    colors = tasks.graphs.COLORBLIND_PALETTE
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(9.2, 2.6), sharey=True)
    titles = ["80 annotators/item, balanced", "20 annotators/item, balanced",
              "80 annotators/item, 20\\% minority"]
    for ax, (n_ann, minority), title in zip(axes, CONDITIONS, titles):
        for i, m in enumerate(methods):
            y = [np.mean([r["detected"] for r in rows
                          if r["scenario"] == "polarization" and r["method"] == m
                          and r["n_ann"] == n_ann and r["minority"] == minority
                          and r["delta"] == d] or [np.nan]) for d in DELTAS]
            ax.plot(DELTAS, y, marker=tasks.graphs.MARKERS[i % len(tasks.graphs.MARKERS)],
                    color=colors[i % len(colors)],
                    label=LEGEND_LABEL.get(m, m), lw=1.4, ms=3.8)
        ax.axhline(ALPHA, ls=":", c="0.5", lw=1)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(r"group effect size $\delta$")
        ax.set_ylim(-0.04, 1.08)
    axes[0].set_ylabel("detection rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print("wrote", out_path)


def main(output_dir, graph_output_dir, label, n_items, n_reps, workers, plot_only):
    _CFG.update(label=label, n_items=n_items)
    out_csv = Path(output_dir) / "metric-comparison.csv"
    if not plot_only:
        run(out_csv, n_items, n_reps, workers)
    rows = list(csv.DictReader(open(out_csv)))
    for r in rows:
        r["delta"] = float(r["delta"]); r["n_ann"] = int(r["n_ann"])
        r["minority"] = float(r["minority"]); r["detected"] = int(r["detected"])
        r["stat"] = float(r["stat"])
    methods = [m for m in dict.fromkeys(r["method"] for r in rows)
               if m != "Krippendorff alpha (pooled)"]
    plot(rows, methods, Path(graph_output_dir) / "metric_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare apunim against agreement-based attribution on "
                    "synthetic data with a known group effect.")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for the CSV result files.")
    parser.add_argument("--graph-output-dir", required=True,
                        help="Directory for graphs.")
    parser.add_argument("--label", default="apunim",
                        help="Name for the apunim arm; record the installed "
                             "apunim version when comparing across versions.")
    parser.add_argument("--n-items", type=int, default=200)
    parser.add_argument("--n-reps", type=int, default=40)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--plot-only", action="store_true",
                        help="Re-draw the figure from an existing CSV.")
    args = parser.parse_args()
    main(Path(args.output_dir), Path(args.graph_output_dir), args.label,
         args.n_items, args.n_reps, args.workers, args.plot_only)
