from pathlib import Path
import itertools
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

from . import sap
from . import synthetic


def mean_pairwise_kappa(annotations: list) -> float | None:
    """
    Compute mean Cohen's kappa over all annotator pairs
    within a single group.
    """
    if len(annotations) < 2:
        return None

    kappas = []
    for a, b in itertools.combinations(annotations, 2):
        try:
            k = cohen_kappa_score(a, b)
            if not np.isnan(k):
                kappas.append(k)
        except ValueError:
            continue

    return np.mean(kappas) if kappas else None


def intergroup_kappa(group_a: list, group_b: list) -> float | None:
    """
    Mean Cohen's kappa across all cross-group annotator pairs.
    """
    kappas = []
    for a in group_a:
        for b in group_b:
            try:
                k = cohen_kappa_score(a, b)
                if not np.isnan(k):
                    kappas.append(k)
            except ValueError:
                continue

    return np.mean(kappas) if kappas else None


def main(sap_path: Path, synthetic_path: Path):
    sap_ds = sap.SapDataset(dataset_path=sap_path)
    synth_ds = synthetic.HundredDataset(dataset_path=synthetic_path)

    sap_df = sap_ds.get_dataset()
    synth_df = synth_ds.get_dataset()

    sap_key = sap_ds.get_comment_key_column()
    synth_key = synth_ds.get_comment_key_column()

    sap_ann = sap_ds.get_annotation_column()
    synth_ann = synth_ds.get_annotation_column()

    # ─────────────────────────────────────────────
    # Align datasets on shared comments
    # ─────────────────────────────────────────────
    merged = sap_df.merge(
        synth_df,
        left_on=sap_key,
        right_on=synth_key,
        suffixes=("_sap", "_syn"),
    )

    print(f"Shared comments: {len(merged)}")

    # ─────────────────────────────────────────────
    # Intragroup agreement
    # ─────────────────────────────────────────────
    sap_kappas = merged[f"{sap_ann}_sap"].apply(mean_pairwise_kappa)
    syn_kappas = merged[f"{synth_ann}_syn"].apply(mean_pairwise_kappa)

    print("Mean intragroup Cohen's κ")
    print(f"  SAP:       {sap_kappas.mean():.3f}")
    print(f"  Synthetic: {syn_kappas.mean():.3f}")

    # ─────────────────────────────────────────────
    # Intergroup agreement (per comment)
    # ─────────────────────────────────────────────
    merged["intergroup_kappa"] = merged.apply(
        lambda row: intergroup_kappa(
            row[f"{sap_ann}_sap"],
            row[f"{synth_ann}_syn"],
        ),
        axis=1,
    )

    print(
        f"Mean intergroup Cohen's κ: "
        f"{merged['intergroup_kappa'].mean():.3f}"
    )

    # ─────────────────────────────────────────────
    # Seaborn histogram
    # ─────────────────────────────────────────────
    sns.histplot(
        merged["intergroup_kappa"].dropna(),
        bins=30,
        kde=True,
    )
    plt.xlabel("Intergroup Cohen's κ")
    plt.ylabel("Number of comments")
    plt.title("Intergroup Annotation Agreement per Comment")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Classify forum comments using taxonomy categories and an LLM."
        )
    )
    parser.add_argument(
        "--sap-path",
        required=True,
        help="Path to the sap CSV file.",
    )
    parser.add_argument(
        "--synthetic-path",
        required=True,
        help="Directory for the synthetic sap CSV file.",
    )
    args = parser.parse_args()
    main(
        sap_path=Path(args.sap_path),
        synthetic_path=Path(args.synthetic_path)
    )
