from pathlib import Path
import itertools
import argparse

import krippendorff
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from . import kumar
from . import synthetic


def cross_group_agreement(
    human_annotations: list[list[int]], synthetic_annotations: list[list[int]]
) -> NDArray:
    """
    Computes mean agreement between two annotation groups for one comment.
    Returns np.nan if either group is empty.
    """
    agreement_per_comment = []
    for human_comment_annotations, synthetic_comment_annotations in zip(
        human_annotations, synthetic_annotations
    ):
        agreement = krippendorff.alpha(
            [human_comment_annotations, synthetic_comment_annotations],
            level_of_measurement="ordinal",
        )
        agreement_per_comment.append(agreement)

    return np.array(agreement_per_comment)


def main(kumar_path: Path, synthetic_path: Path):

    kumar_ds = kumar.KumarDataset(dataset_path=kumar_path)
    synth_ds = synthetic.HundredDataset(dataset_path=synthetic_path)

    kumar_df = kumar_ds.get_dataset()
    synth_df = synth_ds.get_dataset()

    kumar_key = kumar_ds.get_comment_key_column()
    synth_key = synth_ds.get_comment_key_column()

    # ─────────────────────────────────────────────
    # Align datasets on shared comments
    # ─────────────────────────────────────────────
    merged_df = kumar_df.merge(
        synth_df,
        left_on=kumar_key,
        right_on=synth_key,
        suffixes=("_kumar", "_syn"),
    )

    print(f"Shared comments: {len(merged_df)}")

    agreements = cross_group_agreement(
        human_annotations=merged_df.Toxicity_kumar,
        synthetic_annotations=merged_df.Toxicity_syn,
    )
    print(agreements.mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Classify forum comments using taxonomy categories and an LLM."
        )
    )
    parser.add_argument(
        "--kumar-path",
        required=True,
        help="Path to the kumar CSV file.",
    )
    parser.add_argument(
        "--synthetic-path",
        required=True,
        help="Directory for the synthetic kumar CSV file.",
    )
    args = parser.parse_args()
    main(
        kumar_path=Path(args.kumar_path),
        synthetic_path=Path(args.synthetic_path),
    )
