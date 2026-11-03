from pathlib import Path

import pandas as pd

from src.lib.preprocessing import DicesDataset

PATHS = {
    "990": "data/datasets/dices/990/diverse_safety_adversarial_dialog_990.csv"
}

for variant, path in PATHS.items():
    df = DicesDataset(dataset_path=Path(path), variant=variant).get_dataset()

    # one list of gender labels per item
    labels = pd.Series([g for items in df["Gender"] for g in items])
    n_nb = df["Gender"].apply(
        lambda items: sum("binary" in str(g).lower() for g in items)
    )

    print(f"DICES-{variant}: {len(df)} items")
    print("  gender labels:", labels.value_counts().to_dict())
    print(
        "  non-binary raters per item:",
        n_nb.value_counts().sort_index().to_dict(),
    )
    print(
        f"  items with >= 3: {(n_nb >= 3).sum()} | items with > 3: {(n_nb > 3).sum()}"
    )
