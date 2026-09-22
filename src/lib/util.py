import re
from pathlib import Path

import pandas as pd


def skip_if_exists(path: Path) -> bool:
    """
    Returns True (and prints a message) if `path` already exists, so the
    caller can skip recomputing it. Centralized here so every experiment
    step uses the same check/logging behavior.
    """
    if path.exists():
        print(f"Skipping (already exists): {path}")
        return True
    return False


def center_table_latex(latex_str: str) -> str:
    latex_str = re.sub(
        r"(\\begin\{table\}\[[^\]]*\])",
        lambda m: m.group(1) + "\n\\centering",
        latex_str,
    )
    return latex_str


def trim_numeric_col_latex(col: pd.Series, float_format: str) -> pd.Series:
    return pd.to_numeric(
        col,
        errors="coerce",
    ).map(lambda x: "---" if pd.isna(x) else f"{x:{float_format}}")
