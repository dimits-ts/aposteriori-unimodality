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
    return _append_command_to_table_latex(
        latex_str=latex_str, command=r"\centering"
    )


def small_table_latex(latex_str: str, size: str = r"\small") -> str:
    return _append_command_to_table_latex(latex_str=latex_str, command=size)


def trim_numeric_col_latex(col: pd.Series, float_format: str) -> pd.Series:
    return pd.to_numeric(
        col,
        errors="coerce",
    ).map(lambda x: "---" if pd.isna(x) else f"{x:{float_format}}")


def significance_superscript(p):
    if pd.isna(p):
        return ""
    elif p < 0.001:
        return r"$^{***}$"
    elif p < 0.01:
        return r"$^{**}$"
    elif p < 0.05:
        return r"$^{*}$"
    else:
        return ""


def _append_command_to_table_latex(latex_str: str, command: str) -> str:
    return re.sub(
        r"(\\begin\{table\}\[[^\]]*\])",
        lambda m: m.group(1) + f"\n{command}",
        latex_str,
    )
