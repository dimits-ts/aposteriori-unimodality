import re
import zlib
from pathlib import Path

import pandas as pd


def skip_if_exists(path: Path, quiet: bool = False) -> bool:
    """
    Returns True if `path` already exists, so the caller can skip
    recomputing it, and prints a message unless `quiet` (e.g. inside
    progress bars, which plain prints break up). Centralized here so every
    experiment step uses the same check/logging behavior.
    """
    if path.exists():
        if not quiet:
            print(f"Skipping (already exists): {path}")
        return True
    return False


def stable_seed(*args) -> int:
    """
    Deterministic seed across processes and runs, derived from args.
    hash() is salted per process, so use crc32 instead.
    """
    key = "|".join(str(a) for a in args).encode()
    return zlib.crc32(key)


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


# Single source of truth for significance stars: (p-value threshold, stars),
# ordered from strictest to loosest. Both the star rendering below and the
# table-caption legend are derived from this, so they cannot drift apart.
SIGNIFICANCE_LEVELS = (
    (0.001, "***"),
    (0.01, "**"),
    (0.05, "*"),
)

# LaTeX legend, e.g. "$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$."
# Append this to any table caption whose cells use significance_superscript.
SIGNIFICANCE_LEGEND = (
    ", ".join(
        f"$^{{{stars}}}p<{threshold}$"
        for threshold, stars in reversed(SIGNIFICANCE_LEVELS)
    )
    + "."
)


def significance_superscript(p):
    if pd.isna(p):
        return ""
    for threshold, stars in SIGNIFICANCE_LEVELS:
        if p < threshold:
            return f"$^{{{stars}}}$"
    return ""


def _append_command_to_table_latex(latex_str: str, command: str) -> str:
    return re.sub(
        r"(\\begin\{table\}\[[^\]]*\])",
        lambda m: m.group(1) + f"\n{command}",
        latex_str,
    )
