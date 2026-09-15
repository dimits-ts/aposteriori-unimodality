from pathlib import Path


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