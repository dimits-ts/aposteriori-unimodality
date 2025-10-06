from pathlib import Path

import matplotlib.pyplot as plt


def save_plot(path: Path) -> None:
    """
    Saves a plot to the specified filepath.

    :param path:
        The full path (including filename) where the plot will be saved.
    :type path: pathlib.Path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    print(f"Figure saved to {path.resolve()}")
