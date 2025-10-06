import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats

from .tasks import graphs
from .apunim import aposteriori


TITLE_SIZE = 20
LABEL_SIZE = 16
LESSER_LABEL_SIZE = 14
DIFF_COMMENTS_SIZE = 200
INTUITION_SIZE = 50
NUM_BINS = 10


def plot_annotation_distributions(
    graph_dir: Path,
    n_annotators: int = 10,
    n_annotations: int = 100,
    variance: float = 0.3,
    random_seed: int = 42,
) -> None:
    np.random.seed(random_seed)

    unimodal = [
        np.random.normal(loc=5, scale=variance, size=n_annotations)
        for _ in range(n_annotators)
    ]
    bimodal = [
        np.concatenate(
            [
                np.random.normal(3, variance, n_annotations // 2),
                np.random.normal(5, variance, n_annotations // 2),
            ]
        )
        for _ in range(n_annotators)
    ]
    multimodal = [
        np.concatenate(
            [
                np.random.normal(2, variance, n_annotations // 3),
                np.random.normal(5, variance, n_annotations // 3),
                np.random.normal(8, variance, n_annotations // 3),
            ]
        )
        for _ in range(n_annotators)
    ]
    uniform = [
        np.random.uniform(0, 10, n_annotations) for _ in range(n_annotators)
    ]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    def plot_matrix(ax, data, title):
        all_x = []
        all_y = []
        for i, y in enumerate(data):
            all_x.extend([i] * len(y))
            all_y.extend(y)
        all_x = np.array(all_x)
        all_y = np.array(all_y)

        # Compute point density
        xy = np.vstack([all_x, all_y])
        z = scipy.stats.gaussian_kde(xy)(xy)

        sc = ax.scatter(
            all_x,
            all_y,
            c=z,
            s=30,
            edgecolor="black",
            cmap="viridis",
            alpha=0.7,
        )
        ax.set_title(title)
        ax.set_xticks(range(n_annotators))
        ax.set_ylim(1, 10)
        plt.colorbar(sc, ax=ax, label="Density")

    plot_matrix(axs[0, 0], unimodal, "Low Disagreement\nLow Polarization")
    plot_matrix(axs[0, 1], uniform, "High Disagreement\nLow Polarization")
    plot_matrix(axs[1, 0], bimodal, "Low Disagreement\nHigh Polarization")
    plot_matrix(axs[1, 1], multimodal, "High Disagreement\nHigh Polarization")

    graphs.save_plot(graph_dir / "disagreement_vs_polarization.png")


def dfu_plots(colors: list[str], graph_dir: Path) -> None:
    d1 = _truncated_normal(loc=2, scale=1.3, size=INTUITION_SIZE)
    d2 = _truncated_normal(loc=8, scale=1.3, size=INTUITION_SIZE)
    d_all = np.hstack([d1, d2])

    _dfu_plot(
        data=d1,
        graph_path=graph_dir / "ndfu_men.png",
        label="men",
        color=colors[0],
    )
    _dfu_plot(
        data=d2,
        graph_path=graph_dir / "ndfu_women.png",
        label="women",
        color=colors[1],
    )
    _dfu_plot(
        data=d_all,
        graph_path=graph_dir / "ndfu_all.png",
        label="all",
        color=colors[2],
    )
    _combined_dfu_plot(
        datasets=[d1, d2, d_all],
        graph_path=graph_dir / "ndfu_combined.png",
        labels=["men", "women", "all"],
        colors=colors,
    )


def _combined_dfu_plot(
    datasets: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    graph_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    for data, label, color in zip(datasets, labels, colors):
        sns.histplot(
            data,
            bins=NUM_BINS,
            kde=True,
            alpha=0.7,
            color=color,
            label=label,
        )

    # Add all nDFU annotations
    math_text = ""
    for data, label in zip(datasets, labels):
        ndfu_value = aposteriori.dfu(data, bins=NUM_BINS, normalized=True)
        math_text += f"$\\mathbf{{nDFU_{{{label}}}}}={ndfu_value:.3f}$\n"

    plt.legend(loc="center")
    plt.xlabel(f"Toxicity\n{math_text}", fontsize=LABEL_SIZE)
    plt.ylabel(r"\#Comments", fontsize=LABEL_SIZE)

    graphs.save_plot(graph_path)
    plt.close()


def discussion_example(graph_dir: Path) -> None:
    misogynist_comment = """
    ``A: Men are naturally more suited for leadership because they’re more
    decisive. Women are too emotional to handle real responsibility.``
    """
    misandrist_comment = """
    ``B: Men are just too aggressive and can’t work with others.
    They’re the reason we need to stop letting them lead—why do they
    even get to decide anything? ``
    """
    discussion_comment = f"{misogynist_comment}\n{misandrist_comment}"

    d_woman_comment1 = _truncated_normal(
        loc=2, scale=1, size=DIFF_COMMENTS_SIZE
    )
    d_woman_comment2 = _truncated_normal(
        loc=6, scale=1, size=DIFF_COMMENTS_SIZE
    )
    d_man_comment1 = _truncated_normal(loc=6, scale=1, size=DIFF_COMMENTS_SIZE)
    d_man_comment2 = _truncated_normal(loc=2, scale=1, size=DIFF_COMMENTS_SIZE)

    d_woman = np.hstack([d_woman_comment1, d_woman_comment2])
    d_man = np.hstack([d_man_comment1, d_man_comment2])

    _plot_example_individual(
        misogynist_comment,
        d_woman_comment2,
        d_man_comment2,
        graph_dir / "ndfu_comment2.png",
    )
    _plot_example_individual(
        misandrist_comment,
        d_woman_comment1,
        d_man_comment1,
        graph_dir / "ndfu_comment1.png",
    )
    _plot_example_individual(
        discussion_comment,
        d_woman,
        d_man,
        graph_dir / "ndfu_discussion.png",
    )


def _truncated_normal(loc, scale, lower=0, upper=10, size=100):
    a, b = (lower - loc) / scale, (upper - loc) / scale
    return scipy.stats.truncnorm(a, b, loc=loc, scale=scale).rvs(size)


def _dfu_plot(data: np.ndarray, graph_path: Path, color: str, label) -> None:
    ndfu_value = aposteriori.dfu(data, bins=NUM_BINS, normalized=True)
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data,
        bins=NUM_BINS,
        kde=True,
        alpha=0.7,
        color=color,
        label=label,
    )
    math_text = f"$\\mathbf{{nDFU_{{{label}}}}}={ndfu_value:.3f}$"
    plt.xlabel(f"Toxicity\n{math_text}", fontsize=LABEL_SIZE)
    plt.ylabel(r"\#Comments", fontsize=LABEL_SIZE)

    graphs.save_plot(graph_path)
    plt.close()


def _plot_example_individual(
    title: str,
    women_annot: list[float],
    men_annot: list[float],
    graph_path: Path,
):
    ndfu_man = aposteriori.dfu(men_annot, bins=NUM_BINS, normalized=True)
    ndfu_woman = aposteriori.dfu(women_annot, bins=NUM_BINS, normalized=True)
    ndfu_all = aposteriori.dfu(
        np.hstack([ndfu_man, ndfu_woman]), bins=NUM_BINS, normalized=True
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(
        men_annot, bins=NUM_BINS, alpha=0.6, label="Men", kde=True, ax=ax
    )
    sns.histplot(
        women_annot, bins=NUM_BINS, alpha=0.6, label="Women", kde=True, ax=ax
    )
    ax.set_title(title, fontsize=LESSER_LABEL_SIZE)
    ax.legend(loc="upper right")
    ax.set_xlabel(
        f"$nDFU_{{men}}={ndfu_man:.4f}$\n"
        f"$nDFU_{{women}}={ndfu_woman:.4f}$\n"
        f"$nDFU_{{all}}={ndfu_all:.4f}$",
        fontsize=LESSER_LABEL_SIZE,
    )
    ax.set_ylabel(r"\#Comments", fontsize=LABEL_SIZE)
    ax.set_xlim(1, 10)

    graphs.save_plot(graph_path)
    plt.close()


def main(graph_dir: Path):
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
    np.random.seed(seed=42)
    colors = sns.color_palette()

    dfu_plots(colors, graph_dir)
    discussion_example(graph_dir)

    plot_annotation_distributions(graph_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Create demonstrative plots to explain apunim.")
    )
    parser.add_argument(
        "--graph-output-dir",
        required=True,
        help="Directory for the graphs.",
    )
    args = parser.parse_args()
    main(graph_dir=Path(args.graph_output_dir))
