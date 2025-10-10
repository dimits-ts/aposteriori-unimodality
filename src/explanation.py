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


def _discrete_normal(loc, scale, size):
    vals = np.random.normal(loc, scale, size)
    vals = np.clip(
        np.round(vals), 1, 5
    )  # round to nearest integer between 1-5
    return vals


def _prepare_distributions(n_annotators, n_annotations, variance):
    unimodal = [
        _discrete_normal(3, variance, n_annotations)
        for _ in range(n_annotators)
    ]
    bimodal = [
        np.concatenate(
            [
                _discrete_normal(2, variance, n_annotations // 2),
                _discrete_normal(4, variance, n_annotations // 2),
            ]
        )
        for _ in range(n_annotators)
    ]
    multimodal = [
        np.concatenate(
            [
                _discrete_normal(1, variance, n_annotations // 3),
                _discrete_normal(3, variance, n_annotations // 3),
                _discrete_normal(5, variance, n_annotations // 3),
            ]
        )
        for _ in range(n_annotators)
    ]
    uniform = [
        np.random.randint(1, 6, n_annotations) for _ in range(n_annotators)
    ]
    return unimodal, bimodal, multimodal, uniform


def _plot_matrix(
    ax, data, n_annotators, title, horizontal_jitter=0.25, vertical_jitter=0.15
):
    all_x = []
    all_y = []
    for i, values in enumerate(data):
        # Horizontal jitter for annotator separation
        x_jittered = i + np.random.uniform(
            -horizontal_jitter, horizontal_jitter, size=len(values)
        )
        # Vertical jitter to show overlapping points
        y_jittered = values + np.random.uniform(
            -vertical_jitter, vertical_jitter, size=len(values)
        )
        all_x.extend(x_jittered)
        all_y.extend(y_jittered)

    all_x = np.array(all_x)
    all_y = np.array(all_y)

    # Compute KDE for density
    xy = np.vstack([all_x, all_y])
    z = scipy.stats.gaussian_kde(xy)(xy)

    sc = ax.scatter(
        all_x, all_y, c=z, s=40, edgecolor="black", cmap="viridis", alpha=0.7
    )

    ax.set_xlabel("Annotators")
    ax.set_xticks(range(n_annotators))
    ax.set_xticklabels([str(i + 1) for i in range(n_annotators)])
    ax.set_xlim(-0.5, n_annotators - 0.5)

    ax.set_ylabel("Toxicity")
    ax.set_yticks([1, 2, 3, 4, 5])

    ax.set_ylim(0.8, 5.2)

    ax.set_title(title)
    plt.colorbar(sc, ax=ax, label="Density")


def plot_annotation_distributions(
    graph_dir: Path,
    n_annotators: int = 10,
    n_annotations: int = 100,
    variance: float = 0.5,
    random_seed: int = 42,
):
    np.random.seed(random_seed)

    unimodal, bimodal, multimodal, uniform = _prepare_distributions(
        n_annotators, n_annotations, variance
    )

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    _plot_matrix(
        axs[0, 0], unimodal, n_annotators, "Low Disagreement\nLow Polarization"
    )
    _plot_matrix(
        axs[0, 1], uniform, n_annotators, "High Disagreement\nLow Polarization"
    )
    _plot_matrix(
        axs[1, 0], bimodal, n_annotators, "Low Disagreement\nHigh Polarization"
    )
    _plot_matrix(
        axs[1, 1],
        multimodal,
        n_annotators,
        "High Disagreement\nHigh Polarization",
    )

    fig.suptitle(
        "Annotator disagreement (variance) vs polarization (clustering)"
    )

    graphs.save_plot(
        graph_dir / "disagreement_vs_polarization.png"
    )
    plt.close()


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
