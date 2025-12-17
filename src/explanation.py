import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import apunim
from numpy.typing import NDArray

from .tasks import graphs


INTUITION_SIZE = 50
DIFF_COMMENTS_SIZE = 200
NUM_BINS = 10
LABEL_FONTSIZE = 18
LESSER_LABEL_SIZE = 18
SUBTITLE_FONTSIZE = 24


def _discrete_normal(loc, scale, size):
    vals = np.random.normal(loc, scale, size)
    vals = np.clip(
        np.round(vals), 1, 5
    )  # round to nearest integer between 1-5
    return vals


def _prepare_distributions(n_annotators, variance):
    # Define special indices for mean shifts (use the first 20% as "Muslims")
    n_special = int(0.2 * n_annotators)
    special_indices = list(range(n_special))  # first 20% will get mean_shift=1

    def make_distribution(base_means, special_indices=[], variance=variance):
        data = []
        for i in range(n_annotators):
            mean_shift = 1 if i in special_indices else 0
            distributions = [
                _discrete_normal(base_mean + mean_shift, variance, 1)
                for base_mean in base_means
            ]
            combined = (
                np.mean(distributions)
                if len(distributions) > 1
                else distributions[0][0]
            )
            data.append(combined)
        return data

    unimodal = make_distribution([3], variance=variance)
    bimodal = make_distribution(
        [3], special_indices=special_indices, variance=variance
    )
    multimodal = make_distribution(
        [1, 3], special_indices=special_indices, variance=variance * 3
    )
    uniform = list(np.random.randint(1, 6, n_annotators))

    group_labels = [
        1 if i in special_indices else 0 for i in range(n_annotators)
    ]  # 0=Christian, 1=Muslim
    return unimodal, bimodal, multimodal, uniform, group_labels


def _plot_matrix(
    ax, data, group_labels, title, horizontal_jitter=0.25, vertical_jitter=0.15
):
    color_map = {0: "blue", 1: "green"}
    label_map = {0: "Christian", 1: "Muslim"}

    # Scatter points for each group separately to add legend
    for group in [0, 1]:
        idx = [i for i, g in enumerate(group_labels) if g == group]
        x_jittered = np.array(idx) + np.random.uniform(
            -horizontal_jitter, horizontal_jitter, size=len(idx)
        )
        y_jittered = np.array([data[i] for i in idx]) + np.random.uniform(
            -vertical_jitter, vertical_jitter, size=len(idx)
        )
        ax.scatter(
            x_jittered,
            y_jittered,
            c=color_map[group],
            edgecolor="black",
            s=60,
            label=label_map[group],
            alpha=0.7,
        )

    ax.set_xlabel("Annotators", fontsize=LABEL_FONTSIZE)
    ax.set_xlim(-0.5, len(data) - 0.5)
    ax.set_xticks([])
    ax.set_ylabel("Hate Speech", fontsize=LABEL_FONTSIZE)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["☺", "🙂", "😐", "😠", "🤬"], fontsize=LABEL_FONTSIZE)
    ax.set_ylim(0.8, 5.2)
    ax.set_title(title, fontsize=SUBTITLE_FONTSIZE)
    ax.legend(title="Annotator Group", loc="upper right")


def plot_annotation_distributions(
    graph_dir: Path, n_annotators=100, variance=0.3, random_seed=42
):
    np.random.seed(random_seed)

    unimodal, bimodal, multimodal, uniform, group_labels = (
        _prepare_distributions(n_annotators, variance)
    )

    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    _plot_matrix(
        axs[0, 0], unimodal, group_labels, "Low Disagreement\nLow Polarization"
    )
    _plot_matrix(
        axs[0, 1], bimodal, group_labels, "Low Disagreement\nHigh Polarization"
    )
    _plot_matrix(
        axs[1, 0], uniform, group_labels, "High Disagreement\nLow Polarization"
    )
    _plot_matrix(
        axs[1, 1],
        multimodal,
        group_labels,
        "High Disagreement\nHigh Polarization",
    )
    plt.savefig(graph_dir / "disagreement_vs_polarization.png")
    plt.close()


def dfu_plots(colors, graph_dir: Path) -> None:
    d1 = _truncated_normal(loc=2, scale=1.3, size=INTUITION_SIZE)
    d2 = _truncated_normal(loc=8, scale=1.3, size=INTUITION_SIZE)
    d_all = np.hstack([d1, d2])

    _dfu_plot(
        data=d1,
        graph_path=graph_dir / "ndfu_men.png",
        label="Men",
        color=colors[0],
    )
    _dfu_plot(
        data=d2,
        graph_path=graph_dir / "ndfu_women.png",
        label="Women",
        color=colors[1],
    )
    _dfu_plot(
        data=d_all,
        graph_path=graph_dir / "ndfu_all.png",
        label="All",
        color=colors[2],
    )
    _combined_dfu_plot(
        datasets=[d1, d2, d_all],
        graph_path=graph_dir / "ndfu_combined.png",
        labels=["Men", "Women", "All"],
        colors=colors,
    )


def _combined_dfu_plot(
    datasets: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    graph_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    for data, label, color in zip(datasets, labels, colors):
        sns.histplot(
            data,
            bins=NUM_BINS,
            kde=True,
            alpha=0.7,
            color=color,
            label=label,
            ax=ax,
        )

    # Add all nDFU annotations
    math_text = ""
    for data, label in zip(datasets, labels):
        ndfu_value = apunim.dfu(data, bins=NUM_BINS, normalized=True)
        math_text += f"$\\mathbf{{nDFU_{{{label}}}}}={ndfu_value:.3f}$\n"

    plt.legend(loc="center")
    plt.xlabel(f"Toxicity\n{math_text}", fontsize=LABEL_FONTSIZE)
    plt.ylabel(r"\#Annotations", fontsize=LABEL_FONTSIZE)
    plt.title(r"\texttit{``Most women can't drive well.''}")

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
        graph_dir / "ndfu_comment1.png",
    )
    _plot_example_individual(
        misandrist_comment,
        d_woman_comment1,
        d_man_comment1,
        graph_dir / "ndfu_comment2.png",
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
    ndfu_value = apunim.dfu(data, bins=NUM_BINS, normalized=True)

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
    plt.xlabel(f"Toxicity\n{math_text}", fontsize=LABEL_FONTSIZE)
    plt.ylabel(r"\#Comments", fontsize=LABEL_FONTSIZE)

    # individual pictures to be used for slides
    graphs.save_plot(graph_path)
    plt.close()


def _plot_example_individual(
    title: str,
    women_annot: NDArray[np.float64],
    men_annot: NDArray[np.float64],
    graph_path: Path,
):
    ndfu_man = apunim.dfu(men_annot, bins=NUM_BINS, normalized=True)
    ndfu_woman = apunim.dfu(women_annot, bins=NUM_BINS, normalized=True)
    ndfu_all = apunim.dfu(
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
    ax.set_ylabel(r"\#Comments", fontsize=LABEL_FONTSIZE)
    ax.set_xlim(1, 10)

    graphs.save_plot(graph_path)
    plt.close()


def main(graph_dir: Path):
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"text.usetex": False})
    np.random.seed(seed=42)
    colors = sns.color_palette()

    dfu_plots(colors, graph_dir)
    discussion_example(graph_dir)

    plt.rcParams["font.family"] = "Symbola"
    plt.rcParams["text.usetex"] = False
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
