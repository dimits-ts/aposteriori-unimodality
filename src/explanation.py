import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import apunim
from numpy.typing import NDArray
from matplotlib.patches import Patch

from .tasks import graphs


INTUITION_SIZE = 50
DIFF_COMMENTS_SIZE = 200
NUM_BINS = 10


def _apply_hatches(ax, hatch: str):
    for patch in ax.patches:
        patch.set_hatch(hatch)
        patch.set_edgecolor("black")


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
    ax,
    data,
    group_labels,
    title,
    horizontal_jitter=0.25,
    vertical_jitter=0.15,
):
    color_map = {0: "blue", 1: "green"}

    marker_map = {
        0: graphs.MARKERS[0],
        1: graphs.MARKERS[1],
    }

    # Scatter points for each group separately
    for group in [0, 1]:
        idx = [i for i, g in enumerate(group_labels) if g == group]

        x_jittered = np.array(idx) + np.random.uniform(
            -horizontal_jitter,
            horizontal_jitter,
            size=len(idx),
        )

        y_jittered = np.array([data[i] for i in idx]) + np.random.uniform(
            -vertical_jitter,
            vertical_jitter,
            size=len(idx),
        )

        ax.scatter(
            x_jittered,
            y_jittered,
            c=color_map[group],
            marker=marker_map[group],
            edgecolor="black",
            s=65,
            alpha=0.75,
            linewidth=0.7,
        )

    ax.set_xlim(-0.5, len(data) - 0.5)
    ax.set_xticks([])
    ax.minorticks_off()

    # emoji labels that render in standard unicode fonts
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(
        [r"$\heartsuit\heartsuit$", r"$\heartsuit$", "?", "!", "!!"]
    )

    ax.set_ylim(0.8, 5.2)

    ax.set_title(
        title,
        pad=10,
        fontweight="bold",
    )

    # remove unnecessary subplot spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # keep left spine for emoji scale
    ax.spines["left"].set_linewidth(1.0)

    ax.tick_params(axis="x", length=0)


def plot_annotation_distributions(
    graph_dir: Path,
    n_annotators=100,
    variance=0.3,
    random_seed=42,
):
    np.random.seed(random_seed)

    unimodal, bimodal, multimodal, uniform, group_labels = (
        _prepare_distributions(n_annotators, variance)
    )

    fig, axs = plt.subplots(
        2,
        2,
        figsize=(13, 10),
        sharex=True,
        sharey=True,
    )

    # MUCH larger spacing
    plt.subplots_adjust(
        hspace=0.32,
        wspace=0.18,
        left=0.16,
        right=0.96,
        top=0.86,
        bottom=0.16,
    )

    # Low polarization / low disagreement
    _plot_matrix(
        axs[1, 0],
        unimodal,
        group_labels,
        title="Agreement",
    )

    # High polarization / low disagreement
    _plot_matrix(
        axs[0, 0],
        bimodal,
        group_labels,
        title="Group split",
    )

    # High polarization / high disagreement
    _plot_matrix(
        axs[0, 1],
        multimodal,
        group_labels,
        title="Group split, mixed opinions",
    )

    # Low polarization / high disagreement
    _plot_matrix(
        axs[1, 1],
        uniform,
        group_labels,
        title="Overall mixed opinions",
    )

    # clearer matrix-style labels
    fig.supxlabel(
        "Low disagreement $\\rightarrow$ High disagreement",
        fontweight="bold",
        y=0.08,  # avoid legend
    )

    fig.supylabel(
        "Low polarization $\\rightarrow$ High polarization", fontweight="bold"
    )

    fig.suptitle(
        "``We will never stop in our fight against Radical Islamic Terrorism''",
        fontweight="bold",
    )

    legend_handles = [
        plt.Line2D(
            [],
            [],
            linestyle="",
            marker=graphs.MARKERS[1],
            markersize=18,
            markerfacecolor="green",
            markeredgecolor="black",
            label="Muslim",
        ),
        plt.Line2D(
            [],
            [],
            linestyle="",
            marker=graphs.MARKERS[0],
            markersize=18,
            markerfacecolor="blue",
            markeredgecolor="black",
            label="Non-Muslim",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=18,
        frameon=True,
    )

    graphs.save_plot(graph_dir / "disagreement_vs_polarization.png")
    plt.close()


def dfu_plots(colors, graph_dir: Path) -> None:
    d1 = _truncated_normal(loc=2, scale=1.3, size=INTUITION_SIZE)
    d2 = _truncated_normal(loc=8, scale=1.3, size=INTUITION_SIZE)
    d_all = np.hstack([d1, d2])

    _combined_dfu_plot(
        datasets=[d_all, d1, d2],
        graph_path=graph_dir / "ndfu_combined.png",
        labels=["All", "Men", "Women"],
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

    legend_handles = []

    for data, label, color, hatch in zip(
        datasets,
        labels,
        colors,
        graphs.HATCHES,
    ):
        before = len(ax.patches)

        sns.histplot(
            data,
            bins=NUM_BINS,
            kde=True,
            alpha=0.7,
            color=color,
            ax=ax,
        )

        # only apply hatch to newly created bars
        new_patches = ax.patches[before:]

        for patch in new_patches:
            patch.set_hatch(hatch)
            patch.set_edgecolor("black")

        legend_handles.append(
            Patch(
                facecolor=color,
                edgecolor="black",
                hatch=hatch,
                label=label,
                alpha=0.7,
            )
        )

    # Add all nDFU annotations
    math_text = ""
    for data, label in zip(datasets, labels):
        ndfu_value = apunim.dfu(
            data,
            bins=NUM_BINS,
            normalized=True,
        )
        math_text += f"$\\mathbf{{nDFU_{{{label}}}}}={ndfu_value:.3f}$\n"

    plt.legend(handles=legend_handles, loc="upper center")
    plt.xlabel(f"Toxicity\n{math_text}")
    plt.ylabel(r"\#Annotations")
    plt.title(r"\textit{``Most women can't drive well.''}")

    graphs.save_plot(graph_path)
    plt.close()


def discussion_example(graph_dir: Path) -> None:
    misogynist_comment = """
    A: ``Why does the police seem to like
    killing black people?''
    """
    misandrist_comment = """
    B: ``There is a much risk higher of resistance when
    dealing with blacks compared to anyone else.''
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
        d_man_comment1,
        d_woman_comment1,
        graph_dir / "ndfu_comment1.png",
    )
    _plot_example_individual(
        misandrist_comment,
        d_man_comment2,
        d_woman_comment2,
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


def _dfu_plot(
    data: np.ndarray,
    graph_path: Path,
    color: str,
    label,
    hatch: str,
) -> None:
    ndfu_value = apunim.dfu(data, bins=NUM_BINS, normalized=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data,
        bins=NUM_BINS,
        kde=True,
        alpha=0.7,
        color=color,
        ax=ax,
    )

    _apply_hatches(ax, hatch)

    legend_handle = Patch(
        facecolor=color,
        edgecolor="black",
        hatch=hatch,
        label=label,
        alpha=0.7,
    )

    math_text = f"$\\mathbf{{nDFU_{{{label}}}}}={ndfu_value:.3f}$"
    plt.xlabel(f"Toxicity\n{math_text}")
    plt.ylabel(r"\#Annotations")

    ax.legend(handles=[legend_handle])

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
        men_annot,
        bins=NUM_BINS,
        alpha=0.6,
        kde=True,
        ax=ax,
        color=sns.color_palette()[0],
    )
    men_patches = list(ax.patches)

    sns.histplot(
        women_annot,
        bins=NUM_BINS,
        alpha=0.6,
        kde=True,
        ax=ax,
        color=sns.color_palette()[1],
    )

    women_patches = ax.patches[len(men_patches) :]

    for patch in men_patches:
        patch.set_hatch(graphs.HATCHES[0])
        patch.set_edgecolor("black")

    for patch in women_patches:
        patch.set_hatch(graphs.HATCHES[1])
        patch.set_edgecolor("black")

    legend_handles = [
        Patch(
            facecolor=sns.color_palette()[0],
            edgecolor="black",
            hatch=graphs.HATCHES[0],
            label="Black",
            alpha=0.6,
        ),
        Patch(
            facecolor=sns.color_palette()[1],
            edgecolor="black",
            hatch=graphs.HATCHES[1],
            label="White",
            alpha=0.6,
        ),
    ]

    ax.set_title(title)
    ax.legend(handles=legend_handles, loc="upper right")

    ax.set_xlabel(
        f"$nDFU_{{White}}={ndfu_man:.4f}$\n"
        f"$nDFU_{{Black}}={ndfu_woman:.4f}$\n"
        f"$nDFU_{{All}}={ndfu_all:.4f}$",
    )
    ax.set_ylabel(r"\#Annotations")
    ax.set_xlabel("Toxicity")
    ax.set_xlim(1, 10)

    graphs.save_plot(graph_path)
    plt.close()


def main(graph_dir: Path):
    graphs.graph_setup()
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
