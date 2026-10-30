import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import scipy.stats
import apunim
from numpy.typing import NDArray
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..lib import graphs, labels


INTUITION_SIZE = 50
DIFF_COMMENTS_SIZE = 200
NUM_BINS = 10

COLOR_MAP = {
    "Men": graphs.COLORBLIND_PALETTE[0],
    "Women": graphs.COLORBLIND_PALETTE[1],
}

HATCH_MAP = {
    "Men": graphs.HATCHES[0],
    "Women": graphs.HATCHES[1],
    "All": graphs.HATCHES[2],
}

# Annotator groups for the disagreement plots: 0 = majority, 1 = minority.
GROUP_LABELS = {0: "Non-Muslim", 1: "Muslim"}
GROUP_COLORS = {
    0: graphs.COLORBLIND_PALETTE[0],
    1: graphs.COLORBLIND_PALETTE[1],
}
GROUP_MARKERS = {0: graphs.MARKERS[0], 1: graphs.MARKERS[6]}  # circle, X
# Style for panels that draw no group distinction.
UNGROUPED_COLOR = graphs.COLORBLIND_PALETTE[2]
UNGROUPED_MARKER = graphs.MARKERS[3]  # triangle
# Wider than graphs.DEFAULT_FIGSIZE to fit three panels side by side.
DISAGREEMENT_FIGSIZE = (20, 4)
# How many scale points the minority's opinions are shifted up by.
MINORITY_SHIFT = 2

# Discussion figure: three panels (comment 1, comment 2, pooled).
# Rating spread of every group on every comment.
DISCUSSION_SIGMA = 0.5
# Shared bin edges so bars line up across panels and groups.
# Only used for drawing; nDFU is still computed with NUM_BINS.
DISCUSSION_PLOT_BINS = np.linspace(0, 10, 2 * NUM_BINS + 1)
# Density, so the pooled panel (twice the annotations) shares a y-scale.
DISCUSSION_HIST_STAT = "density"
DISCUSSION_TITLE = f"Pooling Annotations Across Items Masks {labels.ASD.short}"
MEAN_LINESTYLES = {
    "Men": graphs.LINESTYLES[1],  # dashed
    "Women": graphs.LINESTYLES[2],  # dotted
}
# Mean lines are drawn this many times thicker than rcParams lines.
MEAN_LINEWIDTH_SCALE = 2


def main(graph_dir: Path):
    graphs.graph_setup()
    np.random.seed(seed=42)

    dfu_plots(graph_dir)
    discussion_example(graph_dir)

    plot_annotation_distributions(graph_dir)


def _discrete_normal(loc, scale, size):
    vals = np.random.normal(loc, scale, size)
    vals = np.clip(
        np.round(vals), 1, 5
    )  # round to nearest integer between 1-5
    return vals


def plot_annotation_distributions(
    graph_dir: Path,
    n_annotators=100,
    variance=0.3,
    random_seed=42,
):
    np.random.seed(random_seed)
    groups = _group_labels(n_annotators)

    # Jitter once so the two systematic-disagreement plots
    # show identical points.
    mixed = _jitter(_mixed_opinions(n_annotators))
    split = _jitter(_minority_split_mixed_opinions(groups, variance))

    fig, axs = plt.subplots(1, 3, sharey=True, figsize=DISAGREEMENT_FIGSIZE)
    _annotation_axis(axs[0], mixed, groups=None, title="Disagreement")
    _annotation_axis(
        axs[1], split, groups=None, title="Systematic Disagreement"
    )
    _annotation_axis(
        axs[2],
        split,
        groups=groups,
        title="Attributable\nSystematic Disagreement",
    )

    _group_legend(axs[2])

    fig.suptitle(
        r"``\textit{We will never stop in our fight against Radical Islamic Terrorism}''",
        y=1.1,  # clear of the two-line panel title
    )
    graphs.save_plot(graph_dir / "disagreement_vs_polarization.png")
    plt.close(fig)


def _group_labels(n_annotators: int, minority_frac=0.2) -> NDArray[np.int_]:
    """The first minority_frac of annotators belong to the minority (1)."""
    n_minority = int(minority_frac * n_annotators)
    return np.array([1] * n_minority + [0] * (n_annotators - n_minority))


def _mixed_opinions(n_annotators: int) -> NDArray[np.float64]:
    return np.random.randint(1, 6, n_annotators).astype(float)


def _minority_split_mixed_opinions(
    groups: NDArray[np.int_], variance: float
) -> NDArray[np.float64]:
    """Each annotator averages two draws around 1 and 3; the minority is
    shifted up by MINORITY_SHIFT points."""
    draws = [
        _discrete_normal(
            base + MINORITY_SHIFT * groups, variance * 3, len(groups)
        )
        for base in (1, 3)
    ]
    return np.mean(draws, axis=0)


def _jitter(
    values: NDArray[np.float64],
    horizontal_jitter=0.25,
    vertical_jitter=0.15,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One column per annotator, with jitter so overlapping points show."""
    n = len(values)
    x = np.arange(n) + np.random.uniform(
        -horizontal_jitter, horizontal_jitter, n
    )
    y = values + np.random.uniform(-vertical_jitter, vertical_jitter, n)
    return x, y


def _annotation_axis(
    ax,
    points: tuple[NDArray[np.float64], NDArray[np.float64]],
    groups: NDArray[np.int_] | None,
    title: str,
) -> None:
    """Scatter of annotations. Without groups, every annotator is drawn in
    a neutral style and no legend is shown."""
    x, y = points

    if groups is None:
        _scatter(ax, x, y, UNGROUPED_COLOR, UNGROUPED_MARKER, label=None)
    else:
        for group, label in GROUP_LABELS.items():
            mask = groups == group
            _scatter(
                ax,
                x[mask],
                y[mask],
                GROUP_COLORS[group],
                GROUP_MARKERS[group],
                label=label,
            )

    _style_annotation_axis(ax, n_annotators=len(x), title=title)


def _group_legend(ax) -> None:
    """Legend for the grouped panel only, since the other panels draw
    no groups."""
    ax.legend(
        markerscale=2,
    )


def _scatter(ax, x, y, color: str, marker: str, label: str | None) -> None:
    ax.scatter(
        x,
        y,
        c=color,
        marker=marker,
        edgecolor="black",
        s=65,
        alpha=0.75,
        linewidth=0.7,
        label=label,
    )


def _style_annotation_axis(ax, n_annotators: int, title: str) -> None:
    ax.set_xlim(-0.5, n_annotators - 0.5)
    ax.set_xticks([])
    ax.minorticks_off()

    # symbolic labels that render under usetex
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(
        [r"$\heartsuit\heartsuit\heartsuit$", r"$\heartsuit$", "?", "!", "!!!"]
    )
    ax.set_ylim(0.8, 5.2)
    ax.tick_params(axis="y", labelsize=plt.rcParams["axes.labelsize"])
    ax.set_title(title)


def dfu_plots(graph_dir: Path) -> None:
    d1 = _truncated_normal(loc=2, scale=1.3, size=INTUITION_SIZE)
    d2 = _truncated_normal(loc=8, scale=1.3, size=INTUITION_SIZE)
    d_all = np.hstack([d1, d2])

    _combined_dfu_plot(
        datasets=[d_all, d1, d2],
        labels=["All", "Men", "Women"],
        colors=[
            graphs.COLORBLIND_PALETTE[2],
            COLOR_MAP["Men"],
            COLOR_MAP["Women"],
        ],
        graph_path=graph_dir / "ndfu_combined.png",
    )


def _combined_dfu_plot(
    datasets: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    graph_path: Path,
) -> None:
    ax = plt.gca()

    legend_handles = []

    for data, label, color in zip(datasets, labels, colors):
        hatch = HATCH_MAP[label]
        before = len(ax.patches)

        sns.histplot(
            data,
            bins=NUM_BINS,
            kde=True,
            alpha=0.5,
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
    """Two comments with opposite (bidirectional) gender polarization,
    and the pooled annotations of both, as one three-panel figure."""
    # Each group rates the comment targeting its own gender as more toxic.
    women_comment2 = _truncated_normal(
        loc=2, scale=DISCUSSION_SIGMA, size=DIFF_COMMENTS_SIZE
    )
    women_comment1 = _truncated_normal(
        loc=6, scale=DISCUSSION_SIGMA, size=DIFF_COMMENTS_SIZE
    )
    men_comment2 = _truncated_normal(
        loc=6, scale=DISCUSSION_SIGMA, size=DIFF_COMMENTS_SIZE
    )
    men_comment1 = _truncated_normal(
        loc=2, scale=DISCUSSION_SIGMA, size=DIFF_COMMENTS_SIZE
    )

    panels = [
        (
            "``Most women do not drive well.''",
            men_comment1,
            women_comment1,
        ),
        (
            "``Most men need closer friendships,\nnot just romantic support.''",
            men_comment2,
            women_comment2,
        ),
        (
            "Both comments pooled",
            np.hstack([men_comment1, men_comment2]),
            np.hstack([women_comment1, women_comment2]),
        ),
    ]

    # Figure size and fonts come from graphs.graph_setup(); constrained
    # layout positions the panels, legend and title without manual offsets.
    fig, axs = plt.subplots(1, len(panels), sharey=True, layout="constrained")
    for ax, (title, men_annot, women_annot) in zip(axs, panels):
        _discussion_panel(ax, title, men_annot, women_annot)

    axs[0].set_ylabel("Density")
    handles = _discussion_legend_handles()
    fig.suptitle(DISCUSSION_TITLE)
    _legend_below(fig, axs, handles)

    graphs.save_plot(graph_dir / "ndfu_discussion.png")
    plt.close(fig)


def _discussion_panel(
    ax,
    title: str,
    men_annot: NDArray[np.float64],
    women_annot: NDArray[np.float64],
) -> None:
    """Histograms per group, a line at each group's mean, and each group's
    nDFU and mean under the x-axis."""
    group_annots = {"Men": men_annot, "Women": women_annot}

    for label, annot in group_annots.items():
        before = len(ax.patches)
        sns.histplot(
            annot,
            bins=DISCUSSION_PLOT_BINS,
            stat=DISCUSSION_HIST_STAT,
            alpha=0.6,
            kde=True,
            ax=ax,
            color=COLOR_MAP[label],
        )
        # only hatch the bars this call created
        for patch in ax.patches[before:]:
            patch.set_hatch(HATCH_MAP[label])
            patch.set_edgecolor("black")

        ax.axvline(annot.mean(), **_mean_line_style(label))

    # One quantity per line, so labels of neighboring panels never
    # collide at the default figure width.
    stats_lines = [
        f"$nDFU_{{{label}}}="
        f"{apunim.dfu(annot, bins=NUM_BINS, normalized=True):.3f}$"
        for label, annot in group_annots.items()
    ] + [
        f"$\\mu_{{{label}}}={annot.mean():.2f}$"
        for label, annot in group_annots.items()
    ]

    ax.set_title(title)
    ax.set_xlabel(
        r"\begin{tabular}{c}Toxicity\\"
        + r"\\".join(stats_lines)
        + r"\end{tabular}"
    )
    ax.set_xlim(0, 10)


def _mean_line_style(label: str) -> dict:
    """Line style for a group's mean, sized from rcParams at draw time
    (after graphs.graph_setup() has run). The black outline keeps the
    line visible over bars of the same color."""
    linewidth = MEAN_LINEWIDTH_SCALE * plt.rcParams["lines.linewidth"]
    return {
        "color": COLOR_MAP[label],
        "linestyle": MEAN_LINESTYLES[label],
        "linewidth": linewidth,
        "path_effects": [
            pe.Stroke(
                linewidth=linewidth + plt.rcParams["lines.linewidth"],
                foreground="black",
            ),
            pe.Normal(),
        ],
    }


def _legend_below(fig, axs, handles) -> None:
    """Place the legend just below the lowest x-label, after constrained
    layout has run."""
    fig.draw_without_rendering()  # run constrained layout once
    fig.set_layout_engine("none")  # freeze it; the legend must not move panels
    renderer = fig.canvas.get_renderer()
    bottom = min(ax.get_tightbbox(renderer).y0 for ax in axs)
    _, y = fig.transFigure.inverted().transform((0, bottom))
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=len(handles),
    )


def _discussion_legend_handles() -> list:
    handles = [
        Patch(
            facecolor=COLOR_MAP[label],
            edgecolor="black",
            hatch=HATCH_MAP[label],
            label=label,
            alpha=0.6,
        )
        for label in ("Men", "Women")
    ]
    handles += [
        Line2D([], [], label=f"Mean ({label})", **_mean_line_style(label))
        for label in ("Men", "Women")
    ]
    return handles


def _truncated_normal(loc, scale, lower=0, upper=10, size=100):
    a, b = (lower - loc) / scale, (upper - loc) / scale
    return scipy.stats.truncnorm(a, b, loc=loc, scale=scale).rvs(size)


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
