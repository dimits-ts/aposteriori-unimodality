import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd


from ..lib import graphs
from ..lib.preprocessing import Dataset
from .shared import (
    DATASET_KEYS,
    HumanDatasets,
    find_annotation_files,
    load_llm_df,
    _available_dataset_keys,
    _key_columns,
    _keyed_series,
    _order_models,
    MAIN_PROMPT_NAMES,
)


# ---------------------------------------------------------------------------
# Shared plotting helpers (subplot grids, legends)
# ---------------------------------------------------------------------------


def _subplot_grid(n: int, ncols: int):
    nrows = -(-n // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, squeeze=False)
    return fig, axes, nrows


def _axis_at(axes, i: int, ncols: int):
    return axes[i // ncols][i % ncols]


def _hide_unused_axes(axes, n_used: int, nrows: int, ncols: int) -> None:
    for j in range(n_used, nrows * ncols):
        _axis_at(axes, j, ncols).set_visible(False)


def _clean_legend(ax, fontsize: int = 10) -> None:
    legend = ax.get_legend()
    if legend is None:
        return
    legend.set_title(None)
    for text in legend.get_texts():
        text.set_fontsize(fontsize)


# ---------------------------------------------------------------------------
# 1. Histograms: human vs. LLM annotation frequencies, one subplot/dataset
# ---------------------------------------------------------------------------


def collect_human_annotations(ds: Dataset) -> np.ndarray:
    col = ds.get_annotation_column()
    values = []
    for entry in ds.get_dataset()[col]:
        if isinstance(entry, (list, np.ndarray)):
            values.extend(entry)
        else:
            values.append(entry)
    return (
        pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    )


def collect_llm_annotations(
    annotations_dir: Path, dataset_key: str, prompt_name: str
) -> dict[str, np.ndarray]:
    out = {}
    for pseudo, path in find_annotation_files(
        annotations_dir, dataset_key, prompt_name
    ).items():
        df = load_llm_df(path)
        out[pseudo] = df["annotation_clean"].dropna().to_numpy()
    return out


def _histogram_records(
    human_vals: np.ndarray, llm_vals: dict[str, np.ndarray]
) -> list[dict]:
    records = [{"value": v, "source": "Human"} for v in human_vals]
    for pseudo, vals in sorted(llm_vals.items()):
        records.extend({"value": v, "source": pseudo} for v in vals)
    return records


def _draw_annotation_histogram(
    ax,
    ds: Dataset,
    annotations_dir: Path,
    dataset_key: str,
    prompt_name: str,
) -> None:
    _LINESTYLES = [
        "-",
        "--",
        "-.",
        ":",
        (0, (3, 1, 1, 1)),
        (0, (5, 1)),
        (0, (1, 1)),
    ]
    human_vals = collect_human_annotations(ds)
    llm_vals = collect_llm_annotations(
        annotations_dir, dataset_key, prompt_name
    )
    records = _histogram_records(human_vals, llm_vals)

    if not records:
        ax.set_visible(False)
        print(f"No annotations found for {dataset_key}; skipping subplot.")
        return

    plot_df = pd.DataFrame(records)
    sources = ["Human"] + sorted(llm_vals.keys())
    sources = [s for s in sources if s in set(plot_df["source"])]
    palette = dict(zip(sources, graphs.COLORBLIND_PALETTE))
    linestyle_cycle = itertools.cycle(_LINESTYLES)
    dash_map = {s: next(linestyle_cycle) for s in sources}

    for source in sources:
        sub = plot_df[plot_df["source"] == source]
        is_human = source == "Human"
        sns.histplot(
            data=sub,
            x="value",
            discrete=True,
            stat="probability",
            common_norm=False,
            element="step",
            fill=False,
            color=palette[source],
            linestyle=dash_map[source],
            linewidth=2.4 if is_human else 1.6,
            alpha=1.0 if is_human else 0.85,
            label=source,
            ax=ax,
        )

    ax.set_title(ds.get_name())
    ax.set_xlabel(f"{ds.get_annotation_column()} value")
    ax.set_ylabel("Proportion")
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


def plot_annotation_histograms(
    human_datasets: HumanDatasets,
    annotations_dir: Path,
    output_path: Path,
    prompt_name: str = "default",
    ncols: int = 2,
    dataset_keys: list[str] = DATASET_KEYS,
) -> None:
    dataset_keys = _available_dataset_keys(human_datasets, dataset_keys)
    fig, axes, nrows = _subplot_grid(len(dataset_keys), ncols)

    for i, key in enumerate(dataset_keys):
        _draw_annotation_histogram(
            _axis_at(axes, i, ncols),
            human_datasets[key],
            annotations_dir,
            key,
            prompt_name,
        )

    _hide_unused_axes(axes, len(dataset_keys), nrows, ncols)

    handles_by_label = {}
    for ax in fig.axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles_by_label.setdefault(label, handle)

    ordered_labels = sorted(
        handles_by_label.keys(), key=lambda label: (label != "Human", label)
    )
    handles = [handles_by_label[label] for label in ordered_labels]

    fig.suptitle(
        f"Human vs. LLM annotation distributions ({prompt_name} prompt)",
        y=1.02,
    )
    fig.legend(
        handles,
        ordered_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(ordered_labels),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    graphs.save_plot(output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1b. Mean-difference-between-prompts bar charts (per model, per dataset)
# ---------------------------------------------------------------------------


def _paired_prompt_annotations(
    annotations_dir: Path,
    dataset_key: str,
    model: str,
    prompt_names: list[str],
) -> pd.DataFrame | None:
    dfs = _load_model_prompt_dfs(
        annotations_dir, dataset_key, model, prompt_names
    )
    if dfs is None:
        return None

    key_cols = _key_columns(dfs)
    series_list = [
        _keyed_series(df, key_cols, prompt_name)
        for prompt_name, df in dfs.items()
    ]
    wide = pd.concat(series_list, axis=1).dropna()
    return wide if not wide.empty else None


def _load_model_prompt_dfs(
    annotations_dir: Path,
    dataset_key: str,
    model: str,
    prompt_names: list[str],
) -> dict[str, pd.DataFrame] | None:
    dfs = {}
    for prompt_name in prompt_names:
        path = find_annotation_files(
            annotations_dir, dataset_key, prompt_name
        ).get(model)
        if path is None:
            return None
        dfs[prompt_name] = load_llm_df(path)
    return dfs


def _prompt_diff_records_for_model(
    annotations_dir: Path,
    dataset_key: str,
    model: str,
    prompt_names: list[str],
    baseline_prompt: str,
    other_prompts: list[str],
) -> list[dict]:
    wide = _paired_prompt_annotations(
        annotations_dir, dataset_key, model, prompt_names
    )
    if wide is None:
        return []
    return [
        {
            "Model": model,
            "Diff": f"{other} - {baseline_prompt}",
            "value": value,
        }
        for other in other_prompts
        for value in wide[other] - wide[baseline_prompt]
    ]


def _prompt_diff_records_for_dataset(
    annotations_dir: Path,
    dataset_key: str,
    models: list[str],
    prompt_names: list[str],
    baseline_prompt: str,
    other_prompts: list[str],
) -> list[dict]:
    records = []
    for model in models:
        records.extend(
            _prompt_diff_records_for_model(
                annotations_dir,
                dataset_key,
                model,
                prompt_names,
                baseline_prompt,
                other_prompts,
            )
        )
    return records


def _collect_prompt_diff_by_dataset(
    annotations_dir: Path,
    dataset_keys: list[str],
    exclude_models: set[str],
    prompt_names: list[str],
    baseline_prompt: str,
    other_prompts: list[str],
) -> dict[str, pd.DataFrame]:
    records_by_dataset = {}
    for key in dataset_keys:
        models = _order_models(
            set(find_annotation_files(annotations_dir, key, baseline_prompt))
            - exclude_models
        )
        records = _prompt_diff_records_for_dataset(
            annotations_dir,
            key,
            models,
            prompt_names,
            baseline_prompt,
            other_prompts,
        )
        if records:
            records_by_dataset[key] = pd.DataFrame(records)
    return records_by_dataset


def _draw_prompt_diff_subplot(
    ax,
    dataset_name: str,
    plot_df: pd.DataFrame,
    other_prompts: list[str],
    baseline_prompt: str,
) -> None:
    diff_order = [
        f"{other} - {baseline_prompt}"
        for other in other_prompts
        if f"{other} - {baseline_prompt}" in set(plot_df["Diff"])
    ]
    model_order = _order_models(set(plot_df["Model"]))
    palette = dict(zip(diff_order, graphs.COLORBLIND_PALETTE[1:]))

    sns.barplot(
        data=plot_df,
        x="Model",
        y="value",
        hue="Diff",
        hue_order=diff_order,
        order=model_order,
        errorbar="se",
        capsize=0.15,
        palette=palette,
        ax=ax,
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(dataset_name)
    ax.set_xlabel("")
    ax.set_ylabel(f"Mean annotation diff vs. {baseline_prompt}")
    ax.tick_params(axis="x", rotation=30)
    _clean_legend(ax)


def plot_prompt_mean_diff(
    human_datasets: HumanDatasets,
    annotations_dir: Path,
    output_path: Path,
    exclude_models: list[str],
    prompt_names: list[str] = MAIN_PROMPT_NAMES,
    baseline_prompt: str = "default",
    ncols: int = 2,
    dataset_keys: list[str] = DATASET_KEYS,
) -> None:
    """
    One subplot per dataset in `dataset_keys`: for each model, the mean
    difference (±SE) between annotations under each non-baseline prompt
    and the baseline, computed item-by-item on the same (comment, persona)
    pairs.
    """
    exclude_models = set(exclude_models)
    other_prompts = [p for p in prompt_names if p != baseline_prompt]
    dataset_keys = _available_dataset_keys(human_datasets, dataset_keys)

    records_by_dataset = _collect_prompt_diff_by_dataset(
        annotations_dir,
        dataset_keys,
        exclude_models,
        prompt_names,
        baseline_prompt,
        other_prompts,
    )
    if not records_by_dataset:
        print(
            "No datasets with >=2 matched prompts found; skipping prompt "
            "mean-diff plot."
        )
        return

    keys_with_data = [k for k in dataset_keys if k in records_by_dataset]
    fig, axes, nrows = _subplot_grid(len(keys_with_data), ncols)

    for i, key in enumerate(keys_with_data):
        _draw_prompt_diff_subplot(
            _axis_at(axes, i, ncols),
            human_datasets[key].get_name(),
            records_by_dataset[key],
            other_prompts,
            baseline_prompt,
        )

    _hide_unused_axes(axes, len(keys_with_data), nrows, ncols)
    fig.suptitle(
        "Mean difference (\u00b1 SE) in LLM annotations across prompt "
        "variants",
        y=1.02,
    )
    fig.tight_layout()
    graphs.save_plot(output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Inherent-polarization histogram grid (companion to the subsampled
# inherent-polarization table): datasets as columns, prompts as rows
# ---------------------------------------------------------------------------

INHERENT_HIST_BINS = 30


def plot_inherent_polarization_histogram_grid(
    long_df: pd.DataFrame,
    output_path: Path,
    dataset_keys: list[str],
    prompt_names: list[str],
    bins: int = INHERENT_HIST_BINS,
) -> None:
    """
    Grid of normalized histograms of per-comment inherent polarization:
    one row per dataset, one column per prompt, with Human and every LLM
    overlaid in each cell.

    `long_df` is the long-format frame from
    `polarization.compute_inherent_polarization_comparison` (columns
    Dataset, Prompt, Source, TextID, value). Each source is normalized by
    its own number of comments (bar heights sum to 1).

    Styling follows the human `apriori.png` inherent-polarization figure:
    colorblind palette with a hatch per source, black bar edges,
    alpha 0.7, x-range [0, 1].
    """
    df = long_df.assign(
        value=pd.to_numeric(long_df["value"], errors="coerce")
    ).dropna(subset=["value"])
    dataset_keys = [k for k in dataset_keys if k in set(df["Dataset"])]
    prompt_names = [p for p in prompt_names if p in set(df["Prompt"])]
    if df.empty or not dataset_keys or not prompt_names:
        print(f"No inherent-polarization values; skipping {output_path}.")
        return

    sources = ["Human"] + _order_models(set(df["Source"]) - {"Human"})
    sources = [s for s in sources if s in set(df["Source"])]
    style = {
        s: (
            graphs.COLORBLIND_PALETTE[i % len(graphs.COLORBLIND_PALETTE)],
            graphs.HATCHES[i % len(graphs.HATCHES)],
        )
        for i, s in enumerate(sources)
    }

    nrows, ncols = len(dataset_keys), len(prompt_names)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for r, key in enumerate(dataset_keys):
        for c, prompt in enumerate(prompt_names):
            ax = axes[r][c]
            for source in sources:
                vals = df[
                    (df["Dataset"] == key)
                    & (df["Prompt"] == prompt)
                    & (df["Source"] == source)
                ]["value"]
                if vals.empty:
                    continue
                color, hatch = style[source]
                before = len(ax.patches)
                sns.histplot(
                    x=vals,
                    bins=bins,
                    binrange=(0, 1),
                    stat="proportion",
                    common_norm=False,
                    alpha=0.7,
                    color=color,
                    ax=ax,
                )
                for patch in ax.patches[before:]:
                    patch.set_hatch(hatch)
                    patch.set_edgecolor("black")
                    patch.set_linewidth(0.3)
            ax.set_xlim(0.1, 1)
            ax.set_ylim(0, 0.3)
            ax.set_ylabel(key.capitalize() if c == 0 else "")
            ax.set_xlabel("")
            if r == 0:
                ax.set_title(prompt.capitalize())

    fig.supylabel("Proportion of comments")
    fig.supxlabel("Unattributable Polarization")
    fig.suptitle("LLM Annotation Exhibits Different UnPol Patterns")

    legend_handles = [
        mpatches.Patch(
            facecolor=style[s][0],
            edgecolor="black",
            hatch=style[s][1],
            label=s,
            alpha=0.4,
        )
        for s in sources
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(sources), 7),
        frameon=False,
    )
    graphs.save_plot(output_path)
    plt.close(fig)
