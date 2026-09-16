import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


from ..lib import graphs
from ..lib.preprocessing import Dataset, LazyDatasetLoader
from .common import (
    find_annotation_files,
    load_llm_df,
    _available_dataset_keys,
    _key_columns,
    _keyed_series,
    _order_models,
    _compute_ndfu_records,
    _limited_sdb_columns,
    _human_sample_dataset,
    MAIN_PROMPT_NAMES,
    PROMPT_COMPARISON_DATASET_KEYS,
    LLMAnnotationDataset,
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

    # Step-line histograms (rather than dodged bars) so up to 7 overlapping
    # distributions (human + 6 models) stay legible. Draw one source at a
    # time so each gets its own color AND line style/width — color alone
    # doesn't scale to 7 overlapping series.
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
    # No per-axis legend; a single shared legend is built once, at the
    # figure level, in plot_annotation_histograms.
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


def plot_annotation_histograms(
    human_datasets: LazyDatasetLoader,
    annotations_dir: Path,
    output_path: Path,
    prompt_name: str = "default",
    ncols: int = 2,
) -> None:
    dataset_keys = _available_dataset_keys(human_datasets) or list(
        human_datasets.keys()
    )
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

    # Collect one legend entry per unique source across all subplots,
    # preserving first-seen order (Human first), then draw it once,
    # horizontally, below the whole figure.
    handles_by_label = {}
    for ax in fig.axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles_by_label.setdefault(label, handle)

    ordered_labels = sorted(
        handles_by_label.keys(), key=lambda l: (l != "Human", l)
    )
    handles = [handles_by_label[l] for l in ordered_labels]

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
    """
    For a single (dataset, model), loads the annotation CSV for each of
    `prompt_names` and aligns them on (text_id + persona attributes) -- the
    key llm_annotate.py's seeding guarantees is shared across every prompt
    variant of a given dataset (see module docstring). Returns None if the
    model is missing any of the requested prompts, or if no items survive
    the alignment; otherwise returns one row per matched (comment, persona)
    item, with one column per prompt holding that prompt's cleaned
    annotation value.
    """
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
    human_datasets: LazyDatasetLoader,
    annotations_dir: Path,
    output_path: Path,
    exclude_models: list[str],
    prompt_names: list[str] = MAIN_PROMPT_NAMES,
    baseline_prompt: str = "default",
    ncols: int = 2,
) -> None:
    """
    One subplot per dataset (skipping any dataset for which fewer than two
    of `prompt_names` were run): for each model, the mean difference --
    with standard-error bars -- between that model's annotations under
    each non-baseline prompt (e.g. "stereotype", "persona") and its
    annotations under `baseline_prompt` ("default"), computed item-by-item
    on the *same* (comment, persona) pairs via `_paired_prompt_annotations`
    so the comparison is apples-to-apples rather than comparing marginal
    distributions.
    """
    exclude_models = set(exclude_models)
    other_prompts = [p for p in prompt_names if p != baseline_prompt]
    dataset_keys = _available_dataset_keys(human_datasets)

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
# 4. Composite apunim grid (datasets x models)
# ---------------------------------------------------------------------------


def _apunim_grid_models(
    annotations_dir: Path, dataset_keys: list[str], prompt_name: str
) -> list[str]:
    found: set[str] = set()
    for key in dataset_keys:
        found |= set(find_annotation_files(annotations_dir, key, prompt_name))
    return _order_models(found)


def _apunim_column_dataset(
    column: str,
    dataset_key: str,
    prompt_name: str,
    human_ds: Dataset | None,
    files: dict[str, Path],
):
    if column == "Human":
        return human_ds
    path = files.get(column)
    if path is None:
        return None
    df = load_llm_df(path)
    return LLMAnnotationDataset(df, dataset_key, column, prompt_name)


def _draw_apunim_column(
    ax,
    column: str,
    ds,
    sdb_columns_limit: int | None,
    is_first_row: bool,
    is_first_col: bool,
    dataset_name: str,
) -> None:
    color = graphs.COLORBLIND_PALETTE[1 if column == "Human" else 2]

    if ds is None:
        ax.axis("off")
        return

    plot_df = _compute_ndfu_records(
        ds, sdb_columns=_limited_sdb_columns(ds, sdb_columns_limit)
    )
    if plot_df.empty:
        ax.axis("off")
        return

    sns.boxplot(
        x="PC Dimension",
        y="nDFU",
        data=plot_df,
        ax=ax,
        fliersize=1,
        linewidth=0.6,
        color=color,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.grid(axis="y", alpha=0.3, linewidth=0.4)

    # Column titles only on the first dataset row; y-axis labels only on
    # the first column -- avoids repeating the same labels across the grid.
    if is_first_row:
        ax.set_title(column)

    if is_first_col:
        ax.set_ylabel(dataset_name.split()[0])
    else:
        ax.set_yticklabels([])


def _draw_apunim_row(
    subfig,
    dataset_key: str,
    dataset_name: str,
    ds_human: Dataset,
    annotations_dir: Path,
    prompt_name: str,
    columns: list[str],
    sdb_columns_limit: int | None,
    is_first_row: bool,
) -> None:
    axes = subfig.subplots(
        nrows=1, ncols=len(columns), squeeze=False
    )[0]

    human_ds = _human_sample_dataset(
        ds_human, annotations_dir, dataset_key, prompt_name
    )
    files = find_annotation_files(annotations_dir, dataset_key, prompt_name)

    for c, column in enumerate(columns):
        ds = _apunim_column_dataset(
            column, dataset_key, prompt_name, human_ds, files
        )
        _draw_apunim_column(
            axes[c],
            column,
            ds,
            sdb_columns_limit,
            is_first_row,
            c == 0,
            dataset_name,
        )


def plot_apunim_grid(
    human_datasets: LazyDatasetLoader,
    annotations_dir: Path,
    output_path: Path,
    prompt_name: str = "default",
    models: list[str] | None = None,
    sdb_columns_limit: int | None = 6,
    title: str = "Default",
) -> None:
    """Plot all datasets and models in a single 2x5 grid using subfigures.

    Each dataset occupies one subfigure (row), with one subplot per
    Human/model column. A single title is placed above each dataset row.

    `title` is the figure-level suptitle. It defaults to the main
    ("default" prompt) plot's title; callers producing a separate grid per
    adversarial prompt (see main()) pass something that just names the
    instruction instead, e.g. "Stereotype Prompt".
    """
    dataset_keys = [
        k for k in PROMPT_COMPARISON_DATASET_KEYS if k in human_datasets
    ]
    if not dataset_keys:
        print(
            "No non-DICES datasets available; skipping composite apunim grid."
        )
        return

    if models is None:
        models = _apunim_grid_models(
            annotations_dir, dataset_keys, prompt_name
        )
    if not models:
        print("No LLM annotation files found; skipping composite apunim grid.")
        return

    columns = ["Human"] + models
    fig = plt.figure(constrained_layout=True, figsize=(8, 2))
    subfigures = fig.subfigures(
        nrows=len(dataset_keys), ncols=1, height_ratios=[1] * len(dataset_keys)
    )

    # matplotlib collapses a single-row subfigures() call to a bare
    # SubFigure rather than an array of length one.
    if len(dataset_keys) == 1:
        subfigures = [subfigures]

    for r, dataset_key in enumerate(dataset_keys):
        _draw_apunim_row(
            subfigures[r],
            dataset_key,
            human_datasets[dataset_key].get_name(),
            human_datasets[dataset_key],
            annotations_dir,
            prompt_name,
            columns,
            sdb_columns_limit,
            is_first_row=(r == 0),
        )

    fig.supylabel("nDFU")
    fig.suptitle(title)
    graphs.save_plot(output_path)
    plt.close(fig)
