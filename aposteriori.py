from typing import Hashable
import math
import numpy as np
import scipy

from ndfu import ndfu


def aposteriori_unimodality(
    annotations: list[np.ndarray],
    annotator_group: list[np.ndarray],
    sample_ratio: float = 0.2,
    bootstrap_steps: int = 100,
) -> dict[Hashable, float]:
    """
    Conducts a statistical test to evaluate whether the polarization of annotations
    can be explained by a specific grouping feature. If the p-value is below the
    significance level, it suggests that the grouping feature contributes to polarization.

    Args:
        annotations (list[np.ndarray]): A list where each element contains the annotations
            for a single comment in a discussion.
        annotator_group (list[np.ndarray]): A list where each element contains the group
            assignments of annotators corresponding to the annotations for each comment.

    Returns:
        dict[Hashable, float]: A dictionary mapping each unique group (factor) to the
            p-value of the statistical test for that group.

    Raises:
        ValueError: If the number of comments in `annotations` and `annotator_group`
            do not match, or if the lengths of annotations and group assignments
            are inconsistent for any comment.
    """
    if len(annotations) != len(annotator_group):
        raise ValueError(
            "The number of comments in `annotations` and `annotator_group` must be the same."
        )

    if len(annotations) == 0:
        return {}

    for annotation, group in zip(annotations, annotator_group):
        if len(annotation) != len(group) or len(annotation) == 0 or len(group) == 0:
            raise ValueError(
                "Annotations and group assignments must have the same length for each comment."
            )

    for annotation, group in zip(annotations, annotator_group):
        if len(annotation) == 0 or len(group) == 0:
            raise ValueError("Comments should have at least one annotation.")

    # Initialize statistics for each group level
    all_annots = []
    for array in annotator_group:
        for key in array:
            all_annots.append(key)

    aposteriori_unit_statistics: dict[Hashable, list] = {
        key: [] for key in np.unique(all_annots)
    }

    # Calculate per-comment statistics for each group level
    for comment_annotations, comment_annotator_group in zip(
        annotations, annotator_group
    ):
        for level in np.unique(annotator_group):
            aposteriori_stat = bootrstrap_level_aposteriori_unit(
                comment_annotations,
                comment_annotator_group,
                level,
                sample_ratio=sample_ratio,
                bootstrap_steps=bootstrap_steps
            )
            aposteriori_unit_statistics[level].append(aposteriori_stat)

    # Aggregate statistics for the entire group
    aposteriori_final_statistics: dict[Hashable, float] = {}
    for level, stats in aposteriori_unit_statistics.items():
        aposteriori_final_statistics[level] = discussion_aposteriori(stats)

    return aposteriori_final_statistics


def bootrstrap_level_aposteriori_unit(
    annotations: np.ndarray,
    annotator_group: np.ndarray,
    level: Hashable,
    sample_ratio: float,
    bootstrap_steps: int,
) -> float:
    level_annotations = annotations[annotator_group == level]
    n_samples = max(math.floor(annotations.shape[0] * sample_ratio), 1)

    s_statistics = []
    for _ in range(bootstrap_steps):
        sample_level_annotations = np.random.choice(
            level_annotations, replace=True, size=n_samples
        )
        sample_annotations = np.random.choice(
            level_annotations, replace=True, size=n_samples
        )
        s_stat = aposteriori_unit(
            global_annotations=sample_annotations,
            level_annotations=sample_level_annotations,
        )
        s_statistics.append(s_stat)

    return float(np.mean(s_statistics))


def discussion_aposteriori(level_aposteriori_statistics: list[float]) -> float:
    """
    Performs a Wilcoxon signed-rank test to determine the significance of differences
    in aposteriori statistics for a specific group.

    Args:
        level_aposteriori_statistics (list[float]): A list of aposteriori statistics for a group.

    Returns:
        float: The p-value of the Wilcoxon test. If no difference is detected, returns 1.
    """
    x = level_aposteriori_statistics
    y = np.zeros_like(level_aposteriori_statistics)
    if np.sum(x - y) == 0:
        return 1.0
    else:
        return scipy.stats.wilcoxon(x, y=y, alternative="greater").pvalue


def aposteriori_unit(
    global_annotations: np.ndarray, level_annotations: np.ndarray
) -> float:
    """
    Computes the difference in normalized distance from unimodality (nDFU) between
    global annotations and annotations from a specific group.

    Args:
        global_annotations (np.ndarray): The full set of annotations for a comment.
        level_annotations (np.ndarray): The subset of annotations corresponding to a specific group.

    Returns:
        float: The difference in nDFU values, which indicates the contribution of the
        group to polarization.
    """
    global_ndfu = ndfu(global_annotations)
    level_ndfu = ndfu(level_annotations)
    return global_ndfu - level_ndfu
