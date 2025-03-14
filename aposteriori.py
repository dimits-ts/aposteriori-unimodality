from typing import Hashable, Any
import math
import numpy as np
import scipy

from ndfu import ndfu


def aposteriori_unimodality(all_annotations: list[int], annotator_group: list[Any], comment_group: list[Any]) -> float:
    all_annotations = np.array(all_annotations)
    annotator_group = np.array(annotator_group)

    all_comment_stats = []
    for curr_comment_id in np.unique(comment_group):
        is_in_curr_comment = comment_group == curr_comment_id
        all_comment_annotations = all_annotations[is_in_curr_comment]
        comment_annotator_groups = annotator_group[is_in_curr_comment]
        
        curr_comment_stats = _aposteriori_comment(all_comment_annotations, comment_annotator_groups)
        all_comment_stats.extend(curr_comment_stats)


    #print("DEBUG: final stats: ", stats)
    return _wilcox(curr_comment_stats)


def _aposteriori_comment(all_comment_annotations: np.ndarray, annotator_group: np.ndarray) -> np.ndarray:
    """
    Generate the aposteriori stat (ndfu diff stat) for each annotation level, for one comment.

    :param all_comment_annotations: An array containing all annotations for the current comment
    :type all_comment_annotations: np.ndarray
    :param annotator_group: An array where each value is a distinct level of the currently considered factor
    :type annotator_group: np.ndarray
    :return: A numpy array containing the ndfu stats for each level of the currently considered factor, for one comment
    :rtype: np.ndarray
    """
    stats = []
    for annot_group in np.unique(annotator_group):
        factor_annotations = all_comment_annotations[annotator_group == annot_group]

        if len(factor_annotations) != 0:
            aposteriori_stat = _ndfu_diff(
                all_comment_annotations, factor_annotations
            )
            stats.append(aposteriori_stat)
    return np.array(stats)



def _wilcox(level_aposteriori_statistics: list[float]) -> float:
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
        return scipy.stats.wilcoxon(x=x, y=y, alternative="less").pvalue


def _ndfu_diff(
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
