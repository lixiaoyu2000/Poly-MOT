"""
one/two-stage category-specific matching strategy, see two-stage details on the Poly-MOT paper
Three implemented matching methods(Greedy, Hungarian, Mutual Nearest Neighbor(MNN))
TODO: to support more matching methods
"""

import lap
import numpy as np
from typing import Tuple


def Hungarian(cost_matrix: np.array, threshold: dict) -> Tuple[list, list, np.array, np.array]:
    """implement hungarian algorithm with lap

    Args:
        cost_matrix (np.array): 3-ndim or 2-ndim, [N_cls, N_det, N_tra], invaild cost equal to np.inf
        threshold (dict): matching threshold to restrict FP matches

    Returns:
        Tuple[list, list, np.array, np.array]: matched det, matched tra, unmatched det, unmatched tra
    """

    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )
    matches, unmatched_a, unmatched_b = [], [], []
    # cost:总代价, x:一个大小为n的数组，用于指定每一行被分配到哪一列, y: 一个大小为n的数组，用于指定每列被分配到哪一行
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=threshold)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    # 没有匹配的det
    unmatched_a = np.where(x < 0)[0]
    # 没有匹配的pre
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b
