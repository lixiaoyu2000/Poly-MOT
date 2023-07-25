"""
category-specific matching strategy, see more details on the Poly-MOT paper
Three implemented matching methods(Greedy, Hungarian, Mutual Nearest Neighbor(MNN))
TODO: to support more matching methods
"""

import lap
import numpy as np
from typing import Tuple


def Hungarian(cost_matrix: np.array, thresholds: dict) -> Tuple[list, list, np.array, np.array]:
    """implement hungarian algorithm with lap

    Args:
        cost_matrix (np.array): 3-ndim [N_cls, N_det, N_tra] or 2-ndim, invaild cost equal to np.inf
        thresholds (dict): matching thresholds to restrict FP matches

    Returns:
        Tuple[list, list, np.array, np.array]: matched det, matched tra, unmatched det, unmatched tra
    """
    assert cost_matrix.ndim == 2 or cost_matrix.ndim == 3, "cost matrix must be valid."
    if cost_matrix.ndim == 2: cost_matrix = cost_matrix[None, :, :]
    assert len(thresholds) == cost_matrix.shape[0], "the number of thresholds should be equal to cost matrix number."

    # solve cost matrix
    m_det, m_tra = [], []
    for cls_idx, cls_cost in enumerate(cost_matrix):
        _, x, y = lap.lapjv(cls_cost, extend_cost=True, cost_limit=thresholds[cls_idx])
        for ix, mx in enumerate(x):
            if mx >= 0:
                assert (ix not in m_det) and (mx not in m_tra) 
                m_det.append(ix)
                m_tra.append(mx)
                            
    # unmatched tra and det
    num_det, num_tra = cost_matrix.shape[1:]
    if len(m_det) == 0:
        um_det, um_tra = np.arange(num_det), np.arange(num_tra)
    else:
        um_det = np.setdiff1d(np.arange(num_det), np.array(m_det))
        um_tra = np.setdiff1d(np.arange(num_tra), np.array(m_tra))

    return m_det, m_tra, um_det, um_tra


def Greedy(cost_matrix: np.array, thresholds: dict) -> Tuple[list, list, np.array, np.array]:
    """implement greedy algorithm

    Args:
        cost_matrix (np.array): 3-ndim [N_cls, N_det, N_tra] or 2-ndim, invaild cost equal to np.inf
        thresholds (dict): matching thresholds to restrict FP matches

    Returns:
        Tuple[list, list, np.array, np.array]: matched det, matched tra, unmatched det, unmatched tra
    """
    assert cost_matrix.ndim == 2 or cost_matrix.ndim == 3, "cost matrix must be valid."
    if cost_matrix.ndim == 2: cost_matrix = cost_matrix[None, :, :]
    assert len(thresholds) == cost_matrix.shape[0], "the number of thresholds should be egual to cost matrix number."
    
    # solve cost matrix
    m_det, m_tra = [], []
    num_det, num_tra = cost_matrix.shape[1:] 
    for cls_idx, cls_cost in enumerate(cost_matrix):
        for det_idx in range(num_det):
            tra_idx = cls_cost[det_idx].argmin()
            if cls_cost[det_idx][tra_idx] <= thresholds[cls_idx]:
                cost_matrix[cls_idx, :, tra_idx] = 1e18
                m_det.append(det_idx)
                m_tra.append(tra_idx)
    
    # unmatched tra and det
    if len(m_det) == 0:
        um_det, um_tra = np.arange(num_det), np.arange(num_tra)
    else:
        um_det = np.setdiff1d(np.arange(num_det), np.array(m_det))
        um_tra = np.setdiff1d(np.arange(num_tra), np.array(m_tra))
    
    return m_det, m_tra, um_det, um_tra

def MNN(cost_matrix: np.array, thresholds: dict) -> Tuple[list, list, np.array, np.array]:
    """implement MNN(Mutual Nearest Neighbor) algorithm

    Args:
        cost_matrix (np.array): 3-ndim [N_cls, N_det, N_tra] or 2-ndim, invaild cost equal to np.inf
        thresholds (dict): matching thresholds to restrict FP matches

    Returns:
        Tuple[list, list, np.array, np.array]: matched det, matched tra, unmatched det, unmatched tra
    """
    assert cost_matrix.ndim == 2 or cost_matrix.ndim == 3, "cost matrix must be valid."
    if cost_matrix.ndim == 2: cost_matrix = cost_matrix[None, :, :]
    assert len(thresholds) == cost_matrix.shape[0], "the number of thresholds should be egual to cost matrix number."
    
    # solve cost matrix
    m_det, m_tra = [], []
    num_det, num_tra = cost_matrix.shape[1:] 
    for cls_idx, cls_cost in enumerate(cost_matrix):
        mask = cls_cost <= thresholds[cls_idx]
        # mutual nearest
        mask = mask \
            * (cls_cost == cls_cost.min(axis= 0)) \
            * (cls_cost == cls_cost.min(axis= 1)[:, np.newaxis])
        m_det += np.where(mask == True)[0].tolist()
        m_tra += np.where(mask == True)[1].tolist()
    
    # unmatched tra and det
    if len(m_det) == 0:
        um_det, um_tra = np.arange(num_det), np.arange(num_tra)
    else:
        um_det = np.setdiff1d(np.arange(num_det), np.array(m_det))
        um_tra = np.setdiff1d(np.arange(num_tra), np.array(m_tra))
    return m_det, m_tra, um_det, um_tra