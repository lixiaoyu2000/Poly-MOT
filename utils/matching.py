"""
one/two-stage category-specific matching strategy, see two-stage details on the Poly-MOT paper
Three implemented matching methods(Greedy, Hungarian, Mutual Nearest Neighbor(MNN))
TODO: to support more matching methods
"""

import numpy as np
from data.script.NUSC_CONSTANT import *
from geometry.nusc_utils import mask_between_boxes


def mask_tras_dets(cls_num, det_labels, tra_labels) -> np.array:
    """
    mask invalid cost between tras and dets
    :return: np.array[bool], [cls_num, det_num, tra_num], True denotes valid (det label == tra label == cls idx)
    """
    det_num, tra_num = len(det_labels), len(tra_labels)
    cls_mask = np.ones(shape=(cls_num, det_num, tra_num)) * np.arange(cls_num)[:, None, None]
    # [det_num, tra_num], True denotes invalid(diff cls)
    same_mask, _ = mask_between_boxes(det_labels, tra_labels)
    # [det_num, tra_num], invalid idx assign -1
    tmp_labels = tra_labels[None, :].repeat(det_num, axis=0)
    tmp_labels[np.where(same_mask)] = -1
    return tmp_labels[None, :, :].repeat(cls_num, axis=0) == cls_mask


def fast_compute_check(metrics: dict, second_metric: str) -> bool:
    """
    Whether cost matrix can be quickly constructed
    :param: dict, similarity metric for each class
    :param: str, similarity metric for second stage
    :return: bool, True -> fast computation
    """
    used_metrics = [m for _, m in metrics.items()] + [second_metric]
    assert len(used_metrics) != 0, 'must have metrics for association'
    return True if len(set(used_metrics) - set(FAST_METRIC)) == 0 else False


def reorder_metrics(metrics: dict) -> dict:
    """
    reorder metrics from {key(class, int): value(metrics, str)} to {key(metrics, str): value(class_labels, list)}
    :param metrics: dict, format: {key(class, int): value(metrics, str)}
    :return: dict, {key(metrics, str): value(class_labels, list)}
    """
    new_metrics = {}
    for cls, metric in metrics:
        if metric in new_metrics: new_metrics[metric].append(cls)
        else: new_metrics[metric] = []
    return new_metrics


def spec_metric_mask(cls_list: list, det_labels: np.array, tra_labels: np.array) -> np.array:
    """
    mask matrix, merge all object instance index of the specific class
    :param cls_list: list, valid category list
    :param det_labels: np.array, class labels of detections
    :param tra_labels: np.array, class labels of trajectories
    :return: np.array[bool], True denotes invalid(the object's category is not specific)
    """
    det_num, tra_num = len(det_labels), len(tra_labels)
    metric_mask = np.ones((det_num, tra_num), dtype=bool)
    merge_det_idx = [idx for idx, cls in enumerate(det_labels) if cls in cls_list]
    merge_tra_idx = [idx for idx, cls in enumerate(tra_labels) if cls in cls_list]
    metric_mask[np.ix_(merge_det_idx, merge_tra_idx)] = False
    return metric_mask
