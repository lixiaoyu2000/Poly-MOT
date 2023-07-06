"""
Non-Maximum Suppression(NMS) ops for the NuScenes dataset
Three implemented NMS versions(blend_nms, no_blend_nms, blend_soft_nms)
TODO: to support more NMS versions
"""
import pdb

import numpy as np
import numba as nb
from typing import List
from geometry import nusc_box
from data.script.NUSC_CONSTANT import *
from geometry.nusc_distance import iou_bev_s, iou_3d_s, giou_bev_s, giou_3d_s, d_eucl_s
from geometry.nusc_distance import iou_bev, iou_3d, giou_bev, giou_3d, d_eucl


def blend_nms(box_infos: dict, metrics: str, thre: float) -> List[int]:
    """
    :param box_infos: dict, a collection of nusc_box info, keys must contain 'np_dets' and 'np_dets_bottom_corners'
    :param metrics: str, similarity metric for nms, five implemented metrics(iou_bev, iou_3d, giou_bev, giou_3d, d_eluc)
    :param thre: float, threshold of filter
    :return: keep box index, List[int]
    """
    assert metrics in ['iou_bev', 'iou_3d', 'giou_bev', 'giou_3d', 'd_eucl'], "unsupported NMS metrics"
    assert 'np_dets' in box_infos and 'np_dets_bottom_corners' in box_infos, 'must contain specified keys'

    infos, corners = box_infos['np_dets'], box_infos['np_dets_bottom_corners']
    sort_idxs, keep = np.argsort(-infos[:, -2]), []
    while sort_idxs.size > 0:
        i = sort_idxs[0]
        keep.append(i)
        # only one box left
        if sort_idxs.size == 1: break
        left, first = [{'np_dets_bottom_corners': corners[idx], 'np_dets': infos[idx]} for idx in [sort_idxs[1:], i]]
        # the return value number varies by distinct metrics
        if metrics not in METRIC: distances = globals()[metrics](first, left)[0]
        else: distances = globals()[metrics](first, left)[1][0]
        sort_idxs = sort_idxs[1:][distances <= thre]
    return keep
