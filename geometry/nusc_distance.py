"""
Similarity metric for computing geometry distance between tracklet and det(cost matrix) or det and det(NMS)
Five implemented Similarity metrics for the NuScenes dataset
half-parallel: iou_bev, iou_3d, giou_bev, giou_3d, d_eucl
Serial: iou_bev_s, iou_3d_s, giou_bev_s, giou_3d_s, d_eucl_s
TODO: make d_eucl parallel; support more similarity metrics; speed up computation

Thanks: Part codes are inspired by SimpleTrack and AB3DMOT and EagerMOT
Code URL: SimpleTrack(https://github.com/tusen-ai/SimpleTrack) AB3DMOT(https://github.com/xinshuoweng/AB3DMOT)
EagerMOT(https://github.com/aleksandrkim61/EagerMOT)
"""

import pdb
import numpy as np
from typing import List, Tuple, Union
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
from utils import expand_dims
from pre_processing.nusc_data_conversion import concat_box_attr
from geometry import PolyArea2D_s, PolyArea2D, yaw_punish_factor, \
    mask_between_boxes, logical_or_mask, loop_convex, loop_inter, nusc_box


def iou_bev_s(box_a: nusc_box, box_b: nusc_box) -> float:
    """
    Serial implementation of iou bev
    :param box_a: nusc_box
    :param box_b: nusc_box
    :return: float, iou between two boxes under Bird's Eye View(BEV)
    """
    if box_b.name != box_a.name:
        return -np.inf
    boxa_corners, boxb_corners = np.array(box_a.bottom_corners_), np.array(box_b.bottom_corners_)
    reca, recb = Polygon(boxa_corners), Polygon(boxb_corners)
    inter_area = reca.intersection(recb).area
    ioubev = inter_area / (box_a.area + box_b.area - inter_area)
    return ioubev


def iou_3d_s(box_a: nusc_box, box_b: nusc_box) -> Tuple[float, float]:
    """
    Serial implementation of 3d iou
    :param box_a: nusc_box
    :param box_b: nusc_box
    :return: [float, float], 3d/bev iou between two boxes
    """
    if box_b.name != box_a.name:
        return -np.inf, -np.inf
    boxa_corners, boxb_corners = np.array(box_a.bottom_corners_), np.array(box_b.bottom_corners_)
    reca, recb = Polygon(boxa_corners), Polygon(boxb_corners)
    inter_area = reca.intersection(recb).area
    ha, hb, za, zb = box_a.wlh[2], box_b.wlh[2], box_a.center[2], box_b.center[2]
    overlap_height = max(0, min(za + ha / 2, zb + hb / 2) - max(za - ha / 2, zb - hb / 2))
    inter_volume = inter_area * overlap_height
    iou3d = inter_volume / (box_a.volume + box_b.volume - inter_volume)
    ioubev = inter_area / (box_a.area + box_b.area - inter_area)
    return ioubev, iou3d


def giou_bev_s(box_a: nusc_box, box_b: nusc_box) -> float:
    """
    Serial implementation of giou(Generalized Intersection over Union) under Bird's Eye View(BEV)
    :param box_a: nusc_box
    :param box_b: nusc_box
    :return: float, giou between two boxes under Bird's Eye View(BEV)
    """
    if box_b.name != box_a.name:
        return -np.inf
    boxa_corners, boxb_corners = np.array(box_a.bottom_corners_), np.array(box_b.bottom_corners_)
    reca, recb = Polygon(boxa_corners), Polygon(boxb_corners)
    inter_area = reca.intersection(recb).area
    union_area = box_a.area + box_b.area - inter_area

    # ConvexHull of two boxes
    all_corners = np.vstack((boxa_corners, boxb_corners))
    C = ConvexHull(all_corners)
    convex_corners = all_corners[C.vertices]

    # obtain convexhull area using vector cross product
    convex_area = PolyArea2D_s(convex_corners)

    # check formula on Poly-MOT paper; value interval -> [-1, 1]
    gioubev = inter_area / union_area - (convex_area - union_area) / convex_area
    return gioubev


def giou_3d_s(box_a: nusc_box, box_b: nusc_box) -> Tuple[float, float]:
    """
    Serial implementation of 3d giou
    :param box_a: nusc_box
    :param box_b: nusc_box
    :return: 3d giou between two boxes
    """
    if box_b.name != box_a.name:
        return -np.inf, -np.inf
    boxa_corners, boxb_corners = np.array(box_a.bottom_corners_), np.array(box_b.bottom_corners_)
    reca, recb = Polygon(boxa_corners), Polygon(boxb_corners)
    ha, hb, za, zb = box_a.wlh[2], box_b.wlh[2], box_a.center[2], box_b.center[2]

    # overlap/union area under 2D/3D view between two boxes
    overlap_height = max(0, min(za + ha / 2, zb + hb / 2) - max(za - ha / 2, zb - hb / 2))
    union_height = max((za + ha / 2), (zb + hb / 2)) - min((zb - hb / 2), (za - ha / 2))
    inter_area = reca.intersection(recb).area
    union_area = box_a.area + box_b.area - inter_area
    inter_volume = inter_area * overlap_height
    union_volume = box_a.volume + box_b.volume - inter_volume

    # convexhull area under 2D/3D view between two boxes
    all_corners = np.vstack((boxa_corners, boxb_corners))
    C = ConvexHull(all_corners)
    convex_corners = all_corners[C.vertices]
    convex_area = PolyArea2D_s(convex_corners)
    convex_volume = convex_area * union_height

    # gioubev and giou3d between two boxes
    giou3d = inter_volume / union_volume - (convex_volume - union_volume) / convex_volume
    gioubev = inter_area / union_area - (convex_area - union_area) / convex_area
    return gioubev, giou3d


def d_eucl_s(box_a: nusc_box, box_b: nusc_box) -> float:
    """
    Serial implementation of Euclidean Distance with yaw angle punish
    :param box_a: nusc_box
    :param box_b: nusc_box
    :return: Eucl distance between two nusc_box
    """
    if box_b.name != box_a.name:
        return np.inf
    boxa_vector, boxb_vector = concat_box_attr(box_a, 'center', 'wlh'), concat_box_attr(box_b, 'center', 'wlh')
    eucl_dis = np.linalg.norm(np.array(boxa_vector - boxb_vector))
    punish_factor = yaw_punish_factor(box_a, box_b)
    return eucl_dis * punish_factor


def d_eucl(boxes_a, boxes_b) -> np.array:
    """
    Serial implementation of Euclidean Distance with yaw angle punish
    :param boxes_a: np.array[nusc_box], a collection of nusc_box
    :param boxes_b: np.array[nusc_box], a collection of nusc_box
    :return: Eucl distance between two collections
    """
    eucl_dis = np.zeros((len(boxes_a), len(boxes_b)))
    for i, boxa in enumerate(boxes_a):
        for j, boxb in enumerate(boxes_b):
            eucl_dis[i, j] = d_eucl_s(boxa, boxb)
    return eucl_dis


def giou_3d(boxes_a: dict, boxes_b: dict) -> Tuple[np.array, np.array]:
    """
    half-parallel implementation of 3d giou. why half? convexhull and intersection are still serial
    'np_dets': np.array, [det_num, 14](x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), det_score, class_label)
    'np_dets_bottom_corners': np.array, [det_num, 4, 2]
    :param boxes_a: dict, a collection of nusc_box info, keys must contain 'np_dets' and 'np_dets_bottom_corners'
    :param boxes_b: dict, a collection of nusc_box info, keys must contain 'np_dets' and 'np_dets_bottom_corners'
    :return: [np.array, np.array], 3d giou/bev giou between two boxes collections
    """
    assert 'np_dets' in boxes_a and 'np_dets_bottom_corners' in boxes_a, 'must contain specified keys'
    assert 'np_dets' in boxes_b and 'np_dets_bottom_corners' in boxes_b, 'must contain specified keys'

    # load info
    infos_a, infos_b = boxes_a['np_dets'], boxes_b['np_dets']  # [box_num, 14]
    bcs_a, bcs_b = boxes_a['np_dets_bottom_corners'], boxes_b['np_dets_bottom_corners']  # [box_num, 4, 2]

    # corner case, 1d array(collection only has one box) to 2d array
    if infos_a.ndim == 1: infos_a, bcs_a = infos_a[None, :], bcs_a[None, :]
    if infos_b.ndim == 1: infos_b, bcs_b = infos_b[None, :], bcs_b[None, :]
    assert infos_a.shape[1] == 14 and infos_b.shape[1] == 14, "dim must be 14"

    # mask matrix, True denotes different(invalid), False denotes same(valid)
    bool_mask, seq_mask = mask_between_boxes(infos_a[:, -1], infos_b[:, -1])
    bool_mask, seq_mask = logical_or_mask(bool_mask, seq_mask, boxes_a, boxes_b)

    # process bottom corners, size and center for parallel computing
    rep_bcs_a, rep_bcs_b = expand_dims(bcs_a, len(bcs_b), 1), expand_dims(bcs_b, len(bcs_a), 0)  # [a_num, b_num, 4, 2]
    wlh_a, wlh_b = expand_dims(infos_a[:, 3:6], len(infos_b), 1), expand_dims(infos_b[:, 3:6], len(infos_a), 0)
    za, zb = expand_dims(infos_a[:, 2], len(infos_b), 1), expand_dims(infos_b[:, 2], len(infos_a), 0)  # [a_num, b_num]
    wa, la, ha = wlh_a[:, :, 0], wlh_a[:, :, 1], wlh_a[:, :, 2]
    wb, lb, hb = wlh_b[:, :, 0], wlh_b[:, :, 1], wlh_b[:, :, 2]

    # polygons
    polys_a, polys_b = [Polygon(bc_a) for bc_a in bcs_a], [Polygon(bc_b) for bc_b in bcs_b]

    # overlap and union height
    ohs = np.maximum(np.zeros_like(ha), np.minimum(za + ha / 2, zb + hb / 2) - np.maximum(za - ha / 2, zb - hb / 2))
    uhs = np.maximum((za + ha / 2), (zb + hb / 2)) - np.minimum((zb - hb / 2), (za - ha / 2))

    # overlap and union area/volume
    inter_areas = loop_inter(polys_a, polys_b, bool_mask)
    inter_volumes = inter_areas * ohs
    union_areas, union_volumes = wa * la + wb * lb - inter_areas, wa * la * ha + wb * lb * hb - inter_volumes

    # convexhull area/volume
    convex_areas = loop_convex(rep_bcs_a, rep_bcs_b, seq_mask)
    convex_volumes = convex_areas * uhs

    # calu gioubev/giou3d and mask invalid value
    gioubev = inter_areas / union_areas - (convex_areas - union_areas) / convex_areas
    giou3d = inter_volumes / union_volumes - (convex_volumes - union_volumes) / convex_volumes
    giou3d[bool_mask], gioubev[bool_mask] = -np.inf, -np.inf

    return gioubev, giou3d


def giou_bev(boxes_a: dict, boxes_b: dict) -> np.array:
    """
    half-parallel implementation of bev giou.
    'np_dets': np.array, [det_num, 14](x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), det_score, class_label)
    'np_dets_bottom_corners': np.array, [det_num, 4, 2]
    :param boxes_a: dict, a collection of nusc_box info, keys must contain 'np_dets' and 'np_dets_bottom_corners'
    :param boxes_b: dict, a collection of nusc_box info, keys must contain 'np_dets' and 'np_dets_bottom_corners'
    :return: np.array, bev giou between two boxes collections
    """
    assert 'np_dets' in boxes_a and 'np_dets_bottom_corners' in boxes_a, 'must contain specified keys'
    assert 'np_dets' in boxes_b and 'np_dets_bottom_corners' in boxes_b, 'must contain specified keys'

    # load info
    infos_a, infos_b = boxes_a['np_dets'], boxes_b['np_dets']  # [box_num, 14]
    bcs_a, bcs_b = boxes_a['np_dets_bottom_corners'], boxes_b['np_dets_bottom_corners']  # [box_num, 4, 2]

    # 1d array to 2d array
    if infos_a.ndim == 1: infos_a, bcs_a = infos_a[None, :], bcs_a[None, :]
    if infos_b.ndim == 1: infos_b, bcs_b = infos_b[None, :], bcs_b[None, :]
    assert infos_a.shape[1] == 14 and infos_b.shape[1] == 14, "dim must be 14"

    # mask matrix, True denotes different, False denotes same
    bool_mask, seq_mask = mask_between_boxes(infos_a[:, -1], infos_b[:, -1])
    bool_mask, seq_mask = logical_or_mask(bool_mask, seq_mask, boxes_a, boxes_b)

    # process bottom corners, size and center for parallel computing
    rep_bcs_a, rep_bcs_b = expand_dims(bcs_a, len(bcs_b), 1), expand_dims(bcs_b, len(bcs_a), 0)  # [a_num, b_num, 4, 2]
    wlh_a, wlh_b = expand_dims(infos_a[:, 3:6], len(infos_b), 1), expand_dims(infos_b[:, 3:6], len(infos_a), 0)
    wa, la, wb, lb = wlh_a[:, :, 0], wlh_a[:, :, 1], wlh_b[:, :, 0], wlh_b[:, :, 1]

    # polygons
    polys_a, polys_b = [Polygon(bc_a) for bc_a in bcs_a], [Polygon(bc_b) for bc_b in bcs_b]

    # overlap and union area
    inter_areas = loop_inter(polys_a, polys_b, bool_mask)
    union_areas = wa * la + wb * lb - inter_areas

    # convexhull area/volume
    convex_areas = loop_convex(rep_bcs_a, rep_bcs_b, seq_mask)

    # calu gioubev and mask invalid value
    gioubev = inter_areas / union_areas - (convex_areas - union_areas) / convex_areas
    gioubev[bool_mask] = -np.inf

    return gioubev


def iou_bev(boxes_a: dict, boxes_b: dict) -> np.array:
    """
    half-parallel implementation of bev iou.
    'np_dets': np.array, [det_num, 14](x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), det_score, class_label)
    'np_dets_bottom_corners': np.array, [det_num, 4, 2]
    :param boxes_a: dict, a collection of nusc_box info, keys must contain 'np_dets' and 'np_dets_bottom_corners'
    :param boxes_b: dict, a collection of nusc_box info, keys must contain 'np_dets' and 'np_dets_bottom_corners'
    :return: np.array, bev iou between two boxes collections
    """
    assert 'np_dets' in boxes_a and 'np_dets_bottom_corners' in boxes_a, 'must contain specified keys'
    assert 'np_dets' in boxes_b and 'np_dets_bottom_corners' in boxes_b, 'must contain specified keys'

    # load info
    infos_a, infos_b = boxes_a['np_dets'], boxes_b['np_dets']  # [box_num, 14]
    bcs_a, bcs_b = boxes_a['np_dets_bottom_corners'], boxes_b['np_dets_bottom_corners']  # [box_num, 4, 2]

    # corner case, 1d array(collection only has one box) to 2d array
    if infos_a.ndim == 1: infos_a, bcs_a = infos_a[None, :], bcs_a[None, :]
    if infos_b.ndim == 1: infos_b, bcs_b = infos_b[None, :], bcs_b[None, :]
    assert infos_a.shape[1] == 14 and infos_b.shape[1] == 14, "dim must be 14"

    # mask matrix, True denotes different, False denotes same
    bool_mask, seq_mask = mask_between_boxes(infos_a[:, -1], infos_b[:, -1])
    bool_mask, _ = logical_or_mask(bool_mask, seq_mask, boxes_a, boxes_b)

    # process bottom corners, size and center for parallel computing
    wlh_a, wlh_b = expand_dims(infos_a[:, 3:6], len(infos_b), 1), expand_dims(infos_b[:, 3:6], len(infos_a), 0)
    wa, la, wb, lb = wlh_a[:, :, 0], wlh_a[:, :, 1], wlh_b[:, :, 0], wlh_b[:, :, 1]

    # polygons
    polys_a, polys_b = [Polygon(bc_a) for bc_a in bcs_a], [Polygon(bc_b) for bc_b in bcs_b]

    # overlap and union area
    inter_areas = loop_inter(polys_a, polys_b, bool_mask)
    union_areas = wa * la + wb * lb - inter_areas

    # calu bev iou and mask invalid value
    ioubev = inter_areas / union_areas
    ioubev[bool_mask] = -np.inf

    return ioubev


def iou_3d(boxes_a: dict, boxes_b: dict) -> Tuple[np.array, np.array]:
    """
    half-parallel implementation of 3d iou.
    'np_dets': np.array, [det_num, 14](x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), det_score, class_label)
    'np_dets_bottom_corners': np.array, [det_num, 4, 2]
    :param boxes_a: dict, a collection of nusc_box info, keys must contain 'np_dets' and 'np_dets_bottom_corners'
    :param boxes_b: dict, a collection of nusc_box info, keys must contain 'np_dets' and 'np_dets_bottom_corners'
    :return: [np.array, np.array], 3d iou/bev iou between two boxes collections
    """
    assert 'np_dets' in boxes_a and 'np_dets_bottom_corners' in boxes_a, 'must contain specified keys'
    assert 'np_dets' in boxes_b and 'np_dets_bottom_corners' in boxes_b, 'must contain specified keys'

    # load info
    infos_a, infos_b = boxes_a['np_dets'], boxes_b['np_dets']  # [box_num, 14]
    bcs_a, bcs_b = boxes_a['np_dets_bottom_corners'], boxes_b['np_dets_bottom_corners']  # [box_num, 4, 2]

    # corner case, 1d array(collection only has one box) to 2d array
    if infos_a.ndim == 1: infos_a, bcs_a = infos_a[None, :], bcs_a[None, :]
    if infos_b.ndim == 1: infos_b, bcs_b = infos_b[None, :], bcs_b[None, :]
    assert infos_a.shape[1] == 14 and infos_b.shape[1] == 14, "dim must be 14"

    # mask matrix, True denotes different, False denotes same
    bool_mask, seq_mask = mask_between_boxes(infos_a[:, -1], infos_b[:, -1])
    bool_mask, _ = logical_or_mask(bool_mask, seq_mask, boxes_a, boxes_b)

    # process bottom corners, size and center for parallel computing
    wlh_a, wlh_b = expand_dims(infos_a[:, 3:6], len(infos_b), 1), expand_dims(infos_b[:, 3:6], len(infos_a), 0)
    za, zb = expand_dims(infos_a[:, 2], len(infos_b), 1), expand_dims(infos_b[:, 2], len(infos_a), 0)  # [a_num, b_num]
    wa, la, ha = wlh_a[:, :, 0], wlh_a[:, :, 1], wlh_a[:, :, 2]
    wb, lb, hb = wlh_b[:, :, 0], wlh_b[:, :, 1], wlh_b[:, :, 2]

    # polygons
    polys_a, polys_b = [Polygon(bc_a) for bc_a in bcs_a], [Polygon(bc_b) for bc_b in bcs_b]

    # overlap height
    ohs = np.maximum(np.zeros_like(ha), np.minimum(za + ha / 2, zb + hb / 2) - np.maximum(za - ha / 2, zb - hb / 2))

    # overlap and union area/volume
    inter_areas = loop_inter(polys_a, polys_b, bool_mask)
    inter_volumes = inter_areas * ohs
    union_areas, union_volumes = wa * la + wb * lb - inter_areas, wa * la * ha + wb * lb * hb - inter_volumes

    # calu bev iou and mask invalid value
    ioubev, iou3d = inter_areas / union_areas, inter_volumes / union_volumes
    ioubev[bool_mask], iou3d[bool_mask] = -np.inf, -np.inf

    return ioubev, iou3d
