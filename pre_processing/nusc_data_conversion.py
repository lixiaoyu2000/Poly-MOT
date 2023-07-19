"""
data format conversion and data concat on the NuScenes dataset
"""

import pdb
import numpy as np
from typing import Tuple, List
from geometry import NuscBox
from data.script.NUSC_CONSTANT import *


def concat_box_attr(nuscbox: NuscBox, *attrs) -> List:
    res = []
    for attr in attrs:
        tmp_attr = getattr(nuscbox, attr)
        if isinstance(tmp_attr, list):
            res += getattr(nuscbox, attr)
        elif isinstance(tmp_attr, (float, int)):
            res +=[tmp_attr]
        elif isinstance(tmp_attr, np.ndarray):
            res += tmp_attr.tolist()
        else: raise Exception("unsupport date format to concat")
    return res


def concat_dict_attr(dictbox: dict, *attrs) -> List:
    res = []
    for attr in attrs:
        if attr == 'detection_name':
            res += [CLASS_SEG_TO_STR_CLASS[dictbox[attr]]]
            continue
        elif attr == 'detection_score':
            res += [dictbox[attr]]
            continue
        res += dictbox[attr]
    return res


def dictdet2array(dets: List[dict], *attrs) -> Tuple[List, np.array]:
    listdets = [concat_dict_attr(det, *attrs) for det in dets if det['detection_name'] in CLASS_SEG_TO_STR_CLASS]
    return listdets, np.array(listdets)


def arraydet2box(dets: np.array, ids: np.array = None):
    # det -> (x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), det_score, class_label)
    if dets.ndim == 1: dets = dets[None, :]
    assert dets.shape[1] == 14, "The number of observed states must satisfy 14"
    NuscBoxes, boxes_bottom_corners = [], []
    for idx, det in enumerate(dets):
        curr_box = NuscBox(center=det[0:3], size=det[3:6], rotation=det[8:12],
                            velocity=tuple(det[6:8].tolist() + [0.0]), score=det[12],
                            name=CLASS_STR_TO_SEG_CLASS[int(det[13])])
        if ids is not None: curr_box.tracking_id = int(ids[idx])
        NuscBoxes.append(curr_box)
        boxes_bottom_corners.append(curr_box.bottom_corners_)
    return np.array(NuscBoxes), np.array(boxes_bottom_corners)
