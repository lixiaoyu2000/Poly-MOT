"""
data format conversion and data concat on the NuScenes dataset
"""

import pdb
import numpy as np
from geometry.nusc_box import nusc_box
from pyquaternion import Quaternion
from typing import List, Tuple, Union
from data.script.NUSC_CONSTANT import *


def concat_box_attr(nuscbox: nusc_box, *attrs) -> List:
    res = []
    for attr in attrs:
        res += getattr(nuscbox, attr)
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


def dictdet2array(dets: List[dict], *attrs) -> Union[List, np.array]:
    listdets = [concat_dict_attr(det, *attrs) for det in dets if det['detection_name'] in CLASS_SEG_TO_STR_CLASS]
    return listdets, np.array(listdets)


def arraydet2box(dets: np.array) -> Union[np.array[nusc_box], np.array]:
    # det -> (x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), det_score, class_label)
    if dets.ndim == 1: dets = dets[None, :]
    assert dets.shape[1] == 14, "The number of observed states must satisfy 14"
    nusc_boxes, boxes_bottom_corners = [], []
    for idx, det in enumerate(dets):
        curr_box = nusc_box(center=det[0:3], size=det[3:6], rotation=det[8:12],
                            velocity=tuple(det[6:8].tolist() + [0.0]), score=det[12],
                            name=CLASS_STR_TO_SEG_CLASS[int(det[13])])
        nusc_boxes.append(curr_box)
        boxes_bottom_corners.append(curr_box.bottom_corners_)
    return np.array(nusc_boxes), np.array(boxes_bottom_corners)
