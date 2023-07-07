"""
3d Box Class inherit from nuscenes.utils.data_classes.box
"""
import pdb

import numpy as np
from typing import List, Tuple
from pyquaternion import Quaternion
from data.script.NUSC_CONSTANT import *
from nuscenes.utils.data_classes import Box


class nusc_box(Box):
    def __init__(self, center: List[float], size: List[float], rotation: List[float], label: int = np.nan,
                 score: float = np.nan, velocity: Tuple = (np.nan, np.nan, np.nan), name: str = None,
                 token: str = None):
        """
        following notes are from nuscenes.utils.data_classes.box
        :param center: Center of box given as x, y, z.
        :param size: Size of box in width, length, height.
        :param rotation: Box orientation.
        :param label: Integer label, optional.
        :param score: Classification score, optional.
        :param velocity: Box velocity in x, y, z direction.
        :param name: Box name, optional. Can be used e.g. for denote category name.
        :param token: Unique string identifier from DB.
        """
        super().__init__(center, size, self.abs_orientation_axisZ(Quaternion(rotation)), 
                         label, score, velocity, name, token)
        assert self.orientation.axis[-1] >= 0
        
        self.tracking_id = None
        self.yaw = self.orientation.radians
        self.name_label = CLASS_SEG_TO_STR_CLASS[name]
        self.bottom_corners_ = self.bottom_corners()[:2].T  # [4, 2]
        self.volume, self.area = self.box_volum(), self.box_bottom_area()

    @staticmethod
    def abs_orientation_axisZ(orientation: Quaternion) -> Quaternion:
        # Double Cover, align with subsequent motion model
        return -orientation if orientation.axis[-1] < 0 else orientation

    def box_volum(self) -> float:
        return self.wlh[0] * self.wlh[1] * self.wlh[2]

    def box_bottom_area(self) -> float:
        return self.wlh[0] * self.wlh[1]
