"""
dataloader of NuScenes dataset
Obtain the observation information(detection) of each frame iteratively
--------ATTENTION: Detector files must be in chronological order-------
"""

import pdb
import numpy as np
from utils.io import load_file
from geometry.nusc_box import nusc_box
from data.script.NUSC_CONSTANT import *
from pre_processing import dictdet2array, arraydet2box, blend_nms


class NuScenesloader:
    def __init__(self, detection_path, first_token_path, config):
        """
        :param detection_path: path of order detection file
        :param first_token_path: path of first frame token for each seq
        :param config: dict, hyperparameter setting
        """
        # detector -> {sample_token:[{det1_info}, {det2_info}], ...}， check the detailed "det_info" at nuscenes.org
        self.detector = load_file(detection_path)["results"]
        self.all_sample_token = list(self.detector.keys())
        self.seq_first_token = load_file(first_token_path)
        self.config, self.data_info = config, {}
        self.SF_thre, self.NMS_thre = config['preprocessing']['SF_thre'], config['preprocessing']['NMS_thre']
        self.NMS_type, self.NMS_metric = config['preprocessing']['NMS_type'], config['preprocessing']['NMS_metric']
        self.seq_id = self.frame_id = 0

    def __getitem__(self, item):
        """
        data_info(dict): {
            'is_first_frame': bool
            'timestamp': int
            'sample_token': str
            'seq_id': int
            'frame_id': int
            'has_velo': bool
            'np_dets': np.array, [det_num, 14](x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), det_score, class_label)
            'np_dets_bottom_corners': np.array, [det_num, 4, 2]
            'box_dets': np.array[nusc_box], [det_num]
            'no_dets': bool, corner case,
            'det_num': int,
        }
        """
        curr_token = self.all_sample_token[item]
        ori_dets = self.detector[curr_token]

        # assign seq and frame id
        if curr_token in self.seq_first_token:
            self.seq_id += 1
            self.frame_id = 1
        else: self.frame_id += 1

        # all categories are blended together and sorted by detection score
        list_dets, np_dets = dictdet2array(ori_dets, 'translation', 'size', 'velocity', 'rotation',
                                           'detection_score', 'detection_name')

        # Score Filter based on category-specific thresholds
        np_dets = np.array([det for det in list_dets if det[-2] > self.SF_thre[det[-1]]])
        box_dets, np_dets_bottom_corners = arraydet2box(np_dets)
        assert len(np_dets) == len(box_dets) == len(np_dets_bottom_corners)

        # NMS, "blend" ref to blend all categories together during NMS
        if len(np_dets) != 0:
            tmp_infos = {'np_dets': np_dets, 'np_dets_bottom_corners': np_dets_bottom_corners}
            keep = globals()[self.NMS_type](box_infos=tmp_infos, metrics=self.NMS_metric, thre=self.NMS_thre)
            keep_num = len(keep)
        # corner case, no det left
        else: keep = keep_num = 0

        print(f"Total {len(list_dets) - keep_num} bboxes are filtered; "
              f"{len(list_dets) - len(np_dets)} during SF, "
              f"{len(np_dets) - keep_num} during NMS, "
              f"Still {keep_num} bboxes left. "
              f"seq id {self.seq_id}, frame id {self.frame_id}, "
              f"Total frame id {item + 1}.")

        # Available information for the current frame
        data_info = {
            'is_first_frame': curr_token in self.seq_first_token,
            'timestamp': item,
            'sample_token': curr_token,
            'seq_id': self.seq_id,
            'frame_id': self.frame_id,
            'has_velo': self.config['basic']['has_velo'],
            'np_dets': np_dets[keep] if keep_num != 0 else np.zeros(0),
            'np_dets_bottom_corners': np_dets_bottom_corners[keep] if keep_num != 0 else np.zeros(0),
            'box_dets': box_dets[keep] if keep_num != 0 else np.zeros(0),
            'no_dets': keep_num == 0,
            'det_num': keep_num,
        }
        return data_info

    def __len__(self):
        return len(self.all_sample_token)
