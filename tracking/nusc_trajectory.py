"""
object's trajectory. A trajectory is a collection of information for each frame(nusc_object.py)
Two core functions: state predict and state update.
The 'state' here is generalized, including attribute info, motion info, geometric info, score info, etc.

In general, Poly-MOT combines count-based and confidence-based strategy to manage trajectory lifecycle. 
Specifically, we use the count-based strategy to initialize and unregister trajectories, while using the 
score-based strategy to penalize mismatched trajectories
"""
import pdb
import numpy as np
from .nusc_life_manage import LifeManagement
from .nusc_score_manage import ScoreManagement
from motion_module.nusc_object import FrameObject
from motion_module.kalman_filter import LinearKalmanFilter, ExtendKalmanFilter


class Trajectory:
    def __init__(self, timestamp: int, config: dict, track_id: int, det_infos: dict):
        # init basic infos
        self.timestamp = timestamp
        self.cfg, self.tracking_id, self.class_label = config, track_id, det_infos['np_array'][-1]
        # manage tracklet's attribute
        self.life_management = LifeManagement(timestamp, config, self.class_label)
        # manage tracklet's score, predict/update/punish trackelet
        self.score_management = ScoreManagement(timestamp, config, self.class_label, det_infos)
        # manage for tracklet's motion/geometric infos
        KF_type = self.cfg['motion_model']['filter'][self.class_label]
        assert KF_type in ['LinearKalmanFilter', 'ExtendKalmanFilter'], "must use specific kalman filter"
        self.motion_management = globals()[KF_type](timestamp, config, track_id, det_infos)
    
    def state_predict(self, timestamp: int) -> None:
        """
        predict trajectory's state
        :param timestamp: current frame id
        """
        self.timestamp = timestamp
        self.life_management.predict(timestamp)
        self.motion_management.predict(timestamp)
        self.score_management.predict(timestamp, self.motion_management[timestamp])
    
    def state_update(self, timestamp: int, det: dict = None) -> None:
        """
        update trajectory's state
        :param timestamp: current frame id
        :param det: dict, detection infos under different data format
            {
                'nusc_box': NuscBox,
                'np_array': np.array,
                'has_velo': bool, whether the detetor has velocity info
            }
        """
        self.timestamp = timestamp
        self.life_management.update(timestamp, det)
        self.motion_management.update(timestamp, det)
        self.score_management.update(timestamp, self.motion_management[timestamp], det)
        
    def __getitem__(self, item) -> FrameObject:
        return self.motion_management[item]
    
    def __len__(self) -> int:
        return len(self.motion_management)
    
    def __repr__(self) -> str:
        repr_str = 'tracklet status: {}, id: {}, score: {}, state: {}'
        return repr_str.format(self.life_management, self.tracking_id, 
                               self.score_management[self.timestamp],
                               self.motion_management[self.timestamp])