"""
kalman filter for trajectory state(motion state) estimation
Two implemented KF version (LKF, EKF)
Three core functions for each model: state init, state predict and state update
Linear Kalman Filter for CA, CV Model, Extend Kalman Filter for CTRA, CTRV, Bicycle Model
Ref: https://en.wikipedia.org/wiki/Kalman_filter
"""
import numpy as np
from nusc_object import FrameObject
from geometry.nusc_box import nusc_box
from pre_processing import arraydet2box
from motion_model import CA, CTRA, BICYCLE

class KalmanFilter:
    def __init__(self, timestamp: int, config: dict, track_id: int, det_infos: dict) -> None:
        # init basic infos, no control input
        self.seq_id = det_infos['seq_id']
        self.initstamp = self.timestamp = timestamp
        self.model = config['motion_model']['model'][self.class_label]
        self.tracking_id, self.class_label = track_id, det_infos['nusc_box'][-1]
        self.dt, self.has_velo = config['basic']['LiDAR_interval'], config['basic']['has_velo']
        # init FrameObject for each frame
        self.frame_objects = {}
    
    def initialize(self, timestamp: int, det: dict) -> None:
        """initialize the filter parameters
        Args:
            timestamp (int): current frame id
            det (dict): detection infos under different data format.
            {
                'nusc_box': nusc_box,
                'np_array': np.array,
                'has_velo': bool, whether the detetor has velocity info
            }
        """
        pass
    
    def predict(self, timestamp: int) -> None:
        """predict tracklet at each frame
        Args:
            timestamp (int): current frame id
        """

        pass
    
    def update(self, timestamp: int, det: dict = None) -> None:
        """update tracklet motion and geometric state
        Args:
            timestamp (int): current frame id
            det (dict, optional): same as self.init. Defaults to None.
        """
        pass
    
    def addFrameObject(self, timestamp: int, tra_info: np.array, mode: str = None) -> None:
        """add predict/update tracklet state to the frameobjects, data 
        format is also implemented in this function.
        frame_objects: {
            frame_id: FrameObject
        }
        Args:
            timestamp (int): current frame id
            tra_info (np.array): Trajectory state estimated by Kalman filter, 
                                 which are organized as following:
            [x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), det_score, class_label]
            mode (str, optional): stage of adding objects, 'update', 'predict'. Defaults to None.
        """
        if mode is None:
            return
        
        box_info, bm_info = arraydet2box(tra_info)
        frame_object = FrameObject()
        if mode == 'update':
            frame_object.update_infos = np.array(tra_info, [self.tracking_id, self.seq_id, timestamp])
            frame_object.update_box = box_info
            frame_object.update_bms = bm_info
        elif mode == 'predict':
            frame_object.predict_infos = np.array(tra_info, [self.tracking_id, self.seq_id, timestamp])
            frame_object.predict_box = box_info
            frame_object.predict_bms = bm_info
        else: raise Exception('mode must be update or predict')
            
    def __getitem__(self, item) -> FrameObject:
        return self.frame_objects[item]

    def __len__(self) -> int:
        return len(self.frame_objects)



class LinearKalmanFilter(KalmanFilter):
    def __init__(self, timestamp: int, config: dict, track_id: int, det_infos: dict) -> None:
        # init basic infos
        super(LinearKalmanFilter, self).__init__(timestamp, config, track_id, det_infos)
        # set motion model, default Constant Acceleration(CA) for LKF
        self.model = globals()[self.model](self.has_velo) if self.model in ['CV', 'CA'] else globals()['CA'](self.has_velo)
        # Transition and Observation Matrices are fixed in the LKF
        self.initialize(det_infos)
        
    def initialize(self, det_infos: dict) -> None:
        # state transition
        self.P = self.model.getInitCovP()
        self.F = self.model.getTransitionF()
        self.Q = self.model.getProcessNoiseQ()
        
        # state to measurement transition
        self.R = self.model.getMeaNoiseR()
        self.H = self.model.getMeaStateH()

        # get different data format tracklet's state
        self.addFrameObject(self.timestamp, det_infos)
        
        
    
    
        
