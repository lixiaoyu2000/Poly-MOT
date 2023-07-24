"""
kalman filter for trajectory state(motion state) estimation
Two implemented KF version (LKF, EKF)
Three core functions for each model: state init, state predict and state update
Linear Kalman Filter for CA, CV Model, Extend Kalman Filter for CTRA, CTRV, Bicycle Model
Ref: https://en.wikipedia.org/wiki/Kalman_filter
"""
import pdb
import numpy as np
from .nusc_object import FrameObject
from .motion_model import CA, CTRA
from pre_processing import arraydet2box, concat_box_attr

class KalmanFilter:
    """kalman filter interface
    """
    def __init__(self, timestamp: int, config: dict, track_id: int, det_infos: dict) -> None:
        # init basic infos, no control input
        self.seq_id = det_infos['seq_id']
        self.initstamp = self.timestamp = timestamp
        self.tracking_id, self.class_label = track_id, det_infos['np_array'][-1]
        self.model = config['motion_model']['model'][self.class_label]
        self.dt, self.has_velo = config['basic']['LiDAR_interval'], config['basic']['has_velo']
        # init FrameObject for each frame
        self.state, self.frame_objects = None, {}
    
    def initialize(self, det: dict) -> None:
        """initialize the filter parameters
        Args:
            det (dict): detection infos under different data format.
            {
                'nusc_box': NuscBox,
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
    
    def getMeasureInfo(self, det: dict = None) -> np.array:
        """convert det box to the measurement info for updating filter
        [x, y, z, w, h, l, (vx, vy, optional), ry]
        Args:
            det (dict, optional): same as self.init. Defaults to None.

        Returns:
            np.array: measurement for updating filter
        """
        if det is None: raise("detection cannot be None")
        
        mea_attr = ('center', 'wlh', 'velocity', 'yaw') if self.has_velo else ('translation', 'size', 'yaw')
        list_det = concat_box_attr(det['nusc_box'], *mea_attr)
        if self.has_velo: list_det.pop(8)
        
        # only for debug, delete on the release version
        # ensure the measure yaw goes around [0, 0, 1]
        assert list_det[-1] == det['nusc_box'].orientation.radians and det['nusc_box'].orientation.axis[-1] >= 0
        assert len(list_det) == 9 if self.has_velo else 7
        
        return np.mat(list_det).T
        
    
    def addFrameObject(self, timestamp: int, tra_info: dict, mode: str = None) -> None:
        """add predict/update tracklet state to the frameobjects, data 
        format is also implemented in this function.
        frame_objects: {
            frame_id: FrameObject
        }
        Args:
            timestamp (int): current frame id
            tra_info (dict): Trajectory state estimated by Kalman filter, 
            {
                'exter_state': np.array, for output file. 
                               [x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), tra_score, class_label]
                'inner_state': np.array, for state estimation. 
                               varies by motion model
            }
            mode (str, optional): stage of adding objects, 'update', 'predict'. Defaults to None.
        """
        # corner case, no tra info
        if mode is None: return
        
        # data format conversion
        inner_info, exter_info = tra_info['inner_state'], tra_info['exter_state']
        extra_info = np.array([self.tracking_id, self.seq_id, timestamp])
        box_info, bm_info = arraydet2box(exter_info, np.array([self.tracking_id]))

        # update each frame infos 
        if mode == 'update':
            frame_object = self.frame_objects[timestamp]
            frame_object.update_bms, frame_object.update_box = bm_info[0], box_info[0]
            frame_object.update_state, frame_object.update_infos = inner_info, np.append(exter_info, extra_info)
        elif mode == 'predict':
            frame_object = FrameObject()
            frame_object.predict_bms, frame_object.predict_box = bm_info[0], box_info[0]
            frame_object.predict_state, frame_object.predict_infos = inner_info, np.append(exter_info, extra_info)
            self.frame_objects[timestamp] = frame_object
        else: raise Exception('mode must be update or predict')
    
    def getOutputInfo(self, state: np.mat) -> np.array:
        """convert state vector in the filter to the output format
        Note that, tra score will be process later
        Args:
            state (np.mat): [state dim, 1], predict or update state estimated by the filter

        Returns:
            np.array: [14(fix), 1], predict or update state under output file format
            output format: [x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), tra_score, class_label]
        """
        
        # return state vector except tra score and tra class
        inner_state = self.model.getOutputInfo(state)
        return np.append(inner_state, np.array([-1, self.class_label]))
    
    def __getitem__(self, item) -> FrameObject:
        return self.frame_objects[item]

    def __len__(self) -> int:
        return len(self.frame_objects)



class LinearKalmanFilter(KalmanFilter):
    """Linear Kalman Filter for linear motion model, such as CV and CA
    """
    def __init__(self, timestamp: int, config: dict, track_id: int, det_infos: dict) -> None:
        # init basic infos
        super(LinearKalmanFilter, self).__init__(timestamp, config, track_id, det_infos)
        # set motion model, default Constant Acceleration(CA) for LKF
        self.model = globals()[self.model](self.has_velo, self.dt) if self.model in ['CV', 'CA'] \
                     else globals()['CA'](self.has_velo, self.dt)
        # Transition and Observation Matrices are fixed in the LKF
        self.initialize(det_infos)
        
    def initialize(self, det_infos: dict) -> None:
        # state transition
        self.F = self.model.getTransitionF()
        self.Q = self.model.getProcessNoiseQ()
        self.SD = self.model.getStateDim()
        self.P = self.model.getInitCovP(self.class_label)
        
        # state to measurement transition
        self.R = self.model.getMeaNoiseR()
        self.H = self.model.getMeaStateH()

        # get different data format tracklet's state
        self.state = self.model.getInitState(det_infos)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': det_infos['np_array']
        }
        self.addFrameObject(self.timestamp, tra_infos, 'predict')
        self.addFrameObject(self.timestamp, tra_infos, 'update')
    
    def predict(self, timestamp: int) -> None:
        # predict state and errorcov
        self.state = self.F * self.state
        self.P = self.F * self.P * self.F.T + self.Q
        
        # convert the state in filter to the output format
        self.model.warpStateYawToPi(self.state)
        output_info = self.getOutputInfo(self.state)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': output_info
        }
        self.addFrameObject(timestamp, tra_infos, 'predict')
        
    def update(self, timestamp: int, det: dict = None) -> None:
        # corner case, no det for updating
        if det is None: return
        
        # update state and errorcov
        meas_info = self.getMeasureInfo(det)
        _res = meas_info - self.H * self.state
        self.model.warpResYawToPi(_res)
        _S = self.H * self.P * self.H.T + self.R
        _KF_GAIN = self.P * self.H.T * _S.I
        
        self.state += _KF_GAIN * _res
        self.P = (np.mat(np.identity(self.SD)) - _KF_GAIN * self.H) * self.P
        
        # output updated state to the result file
        self.model.warpStateYawToPi(self.state)
        output_info = self.getOutputInfo(self.state)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': output_info
        }
        self.addFrameObject(timestamp, tra_infos, 'update')
        
        
class ExtendKalmanFilter(KalmanFilter):
    def __init__(self, timestamp: int, config: dict, track_id: int, det_infos: dict) -> None:
        super().__init__(timestamp, config, track_id, det_infos)
        # set motion model, default Constant Acceleration and Turn Rate(CTRA) for EKF
        self.model = globals()[self.model](self.has_velo, self.dt) if self.model in ['CTRA', 'BICYCLE'] \
                     else globals()['CTRA'](self.has_velo, self.dt)
        # Transition and Observation Matrices are changing in the EKF
        self.initialize(det_infos)
    
    def initialize(self, det_infos: dict) -> None:
        # init errorcov categoty-specific
        self.SD = self.model.getStateDim()
        self.P = self.model.getInitCovP(self.class_label)
        
        # set noise matrix(fixed)
        self.Q = self.model.getProcessNoiseQ()
        self.R = self.model.getMeaNoiseR()

        # get different data format tracklet's state
        self.state = self.model.getInitState(det_infos)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': det_infos['np_array']
        }
        self.addFrameObject(self.timestamp, tra_infos, 'predict')
        self.addFrameObject(self.timestamp, tra_infos, 'update')
        
    def predict(self, timestamp: int) -> None:
        # get jacobian matrix F using the final estimated state of the previous frame
        self.F = self.model.getTransitionF(self.state)
        
        # state and errorcov transition
        self.state = self.model.stateTransition(self.state)
        self.P = self.F * self.P * self.F.T + self.Q
        
        # convert the state in filter to the output format
        self.model.warpStateYawToPi(self.state)
        output_info = self.getOutputInfo(self.state)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': output_info
        }
        self.addFrameObject(timestamp, tra_infos, 'predict')
    
    def update(self, timestamp: int, det: dict = None) -> None:
        # corner case, no det for updating
        if det is None: return
        
        # get measure infos for updating, and project state into meausre space
        meas_info = self.getMeasureInfo(det)
        state_info = self.model.StateToMeasure(self.state)
        
        # get state residual, and warp angle diff inplace
        _res = meas_info - state_info
        self.model.warpResYawToPi(_res)
        
        # get jacobian matrix H using the predict state
        self.H = self.model.getMeaStateH(self.state)
        
        # obtain KF gain and update state and errorcov
        _S = self.H * self.P * self.H.T + self.R
        _KF_GAIN = self.P * self.H.T * _S.I
        _I_KH = np.mat(np.identity(self.SD)) - _KF_GAIN * self.H
        
        self.state += _KF_GAIN * _res
        self.P = _I_KH * self.P * _I_KH.T + _KF_GAIN * self.R * _KF_GAIN.T
        
        # output updated state to the result file
        self.model.warpStateYawToPi(self.state)
        output_info = self.getOutputInfo(self.state)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': output_info
        }
        self.addFrameObject(timestamp, tra_infos, 'update')
        
        
        
        
        
        
        
            
        
        
        
        
        
        
        
    
    
        
