"""
motion model of trajectory, notice that objects of different categories often exhibit various motion patterns.
Five implemented motion models, including
- Two linear model: Constant Acceleration(CA), Constant Velocity(CV)
- Three non-linear model: Constant Turn Rate and Acceleration(CTRA), Constant Turn Rate and Velocity(CTRV), Bicycle Model
"""
import numpy as np
from pyquaternion import Quaternion
from utils.math import warp_to_pi
from data.script.NUSC_CONSTANT import *

class CA: 
    """Constant Acceleration Motion Model
    Basic info:
        State vector: [x, y, z, w, l, h, vx, vy, vz, ax, ay, az, ry]
        Measure vector: [x, y, z, w, l, h, (vx, vy, optional), ry]
    """
    def __init__(self, has_velo: bool, dt: float) -> None:
        self.has_velo, self.dt, self.SD = has_velo, dt, 13
        self.MD = 9 if self.has_velo else 7
    
    def getInitState(self, det_infos: dict) -> np.mat:
        """from detection init tracklet
        Acceleration and velocity on the z-axis are both set to 0

        Args:
            det (dict): detection infos under different data format.
            {
                'nusc_box': NuscBox,
                'np_array': np.array,
                'has_velo': bool, whether the detetor has velocity info
            }

        Returns:
            np.mat: [state dim, 1], state vector
        """
        init_state = np.zeros(shape=self.SD)
        det, det_box = det_infos['np_array'], det_infos['nusc_box']
        
        # set x, y, z, w, l, h, (vx, vy, if velo is valid)
        init_state[:6] = det[:6]
        if self.has_velo: init_state[6:8] = det[6:8]
        
        # set yaw
        init_state[-1] = det_box.yaw
        
        # only for debug
        q = Quaternion(det[8:12].tolist())
        q = -q if q.axis[-1] < 0 else q
        assert q.radians == det_box.yaw
        
        return np.mat(init_state).T
    
    def getInitCovP(self, cls_label: int) -> np.mat:
        """init errorcov. Generally, the CA model can converge quickly, 
        so not particularly sensitive to initialization

        Args:
            cls_label (int): set init errorcov category-specific. 

        Returns:
            np.mat: [state dim, state dim], Initialized covariance matrix
        """
        return np.mat(np.eye(self.SD)) * 0.01
    
    def getProcessNoiseQ(self) -> np.mat:
        """get process noise matrix. The value set is somewhat arbitrary

        Returns:
            np.mat: process noise matrix(fix)
        """
        return np.mat(np.eye(self.SD)) * 100
    
    def getTransitionF(self) -> np.mat:
        """get state transition matrix.
        obtain matrix in the motion_module/script/Linear_kinect_jacobian.ipynb
        Returns:
            np.mat: [state dim, state dim], state transition matrix
        """
        dt = self.dt
        F = np.mat([[1, 0, 0, 0, 0, 0, dt,  0, 0, 0.5*dt**2,         0,  0, 0],
                    [0, 1, 0, 0, 0, 0,  0, dt, 0,         0, 0.5*dt**2,  0, 0],
                    [0, 0, 1, 0, 0, 0,  0,  0, 0,         0,         0,  0, 0],
                    [0, 0, 0, 1, 0, 0,  0,  0, 0,         0,         0,  0, 0],
                    [0, 0, 0, 0, 1, 0,  0,  0, 0,         0,         0,  0, 0],
                    [0, 0, 0, 0, 0, 1,  0,  0, 0,         0,         0,  0, 0],
                    [0, 0, 0, 0, 0, 0,  1,  0, 0,        dt,         0,  0, 0],
                    [0, 0, 0, 0, 0, 0,  0,  1, 0,         0,        dt,  0, 0],
                    [0, 0, 0, 0, 0, 0,  0,  0, 0,         0,         0,  0, 0],
                    [0, 0, 0, 0, 0, 0,  0,  0, 0,         1,         0,  0, 0],
                    [0, 0, 0, 0, 0, 0,  0,  0, 0,         0,         1,  0, 0],
                    [0, 0, 0, 0, 0, 0,  0,  0, 0,         0,         0,  0, 0],
                    [0, 0, 0, 0, 0, 0,  0,  0, 0,         0,         0,  0, 1]])
        return F
        
    def getMeaNoiseR(self) -> np.mat:
        """get measurement noise matrix. The value set is also somewhat arbitrary
        Returns:
            np.mat: measure noise matrix(fix)
        """
        return np.mat(np.eye(self.MD)) * 0.001
    
    def getMeaStateH(self) -> np.mat:
        """get state to measure transition matrix.
        obtain matrix in the motion_module/script/Linear_kinect_jacobian.ipynb
        Returns:
            np.mat: [measure dim, state dim], state to measure transition matrix
        """
        if self.has_velo:
            H = np.mat([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        else:
            H = np.mat([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        return H
    
    @staticmethod
    def getOutputInfo(state: np.mat) -> np.array:
        """convert state vector in the filter to the output format
        Note that, tra score will be process later
        Args:
            state (np.mat): [state dim, 1], predict or update state estimated by the filter

        Returns:
            np.array: [12(fix), 1], predict or update state under output file format
            output format: [x, y, z, w, l, h, vx, vy, ry(orientation, 1x4)]
        """
        rotation = Quaternion(axis=(0, 0, 1), radians=state[-1, 0]).q
        list_state = state.T.tolist()[0][:8] + rotation.tolist()
        return np.array(list_state)
    
    @staticmethod
    def warpResYawToPi(res: np.mat) -> np.mat:
        """warp res yaw to [-pi, pi) in place

        Args:
            res (np.mat): [measure dim, 1]
            res infos -> [x, y, z, w, l, h, (vx, vy, optional), ry]

        Returns:
            np.mat: [measure dim, 1], residual warped to [-pi, pi)
        """
        res[-1, 0] = warp_to_pi(res[-1, 0])
        return res
    
    @staticmethod
    def warpStateYawToPi(state: np.mat) -> np.mat:
        """warp state yaw to [-pi, pi) in place

        Args:
            state (np.mat): [state dim, 1]
            State vector: [x, y, z, w, l, h, vx, vy, vz, ax, ay, az, ry]

        Returns:
            np.mat: [state dim, 1], state after warping
        """
        state[-1, 0] = warp_to_pi(state[-1, 0])
        return state
    
    def getStateDim(self) -> int:
        return self.SD
    
    def getMeasureDim(self) -> int:
        return self.MD
    

        
class CTRA: 
    """Constant Acceleration and Turn Rate Motion Model
    Basic info:
        State vector: [x, y, z, w, l, h, v, a, ry, ry_rate]
        Measure vector: [x, y, z, w, l, h, (vx, vy, optional), ry]
    """
    def __init__(self, has_velo: bool, dt: float) -> None:
        self.has_velo, self.dt, self.SD = has_velo, dt, 10
        self.MD = 9 if self.has_velo else 7
        
    def getInitState(self, det_infos: dict) -> np.mat:
        """from detection init tracklet
        Acceleration and yaw(turn) rate are both set to 0. when velociy
        on X/Y-Axis are available, the combined velocity is also set to 0

        Args:
            det (dict): detection infos under different data format.
            {
                'nusc_box': NuscBox,
                'np_array': np.array,
                'has_velo': bool, whether the detetor has velocity info
            }

        Returns:
            np.mat: [state dim, 1], state vector
        """
        init_state = np.zeros(shape=self.SD)
        det, det_box = det_infos['np_array'], det_infos['nusc_box']
        
        # set x, y, z, w, l, h, (v, if velo is valid)
        init_state[:6] = det[:6]
        if self.has_velo: init_state[6] = np.hypot(det[6], det[7])
        
        # set yaw
        init_state[-2] = det_box.yaw
        
        # only for debug
        q = Quaternion(det[8:12].tolist())
        q = -q if q.axis[-1] < 0 else q
        assert q.radians == det_box.yaw
        
        return np.mat(init_state).T
    
    def getInitCovP(self, cls_label: int) -> np.mat:
        """init errorcov. In general, when the speed is observable, 
        the CTRA model can converge quickly, but when the speed is not measurable, 
        we need to carefully set the initial covariance to help the model converge

        Args:
            cls_label (int): set init errorcov category-specific. 

        Returns:
            np.mat: [state dim, state dim], Initialized covariance matrix
        """
        cls_name = CLASS_STR_TO_SEG_CLASS[cls_label]
        vector_p = CTRA_INIT_EFKP[cls_name] if cls_name in CTRA_INIT_EFKP else CTRA_INIT_EFKP['car']
        
        return np.mat(np.diag(vector_p)) if not self.has_velo else np.mat(np.diag('car')) 
    
    def getProcessNoiseQ(self) -> np.mat:
        """get process noise matrix. The value set is somewhat arbitrary

        Returns:
            np.mat: process noise matrix(fix)
        """
        return np.mat(np.eye(self.SD)) * 1
    
    def getMeaNoiseR(self) -> np.mat:
        """get measurement noise matrix. The value set is also somewhat arbitrary
        Returns:
            np.mat: measure noise matrix(fix)
        """
        return np.mat(np.eye(self.MD)) * 1
    
    def stateTransition(self, state: np.mat) -> np.mat:
        """state transition, 
        obtain analytical solutions in the motion_module/script/CTRA_kinect_jacobian.ipynb
        Args:
            state (np.mat): [state dim, 1] the estimated state of the previous frame

        Returns:
            np.mat: [state dim, 1] the predict state of the current frame
        """
        
        assert state.shape == (10, 1), "state vector number in CTRA must equal to 10"
        
        dt = self.dt
        x, y, z, w, l, h, v, a, theta, omega = state.T.tolist()[0]
        yaw_sin, yaw_cos = np.sin(theta), np.cos(theta)
        next_v, next_ry = v + a * dt, theta + omega * dt
        
        # corner case(tiny yaw rate), prevent divide-by-zero overflow
        if abs(omega) < 0.001:
            displacement = v * dt + a * dt ** 2 / 2
            predict_state = [x + displacement * yaw_cos,
                             y + displacement * yaw_sin,
                             z, w, l, h, 
                             next_v, a, 
                             next_ry, omega]
        else:
            ry_rate_inv_square = 1.0 / (omega * omega)
            next_yaw_sin, next_yaw_cos = np.sin(next_ry), np.cos(next_ry)
            predict_state = [x + ry_rate_inv_square * (next_v * omega * next_yaw_sin + a * next_yaw_cos - v * omega * yaw_sin - a * yaw_cos),
                             y + ry_rate_inv_square * (-next_v * omega * next_yaw_cos + a * next_yaw_sin + v * omega * yaw_cos - a * yaw_sin),
                             z, w, l, h,
                             next_v, a, 
                             next_ry, omega]
        
        return np.mat(predict_state).T  
        
        
    
    def StateToMeasure(self, state: np.mat) -> np.mat:
        """get state vector in the measure space
        state vector -> [x, y, z, w, l, h, v, a, ry, ry_rate]
        measure space -> [x, y, z, w, l, h, (vx, vy, optional), ry]

        Args:
            state (np.mat): [state dim, 1] the predict state of the current frame

        Returns:
            np.mat: [measure dim, 1] state vector projected in the measure space
        """
        assert state.shape == (10, 1), "state vector number in CTRA must equal to 10"
        
        x, y, z, w, l, h, v, _, theta, _ = state.T.tolist()[0]
        if self.has_velo:
            state_info = [x, y, z,
                          w, l, h,
                          v * np.cos(theta),
                          v * np.sin(theta),
                          theta]
        else:
            state_info = [x, y, z,
                          w, l, h,
                          theta]
        
        return np.mat(state_info).T
         
        
    
    def getTransitionF(self, state: np.mat) -> np.mat:
        """get state transition matrix.
        obtain matrix in the motion_module/script/CTRA_kinect_jacobian.ipynb
        
        Args:
            state (np.mat): [state dim, 1] the estimated state of the previous frame
        
        Returns:
            np.mat: [state dim, state dim], d(stateTransition) / d(state) at previous_state
        """
        dt = self.dt
        x, y, z, w, l, h, v, a, theta, omega = state.T.tolist()[0]
        
        # corner case, tiny turn rate
        if abs(omega) < 0.001:
            F = np.mat([[1, 0, 0, 0, 0, 0, dt*cos(yaw),  dt**2*cos(yaw)/2, -(a*dt**2/2 + dt*v)*sin(yaw), 0],
                        [0, 1, 0, 0, 0, 0, dt*sin(yaw),  dt**2*sin(yaw)/2,  (a*dt**2/2 + dt*v)*cos(yaw), 0],
                        [0, 0, 1, 0, 0, 0,           0,                 0,                            0, 0],
                        [0, 0, 0, 1, 0, 0,           0,                 0,                            0, 0],
                        [0, 0, 0, 0, 1, 0,           0,                 0,                            0, 0],
                        [0, 0, 0, 0, 0, 1,          0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, dt,0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1,0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0,1,dt],
                        [0, 0, 0, 0, 0, 0, 0, 0,0,1]])
                                
        
        
        return F
    
    def getMeaStateH(self, state: np.mat) -> np.mat:
        """get state to measure transition matrix.
        obtain matrix in the motion_module/script/CTRA_kinect_jacobian.ipynb
        
        Args:
            state (np.mat): [state dim, 1] the predict state of the current frame
        
        Returns:
            np.mat: [measure dim, state dim], d(StateToMeasure) / d(state) at predict_state
        """
        pass
    
    
        
    