"""
kalman filter for trajectory state(motion state) estimation
Two implemented KF version (LKF, EKF)
Three core functions for each model: state init, state predict and state update
Linear Kalman Filter for CA, CV Model, Extend Kalman Filter for CTRA, CTRV, Bicycle Model
Ref: https://en.wikipedia.org/wiki/Kalman_filter
"""
from abc import ABC
from abc import abstractmethod
from 

class KalmanFilter(ABC):
    @abstractmethod
    def init(self, det: dict) -> None:
        """initialize the filter parameters
        Args:
            det (dict): detection infos under different data format.
            {
                'nusc_box': nusc_box,
                'np_array': np.array,
                'has_velo': bool, whether the detetor has velocity info
            }
        """
    
    @abstractmethod
    def predict(self, timestamp: int) -> None:
        """predict tracklet at each frame
        Args:
            timestamp (int): current frame id
        """

        pass
    
    @abstractmethod
    def update(self, timestamp: int, det: dict = None) -> None:
        """update tracklet motion and geometric state
        Args:
            timestamp (int): current frame id
            det (dict, optional): same as self.init. Defaults to None.
        """
        pass



class LinearKalmanFilter(KalmanFilter):
    def __init__(self, timestamp: int, config: dict, track_id: int, det_infos: dict) -> None:
        # init basic infos
        self.tracking_id, self.class_label = track_id, det_infos['nusc_box'][-1]
        self.dt = config['basic']['LiDAR_interval']
        self.initstamp = self.timestamp = timestamp
        
        # set motion model, default Constant Acceleration(CA)
        model = config['motion_model']['model'][self.class_label]
        self.motion_model = model if model in ['CV', 'CA'] else 'CA'
        
        # init Transition and Observation Matrix 
        
