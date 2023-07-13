"""
motion model of trajectory, notice that objects of different categories often exhibit various motion patterns.
Five implemented motion models, including
- Two linear model: Constant Acceleration(CA), Constant Velocity(CV)
- Three non-linear model: Constant Turn Rate and Acceleration(CTRA), Constant Turn Rate and Velocity(CTRV), Bicycle Model
"""
import numpy as np
from data.script.NUSC_CONSTANT import *

class CA: 
    """Constant Acceleration Motion Model
    Basic info:
        State vector: [x, y, z, w, l, h, vx, vy, vz, ax, ay, az, ry]
        Measure vector: [x, y, z, w, l, h, (vx, vy, optional), ry]
    """
    def __init__(self, has_velo: bool, dt: float) -> None:
        self.has_velo, self.dt = has_velo, dt
    
    def getInitState(self, det_infos: dict) -> np.mat:
        """from detection init tracklet

        Args:
            det (dict): detection infos under different data format.
            {
                'nusc_box': NuscBox,
                'np_array': np.array,
                'has_velo': bool, whether the detetor has velocity info
            }

        Returns:
            np.mat: state vector, [13, 1]
        """
        pass
    
    def getInitCovP(self, cls_label: int) -> np.mat:
        pass
    
    def getProcessNoiseQ(self) -> np.mat:
        pass
    
    def getTransitionF(self) -> np.mat:
        pass
    
    def getMeaNoiseR(self) -> np.mat:
        pass
    
    def getMeaStateH(self) -> np.mat:
        pass
    
    def getOutputInfo(self, state: np.mat) -> np.array:
        pass
    
    def getStateDim(self) -> int:
        return 13
        
        
        
        
    