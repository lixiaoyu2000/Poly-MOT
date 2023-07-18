"""
Information for each frame of the trajectory.
"""

class FrameObject:
    def __init__(self) -> None:
        # output infos
        self.update_bms, self.update_infos, self.update_box =  None, None, None
        self.predict_bms, self.predict_infos, self.predict_box = None, None, None
        
        # infos for state transition
        self.update_state, self.predict_state = None, None
        
    def __repr__(self) -> str:
        repr_str = 'Predict state: {}, Update state: {}'
        return repr_str.format(self.predict_infos, self.update_infos)