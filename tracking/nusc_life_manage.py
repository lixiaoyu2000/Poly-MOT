"""
Count/Confidence-based Trajectory Lifecycle Management Module.
Trajectory state(tentative/active/death) transition and tracking score punish
TODO: to support Confidence-based method and check the correctness of Confidence-based

Thanks: Confidence-based management codes are inspired by CBMOT.
Code URL: CBMOT(https://github.com/cogsys-tuebingen/CBMOT)
"""
import numpy as np


class LifeManagement:
    def __init__(self, timestamp: int, config: dict, class_label: int):
        self.cfg = config['life_cycle']
        self.time_since_update = 0
        self.curr_time = self.init_time = timestamp
        self.hit, self.miss_num, self.state_switch = 1, 0, False
        self.min_hit, self.max_age = self.cfg['min_hit'][class_label], self.cfg['max_age'][class_label]
        self.state = 'active' if self.min_hit <= 1 or timestamp <= self.min_hit else 'tentative'

    def predict(self, timestamp: int) -> None:
        """
        predict tracklet lifecycle
        :param timestamp: int, current timestamp, frame id
        """
        self.miss_num += 1
        self.curr_time = timestamp
        self.time_since_update += 1

    def update(self, timestamp: int, det = None) -> None:
        """
        update tracklet lifecycle status, switch tracklet's state (tentative/dead/active)
        :param timestamp: int, current timestamp, frame id
        :param det: matched detection at current frame
        """
        if det is not None:
            self.hit += 1
            self.miss_num, self.time_since_update = 0, 0

        if self.state == 'tentative':
            if self.time_since_update > 0:
                self.state_switch, self.state = True, 'dead'
            elif (self.hit >= self.min_hit) or (timestamp <= self.min_hit):
                self.state_switch, self.state = True, 'active'
            else: self.state_switch = False
        elif self.state == 'active':
            if self.time_since_update >= self.max_age:
                self.state_switch, self.state = True, 'dead'
            else: self.state_switch = False
        else: raise Exception("dead trajectory cannot be updated")




