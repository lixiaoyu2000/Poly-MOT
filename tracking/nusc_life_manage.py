"""
Count-based Trajectory Lifecycle Management Module.
Trajectory state(tentative/active/death) transition and tracking score punish
"""

class LifeManagement:
    def __init__(self, timestamp: int, config: dict, class_label: int):
        self.cfg = config['life_cycle']
        self.time_since_update, self.hit = 0, 1
        self.curr_time = self.init_time = timestamp
        self.min_hit, self.max_age = self.cfg['min_hit'][class_label], self.cfg['max_age'][class_label]
        self.state = 'active' if self.min_hit <= 1 or timestamp <= self.min_hit else 'tentative'

    def predict(self, timestamp: int) -> None:
        """
        predict tracklet lifecycle
        :param timestamp: int, current timestamp, frame id
        """
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
            self.time_since_update = 0

        if self.state == 'tentative':
            if (self.hit >= self.min_hit) or (timestamp <= self.min_hit):
                self.state = 'active'
            elif self.time_since_update > 0:
                self.state = 'dead'
            else: pass
        elif self.state == 'active':
            if self.time_since_update >= self.max_age:
                self.state = 'dead'
            else: pass
        else: raise Exception("dead trajectory cannot be updated")
        
    def __repr__(self) -> str:
        repr_str = 'init_timestamp: {}, time_since_update: {}, hit: {}, state: {}'
        return repr_str.format(self.init_time, self.time_since_update, self.hit, self.state)




