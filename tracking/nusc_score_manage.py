"""
assign tracklet confidence score 
predict, update, punish tracklet score under category-specific way.

TODO: to support Confidence-based method to init and kill tracklets
Code URL: CBMOT(https://github.com/cogsys-tuebingen/CBMOT)
"""
import pdb
from motion_module.nusc_object import FrameObject

class ScoreObject:
    def __init__(self) -> None:
        self.raw_score = self.final_score = -1
        self.predict_score = self.update_score = -1
        
    def __repr__(self) -> str:
        repr_str = 'Raw score: {}, Predict score: {}, Update score: {}, Final score: {}.'
        return repr_str.format(self.raw_score, self.predict_score, self.update_score, self.final_score)

class ScoreManagement:
    def __init__(self, timestamp: int, cfg: dict, cls_label: int, det_infos: dict) -> None:
        self.initstamp = timestamp
        self.frame_objects, self.dr = {}, cfg['life_cycle']['decay_rate'][cls_label]
        self.initialize(det_infos)
    
    def initialize(self, det_infos: dict) -> None:
        """init tracklet confidence score, no inplace ops needed

        Args:
            det_infos (dict): dict, detection infos under different data format
            {
                'nusc_box': NuscBox,
                'np_array': np.array,
                'has_velo': bool, whether the detetor has velocity info
            }
        """
        score_obj = ScoreObject()
        score_obj.raw_score = score_obj.final_score = det_infos['nusc_box'].score
        score_obj.predict_score = score_obj.update_score = det_infos['nusc_box'].score
        self.frame_objects[self.initstamp] = score_obj
    
    def predict(self, timestamp: int, pred_obj: FrameObject = None) -> None:
        """decay tracklet confidence score, change score in the predict infos inplace.

        Args:
            timestamp (int): current frame id
            pred_obj (FrameObject): nusc box/infos predicted by the filter
        """
        score_obj = ScoreObject()
        prev_score = self.frame_objects[timestamp - 1].final_score
        score_obj.raw_score, score_obj.predict_score = prev_score, prev_score * self.dr
        self.frame_objects[timestamp] = score_obj
        
        # assign tracklet score inplace
        pred_obj.predict_box.score = pred_obj.predict_infos[-5] = score_obj.predict_score
        
        
    def update(self, timestamp: int, update_obj: FrameObject, raw_det: dict = None) -> None:
        """Update trajectory confidence scores inplace directly using matched det

        Args:
            timestamp (int): current frame id
            update_obj (FrameObject): nusc box/infos updated by the filter
            raw_det (dict, optional): same as data format in the init function. Defaults to None.
        """
        score_obj = self.frame_objects[timestamp]
        if raw_det is None:
            score_obj.final_score = score_obj.predict_score
            return

        # assign score objects and output scores
        score_obj.update_score = score_obj.final_score = raw_det['nusc_box'].score
        update_obj.update_box.score = update_obj.update_infos[-5] = raw_det['nusc_box'].score
    
    def __getitem__(self, item) -> ScoreObject:
        return self.frame_objects[item]

    def __len__(self) -> int:
        return len(self.frame_objects)
        
        
    
    