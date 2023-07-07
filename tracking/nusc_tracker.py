"""
Tracker, Core of Poly-MOT.
Tracklet prediction and punishment, cost matrix construction, tracking id assignment, tracklet update and init, and output file

TODO: consider tentative tracklets, Reorganize the output format
TODO: delete debug log in the release version
"""

import pdb
import numpy as np
from nusc_trajectory import Trajectory
from data.script.NUSC_CONSTANT import *
from geometry.nusc_distance import iou_bev, iou_3d, giou_bev, giou_3d, d_eucl
from utils.matching import mask_tras_dets, fast_compute_check, reorder_metrics, spec_metric_mask, \
    linear_assignment


class Tracker:
    def __init__(self, config):
        self.cfg = config
        # Hyper parameters
        self.is_debug = self.cfg['debug']['is_debug']
        self.cls_num = self.cfg['basic']['CLASS_NUM']
        self.f_thre, self.s_thre = config['association']['first_thre'], config['association']['second_thre']
        self.two_stage, self.algorithm = config['association']['two_stage'], config['association']['algorithm']
        self.punish_num, self.metrics = config['output']['punish_num'], config['association']['category_metrics']
        self.second_metric = config['association']['second_metric']
        self.post_nms_cfg = config['output']
        self.fast = fast_compute_check(self.metrics, self.second_metric)
        self.re_metrics = reorder_metrics(self.metrics)
        # init, notice that no tentative trajectories in Poly-MOT best experiments.
        # self.xx_tras -> {tracking id(int): trajectory}
        self.active_tras, self.tentative_tras, self.dead_tras, self.valid_tras = {}, {}, {}, {}
        self.id_seed, self.frame_id, self.seq_id, self.det_infos, self.tra_infos = 0, None, None, None, None

    def reset(self) -> None:
        """
        Initialize Tracker for each new seq
        """
        self.active_tras, self.tentative_tras, self.dead_tras, self.valid_tras = {}, {}, {}, {}
        self.id_seed, self.frame_id, self.seq_id, self.det_infos, self.tra_infos = 0, None, None, None, None

    def tracking(self, data_info: dict) -> None:
        """
        :param data_info: the observation information(detection) of each frame
        :return: update data_info: {
            'np_track_res': np.array, [num, 17] add 'tracking_id', 'seq_id', 'frame_id'
            'box_track_res': np.array[nusc_box], [num,]
            'no_val_track_result'[optimal]: bool
        }
        """
        self.det_infos, self.frame_id, self.seq_id = data_info, data_info['frame_id'], data_info['seq_id']
        
        # step0. reinit tracker for each seq
        if self.det_infos['is_first_frame']: self.reset()

        # step1. predict all valid trajectories
        self.tras_predict()

        # step2. if there is no dets, we will punish all valid trajectories
        if self.det_infos['no_dets']:
            self.tras_punish(data_info)
            if self.post_nms_cfg['post_nms']: self.post_nms_tras(data_info)
            return

        # step3. associate current frame detections with existing trajectories
        tracking_ids = self.data_association()

        # step4. use observations(detection) to update the corresponding trajectories
        # and output unmatch trajectories(up to punish_num) states, and output new trajectories states
        dict_track_res = self.tras_update(tracking_ids, data_info)
        if len(dict_track_res['np_track_res']) == 0: return 
            
        # step5. update and output tracking results
        data_info.update({
            'np_track_res': dict_track_res['np_track_res'],
            'box_track_res': dict_track_res['box_track_res'],
            'bm_track_res': dict_track_res['bottom_corners_track_res'],
        })
        
        # whether to use post-predict to reduce FP prediction
        if self.post_nms_cfg['post_nms']: self.post_nms_tras(data_info)

    def tras_predict(self) -> None:
        """
        State Prediction for all trajectories
        get self.tra_infos: {
            'np_tras': np.array, [valid_tra_num, 14]
            'np_tras_bottom_corners': np.array[nusc_box], [valid_tra_num,]
            'all_valid_ids': np.array, [valid_tra_num,]
            'all_valid_boxes': np.array[nusc_box], [valid_tra_num,]
            'tra_num': len(all_valid_ids)
        }
        """
        # debug, Check for tracklets with duplicate states
        if self.is_debug: self.debug()
        pred_infos, pred_bms, pred_boxes, all_valid_ids = [], [], [], []

        # corner case(such as first frame..), no valid tracklets
        if len(self.valid_tras) == 0: return

        # predict tracklet for data association
        for tra_id, tra in self.valid_tras.items():
            # only for debug
            if self.is_debug: assert tra_id not in self.dead_tras

            # predict each valid tracklet state
            tra.state_predict(timestamp=self.frame_id)
            pred_object = tra[self.frame_id]
            
            all_valid_ids.append(tra_id)
            pred_boxes.append(pred_object.pred_box)
            pred_infos.append(pred_object.pred_infos)
            pred_bms.append(pred_object.pred_box.bottom_corners_)
            
        # only for debug
        if self.is_debug:
            self.valid_tras = self.merge_valid_tras()
            assert len(all_valid_ids) == len(self.valid_tras)
        self.tra_infos = {
            'np_tras': np.array(pred_infos),  # info dim: 17, add 'tracking_id', 'seq_id', 'frame_id'
            'np_tras_bottom_corners': np.array(pred_bms),
            'all_valid_ids': np.array(all_valid_ids),
            'all_valid_boxes': np.array(pred_boxes),
            'tra_num': len(all_valid_ids)
        }

    def tras_punish(self, data_info: dict) -> None:
        """
        handle the corner case where there is no detection at current frame
        also can be seen as "short-cut"
        :param data_info: dict, output file
        :return: update pure predict state, up to output punish_num frame
        update data_info: {
            'np_track_res': np.array, [num, 17] add 'tracking_id', 'seq_id', 'frame_id'
            'box_track_res': np.array[nusc_box], [num,]
        }
        """
        # no valid tras after predicting(valid ids is empty) or no valid tras at prev frame(None)
        if self.tra_infos is None or self.tra_infos['tra_num'] == 0: 
            data_info.update({'no_val_track_result': True})
            return
        
        # manage trajectory
        dict_track_res = self.tras_update(tracking_ids=np.array([]), data_info=data_info)
        if len(dict_track_res['np_track_res']) == 0: 
            data_info.update({'no_val_track_result': True})
            return 
            
        # output punishment tracking results
        data_info.update({
            'np_track_res': dict_track_res['np_track_res'],
            'box_track_res': dict_track_res['box_track_res'],
            'bm_track_res': dict_track_res['bottom_corners_track_res'],
        })


    def tras_update(self, tracking_ids: np.array, data_info: dict) -> dict:
        """
        update the corresponding trajectories with observations, init new tras,
        and punish unmatched tracklets
        :param tracking_ids: np.array, tracking id of each detection
        :param data_info: dict, dets infos at the current frame
        :return: dict, valid estimated states(updated tras, new tras, valid unmatched tras)
            {
            'np_track_res': np.array, [valid_tra_num, 17],
            'box_track_res': np.array[nusc_box], [valid_tra_num,]
            'bm_track_res': np.array, [valid_tra_num, 4, 2]
            }
        """
        tracking_ids = tracking_ids.tolist()
        assert len(tracking_ids) == self.det_infos['det_num'] and len(self.valid_tras) != 0
        np_res, box_res, bm_res = [], [], []
        new_tras, ten_tras, act_tras = {}, {}, {}
        
        # iterative detections, use measurement(dets) to correct tracklet
        for det_idx, tra_id in enumerate(tracking_ids):
            dict_det = {
                'nusc_box': data_info['box_dets'][det_idx],
                'np_array': data_info['np_dets'][det_idx],
                'has_velo': data_info['has_velo']
            }
            if self.is_debug: assert tra_id not in self.dead_tras
            if tra_id in self.valid_tras:
                # update exist trajectory
                tra = self.valid_tras[tra_id]
                tra.state_update(timestamp=self.frame_id, det=dict_det)
            else:
                # init new trajectory
                tra = Trajectory(timestamp=self.frame_id,
                                 config=self.cfg,
                                 track_id=tra_id,
                                 det_infos=dict_det)
                new_tras[tra_id] = tra
        
        # merge all tras, include exist trajectories and newly generated trajectory
        tmp_merge_tras = {**self.valid_tras, **new_tras}
        
        # iterative trajectories, punish and output
        for tra_id, tra in tmp_merge_tras.items():
            # update unmatched tracklets
            if tra_id not in tracking_ids: 
                tra.state_update(timestamp=self.frame_id, det=None)
            update_object = tra[self.frame_id]
            # only active tracklets' state are output to the result file
            if tra.life_management.state == 'active':
                act_tras[tra_id] = tra
                if update_object.update_infos is not None:
                    np_res.append(update_object.update_infos)
                    box_res.append(update_object.update_box)
                    bm_res.append(update_object.update_bms)
                elif tra.life_management.time_since_update <= self.punish_num:
                    np_res.append(update_object.predict_infos)
                    box_res.append(update_object.predict_box)
                    bm_res.append(update_object.predict_bms)
            elif tra.life_management.state == 'tentative':
                ten_tras[tra_id] = tra
            elif tra.life_managemnet.state == 'dead':
                assert tra_id not in self.dead_tras
                self.dead_tras[tra_id] = tra
            else: raise Exception('Tracjectory state only have three attributes')
            
        # reorganize active/dead/tentative trajectories
        self.active_tras, self.tentative_tras = act_tras, ten_tras
        self.valid_tras = {**self.active_tras, **self.tentative_tras}
        
        dict_track_res = {
            'np_track_res': np_res,
            'box_track_res': box_res,
            'bm_track_res': bm_res,
        }
        
        return dict_track_res   

    def data_association(self) -> np.array:
        """
        Associate the track and the detection, and assign a tracking id to each detection
        :return: np.array, tracking ids of each detection
        """
        # corner case, no valid trajectory. quickly assign each det tracking id
        if len(self.valid_tras) == 0:
            ids = np.arange(self.id_seed, self.id_seed + self.det_infos['det_num'], dtype=int)
            self.id_seed += self.det_infos['det_num']
        else:
            cost_matrices = self.compute_cost()
            ids = self.matching_cost(cost_matrices)
        return ids

    def compute_cost(self) -> dict:
        """
        Construct the cost matrix between the trajectory and the detection
        :return: dict, a collection of cost matrices,
        one-stage: np.array, [cls_num, det_num, tra_num], two-stage: np.array, [det_num, tra_num]
        """
        assert self.tra_infos is not None and self.tra_infos['tra_num'] is not None
        det_num, tra_num = self.det_infos['det_num'], self.tra_infos['tra_num']
        det_labels, tra_labels = self.det_infos['np_dets'][:, -1], self.tra_infos['np_tras'][:, -4]

        # [cls_num, det_num, tra_num], True denotes valid (det label == tra label == cls idx)
        valid_mask = mask_tras_dets(self.cls_num, det_labels, tra_labels)
        self.tra_cost_infos = {'np_dets': self.tra_infos['np_tras'][:, :-3],
                               'np_dets_bottom_corners': self.tra_infos['np_tras_bottom_corners']}

        if self.fast:
            # metrics only have giou_3d/giou_bev
            first_cost, two_cost = giou_3d(self.det_infos, self.tra_cost_infos)
            first_cost = first_cost[None, :, :].repeat(self.cls_num, axis=0)
        else:
            two_cost = globals()[self.second_metric](self.det_infos, self.tra_cost_infos)
            first_cost = np.zeros((self.cls_num, det_num, tra_num))
            for metric, cls_list in self.re_metrics.items():
                # True denotes invalid(the object's category is not specific)
                self.tra_cost_infos['mask'] = spec_metric_mask(cls_list, det_labels, tra_labels)
                if metric in METRIC:
                    cost1, _ = globals()[metric](self.det_infos, self.tra_cost_infos)
                else:
                    cost1 = globals()[metric](self.det_infos, self.tra_cost_infos)
                first_cost[cls_list] = cost1

        # mask invalid value
        first_cost[np.where(~valid_mask)] = -np.inf

        # Due to the execution speed of python,
        # construct the two-stage cost matrix under half-parallel framework is very tricky, 
        # we strongly recommend only use giou_bev as two-stage metric to build the cost matrix
        return {'one_stage': 1 - first_cost, 'two_stage': 1 - two_cost if two_cost is not None else None}

    def matching_cost(self, cost_matrices: dict) -> np.array:
        """
        Solve the matching pair according to the cost matrix
        :param cost_matrices: cost matrices between dets and tras construct in the one/two stage
        :return: np.array, tracking id of each detection
        """
        cost1, cost2 = cost_matrices['one_stage'], cost_matrices['two_stage']
        # m_tras_1 is not the tracking id, but is the index of tracklet in the all valid trajectories
        m_dets_1, m_tras_1, um_dets_1, um_tras_1 = linear_assignment(cost1, self.f_thre, self.algorithm)
        if self.two_stage:
            inf_cost = np.ones_like(cost2) * np.inf
            inf_cost[np.ix_(um_dets_1, um_tras_1)] = 0
            cost2 += inf_cost
            m_dets_2, m_tras_2, _, _ = linear_assignment(cost2, self.s_thre, self.algorithm)
            m_dets_1 += m_dets_2
            m_tras_1 += m_tras_2

        assert len(m_dets_1) == len(m_tras_1), "as the pair, number of the matched tras and dets must be equal"
        # corner case, no matching pair after matching
        if len(m_dets_1) == 0:
            ids = np.arange(self.id_seed, self.id_seed + self.det_infos['det_num'], dtype=int)
            self.id_seed += self.det_infos['det_num']
            return ids
        else:
            ids, match_pairs = [], {key: value for key, value in zip(m_dets_1, m_tras_1)}
            all_valid_ids = self.tra_infos['all_valid_ids']
            for det_idx in range(self.det_infos['det_num']):
                if det_idx not in m_dets_1:
                    ids.append(self.id_seed)
                    self.id_seed += 1
                else:
                    ids.append(all_valid_ids[match_pairs[det_idx]])

        return np.array(ids)

    def merge_valid_tras(self) -> dict:
        """
        Get all valid trajectories, 'valid' denotes that 'active' and 'tentative'
        :return: dict, merge active tracklets and tentative tracklets
        """
        return {**self.active_tras, **self.tentative_tras}
    
    def post_nms_tras(self, data_info) -> None:
        """
        use post-predict to reduce FP prediction
        :param data_info: the final tracking result at each frame
        :retrun: no return, but filter FP results in the data_info
        TODO: move this function into 'main' Function
        """
        post_metric = self.post_nms_cfg['NMS_metric']
        post_thre = self.post_nms_cfg['NMS_thre']
        post_type = self.post_nms_cfg['NMS_type']
        tmp_tra_infos = {'np_dets': data_info['np_track_res'][0: -3], 
                         'np_dets_bottom_corners': data_info['bm_track_res']}
        keep = globals()[post_type](box_infos=tmp_tra_infos, metrics=post_metric, thre=post_thre)
        if len(keep) == 0: 
            data_info.update({'no_val_track_result': True})
        else: 
            data_info['np_track_res'] = data_info['np_track_res'][keep]
            data_info['box_track_res'] = data_info['box_track_res'][keep]

    def debug(self) -> None:
        """
        only for debug
        TODO: delete at public version
        """
        # 检查三类轨迹是否有重复的key
        assert len(self.tentative_tras.keys() & self.active_tras.keys()) == 0
        assert len(self.active_tras.keys() & self.dead_tras.keys()) == 0
        assert len(self.tentative_tras.keys() & self.dead_tras.keys()) == 0
