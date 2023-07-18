import yaml, argparse, time, os, json, multiprocessing
from dataloader.nusc_loader import NuScenesloader
from tracking.nusc_tracker import Tracker
from tqdm import tqdm
import pdb

parser = argparse.ArgumentParser()
# running configurations
parser.add_argument('--process', type=int, default=1)
# paths
localtime = ''.join(time.asctime(time.localtime(time.time())).split(' '))
parser.add_argument('--nusc_path', type=str, default='/mnt/intel/jupyterhub/yueling.shen/nuscenes')
parser.add_argument('--config_path', type=str, default='config/nusc_config.yaml')
parser.add_argument('--detection_path', type=str, default='data/detector/val/val_centerpoint.json')
parser.add_argument('--first_token_path', type=str, default='data/utils/first_token_table/trainval/nusc_first_token.json')
parser.add_argument('--result_path', type=str, default='result/' + localtime)
parser.add_argument('--eval_path', type=str, default='eval_result/')
args = parser.parse_args()


def main(result_path, token, process, nusc_loader):
    # PolyMOT modal is completely dependent on the detector modal
    result = {
        "results": {},
        "meta": {
            "use_camera": True,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }
    }
    # tracking and output file
    nusc_tracker = Tracker(config=nusc_loader.config)
    for frame_data in tqdm(nusc_loader, desc='Running', total=len(nusc_loader) // process, position=token):
        if process > 1 and frame_data['seq_id'] % process != token:
            continue
        sample_token = frame_data['sample_token']
        # track each sequence
        nusc_tracker.tracking(frame_data)
        """
        only for debug
        {
            'np_track_res': np.array, [num, 17] add 'tracking_id', 'seq_id', 'frame_id'
            'box_track_res': np.array[NuscBox], [num,]
            'no_val_track_result': bool
        }
        """
        # output process
        sample_results = []
        if 'no_val_track_result' not in frame_data:
            for predict_box in frame_data['box_track_res']:
                box_result = {
                    "sample_token": sample_token,
                    "translation": [float(predict_box.center[0]), float(predict_box.center[1]), float(predict_box.center[2])],
                    "size": [float(predict_box.wlh[0]), float(predict_box.wlh[1]), float(predict_box.wlh[2])],
                    "rotation": [float(predict_box.orientation[0]), float(predict_box.orientation[1]),
                                    float(predict_box.orientation[2]), float(predict_box.orientation[3])],
                    "velocity": [float(predict_box.velocity[0]), float(predict_box.velocity[1])],
                    "tracking_id": str(predict_box.tracking_id),
                    "tracking_name": predict_box.name,
                    "tracking_score": predict_box.score,
                }
                sample_results.append(box_result.copy())
                
        # add to the output file
        if sample_token in result["results"]:
            result["results"][sample_token] = result["results"][sample_token] + sample_results
        else:
            result["results"][sample_token] = sample_results
    
    # sort track result by the tracking score
    for sample_token in result["results"].keys():
        confs = sorted(
            [
                (-d["tracking_score"], ind)
                for ind, d in enumerate(result["results"][sample_token])
            ]
        )
        result["results"][sample_token] = [
            result["results"][sample_token][ind]
            for _, ind in confs[: min(500, len(confs))]
        ]
        
    # write file
    if (process > 1):
        json.dump(result, open(result_path + str(token) +".json", "w"))
    else:
        json.dump(result, open(result_path +"results.json", "w"))

def eval(result_path, eval_path, nusc_path):
    from nuscenes.eval.tracking.evaluate import TrackingEval
    from nuscenes.eval.common.config import config_factory as track_configs
    cfg = track_configs("tracking_nips_2019") 
    nusc_eval = TrackingEval(
        config=cfg,
        result_path=result_path,
        eval_set="val",
        output_dir=eval_path,
        verbose=True,
        nusc_version="v1.0-trainval",
        nusc_dataroot=nusc_path,
    )
    print("result in " + result_path)
    metrics_summary = nusc_eval.main()


if __name__ == "__main__":
    os.makedirs(args.result_path, exist_ok=True)
    os.makedirs(args.eval_path, exist_ok=True)
    # load config and dataloader
    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.Loader)
    nusc_loader = NuScenesloader(args.detection_path, args.first_token_path, config)    
    print('writing result in folder: ' + os.path.abspath(args.result_path))
    if args.process > 1:
        result_temp_path = args.result_path + '/temp_result'
        os.makedirs(result_temp_path, exist_ok=True)
        pool = multiprocessing.Pool(args.process)
        for token in range(args.process):
            pool.apply_async(main, args = (result_temp_path, token, args.process, nusc_loader))
        pool.close()
        pool.join()
        results = {'results': {}, 'meta': {}}
        # combine the results of each process
        for token in range(args.process):
            result = json.load(open(os.path.join(result_temp_path, str(token) + '.json'), 'r'))
            results["results"].update(result["results"])
            results["meta"].update(result["meta"])
        json.dump(results, open(args.result_path + '/results.json', "w"))
    else:
        main(args.result_path, 0, 1, nusc_loader)
    # eval result
    eval(args.result_path, args.eval_path, args.nusc_path)