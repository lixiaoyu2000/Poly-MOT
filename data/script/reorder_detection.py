"""
Organize detector files in chronological order on the NuScenes dataset
"""

import os, json, sys
sys.path.append('../..')
from utils.io import load_file
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

OUTPUT_ROOT_PATH = "../detector/"


def reorder_detection(detector_path, dataset_path, dataset_name='NuScenes',
                      dataset_version='trainval', detector_name='centerpoint'):
    """
    :param detector_path: path of detection file
    :param dataset_path: root path of dataset file
    :param dataset_name: name of dataset
    :param dataset_version: version(split) of dataset (trainval/test)
    :param detector_name: name of detector eg: CenterPoint..
    :return: Reorganized detection files .json
    """
    assert dataset_version in ['trainval', 'test'] and dataset_name in ['NuScenes', 'Waymo'], \
        "unsupported dataset or data version"

    if dataset_name == 'NuScenes':
        nusc = NuScenes(version='v1.0-' + dataset_version, dataroot=dataset_path,
                        verbose=True)
        frame_num = 6019 if dataset_version == 'trainval' else 6008

        # load detector file
        chaos_detector_json = load_file(detector_path)
        assert len(chaos_detector_json['results']) == frame_num, "wrong detection result"
        first_token_path = '../utils/first_token_table/{}/nusc_first_token.json'.format(dataset_version)
        all_token_table = from_first_to_all(nusc, first_token_path)
        assert len(all_token_table) == frame_num

        # reorder file
        order_file = {
            "results": {token: chaos_detector_json['results'][token] for token in all_token_table},
            "meta": chaos_detector_json["meta"]
        }

        # output file
        version = 'val' if dataset_version == "trainval" else 'test'
        os.makedirs(OUTPUT_ROOT_PATH + version, exist_ok=True)
        OUTPUT_PATH = OUTPUT_ROOT_PATH + version + f"/{version}_{detector_name}.json"
        print(f"write order detection file to {OUTPUT_PATH}")
        json.dump(order_file, open(OUTPUT_PATH, "w"))

    else:
        raise Exception("Waymo dataset is not currently supported")


def from_first_to_all(nusc, first_token_path):
    """
    :param nusc: NuScenes class
    :param first_token_path: path of first frame token for each seq
    :return: list format token table
    """
    first_token_table, seq_num = load_file(first_token_path), 150
    assert len(first_token_table) == seq_num, "wrong token table"
    all_token_table = []
    for first_token in first_token_table:
        curr_token = first_token
        while curr_token != '':
            all_token_table.append(curr_token)
            curr_token = nusc.get('sample', curr_token)['next']

    return all_token_table

if __name__ == "__main__":
    reorder_detection(
        detector_path='../detector/raw_detector/infos_val_10sweeps_withvelo_filter_True.json',
        dataset_path='/mnt/share/sda-8T/rj/Dateset/Nuscenes/data/nuscenes',
        dataset_name='NuScenes',
        dataset_version='trainval',
        detector_name='centerpoint'
    )