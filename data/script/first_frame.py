"""
get first frame token for every seq on the NuScenes dataset
TODO: support Waymo dataset
"""

import os, json, sys
sys.path.append('../..')
from utils.io import load_file
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

FIRST_TOKEN_ROOT_PATH = '../utils/first_token_table/'


def extract_first_token(dataset_path, detector_path, dataset_name='NuScenes', dataset_version='trainval'):
    """
    :param dataset_path: path of dataset
    :param dataset_name: name of dataset
    :param detector_path: path of detection file
    :param dataset_version: version(split) of dataset (trainval/test)
    :return: first frame token table .json
    """
    assert dataset_version in ['trainval', 'test'] and dataset_name in ['NuScenes', 'Waymo'], \
        "unsupported dataset or data version"

    if dataset_name == 'NuScenes':
        nusc = NuScenes(version='v1.0-' + dataset_version, dataroot=dataset_path,
                        verbose=True)
        frame_num = 6019 if dataset_version == 'trainval' else 6008
        seq_num = 150

        # load detector file
        detector_json = load_file(detector_path)
        assert len(detector_json['results']) == frame_num, "wrong detection result"

        # get first frame token of each seq
        first_token_table = []
        print("Extracting first frame token...")
        for sample_token in tqdm(detector_json['results']):
            if nusc.get('sample', sample_token)['prev'] == '':
                first_token_table.append(sample_token)
        assert len(first_token_table) == seq_num, "wrong detection result"

        # write token table
        os.makedirs(FIRST_TOKEN_ROOT_PATH + dataset_version, exist_ok=True)
        FIRST_TOKEN_PATH = FIRST_TOKEN_ROOT_PATH + dataset_version + "/nusc_first_token.json"
        print(f"write token table to {FIRST_TOKEN_PATH}")
        json.dump(first_token_table, open(FIRST_TOKEN_PATH, "w"))

    else:
        raise Exception("Waymo dataset is not currently supported")


if __name__ == "__main__":
    extract_first_token(
        dataset_path='/mnt/share/sda-8T/rj/Dateset/Nuscenes/data/nuscenes',
        detector_path='../detector/raw_detector/infos_val_10sweeps_withvelo_filter_True.json',
        dataset_name='NuScenes',
        dataset_version='trainval'
    )
