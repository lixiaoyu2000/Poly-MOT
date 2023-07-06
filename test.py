import yaml
import pdb
import time
from tqdm import tqdm
import numpy as np
from dataloader.nusc_loader import NuScenesloader
from geometry.nusc_distance import giou_3d, giou_3d_s, iou_3d, iou_3d_s, giou_bev, giou_bev_s, iou_bev, iou_bev_s

if __name__ == "__main__":
    config_path = 'config/nusc_config.yaml'
    detection_path = 'data/detector/val/val_centerpoint.json'
    first_token_path = 'data/utils/first_token_table/trainval/nusc_first_token.json'
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    nusc_loader = NuScenesloader(detection_path, first_token_path, config)
    for i, frame_data in enumerate(nusc_loader):
        if i == 0: st = time.time()

    print(time.time() - st)
