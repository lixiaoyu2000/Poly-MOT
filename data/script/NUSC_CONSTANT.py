"""
Public constant accessible to all files
"""

import numpy as np

# metrics with two return values
METRIC = ['iou_3d', 'giou_3d']
FAST_METRIC = ['giou_3d', 'giou_bev']

# category name(str) <-> category label(int)
CLASS_SEG_TO_STR_CLASS = {'bicycle': 0, 'bus': 1, 'car': 2, 'motorcycle': 3, 'pedestrian': 4, 'trailer': 5, 'truck': 6}
CLASS_STR_TO_SEG_CLASS = {0: 'bicycle', 1: 'bus', 2: 'car', 3: 'motorcycle', 4: 'pedestrian', 5: 'trailer', 6: 'truck'}

# math
PI, TWO_PI = np.pi, 2 * np.pi

# init EKFP for different non-linear motion model
CTRA_INIT_EFKP = {
    # [x, y, z, w, l, h, v, a, theta, omega]
    'bus': [10, 10, 10, 10, 10, 10, 10, 10, 1000, 10],
    'car': [4, 4, 4, 4, 4, 4, 1000, 4, 1, 0.1],
    'trailer': [10, 10, 10, 10, 10, 10, 10, 10, 1000, 10],
    'truck': [10, 10, 10, 10, 10, 10, 10, 10, 1000, 10],
    'pedestrian': [10, 10, 10, 10, 10, 10, 10, 10, 1000, 10]
}
BIC_INIT_EKFP = {
    # [x, y, z, w, l, h, v, a, theta, sigma]
    'bicycle': [10, 10, 10, 10, 10, 10, 10000, 10, 10, 10],
    'motorcycle': [4, 4, 4, 4, 4, 4, 100, 4, 4, 1],
}