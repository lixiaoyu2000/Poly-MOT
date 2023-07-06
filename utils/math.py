"""
math script for the NuScenes dataset
"""

import numpy as np


def expand_dims(array: np.array, expand_len: int, dim: int) -> np.array:
    return np.expand_dims(array, dim).repeat(expand_len, axis=dim)
