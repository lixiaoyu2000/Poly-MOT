"""
math script
"""

import numpy as np


def expand_dims(array: np.array, expand_len: int, dim: int) -> np.array:
    return np.expand_dims(array, dim).repeat(expand_len, axis=dim)

def warp_to_pi(yaw: float) -> float:
    """warp yaw to [-pi, pi)

    Args:
        yaw (float): raw angle

    Returns:
        float: raw angle after warping
    """
    while yaw >= np.pi:
        yaw -= 2 * np.pi
    while yaw < -np.pi:
        yaw += 2 * np.pi
    return yaw
