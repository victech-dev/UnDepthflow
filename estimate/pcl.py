import numpy as np


def populate_pcd(depth, K):
    H, W = depth.shape[:2]
    py, px = np.mgrid[:H,:W]
    xyz = np.stack([px, py, np.ones_like(px)], axis=-1) * np.atleast_3d(depth)
    xyz = np.reshape(xyz, (-1, 3)) @ np.linalg.inv(K).T
    return xyz

