import numpy as np
import utils


def populate_pc(depth, nK):
    H, W = depth.shape[:2]
    K = utils.unnormalize_K(nK, (W, H))
    py, px = np.mgrid[:H,:W]
    xyz = np.stack([px, py, np.ones_like(px)], axis=-1) * np.atleast_3d(depth)
    xyz = np.reshape(xyz, (-1, 3)) @ np.linalg.inv(K).T
    return xyz

