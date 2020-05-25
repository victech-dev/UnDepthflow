import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
import functools
from scipy.spatial.transform import Rotation as R

from estimate.pcdlib import COORD_FRAMES

class IcpVis(object):
    def __init__(self, pcd1, pcd2, tfm):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()
        self.vis.register_key_callback(32, functools.partial(self.load_frame, key='next')) 
        for geo in COORD_FRAMES:
            self.vis.add_geometry(geo)

        self.pcd1 = o3d.geometry.PointCloud()
        self.pcd1.points = o3d.utility.Vector3dVector(pcd1)
        self.pcd1.paint_uniform_color([1, 0.706, 0])
        self.pcd1_raw = pcd1

        self.pcd2 = o3d.geometry.PointCloud()
        self.pcd2.points = o3d.utility.Vector3dVector(pcd2)
        self.pcd2.paint_uniform_color([0, 0.651, 0.929])
        self.pcd2_raw = pcd2

        self.vis.add_geometry(self.pcd1)
        self.vis.add_geometry(self.pcd2)

        # set camera pose properly
        vc = self.vis.get_view_control()
        cam_params = vc.convert_to_pinhole_camera_parameters()
        new_extrinsic = np.copy(cam_params.extrinsic)
        new_extrinsic[:3,:3] = R.from_euler('x', 10, degrees=True).as_matrix()
        new_extrinsic[:3,3] = [0, 0.3, 0]
        new_cam_params = o3d.camera.PinholeCameraParameters()
        new_cam_params.intrinsic = cam_params.intrinsic
        new_cam_params.extrinsic = new_extrinsic
        vc.convert_from_pinhole_camera_parameters(new_cam_params)

        self.index = 0
        self.tfm = tfm

    def load_frame(self, vis, key=None):
        self.index = 1 - self.index

        self.pcd1.points = o3d.utility.Vector3dVector(self.pcd1_raw)
        if self.index == 1:
            self.pcd1.transform(self.tfm)
        
        self.vis.update_geometry(self.pcd1)

    def run(self):
        self.vis.run()

    def clear(self):
        self.vis.destroy_window()
        del self.vis


def cv2_solve_icp(pc1, pc2, filter_far=True):
    _, pcn1 = cv2.ppf_match_3d.computeNormalsPC3d(pc1, 8, False, (0,0,0))
    _, pcn2 = cv2.ppf_match_3d.computeNormalsPC3d(pc2, 8, False, (0,0,0))
    if filter_far:
        dist1 = np.linalg.norm(pcn1[:,:3], axis=-1)
        idx1 = np.argsort(dist1)
        pcn1 = pcn1[idx1[:len(idx1)//2]]
        dist2 = np.linalg.norm(pcn2[:,:3], axis=-1)
        idx2 = np.argsort(dist2)
        pcn2 = pcn2[idx2[:len(idx2)//2]]
    icp = cv2.ppf_match_3d_ICP(iterations=100, tolerence=1e-3, rejectionScale=1.0, numLevels=3)
    _, _, tfm = icp.registerModelToScene(pcn1, pcn2)
    return tfm


if __name__ == "__main__":
    import cv2
    from pathlib import Path
    import time

    from estimate import populate_pcd
    import utils

    K_org, baseline = utils.query_K('victech')
    K_new = utils.rescale_K(K_org, (640, 480), (128, 96))

    for i in range(3, 90):
        root_dir = Path("M:\\Users\\sehee\\camera_taker\\undist_fisheye\\depthL")
        print('* ICP:', i, i+1)
        depth1 = cv2.imread(str(root_dir/f'{i:06d}.pfm'), -1)
        depth2 = cv2.imread(str(root_dir/f'{i+1:06d}.pfm'), -1)
        pc1 = populate_pcd(depth1, K_new)
        pc2 = populate_pcd(depth2, K_new)

        t0 = time.time()
        tfm = cv2_solve_icp(pc1, pc2)
        t1 = time.time()
        print(t1 - t0, tfm)

        # pcd1_t = np.concatenate([pcd1, np.ones_like(pcd1[:,:1])], axis=-1)
        # pcd1_t = pcd1_t @ tfm.T
        # pcd1_t = pcd1_t[:,:3] / pcd1_t[:,-1:]
        vis = IcpVis(pc1, pc2, tfm)
        vis.run()
        vis.clear()

