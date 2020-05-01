import numpy as np
import open3d as o3d
from pathlib import Path
import functools
import cv2
from pcd_utils.core import COORD_FRAMES, populate_pcd

class NavScene(object):
    def __init__(self, feeder):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()
        self.vis.register_key_callback(8, functools.partial(self.load_frame, key='prev'))
        self.vis.register_key_callback(32, functools.partial(self.load_frame, key='next')) 
        for geo in COORD_FRAMES:
            self.vis.add_geometry(geo)
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)

        self.feeder = feeder
        self.index = 0
        self.load_frame(self.vis)

    def load_frame(self, vis, key=None):
        if key == 'prev':
            self.index -= 1
        if key == 'next':
            self.index += 1

        img, depth, K = self.feeder(self.index)
        xyz = populate_pcd(depth, K)
        rgb = np.reshape(img, (-1, 3)) / 255
        self.pcd.points = o3d.utility.Vector3dVector(xyz)
        self.pcd.colors = o3d.utility.Vector3dVector(rgb)
        self.vis.update_geometry(self.pcd)

        # from imgtool import imshow
        # depth2 = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)
        # depth2 = cv2.convertScaleAbs(depth2, alpha=255)
        # show = np.concatenate([img, img], axis=0)
        # show[img.shape[0]:,:] = np.atleast_3d(depth2)
        # imshow(show, wait=False, norm=False)

    def run(self):
        self.vis.run()

    def clear(self):
        self.vis.destroy_window()
        del self.vis
