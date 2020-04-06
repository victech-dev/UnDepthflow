import numpy as np
import open3d as o3d
from pathlib import Path
import functools

def _create_axis_bar():
    LEN, DIV, RADIUS = 20, 1, 0.02
    color = np.eye(3)
    rot = [o3d.geometry.get_rotation_matrix_from_xyz([0,0.5*np.pi,0]), 
        o3d.geometry.get_rotation_matrix_from_xyz([-0.5*np.pi,0,0]), 
        np.eye(3)]
    bar = []
    for c,r in zip(color, rot):
        color_blend = False
        for pos in np.arange(0, LEN, DIV):
            b = o3d.geometry.TriangleMesh.create_cylinder(radius=RADIUS, height=DIV)
            b.paint_uniform_color(c*0.5 + 0.5 if color_blend else c); color_blend = not color_blend
            b.translate([0,0,pos + DIV/2])
            b.rotate(r, center=False)
            bar.append(b)
    return bar

COORD_FRAMES = _create_axis_bar()


def populate_pcd(img, depth, K):
    H, W = img.shape[:2]
    py, px = np.mgrid[:H,:W]
    xyz = np.stack([px, py, np.ones_like(px)], axis=-1) * np.expand_dims(depth, axis=-1)
    xyz = np.reshape(xyz, (-1, 3)) @ np.linalg.inv(K).T
    rgb = np.reshape(img, (-1, 3)) / 255.0
    # remove 0-depth area
    mask = (depth > 0).flatten()
    return xyz[mask], rgb[mask]
        

def show_pcd(img, depth, K):
    ''' visualize point cloud for single scene'''
    xyz, rgb = populate_pcd(img, depth, K)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.visualization.draw_geometries([pcd] + COORD_FRAMES)


class NavScene(object):
    def __init__(self, feeder):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()
        self.vis.register_key_callback(44, functools.partial(self.load_frame, key='prev'))
        self.vis.register_key_callback(46, functools.partial(self.load_frame, key='next')) 
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
        xyz, rgb = populate_pcd(img, depth, K)
        self.pcd.points = o3d.utility.Vector3dVector(xyz)
        self.pcd.colors = o3d.utility.Vector3dVector(rgb)
        self.vis.update_geometry(self.pcd)

    def run(self):
        self.vis.run()

    def clear(self):
        self.vis.destroy_window()
        del self.vis
