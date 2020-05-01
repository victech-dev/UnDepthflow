import numpy as np
import open3d as o3d
from pathlib import Path
import functools
import cv2

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


def populate_pcd(depth, K):
    H, W = depth.shape[:2]
    py, px = np.mgrid[:H,:W]
    xyz = np.stack([px, py, np.ones_like(px)], axis=-1) * np.atleast_3d(depth)
    xyz = np.reshape(xyz, (-1, 3)) @ np.linalg.inv(K).T
    return xyz

        
def show_pcd(img, depth, K):
    ''' visualize point cloud for single scene'''
    assert np.all(img.shape[:2] == depth.shape[:2]) and img.shape[2] == 3 and img.dtype == np.uint8
    xyz = populate_pcd(depth, K)
    rgb = np.reshape(img, (-1, 3)) / 255
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.visualization.draw_geometries([pcd] + COORD_FRAMES)


