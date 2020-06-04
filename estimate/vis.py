import open3d as o3d
import numpy as np
import functools
from scipy.spatial.transform import Rotation as R
from estimate.pcl import populate_pc

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


def show_pcd(img, depth, nK):
    ''' visualize point cloud for single scene'''
    assert np.all(img.shape[:2] == depth.shape[:2]) and img.shape[2] == 3 and img.dtype == np.uint8
    xyz = populate_pc(depth, nK)
    rgb = np.reshape(img, (-1, 3)) / 255
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.visualization.draw_geometries([pcd] + COORD_FRAMES)


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

        self.feeder = feeder
        self.index = 0
        self.load_frame(self.vis)

    def load_frame(self, vis, key=None):
        if key == 'prev':
            self.index -= 1
        if key == 'next':
            self.index += 1

        img, depth, nK = self.feeder(self.index)
        xyz = populate_pc(depth, nK)
        rgb = np.reshape(img, (-1, 3)) / 255
        self.pcd.points = o3d.utility.Vector3dVector(xyz)
        self.pcd.colors = o3d.utility.Vector3dVector(rgb)
        self.vis.update_geometry(self.pcd)

    def run(self):
        self.vis.run()

    def clear(self):
        self.vis.destroy_window()
        del self.vis
