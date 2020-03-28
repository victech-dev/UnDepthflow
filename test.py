import tensorflow as tf
from tensorflow.python.platform import app
import os
import numpy as np
import imgtool
import cv2
import open3d as o3d

from autoflags import opt, autoflags
from eval.evaluate_flow import load_gt_flow_kitti, get_scaled_intrinsic_matrix, scale_intrinsics
from eval.evaluate_mask import load_gt_mask
from eval.evaluation_utils import width_to_focal
from eval.test_model import TestModel

def predict_depth_single(sess, model, image1, image2, image1r, image2r, K, fxb):
    height, width = image1.shape[:2] # original height, width
    # session run
    outputs = model(sess, image1, image2, image1r, image2r, K)
    disp0 = outputs['stereo']['disp'][0] # [1, H, W, (ltr,rtl)]
    pred_disp = disp0[0,:,:,0:1]
    # depth from disparity
    pred_disp = width * cv2.resize(pred_disp, (width, height))
    pred_depth = fxb / pred_disp
    return pred_depth

def predict_depth_single_gt_2015(sess, model, i):
    gt_dir = opt.gt_2015_dir
    img1 = imgtool.imread(os.path.join(gt_dir, "image_2", str(i).zfill(6) + "_10.png"))
    img2 = imgtool.imread(os.path.join(gt_dir, "image_2", str(i).zfill(6) + "_11.png"))
    img1r = imgtool.imread(os.path.join(gt_dir, "image_3", str(i).zfill(6) + "_10.png"))
    img2r = imgtool.imread(os.path.join(gt_dir, "image_3", str(i).zfill(6) + "_11.png"))
    K = get_scaled_intrinsic_matrix(os.path.join(gt_dir, "calib_cam_to_cam", str(i).zfill(6) + ".txt"), 1.0, 1.0)
    fxb = width_to_focal[img1.shape[1]] * 0.54
    depth = predict_depth_single(sess, model, img1, img2, img1r, img2r, K, fxb)
    # import time
    # time0 = time.time()
    # for _ in range(1000):
    #     predict_depth_single(sess, model, img1, img2, img1r, img2r, K, fxb)
    # time1 = time.time()
    # print("**** 1000 elapsed:", time1 - time0)
    return img1, depth, K

def predict_depth_vicimg(sess, model, imgnameL, imgnameR):
    imgL = imgtool.imread(imgnameL)
    imgR = imgtool.imread(imgnameR)
    K = [9.5061071654182354e+02, 0.0, 5.8985625846591154e+02, 0.0, 9.5061071654182354e+02, 3.9634126783635918e+02, 0, 0, 1]
    K = np.array(K).reshape(3,3)
    fxb = 9.5061071654182354e+02 / 8.2988120552523057 # Q[3,4]/Q[4,3]
    depth = predict_depth_single(sess, model, imgL, imgL, imgR, imgR, K, fxb)
    return imgL, depth, K

def create_axis_bar():
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

def count_weights():
    var_pose = list(set(tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=".*pose_net.*")))
    var_depth = list(set(tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=".*(depth_net|feature_net_disp).*")))
    var_flow = list(set(tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=".*(flow_net|feature_net_flow).*")))
    if opt.mode == "depthflow":
        var_train_list = var_pose + var_depth + var_flow
    elif opt.mode == "depth":
        var_train_list = var_pose + var_depth
    elif opt.mode == "flow":
        var_train_list = var_flow
    else:
        var_train_list = var_depth
    sizes = [np.prod(v.shape.as_list()) for v in var_train_list]
    print("*** Total weight count:", np.sum(sizes))

def show_pcd(img, depth, K):
    if hasattr(show_pcd, 'axisbar'):
        axisbar = getattr(show_pcd, 'axisbar')
    else:
        axisbar = create_axis_bar()
        setattr(show_pcd, 'axisbar', axisbar)
    py, px = np.mgrid[:img.shape[0],:img.shape[1]]
    depth = np.clip(depth, 0, 100) # some clipping required here
    xyz = np.stack([px, py, np.ones_like(px)], axis=-1) * np.expand_dims(depth, axis=-1)
    xyz = np.reshape(xyz, (-1, 3)) @ np.linalg.inv(K).T
    rgb = np.reshape(img, (-1, 3)) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    # np.savetxt('./scene.txt', np.hstack([xyz.astype(np.float32), rgb.astype(np.float32)]))
    # pcd = o3d.io.read_point_cloud("./scene.txt", format='xyzrgb')
    o3d.visualization.draw_geometries([pcd] + axisbar)


def main(unused_argv):
    opt.trace = '' # this should be empty because we have no output when testing
    opt.batch_size = 1
    opt.mode = 'stereo'
    opt.pretrained_model = '.results_original/model-stereo'
    Model, _ = autoflags()

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        print('Constructing models and inputs.')
        with tf.variable_scope(tf.get_variable_scope()) as vs:
            with tf.name_scope("test_model"):
                model = TestModel(Model, vs)

        # Create a saver.
        saver = tf.train.Saver(max_to_keep=10)

        # Make training session.
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.log_device_placement = False
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, opt.pretrained_model)

        # # point cloud test of KITTI 2015 gt
        for i in range(200):
            img, depth, K = predict_depth_single_gt_2015(sess, model, i)
            show_pcd(img, depth, K)

        # # point cloud test of office image
        # data_dir = 'M:\\Users\\sehee\\StereoCalibrationExample_200313_1658'
        # imgnameL = os.path.join(data_dir, 'photo04_L.jpg')
        # imgnameR = os.path.join(data_dir, 'photo04_R.jpg')
        # img, depth, K = predict_depth_vicimg(sess, model, imgnameL, imgnameR)
        # depth = np.clip(depth, 0, 20)
        # show_pcd(img, depth, K)


if __name__ == '__main__':
    app.run()
