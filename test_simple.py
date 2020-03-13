import tensorflow as tf

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from monodepth_dataloader import MonodepthDataloader
from models import *

from eval.evaluate_flow import load_gt_flow_kitti, get_scaled_intrinsic_matrix, scale_intrinsics
from eval.evaluate_mask import load_gt_mask
from eval.evaluation_utils import width_to_focal
from loss_utils import average_gradients

from test import test
import os
import numpy as np
import imgtool
import cv2
import open3d as o3d

FLAGS = flags.FLAGS

flags.DEFINE_string('pretrained_model', '',
                    'filepath of a pretrained model to initialize from.')
flags.DEFINE_string(
    'mode', '',
    'selection from four modes of ["flow", "depth", "depthflow", "stereo"]')

flags.DEFINE_string('data_dir', '', 'root filepath of data.')
flags.DEFINE_string('gt_2012_dir', '',
                    'directory of ground truth of kitti 2012')
flags.DEFINE_string('gt_2015_dir', '',
                    'directory of ground truth of kitti 2015')

flags.DEFINE_integer("img_height", 256, "Image height")
flags.DEFINE_integer("img_width", 832, "Image width")

flags.DEFINE_float("depth_smooth_weight", 10.0, "Weight for depth smoothness")
flags.DEFINE_float("ssim_weight", 0.85,
                   "Weight for using ssim loss in pixel loss")
flags.DEFINE_float("flow_smooth_weight", 10.0, "Weight for flow smoothness")
flags.DEFINE_float("flow_consist_weight", 0.01, "Weight for flow consistent")
flags.DEFINE_float("flow_diff_threshold", 4.0,
                   "threshold when comparing optical flow and rigid flow ")

flags.DEFINE_string('eval_pose', '', 'pose seq to evaluate')

flags.DEFINE_integer("num_scales", 4, "Number of scales: 1/2^0, 1/2^1, ..., 1/2^(n-1)") #FLAGS.num_scales = 4
flags.DEFINE_boolean('eval_flow', False, '')
flags.DEFINE_boolean('eval_depth', False, '')
flags.DEFINE_boolean('eval_mask', False, '')
opt = FLAGS

def predict_depth_single(sess, eval_model, image1, image1r, K, fxb):
    height, width = image1.shape[:2] # original height, width
    # fit to model input
    img1 = imgtool.imresize(image1, (opt.img_height, opt.img_width))
    img1r = imgtool.imresize(image1r, (opt.img_height, opt.img_width))
    # prepend batch dimension
    img1 = np.expand_dims(img1, axis=0)
    img1r = np.expand_dims(img1r, axis=0)
    # zoom K
    K = scale_intrinsics(K, opt.img_width / width, opt.img_height / height)
    # session run
    pred_disp, _ = sess.run(
        [eval_model.pred_disp, eval_model.pred_mask],
        feed_dict = {
            eval_model.input_1: img1, eval_model.input_2: img1,
            eval_model.input_r: img1r, eval_model.input_2r: img1r, 
            eval_model.input_intrinsic: K})
    # depth from disparity
    pred_disp = np.squeeze(pred_disp)
    pred_disp = width * cv2.resize(pred_disp, (width, height))
    pred_depth = fxb / pred_disp
    return pred_depth

def predict_depth_single_gt_2015(sess, eval_model, i):
    gt_dir = opt.gt_2015_dir
    img1 = imgtool.imread(os.path.join(gt_dir, "image_2", str(i).zfill(6) + "_10.png"))
    img1r = imgtool.imread(os.path.join(gt_dir, "image_3", str(i).zfill(6) + "_10.png"))
    K = get_scaled_intrinsic_matrix(os.path.join(gt_dir, "calib_cam_to_cam", str(i).zfill(6) + ".txt"), 1.0, 1.0)
    fxb = width_to_focal[img1.shape[1]] * 0.54
    depth = predict_depth_single(sess, eval_model, img1, img1r, K, fxb)
    # import time
    # time0 = time.time()
    # for _ in range(1000):
    #     predict_depth_single(sess, eval_model, img1, img1r, K, fxb)
    # time1 = time.time()
    # print("**** 1000 elapsed:", time1 - time0)
    return img1, depth, K

def create_axis_bar():
    LEN, DIV, RADIUS = 20, 1, 0.2
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

def main(unused_argv):
    from datafind import kitti_data_find
    #VICTECH stereo test
    kitti_data_find()
    FLAGS.mode = 'stereo'
    FLAGS.pretrained_model = './stereo_results/model-297503'
    #VICTECH

    print('Constructing models and inputs.')

    if FLAGS.mode == "depthflow":  # stage 3: train depth and flow together
        Model = Model_depthflow
        Model_eval = Model_eval_depthflow

        opt.eval_flow = True
        opt.eval_depth = True
        opt.eval_mask = True
    elif FLAGS.mode == "depth":  # stage 2: train depth
        Model = Model_depth
        Model_eval = Model_eval_depth

        opt.eval_flow = True
        opt.eval_depth = True
        opt.eval_mask = False
    elif FLAGS.mode == "flow":  # stage 1: train flow
        Model = Model_flow
        Model_eval = Model_eval_flow

        opt.eval_flow = True
        opt.eval_depth = False
        opt.eval_mask = False
    elif FLAGS.mode == "stereo":
        Model = Model_stereo
        Model_eval = Model_eval_stereo

        opt.eval_flow = False
        opt.eval_depth = True
        opt.eval_mask = False
    else:
        raise "mode must be one of flow, depth, depthflow or stereo"

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        image1 = tf.placeholder(tf.float32, [1, opt.img_height, opt.img_width, 3], name='dummy_input_1')
        image1r = tf.placeholder(tf.float32, [1, opt.img_height, opt.img_width, 3], name='dummy_input_1r')
        image2 = tf.placeholder(tf.float32, [1, opt.img_height, opt.img_width, 3], name='dummy_input_2')
        image2r = tf.placeholder(tf.float32, [1, opt.img_height, opt.img_width, 3], name='dummy_input_2r')
        cam2pix = tf.placeholder(tf.float32, [1, 4, 3, 3], name='dummy_cam2pix')
        pix2cam = tf.placeholder(tf.float32, [1, 4, 3, 3], name='dummy_pix2cam')

        with tf.variable_scope(tf.get_variable_scope()) as vs:
            with tf.name_scope("model") as ns:
                model = Model(image1, image2, image1r, image2r, 
                    cam2pix, pix2cam, reuse_scope=False, scope=vs)
                eval_model = Model_eval(scope=vs)

        # Create a saver.
        saver = tf.train.Saver(max_to_keep=10)

        # Make training session.
        config = tf.ConfigProto(device_count={'GPU':0})
        sess = tf.Session(config=config)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, FLAGS.pretrained_model)

        if opt.eval_flow:
            gt_flows_2012, noc_masks_2012 = load_gt_flow_kitti("kitti_2012")
            gt_flows_2015, noc_masks_2015 = load_gt_flow_kitti("kitti")
            gt_masks = load_gt_mask()
        else:
            gt_flows_2012, noc_masks_2012, gt_flows_2015, noc_masks_2015, gt_masks = \
              None, None, None, None, None

        # # evaluate KITTI gt 2012/2015
        # test(sess, eval_model, 0, gt_flows_2012, noc_masks_2012,
        #         gt_flows_2015, noc_masks_2015, gt_masks)

        # # show depth 
        # depth /= 10; depth = np.clip(depth, 0, 1)
        # cv2.imshow('depth', depth)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # point cloud test of KITTI 2015 gt
        axis_bar = create_axis_bar()
        for i in range(200):
            img, depth, K = predict_depth_single_gt_2015(sess, eval_model, i)
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
            o3d.visualization.draw_geometries([pcd] + axis_bar)


if __name__ == '__main__':
    app.run()
