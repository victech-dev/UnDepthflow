import tensorflow as tf
from tensorflow.python.platform import app
import os
import numpy as np
import imgtool
import cv2
import open3d as o3d
from pathlib import Path
import re

from opt_utils import opt, autoflags
from eval.evaluate_flow import load_gt_flow_kitti, get_scaled_intrinsic_matrix, scale_intrinsics
from eval.evaluate_mask import load_gt_mask
from eval.evaluation_utils import width_to_focal
from eval.test_model import TestModel
from eval.pcd_utils import NavScene


def predict_depth_vicimg(sess, model, imgnameL, imgnameR):
    imgL = imgtool.imread(imgnameL)
    if imgL.shape[2] == 4: 
        imgL = imgL[:,:,:3]
    imgR = imgtool.imread(imgnameR)
    if imgR.shape[2] == 4: 
        imgR = imgR[:,:,:3]
    K = [320.0, 0.0, 320.0, 0.0, 320.0, 240.0, 0, 0, 1]
    K = np.array(K).reshape(3,3)
    fxb = 320.0 * 0.25 # baseline???

    height, width = imgL.shape[:2] # original height, width
    # session run
    imgL_fit = imgtool.imresize(imgL, (opt.img_height, opt.img_width))
    imgR_fit = imgtool.imresize(imgR, (opt.img_height, opt.img_width))
    disp0 = sess.run(model.pred_disp, feed_dict={model.input_L: imgL_fit[None], model.input_R: imgR_fit[None]})
    pred_disp = disp0[0,:,:,0:1]

    # depth from disparity
    pred_disp = width * cv2.resize(pred_disp, (width, height))
    depth = fxb / pred_disp
    return imgL, depth, K


def main(unused_argv):
    opt.trace = '' # this should be empty because we have no output when testing
    opt.batch_size = 1
    opt.mode = 'stereosv'
    opt.pretrained_model = '.results_stereosv/model-145003'
    Model, Model_eval = autoflags()

    with tf.Graph().as_default(), tf.device('/gpu:0'):
        print('Constructing models and inputs.')
        imageL = tf.placeholder(tf.float32, [1, opt.img_height, opt.img_width, 3], name='dummy_input_L')
        imageR = tf.placeholder(tf.float32, [1, opt.img_height, opt.img_width, 3], name='dummy_input_R')
        dispL = tf.placeholder(tf.float32, [1, opt.img_height, opt.img_width, 3], name='dummy_disp_L')
        dispR = tf.placeholder(tf.float32, [1, opt.img_height, opt.img_width, 3], name='dummy_disp_R')
        with tf.variable_scope(tf.get_variable_scope()) as vs:
            with tf.name_scope("test_model"):
                _ = Model(imageL, imageR, dispL, dispR, reuse_scope=False, scope=vs)
                model = Model_eval(scope=vs)

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

        ''' point cloud test of KITTI 2015 gt '''
        # def ns_feeder(index):
        #     i = index % 200
        #     img1 = imgtool.imread(os.path.join(opt.gt_2015_dir, "image_2", f"{i:06}_10.png"))
        #     img2 = imgtool.imread(os.path.join(opt.gt_2015_dir, "image_2", f"{i:06}_11.png"))
        #     img1r = imgtool.imread(os.path.join(opt.gt_2015_dir, "image_3", f"{i:06}_10.png"))
        #     img2r = imgtool.imread(os.path.join(opt.gt_2015_dir, "image_3", f"{i:06}_10.png"))
        #     K = get_scaled_intrinsic_matrix(os.path.join(opt.gt_2015_dir, "calib_cam_to_cam", str(i).zfill(6) + ".txt"), 1.0, 1.0)
        #     fxb = width_to_focal[img1.shape[1]] * 0.54
        #     depth = model.predict_depth(sess, img1, img2, img1r, img2r, K, fxb)
        #     return img1, np.clip(depth, 0, 100), K

        ''' point cloud test of KITTI raw data '''
        # data_dir = Path('M:\\datasets\\kitti_data\\kitti_raw_data\\2011_09_26\\2011_09_26_drive_0117_sync')
        # imgnamesL = sorted(Path(data_dir/'image_02'/'data').glob('*.png'), key=lambda v: int(v.stem))
        # imgnamesR = sorted(Path(data_dir/'image_03'/'data').glob('*.png'), key=lambda v: int(v.stem))
        # def ns_feeder(index):
        #     img1 = imgtool.imread(imgnamesL[index % len(imgnamesL)])
        #     img1r = imgtool.imread(imgnamesR[index % len(imgnamesR)])
        #     K = get_scaled_intrinsic_matrix(data_dir.parent/'calib_cam_to_cam.txt', 1.0, 1.0)
        #     depth = model.predict_depth(sess, img1, img1, img1r, img1r, K, 720 * 0.54)
        #     return img1, np.clip(depth, 0, 100), K

        ''' point cloud test of office image '''
        # data_dir = Path('M:\\Users\\sehee\\StereoCapture_200316_1400\\seq1')
        # imgnamesL = list(Path(data_dir).glob('*_L.jpg'))
        # def ns_feeder(index):
        #     imgnameL = imgnamesL[index % len(imgnamesL)]
        #     imgnameR = (data_dir/imgnameL.stem.replace('_L', '_R')).with_suffix('.jpg')
        #     img, depth, K = predict_depth_vicimg(sess, model, str(imgnameL), str(imgnameR))
        #     return img, np.clip(depth, 0, 30), K

        ''' point cloud test of UnrealStereo data '''
        data_dir = Path('M:\\datasets\\unrealstereo\\arch1_913')
        imgnamesL = [f for f in (data_dir/'imL').glob('*.png') if not f.stem.startswith('gray')]
        imgnamesL = sorted(imgnamesL, key=lambda v: int(v.stem))
        def ns_feeder(index):
            imgnameL = imgnamesL[index % len(imgnamesL)]
            imgnameR = (data_dir/'imR'/imgnameL.stem).with_suffix('.png')
            img, depth, K = predict_depth_vicimg(sess, model, str(imgnameL), str(imgnameR))
            return img, np.clip(depth, 0, 30), K

        scene = NavScene(ns_feeder)
        scene.run()
        scene.clear()


if __name__ == '__main__':
    app.run()
