import tensorflow as tf
from tensorflow.python.platform import app
import os
import numpy as np
import imgtool
import cv2
import open3d as o3d
from pathlib import Path

from autoflags import opt, autoflags
from eval.evaluate_flow import load_gt_flow_kitti, get_scaled_intrinsic_matrix, scale_intrinsics
from eval.evaluate_mask import load_gt_mask
from eval.evaluation_utils import width_to_focal
from eval.test_model import TestModel
from eval.pcd_utils import NavScene


def predict_depth_vicimg(sess, model, imgnameL, imgnameR):
    imgL = imgtool.imread(imgnameL)
    imgR = imgtool.imread(imgnameR)
    K = [9.5061071654182354e+02, 0.0, 5.8985625846591154e+02, 0.0, 9.5061071654182354e+02, 3.9634126783635918e+02, 0, 0, 1]
    K = np.array(K).reshape(3,3)
    fxb = 9.5061071654182354e+02 / 8.2988120552523057 # Q[3,4]/Q[4,3]
    depth = model.predict_depth(sess, imgL, imgL, imgR, imgR, K, fxb)
    return imgL, depth, K


def main(unused_argv):
    opt.trace = '' # this should be empty because we have no output when testing
    opt.batch_size = 1
    opt.mode = 'stereo'
    opt.pretrained_model = '.results_best/model-stereo'
    Model, _ = autoflags()

    with tf.Graph().as_default(), tf.device('/gpu:0'):
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

        # point cloud test of KITTI 2015 gt
        def ns_feeder(index):
            i = index % 200
            img1 = imgtool.imread(os.path.join(opt.gt_2015_dir, "image_2", f"{i:06}_10.png"))
            img2 = imgtool.imread(os.path.join(opt.gt_2015_dir, "image_2", f"{i:06}_11.png"))
            img1r = imgtool.imread(os.path.join(opt.gt_2015_dir, "image_3", f"{i:06}_10.png"))
            img2r = imgtool.imread(os.path.join(opt.gt_2015_dir, "image_3", f"{i:06}_10.png"))
            K = get_scaled_intrinsic_matrix(os.path.join(opt.gt_2015_dir, "calib_cam_to_cam", str(i).zfill(6) + ".txt"), 1.0, 1.0)
            fxb = width_to_focal[img1.shape[1]] * 0.54
            depth = model.predict_depth(sess, img1, img2, img1r, img2r, K, fxb)
            return img1, np.clip(depth, 0, 100), K

        # point cloud test of office image
        # data_dir = Path('M:\\Users\\sehee\\StereoCapture_200316_1400\\seq1')
        # imgnamesL = list(Path(data_dir).glob('*_L.jpg'))
        # def ns_feeder(index):
        #     imgnameL = imgnamesL[index % len(imgnamesL)]
        #     imgnameR = (data_dir/imgnameL.stem.replace('_L', '_R')).with_suffix('.jpg')
        #     img, depth, K = predict_depth_vicimg(sess, model, str(imgnameL), str(imgnameR))
        #     return img, np.clip(depth, 0, 30), K

        scene = NavScene(ns_feeder)
        scene.run()
        scene.clear()


if __name__ == '__main__':
    app.run()
