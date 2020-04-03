import tensorflow as tf
import numpy as np
import cv2
import os

from autoflags import opt
import imgtool
from eval.evaluate_flow import get_scaled_intrinsic_matrix, scale_intrinsics
from eval.evaluation_utils import width_to_focal
from post_processing import edge_aware_upscale
from monodepth_dataloader import get_multi_scale_intrinsics

''' VICTECH
This TestModel is somewhat different with Model_eval_xxx.
Since for debug/test/evaluate purpose, we need to access intermediate 
loss/image/depth/mask tensor of actual training model, 
Model_eval_xxx is not sufficient to do this.
Therefore, we make wrapper class of actual training model and 
use this when we want to dig into the graph.
'''
class TestModel(object):
    def __init__(self, train_model, scope):
        with tf.variable_scope(scope, reuse=False):
            img_shape = [1, opt.img_height, opt.img_width, 3]
            self.image1 = tf.placeholder(tf.float32, img_shape, name='input_image1')
            self.image2 = tf.placeholder(tf.float32, img_shape, name='input_image2')
            self.image1r = tf.placeholder(tf.float32, img_shape, name='input_image1r')
            self.image2r = tf.placeholder(tf.float32, img_shape, name='input_image2r')
            self.intrinsic = tf.placeholder(tf.float32, [3, 3])
            self.cam2pix, self.pix2cam = get_multi_scale_intrinsics(self.intrinsic, opt.num_scales)
            self.cam2pix = tf.expand_dims(self.cam2pix, axis=0)
            self.pix2cam = tf.expand_dims(self.pix2cam, axis=0)
            self.model = train_model(self.image1, self.image2, self.image1r, self.image2r, 
                self.cam2pix, self.pix2cam, reuse_scope=False, scope=scope)
            self.outputs = self.model.outputs

    def __call__(self, sess, img1, img2, img1r, img2r, K, queries=None):
        H0, W0 = img1.shape[0:2]
        img1 = imgtool.imresize(img1, (opt.img_height, opt.img_width))
        img2 = imgtool.imresize(img2, (opt.img_height, opt.img_width))
        img1r = imgtool.imresize(img1r, (opt.img_height, opt.img_width))
        img2r = imgtool.imresize(img2r, (opt.img_height, opt.img_width))

        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)
        img1r = np.expand_dims(img1r, axis=0)
        img2r = np.expand_dims(img2r, axis=0)
        img1 = (img1 / 255).astype(np.float32)
        img2 = (img2 / 255).astype(np.float32)
        img1r = (img1r / 255).astype(np.float32)
        img2r = (img2r / 255).astype(np.float32)

        K = scale_intrinsics(K, opt.img_width / W0, opt.img_height / H0)
        outputs = sess.run(self.model.outputs if queries is None else queries, 
                feed_dict = {self.image1: img1, self.image2: img2, 
                self.image1r: img1r, self.image2r: img2r, self.intrinsic: K})
        return outputs

    def predict_depth(self, sess, image1, image2, image1r, image2r, K, fxb, pp=True):
        height, width = image1.shape[:2] # original height, width
        # session run
        query = self.outputs['stereo']['disp'][0] # [1, H, W, (ltr,rtl)]
        disp0 = self(sess, image1, image2, image1r, image2r, K, query)
        pred_disp = disp0[0,:,:,0:1]
        # depth from disparity
        pred_disp = width * cv2.resize(pred_disp, (width, height))
        pred_depth = fxb / pred_disp
        # post processing (downsample to actual pwc output size and do edge aware upscaling)
        if pp:
            depth_ds = cv2.resize(pred_depth, (opt.img_width//4, opt.img_height//4), interpolation=cv2.INTER_AREA)
            pred_depth = edge_aware_upscale(depth_ds, height, width)
        return pred_depth

    def predict_depth_gt_2015(self, sess, i):
        gt_dir = opt.gt_2015_dir
        img1 = imgtool.imread(os.path.join(gt_dir, "image_2", f"{i:06}_10.png"))
        img2 = imgtool.imread(os.path.join(gt_dir, "image_2", f"{i:06}_11.png"))
        img1r = imgtool.imread(os.path.join(gt_dir, "image_3", f"{i:06}_10.png"))
        img2r = imgtool.imread(os.path.join(gt_dir, "image_3", f"{i:06}_10.png"))
        K = get_scaled_intrinsic_matrix(os.path.join(gt_dir, "calib_cam_to_cam", str(i).zfill(6) + ".txt"), 1.0, 1.0)
        fxb = width_to_focal[img1.shape[1]] * 0.54
        depth = self.predict_depth(sess, img1, img2, img1r, img2r, K, fxb)
        return img1, depth, K

        

