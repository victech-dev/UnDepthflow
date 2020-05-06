from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from opt_utils import opt
from nets.pwc_disp import pwc_disp, feature_pyramid_disp
from core_warp import inv_warp_flow, fwd_warp_flow
from loss_utils import charbonnier_loss
from pcdlib import tf_populate_pcd, tf_detect_plane_xz

class MonodepthModel(object):
    """monodepth model"""

    def __init__(self, mode, left, right, left_feature, right_feature, left_disp, right_disp, reuse_variables=None):
        self.mode = mode
        self.left = left
        self.right = right
        self.left_feature = left_feature
        self.right_feature = right_feature
        self.reuse_variables = reuse_variables

        self.build_model()

        if self.mode == 'train':
            self.build_outputs()

            dispL_pyramid = self.scale_pyramid(left_disp, 4)
            dispR_pyramid = self.scale_pyramid(right_disp, 4)
            SCALE_FACTOR = np.array([1.0, 0.8, 0.6, 0.4])

            loss = 0
            disp_L1_loss = []
            for s in range(4):
                left_pixel_error = opt.img_width * (dispL_pyramid[s] - self.disp_left_est[s])
                right_pixel_error = opt.img_width * (dispR_pyramid[s] - self.disp_right_est[s])
                disp_L1_loss.append(0.5 * (tf.reduce_mean(tf.abs(left_pixel_error)) + tf.reduce_mean(tf.abs(right_pixel_error))))

                if opt.loss_metric == 'l1-log': # l1 of log
                    left_error = tf.abs(tf.log(1.0 + dispL_pyramid[s]) - tf.log(1.0 + self.disp_left_est[s]))
                    right_error = tf.abs(tf.log(1.0 + dispR_pyramid[s]) - tf.log(1.0 + self.disp_right_est[s]))
                    loss += SCALE_FACTOR[s] * (tf.reduce_mean(left_error) + tf.reduce_mean(right_error))
                elif opt.loss_metric == 'charbonnier':
                    loss += 0.1 * SCALE_FACTOR[s] * (charbonnier_loss(left_pixel_error) + charbonnier_loss(right_pixel_error))
                else:
                    raise ValueError('! Unsupported loss metric')

            self.total_loss = loss
            self.disp_L1_loss = disp_L1_loss[0]

    def scale_pyramid(self, img, num_scales):
        downsample = tf.keras.layers.AveragePooling2D(2)
        scaled_imgs = [img]
        for _ in range(1, num_scales):
            scaled_imgs.append(downsample(scaled_imgs[-1]))
        return scaled_imgs

    def generate_flow_left(self, disp, scale):
        W = opt.img_width // (2**scale)
        ltr_flow = -disp * W
        ltr_flow = tf.concat([ltr_flow, tf.zeros_like(ltr_flow)], axis=3)
        return ltr_flow

    def generate_flow_right(self, disp, scale):
        return self.generate_flow_left(-disp, scale)

    def build_model(self):
        with slim.arg_scope(
            [slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            with tf.variable_scope('depth_net', reuse=self.reuse_variables):

                self.left_pyramid = self.scale_pyramid(self.left, 4)
                if self.mode == 'train':
                    self.right_pyramid = self.scale_pyramid(self.right, 4)

                self.model_input = tf.concat([self.left, self.right], 3)

                self.disp1, self.disp2, self.disp3, self.disp4, self.disp_pyr = pwc_disp(
                    self.left, self.right, self.left_feature, self.right_feature)

    def build_outputs(self):
        # STORE DISPARITIES
        H = opt.img_height
        W = opt.img_width
        with tf.variable_scope('disparities'):
            self.disp_est = [self.disp1, self.disp2, self.disp3, self.disp4]
            self.disp_left_est = [d[:, :, :, 0:1] for d in self.disp_est]
            self.disp_right_est = [d[:, :, :, 1:2] for d in self.disp_est]


class Model_stereosv(object):
    def __init__(self,
                 imageL=None,
                 imageR=None,
                 dispL=None,
                 dispR=None,
                 reuse_scope=False,
                 scope=None):

        with tf.variable_scope(scope, reuse=reuse_scope):
            left_feature = feature_pyramid_disp(imageL, reuse=False)
            right_feature = feature_pyramid_disp(imageR, reuse=True)

            model = MonodepthModel('train', imageL, imageR, left_feature, right_feature, dispL, dispR)
            outputs = dict(disp=[model.disp1, model.disp2, model.disp3, model.disp4])

        self.loss = model.total_loss
        self.outputs = dict(stereo=outputs)

        # Create summaries once when multiple models are created in multiple gpu
        if not tf.get_collection(tf.GraphKeys.SUMMARIES, scope=f'stereosv_losses/.*'):
            with tf.name_scope('stereosv_losses/'):
                tf.summary.scalar('total_loss', model.total_loss)
                tf.summary.scalar('disp_L1_loss', model.disp_L1_loss)


class Model_eval_stereosv(object):
    def __init__(self, scope=None):
        with tf.variable_scope(scope, reuse=True):
            input_uint8_L = tf.placeholder(tf.uint8, [1, opt.img_height, opt.img_width, 3], name='raw_input_L')
            input_uint8_R = tf.placeholder(tf.uint8, [1, opt.img_height, opt.img_width, 3], name='raw_input_R')
            K = tf.placeholder(tf.float32, [1, 3, 3], name='cam_intrinsic_input')
            baseline = tf.placeholder(tf.float32, [1], name='baseline_input')

            input_L = tf.image.convert_image_dtype(input_uint8_L, dtype=tf.float32)
            input_R = tf.image.convert_image_dtype(input_uint8_R, dtype=tf.float32)

            featureL_disp = feature_pyramid_disp(input_L, reuse=True)
            featureR_disp = feature_pyramid_disp(input_R, reuse=True)

            model = MonodepthModel('test', input_L, input_R, featureL_disp, featureR_disp, None, None)
            pred_disp = [model.disp1, model.disp2, model.disp3, model.disp4]
            pred_tmap = self.build_traversability_map(model.disp_pyr[:, :, :, 0:1], K, baseline)

        self.input_L = input_uint8_L
        self.input_R = input_uint8_R
        self.input_K = K
        self.input_baseline = baseline

        self.pred_disp = pred_disp[0][:, :, :, 0:1]
        self.pred_tmap = pred_tmap


    def build_traversability_map(self, disp, K0, baseline):
        '''
        disp: normalized disparity of shape [B, H//4, W//4, 1] (bottom of pyramid)
        K0: rescaled already from original size to [opt.img_width, opt.img_height]
        '''
        _, h1, w1, _ = tf.unstack(tf.shape(disp))
        rw = tf.cast(w1, tf.float32) / opt.img_width
        rh = tf.cast(h1, tf.float32) / opt.img_height

        # rescale intrinsic (note K should be rescaled already from original size to [opt.img_width, opt.img_height])
        K_scale = tf.convert_to_tensor([[rw, 0, 0.5*(rw-1)], [0, rh, 0.5*(rh-1)], [0, 0, 1]], dtype=tf.float32)
        K1 = K_scale[None,:,:] @ K0

        # construct point cloud
        fxb = K1[:,0,0] * baseline
        depth = fxb[:,None,None,None] / (tf.cast(w1, tf.float32) * disp)
        xyz = tf_populate_pcd(depth, K1)
        plane_xz = tf_detect_plane_xz(xyz)

        # Condition 1: thresh below camera
        cond1 = tf.cast(xyz[:,:,:,1] > 0.3, tf.float32) 
        # Condition 2: y component of normal vector
        cond2 = tf.cast(plane_xz > 0.85, tf.float32)
        return cond1 * cond2

