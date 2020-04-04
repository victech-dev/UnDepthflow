# Copyright UCL Business plc 2017. Patent Pending. All rights reserved. 
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence, 
# please contact info@uclb.com
"""
Adopted from https://github.com/mrharicot/monodepth
Please see LICENSE_monodepth for details
"""

from __future__ import absolute_import, division, print_function
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from autoflags import opt
from nets.pwc_disp import pwc_disp
from core_warp import inv_warp_flow, fwd_warp_flow
from loss_utils import disp_smoothness, SSIM

class MonodepthModel(object):
    """monodepth model"""

    def __init__(self, mode, left, right, left_feature, right_feature, reuse_variables=None):
        self.mode = mode
        self.left = left
        self.right = right
        self.left_feature = left_feature
        self.right_feature = right_feature
        self.reuse_variables = reuse_variables

        self.build_model()
        if self.mode == 'test':
            return

        self.build_outputs()
        self.build_losses()

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

                self.disp1, self.disp2, self.disp3, self.disp4 = pwc_disp(
                    self.left, self.right, self.left_feature, self.right_feature)

    def build_outputs(self):
        # STORE DISPARITIES
        H = opt.img_height
        W = opt.img_width
        with tf.variable_scope('disparities'):
            self.disp_est = [self.disp1, self.disp2, self.disp3, self.disp4]
            self.disp_left_est = [d[:, :, :, 0:1] for d in self.disp_est]
            self.disp_right_est = [d[:, :, :, 1:2] for d in self.disp_est]

        if self.mode == 'test':
            return

        self.ltr_flow = [
            self.generate_flow_left(self.disp_left_est[i], i) for i in range(4)
        ]
        self.rtl_flow = [
            self.generate_flow_right(self.disp_right_est[i], i) for i in range(4)
        ]

        self.right_occ_mask = [tf.clip_by_value(fwd_warp_flow(1, f), 0, 1) for f in self.ltr_flow]
        self.left_occ_mask = [tf.clip_by_value(fwd_warp_flow(1, f), 0, 1) for f in self.rtl_flow]

        self.right_occ_mask_avg = [
            tf.reduce_mean(self.right_occ_mask[i]) + 1e-12 for i in range(4)
        ]
        self.left_occ_mask_avg = [
            tf.reduce_mean(self.left_occ_mask[i]) + 1e-12 for i in range(4)
        ]

        # GENERATE IMAGES
        with tf.variable_scope('images'):
            self.left_est = [inv_warp_flow(img, f) for img, f in zip(self.right_pyramid, self.ltr_flow)]
            self.right_est = [inv_warp_flow(img, f) for img, f in zip(self.left_pyramid, self.rtl_flow)]

        # LR CONSISTENCY
        with tf.variable_scope('left-right'):
            self.right_to_left_disp = [inv_warp_flow(disp, f) for disp, f in zip(self.disp_right_est, self.ltr_flow)]
            self.left_to_right_disp = [inv_warp_flow(disp, f) for disp, f in zip(self.disp_left_est, self.rtl_flow)]

        # DISPARITY SMOOTHNESS
        with tf.variable_scope('smoothness'):
            self.disp_left_smoothness = disp_smoothness(self.disp_left_est, self.left_pyramid)
            self.disp_right_smoothness = disp_smoothness(self.disp_right_est, self.right_pyramid)

    def build_losses(self):
        with tf.variable_scope('losses', reuse=self.reuse_variables):
            # IMAGE RECONSTRUCTION (L1)
            self.l1_left = [
                tf.abs(self.left_est[i] - self.left_pyramid[i]) *
                self.left_occ_mask[i] for i in range(4)
            ]
            self.l1_reconstruction_loss_left = [
                tf.reduce_mean(l) / self.left_occ_mask_avg[i]
                for i, l in enumerate(self.l1_left)
            ]

            self.l1_right = [
                tf.abs(self.right_est[i] - self.right_pyramid[i]) *
                self.right_occ_mask[i] for i in range(4)
            ]
            self.l1_reconstruction_loss_right = [
                tf.reduce_mean(l) / self.right_occ_mask_avg[i]
                for i, l in enumerate(self.l1_right)
            ]

            # SSIM
            self.ssim_loss_left = [tf.reduce_mean(SSIM(x * m, y * m)) / denom
                for x, y, m, denom in zip(self.left_est, self.left_pyramid, self.left_occ_mask, self.left_occ_mask_avg)]
            self.ssim_loss_right = [tf.reduce_mean(SSIM(x * m, y * m)) / denom
                for x, y, m, denom in zip(self.right_est, self.right_pyramid, self.right_occ_mask, self.right_occ_mask_avg)]

            # WEIGTHED SUM
            self.image_loss_right = [
                opt.ssim_weight * self.ssim_loss_right[i] +
                (1 - opt.ssim_weight) * self.l1_reconstruction_loss_right[i] for i in range(4)
            ]
            self.image_loss_left = [
                opt.ssim_weight * self.ssim_loss_left[i] +
                (1 - opt.ssim_weight) * self.l1_reconstruction_loss_left[i] for i in range(4)
            ]
            self.image_loss = tf.add_n(self.image_loss_left + self.image_loss_right)

            # DISPARITY SMOOTHNESS
            self.disp_left_loss = [tf.reduce_mean(tf.abs(s)) for s in self.disp_left_smoothness]
            self.disp_right_loss = [tf.reduce_mean(tf.abs(s)) for s in self.disp_right_smoothness]
            self.disp_gradient_loss = tf.add_n(self.disp_left_loss + self.disp_right_loss)

            # LR CONSISTENCY
            self.lr_left_loss  = [tf.reduce_mean(tf.abs(self.right_to_left_disp[i] - self.disp_left_est[i])*self.left_occ_mask[i]) / \
                                  self.left_occ_mask_avg[i]  for i in range(4)]
            self.lr_right_loss = [tf.reduce_mean(tf.abs(self.left_to_right_disp[i] - self.disp_right_est[i])*self.right_occ_mask[i]) / \
                                  self.right_occ_mask_avg[i] for i in range(4)]
            self.lr_loss = tf.add_n(self.lr_left_loss + self.lr_right_loss)

            # TOTAL LOSS
            self.total_loss = self.image_loss + opt.disp_smooth_weight * self.disp_gradient_loss + opt.lr_loss_weight * self.lr_loss

    def build_summaries(self):
        # SUMMARIES
        with tf.device('/cpu:0'):
            for i in range(4):
                tf.summary.scalar(
                    'ssim_loss_' + str(i),
                    self.ssim_loss_left[i] + self.ssim_loss_right[i])
                tf.summary.scalar(
                    'l1_loss_' + str(i),
                    self.l1_reconstruction_loss_left[i] +
                    self.l1_reconstruction_loss_right[i])
                tf.summary.scalar(
                    'image_loss_' + str(i),
                    self.image_loss_left[i] + self.image_loss_right[i])
                tf.summary.scalar(
                    'disp_gradient_loss_' + str(i),
                    self.disp_left_loss[i] + self.disp_right_loss[i])
                tf.summary.scalar(
                    'lr_loss_' + str(i),
                    self.lr_left_loss[i] + self.lr_right_loss[i])
                tf.summary.image(
                    'disp_left_est_' + str(i),
                    self.disp_left_est[i],
                    max_outputs=4)
                tf.summary.image(
                    'disp_right_est_' + str(i),
                    self.disp_right_est[i],
                    max_outputs=4)
                tf.summary.image(
                    'occ_left_est_' + str(i),
                    self.left_occ_mask[i],
                    max_outputs=4)
                tf.summary.image(
                    'occ_right_est_' + str(i),
                    self.right_occ_mask[i],
                    max_outputs=4)

                if False: #self.params.full_summary:
                    tf.summary.image(
                        'left_est_' + str(i),
                        self.left_est[i],
                        max_outputs=4)
                    tf.summary.image(
                        'right_est_' + str(i),
                        self.right_est[i],
                        max_outputs=4)
                    tf.summary.image(
                        'l1_left_' + str(i),
                        self.l1_left[i],
                        max_outputs=4)
                    tf.summary.image(
                        'l1_right_' + str(i),
                        self.l1_right[i],
                        max_outputs=4)

            if False: #self.params.full_summary:
                tf.summary.image(
                    'left',
                    self.left,
                    max_outputs=4)
                tf.summary.image(
                    'right',
                    self.right,
                    max_outputs=4)

def disp_godard(left_img, right_img, left_feature, right_feature, is_training=True):
    mode = 'train' if is_training else 'test'
    model = MonodepthModel(mode, left_img, right_img, left_feature, right_feature)
    outputs = dict(disp=[model.disp1, model.disp2, model.disp3, model.disp4])
    if is_training:
        outputs['total_loss'] = model.total_loss
        outputs['image_loss'] = model.image_loss
        outputs['disp_gradient_loss'] = model.disp_gradient_loss
        outputs['lr_loss'] = model.lr_loss
        outputs['left_occ_mask'] = model.left_occ_mask
        outputs['right_occ_mask'] = model.right_occ_mask
        outputs['left_pyramid'] = model.left_pyramid
        outputs['right_pyramid'] = model.right_pyramid
    return outputs

