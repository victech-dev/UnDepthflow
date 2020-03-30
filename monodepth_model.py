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
        scaled_imgs = [img]
        s = tf.shape(img)
        h = s[1]
        w = s[2]
        for i in range(num_scales - 1):
            ratio = 2**(i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
        return scaled_imgs

    def generate_flow_left(self, disp, scale):
        batch_size = opt.batch_size
        H = opt.img_height // (2**scale)
        W = opt.img_width // (2**scale)
        zero_flow = tf.zeros([batch_size, H, W, 1])
        ltr_flow = -disp * W
        ltr_flow = tf.concat([ltr_flow, zero_flow], axis=3)
        return ltr_flow

    def generate_flow_right(self, disp, scale):
        return self.generate_flow_left(-disp, scale)

    def generate_transformed(self, img, flow, scale):
        return inv_warp_flow(img, flow)

    ''' undepthflow (with disp norm, index bug fixed) '''
    def get_disparity_smoothness_2nd(self, disp, pyramid):
        LOSS_COEFFS = [1.0, 0.28772402, 0.10591773, 0.04457521]
        WEIGHT_COEFFS = [1.0, 0.7, 0.5, 0.33]
        def _loss(s, gray): # Iyy,Ixx
            g = tf.concat(tf.image.image_gradients(gray), axis=-1) # [B,H,W,(dy,dx)]
            iyy = tf.image.image_gradients(g[:,:,:,0:1])[0]
            ixx = tf.image.image_gradients(g[:,:,:,1:2])[1]
            iyyixx = tf.concat([tf.roll(iyy, 1, axis=1), tf.roll(ixx, 1, axis=2)], axis=-1)
            return LOSS_COEFFS[s]*iyyixx
        def _weights(s, img): # Iy,Ix
            g = tf.stack(tf.image.image_gradients(img), axis=-1) # [B,H,W,C,(dy,dx)]
            g = tf.reduce_mean(tf.abs(g), axis=3) # [B,H,W,(dy,dx)]
            return tf.exp(-10*WEIGHT_COEFFS[s]*g)
        disp_mean = [tf.reduce_mean(d, axis=[1,2], keepdims=True) for d in disp]
        disp = [d / m for d, m in zip(disp, disp_mean)]
        loss = [_loss(s, d) for s, d in enumerate(disp)]
        weights = [_weights(s, img) for s, img in enumerate(pyramid)]
        return [l*w for l, w in zip(loss, weights)] # [batch, height, width, 2]

    ''' monodepth2 version (with disp norm) '''
    def get_disparity_smoothness_monodepth2(self, disp, pyramid):
        # same as monodepth, index bug fixed
        LOSS_COEFFS = [1.0, 0.46514253, 0.22323016, 0.10952386]
        WEIGHT_COEFFS = [1.0, 0.6, 0.4, 0.3]
        def _loss(s, gray): # Iy,Ix vanila gradient
            g = tf.concat(tf.image.image_gradients(gray), axis=-1) # [B,H,W,(dy,dx)]
            return LOSS_COEFFS[s]*g
        def _weights(s, img): # Iy,Ix vanila gradient
            g = tf.stack(tf.image.image_gradients(img), axis=-1) # [B,H,W,C,(dy,dx)]
            g = tf.reduce_mean(tf.abs(g), axis=3) # [B,H,W,(dy,dx)]
            return tf.exp(-WEIGHT_COEFFS[s]*g)
        disp_mean = [tf.reduce_mean(d, axis=[1,2], keepdims=True) for d in disp]
        disp = [d / m for d, m in zip(disp, disp_mean)]
        loss = [_loss(s, d) for s, d in enumerate(disp)]
        weights = [_weights(s, img) for s, img in enumerate(pyramid)]
        return [l*w for l, w in zip(loss, weights)] # [batch, height, width, 2]

    ''' depth smoothness version of undepthflow (with depth norm) '''
    def get_disparity_smoothness_2nd_v2(self, disp, pyramid):
        LOSS_COEFFS = [1.0, 0.32506809, 0.13110368, 0.06063714]
        WEIGHT_COEFFS = [1.0, 0.7, 0.5, 0.33]
        def _loss(s, gray): # Iyy,Ixx as Laplacian
            # x3 scale compared to vanila gradient of gradient caused by 3x3 Laplacian filter
            fxx = np.array([[1,-2,1]]*3, dtype=np.float32)
            filters = np.expand_dims(np.stack([fxx.T, fxx], axis=-1), axis=2)
            i_pad = tf.pad(gray, [[0,0],[1,1],[1,1],[0,0]], 'SYMMETRIC')
            iyyixx = tf.nn.conv2d(i_pad, filters, 1, padding='VALID')
            return LOSS_COEFFS[s]*iyyixx # [B,H,W,(dyy,dxx)]
        def _weights(s, img): # Iy,Ix as Sobel
            # x10 scale compared to vanila gradient caused by rgb weight and sobel filter
            rgb_weight = tf.constant([0.897, 1.761, 0.342], dtype=tf.float32)
            sobel = tf.image.sobel_edges(img) # [batch, height, width, 3, 2]
            sobel_weighted = sobel * rgb_weight[None,None,None,:,None]
            sobel_abs = tf.abs(sobel_weighted)
            g = tf.reduce_max(sobel_abs, axis=3) # [batch, height, width, (dy,dx)]
            return tf.exp(-WEIGHT_COEFFS[s]*g)
        depth = [1.0 / (d+1e-3) for d in disp]
        depth_mean = [tf.reduce_mean(d, axis=[1,2], keepdims=True) for d in depth]
        depth = [d / m for d, m in zip(depth, depth_mean)]    
        loss = [_loss(s, d) for s, d in enumerate(depth)]
        weights = [_weights(s, img) for s, img in enumerate(pyramid)]
        return [l*w for l, w in zip(loss, weights)] # [batch, height, width, 2]

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
            self.generate_flow_right(self.disp_right_est[i], i)
            for i in range(4)
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
            self.left_est = [
                self.generate_transformed(self.right_pyramid[i],
                                          self.ltr_flow[i], i)
                for i in range(4)
            ]
            self.right_est = [
                self.generate_transformed(self.left_pyramid[i],
                                          self.rtl_flow[i], i)
                for i in range(4)
            ]

        # LR CONSISTENCY
        with tf.variable_scope('left-right'):
            self.right_to_left_disp = [
                self.generate_transformed(self.disp_right_est[i],
                                          self.ltr_flow[i], i)
                for i in range(4)
            ]
            self.left_to_right_disp = [
                self.generate_transformed(self.disp_left_est[i],
                                          self.rtl_flow[i], i)
                for i in range(4)
            ]

        # DISPARITY SMOOTHNESS
        with tf.variable_scope('smoothness'):
            if opt.smooth_mode == 'monodepth2':
                self.disp_left_smoothness = self.get_disparity_smoothness_monodepth2(
                    self.disp_left_est, self.left_pyramid)
                self.disp_right_smoothness = self.get_disparity_smoothness_monodepth2(
                    self.disp_right_est, self.right_pyramid)
            elif opt.smooth_mode == 'undepthflow':
                self.disp_left_smoothness = self.get_disparity_smoothness_2nd(
                    self.disp_left_est, self.left_pyramid)
                self.disp_right_smoothness = self.get_disparity_smoothness_2nd(
                    self.disp_right_est, self.right_pyramid)
            elif opt.smooth_mode == 'sehee':
                self.disp_left_smoothness = self.get_disparity_smoothness_2nd_v2(
                    self.disp_left_est, self.left_pyramid)
                self.disp_right_smoothness = self.get_disparity_smoothness_2nd_v2(
                    self.disp_right_est, self.right_pyramid)

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
            self.ssim_left = [
                tf.image.ssim(est * mask, tgt * mask, 1.0, filter_size=3, filter_sigma=256)
                for est, tgt, mask in zip(self.left_est, self.left_pyramid, self.left_occ_mask)]
            self.ssim_loss_left = [
                tf.reduce_mean(tf.clip_by_value(0.5 * (1 - ssim), 0, 1)) / denom
                for ssim, denom in zip(self.ssim_left, self.left_occ_mask_avg)]

            self.ssim_right = [
                tf.image.ssim(est * mask, tgt * mask, 1.0, filter_size=3, filter_sigma=256)
                for est, tgt, mask in zip(self.right_est, self.right_pyramid, self.right_occ_mask)]
            self.ssim_loss_right = [
                tf.reduce_mean(tf.clip_by_value(0.5 * (1 - ssim), 0, 1)) / denom
                for ssim, denom in zip(self.ssim_right, self.right_occ_mask_avg)]

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
            self.total_loss = self.image_loss + opt.depth_smooth_weight * self.disp_gradient_loss + 1.0 * self.lr_loss

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
                        'ssim_left_' + str(i),
                        self.ssim_left[i],
                        max_outputs=4)
                    tf.summary.image(
                        'ssim_right_' + str(i),
                        self.ssim_right[i],
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
    return outputs

