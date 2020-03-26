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

from nets.pwc_disp import pwc_disp
from core_warp import inv_warp_flow
from optical_flow_warp_fwd import transformerFwd

#DEBUG!!!!
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS
#DEBUG!!!!

class MonodepthModel(object):
    """monodepth model"""

    def __init__(self,
                 params,
                 mode,
                 left,
                 right,
                 left_feature,
                 right_feature,
                 reuse_variables=None,
                 model_index=0):
        self.params = params
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
        # 

        self.build_losses()

        with tf.name_scope('monodepth'):
            tf.summary.scalar('image_loss', self.image_loss)
            tf.summary.scalar('disp_gradient_loss', self.disp_gradient_loss)
            tf.summary.scalar('lr_loss', self.lr_loss)

#         self.build_summaries()     

    def gradient_x(self, img):
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx

    def gradient_y(self, img):
        gy = img[:, :-1, :, :] - img[:, 1:, :, :]
        return gy

    def upsample_nn(self, x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

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
        batch_size = self.params.batch_size
        H = self.params.height // (2**scale)
        W = self.params.width // (2**scale)
        zero_flow = tf.zeros([batch_size, H, W, 1])
        ltr_flow = -disp * W
        ltr_flow = tf.concat([ltr_flow, zero_flow], axis=3)
        return ltr_flow

    def generate_flow_right(self, disp, scale):
        return self.generate_flow_left(-disp, scale)

    def generate_transformed(self, img, flow, scale):
        return inv_warp_flow(img, flow)

    def SSIM(self, x, y):
        C1 = 0.01**2
        C2 = 0.03**2

        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

        sigma_x = slim.avg_pool2d(x**2, 3, 1, 'VALID') - mu_x**2
        sigma_y = slim.avg_pool2d(y**2, 3, 1, 'VALID') - mu_y**2
        sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

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

    def get_disp(self, x):
        disp = 0.3 * self.conv(x, 2, 3, 1, tf.nn.sigmoid)
        return disp

    def conv(self,
             x,
             num_out_layers,
             kernel_size,
             stride,
             activation_fn=tf.nn.elu):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(
            p_x,
            num_out_layers,
            kernel_size,
            stride,
            'VALID',
            activation_fn=activation_fn)

    def conv_block(self, x, num_out_layers, kernel_size):
        conv1 = self.conv(x, num_out_layers, kernel_size, 1)
        conv2 = self.conv(conv1, num_out_layers, kernel_size, 2)
        return conv2

    def maxpool(self, x, kernel_size):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.max_pool2d(p_x, kernel_size)

    def resconv(self, x, num_layers, stride):
        do_proj = tf.shape(x)[3] != num_layers or stride == 2
        shortcut = []
        conv1 = self.conv(x, num_layers, 1, 1)
        conv2 = self.conv(conv1, num_layers, 3, stride)
        conv3 = self.conv(conv2, 4 * num_layers, 1, 1, None)
        if do_proj:
            shortcut = self.conv(x, 4 * num_layers, 1, stride, None)
        else:
            shortcut = x
        return tf.nn.elu(conv3 + shortcut)

    def resblock(self, x, num_layers, num_blocks):
        out = x
        for i in range(num_blocks - 1):
            out = self.resconv(out, num_layers, 1)
        out = self.resconv(out, num_layers, 2)
        return out

    def upconv(self, x, num_out_layers, kernel_size, scale):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_out_layers, kernel_size, 1)
        return conv

    def deconv(self, x, num_out_layers, kernel_size, scale):
        p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale,
                                     'SAME')
        return conv[:, 3:-1, 3:-1, :]

    def build_pwc(self):
        self.disp1, self.disp2, self.disp3, self.disp4 = pwc_disp(
            self.left, self.right, self.left_feature, self.right_feature)

    def build_vgg(self):
        #set convenience functions
        conv = self.conv
        if self.params.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv

        with tf.variable_scope('encoder'):
            conv1 = self.conv_block(self.model_input, 32, 7)  # H/2
            conv2 = self.conv_block(conv1, 64, 5)  # H/4
            conv3 = self.conv_block(conv2, 128, 3)  # H/8
            conv4 = self.conv_block(conv3, 256, 3)  # H/16
            conv5 = self.conv_block(conv4, 512, 3)  # H/32
            conv6 = self.conv_block(conv5, 512, 3)  # H/64
            conv7 = self.conv_block(conv6, 512, 3)  # H/128

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = conv2
            skip3 = conv3
            skip4 = conv4
            skip5 = conv5
            skip6 = conv6

        with tf.variable_scope('decoder'):
            upconv7 = upconv(conv7, 512, 3, 2)  #H/64
            concat7 = tf.concat([upconv7, skip6], 3)
            iconv7 = conv(concat7, 512, 3, 1)

            upconv6 = upconv(iconv7, 512, 3, 2)  #H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6 = conv(concat6, 512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2)  #H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5 = conv(concat5, 256, 3, 1)

            upconv4 = upconv(iconv5, 128, 3, 2)  #H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4 = conv(concat4, 128, 3, 1)
            self.disp4 = self.get_disp(iconv4)
            udisp4 = self.upsample_nn(self.disp4, 2)

            upconv3 = upconv(iconv4, 64, 3, 2)  #H/4
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3 = conv(concat3, 64, 3, 1)
            self.disp3 = self.get_disp(iconv3)
            udisp3 = self.upsample_nn(self.disp3, 2)

            upconv2 = upconv(iconv3, 32, 3, 2)  #H/2
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2 = conv(concat2, 32, 3, 1)
            self.disp2 = self.get_disp(iconv2)
            udisp2 = self.upsample_nn(self.disp2, 2)

            upconv1 = upconv(iconv2, 16, 3, 2)  #H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1 = conv(concat1, 16, 3, 1)
            self.disp1 = self.get_disp(iconv1)

    def build_resnet50(self):
        #set convenience functions
        conv = self.conv
        if self.params.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv

        with tf.variable_scope('encoder'):
            conv1 = conv(self.model_input, 64, 7, 2)  # H/2  -   64D
            pool1 = self.maxpool(conv1, 3)  # H/4  -   64D
            conv2 = self.resblock(pool1, 64, 3)  # H/8  -  256D
            conv3 = self.resblock(conv2, 128, 4)  # H/16 -  512D
            conv4 = self.resblock(conv3, 256, 6)  # H/32 - 1024D
            conv5 = self.resblock(conv4, 512, 3)  # H/64 - 2048D

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4

        # DECODING
        with tf.variable_scope('decoder'):
            upconv6 = upconv(conv5, 512, 3, 2)  #H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6 = conv(concat6, 512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2)  #H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5 = conv(concat5, 256, 3, 1)

            upconv4 = upconv(iconv5, 128, 3, 2)  #H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4 = conv(concat4, 128, 3, 1)
            self.disp4 = self.get_disp(iconv4)
            udisp4 = self.upsample_nn(self.disp4, 2)

            upconv3 = upconv(iconv4, 64, 3, 2)  #H/4
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3 = conv(concat3, 64, 3, 1)
            self.disp3 = self.get_disp(iconv3)
            udisp3 = self.upsample_nn(self.disp3, 2)

            upconv2 = upconv(iconv3, 32, 3, 2)  #H/2
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2 = conv(concat2, 32, 3, 1)
            self.disp2 = self.get_disp(iconv2)
            udisp2 = self.upsample_nn(self.disp2, 2)

            upconv1 = upconv(iconv2, 16, 3, 2)  #H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1 = conv(concat1, 16, 3, 1)
            self.disp1 = self.get_disp(iconv1)

    def build_model(self):
        with slim.arg_scope(
            [slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            with tf.variable_scope('depth_net', reuse=self.reuse_variables):

                self.left_pyramid = self.scale_pyramid(self.left, 4)
                if self.mode == 'train':
                    self.right_pyramid = self.scale_pyramid(self.right, 4)

                if self.params.do_stereo:
                    self.model_input = tf.concat([self.left, self.right], 3)
                else:
                    self.model_input = self.left

                #build model
                if self.params.encoder == 'vgg':
                    self.build_vgg()
                elif self.params.encoder == 'resnet50':
                    self.build_resnet50()
                elif self.params.encoder == 'pwc':
                    self.build_pwc()
                else:
                    return None

    def build_outputs(self):
        # STORE DISPARITIES
        H = self.params.height
        W = self.params.width
        with tf.variable_scope('disparities'):
            self.disp_est = [self.disp1, self.disp2, self.disp3, self.disp4]
            self.disp_left_est = [
                tf.expand_dims(d[:, :, :, 0], 3) for d in self.disp_est
            ]
            self.disp_right_est = [
                tf.expand_dims(d[:, :, :, 1], 3) for d in self.disp_est
            ]

        if self.mode == 'test':
            return

        self.ltr_flow = [
            self.generate_flow_left(self.disp_left_est[i], i) for i in range(4)
        ]
        self.rtl_flow = [
            self.generate_flow_right(self.disp_right_est[i], i)
            for i in range(4)
        ]

        self.right_occ_mask = [
            tf.clip_by_value(
                transformerFwd(
                    tf.ones(
                        shape=[
                            self.params.batch_size, H // (2**i), W // (2**i), 1
                        ],
                        dtype='float32'),
                    self.ltr_flow[i], [H // (2**i), W // (2**i)]),
                clip_value_min=0.0,
                clip_value_max=1.0) for i in range(4)
        ]

        self.left_occ_mask = [
            tf.clip_by_value(
                transformerFwd(
                    tf.ones(
                        shape=[
                            self.params.batch_size, H // (2**i), W // (2**i), 1
                        ],
                        dtype='float32'),
                    self.rtl_flow[i], [H // (2**i), W // (2**i)]),
                clip_value_min=0.0,
                clip_value_max=1.0) for i in range(4)
        ]

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
            if FLAGS.smooth_mode == 'monodepth2':
                self.disp_left_smoothness = self.get_disparity_smoothness_monodepth2(
                    self.disp_left_est, self.left_pyramid)
                self.disp_right_smoothness = self.get_disparity_smoothness_monodepth2(
                    self.disp_right_est, self.right_pyramid)
            elif FLAGS.smooth_mode == 'undepthflow':
                self.disp_left_smoothness = self.get_disparity_smoothness_2nd(
                    self.disp_left_est, self.left_pyramid)
                self.disp_right_smoothness = self.get_disparity_smoothness_2nd(
                    self.disp_right_est, self.right_pyramid)
            elif FLAGS.smooth_mode == 'undepthflow_v2':
                self.disp_left_smoothness = self.get_disparity_smoothness_2nd_v2(
                    self.disp_left_est, self.left_pyramid)
                self.disp_right_smoothness = self.get_disparity_smoothness_2nd_v2(
                    self.disp_right_est, self.right_pyramid)

    def build_losses(self):
        with tf.variable_scope('losses', reuse=self.reuse_variables):
            # IMAGE RECONSTRUCTION
            # L1
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
                self.SSIM(self.left_est[i] * self.left_occ_mask[i],
                          self.left_pyramid[i] * self.left_occ_mask[i])
                for i in range(4)
            ]
            self.ssim_loss_left = [
                tf.reduce_mean(s) / self.left_occ_mask_avg[i]
                for i, s in enumerate(self.ssim_left)
            ]

            self.ssim_right = [
                self.SSIM(self.right_est[i] * self.right_occ_mask[i],
                          self.right_pyramid[i] * self.right_occ_mask[i])
                for i in range(4)
            ]
            self.ssim_loss_right = [
                tf.reduce_mean(s) / self.right_occ_mask_avg[i]
                for i, s in enumerate(self.ssim_right)
            ]

            # WEIGTHED SUM
            self.image_loss_right = [
                self.params.alpha_image_loss * self.ssim_loss_right[i] +
                (1 - self.params.alpha_image_loss
                 ) * self.l1_reconstruction_loss_right[i] for i in range(4)
            ]
            self.image_loss_left = [
                self.params.alpha_image_loss * self.ssim_loss_left[i] +
                (1 - self.params.alpha_image_loss
                 ) * self.l1_reconstruction_loss_left[i] for i in range(4)
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
            self.total_loss = self.image_loss + self.params.disp_gradient_loss_weight * self.disp_gradient_loss + self.params.lr_loss_weight * self.lr_loss

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

                if self.params.full_summary:
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

            if self.params.full_summary:
                tf.summary.image(
                    'left',
                    self.left,
                    max_outputs=4)
                tf.summary.image(
                    'right',
                    self.right,
                    max_outputs=4)

monodepth_parameters = namedtuple('parameters', 'encoder, '
                                  'do_stereo, '
                                  'wrap_mode, '
                                  'use_deconv, '
                                  'alpha_image_loss, '
                                  'disp_gradient_loss_weight, '
                                  'lr_loss_weight, '
                                  'full_summary, height, width, batch_size')


def disp_godard(left_img,
                right_img,
                left_feature,
                right_feature,
                opt,
                is_training=True):
    params = monodepth_parameters(
        encoder="pwc",
        do_stereo=True,
        wrap_mode='border',
        use_deconv=False,
        alpha_image_loss=opt.ssim_weight,
        disp_gradient_loss_weight=opt.depth_smooth_weight,
        lr_loss_weight=1.0,
        full_summary=False,
        height=opt.img_height,
        width=opt.img_width,
        batch_size=int(left_img.get_shape()[0]))
    if is_training:
        model = MonodepthModel(params, "train", left_img, right_img,
                               left_feature, right_feature)
        return [model.disp1, model.disp2, model.disp3,
                model.disp4], model.total_loss
    else:
        model = MonodepthModel(params, "test", left_img, right_img,
                               left_feature, right_feature)
        return [model.disp1, model.disp2, model.disp3, model.disp4]

