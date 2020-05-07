# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf
from tensorflow.layers import Conv2D
import functools

from opt_utils import opt
from core_warp import inv_warp_flow

_leaky_relu = functools.partial(tf.nn.leaky_relu, alpha=0.1)
_l2 = tf.keras.regularizers.l2(opt.weight_decay)
_conv2d = functools.partial(Conv2D, padding='same', activation=_leaky_relu, kernel_initializer='glorot_normal', kernel_regularizer=_l2)


def feature_pyramid_disp(image, reuse):
    with tf.variable_scope('feature_net_disp', reuse=reuse):
        cnv1 = _conv2d(16, 3, 2, name='cnv1')(image)
        cnv2 = _conv2d(16, 3, 1, name='cnv2')(cnv1)
        cnv3 = _conv2d(32, 3, 2, name='cnv3')(cnv2)
        cnv4 = _conv2d(32, 3, 1, name='cnv4')(cnv3)
        cnv5 = _conv2d(64, 3, 2, name='cnv5')(cnv4)
        cnv6 = _conv2d(64, 3, 1, name='cnv6')(cnv5)
        cnv7 = _conv2d(96, 3, 2, name='cnv7')(cnv6)
        cnv8 = _conv2d(96, 3, 1, name='cnv8')(cnv7)
        cnv9 = _conv2d(128, 3, 2, name='cnv9')(cnv8)
        cnv10 = _conv2d(128, 3, 1, name='cnv10')(cnv9)
        cnv11 = _conv2d(192, 3, 2, name='cnv11')(cnv10)
        cnv12 = _conv2d(192, 3, 1, name='cnv12')(cnv11)
        return cnv2, cnv4, cnv6, cnv8, cnv10, cnv12


def cost_volumn(feature1, feature2, d=4):
    batch_size, H, W, feature_num = map(int, feature1.get_shape()[0:4])
    feature2 = tf.pad(feature2, [[0, 0], [0, 0], [d, d], [0, 0]], "CONSTANT")
    cv = []
    for i in range(1):
        for j in range(2 * d + 1):
            cv.append(
                tf.reduce_mean(
                    feature1 * feature2[:, i:(i + H), j:(j + W), :],
                    axis=3,
                    keep_dims=True))
    return tf.concat(cv, axis=3)


def optical_flow_decoder_dc(inputs, level):
    cnv1 = _conv2d(128, 3, 1, name=f'cnv1_fd_{level}')(inputs)
    cnv2 = _conv2d(128, 3, 1, name=f'cnv2_fd_{level}')(cnv1)
    cnv3 = _conv2d(96, 3, 1, name=f'cnv3_fd_{level}')(tf.concat([cnv1, cnv2], axis=3))
    cnv4 = _conv2d(64, 3, 1, name=f'cnv4_fd_{level}')(tf.concat([cnv2, cnv3], axis=3))
    cnv5 = _conv2d(32, 3, 1, name=f'cnv5_fd_{level}')(tf.concat([cnv3, cnv4], axis=3))
    flow_x = _conv2d(1, 3, 1, activation=None, name=f'cnv6_fd_{level}')(tf.concat([cnv4, cnv5], axis=3))
    flow_y = tf.zeros_like(flow_x)
    flow = tf.concat([flow_x, flow_y], axis=3)
    return flow, cnv5


def context_net(inputs):
    cnv1 = _conv2d(128, 3, 1, dilation_rate=1, name="cnv1_cn")(inputs)
    cnv2 = _conv2d(128, 3, 1, dilation_rate=2, name="cnv2_cn")(cnv1)
    cnv3 = _conv2d(128, 3, 1, dilation_rate=4, name="cnv3_cn")(cnv2)
    cnv4 = _conv2d(96, 3, 1, dilation_rate=8,  name="cnv4_cn")(cnv3)
    cnv5 = _conv2d(64, 3, 1, dilation_rate=16, name="cnv5_cn")(cnv4)
    cnv6 = _conv2d(32, 3, 1, dilation_rate=1, name="cnv6_cn")(cnv5)
    flow_x = _conv2d(1, 3, 1, dilation_rate=1, activation=None, name="cnv7_cn")(cnv6)
    flow_y = tf.zeros_like(flow_x)
    flow = tf.concat([flow_x, flow_y], axis=3)
    return flow


def construct_model_pwc_full_disp(feature1, feature2, image1, neg=False):
    batch_size, H, W, color_channels = map(int, image1.get_shape()[0:4])
    upsampling_x2 = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')
    upsampling_x4 = tf.keras.layers.UpSampling2D(size=4, interpolation='bilinear')

    #############################
    feature1_1, feature1_2, feature1_3, feature1_4, feature1_5, feature1_6 = feature1
    feature2_1, feature2_2, feature2_3, feature2_4, feature2_5, feature2_6 = feature2

    cv6 = cost_volumn(feature1_6, feature2_6, d=4)
    flow6, _ = optical_flow_decoder_dc(cv6, level=6)
    if neg:
        flow6 = -tf.nn.relu(-flow6)
    else:
        flow6 = tf.nn.relu(flow6)

    flow6to5 = upsampling_x2(flow6) * 2.0
    feature2_5w = inv_warp_flow(feature2_5, flow6to5)
    cv5 = cost_volumn(feature1_5, feature2_5w, d=4)
    flow5, _ = optical_flow_decoder_dc(
        tf.concat(
            [cv5, feature1_5, flow6to5], axis=3), level=5)
    flow5 = flow5 + flow6to5
    if neg:
        flow5 = -tf.nn.relu(-flow5)
    else:
        flow5 = tf.nn.relu(flow5)

    flow5to4 = upsampling_x2(flow5) * 2.0
    feature2_4w = inv_warp_flow(feature2_4, flow5to4)
    cv4 = cost_volumn(feature1_4, feature2_4w, d=4)
    flow4, _ = optical_flow_decoder_dc(
        tf.concat(
            [cv4, feature1_4, flow5to4[:, :, :, 0:1]], axis=3), level=4)
    flow4 = flow4 + flow5to4
    if neg:
        flow4 = -tf.nn.relu(-flow4)
    else:
        flow4 = tf.nn.relu(flow4)

    flow4to3 = upsampling_x2(flow4) * 2.0
    feature2_3w = inv_warp_flow(feature2_3, flow4to3)
    cv3 = cost_volumn(feature1_3, feature2_3w, d=4)
    flow3, _ = optical_flow_decoder_dc(
        tf.concat(
            [cv3, feature1_3, flow4to3[:, :, :, 0:1]], axis=3), level=3)
    flow3 = flow3 + flow4to3
    if neg:
        flow3 = -tf.nn.relu(-flow3)
    else:
        flow3 = tf.nn.relu(flow3)

    flow3to2 = upsampling_x2(flow3) * 2.0
    feature2_2w = inv_warp_flow(feature2_2, flow3to2)
    cv2 = cost_volumn(feature1_2, feature2_2w, d=4)
    flow2_raw, f2 = optical_flow_decoder_dc(
        tf.concat(
            [cv2, feature1_2, flow3to2[:, :, :, 0:1]], axis=3), level=2)
    flow2_raw = flow2_raw + flow3to2
    if neg:
        flow2_raw = -tf.nn.relu(-flow2_raw)
    else:
        flow2_raw = tf.nn.relu(flow2_raw)

    flow2 = context_net(tf.concat(
        [flow2_raw[:, :, :, 0:1], f2], axis=3)) + flow2_raw
    if neg:
        flow2 = -tf.nn.relu(-flow2)
    else:
        flow2 = tf.nn.relu(flow2)

    disp0 = flow2[:, :, :, 0:1] / (W / (2**2)) # 1/4 of input size
    disp1 = flow3[:, :, :, 0:1] / (W / (2**3)) # 1/8 of input size
    disp2 = flow4[:, :, :, 0:1] / (W / (2**4)) # 1/16 of input size
    disp3 = flow5[:, :, :, 0:1] / (W / (2**5)) # 1/32 of input size

    if neg:
        return -disp0, -disp1, -disp2, -disp3
    else:
        return disp0, disp1, disp2, disp3


def pwc_disp(image1, image2, feature1, feature2):
    min_disp = 1e-6

    with tf.variable_scope('left_disp'):
        ltr_disp = construct_model_pwc_full_disp(
            feature1, feature2, image1, neg=True)

    with tf.variable_scope('right_disp'):
        rtl_disp = construct_model_pwc_full_disp(
            feature2, feature1, image2, neg=False)

    return [
        tf.concat(
            [ltr + min_disp, rtl + min_disp], axis=3)
        for ltr, rtl in zip(ltr_disp, rtl_disp)
    ]
