import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, UpSampling2D, AveragePooling2D
from tensorflow.keras import Sequential, Model, Input
import tensorflow_addons as tfa
import functools

from opt_helper import opt
from loss_utils import charbonnier_loss
from pcdlib import tf_populate_pcd, tf_detect_plane_xz

_leaky_relu = functools.partial(tf.nn.leaky_relu, alpha=0.1)
_reg = tf.keras.regularizers.l2(opt.weight_decay)
_conv2d = functools.partial(Conv2D, padding='same', activation=_leaky_relu, kernel_regularizer=_reg)

class FeaturePyramid(Layer):
    def __init__(self, *args, **kwargs):
        super(FeaturePyramid, self).__init__(*args, **kwargs)
        self._cnvs = []
        self._cnvs.append(_conv2d(16, 3, 2, name='cnv1'))
        self._cnvs.append(_conv2d(16, 3, 1, name='cnv2'))
        self._cnvs.append(_conv2d(32, 3, 2, name='cnv3'))
        self._cnvs.append(_conv2d(32, 3, 1, name='cnv4'))
        self._cnvs.append(_conv2d(64, 3, 2, name='cnv5'))
        self._cnvs.append(_conv2d(64, 3, 1, name='cnv6'))
        self._cnvs.append(_conv2d(96, 3, 2, name='cnv7'))
        self._cnvs.append(_conv2d(96, 3, 1, name='cnv8'))
        self._cnvs.append(_conv2d(128, 3, 2, name='cnv9'))
        self._cnvs.append(_conv2d(128, 3, 1, name='cnv10'))
        self._cnvs.append(_conv2d(192, 3, 2, name='cnv11'))
        self._cnvs.append(_conv2d(192, 3, 1, name='cnv12'))

    def call(self, inputs):
        x = [inputs]
        for cnv in self._cnvs:
            x.append(cnv(x[-1]))
        return x[2], x[4], x[6], x[8], x[10], x[12]


class DispDecoder(Layer):
    def __init__(self, *args, **kwargs):
        super(DispDecoder, self).__init__(*args, **kwargs)
        self._cnv1 = _conv2d(128, 3, 1, name=f'cnv1')
        self._cnv2 = _conv2d(128, 3, 1, name=f'cnv2')
        self._cnv3 = _conv2d(96, 3, 1, name=f'cnv3')
        self._cnv4 = _conv2d(64, 3, 1, name=f'cnv4')
        self._cnv5 = _conv2d(32, 3, 1, name=f'cnv5')
        self._cnv6 = _conv2d(1, 3, 1, activation=None, name=f'cnv6')

    def call(self, inputs):
        out1 = self._cnv1(inputs)
        out2 = self._cnv2(out1)
        out3 = self._cnv3(tf.concat([out1, out2], axis=3))
        out4 = self._cnv4(tf.concat([out2, out3], axis=3))
        out5 = self._cnv5(tf.concat([out3, out4], axis=3))
        flow_x = self._cnv6(tf.concat([out4, out5], axis=3))
        flow_y = tf.zeros_like(flow_x)
        flow = tf.concat([flow_y, flow_x], axis=3)
        return flow, out5


class ContextNet(Layer):
    def __init__(self, *args, **kwargs):
        super(ContextNet, self).__init__(*args, **kwargs)
        self._cnvs = []
        self._cnvs.append(_conv2d(128, 3, 1, dilation_rate=1, name='cnv1'))
        self._cnvs.append(_conv2d(128, 3, 1, dilation_rate=2, name='cnv2'))
        self._cnvs.append(_conv2d(128, 3, 1, dilation_rate=4, name='cnv3'))
        self._cnvs.append(_conv2d(96, 3, 1, dilation_rate=8,  name='cnv4'))
        self._cnvs.append(_conv2d(64, 3, 1, dilation_rate=16, name='cnv5'))
        self._cnvs.append(_conv2d(32, 3, 1, dilation_rate=1, name='cnv6'))
        self._cnvs.append(_conv2d(1, 3, 1, dilation_rate=1, activation=None, name=f'cnv7'))

    def call(self, inputs):
        out = inputs
        for cnv in self._cnvs:
            out = cnv(out)
        flow_x, flow_y = out, tf.zeros_like(out)
        return tf.concat([flow_y, flow_x], axis=3)


class PwcNet_Single(Layer):
    def __init__(self, neg, *args, **kwargs):
        super(PwcNet_Single, self).__init__(*args, **kwargs)
        self.neg = neg
        self._cn = ContextNet(name='cn')
        self._dec2 = DispDecoder(name='dec2')
        self._dec3 = DispDecoder(name='dec3')
        self._dec4 = DispDecoder(name='dec4')
        self._dec5 = DispDecoder(name='dec5')
        self._dec6 = DispDecoder(name='dec6')

    @staticmethod
    def _cost_volumn(feature1, feature2, d=4):
        W = feature1.shape[2]
        feature2 = tf.pad(feature2, [[0, 0], [0, 0], [d, d], [0, 0]], "CONSTANT")
        cv = []
        for j in range(2 * d + 1):
            cv.append(tf.reduce_mean(
                feature1 * feature2[:, :, j:(j + W), :],
                axis=3, keepdims=True))
        return tf.concat(cv, axis=3)

    # def _cost_volumn2(self, feature1, feature2, d=4):
    #     #feature2 = tf.pad(feature2, [[0, 0], [0, 0], [d, d], [0, 0]], "CONSTANT")
    #     cv = tfa.layers.CorrelationCost(1, d, 1, 1, d, 'channels_last')([feature1, feature2])
    #     return cv

    @staticmethod
    def _inv_warp_flow(image, flow):
        return tfa.image.dense_image_warp(image, -flow)

    def call(self, inputs):
        _, feature1_2, feature1_3, feature1_4, feature1_5, feature1_6 = inputs[:6]
        _, feature2_2, feature2_3, feature2_4, feature2_5, feature2_6 = inputs[6:]

        upsampling_x2 = UpSampling2D(size=2, interpolation='bilinear')

        cv6 = self._cost_volumn(feature1_6, feature2_6, d=4)
        flow6, _ = self._dec6(cv6)
        if self.neg:
            flow6 = -tf.nn.relu(-flow6)
        else:
            flow6 = tf.nn.relu(flow6)

        flow6to5 = upsampling_x2(flow6) * 2.0
        feature2_5w = self._inv_warp_flow(feature2_5, flow6to5)
        cv5 = self._cost_volumn(feature1_5, feature2_5w, d=4)
        flow5, _ = self._dec5(tf.concat([cv5, feature1_5, flow6to5], axis=3))
        flow5 = flow5 + flow6to5
        if self.neg:
            flow5 = -tf.nn.relu(-flow5)
        else:
            flow5 = tf.nn.relu(flow5)

        flow5to4 = upsampling_x2(flow5) * 2.0
        feature2_4w = self._inv_warp_flow(feature2_4, flow5to4)
        cv4 = self._cost_volumn(feature1_4, feature2_4w, d=4)
        flow4, _ = self._dec4(tf.concat([cv4, feature1_4, flow5to4[:, :, :, 1:2]], axis=3))
        flow4 = flow4 + flow5to4
        if self.neg:
            flow4 = -tf.nn.relu(-flow4)
        else:
            flow4 = tf.nn.relu(flow4)

        flow4to3 = upsampling_x2(flow4) * 2.0
        feature2_3w = self._inv_warp_flow(feature2_3, flow4to3)
        cv3 = self._cost_volumn(feature1_3, feature2_3w, d=4)
        flow3, _ = self._dec3(tf.concat([cv3, feature1_3, flow4to3[:, :, :, 1:2]], axis=3))
        flow3 = flow3 + flow4to3
        if self.neg:
            flow3 = -tf.nn.relu(-flow3)
        else:
            flow3 = tf.nn.relu(flow3)

        flow3to2 = upsampling_x2(flow3) * 2.0
        feature2_2w = self._inv_warp_flow(feature2_2, flow3to2)
        cv2 = self._cost_volumn(feature1_2, feature2_2w, d=4)
        flow2_raw, f2 = self._dec2(tf.concat([cv2, feature1_2, flow3to2[:, :, :, 1:2]], axis=3))
        flow2_raw = flow2_raw + flow3to2
        if self.neg:
            flow2_raw = -tf.nn.relu(-flow2_raw)
        else:
            flow2_raw = tf.nn.relu(flow2_raw)

        flow2 = self._cn(tf.concat([flow2_raw[:, :, :, 1:2], f2], axis=3)) + flow2_raw
        if self.neg:
            flow2 = -tf.nn.relu(-flow2)
        else:
            flow2 = tf.nn.relu(flow2)

        W = opt.img_width
        disp0 = flow2[:, :, :, 1:2] / (W / (2**2)) # 1/4 of input size
        disp1 = flow3[:, :, :, 1:2] / (W / (2**3)) # 1/8 of input size
        disp2 = flow4[:, :, :, 1:2] / (W / (2**4)) # 1/16 of input size
        disp3 = flow5[:, :, :, 1:2] / (W / (2**5)) # 1/32 of input size

        if self.neg:
            return -disp0, -disp1, -disp2, -disp3
        else:
            return disp0, disp1, disp2, disp3


class DispNet(Model):
    def __init__(self, *args, **kwargs):
        super(DispNet, self).__init__(*args, **kwargs)
        self._feat = FeaturePyramid(name='feature_net_disp')
        self._pwcL = PwcNet_Single(True, name='left_disp')
        self._pwcR = PwcNet_Single(False, name='right_disp')

    @staticmethod
    def _scale_pyramid(img, pool_size0, pool_size1, num_scales):
        scaled_imgs = [img if pool_size0 == 1 else AveragePooling2D(pool_size0)(img)]
        downsample1 = AveragePooling2D(pool_size1)
        for _ in range(1, num_scales):
            scaled_imgs.append(downsample1(scaled_imgs[-1]))
        return scaled_imgs

    def call(self, inputs, training=None):
        imgL, imgR = inputs[:2]

        featL = self._feat(imgL)
        featR = self._feat(imgR)
        pred_dispL = self._pwcL(featL + featR)
        pred_dispR = self._pwcR(featR + featL)

        loss = 0.0
        if training == True:
            dispL, dispR = inputs[2:]
            dispL_pyr = self._scale_pyramid(dispL, 4, 2, 4)
            dispR_pyr = self._scale_pyramid(dispR, 4, 2, 4)
            SCALE_FACTOR = [1.0, 0.8, 0.6, 0.4]

            for s in range(4):
                left_pixel_error = opt.img_width * (dispL_pyr[s] - pred_dispL[s])
                right_pixel_error = opt.img_width * (dispR_pyr[s] - pred_dispR[s])
                if s == 0:
                    pixel_error = 0.5 * tf.reduce_mean(tf.abs(left_pixel_error) + tf.abs(right_pixel_error))
                    self.add_metric(pixel_error, name='epe', aggregation='mean')

                if opt.loss_metric == 'l1-log': # l1 of log
                    left_error = tf.abs(tf.math.log(1.0 + dispL_pyr[s]) - tf.math.log(1.0 + pred_dispL[s]))
                    right_error = tf.abs(tf.math.log(1.0 + dispR_pyr[s]) - tf.math.log(1.0 + pred_dispR[s]))
                    loss += SCALE_FACTOR[s] * tf.reduce_mean(left_error + right_error)
                elif opt.loss_metric == 'charbonnier':
                    loss += 0.1 * SCALE_FACTOR[s] * (charbonnier_loss(left_pixel_error) + charbonnier_loss(right_pixel_error))
                else:
                    raise ValueError('! Unsupported loss metric')
        self.add_loss(loss, inputs=True)
            
        if training == True:
            return tuple()
        else:
            return pred_dispL[:1] + pred_dispR[:1]

    @tf.function
    def predict_single(self, imgL, imgR):
        imgL = tf.cast(imgL, tf.float32) / 255
        imgR = tf.cast(imgR, tf.float32) / 255
        dispL, dispR = self([imgL[None], imgR[None]], False)
        return dispL[0], dispR[0]


class TraversabilityDecoder(Layer):
    def __init__(self, neg, *args, **kwargs):
        super(TraversabilityDecoder, self).__init__(*args, **kwargs)

    def call(self, inputs):
        '''
        disp: normalized disparity of shape [B, H//4, W//4, 1] (bottom of pyramid)
        K0: rescaled already from original size to [opt.img_width, opt.img_height]
        '''
        disp, K0, baseline = inputs
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


# if __name__ == '__main__':
#     import cv2
#     import numpy as np
#     from pathlib import Path
#     import time

#     disp_net = DispNet(name='depth_net')
#     imgL0 = np.ones((384, 512, 3), np.uint8)
#     imgR0 = np.ones((384, 512, 3), np.uint8)
#     disp_net.predict_single(imgL0, imgR0)
#     disp_net.load_weights('.results_stereosv/model-tf2')

#     # point cloud test of office image of inbo.yeo 
#     data_dir = Path('M:\\Users\\sehee\\camera_taker\\undist_fisheye')
#     imgnamesL = sorted(Path(data_dir/'imL').glob('*.png'), key=lambda v: int(v.stem))
#     for index in range(len(imgnamesL)):
#         imgnameL = imgnamesL[index % len(imgnamesL)]
#         imgnameR = (data_dir/'imR'/imgnameL.stem).with_suffix('.png')

#         imgL = cv2.cvtColor(cv2.imread(str(imgnameL)), cv2.COLOR_BGR2RGB)
#         imgR = cv2.cvtColor(cv2.imread(str(imgnameR)), cv2.COLOR_BGR2RGB)
#         imgL = cv2.resize(imgL, (opt.img_width, opt.img_height), interpolation=cv2.INTER_LINEAR)
#         imgR = cv2.resize(imgR, (opt.img_width, opt.img_height), interpolation=cv2.INTER_LINEAR)

#         t0 = time.time()
#         dispL, _ = disp_net.predict_single(imgL, imgR)
#         t1 = time.time()
#         print("* elspaed:", t1 - t0)
#         imgtool.imshow(dispL.numpy())
#         #imgtool.imshow(disp0)
