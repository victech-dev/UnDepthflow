import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, UpSampling2D
from tensorflow.keras import Sequential, Model, Input
import tensorflow_addons as tfa
import functools

# from opt_utils import opt
#DEBUG!!!!!!
from collections import namedtuple
Options = namedtuple('Option', 'weight_decay img_height img_width')
opt = Options(1e-4, 384, 512)
#DEBUG!!!!!!

_leaky_relu = functools.partial(tf.nn.leaky_relu, alpha=0.1)
_l2 = tf.keras.regularizers.l2(opt.weight_decay)
_conv2d = functools.partial(Conv2D, padding='same', activation=_leaky_relu, kernel_regularizer=_l2)

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

    def _cost_volumn(self, feature1, feature2, d=4):
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

    def _inv_warp_flow(self, image, flow):
        return tfa.image.dense_image_warp(image, -flow)

    def call(self, inputs):
        feature1_1, feature1_2, feature1_3, feature1_4, feature1_5, feature1_6 = inputs[:6]
        feature2_1, feature2_2, feature2_3, feature2_4, feature2_5, feature2_6 = inputs[6:]

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

    ######@tf.function
    @tf.function(input_signature=[[tf.TensorSpec([None, 384, 512, 3], tf.float32), tf.TensorSpec([None, 384, 512, 3], tf.float32)]])
    def call(self, inputs, training=None):
        print("***** Tracing with args: ", inputs, training)
        imgL, imgR = inputs
        featL = self._feat(imgL)
        featR = self._feat(imgR)
        dispL = self._pwcL(featL + featR)
        dispR = self._pwcR(featR + featL)
        return dispL + dispR

# def disp_net():
#     feat = FeaturePyramid(name='feature_net_disp')
#     pwcL = PwcNet_Single(True, name='left_disp')
#     pwcR = PwcNet_Single(False, name='right_disp')
#     imgL = Input(shape=(384, 512, 3), dtype='float32')
#     imgR = Input(shape=(384, 512, 3), dtype='float32')
#     featL = feat(imgL)
#     featR = feat(imgR)
#     dispL = pwcL(featL + featR)
#     dispR = pwcR(featR + featL)
#     model = Model([imgL, imgR], dispL + dispR)
#     return model


def imshow(img, name='imshow', rgb=True, wait=True, norm=True):
    import numpy as np
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    if len(img.shape)==3 and img.shape[2]==3 and rgb:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if norm and (img.dtype==np.float32 or img.dtype==np.float):
        img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
        img = cv2.convertScaleAbs(img, alpha=255)
    cv2.imshow(name, img)
    if wait:
        cv2.waitKey(0)

if __name__ == '__main__':
    import cv2
    import numpy as np
    from pathlib import Path
    import time

    disp_net = DispNet(name='depth_net')
    disp_net.build([(None, 384, 512, 3), (None, 384, 512, 3)])
    #disp_net = disp_net()
    # imgL0 = np.ones((1, 384, 512, 3), np.float32)
    # imgR0 = np.ones((1, 384, 512, 3), np.float32)
    # disp_net([imgL0, imgR0])

    disp_net.load_weights('.results_stereosv/model-tf2')

    # var_list = disp_net.trainable_variables
    # var_dict = dict([(v.name.split(':')[0], v) for v in var_list])
    # rename(var_dict)
    # disp_net.save_weights('.results_stereosv/model-tf2')

    ''' point cloud test of office image of inbo.yeo '''
    data_dir = Path('M:\\Users\\sehee\\camera_taker\\undist_fisheye')
    imgnamesL = sorted(Path(data_dir/'imL').glob('*.png'), key=lambda v: int(v.stem))
    for index in range(len(imgnamesL)):
        imgnameL = imgnamesL[index % len(imgnamesL)]
        imgnameR = (data_dir/'imR'/imgnameL.stem).with_suffix('.png')

        imgL = cv2.cvtColor(cv2.imread(str(imgnameL)), cv2.COLOR_BGR2RGB)
        imgR = cv2.cvtColor(cv2.imread(str(imgnameR)), cv2.COLOR_BGR2RGB)
        imgL = cv2.resize(imgL, (opt.img_width, opt.img_height), interpolation=cv2.INTER_LINEAR)
        imgR = cv2.resize(imgR, (opt.img_width, opt.img_height), interpolation=cv2.INTER_LINEAR)
        imgL = (imgL / 255).astype(np.float32)
        imgR = (imgR / 255).astype(np.float32)

        t0 = time.time()
        disp0, *_ = disp_net([imgL[None], imgR[None]])
        t1 = time.time()
        print("* elspaed:", t1 - t0)
        disp0 = disp0[0]
        imshow(disp0.numpy())



