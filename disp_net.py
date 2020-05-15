import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, UpSampling2D, AveragePooling2D, LeakyReLU
from tensorflow.keras import Sequential, Model, Input
import tensorflow_addons as tfa
import functools

from opt_helper import opt
from losses import charbonnier_loss

_reg = tf.keras.regularizers.l2(opt.weight_decay)
_conv2d = functools.partial(Conv2D, padding='same', kernel_regularizer=_reg)

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
        act = LeakyReLU(0.1)
        out = [inputs]
        for cnv in self._cnvs:
            out.append(act(cnv(out[-1])))
        return out[2], out[4], out[6], out[8], out[10], out[12]


class FlowDecoder(Layer):
    def __init__(self, *args, **kwargs):
        super(FlowDecoder, self).__init__(*args, **kwargs)
        self._cnv1 = _conv2d(128, 3, 1, name=f'cnv1')
        self._cnv2 = _conv2d(128, 3, 1, name=f'cnv2')
        self._cnv3 = _conv2d(96, 3, 1, name=f'cnv3')
        self._cnv4 = _conv2d(64, 3, 1, name=f'cnv4')
        self._cnv5 = _conv2d(32, 3, 1, name=f'cnv5')
        self._cnv6 = _conv2d(1, 3, 1, name=f'cnv6')

    def call(self, inputs):
        act = LeakyReLU(0.1)
        out1 = act(self._cnv1(inputs))
        out2 = act(self._cnv2(out1))
        out3 = act(self._cnv3(tf.concat([out1, out2], axis=3)))
        out4 = act(self._cnv4(tf.concat([out2, out3], axis=3)))
        out5 = act(self._cnv5(tf.concat([out3, out4], axis=3)))
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
        self._cnvs.append(_conv2d(1, 3, 1, dilation_rate=1, name=f'cnv7'))

    def call(self, inputs):
        act = LeakyReLU(0.1)
        out = inputs
        for i, cnv in enumerate(self._cnvs):
            out = act(cnv(out)) if i < len(self._cnvs[:-1]) else cnv(out)
        flow_x, flow_y = out, tf.zeros_like(out)
        return tf.concat([flow_y, flow_x], axis=3)


class PwcNet_Single(Layer):
    def __init__(self, *args, **kwargs):
        super(PwcNet_Single, self).__init__(*args, **kwargs)
        self._cn = ContextNet(name='cn')
        self._dec2 = FlowDecoder(name='dec2')
        self._dec3 = FlowDecoder(name='dec3')
        self._dec4 = FlowDecoder(name='dec4')
        self._dec5 = FlowDecoder(name='dec5')
        self._dec6 = FlowDecoder(name='dec6')

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

        neg = 'left' in self.name
        upsampling_x2 = UpSampling2D(size=2, interpolation='bilinear')

        cv6 = self._cost_volumn(feature1_6, feature2_6, d=4)
        flow6, _ = self._dec6(cv6)
        if neg:
            flow6 = -tf.nn.relu(-flow6)
        else:
            flow6 = tf.nn.relu(flow6)

        flow6to5 = upsampling_x2(flow6) * 2.0
        feature2_5w = self._inv_warp_flow(feature2_5, flow6to5)
        cv5 = self._cost_volumn(feature1_5, feature2_5w, d=4)
        flow5, _ = self._dec5(tf.concat([cv5, feature1_5, flow6to5], axis=3))
        flow5 = flow5 + flow6to5
        if neg:
            flow5 = -tf.nn.relu(-flow5)
        else:
            flow5 = tf.nn.relu(flow5)

        flow5to4 = upsampling_x2(flow5) * 2.0
        feature2_4w = self._inv_warp_flow(feature2_4, flow5to4)
        cv4 = self._cost_volumn(feature1_4, feature2_4w, d=4)
        flow4, _ = self._dec4(tf.concat([cv4, feature1_4, flow5to4[:, :, :, 1:2]], axis=3))
        flow4 = flow4 + flow5to4
        if neg:
            flow4 = -tf.nn.relu(-flow4)
        else:
            flow4 = tf.nn.relu(flow4)

        flow4to3 = upsampling_x2(flow4) * 2.0
        feature2_3w = self._inv_warp_flow(feature2_3, flow4to3)
        cv3 = self._cost_volumn(feature1_3, feature2_3w, d=4)
        flow3, _ = self._dec3(tf.concat([cv3, feature1_3, flow4to3[:, :, :, 1:2]], axis=3))
        flow3 = flow3 + flow4to3
        if neg:
            flow3 = -tf.nn.relu(-flow3)
        else:
            flow3 = tf.nn.relu(flow3)

        flow3to2 = upsampling_x2(flow3) * 2.0
        feature2_2w = self._inv_warp_flow(feature2_2, flow3to2)
        cv2 = self._cost_volumn(feature1_2, feature2_2w, d=4)
        flow2_raw, f2 = self._dec2(tf.concat([cv2, feature1_2, flow3to2[:, :, :, 1:2]], axis=3))
        flow2_raw = flow2_raw + flow3to2
        if neg:
            flow2_raw = -tf.nn.relu(-flow2_raw)
        else:
            flow2_raw = tf.nn.relu(flow2_raw)

        flow2 = self._cn(tf.concat([flow2_raw[:, :, :, 1:2], f2], axis=3)) + flow2_raw
        if neg:
            flow2 = -tf.nn.relu(-flow2)
        else:
            flow2 = tf.nn.relu(flow2)

        W = opt.img_width
        disp0 = flow2[:, :, :, 1:2] / (W / (2**2)) # 1/4 of input size
        disp1 = flow3[:, :, :, 1:2] / (W / (2**3)) # 1/8 of input size
        disp2 = flow4[:, :, :, 1:2] / (W / (2**4)) # 1/16 of input size
        disp3 = flow5[:, :, :, 1:2] / (W / (2**5)) # 1/32 of input size

        # Note that flow is calculated as signed value, but 
        # we are returning disp as positive value for both left and right
        if neg:
            return -disp0, -disp1, -disp2, -disp3
        else:
            return disp0, disp1, disp2, disp3


def scale_pyramid(img, pool_size0, pool_size1, num_scales):
    scaled_imgs = [img if pool_size0 == 1 else AveragePooling2D(pool_size0)(img)]
    downsample1 = AveragePooling2D(pool_size1)
    for _ in range(1, num_scales):
        scaled_imgs.append(downsample1(scaled_imgs[-1]))
    return scaled_imgs


def create_model(training=False):
    # do we need (static) batch_size=1 for inference model here for memory optimization ??
    imgL = Input(shape=(opt.img_height, opt.img_width, 3), dtype='float32')
    imgR = Input(shape=(opt.img_height, opt.img_width, 3), dtype='float32')

    feat = FeaturePyramid(name='feature_net_disp')
    pwcL = PwcNet_Single(name='left_disp')
    pwcR = PwcNet_Single(name='right_disp')

    featL = feat(imgL)
    featR = feat(imgR)
    pred_dispL = pwcL(featL + featR)
    pred_dispR = pwcR(featR + featL)

    if training == True:
        # loss, metric during training
        dispL = Input(shape=(384, 512, 1), dtype='float32')
        dispR = Input(shape=(384, 512, 1), dtype='float32')
        model = tf.keras.Model([imgL, imgR, dispL, dispR], [])

        dispL_pyr = scale_pyramid(dispL, 4, 2, 4)
        dispR_pyr = scale_pyramid(dispR, 4, 2, 4)
        SCALE_FACTOR = [1.0, 0.8, 0.6, 0.4]
        for s in range(4):
            left_pixel_error = opt.img_width * (dispL_pyr[s] - pred_dispL[s])
            right_pixel_error = opt.img_width * (dispR_pyr[s] - pred_dispR[s])
            if s == 0:
                pixel_error = 0.5 * tf.reduce_mean(tf.abs(left_pixel_error) + tf.abs(right_pixel_error))
                model.add_metric(pixel_error, name='epe', aggregation='mean')

            if opt.loss_metric == 'l1-log': # l1 of log
                left_error = tf.abs(tf.math.log(1.0 + dispL_pyr[s]) - tf.math.log(1.0 + pred_dispL[s]))
                right_error = tf.abs(tf.math.log(1.0 + dispR_pyr[s]) - tf.math.log(1.0 + pred_dispR[s]))
                loss = tf.reduce_mean(left_error + right_error)
            elif opt.loss_metric == 'charbonnier':
                loss = 0.1 * (charbonnier_loss(left_pixel_error) + charbonnier_loss(right_pixel_error))
            else:
                raise ValueError('! Unsupported loss metric')
            model.add_loss(SCALE_FACTOR[s] * loss, inputs=True)
    else:
        model = tf.keras.Model([imgL, imgR], pred_dispL[:1] + pred_dispR[:1])
    return model


if __name__ == '__main__':
    import cv2
    import numpy as np
    from pathlib import Path
    import time
    import utils
    import functools

    disp_net = create_model()
    disp_net.load_weights('.results_stereosv/weights-tf2')
    predict = tf.function(functools.partial(disp_net.call, training=None, mask=None))
 
    # point cloud test of office image of inbo.yeo 
    data_dir = Path('M:\\Users\\sehee\\camera_taker\\undist_fisheye')
    imgnamesL = sorted(Path(data_dir/'imL').glob('*.png'), key=lambda v: int(v.stem))
    for index in range(len(imgnamesL)):
        imgnameL = imgnamesL[index % len(imgnamesL)]
        imgnameR = (data_dir/'imR'/imgnameL.stem).with_suffix('.png')

        imgL = utils.imread(str(imgnameL))
        imgR = utils.imread(str(imgnameR))
        imgL, imgR = utils.resize_image_pairs(imgL, imgR, (opt.img_width, opt.img_height), np.float32)

        t0 = time.time()
        dispL, _ = predict([imgL[None], imgR[None]])
        t1 = time.time()
        print("* elspaed:", t1 - t0)
        utils.imshow(dispL[0].numpy())
        #utils.imshow(disp0)
