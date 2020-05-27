import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, UpSampling2D, AveragePooling2D, Conv2DTranspose
from tensorflow.keras import Sequential, Model, Input
import tensorflow_addons as tfa
import functools

from opt_helper import opt
from losses import charbonnier_loss

_reg = tf.keras.regularizers.l2(opt.weight_decay)

class ConvBlock(Layer):
    def __init__(self, filters, kernel_size, strides, padding='same', dilation_rate=1, linear=False, *args, **kwargs):
        super(ConvBlock, self).__init__(*args, **kwargs)
        self._cnv = Conv2D(filters, kernel_size, strides, padding, 
            dilation_rate=dilation_rate, kernel_initializer='he_normal', kernel_regularizer=_reg)
        self._linear = linear

    def call(self, inputs):
        x = self._cnv(inputs)
        return x if self._linear else tf.nn.leaky_relu(x, 0.1)


class FeaturePyramid(Layer):
    def __init__(self, *args, **kwargs):
        super(FeaturePyramid, self).__init__(*args, **kwargs)
        self._cnvs = []
        filters = [16, 32, 64, 96, 128, 196]
        for i, f in enumerate(filters):
            self._cnvs.append(ConvBlock(f, 3, 2, name=f'cnv_{i+1}_1'))
            self._cnvs.append(ConvBlock(f, 3, 1, name=f'cnv_{i+1}_2'))
            self._cnvs.append(ConvBlock(f, 3, 1, name=f'cnv_{i+1}_3'))

    def call(self, inputs):
        out = [inputs]
        for cnv in self._cnvs:
            out.append(cnv(out[-1]))
        return out[3], out[6], out[9], out[12], out[15], out[18]


class FlowDecoder(Layer):
    def __init__(self, *args, **kwargs):
        super(FlowDecoder, self).__init__(*args, **kwargs)
        self._cnv1 = ConvBlock(128, 3, 1, name=f'cnv1')
        self._cnv2 = ConvBlock(128, 3, 1, name=f'cnv2')
        self._cnv3 = ConvBlock(96, 3, 1, name=f'cnv3')
        self._cnv4 = ConvBlock(64, 3, 1, name=f'cnv4')
        self._cnv5 = ConvBlock(32, 3, 1, name=f'cnv5')
        self._cnv6 = ConvBlock(2, 3, 1, linear=True, name=f'cnv6')

    def call(self, inputs):
        x0 = inputs
        x1 = self._cnv1(x0)
        x2 = self._cnv2(tf.concat([x0, x1], axis=-1))
        x3 = self._cnv3(tf.concat([x1, x2], axis=-1))
        x4 = self._cnv4(tf.concat([x2, x3], axis=-1))
        x5 = self._cnv5(tf.concat([x3, x4], axis=-1))
        upfeat = tf.concat([x4, x5], axis=-1)
        flow = self._cnv6(upfeat)
        return flow, upfeat


class ContextNet(Layer):
    def __init__(self, *args, **kwargs):
        super(ContextNet, self).__init__(*args, **kwargs)
        self._cnvs = []
        self._cnvs.append(ConvBlock(128, 3, 1, dilation_rate=1, name='cnv1'))
        self._cnvs.append(ConvBlock(128, 3, 1, dilation_rate=2, name='cnv2'))
        self._cnvs.append(ConvBlock(128, 3, 1, dilation_rate=4, name='cnv3'))
        self._cnvs.append(ConvBlock(96, 3, 1, dilation_rate=8,  name='cnv4'))
        self._cnvs.append(ConvBlock(64, 3, 1, dilation_rate=16, name='cnv5'))
        self._cnvs.append(ConvBlock(32, 3, 1, dilation_rate=1, name='cnv6'))
        self._cnvs.append(ConvBlock(2, 3, 1, dilation_rate=1, linear=True, name=f'cnv7'))

    def call(self, inputs):
        flow, x = inputs
        for cnv in self._cnvs:
            x = cnv(x) 
        return flow + x


class FlowPyramid(Layer):
    def __init__(self, *args, **kwargs):
        super(PwcNet_Single, self).__init__(*args, **kwargs)
        self._cn = ContextNet(name='cn')
        self._dec2 = FlowDecoder(name='dec2')
        self._dec3 = FlowDecoder(name='dec3')
        self._dec4 = FlowDecoder(name='dec4')
        self._dec5 = FlowDecoder(name='dec5')
        self._dec6 = FlowDecoder(name='dec6')
        self._upflow3 = Conv2DTranspose(2, 4, 2, padding='same', name='upflow3')
        self._upfeat3 = Conv2DTranspose(2, 4, 2, padding='same', name='upfeat3')
        self._upflow4 = Conv2DTranspose(2, 4, 2, padding='same', name='upflow4')
        self._upfeat4 = Conv2DTranspose(2, 4, 2, padding='same', name='upfeat4')
        self._upflow5 = Conv2DTranspose(2, 4, 2, padding='same', name='upflow5')
        self._upfeat5 = Conv2DTranspose(2, 4, 2, padding='same', name='upfeat5')
        self._upflow6 = Conv2DTranspose(2, 4, 2, padding='same', name='upflow6')
        self._upfeat6 = Conv2DTranspose(2, 4, 2, padding='same', name='upfeat6')

    @staticmethod
    def _cost_volumn(feature1, feature2, d=4):
        cv = tfa.layers.CorrelationCost(1, d, 1, 1, d, 'channels_last')([feature1, feature2])
        return tf.nn.leaky_relu(cv, alpha=0.1)
        # _, H, W, _ = tf.unstack(tf.shape(feature1))
        # feature2 = tf.pad(feature2, [[0, 0], [d, d], [d, d], [0, 0]], "CONSTANT")
        # cv = []
        # for i in range(2 * d + 1):
        #     for j in range(2 * d + 1):
        #         cv.append(tf.reduce_mean(
        #             feature1 * feature2[:, i:(i+H), j:(j+W), :],
        #             axis=3, keepdims=True))
        # return tf.concat(cv, axis=3)

    @staticmethod
    def _inv_warp_flow(image, flow):
        return tfa.image.dense_image_warp(image, -flow)

    def call(self, inputs):
        _, feature1_2, feature1_3, feature1_4, feature1_5, feature1_6 = inputs[:6]
        _, feature2_2, feature2_3, feature2_4, feature2_5, feature2_6 = inputs[6:]

        cv6 = self._cost_volumn(feature1_6, feature2_6, d=4)
        flow6, feat6 = self._dec6(cv6)
        upflow6 = self._upflow6(flow6) * 0.625
        upfeat6 = self._upfeat6(feat6)

        feature2_5w = self._inv_warp_flow(feature2_5, upflow6)
        cv5 = self._cost_volumn(feature1_5, feature2_5w, d=4)
        flow5, feat5 = self._dec5(tf.concat([cv5, feature1_5, upflow6, upfeat6], axis=-1))
        upflow5 = self._upflow5(flow5) * 1.25
        upfeat5 = self._upfeat5(feat5)

        feature2_4w = self._inv_warp_flow(feature2_4, upflow5)
        cv4 = self._cost_volumn(feature1_4, feature2_4w, d=4)
        flow4, feat4 = self._dec4(tf.concat([cv4, feature1_4, upflow5, upfeat5], axis=-1))
        upflow4 = self._upflow4(flow4) * 2.5
        upfeat4 = self._upfeat4(feat4)

        feature2_3w = self._inv_warp_flow(feature2_3, upflow4)
        cv3 = self._cost_volumn(feature1_3, feature2_3w, d=4)
        flow3, feat3 = self._dec3(tf.concat([cv3, feature1_3, upflow4, upfeat4], axis=3))
        upflow3 = self._upflow3(flow3) * 5
        upfeat3 = self._upfeat3(feat3)

        feature2_2w = self._inv_warp_flow(feature2_2, upflow3)
        cv2 = self._cost_volumn(feature1_2, feature2_2w, d=4)
        flow2, feat2 = self._dec2(tf.concat([cv2, feature1_2, upflow3, upfeat3], axis=3))
        flow2 = self._cn([flow2, feat2])

        # flow of source(img1) to target(img2), size = [1/64, 1/32, 1/16, 1/8, 1/4]
        return flow6, flow5, flow4, flow3, flow2


def scale_pyramid(img, pool_size0, pool_size1, num_scales):
    downsample0 = AveragePooling2D(pool_size0)
    downsample1 = AveragePooling2D(pool_size1)
    scale = 1.0 / pool_size0
    scaled_imgs = [img if pool_size0 == 1 else downsample0(img * scale)]
    for _ in range(1, num_scales):
        scale /= pool_size1
        scaled_imgs.append(downsample1(scaled_imgs[-1] * scale))
    return scaled_imgs[::-1] # ordered by smaller to bigger size

def create_model(training=False):
    batch_size = opt.batch_size if training else 1
    imgL = Input(shape=(opt.img_height, opt.img_width, 3), batch_size=batch_size, dtype='float32')
    imgR = Input(shape=(opt.img_height, opt.img_width, 3), batch_size=batch_size, dtype='float32')

    feat = FeaturePyramid(name='feature_net_disp')
    flow = FlowPyramid(name='flow_pyramid')

    featL = feat(imgL)
    featR = feat(imgR)
    pyrL_pred = flow(featL + featR)
    pyrR_pred = flow(featR + featL)

    if training == True:
        # loss, metric during training
        dispL_true = Input(shape=(opt.img_height, opt.img_width, 1), batch_size=batch_size, dtype='float32')
        dispR_true = Input(shape=(opt.img_height, opt.img_width, 1), batch_size=batch_size, dtype='float32')
        model = tf.keras.Model([imgL, imgR, dispL_true, dispR_true], [])

        alphas = [0.32, 0.08, 0.02, 0.01, 0.005]
        flowL_x_true = -dispL_true * opt.img_width
        flowR_x_true = dispR_true * opt.img_width
        flowL_true = tf.concat([tf.zeros_like(flowL_x_true), flowL_x_true], axis=-1)
        flowR_true = tf.concat([tf.zeros_like(flowR_x_true), flowR_x_true], axis=-1)
        pyrL_true = scale_pyramid(flowL_true, 4, 2, 5)
        pyrR_true = scale_pyramid(flowR_true, 4, 2, 5)
        for s in range(5):
            epeL = tf.norm(pyrL_pred[s] - pyrL_true[s], axis=-1)
            epeR = tf.norm(pyrR_pred[s] - pyrR_true[s], axis=-1)
            if s == 0:
                # end-point-error
                aepe = 0.5 * tf.reduce_mean(epeL + epeR)
                model.add_metric(aepe, name='AEPE', aggregation='mean')

            if opt.loss_fn == 'l2norm': 
                loss = 0.5 * tf.reduce_mean(epeL + epeR)
            else:
                raise ValueError('! Unsupported loss metric')
            model.add_loss(alphas[s] * loss, inputs=True)
    else:
        model = tf.keras.Model([imgL, imgR], pyrL_pred[-1:] + pyrR_pred[-1:])
    return model


if __name__ == '__main__':
    import cv2
    import numpy as np
    from pathlib import Path
    import time
    import utils
    import functools

    disp_net = create_model(training=False)
    disp_net.load_weights('.results_stereosv/weights-010.h5')
    disp_net.summary()
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
        disp = dispL[0].numpy()
        print("* elspaed:", t1 - t0, np.min(disp), np.max(disp))
        if utils.imshow(disp) == 27:
            break
        #utils.imshow(disp0)
