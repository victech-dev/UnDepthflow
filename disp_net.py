import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, UpSampling2D, AveragePooling2D, LeakyReLU, Activation
from tensorflow.keras import Sequential, Model, Input
import functools

from opt_helper import opt
from losses import charbonnier_loss
from core_warp import inv_warp_flow_x

_reg = tf.keras.regularizers.l2(opt.weight_decay)

class ConvBlock(Layer):
    def __init__(self, filters, kernel_size, strides, dilation_rate=1, linear=False, *args, **kwargs):
        super(ConvBlock, self).__init__(*args, **kwargs)
        self._cnv = Conv2D(filters, kernel_size, strides, 
            padding='same', dilation_rate=dilation_rate, 
            kernel_initializer='he_normal', kernel_regularizer=_reg)
        self._act_fn = Activation('linear') if linear else LeakyReLU(0.1)

    def call(self, inputs):
        return self._act_fn(self._cnv(inputs))


class FeaturePyramid(Layer):
    def __init__(self, *args, **kwargs):
        super(FeaturePyramid, self).__init__(*args, **kwargs)
        self._cnvs = []
        self._cnvs.append(ConvBlock(16, 3, 2, name='cnv1'))
        self._cnvs.append(ConvBlock(16, 3, 1, name='cnv2'))
        self._cnvs.append(ConvBlock(32, 3, 2, name='cnv3'))
        self._cnvs.append(ConvBlock(32, 3, 1, name='cnv4'))
        self._cnvs.append(ConvBlock(64, 3, 2, name='cnv5'))
        self._cnvs.append(ConvBlock(64, 3, 1, name='cnv6'))
        self._cnvs.append(ConvBlock(96, 3, 2, name='cnv7'))
        self._cnvs.append(ConvBlock(96, 3, 1, name='cnv8'))
        self._cnvs.append(ConvBlock(128, 3, 2, name='cnv9'))
        self._cnvs.append(ConvBlock(128, 3, 1, name='cnv10'))
        self._cnvs.append(ConvBlock(192, 3, 2, name='cnv11'))
        self._cnvs.append(ConvBlock(192, 3, 1, name='cnv12'))

    def call(self, inputs):
        out = [inputs]
        for cnv in self._cnvs:
            out.append(cnv(out[-1]))
        return out[2], out[4], out[6], out[8], out[10], out[12]


class FlowDecoder(Layer):
    def __init__(self, *args, **kwargs):
        super(FlowDecoder, self).__init__(*args, **kwargs)
        self._cnv1 = ConvBlock(128, 3, 1, name=f'cnv1')
        self._cnv2 = ConvBlock(128, 3, 1, name=f'cnv2')
        self._cnv3 = ConvBlock(96, 3, 1, name=f'cnv3')
        self._cnv4 = ConvBlock(64, 3, 1, name=f'cnv4')
        self._cnv5 = ConvBlock(32, 3, 1, name=f'cnv5')
        self._cnv6 = ConvBlock(1, 3, 1, linear=True, name=f'cnv6')

    def call(self, inputs):
        out1 = self._cnv1(inputs)
        out2 = self._cnv2(out1)
        out3 = self._cnv3(tf.concat([out1, out2], axis=3))
        out4 = self._cnv4(tf.concat([out2, out3], axis=3))
        out5 = self._cnv5(tf.concat([out3, out4], axis=3))
        flow_x = self._cnv6(tf.concat([out4, out5], axis=3))
        return flow_x, out5


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
        self._cnvs.append(ConvBlock(1, 3, 1, dilation_rate=1, linear=True, name=f'cnv7'))

    def call(self, inputs):
        out = inputs
        for cnv in self._cnvs:
            out = cnv(out)
        return out # flow_x


class PwcNet_Single(Layer):
    def __init__(self, *args, **kwargs):
        super(PwcNet_Single, self).__init__(*args, **kwargs)
        self._cn = ContextNet(name='cn')
        self._dec2 = FlowDecoder(name='dec2')
        self._dec3 = FlowDecoder(name='dec3')
        self._dec4 = FlowDecoder(name='dec4')
        self._dec5 = FlowDecoder(name='dec5')
        self._dec6 = FlowDecoder(name='dec6')
        self._upsampling_x2 = UpSampling2D(size=2, interpolation='bilinear')
        self._neg_flow = True if 'left' in self.name else False

    @staticmethod
    def _cost_volumn(feature1, feature2, d=4):
        W = feature1.shape[2]
        feature2 = tf.pad(feature2, [[0, 0], [0, 0], [d, d], [0, 0]], "CONSTANT")
        cv = []
        for j in range(2 * d + 1):
            cv.append(tf.reduce_mean(
                feature1 * feature2[:, :, j:(j + W), :],
                axis=3, keepdims=True))
        cv = tf.concat(cv, axis=3)
        return tf.nn.leaky_relu(cv, alpha=0.1)

    def _flow_act_fn(self, flow):
        if self._neg_flow:
            return -tf.nn.relu(-flow)
        else:
            return tf.nn.relu(flow)

    def call(self, inputs):
        _, feature1_2, feature1_3, feature1_4, feature1_5, feature1_6 = inputs[:6]
        _, feature2_2, feature2_3, feature2_4, feature2_5, feature2_6 = inputs[6:]
        W = opt.img_width

        cv6 = self._cost_volumn(feature1_6, feature2_6, d=4)
        flow6, _ = self._dec6(cv6)
        flow6 = self._flow_act_fn(flow6)

        flow6to5 = self._upsampling_x2(flow6)
        feature2_5w = inv_warp_flow_x(feature2_5, flow6to5 * 0.65)
        cv5 = self._cost_volumn(feature1_5, feature2_5w, d=4)
        flow5, _ = self._dec5(tf.concat([cv5, feature1_5, flow6to5], axis=3))
        flow5 = flow5 + flow6to5
        flow5 = self._flow_act_fn(flow5)

        flow5to4 = self._upsampling_x2(flow5)
        feature2_4w = inv_warp_flow_x(feature2_4, flow5to4 * 1.25)
        cv4 = self._cost_volumn(feature1_4, feature2_4w, d=4)
        flow4, _ = self._dec4(tf.concat([cv4, feature1_4, flow5to4], axis=3))
        flow4 = flow4 + flow5to4
        flow4 = self._flow_act_fn(flow4)

        flow4to3 = self._upsampling_x2(flow4)
        feature2_3w = inv_warp_flow_x(feature2_3, flow4to3 * 2.5)
        cv3 = self._cost_volumn(feature1_3, feature2_3w, d=4)
        flow3, _ = self._dec3(tf.concat([cv3, feature1_3, flow4to3], axis=3))
        flow3 = flow3 + flow4to3
        flow3 = self._flow_act_fn(flow3)

        flow3to2 = self._upsampling_x2(flow3)
        feature2_2w = inv_warp_flow_x(feature2_2, flow3to2 * 5)
        cv2 = self._cost_volumn(feature1_2, feature2_2w, d=4)
        flow2_raw, f2 = self._dec2(tf.concat([cv2, feature1_2, flow3to2], axis=3))
        flow2_raw = flow2_raw + flow3to2
        flow2_raw = self._flow_act_fn(flow2_raw)

        flow2 = self._cn(tf.concat([flow2_raw, f2], axis=3)) + flow2_raw
        flow2 = self._flow_act_fn(flow2)

        # normalized signed disparity
        disp0 = flow2 * 5 / (W / (2**2)) # 1/4 of input size
        disp1 = flow3 * 2.5 / (W / (2**3)) # 1/8 of input size
        disp2 = flow4 * 1.25 / (W / (2**4)) # 1/16 of input size
        disp3 = flow5 * 0.65 / (W / (2**5)) # 1/32 of input size

        # Note that flow is calculated as signed value, but 
        # we are returning disp as positive value for both left and right
        if self._neg_flow:
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
    batch_size = opt.batch_size if training else 1
    imgL = Input(shape=(opt.img_height, opt.img_width, 3), batch_size=batch_size, dtype='float32')
    imgR = Input(shape=(opt.img_height, opt.img_width, 3), batch_size=batch_size, dtype='float32')

    feat = FeaturePyramid(name='feature_net_disp')
    pwcL = PwcNet_Single(name='left_disp')
    pwcR = PwcNet_Single(name='right_disp')

    featL = feat(imgL)
    featR = feat(imgR)
    pyrL_pred = pwcL(featL + featR)
    pyrR_pred = pwcR(featR + featL)

    if training == True:
        # loss, metric during training
        dispL = Input(shape=(opt.img_height, opt.img_width, 1), batch_size=batch_size, dtype='float32')
        dispR = Input(shape=(opt.img_height, opt.img_width, 1), batch_size=batch_size, dtype='float32')
        model = tf.keras.Model([imgL, imgR, dispL, dispR], [])

        pyrL_true = scale_pyramid(dispL, 4, 2, 4)
        pyrR_true = scale_pyramid(dispR, 4, 2, 4)
        SCALE_FACTOR = [1.0, 0.8, 0.6, 0.4]
        for s in range(4):
            epeL = opt.img_width * tf.abs(pyrL_pred[s] - pyrL_true[s])
            epeR = opt.img_width * tf.abs(pyrR_pred[s] - pyrR_true[s])
            if s == 0:
                # end-point-error
                pixel_error = 0.5 * tf.reduce_mean(epeL + epeR)
                model.add_metric(pixel_error, name='epe', aggregation='mean')
                # depth error w.r.t reference camera
                ref_fxb = 45.13725045708206 # reference f*b = 467.83661057 * 0.120601 * (512/640)
                max_depth = 50.0
                disp2depth = lambda x: tf.clip_by_value(ref_fxb / tf.maximum(opt.img_width * x, 1e-6), 0.0, max_depth)
                depthL_true, depthL_pred, depthR_true, depthR_pred = map(disp2depth, [pyrL_true[s], pyrL_pred[s], pyrR_true[s], pyrR_pred[s]])
                depth_error = 0.5 * tf.reduce_mean(tf.abs(depthL_pred - depthL_true) + tf.abs(depthR_pred - depthR_true))
                model.add_metric(depth_error, name='depth_error', aggregation='mean')

            if opt.loss_metric == 'l1-log': # l1 of log
                eps = 1e-6
                left_error = tf.abs(tf.math.log(eps + pyrL_true[s]) - tf.math.log(eps + pyrL_pred[s]))
                right_error = tf.abs(tf.math.log(eps + pyrR_true[s]) - tf.math.log(eps + pyrR_pred[s]))
                loss = tf.reduce_mean(left_error + right_error)
            elif opt.loss_metric == 'charbonnier':
                loss = 0.1 * (charbonnier_loss(epeL) + charbonnier_loss(epeR))
            else:
                raise ValueError('! Unsupported loss metric')
            model.add_loss(SCALE_FACTOR[s] * loss, inputs=True)
    else:
        model = tf.keras.Model([imgL, imgR], pyrL_pred[:1] + pyrR_pred[:1])
    return model


if __name__ == '__main__':
    import cv2
    import numpy as np
    from pathlib import Path
    import time
    import utils
    import functools

    disp_net = create_model(training=False)
    disp_net.load_weights('.results_stereosv/weights-log.h5')
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
