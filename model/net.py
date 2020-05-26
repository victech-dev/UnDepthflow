import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Layer, Conv2D, BatchNormalization, ReLU, Lambda, UpSampling2D, AveragePooling2D
import tensorflow_addons as tfa
import functools
# # # # #DEBUG!!!
# # # # import sys
# # # # sys.path.append('C:\\workspace\\UnDepthflow')
# # # # #DEBUG!!!

from opt_helper import opt
from losses import charbonnier_loss

from model.modules import *


def factory(type):
    if type == 'level_0':
        return CMD240x240(bn=True)
    elif type == 'level_1':
        return CMD120x120(bn=True)
    elif type == 'level_2':
        return CMD60x60(bn=True)
    elif type == 'level_3':
        return CMDTop(bn=True)
    elif type == 'level_4':
        return CMDTop(bn=True)
    assert 0, 'Correspondence Map Decoder bad creation: ' + type


class InvWarpFlow(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        img, flow = inputs
        return tfa.image.dense_image_warp(img, -flow)    


def scale_pyramid(img, pool_size0, pool_size1, num_scales):
    scaled_imgs = [img if pool_size0 == 1 else AveragePooling2D(pool_size0)(img)]
    downsample1 = AveragePooling2D(pool_size1)
    for _ in range(1, num_scales):
        scaled_imgs.append(downsample1(scaled_imgs[-1]))
    return scaled_imgs[::-1]


def dgc_net(training, mask=False):
    batch_size = opt.batch_size if training else 1
    img1 = Input(shape=(240, 240, 3), batch_size=batch_size, dtype='uint8')
    img2 = Input(shape=(240, 240, 3), batch_size=batch_size, dtype='uint8')

    feat = vgg16_pyramid()
    x1 = tf.cast(img1, tf.float32)
    x1 = tf.keras.applications.vgg16.preprocess_input(x1)
    x2 = tf.cast(img2, tf.float32)
    x2 = tf.keras.applications.vgg16.preprocess_input(x2)
    target_pyr = feat(x1)
    source_pyr = feat(x2)

    # do feature normalisation
    feat_top_pyr_trg = tf.math.l2_normalize(target_pyr[-1], axis=-1, epsilon=1e-6)
    feat_top_pyr_src = tf.math.l2_normalize(source_pyr[-1], axis=-1, epsilon=1e-6)

    # do correlation
    corr1 = CorrelationVolume()([feat_top_pyr_trg, feat_top_pyr_src])
    corr1 = tf.math.l2_normalize(tf.nn.relu(corr1), axis=-1, epsilon=1e-6)

    # correspondence map decoding, for each level of the feature pyramid
    init_map = tf.zeros_like(corr1[:,:,:,:2])
    est_grid = factory('level_4')([corr1, init_map], training=training)
    estimates_grid = [est_grid]

    upsampling_x2 = UpSampling2D(size=2, interpolation='bilinear')
    inv_warp_flow = InvWarpFlow()
    for k in reversed(range(4)):
        p1, p2 = target_pyr[k], source_pyr[k]
        est_map = upsampling_x2(estimates_grid[-1])
        p1_w = inv_warp_flow([p1, est_map])
        est_map = factory(f'level_{k}')([p1_w, p2, est_map], training=training)
        estimates_grid.append(est_map)

    # if self.mask:
    #     self.matchability_net = MatchabilityNet(in_channels=128, bn=True)
    # matchability = None
    # if self.mask:
    #     matchability = self.matchability_net(x1=p1_w, x2=p2)

    if training == True:
        # loss, metric during training

        # note disp is normalized disp
        disp = Input(shape=(240, 240, 1), batch_size=batch_size, dtype='float32')
        model = tf.keras.Model([img1, img2, disp], [])

        flow = tf.concat([tf.zeros_like(disp), disp], axis=-1)
        pyr_true = scale_pyramid(flow, 1, 2, 5)
        pyr_true = [f * (15.0 * 2**i) for i, f in enumerate(pyr_true)]
        pyr_pred = estimates_grid

        SCALE_FACTOR = [1, 1, 1, 1, 1]
        for s in range(5):
            epe = tf.norm(pyr_pred[s] - pyr_true[s], axis=-1)
            if s == 4:
                # end-point-error
                pixel_error = tf.reduce_mean(epe)
                model.add_metric(pixel_error, name='epe', aggregation='mean')

            if opt.loss_metric == 'charbonnier':
                diff = tf.abs(pyr_pred[s] - pyr_true[s])
                loss = charbonnier_loss(diff)
            else:
                raise ValueError('! Unsupported loss metric')
            model.add_loss(SCALE_FACTOR[s] * loss, inputs=True)
    else:
        model = tf.keras.Model([img1, img2], estimates_grid[-1])
    return model

if __name__ == '__main__':
    import cv2
    import numpy as np
    from pathlib import Path
    import time
    import utils
    import functools

    model = dgc_net(training=True)
    #disp_net.load_weights('.results_stereosv/weights-log.h5')
    model.summary()
    predict = tf.function(functools.partial(model.call, training=False, mask=None))

    # for _ in range(100):
    #     img1 = np.random.randint(0, 256, (1, 240, 240, 3), dtype=np.uint8)
    #     img2 = np.random.randint(0, 256, (1, 240, 240, 3), dtype=np.uint8)

    #     t0 = time.time()
    #     outputs = predict([img1, img2])
    #     t1 = time.time()
    #     print("* elspaed:", t1 - t0)
 
    # # point cloud test of office image of inbo.yeo 
    # data_dir = Path('M:\\Users\\sehee\\camera_taker\\undist_fisheye')
    # imgnamesL = sorted(Path(data_dir/'imL').glob('*.png'), key=lambda v: int(v.stem))
    # for index in range(len(imgnamesL)):
    #     imgnameL = imgnamesL[index % len(imgnamesL)]
    #     imgnameR = (data_dir/'imR'/imgnameL.stem).with_suffix('.png')

    #     imgL = utils.imread(str(imgnameL))
    #     imgR = utils.imread(str(imgnameR))
    #     imgL, imgR = utils.resize_image_pairs(imgL, imgR, (opt.img_width, opt.img_height), np.float32)

    #     t0 = time.time()
    #     dispL, _ = predict([imgL[None], imgR[None]])
    #     t1 = time.time()
    #     disp = dispL[0].numpy()
    #     print("* elspaed:", t1 - t0, np.min(disp), np.max(disp))
    #     if utils.imshow(disp) == 27:
    #         break
    #     #utils.imshow(disp0)

