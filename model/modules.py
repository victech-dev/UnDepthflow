import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Layer, Conv2D, BatchNormalization, ReLU, Lambda


class ConvBlock(Layer):
    def __init__(self, filters, kernel_size=3, stride=1, dilation=1, bn=False, *args, **kwargs):
        super(ConvBlock, self).__init__(*args, **kwargs)
        self._cnv = Conv2D(filters, kernel_size, stride, 'same', dilation_rate=dilation)
        self._maybe_bn = BatchNormalization() if bn else Lambda(lambda x: x)
        self._act_fn = ReLU()

    def call(self, inputs, training=None):
        x = self._cnv(inputs)
        x = self._maybe_bn(x, training=training)
        return self._act_fn(x)


class CorrelationVolume(Layer):
    """
    Implementation by Ignacio Rocco
    paper: https://arxiv.org/abs/1703.05593
    project: https://github.com/ignacio-rocco/cnngeometric_pytorch
    """
    def __init__(self, *args, **kwargs):
        super(CorrelationVolume, self).__init__(*args, **kwargs)
    
    def call(self, inputs):
        feature_A, feature_B = inputs
        _, H, W, C = tf.unstack(tf.shape(feature_A))
        feature_A = tf.reshape(tf.transpose(feature_A, perm=[0, 3, 2, 1]), [-1, C, W*H]) # [B, C, W*H]
        feature_B = tf.reshape(feature_B, [-1, H*W, C]) # [B, H*W, C]
        feature_mul = feature_B @ feature_A # [B, H*W, W*H]
        correlation_tensor = tf.reshape(feature_mul, [-1, H, W, W*H]) # [B, H, W, W*H]
        return correlation_tensor


# class MatchabilityNet(Layer):
#     """ 
#     Matchability network to predict a binary mask
#     """
#     def __init__(self, in_channels, bn=False):
#         super(MatchabilityNet, self).__init__()
#         self.conv0 = conv_blck(in_channels, 64, bn=bn)
#         self.conv1 = conv_blck(self.conv0[0].out_channels, 32, bn=bn)
#         self.conv2 = conv_blck(self.conv1[0].out_channels, 16, bn=bn)
#         self.conv3 = nn.Conv2d(self.conv2[0].out_channels, 1, kernel_size=1)

#     def forward(self, x1, x2):
#         x = torch.cat((x1, x2), 1)
#         x = self.conv3(self.conv2(self.conv1(self.conv0(x))))
#         return x


class CMDTop(Layer):
    def __init__(self, bn=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        chan = [128, 128, 96, 64, 32]
        self._cnvs = [ConvBlock(oc, bn=bn) for oc in chan]
        self._cnvs.append(Conv2D(2, 3, 1, 'same'))

    def call(self, inputs, training=None):
        x = tf.concat(inputs, axis=-1)
        for cnv in self._cnvs:
            x = cnv(x, training=training)
        return x


class CMD60x60(Layer):
    def __init__(self, bn=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        chan = [(128,1), (96,2), (64,3), (32,4)]
        self._cnvs = [ConvBlock(oc, dilation=d, bn=bn) for oc, d in chan]
        self._cnvs.append(Conv2D(2, 3, 1, 'same'))

    def call(self, inputs, training=None):
        x = tf.concat(inputs, axis=-1)
        for cnv in self._cnvs:
            x = cnv(x, training=training)
        return x


class CMD120x120(Layer):
    def __init__(self, bn=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        chan = [(128,1), (96,4), (64,6), (32,8)]
        self._cnvs = [ConvBlock(oc, dilation=d, bn=bn) for oc, d in chan]
        self._cnvs.append(Conv2D(2, 3, 1, 'same'))

    def call(self, inputs, training=None):
        x = tf.concat(inputs, axis=-1)
        for cnv in self._cnvs:
            x = cnv(x, training=training)
        return x
        

class CMD240x240(Layer):
    def __init__(self, bn=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        chan = [(128,1), (96,4), (64,12), (32,16)]
        self._cnvs = [ConvBlock(oc, dilation=d, bn=bn) for oc, d in chan]
        self._cnvs.append(Conv2D(2, 3, 1, 'same'))

    def call(self, inputs, training=None):
        x = tf.concat(inputs, axis=-1)
        for cnv in self._cnvs:
            x = cnv(x, training=training)
        return x


def vgg16_pyramid():
    source_model = tf.keras.applications.VGG16(include_top=False, input_shape=(240, 240, 3), weights='imagenet')
    feat_names = ['block1_conv1', 'block1_pool', 'block2_pool', 'block3_pool', 'block4_pool']
    feat_ext = Model(inputs=source_model.inputs, outputs=[source_model.get_layer(name).output for name in feat_names])
    feat_ext.trainable = False
    return feat_ext
