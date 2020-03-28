import tensorflow as tf
import numpy as np

from autoflags import opt
import imgtool
from eval.evaluate_flow import scale_intrinsics
from monodepth_dataloader import get_multi_scale_intrinsics

''' VICTECH
This TestModel is somewhat different with Model_eval_xxx.
Since for debug/test/evaluate purpose, we need to access intermediate 
loss/image/depth/mask tensor of actual training model, 
Model_eval_xxx is not sufficient to do this.
Therefore, we make wrapper class of actual training model and 
use this when we want to dig into the graph.
'''
class TestModel(object):
    def __init__(self, train_model, scope):
        with tf.variable_scope(scope, reuse=False):
            img_shape = [1, opt.img_height, opt.img_width, 3]
            self.image1 = tf.placeholder(tf.float32, img_shape, name='input_image1')
            self.image2 = tf.placeholder(tf.float32, img_shape, name='input_image2')
            self.image1r = tf.placeholder(tf.float32, img_shape, name='input_image1r')
            self.image2r = tf.placeholder(tf.float32, img_shape, name='input_image2r')
            self.intrinsic = tf.placeholder(tf.float32, [3, 3])
            self.cam2pix, self.pix2cam = get_multi_scale_intrinsics(self.intrinsic, opt.num_scales)
            self.cam2pix = tf.expand_dims(self.cam2pix, axis=0)
            self.pix2cam = tf.expand_dims(self.pix2cam, axis=0)
            self.model = train_model(self.image1, self.image2, self.image1r, self.image2r, 
                self.cam2pix, self.pix2cam, reuse_scope=False, scope=scope)
            self.outputs = self.model.outputs

    def __call__(self, sess, img1, img2, img1r, img2r, K):
        orig_H, orig_W = img1.shape[0:2]
        img1 = imgtool.imresize(img1, (opt.img_height, opt.img_width))
        img2 = imgtool.imresize(img2, (opt.img_height, opt.img_width))
        img1r = imgtool.imresize(img1r, (opt.img_height, opt.img_width))
        img2r = imgtool.imresize(img2r, (opt.img_height, opt.img_width))

        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)
        img1r = np.expand_dims(img1r, axis=0)
        img2r = np.expand_dims(img2r, axis=0)
        img1 = (img1 / 255).astype(np.float32)
        img2 = (img2 / 255).astype(np.float32)
        img1r = (img1r / 255).astype(np.float32)
        img2r = (img2r / 255).astype(np.float32)

        K = scale_intrinsics(K, opt.img_width / orig_W, opt.img_height / orig_H)
        outputs = sess.run(self.model.outputs, 
                feed_dict = {self.image1: img1, self.image2: img2, 
                self.image1r: img1r, self.image2r: img2r, self.intrinsic: K})
        return outputs
