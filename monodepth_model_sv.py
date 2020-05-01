from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from opt_utils import opt
from nets.pwc_disp import pwc_disp, feature_pyramid_disp
from core_warp import inv_warp_flow, fwd_warp_flow
from loss_utils import charbonnier_loss
from pcd_utils import tf_populate_pcd

class MonodepthModel(object):
    """monodepth model"""

    def __init__(self, mode, left, right, left_feature, right_feature, left_disp, right_disp, reuse_variables=None):
        self.mode = mode
        self.left = left
        self.right = right
        self.left_feature = left_feature
        self.right_feature = right_feature
        self.reuse_variables = reuse_variables

        self.build_model()

        if self.mode == 'train':
            self.build_outputs()

            dispL_pyramid = self.scale_pyramid(left_disp, 4)
            dispR_pyramid = self.scale_pyramid(right_disp, 4)
            SCALE_FACTOR = np.array([1.0, 0.8, 0.6, 0.4])

            loss = 0
            disp_L1_loss = []
            for s in range(4):
                left_pixel_error = opt.img_width * (dispL_pyramid[s] - self.disp_left_est[s])
                right_pixel_error = opt.img_width * (dispR_pyramid[s] - self.disp_right_est[s])
                disp_L1_loss.append(0.5 * (tf.reduce_mean(tf.abs(left_pixel_error)) + tf.reduce_mean(tf.abs(right_pixel_error))))

                if opt.loss_metric == 'l1-log': # l1 of log
                    left_error = tf.abs(tf.log(1.0 + dispL_pyramid[s]) - tf.log(1.0 + self.disp_left_est[s]))
                    right_error = tf.abs(tf.log(1.0 + dispR_pyramid[s]) - tf.log(1.0 + self.disp_right_est[s]))
                    loss += SCALE_FACTOR[s] * (tf.reduce_mean(left_error) + tf.reduce_mean(right_error))
                elif opt.loss_metric == 'charbonnier':
                    loss += 0.1 * SCALE_FACTOR[s] * (charbonnier_loss(left_pixel_error) + charbonnier_loss(right_pixel_error))
                else:
                    raise ValueError('! Unsupported loss metric')

            self.total_loss = loss
            self.disp_L1_loss = disp_L1_loss[0]

    def scale_pyramid(self, img, num_scales):
        downsample = tf.keras.layers.AveragePooling2D(2)
        scaled_imgs = [img]
        for _ in range(1, num_scales):
            scaled_imgs.append(downsample(scaled_imgs[-1]))
        return scaled_imgs

    def generate_flow_left(self, disp, scale):
        W = opt.img_width // (2**scale)
        ltr_flow = -disp * W
        ltr_flow = tf.concat([ltr_flow, tf.zeros_like(ltr_flow)], axis=3)
        return ltr_flow

    def generate_flow_right(self, disp, scale):
        return self.generate_flow_left(-disp, scale)

    def build_model(self):
        with slim.arg_scope(
            [slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            with tf.variable_scope('depth_net', reuse=self.reuse_variables):

                self.left_pyramid = self.scale_pyramid(self.left, 4)
                if self.mode == 'train':
                    self.right_pyramid = self.scale_pyramid(self.right, 4)

                self.model_input = tf.concat([self.left, self.right], 3)

                self.disp1, self.disp2, self.disp3, self.disp4, self.disp_pyr = pwc_disp(
                    self.left, self.right, self.left_feature, self.right_feature)

    def build_outputs(self):
        # STORE DISPARITIES
        H = opt.img_height
        W = opt.img_width
        with tf.variable_scope('disparities'):
            self.disp_est = [self.disp1, self.disp2, self.disp3, self.disp4]
            self.disp_left_est = [d[:, :, :, 0:1] for d in self.disp_est]
            self.disp_right_est = [d[:, :, :, 1:2] for d in self.disp_est]


class Model_stereosv(object):
    def __init__(self,
                 imageL=None,
                 imageR=None,
                 dispL=None,
                 dispR=None,
                 reuse_scope=False,
                 scope=None):

        with tf.variable_scope(scope, reuse=reuse_scope):
            left_feature = feature_pyramid_disp(imageL, reuse=False)
            right_feature = feature_pyramid_disp(imageR, reuse=True)

            model = MonodepthModel('train', imageL, imageR, left_feature, right_feature, dispL, dispR)
            outputs = dict(disp=[model.disp1, model.disp2, model.disp3, model.disp4])

        self.loss = model.total_loss
        self.outputs = dict(stereo=outputs)

        # Create summaries once when multiple models are created in multiple gpu
        if not tf.get_collection(tf.GraphKeys.SUMMARIES, scope=f'stereosv_losses/.*'):
            with tf.name_scope('stereosv_losses/'):
                tf.summary.scalar('total_loss', model.total_loss)
                tf.summary.scalar('disp_L1_loss', model.disp_L1_loss)


class Model_eval_stereosv(object):
    def __init__(self, scope=None):
        with tf.variable_scope(scope, reuse=True):
            input_uint8_L = tf.placeholder(tf.uint8, [1, opt.img_height, opt.img_width, 3], name='raw_input_L')
            input_uint8_R = tf.placeholder(tf.uint8, [1, opt.img_height, opt.img_width, 3], name='raw_input_R')
            K = tf.placeholder(tf.float32, [1, 3, 3], name='cam_intrinsic_input')
            baseline = tf.placeholder(tf.float32, [1], name='baseline_input')

            input_L = tf.image.convert_image_dtype(input_uint8_L, dtype=tf.float32)
            input_R = tf.image.convert_image_dtype(input_uint8_R, dtype=tf.float32)

            featureL_disp = feature_pyramid_disp(input_L, reuse=True)
            featureR_disp = feature_pyramid_disp(input_R, reuse=True)

            model = MonodepthModel('test', input_L, input_R, featureL_disp, featureR_disp, None, None)
            pred_disp = [model.disp1, model.disp2, model.disp3, model.disp4]
            pred_normal = self.build_normal_map(model.disp_pyr[:, :, :, 0:1], K, baseline)

        self.input_L = input_uint8_L
        self.input_R = input_uint8_R
        self.input_K = K
        self.input_baseline = baseline

        self.pred_disp = pred_disp[0][:, :, :, 0:1]
        self.pred_normal = pred_normal

    def _detect_plane_xz(self, A):
        A_max = tf.reduce_max(tf.abs(A), axis=(-2,-1))
        eval_zero = tf.cast(A_max < 1e-6, tf.float32)

        A_scaled = A / tf.maximum(A_max[...,None,None], 1e-6)
        a00, a11, a22 = A_scaled[...,0,0], A_scaled[...,1,1], A_scaled[...,2,2]
        a01, a02, a12 = A_scaled[...,0,1], A_scaled[...,0,2], A_scaled[...,1,2]

        norm = a01**2 + a02**2 + a12**2
        q = (a00 + a11 + a22) / 3.0
        b00, b11, b22 = a00 - q, a11 - q, a22 - q
        p = tf.sqrt((b00**2 + b11**2 + b22**2 + 2.0 * norm) / 6.0)
        eig_triple = tf.cast(p < 1e-6, tf.float32) * (1.0 - eval_zero)

        c00 = b11 * b22 - a12 * a12
        c01 = a01 * b22 - a12 * a02
        c02 = a01 * a12 - b11 * a02
        det = (b00 * c00 - a01 * c01 + a02 * c02) / tf.maximum(p, 1e-6)**3
        half_det = tf.clip_by_value(0.5 * det, -1.0, 1.0)
        angle = tf.math.acos(half_det) / 3.0

        # beta0, beta1, beta2
        beta2 = tf.math.cos(angle) * 2.0
        beta0 = tf.math.cos(angle + np.pi * 2/3) * 2.0
        beta1 = -(beta0 + beta2)
        betas = tf.stack([beta0, beta1, beta2], axis=-1)

        # eval0 <= eval1 <= eval2
        # Merge 3 cases: all zeros, triple roots, ordinal case.
        # Note this actually should be rescaled to initial values (eig *= A_max), 
        # but we are interesting in eigenvector, so let's just skip it. 
        evals = q[...,None] + p[...,None] * betas
        evals = (eig_triple * q)[...,None] + ((1.0 - eval_zero) * (1.0 - eig_triple))[...,None] * evals
        eval0, eval1, eval2 = tf.unstack(evals, axis=-1)

        # y-magnitude of eigenvector which has the smallest eigenvalue 
        # (= y-magnitude of normal vector from local pcd)
        denom = (eval0 - eval1) * (eval0 - eval2)
        evec_zero = tf.cast(denom < 1e-6, tf.float32)
        ny2_num = eval0**2 - (a00 + a22) * eval0 + (a00 * a22 - a02**2)
        ny2 = tf.clip_by_value(ny2_num / tf.maximum(denom, 1e-6), 0, 1)
        return (1.0 - evec_zero) * tf.sqrt(ny2)

    def build_normal_map(self, disp, K0, baseline):
        '''
        disp: normalized disparity of shape [B, H//4, W//4, 1] (bottom of pyramid)
        K0: rescaled already from original size to [opt.img_width, opt.img_height]
        '''
        _, h1, w1, _ = tf.unstack(tf.shape(disp))
        rw = tf.cast(w1, tf.float32) / opt.img_width
        rh = tf.cast(h1, tf.float32) / opt.img_height

        #DEBUG!!!! K 의 rescale 을 사실은 original image size 에 대해서 해야하는데..
        #DEBUG!!!! 최종 return 되는 normal 에 padding 이 껴있어야하는데...

        # rescale intrinsic (note K should be rescaled already from original size to [opt.img_width, opt.img_height])
        K_scale = tf.convert_to_tensor([[rw, 0, 0.5*(rw-1)], [0, rh, 0.5*(rh-1)], [0, 0, 1]], dtype=tf.float32)
        K1 = K_scale[None,:,:] @ K0

        # construct point cloud
        fxb = K1[:,0,0] * baseline
        depth = fxb[:,None,None,None] / (tf.cast(w1, tf.float32) * disp)
        xyz = tf_populate_pcd(depth, K1)

        # valid padding: shape of sigma = [b, H-ksize+1, W-ksize+1, 3, 3]
        ksize = 5
        P = tf.nn.avg_pool(xyz, ksize, 1, 'VALID')
        xyz2 = xyz[:,:,:,:,None] @ xyz[:,:,:,None,:]
        xyz2_4d = tf.reshape(xyz2, tf.concat([tf.shape(xyz2)[:-2], [9]], 0))
        S_4d = tf.nn.avg_pool(xyz2_4d, ksize, 1, 'VALID')
        S = tf.reshape(S_4d, tf.concat([tf.shape(S_4d)[:-1], [3, 3]], 0))
        sigma = S - P[:,:,:,:,None] @ P[:,:,:,None,:]

        return self._detect_plane_xz(sigma)

import cv2
from misc import read_pfm

def _solve_eigh_3x3(A):
    A_max = tf.reduce_max(tf.abs(A), axis=(-2,-1))
    eval_zero = tf.cast(A_max < 1e-6, tf.float32)

    A_scaled = A / tf.maximum(A_max[...,None,None], 1e-6)
    a00, a11, a22 = A_scaled[...,0,0], A_scaled[...,1,1], A_scaled[...,2,2]
    a01, a02, a12 = A_scaled[...,0,1], A_scaled[...,0,2], A_scaled[...,1,2]

    norm = a01**2 + a02**2 + a12**2
    q = (a00 + a11 + a22) / 3.0
    b00, b11, b22 = a00 - q, a11 - q, a22 - q
    p = tf.sqrt((b00**2 + b11**2 + b22**2 + 2.0 * norm) / 6.0)
    eig_triple = tf.cast(p < 1e-6, tf.float32) * (1.0 - eval_zero)

    c00 = b11 * b22 - a12 * a12
    c01 = a01 * b22 - a12 * a02
    c02 = a01 * a12 - b11 * a02
    det = (b00 * c00 - a01 * c01 + a02 * c02) / tf.maximum(p, 1e-6)**3
    half_det = tf.clip_by_value(0.5 * det, -1.0, 1.0)
    angle = tf.math.acos(half_det) / 3.0

    # beta0, beta1, beta2
    beta2 = tf.math.cos(angle) * 2.0
    beta0 = tf.math.cos(angle + np.pi * 2/3) * 2.0
    beta1 = -(beta0 + beta2)
    betas = tf.stack([beta0, beta1, beta2], axis=-1)

    # eval0 <= eval1 <= eval2
    # Merge 3 cases: all zeros, triple roots, ordinal case.
    # Note this actually should be rescaled to initial values (eig *= A_max), 
    # but we are interesting in eigenvector, so let's just skip it. 
    evals = q[...,None] + p[...,None] * betas
    evals = (eig_triple * q)[...,None] + ((1.0 - eval_zero) * (1.0 - eig_triple))[...,None] * evals
    eval0, eval1, eval2 = tf.unstack(evals, axis=-1)

    # y-magnitude of eigenvector which has the smallest eigenvalue 
    # (= y-magnitude of normal vector from local pcd)
    denom = (eval0 - eval1) * (eval0 - eval2)
    evec_zero = tf.cast(denom < 1e-6, tf.float32)
    ny2_num = (eval0**2 - (a00 + a22) * eval0 + (a00*a22 - a02**2))
    ny2 = tf.maximum(ny2_num, 0) / tf.maximum(denom, 1e-6)
    return (1.0 - evec_zero) * tf.sqrt(ny2)

def populate_pcd(depth, K):
    H, W = depth.shape[:2]
    py, px = np.mgrid[:H,:W]
    xyz = np.stack([px, py, np.ones_like(px)], axis=-1) * np.atleast_3d(depth)
    xyz = np.reshape(xyz, (-1, 3)) @ np.linalg.inv(K).T
    return np.reshape(xyz, (H, W, 3))

def resize_disp_with_K(disp, K0, size):
    h0, w0 = disp.shape[:2]
    w1, h1 = size
    rw, rh = w1 / w0, h1 / h0
    K_scale = np.array([[rw, 0, 0.5*(rw-1)], [0, rh, 0.5*(rh-1)], [0, 0, 1]], dtype=np.float32)
    K1 = K_scale @ K0
    interpolation = cv2.INTER_LINEAR if w1 > w0 else cv2.INTER_AREA
    disp2 = cv2.resize(disp, (w1, h1), interpolation=interpolation) * rw
    return disp2, K1

def np_run(disp, K):
    disp, K = resize_disp_with_K(disp, K, (128,96))
    xyz = populate_pcd(K[0,0] * 0.120601 / disp, K)
    xyz = np.pad(xyz, ((1,0),(1,0),(0,0))) # [1+H, 1+W, 3]

    IP = np.cumsum(xyz, axis=1)
    IP = np.cumsum(IP, axis=0)
    IS = np.cumsum(xyz[:,:,:,None] @ xyz[:,:,None,:], axis=1)
    IS = np.cumsum(IS, axis=0)

    # valid padding: shape of P,S = [H - ksize + 1, W - ksize + 1, 4(, 4)]
    ksize = 5
    P = (IP[ksize:,ksize:] + IP[:-ksize,:-ksize] - IP[:-ksize,ksize:] - IP[ksize:,:-ksize]) / ksize**2
    S = (IS[ksize:,ksize:] + IS[:-ksize,:-ksize] - IS[:-ksize,ksize:] - IS[ksize:,:-ksize]) / ksize**2

    sigma = S - P[:,:,:,None] @ P[:,:,None,:]
    w, v = np.linalg.eig(sigma)
    axis_idx = np.argmin(w, axis=2)
    n = np.take_along_axis(v, axis_idx[:,:,None,None], axis=3)
    n = np.squeeze(n)
    return np.abs(n[:,:,1])

    n = n[:,:,1] # y component of normal vector
    n = np.abs(n) 

    n_pad = np.zeros([n.shape[0] + ksize - 1, n.shape[1] + ksize - 1], n.dtype)
    n_pad[ksize//2:-ksize//2+1, ksize//2:-ksize//2+1] = n
    floor = np.logical_and(n_pad > 0.9, xyz[:,:,1] > 0.1).astype(np.float32)
    floor = cvt_topdown(floor, K)
    exit()

def tf_run(disp, K0):
    disp, _ = resize_disp_with_K(disp, K0, (128,96))
    disp = tf.convert_to_tensor(disp)[None,:,:,None]

    K0 = tf.convert_to_tensor(K0)[None]
    _, h1, w1, _ = tf.unstack(tf.shape(disp))
    rw = tf.cast(w1, tf.float32) / 640 #DEBUG actually img_width
    rh = tf.cast(h1, tf.float32) / 480 #DEBUG actually img_height

    K_scale = tf.convert_to_tensor([[rw, 0, 0.5*(rw-1)], [0, rh, 0.5*(rh-1)], [0, 0, 1]], dtype=tf.float32)
    K1 = K_scale[None,:,:] @ K0
    K1_inv = tf.linalg.inv(K1)

    # construct point cloud
    fxb = K1[:,0,0] * 0.120601
    depth = fxb[:,None,None,None] / disp
    px, py = tf.meshgrid(tf.range(w1), tf.range(h1))
    px, py = tf.cast(px, tf.float32), tf.cast(py, tf.float32)
    xyz = tf.stack([px, py, tf.ones_like(px)], axis=-1)[None] * depth
    xyz = tf.squeeze(K1_inv[:,None,None,:,:] @ xyz[:,:,:,:,None], axis=-1) # [b, H, W, 3]

    # valid padding: shape of sigma = [b, H-ksize+1, W-ksize+1, 3, 3]
    ksize = 5
    P = tf.nn.avg_pool(xyz, ksize, 1, 'VALID')
    xyz2 = xyz[:,:,:,:,None] @ xyz[:,:,:,None,:]
    xyz2_4d = tf.reshape(xyz2, tf.concat([tf.shape(xyz2)[:-2], [9]], 0))
    S_4d = tf.nn.avg_pool(xyz2_4d, ksize, 1, 'VALID')
    S = tf.reshape(S_4d, tf.concat([tf.shape(S_4d)[:-1], [3, 3]], 0))
    sigma = S - P[:,:,:,:,None] @ P[:,:,:,None,:]

    return np.array(_solve_eigh_3x3(sigma))



if __name__ == '__main__':
    tf.enable_eager_execution()

    #_solve_eigh_3x3(tf.convert_to_tensor([[0,2,4],[2,0,1],[4,1,0]], dtype=tf.float32))

    disp, _ = read_pfm(f'M:\\Users\\sehee\\camera_taker\\undist_fisheye\\dispL\\000003.pfm')
    K = [[467.83661057, 0.0, 284.1095847], [0.0, 467.83661057, 256.36649503], [0.0, 0.0, 1.0]]
    K = np.array(K, np.float32)

    ret0 = np_run(disp, K)
    ret1 = tf_run(disp, K)
    np.set_printoptions(precision=6, suppress=True)
    print('* All-close:', np.allclose(ret0, ret1[0]))
    print('* Max L1 diff:', np.max(np.abs(ret0 - ret1[0])))
    print(ret0[50,50:60])
    print(ret1[0, 50,50:60])

    #floor = (ret1[0,:,:,1] > 0.9).astype(np.float32)
    from imgtool import imshow
    imshow(ret1[0])
    #imshow(floor)

