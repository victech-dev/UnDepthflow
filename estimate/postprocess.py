import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
import cv2
from opt_helper import opt


def tf_populate_pcd(depth, K):
    K_inv = tf.linalg.inv(K)
    _, H, W, _ = tf.unstack(tf.shape(depth))
    px, py = tf.meshgrid(tf.range(W), tf.range(H))
    px, py = tf.cast(px, tf.float32), tf.cast(py, tf.float32)
    xyz = tf.stack([px, py, tf.ones_like(px)], axis=-1)[None] * depth
    xyz = tf.squeeze(K_inv[:,None,None,:,:] @ xyz[:,:,:,:,None], axis=-1) # [b, H, W, 3]
    return xyz


def tf_detect_plane_xz(xyz):
    # Calculate covariance matrix of neighboring points
    # http://jacoposerafin.com/wp-content/uploads/bogoslavskyi13ecmr.pdf
    ksize = 5
    P = tf.nn.avg_pool2d(xyz, ksize, 1, 'SAME')
    xyz2 = xyz[:,:,:,:,None] @ xyz[:,:,:,None,:]
    xyz2_4d = tf.reshape(xyz2, tf.concat([tf.shape(xyz2)[:-2], [9]], 0))
    S_4d = tf.nn.avg_pool2d(xyz2_4d, ksize, 1, 'SAME')
    S = tf.reshape(S_4d, tf.concat([tf.shape(S_4d)[:-1], [3, 3]], 0))
    sigma = S - P[:,:,:,:,None] @ P[:,:,:,None,:]

    # eigenvalue solver of 3x3 symmetric matrix
    # much faster than tf.linalg.eigh when input is multiple matrices
    # https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
    A = sigma
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

    # Merge 3 cases: all zeros, triple roots, ordinal case. (eval0 <= eval1 <= eval2)
    # Note this actually should be rescaled to initial values (eig *= A_max), 
    # but we are interesting in eigenvector, so let's just skip it. 
    evals = q[...,None] + p[...,None] * betas
    evals = (eig_triple * q)[...,None] + ((1.0 - eval_zero) * (1.0 - eig_triple))[...,None] * evals
    eval0, eval1, eval2 = tf.unstack(evals, axis=-1)

    # Calculate y-magnitude of eigenvector which has the smallest eigenvalue 
    # (= y-magnitude of normal vector from local pcd)
    # by using eigenvector-eigenvalue identity from https://arxiv.org/pdf/1908.03795.pdf
    denom2 = (eval0 - eval1) * (eval0 - eval2)
    denom = tf.sqrt(tf.maximum(denom2, 0))
    evec_zero = tf.cast(denom < 1e-6, tf.float32)
    ny2_num = eval0**2 - (a00 + a22) * eval0 + (a00 * a22 - a02**2)
    ny_num = tf.sqrt(tf.maximum(ny2_num, 0))
    ny = tf.clip_by_value(ny_num / tf.maximum(denom, 1e-6), 0, 1)
    return (1.0 - evec_zero) * ny


''' Traversability map decoder '''
class TmapDecoder(Layer):
    def __init__(self, disp_net, *args, **kwargs):
        super(TmapDecoder, self).__init__(*args, **kwargs)
        self._disp_net = disp_net

    @tf.function
    def call(self, inputs):
        # Note K0 should already be scaled from original image size to nn-input size 
        imgL, imgR, K0, baseline = inputs
        _, h0, w0, _ = tf.unstack(tf.shape(imgL))

        # decode disparity
        disp, _ = self._disp_net([imgL, imgR], False)

        # rescale intrinsic
        _, h1, w1, _ = tf.unstack(tf.shape(disp))
        rw = tf.cast(w1, tf.float32) / tf.cast(w0, tf.float32)
        rh = tf.cast(h1, tf.float32) / tf.cast(h0, tf.float32)
        K_scale = tf.convert_to_tensor([[rw, 0, 0.5*(rw-1)], [0, rh, 0.5*(rh-1)], [0, 0, 1]], dtype=tf.float32)
        K1 = K_scale[None,:,:] @ K0

        # construct point cloud
        fxb = K1[:,0,0] * baseline
        disp = tf.cast(w1, tf.float32) * tf.maximum(disp, 1e-6)
        depth = fxb[:,None,None,None] / disp
        xyz = tf_populate_pcd(depth, K1)

        # detect xz-plane from pcd
        plane_xz = tf_detect_plane_xz(xyz)

        # Condition 1: thresh below camera
        cond1 = tf.cast(xyz[:,:,:,1] > 0.3, tf.float32) 
        # Condition 2: y component of normal vector
        cond2 = tf.cast(plane_xz > 0.85, tf.float32)
        return depth, cond1 * cond2


def warp_topdown(img, K, elevation, fov=5, ppm=20):
    '''
    img: image to warp
    K: camera intrinsic
    elevation: elevation of floor w.r.t the camera (= camera height from floor)
    fov: field of view as meter
    ppm: pixel per meter, new image size = (2* fov * ppm, fov * ppm)
    '''
    fy, cy = K[1,1], K[1,2]
    z_front = fy * elevation / (img.shape[0] - cy)
    src = np.zeros((4, 3), np.float32)
    
    src[0] = [-fov, elevation, z_front+fov]
    src[1] = [fov, elevation, z_front+fov]
    src[2] = [-fov, elevation, z_front]
    src[3] = [fov, elevation, z_front]
    src = src @ K.T
    src /= src[:,2:]
    
    H,W = int(fov * ppm), int(2 * fov * ppm)
    dst = np.array([[0, 0], [W-1, 0], [0, H-1], [W-1, H-1]], np.float32)
    tfm = cv2.getPerspectiveTransform(src[:,None,:-1], dst[:,None,:])
    return cv2.warpPerspective(img, tfm, (W,H))


def get_visual_odometry(tmap, max_angle=30, search_range=(2, 2), passage_width=1, ppm=20):
    Hs, Ws = tmap.shape[:2]
    Wd, Hd = (int(search_range[0] * ppm), int(search_range[1] * ppm))
    xc, y1 = 0.5*(Ws-1), Hs-1 # bottom center coord of tmap
    x0 = xc - 0.5 * search_range[0] * ppm # left
    x1 = xc + 0.5 * search_range[0] * ppm # right
    y0 = y1 - search_range[1] * ppm # top
    src_pts = np.float32([[x0, y0], [x0, y1], [x1, y1]])

    ksize = int(passage_width * ppm)
    kernel_1d = np.sin(np.linspace(0, np.pi, ksize))

    tmax = -1
    offset0, angle0 = 0, 0
    for angle in np.linspace(-max_angle, max_angle, 30):
        rot = np.deg2rad(angle)
        dst_pts = np.float32([[(Hd-1)*np.tan(rot),0], [0,Hd-1], [Wd-1, Hd-1]])
        tfm = cv2.getAffineTransform(src_pts, dst_pts)
        w = cv2.warpAffine(tmap, tfm, (Wd, Hd))
        w_1d = np.sum(w, axis=0) # [H,W] => [W]
        est_passage = np.correlate(w_1d, kernel_1d, mode='valid')
        max_idx = np.argmax(est_passage)
        if est_passage[max_idx] > tmax:
            tmax = est_passage[max_idx]
            offset0 = (max_idx - 0.5 * (Wd-ksize)) / ppm
            angle0 = angle

    return offset0, angle0

    # # display guide
    # offset, angle = max_passage_track
    # row0, col0 = y1, xc + offset*ppm
    # row1, col1 = y0, xc + offset*ppm + (Hd-1) * np.tan(-np.deg2rad(angle))
    # guide = cv2.cvtColor(cv2.convertScaleAbs(tmap, alpha=255), cv2.COLOR_GRAY2RGB)
    # cv2.line(guide, (int(col0), int(row0)), (int(col1), int(row1)), (0,255,0), thickness=3)
    # imgtool.imshow(guide, wait=False)
    # print(max_passage, max_passage_track)
