import tensorflow as tf
from tensorflow.keras.layers import Layer, Input
import numpy as np
import cv2
from opt_helper import opt
import utils

class PopulatePointCloud(Layer):
    def __init__(self, *args, **kwargs):
        super(PopulatePointCloud, self).__init__(*args, **kwargs)
        self._grid = None
        self._H = 0
        self._W = 0

    def build(self, input_shape):
        depth_shape = input_shape[0]
        self._H, self._W = depth_shape[1:3]
        gx, gy = np.meshgrid(np.arange(self._W), np.arange(self._H))
        grid = np.stack([gx, gy, np.ones_like(gx)], axis=-1)[None]
        grid = grid.astype(np.float32)
        init_grid = tf.constant_initializer(grid)
        self._grid = self.add_weight(shape=(1, self._H, self._W, 3), initializer=init_grid, trainable=False)

    def call(self, inputs):
        depth, nK = inputs

        # convert nK(= normalized K) to inv_K
        H, W = self._H, self._W
        iH, iW = 1/H, 1/W
        infx, ncx = 1/nK[:,0,0], nK[:,0,2]
        infy, ncy = 1/nK[:,1,1], nK[:,1,2]
        iK00, iK02 = iW*infx, infx*(0.5*iW - ncx)
        iK11, iK12 = iH*infy, infy*(0.5*iH - ncy)

        xyz = depth * self._grid
        x, y, z = tf.unstack(xyz, axis=-1)
        tx = iK00[:,None,None] * x + iK02[:,None,None] * z
        ty = iK11[:,None,None] * y + iK12[:,None,None] * z
        tz = z
        return tf.stack([tx, ty, tz], axis=-1)


def tf_detect_plane_xz(xyz):
    def _outer_prod3(vec3):
        x, y, z = tf.unstack(vec3, axis=-1)
        x2, y2, z2, xy, yz, zx = x**2, y**2, z**2, x*y, y*z, z*x
        c0 = tf.stack([x2, xy, zx], axis=-1)
        c1 = tf.stack([xy, y2, yz], axis=-1)
        c2 = tf.stack([zx, yz, z2], axis=-1)
        return tf.stack([c0, c1, c2], axis=-1)

    # Calculate covariance matrix of neighboring points
    # http://jacoposerafin.com/wp-content/uploads/bogoslavskyi13ecmr.pdf
    ksize = 5
    P = tf.nn.avg_pool2d(xyz, ksize, 1, 'SAME')
    xyz2 = _outer_prod3(xyz)
    xyz2_4d = tf.reshape(xyz2, tf.concat([tf.shape(xyz2)[:-2], [9]], 0))
    S_4d = tf.nn.avg_pool2d(xyz2_4d, ksize, 1, 'SAME')
    S = tf.reshape(S_4d, tf.concat([tf.shape(S_4d)[:-1], [3, 3]], 0))
    sigma = S - _outer_prod3(P)

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
def tmap_decoder(disp_net):
    imgL = Input(shape=(opt.img_height, opt.img_width, 3), batch_size=1, dtype='float32')
    imgR = Input(shape=(opt.img_height, opt.img_width, 3), batch_size=1, dtype='float32')
    nK = Input(shape=(3, 3), batch_size=1, dtype='float32')
    baseline = Input(shape=(), batch_size=1, dtype='float32')

    # decode disparity
    featL = disp_net.feat(imgL)
    featR = disp_net.feat(imgR)
    pyr_disp = disp_net.pwcL(featL + featR)
    disp = pyr_disp[0]

    # construct point cloud
    W = disp.shape[2]
    fxb = (nK[:,0,0] * W) * baseline
    disp = W * tf.maximum(disp, 1e-6)
    depth = fxb[:,None,None,None] / disp
    xyz = PopulatePointCloud()([depth, nK])

    # detect xz-plane from pcd
    plane_xz = tf_detect_plane_xz(xyz)

    # Condition 1: thresh below camera
    cond1 = tf.cast(xyz[:,:,:,1] > 0.25, tf.float32) 
    # Condition 2: y component of normal vector
    cond2 = tf.cast(plane_xz > 0.85, tf.float32)

    return tf.keras.Model([imgL, imgR, nK, baseline], [depth, cond1 * cond2])


def warp_topdown(img, nK, elevation, fov=5, ppm=20):
    '''
    img: image to warp
    nK: normalized camera intrinsic
    elevation: elevation of floor w.r.t the camera (= camera height from floor)
    fov: field of view as meter
    ppm: pixel per meter, new image size = (2* fov * ppm, fov * ppm)
    '''
    # let zn = z [n]ear = minimum z among floor pixels (elevation below camera)
    # this can be calculated from eq K * [0 elevation zn].T = zn * [0 imgH-1 1].T
    K = utils.unnormalize_K(nK, (img.shape[1], img.shape[0]))
    fy, cy = K[1,1], K[1,2]
    zn = fy * elevation / (img.shape[0] - 1 - cy)

    src = np.zeros((4, 3), np.float32)
    src[0] = [-fov, elevation, zn + fov]
    src[1] = [fov, elevation, zn + fov]
    src[2] = [-fov, elevation, zn]
    src[3] = [fov, elevation, zn]
    src = src @ K.T
    src /= src[:,2:]
    
    H,W = int(fov * ppm), int(2 * fov * ppm)
    dst = np.array([[0, 0], [W-1, 0], [0, H-1], [W-1, H-1]], np.float32)
    tfm = cv2.getPerspectiveTransform(src[:,None,:-1], dst[:,None,:])
    return cv2.warpPerspective(img, tfm, (W,H)), zn


def get_visual_odometry(tmap, zn, max_angle=30, search_range=(2, 2), passage_width=3, ppm=20):
    ''' tmap: topdown of original traversibility map '''
    Hs, Ws = tmap.shape[:2]
    Wd, Hd = (int((search_range[0] + passage_width) * ppm), int(search_range[1] * ppm))
    xc, y1 = 0.5*(Ws-1), Hs-1 # bottom center coord of tmap
    x0 = xc - 0.5 * (search_range[0] + passage_width) * ppm # left
    x1 = xc + 0.5 * (search_range[0] + passage_width) * ppm # right
    y0 = y1 - search_range[1] * ppm # top
    src_pts = np.float32([[x0, y0], [x0, y1], [x1, y1]])

    ksize = int(passage_width * ppm)
    kernel_1d = np.sin(np.linspace(0, np.pi, ksize)) ** 2

    # generate correlation map (rows for different angle, cols for different offset)
    angle_res = 30
    corr = np.zeros((angle_res, Wd - ksize + 1), np.float32)
    for i, angle in enumerate(np.linspace(-max_angle, max_angle, angle_res)):
        rot = np.deg2rad(angle)
        dst_pts = np.float32([[(Hd-1)*np.tan(rot),0], [0,Hd-1], [Wd-1, Hd-1]])
        tfm = cv2.getAffineTransform(src_pts, dst_pts)
        w = cv2.warpAffine(tmap, tfm, (Wd, Hd))
        w_1d = np.sum(w, axis=0) # [H,W] => [W]
        corr[i] = np.correlate(w_1d, kernel_1d, mode='valid')

    _, _, _, (xi, yi) = cv2.minMaxLoc(corr)
    offset = (xi - 0.5 * (Wd - ksize)) / ppm
    angle = 2 * max_angle * (yi / (angle_res - 1)) - max_angle

    # convert offset/angle to cte/yaw error
    cam_pos = (-offset, -zn)
    ye = np.deg2rad(angle)
    cte = np.cos(-ye) * cam_pos[0] - np.sin(-ye) * cam_pos[1]
    return cte, ye


def get_minimap(tmap, zn, cte, ye, ppm=20):        
    Hs, Ws = tmap.shape[:2]
    pad_bottom = int(np.ceil(zn * ppm))
    mm = cv2.copyMakeBorder(tmap, 0, pad_bottom, 0, 0, cv2.BORDER_CONSTANT)
    mm = cv2.convertScaleAbs(mm, alpha=255)
    mm = cv2.cvtColor(mm, cv2.COLOR_GRAY2RGB)
    def _l2m(wpt): # convert [l]ocal xz position to [m]inimap pixel location
        x = 0.5 * (Ws - 1) + wpt[0] * ppm
        y = Hs - 1 + zn * ppm - wpt[1] * ppm
        return (int(round(x)), int(round(y)))
    # show robot pos
    cv2.circle(mm, _l2m([0,0]), 3, (255, 255, 0), -1)
    # show track line
    rot = np.array([[np.cos(ye), -np.sin(ye)], [np.sin(ye), np.cos(ye)]])
    pt0 = [-cte, 0] @ rot.T
    pt1 = [-cte, zn + 2] @ rot.T
    cv2.line(mm, _l2m(pt0), _l2m(pt1), (0,255,0), thickness=1)
    return mm

