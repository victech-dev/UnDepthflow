import tensorflow as tf
from tensorflow.keras.layers import Layer, Input
import numpy as np
import cv2
from opt_helper import opt
import utils


class PopulatePointCloud(Layer):
    def __init__(self, *args, **kwargs):
        super(PopulatePointCloud, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        depth_shape = input_shape[0]
        H, W = depth_shape[1:3]
        gx, gy = np.meshgrid(np.arange(W), np.arange(H))
        grid = np.stack([gx, gy, np.ones_like(gx)], axis=-1)[None]
        grid = grid.astype(np.float32)
        init_grid = tf.constant_initializer(grid)
        self.gridbase = self.add_weight('gridbase', shape=(1, H, W, 3), initializer=init_grid, trainable=False)

    def call(self, inputs):
        depth, nK = inputs

        # convert nK(= normalized K) to inv_K
        H, W = depth.shape[1:3]
        iH, iW = 1/H, 1/W
        infx, ncx = 1/nK[:,0,0], nK[:,0,2]
        infy, ncy = 1/nK[:,1,1], nK[:,1,2]
        iK00, iK02 = iW*infx, infx*(0.5*iW - ncx)
        iK11, iK12 = iH*infy, infy*(0.5*iH - ncy)

        xyz = depth * self.gridbase
        x, y, z = tf.unstack(xyz, axis=-1)
        tx = iK00[:,None,None] * x + iK02[:,None,None] * z
        ty = iK11[:,None,None] * y + iK12[:,None,None] * z
        tz = z
        return tf.stack([tx, ty, tz], axis=-1)


''' traversability decoder '''
def tmap_decoder(disp_net, with_depth=False):
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
    B, H, W = disp.shape[0:3]
    fx, cx = nK[:,0,0] * W, nK[:,0,2] * W - 0.5
    fxb = fx * baseline
    disp = W * tf.maximum(disp, 1e-6)
    depth = fxb[:,None,None,None] / disp
    pc = PopulatePointCloud()([depth, nK])

    # simple raycasting on xz-plane at given elevation
    elevation = 0.15 # meter under camera
    true_idx = tf.range(H-1, dtype=tf.int32)[None,:,None] # [1,H,1]
    false_idx = -tf.ones([1, 1, 1], dtype=np.int32) #[1,1,1]
    raycast = (pc[:,:-1,:,1] - elevation) * (pc[:,1:,:,1] - elevation) <= 0.0
    hit_v = tf.where(raycast, true_idx, false_idx)
    hit_v = tf.reduce_max(hit_v, axis=1) # [B, H, W] -> [B, W]

    # collect traversability contour from hit result (on given xz-plane)
    flat_pc = tf.reshape(pc, (-1, 3))
    flat_offset0 = tf.range(B, dtype=tf.int32)[None,:] * (H*W) # batch offset
    flat_offset0 += tf.range(W, dtype=tf.int32)[None,:] # u offset
    flat_offset0 += hit_v * W # v offset
    flat_offset1 = flat_offset0 + W
    p0, p1 = tf.gather(flat_pc, flat_offset0), tf.gather(flat_pc, flat_offset1)
    alpha = tf.clip_by_value((elevation - p0[:,:,1]) / tf.maximum(p1[:,:,1] - p0[:,:,1], 1e-6), 0, 1)
    contour = p0 + (p1 - p0) * alpha[:,:,None]

    # convert contour to polar coordinate
    xs, zs = contour[:,:,0], tf.maximum(contour[:,:,2], 0)
    p_abs = tf.math.sqrt(xs**2 + zs**2)
    p_abs = p_abs * tf.cast(hit_v >= 0, tf.float32)

    if with_depth:
        return tf.keras.Model([imgL, imgR, nK, baseline], [p_abs, depth])
    else:
        return tf.keras.Model([imgL, imgR, nK, baseline], p_abs)


def generate_gmap(p_abs, nK):
    W = len(p_abs)
    fx, cx = nK[0,0] * W, nK[0,2] * W - 0.5
    p_arg = 0.5 * np.pi - np.arctan((np.arange(W, dtype=np.float32) - cx) / fx)

    # test end of aisle
    u_center = int(np.round(cx))
    eoa = np.all(p_abs[u_center-1:u_center+2] < 1.2)

    # prepare gmap (open space)
    ppm = 20
    range_x, range_z = 4, 5
    gH, gW = range_z*ppm, range_x*ppm*2 + 1
    gmap = np.zeros((gH, gW), np.uint8) # 1: open grid, 0: unknown

    # draw occlusion contour
    gpos = np.stack([p_abs * np.cos(p_arg) * ppm + (gW-1) * 0.5, (gH-1) - p_abs * np.sin(p_arg)*ppm], axis=-1)
    gpos = np.round(gpos).astype(np.int32)
    gpos_clip = np.stack([np.clip(gpos[:,0], 0, gW-1), np.clip(gpos[:,1], 0, gH-1)], axis=-1)
    for i in range(W-1):
        cv2.line(gmap, tuple(gpos_clip[i]), tuple(gpos_clip[i+1]), 255, thickness=1)
    cv2.line(gmap, ((gW-1)//2, gH-1), tuple(gpos_clip[0]), 255, thickness=1)
    cv2.line(gmap, ((gW-1)//2, gH-1), tuple(gpos_clip[-1]), 255, thickness=1)

    # fill outside / inverse
    gmap = np.pad(gmap, ((1,1), (1,1)), mode='constant')
    _, gmap, _, _ = cv2.floodFill(gmap, None, (0,0), 255) # fill occ/unknown
    gmap = 255 - gmap[1:-1,1:-1]

    return (gmap / 255).astype(np.float32), eoa


def get_visual_odometry(gmap, max_angle=30, search_range=(2, 2), passage_width=1.5, ppm=20):
    Hs, Ws = gmap.shape[:2]
    Wd, Hd = (int((search_range[0] + passage_width) * ppm), int(search_range[1] * ppm))
    xc, y1 = 0.5*(Ws-1), Hs-1 # bottom center coord of gmap
    x0 = xc - 0.5 * (search_range[0] + passage_width) * ppm # left
    x1 = xc + 0.5 * (search_range[0] + passage_width) * ppm # right
    y0 = y1 - search_range[1] * ppm # top
    src_pts = np.float32([[x0, y0], [x0, y1], [x1, y1]])

    ksize = int(passage_width * ppm)
    kernel_1d = np.sin(np.linspace(0, np.pi, ksize))

    # generate correlation map (rows for different angle, cols for different offset)
    angle_res = 30
    corr = np.zeros((angle_res, Wd - ksize + 1), np.float32)
    for i, angle in enumerate(np.linspace(-max_angle, max_angle, angle_res)):
        rot = np.deg2rad(angle)
        dst_pts = np.float32([[(Hd-1)*np.tan(rot),0], [0,Hd-1], [Wd-1, Hd-1]])
        tfm = cv2.getAffineTransform(src_pts, dst_pts)
        w = cv2.warpAffine(gmap, tfm, (Wd, Hd))
        w_1d = np.sum(w, axis=0) # [H,W] => [W]
        corr[i] = np.correlate(w_1d, kernel_1d, mode='valid')

    _, _, _, (xi, yi) = cv2.minMaxLoc(corr)
    offset = (xi - 0.5 * (Wd - ksize)) / ppm
    angle = 2 * max_angle * (yi / (angle_res - 1)) - max_angle

    # convert offset/angle to cte/yaw error
    cam_pos = (-offset, 0)
    ye = np.deg2rad(angle)
    cte = np.cos(-ye) * cam_pos[0] - np.sin(-ye) * cam_pos[1]

    return cte, ye


def get_minimap(gmap, cte, ye, ppm=20):        
    Hs, Ws = gmap.shape[:2]
    mm = cv2.convertScaleAbs(gmap, alpha=255)
    mm = cv2.cvtColor(mm, cv2.COLOR_GRAY2RGB)
    def _l2m(wpt): # convert [l]ocal xz position to [m]inimap pixel location
        x = 0.5 * (Ws - 1) + wpt[0] * ppm
        y = Hs - 1 - wpt[1] * ppm
        return (int(round(x)), int(round(y)))
    # show robot pos
    cv2.circle(mm, _l2m([0,0]), 3, (255, 255, 0), -1)
    # show track line
    rot = np.array([[np.cos(ye), -np.sin(ye)], [np.sin(ye), np.cos(ye)]])
    pt0 = [-cte, 0] @ rot.T
    pt1 = [-cte, 2] @ rot.T
    cv2.line(mm, _l2m(pt0), _l2m(pt1), (0,255,0), thickness=1)
    return mm


