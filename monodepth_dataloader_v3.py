import tensorflow as tf
import re
import numpy as np
import imgtool
import cv2
import functools
from opt_utils import opt
from misc import read_pfm

def inject_strong_contrast(img, alpha):
    ''' modified from https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv '''
    brightness = (alpha ** 0.5) * 0.3
    contrast = np.interp(alpha, [0, 1], [1.0, 3]) 
    scale = (1 - brightness) * contrast
    shift = 127 * (1 - contrast) + 255 * brightness * contrast
    return cv2.addWeighted(img, scale, img, 0, shift)

def inject_bayer_pattern_noise(img, pattern='GB'):
    h, w = img.shape[:2]
    gk = np.zeros([h,w], img.dtype)
    bk = np.zeros([h,w], img.dtype)
    rk = np.zeros([h,w], img.dtype)

    # green kernel
    if pattern == 'GB' or pattern == 'GR':
        gk[::2,::2] = 1; gk[1::2,1::2] = 1
    elif pattern == 'BG' or pattern == 'RG':
        gk[::2,1::2] = 1; gk[1::2,::2] = 1

    # blue, red kernel
    if pattern == 'GB':
        bk[1::2,::2] = 1; rk[::2,1::2] = 1
    elif pattern == 'BG':
        bk[1::2,1::2] = 1; rk[::2,::2] = 1
    elif pattern == 'GR':
        bk[::2,1::2] = 1; rk[1::2,::2] = 1
    elif pattern == 'RG':
        bk[::2,::2] = 1; rk[1::2,1::2] = 1
        
    bayer = img[:,:,0] * rk + img[:,:,1] * gk + img[:,:,2] * bk
    cvt_code = dict(GB=cv2.COLOR_BayerGB2RGB, BG=cv2.COLOR_BayerBG2RGB, 
                    GR=cv2.COLOR_BayerGR2RGB, RG=cv2.COLOR_BayerRG2RGB)
    return cv2.cvtColor(bayer, cvt_code[pattern])

def radial_blur(img, iterations):
    h, w = img.shape[:2]
    cx, cy = (w-1) / 2, (h-1) / 2
    cx = np.random.uniform(0.8 * cx, 1.2 * cx)
    cy = np.random.uniform(0.8 * cy, 1.2 * cy)
    zo_mapx, zo_mapy = map(lambda x: x.astype(np.float32), np.meshgrid(range(w), range(h)))
    zi_mapx, zi_mapy = map(lambda x: x.astype(np.float32), np.meshgrid(range(w), range(h)))
    zo_mapx += (zo_mapx - cx) * (2/w) # 1 pixel offset for border of image
    zo_mapy += (zo_mapy - cy) * (2/h)
    zi_mapx -= (zi_mapx - cx) * (2/w)
    zi_mapy -= (zi_mapy - cy) * (2/h)

    for _ in range(iterations):    
        zo = cv2.remap(img, zo_mapx, zo_mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        zi = cv2.remap(img, zi_mapx, zi_mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        img = cv2.addWeighted(zo, 0.5, zi, 0.5, 0) # blend back to src
    return img

def elastic_distorter(W, H, alpha):
    sigma = int(64 * np.random.uniform(1.0, 2.0)) | 1
    delta = cv2.GaussianBlur((np.random.rand(64, 64, 2).astype(np.float32) * 2 - 1), (sigma, sigma), 0)
    scale = alpha / np.max(np.abs(delta))
    delta = cv2.resize(scale * delta, (W,H))
    dx, dy = cv2.split(delta)
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x, y = np.clip(x+dx, 0, W-1).astype(np.float32), np.clip(y+dy, 0, H-1).astype(np.float32)
    return functools.partial(cv2.remap, map1=x, map2=y, 
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def read_image(imgL_path, imgR_path, dispL_path, dispR_path):
    imgL_path = imgL_path.numpy().decode()
    imgR_path = imgR_path.numpy().decode()
    dispL_path = dispL_path.numpy().decode()
    dispR_path = dispR_path.numpy().decode()

    # note cv2.imread with IMREAD_COLOR would return 3-channels image (without alpha channel)
    imgL = cv2.cvtColor(cv2.imread(imgL_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    imgR = cv2.cvtColor(cv2.imread(imgR_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    dispL, _ = read_pfm(dispL_path)
    dispR, _ = read_pfm(dispR_path)
    H, W = imgL.shape[:2]

    # hue shift
    if opt.hue_delta > 0.0:
        imgL_hsv = cv2.cvtColor(imgL, cv2.COLOR_RGB2HSV)
        imgR_hsv = cv2.cvtColor(imgR, cv2.COLOR_RGB2HSV)
        # apply hue noise
        hmax = round(opt.hue_delta * 90)
        h1 = np.random.randint(-hmax, hmax+1)
        h2 = np.clip(np.random.randint(-hmax, hmax+1), h1-hmax, h1+hmax)
        imgL_hsv[:,:,0] = (imgL_hsv[:,:,0].astype(np.int) + h1) % 180
        imgR_hsv[:,:,0] = (imgR_hsv[:,:,0].astype(np.int) + h2) % 180
        imgL = cv2.cvtColor(imgL_hsv, cv2.COLOR_HSV2RGB)
        imgR = cv2.cvtColor(imgR_hsv, cv2.COLOR_HSV2RGB)
    
    # rgb shift
    if isinstance(opt.rgb_shift, (list,tuple)) and len(opt.rgb_shift) == 3:
        smax = np.array(opt.rgb_shift, np.float) * 255
        s1 = np.random.uniform(-smax, smax)
        s2 = np.clip(np.random.uniform(-smax, smax), s1-smax, s1+smax)
        imgL = cv2.add(imgL, s1[None])
        imgR = cv2.add(imgR, s2[None])

    # apply strong contrast
    if opt.strong_contrast:
        alpha1 = np.random.uniform(0.0, 1.0)
        alpha2 = np.clip(alpha1 * np.random.uniform(0.9, 1.1), 0.0, 1.0)
        imgL = inject_strong_contrast(imgL, alpha1)
        imgR = inject_strong_contrast(imgR, alpha2)

    # gamma transform
    if isinstance(opt.gamma_transform, (list,tuple)) and len(opt.gamma_transform) == 2:
        gamma = np.random.uniform(opt.gamma_transform[0], opt.gamma_transform[1])
        table = (np.arange(0, 256.0 / 255, 1.0 / 255) ** gamma) * 255
        table = table.astype(np.uint8)
        imgL = cv2.LUT(imgL, table)
        imgR = cv2.LUT(imgR, table)

    # random zoom-in
    if opt.zoomin_scale > 1.0:
        sx, sy = np.random.uniform(1.0, opt.zoomin_scale, size=2)
        tx = np.random.uniform(0.0, (W-1)*(sx-1))
        ty = np.random.uniform(0.0, (H-1)*(sy-1))
        warp = np.array([[sx, 0, -tx], [0, sy, -ty]], np.float32)
        imgL = cv2.warpAffine(imgL, warp, (W,H), borderMode=cv2.BORDER_REPLICATE)
        imgR = cv2.warpAffine(imgR, warp, (W,H), borderMode=cv2.BORDER_REPLICATE)
        dispL = sx * cv2.warpAffine(dispL, warp, (W,H), borderMode=cv2.BORDER_REPLICATE)
        dispR = sx * cv2.warpAffine(dispR, warp, (W,H), borderMode=cv2.BORDER_REPLICATE)

    # elastic distortion
    if opt.elastic_distort > 0.0:
        alpha1, alpha2 = np.random.rand(2) * opt.elastic_distort
        distortL = elastic_distorter(W, H, alpha1)
        distortR = elastic_distorter(W, H, alpha2)
        imgL, dispL = map(distortL, (imgL, dispL))
        imgR, dispR = map(distortR, (imgR, dispR))

    # bayer_patter noise
    if opt.bayer_pattern:
        imgL = inject_bayer_pattern_noise(imgL, opt.bayer_pattern)
        imgR = inject_bayer_pattern_noise(imgR, opt.bayer_pattern)

    # inject radial blur
    if opt.radial_blur > 0:
        iterations = np.random.randint(0, opt.radial_blur + 1)
        imgL = radial_blur(imgL, iterations)
        imgR = radial_blur(imgR, iterations)

    imgL = (imgL / 255).astype(np.float32)
    imgL = cv2.resize(imgL, (opt.img_width, opt.img_height), interpolation=cv2.INTER_AREA)
    imgR = (imgR / 255).astype(np.float32)
    imgR = cv2.resize(imgR, (opt.img_width, opt.img_height), interpolation=cv2.INTER_AREA)

    dispL = cv2.resize(dispL, (opt.img_width, opt.img_height), interpolation=cv2.INTER_AREA)
    dispR = cv2.resize(dispR, (opt.img_width, opt.img_height), interpolation=cv2.INTER_AREA)
    return imgL, imgR, np.atleast_3d(dispL), np.atleast_3d(dispR)

def batch_from_dataset():
    ds = tf.data.TextLineDataset(opt.train_file)

    # convert line to (path0, path1, ... path4)
    def _line2path(x):
        splits = tf.strings.split([x]).values
        imgL_path = tf.strings.join([opt.data_dir, splits[0]])
        imgR_path = tf.strings.join([opt.data_dir, splits[1]])
        dispL_path = tf.strings.join([opt.data_dir, splits[2]])
        dispR_path = tf.strings.join([opt.data_dir, splits[3]])
        return imgL_path, imgR_path, dispL_path, dispR_path
    ds = ds.map(_line2path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # shuffle whole dataset
    num_lines = sum(1 for line in open(opt.train_file))
    ds = ds.cache().shuffle(num_lines)

    # load image
    def _loaditems(imgL_path, imgR_path, dispL_path, dispR_path):
        imgL, imgR, dispL, dispR = tf.py_function(read_image, 
            [imgL_path, imgR_path, dispL_path, dispR_path], 
            (tf.float32, tf.float32, tf.float32, tf.float32))
        return imgL, imgR, dispL, dispR
    ds = ds.map(_loaditems, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # repeat, batch
    ds = ds.repeat(-1).batch(opt.batch_size)

    # dimension hint required
    def _setshape(imgL, imgR, dispL, dispR):
        imgL.set_shape([opt.batch_size, opt.img_height, opt.img_width, 3])
        imgR.set_shape([opt.batch_size, opt.img_height, opt.img_width, 3])
        dispL.set_shape([opt.batch_size, opt.img_height, opt.img_width, 1])
        dispR.set_shape([opt.batch_size, opt.img_height, opt.img_width, 1])
        return imgL, imgR, dispL, dispR
    ds = ds.map(_setshape, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # prefetch
    ds = ds.prefetch(16)
    return iter(ds)

if __name__ == '__main__':
    import os
    from collections import namedtuple

    opt.data_dir = 'E:/datasets/dexter/'
    opt.train_file = './filenames/dexter_filenames.txt'
    opt.batch_size = 1

    ds_iter = batch_from_dataset()
    for i in range(256):
        imgL, imgR, dispL, dispR = next(ds_iter)
        imgL, imgR = imgL[0].numpy(), imgR[0].numpy()
        dispL, dispR = dispL[0].numpy(), dispR[0].numpy()
        H, W = imgL.shape[:2]
        tiled = np.zeros((H*2, W*2, 3), np.uint8)
        tiled[:H,:W] = cv2.convertScaleAbs(imgL, alpha=255)
        tiled[:H,W:] = cv2.convertScaleAbs(imgR, alpha=255)
        disp = np.concatenate([dispL, dispR], axis=1)
        disp = cv2.normalize(disp, None, 0, 1, cv2.NORM_MINMAX)
        disp = cv2.convertScaleAbs(disp, alpha=255)
        tiled[H:] = np.atleast_3d(disp)
        imgtool.imshow(tiled)


