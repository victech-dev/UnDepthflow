import tensorflow as tf
import re
import numpy as np
import imgtool
import cv2
from opt_utils import opt

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
    if not hasattr(radial_blur, 'ZOOM_MAP'):
        h, w = img.shape[:2]
        cx, cy = (w-1) / 2, (h-1) / 2
        zoomout_mapx, zoomout_mapy = map(lambda x: x.astype(np.float32), np.meshgrid(range(w), range(h)))
        zoomin_mapx, zoomin_mapy = map(lambda x: x.astype(np.float32), np.meshgrid(range(w), range(h)))
        # 1 pixel offset for border of image
        zoomout_mapx += (zoomout_mapx - cx) * (2/w) 
        zoomout_mapy += (zoomout_mapy - cy) * (2/h)
        zoomin_mapx -= (zoomin_mapx - cx) * (2/w)
        zoomin_mapy -= (zoomin_mapy - cy) * (2/h)
        radial_blur.ZOOM_MAP = (zoomout_mapx, zoomout_mapy, zoomin_mapx, zoomin_mapy)

    zoomout_mapx, zoomout_mapy, zoomin_mapx, zoomin_mapy = radial_blur.ZOOM_MAP
    assert img.shape[:2] == zoomout_mapx.shape[:2]

    for _ in range(iterations):    
        zo = cv2.remap(img, zoomout_mapx, zoomout_mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        zi = cv2.remap(img, zoomin_mapx, zoomin_mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        img = cv2.addWeighted(zo, 0.5, zi, 0.5, 0) # blend back to src
    return img

def read_image(imgL_path, imgR_path):
    if isinstance(imgL_path, bytes):
        imgL_path = imgL_path.decode()
    if isinstance(imgR_path, bytes):
        imgR_path = imgR_path.decode()

    # note cv2.imread with IMREAD_COLOR would return 3-channels image (without alpha channel)
    imgL = cv2.cvtColor(cv2.imread(imgL_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    imgR = cv2.cvtColor(cv2.imread(imgR_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    # noise in HLS space (hue, brightness, contrasts)
    if opt.hue_delta > 0.0 or opt.brightness_delta > 0.0 or opt.contrast_scale > 1.0:
        imgL_hls = cv2.cvtColor(imgL, cv2.COLOR_RGB2HLS)
        imgR_hls = cv2.cvtColor(imgR, cv2.COLOR_RGB2HLS)
        # apply hue noise
        hmax = round(opt.hue_delta * 90)
        h1 = np.random.randint(-hmax, hmax+1)
        h2 = np.clip(np.random.randint(-hmax, hmax+1), h1-hmax, h1+hmax)
        imgL_hls[:,:,0] = (imgL_hls[:,:,0].astype(np.int) + h1) % 180
        imgR_hls[:,:,0] = (imgR_hls[:,:,0].astype(np.int) + h2) % 180
        # apply brightness/contrast noise
        alpha1 = np.random.uniform(1.0, opt.contrast_scale)
        alpha2 = np.clip(alpha1 * np.random.uniform(0.75, 1.25), 1.0, opt.contrast_scale)
        beta1 = np.random.uniform(0.0, opt.brightness_delta)
        beta2 = np.clip(beta1 * np.random.uniform(0.75, 1.25), 0.0, opt.brightness_delta)
        lumL, lumR = imgL_hls[:,:,1] / 255, imgR_hls[:,:,1] / 255
        lum_meanL, lum_meanR = np.mean(lumL), np.mean(lumR)
        lumL = alpha1 * (lumL - lum_meanL) + lum_meanL + beta1
        lumR = alpha2 * (lumR - lum_meanR) + lum_meanR + beta2
        imgL_hls[:,:,1] = np.clip(lumL * 255, 0, 255)
        imgR_hls[:,:,1] = np.clip(lumR * 255, 0, 255)
        imgL = cv2.cvtColor(imgL_hls, cv2.COLOR_HLS2RGB)
        imgR = cv2.cvtColor(imgR_hls, cv2.COLOR_HLS2RGB)

    # rgb shift
    if isinstance(opt.rgb_shift, (list,tuple)) and len(opt.rgb_shift) == 3:
        smax = np.array(opt.rgb_shift, np.float) * 255
        s1 = np.random.uniform(-smax, smax)
        s2 = np.clip(np.random.uniform(-smax, smax), s1-smax, s1+smax)
        imgL = cv2.add(imgL, s1[None])
        imgR = cv2.add(imgR, s2[None])

    # gamma transform
    if isinstance(opt.gamma_transform, (list,tuple)) and len(opt.gamma_transform) == 2:
        gamma = np.random.uniform(opt.gamma_transform[0], opt.gamma_transform[1])
        table = (np.arange(0, 256.0 / 255, 1.0 / 255) ** gamma) * 255
        table = table.astype(np.uint8)
        imgL = cv2.LUT(imgL, table)
        imgR = cv2.LUT(imgR, table)

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

    return imgL, imgR


def read_disparity(disp_path):
    disp = read_pfm(disp_path)
    disp = cv2.resize(disp, (opt.img_width, opt.img_height), interpolation=cv2.INTER_AREA)
    return np.atleast_3d(disp)


def read_pfm(file):
    if isinstance(file, bytes):
        file = file.decode()
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip().decode('utf-8')
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip().decode('utf-8'))
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    #return data, scale
    return data.astype(np.float32)

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
        imgL, imgR = tf.py_func(read_image, [imgL_path, imgR_path], (tf.float32, tf.float32))
        dispL = tf.py_func(read_disparity, [dispL_path], tf.float32)
        dispR = tf.py_func(read_disparity, [dispR_path], tf.float32)
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

    # return as element
    iterator = ds.make_one_shot_iterator()
    next_element = iterator.get_next()
    return next_element

if __name__ == '__main__':
    import os
    from collections import namedtuple

    opt.data_dir = 'M:/datasets/dexter/'
    opt.train_file = './filenames/dexter_filenames.txt'
    opt.batch_size = 1

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        element = batch_from_dataset()
        sess = tf.Session()
        for i in range(256):
            imgL, imgR, dispL, dispR = sess.run(element)
            imgL, imgR = imgL[0], imgR[0]
            dispL, dispR = dispL[0], dispR[0]
            H, W = imgL.shape[:2]
            tiled = np.zeros((H*2, W*2, 3), np.uint8)
            tiled[:H,:W] = cv2.convertScaleAbs(imgL, alpha=255)
            tiled[:H,W:] = cv2.convertScaleAbs(imgR, alpha=255)
            disp = np.concatenate([dispL, dispR], axis=1)
            disp = cv2.normalize(disp, None, 0, 1, cv2.NORM_MINMAX)
            disp = cv2.convertScaleAbs(disp, alpha=255)
            tiled[H:] = np.atleast_3d(disp)
            imgtool.imshow(tiled)


