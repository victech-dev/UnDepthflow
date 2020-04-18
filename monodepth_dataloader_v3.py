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


def apply_brightness_contrast(img, brightness = 0, contrast = 0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(img, alpha_b, img, 0, gamma_b)
    else:
        buf = img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf    


def read_image(image_path):
    if isinstance(image_path, bytes):
        image_path = image_path.decode()
    img = imgtool.imread(image_path)
    
    # if image has 4 channels, we assume that this is RGBA png format
    if img.shape[2] == 4:
        img = img[:,:,:3]
    
    # bayer_patter noise
    if opt.bayer_pattern:
        img = inject_bayer_pattern_noise(img, opt.bayer_pattern)
    img = (img / 255).astype(np.float32)
    img = cv2.resize(img, (opt.img_width, opt.img_height), interpolation=cv2.INTER_AREA)
    return img


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
        imgL = tf.py_func(read_image, [imgL_path], tf.float32)
        imgR = tf.py_func(read_image, [imgR_path], tf.float32)
        dispL = tf.py_func(read_disparity, [dispL_path], tf.float32)
        dispR = tf.py_func(read_disparity, [dispR_path], tf.float32)

        if opt.hue_delta > 0.0:
            h1 = opt.hue_delta * tf.random.uniform([], -1.0, 1.0)
            h2 = h1 * tf.random.uniform([], 0.9, 1.1)
            imgL = tf.image.adjust_hue(imgL, h1)
            imgR = tf.image.adjust_hue(imgR, h2)

        if opt.brightness_delta > 0.0:
            b1 = opt.brightness_delta * tf.random.uniform([], -1.0, 1.0)
            b2 = b1 * tf.random.uniform([], 0.9, 1.1)
            imgL = tf.image.adjust_brightness(imgL, b1)
            imgR = tf.image.adjust_brightness(imgR, b2)
        
        if opt.contrast_scale > 1.0:
            c1 = tf.random.uniform([], 1.0, opt.contrast_scale)
            c2 = c1 * tf.random.uniform([], 0.9, 1.1)
            imgL = tf.image.adjust_contrast(imgL, c1)
            imgR = tf.image.adjust_contrast(imgR, c2)

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

    opt = {}
    opt['data_dir'] = 'M:/datasets/dexter/'
    opt['train_file'] = './filenames/dexter_filenames.txt'
    opt['batch_size'] = 4
    opt['img_height'] = 384
    opt['img_width'] = 512
    opt['num_scales'] = 4
    opt['bayer_pattern'] = 'GB'
    opt['hue_delta'] = 0.08
    opt['brightness_delta'] = 0.15
    opt['contrast_scale'] = 2.0
    Option = namedtuple('Option', opt.keys())
    opt = Option(**opt)

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        element = batch_from_dataset()
        sess = tf.Session()
        for i in range(256):
            imgL, imgR, dispL, dispR = sess.run(element)
            imgtool.imshow(imgL[0])
            imgtool.imshow(dispL[0])


