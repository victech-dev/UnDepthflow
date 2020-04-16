import tensorflow as tf
import re
import numpy as np
import imgtool

def read_image(opt, image_path, get_shape=False):
    # tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
    file_extension = tf.strings.substr(image_path, -3, 3)
    file_cond = tf.equal(file_extension, 'jpg')
    image = tf.cond(
        file_cond, lambda: tf.image.decode_jpeg(tf.io.read_file(image_path)),
        lambda: tf.image.decode_png(tf.io.read_file(image_path)))
    # if image has 4 channels, we assume that this is RGBA png format
    image = tf.cond(tf.equal(tf.shape(image)[2], 3), lambda: image, lambda: image[:,:,:3])
    orig_height = tf.shape(image)[0]
    orig_width = tf.shape(image)[1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [opt.img_height, opt.img_width], tf.image.ResizeMethod.AREA)
    return (image, orig_height, orig_width) if get_shape else image

def read_disparity(opt, disp_path, H, W):
    disp = tf.py_func(read_pfm, [disp_path], tf.float32)
    disp = tf.cond(tf.equal(tf.rank(disp), 3), lambda: disp, lambda: tf.expand_dims(disp, -1))
    #DEBUG!! need to fix this
    #disp.set_shape([H, W, 1])
    #disp /= 640 # normalize <-- UnrealStereo need this, Dexter already normalized
    disp.set_shape([480, 640, 1])
    #DEBUG!!
    disp = tf.image.resize(disp, [opt.img_height, opt.img_width], tf.image.ResizeMethod.AREA)
    return disp

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

def batch_from_dataset(opt):
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
        imgL, H, W = read_image(opt, imgL_path, get_shape=True)
        imgR = read_image(opt, imgR_path)
        dispL = read_disparity(opt, dispL_path, H, W)
        dispR = read_disparity(opt, dispR_path, H, W)
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
    import cv2

    opt = {}
    opt['data_dir'] = 'M:/datasets/unrealstereo/'
    opt['train_file'] = './filenames/filenames.txt'
    opt['batch_size'] = 4
    opt['img_height'] = 384
    opt['img_width'] = 512
    opt['num_scales'] = 4
    Option = namedtuple('Option', opt.keys())
    opt = Option(**opt)

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        element = batch_from_dataset(opt)
        sess = tf.Session()
        for i in range(256):
            imgL, imgR, dispL, dispR = sess.run(element)
            imgtool.imshow(imgL[0])
            imgtool.imshow(dispL[0])


