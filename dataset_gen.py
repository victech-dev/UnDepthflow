# VICTECH dataset generator for single gpu

import tensorflow as tf

def rescale_intrinsics(raw_cam_mat, opt, orig_height, orig_width):
    fx = raw_cam_mat[0, 0]
    fy = raw_cam_mat[1, 1]
    cx = raw_cam_mat[0, 2]
    cy = raw_cam_mat[1, 2]
    r1 = tf.stack(
        [fx * opt.img_width / orig_width, 0, cx * opt.img_width / orig_width])
    r2 = tf.stack([
        0, fy * opt.img_height / orig_height, cy * opt.img_height / orig_height
    ])
    r3 = tf.constant([0., 0., 1.])
    return tf.stack([r1, r2, r3])


def get_multi_scale_intrinsics(raw_cam_mat, num_scales):
    proj_cam2pix = []
    # Scale the intrinsics accordingly for each scale
    for s in range(num_scales):
        fx = raw_cam_mat[0, 0] / (2**s)
        fy = raw_cam_mat[1, 1] / (2**s)
        cx = raw_cam_mat[0, 2] / (2**s)
        cy = raw_cam_mat[1, 2] / (2**s)
        r1 = tf.stack([fx, 0, cx])
        r2 = tf.stack([0, fy, cy])
        r3 = tf.constant([0., 0., 1.])
        proj_cam2pix.append(tf.stack([r1, r2, r3]))
    proj_cam2pix = tf.stack(proj_cam2pix)
    proj_pix2cam = tf.linalg.inv(proj_cam2pix)
    proj_cam2pix.set_shape([num_scales, 3, 3])
    proj_pix2cam.set_shape([num_scales, 3, 3])
    return proj_cam2pix, proj_pix2cam


def make_intrinsics_matrix(fx, fy, cx, cy):
    # Assumes batch input
    batch_size = fx.get_shape().as_list()[0]
    zeros = tf.zeros_like(fx)
    r1 = tf.stack([fx, zeros, cx], axis=1)
    r2 = tf.stack([zeros, fy, cy], axis=1)
    r3 = tf.constant([0., 0., 1.], shape=[1, 3])
    r3 = tf.tile(r3, [batch_size, 1])
    intrinsics = tf.stack([r1, r2, r3], axis=1)
    return intrinsics


def data_augmentation(im, intrinsics, out_h, out_w):
    # Random scaling
    def random_scaling(im, intrinsics):
        batch_size, in_h, in_w, _ = im.get_shape().as_list()
        scaling = tf.random.uniform([2], 1, 1.15)
        x_scaling = scaling[0]
        y_scaling = scaling[1]
        out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
        out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
        im = tf.image.resize_area(im, [out_h, out_w])
        fx = intrinsics[:, 0, 0] * x_scaling
        fy = intrinsics[:, 1, 1] * y_scaling
        cx = intrinsics[:, 0, 2] * x_scaling
        cy = intrinsics[:, 1, 2] * y_scaling
        intrinsics = make_intrinsics_matrix(fx, fy, cx, cy)
        return im, intrinsics

    # Random cropping
    def random_cropping(im, intrinsics, out_h, out_w):
        # batch_size, in_h, in_w, _ = im.get_shape().as_list()
        batch_size, in_h, in_w, _ = tf.unstack(tf.shape(im))
        offset_y = tf.random.uniform(
            [1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
        offset_x = tf.random.uniform(
            [1], 0, in_w - out_w + 1, dtype=tf.int32)[0]
        im = tf.image.crop_to_bounding_box(im, offset_y, offset_x, out_h,
                                           out_w)
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2] - tf.cast(offset_x, dtype=tf.float32)
        cy = intrinsics[:, 1, 2] - tf.cast(offset_y, dtype=tf.float32)
        intrinsics = make_intrinsics_matrix(fx, fy, cx, cy)
        return im, intrinsics

    im, intrinsics = random_scaling(im, intrinsics)
    im, intrinsics = random_cropping(im, intrinsics, out_h, out_w)
    return im, intrinsics

def read_image(opt, image_path, get_shape=False):
    # tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
    file_extension = tf.strings.substr(image_path, -3, 3)
    file_cond = tf.equal(file_extension, 'jpg')
    image = tf.cond(
        file_cond, lambda: tf.image.decode_jpeg(tf.io.read_file(image_path)),
        lambda: tf.image.decode_png(tf.io.read_file(image_path)))
    orig_height = tf.cast(tf.shape(image)[0], "float32")
    orig_width = tf.cast(tf.shape(image)[1], "float32")
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [opt.img_height, opt.img_width], tf.image.ResizeMethod.AREA)
    return (image, orig_height, orig_width) if get_shape else image

def batch_from_dataset(opt):
    ds = tf.data.TextLineDataset(opt.train_file)

    # convert line to (path0, path1, ... path4)
    def _line2path(x):
        splits = tf.strings.split([x]).values
        left_image_path = tf.strings.join([opt.data_dir, splits[0]])
        right_image_path = tf.strings.join([opt.data_dir, splits[1]])
        next_left_image_path = tf.strings.join([opt.data_dir, splits[2]])
        next_right_image_path = tf.strings.join([opt.data_dir, splits[3]])
        cam_intrinsic_path = tf.strings.join([opt.data_dir, splits[4]])
        return left_image_path, right_image_path, next_left_image_path, next_right_image_path, cam_intrinsic_path
    ds = ds.map(_line2path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # shuffle whole dataset
    num_lines = sum(1 for line in open(opt.train_file))
    ds = ds.cache().shuffle(num_lines)

    # load image
    def _loaditems(left_image_path, right_image_path, next_left_image_path, next_right_image_path, cam_intrinsic_path):
        left_image_o, orig_height, orig_width = read_image(opt, left_image_path, get_shape=True)
        right_image_o = read_image(opt, right_image_path)
        next_left_image_o = read_image(opt, next_left_image_path)
        next_right_image_o = read_image(opt, next_right_image_path)

        # randomly flip images
        do_flip = tf.random.uniform([], 0, 1)
        left_image, right_image = tf.cond(do_flip > 0.5,
            lambda: (tf.image.flip_left_right(right_image_o), tf.image.flip_left_right(left_image_o)),
            lambda: (left_image_o, right_image_o))
        next_left_image, next_right_image = tf.cond(do_flip > 0.5,
            lambda: (tf.image.flip_left_right(next_right_image_o), tf.image.flip_left_right(next_left_image_o)),
            lambda: (next_left_image_o, next_right_image_o))

        do_flip_fb = tf.random.uniform([], 0, 1)
        left_image, right_image, next_left_image, next_right_image = tf.cond(do_flip_fb > 0.5,
            lambda: (next_left_image, next_right_image, left_image, right_image),
            lambda: (left_image, right_image, next_left_image, next_right_image))

        # randomly augment images
        #         do_augment  = tf.random.uniform([], 0, 0)
        #         image_list = [left_image, right_image, next_left_image, next_right_image]
        #         left_image, right_image, next_left_image, next_right_image = tf.cond(do_augment > 0.5, 
        #                                                                              lambda: self.augment_image_list(image_list), 
        #                                                                              lambda: image_list)

        left_image.set_shape([None, None, 3])
        right_image.set_shape([None, None, 3])
        next_left_image.set_shape([None, None, 3])
        next_right_image.set_shape([None, None, 3])

        raw_cam_contents = tf.io.read_file(cam_intrinsic_path)
        raw_cam_contents = tf.strings.strip(raw_cam_contents)
        last_line = tf.strings.split([raw_cam_contents], sep="\n").values[-1]
        raw_cam_vec = tf.strings.to_number(tf.strings.split([last_line]).values[1:])
        raw_cam_mat = tf.reshape(raw_cam_vec, [3, 4])
        raw_cam_mat = raw_cam_mat[0:3, 0:3]
        raw_cam_mat = rescale_intrinsics(raw_cam_mat, opt, orig_height, orig_width)
        proj_cam2pix, proj_pix2cam = get_multi_scale_intrinsics(raw_cam_mat, opt.num_scales)

        return left_image, right_image, next_left_image, next_right_image, proj_cam2pix, proj_pix2cam
    ds = ds.map(_loaditems, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # repeat, batch
    ds = ds.repeat(-1).batch(opt.batch_size)

    # dimension hint required
    def _setshape(img1, img1r, img2, img2r, cam2pix, pix2cam):
        img1.set_shape([opt.batch_size, opt.img_height, opt.img_width, 3])
        img1r.set_shape([opt.batch_size, opt.img_height, opt.img_width, 3])
        img2.set_shape([opt.batch_size, opt.img_height, opt.img_width, 3])
        img2r.set_shape([opt.batch_size, opt.img_height, opt.img_width, 3])
        return img1, img1r, img2, img2r, cam2pix, pix2cam
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
    found = 'M:\\datasets\\kitti_data'
    opt['data_dir'] = os.path.join(found, 'kitti_raw_data') + os.path.sep
    opt['train_file'] = './filenames/kitti_train_files_png_4frames.txt'
    opt['batch_size'] = 4
    opt['img_height'] = 256
    opt['img_width'] = 832
    opt['num_scales'] = 4
    Option = namedtuple('Option', opt.keys())
    opt = Option(**opt)

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        element = batch_from_dataset(opt)
        sess = tf.Session()
        import time
        t0 = time.time()
        for i in range(256):
            img1, img1r, img2, img2r, cam2pix, pix2cam = sess.run(element)
            print(pix2cam)
            cv2.imshow('hoho', img1[0])
            cv2.waitKey(0)
        t1 = time.time()
        print("*** elapsed 256:", t1 - t0)
