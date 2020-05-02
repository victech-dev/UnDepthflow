import tensorflow as tf
from tensorflow.python.platform import app
import os
import numpy as np
import imgtool
import cv2
import open3d as o3d
from pathlib import Path
import re
from misc import write_pfm, query_K, resize_image_pairs
from tqdm import tqdm
import time

from opt_utils import opt, autoflags
from pcdlib import NavScene
from eval.evaluate_flow import load_gt_flow_kitti, get_scaled_intrinsic_matrix, scale_intrinsics
from eval.evaluate_mask import load_gt_mask
from eval.evaluation_utils import width_to_focal
from eval.test_model import TestModel

def predict_disp(sess, model, imgnameL, imgnameR):
    imgL = imgtool.imread(imgnameL, mode='RGB')
    imgR = imgtool.imread(imgnameR, mode='RGB')

    # denoising?
    # imgL = cv2.blur(imgL, ksize=(3,3))
    # imgR = cv2.blur(imgR, ksize=(3,3))
    # imgL = cv2.fastNlMeansDenoisingColored(imgL, None, 10, 10, 7, 21)
    # imgR = cv2.fastNlMeansDenoisingColored(imgR, None, 10, 10, 7, 21)
    # imgL = cv2.bilateralFilter(imgL, 11, 17, 17)
    # imgR = cv2.bilateralFilter(imgR, 11, 17, 17)

    # session run
    imgL_fit = cv2.resize(imgL, (opt.img_width, opt.img_height), interpolation=cv2.INTER_AREA)
    imgR_fit = cv2.resize(imgR, (opt.img_width, opt.img_height), interpolation=cv2.INTER_AREA)

    pred_disp = sess.run(model.pred_disp, 
        feed_dict={model.input_L: imgL_fit[None], model.input_R: imgR_fit[None]})
    pred_disp = np.squeeze(pred_disp) # [1, h, w, 1] => [h, w]

    height, width = imgL.shape[:2] # original height, width
    pred_disp = width * cv2.resize(pred_disp, (width, height))
    return imgL, pred_disp


def predict_depth(sess, model, imgnameL, imgnameR, cat):
    imgL = imgtool.imread(imgnameL, mode='RGB')
    imgR = imgtool.imread(imgnameR, mode='RGB')
    K, baseline = query_K(cat)

    # rescale to fit nn-input
    imgL_fit, imgR_fit, K_fit = resize_image_pairs(imgL, imgR, (opt.img_width, opt.img_height), K=K)
    baseline = np.array(baseline, np.float32)

    # session run
    t0 = time.time()
    pred_disp, pred_tmap = sess.run([model.pred_disp, model.pred_tmap], 
        feed_dict={model.input_L: imgL_fit[None], model.input_R: imgR_fit[None], 
            model.input_K: K_fit[None], model.input_baseline: baseline[None]})
    t1 = time.time()
    print("* elspaed:", t1 - t0)
    pred_disp = np.squeeze(pred_disp) # [1, h, w, 1] => [h, w]

    height, width = imgL.shape[:2] # original height, width
    pred_disp = width * cv2.resize(pred_disp, (width, height))
    depth = K[0,0] * baseline / pred_disp

    tmap = cv2.resize(pred_tmap[0], (640, 480), interpolation=cv2.INTER_LINEAR)
    green = np.zeros((480, 640, 3), np.uint8)
    green[:,:,1] = 255
    green = cv2.addWeighted(green, 0.5, imgL, 0.5, 0.0)
    imgL[tmap > 0.5] = green[tmap > 0.5]
    imgtool.imshow(imgL, wait=False)

    return imgL, depth, K

def main(unused_argv):
    opt.trace = '' # this should be empty because we have no output when testing
    opt.batch_size = 1
    opt.mode = 'stereosv'
    opt.pretrained_model = '.results_stereosv/model-log'
    Model, Model_eval = autoflags()

    with tf.device('/gpu:0'), tf.Graph().as_default():
        print('Constructing models and inputs.')
        imageL = tf.placeholder(tf.float32, [1, opt.img_height, opt.img_width, 3], name='dummy_input_L')
        imageR = tf.placeholder(tf.float32, [1, opt.img_height, opt.img_width, 3], name='dummy_input_R')
        dispL = tf.placeholder(tf.float32, [1, opt.img_height, opt.img_width, 3], name='dummy_disp_L')
        dispR = tf.placeholder(tf.float32, [1, opt.img_height, opt.img_width, 3], name='dummy_disp_R')
        with tf.variable_scope(tf.get_variable_scope()) as vs:
            with tf.name_scope("test_model"):
                _ = Model(imageL, imageR, dispL, dispR, reuse_scope=False, scope=vs)
                model = Model_eval(scope=vs)

        # count weights
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        sizes = [np.prod(v.shape.as_list()) for v in var_list]
        print("*** Total weight count:", np.sum(sizes))

        # Create a saver.
        saver = tf.train.Saver(max_to_keep=10)

        # Make training session.
        config = tf.ConfigProto()
        #config.allow_soft_placement = True
        #config.log_device_placement = False
        config.gpu_options.allow_growth = True
        #config.gpu_options.per_process_gpu_memory_fraction=0.6
        sess = tf.Session(config=config)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, opt.pretrained_model)

        ''' point cloud test of office image of inbo.yeo '''
        data_dir = Path('M:\\Users\\sehee\\camera_taker\\undist_fisheye')
        imgnamesL = sorted(Path(data_dir/'imL').glob('*.png'), key=lambda v: int(v.stem))
        def ns_feeder(index):
            imgnameL = imgnamesL[index % len(imgnamesL)]
            imgnameR = (data_dir/'imR'/imgnameL.stem).with_suffix('.png')
            img, depth, K = predict_depth(sess, model, str(imgnameL), str(imgnameR), cat='victech')
            return img, np.clip(depth, 0, 30), K

        ''' generate disparity map prediction '''
        # for imgnameL in tqdm(imgnamesL):
        #     imgnameR = (data_dir/'imR'/imgnameL.stem).with_suffix('.png')
        #     _, disp = predict_disp(sess, model, str(imgnameL), str(imgnameR))
        #     outpath = (data_dir/'dispL'/imgnameL.stem).with_suffix('.pfm')
        #     outpath.parent.mkdir(parents=True, exist_ok=True)
        #     write_pfm(str(outpath), disp)
        # exit()

        ''' point cloud test of office image of kimys '''
        # data_dir = Path('M:\\Users\\sehee\\StereoCapture_200316_1400\\seq1')
        # imgnamesL = list(Path(data_dir).glob('*_L.jpg'))
        # def ns_feeder(index):
        #     imgnameL = imgnamesL[index % len(imgnamesL)]
        #     imgnameR = (data_dir/imgnameL.stem.replace('_L', '_R')).with_suffix('.jpg')
        #     img, depth, K = predict_depth_vicimg(sess, model, str(imgnameL), str(imgnameR))
        #     return img, np.clip(depth, 0, 30), K

        ''' point cloud test of dexter data '''
        # #data_dir = Path('M:\\datasets\\dexter\\arch1_913')
        # data_dir = Path('M:\\datasets\\dexter\\\KingWashLaundromat')
        # imgnamesL = [f for f in (data_dir/'imL').glob('*.png') if not f.stem.startswith('gray')]
        # imgnamesL = sorted(imgnamesL, key=lambda v: int(v.stem))
        # def ns_feeder(index):
        #     imgnameL = imgnamesL[index % len(imgnamesL)]
        #     imgnameR = (data_dir/'imR'/imgnameL.stem).with_suffix('.png')
        #     img, depth, K = predict_depth(sess, model, str(imgnameL), str(imgnameR), cat='dexter')
        #     return img, np.clip(depth, 0, 30), K

        ''' point cloud test of RealSense data '''
        # data_dir = Path('M:\\datasets\\realsense\\rs_1584686924\\img')
        # idx_regex = re.compile('.*-([0-9]+)$')
        # images = sorted(Path(data_dir).glob('rs-output-Color-*.png'), key=lambda v: int(idx_regex.search(v.stem).group(1)))
        # def ns_feeder(index):
        #     imgname = images[index % len(images)]
        #     depthname = (data_dir/imgname.stem.replace('Color', 'Depth')).with_suffix('.png')
        #     img = imgtool.imread(imgname)
        #     depth = imgtool.imread(depthname) * 0.001
        #     K = np.array([[613, 0, 332], [0, 613, 242], [0, 0, 1]])
        #     K_depth = np.array([[385.345, 0, 320.409], [0, 385.345, 244.852], [0, 0, 1]])
        #     return img, np.clip(depth, 0, 30), K_depth

        scene = NavScene(ns_feeder)
        scene.run()
        scene.clear()


if __name__ == '__main__':
    app.run()
