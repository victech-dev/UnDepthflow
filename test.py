import tensorflow as tf
import os
import numpy as np
import utils
import cv2
import open3d as o3d
from pathlib import Path
import re
from tqdm import tqdm
import time
import functools

from disp_net import create_model
from opt_helper import opt, autoflags
from estimate import NavScene, tmap_decoder, warp_topdown, get_visual_odometry, get_minimap


def predict_disp(model, imgnameL, imgnameR):
    imgL = utils.imread(imgnameL)
    imgR = utils.imread(imgnameR)
    height, width = imgL.shape[:2] # original height, width

    # denoising?
    # imgL = cv2.blur(imgL, ksize=(3,3))
    # imgR = cv2.blur(imgR, ksize=(3,3))
    # imgL = cv2.fastNlMeansDenoisingColored(imgL, None, 10, 10, 7, 21)
    # imgR = cv2.fastNlMeansDenoisingColored(imgR, None, 10, 10, 7, 21)
    # imgL = cv2.bilateralFilter(imgL, 11, 17, 17)
    # imgR = cv2.bilateralFilter(imgR, 11, 17, 17)

    # session run
    imgL_fit, imgR_fit = utils.resize_image_pairs(imgL, imgR, (opt.img_width, opt.img_height), np.float32)
    dispL, _ = model([imgL_fit[None], imgR_fit[None]])
    dispL = np.squeeze(dispL) # [h, w]
    dispL = width * cv2.resize(dispL, (width, height))
    return imgL, dispL


def predict_tmap(tf_pred, imgnameL, imgnameR, cat='victech'):
    imgL = utils.imread(imgnameL)
    imgR = utils.imread(imgnameR)
    height, width = imgL.shape[:2] # original height, width
    K, baseline = utils.query_K(cat)
    baseline = np.array(baseline, np.float32)

    # rescale to fit nn-input
    imgL_fit, imgR_fit, K_fit = utils.resize_image_pairs(imgL, imgR, (opt.img_width, opt.img_height), np.float32, K)

    # calculate cte/ye
    t0 = time.time()
    depth, tmap = tf_pred([imgL_fit[None], imgR_fit[None], K_fit[None], baseline[None]])
    depth = cv2.resize(np.squeeze(depth), (width, height))
    tmap = np.squeeze(tmap)
    K_tmap = utils.rescale_K(K, (width, height), (tmap.shape[1], tmap.shape[0]))
    topdown, zn = warp_topdown(tmap, K_tmap, elevation=0.5, fov=5, ppm=20)
    cte, ye = get_visual_odometry(topdown, zn, max_angle=30)
    t1 = time.time()
    print("* elspaed:", Path(imgnameL).stem, t1 - t0, "cte:", cte, "ye:", ye)

    # display minimap
    minimap = get_minimap(topdown, zn, cte, ye)
    utils.imshow(minimap, wait=False)

    # mark traversability to pcd image
    green = np.full((height, width, 3), (0, 255, 0), np.uint8)
    tmap_enlarged = cv2.resize(tmap, (width, height), interpolation=cv2.INTER_NEAREST)
    alpha = np.atleast_3d(1 - 0.5 * tmap_enlarged)
    img_to_show = np.clip(imgL * alpha + green * (1 - alpha), 0, 255).astype(np.uint8)

    return img_to_show, depth, K


if __name__ == '__main__':
    autoflags()
    opt.trace = '' # this should be empty because we have no output when testing
    opt.batch_size = 1
    opt.pretrained_model = '.results_stereosv/weights-tf2'

    print('* Restoring model')
    disp_net = create_model(training=False)
    disp_net.load_weights(opt.pretrained_model)
    tmap_dec = tmap_decoder(disp_net)
    tf_pred = tf.function(functools.partial(tmap_dec.call, training=None, mask=None))

    # tmap_dec([np.zeros((1, 384, 512, 3), np.float32), np.zeros((1, 384, 512, 3), np.float32), np.eye(3, dtype=np.float32)[None], np.array([1], np.float32)])
    # tmap_dec.save('.results_stereosv/fullmodel-tf2', save_format='tf', include_optimizer=False)

    ''' point cloud test of office image of inbo.yeo '''
    data_dir = Path('M:\\Users\\sehee\\camera_taker\\undist_fisheye')
    imgnamesL = sorted(Path(data_dir/'imL').glob('*.png'), key=lambda v: int(v.stem))
    def ns_feeder(index):
        imgnameL = imgnamesL[index % len(imgnamesL)]
        imgnameR = (data_dir/'imR'/imgnameL.stem).with_suffix('.png')
        img, depth, K = predict_tmap(tf_pred, str(imgnameL), str(imgnameR))
        return img, np.clip(depth, 0, 30), K

    ''' generate disparity map prediction '''
    # for imgnameL in tqdm(imgnamesL):
    #     imgnameR = (data_dir/'imR'/imgnameL.stem).with_suffix('.png')
    #     _, disp = predict_disp(disp_net, str(imgnameL), str(imgnameR))
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
    #     img, depth, K = predict_depth_vicimg(disp_net, str(imgnameL), str(imgnameR))
    #     return img, np.clip(depth, 0, 30), K

    ''' point cloud test of dexter data '''
    # #data_dir = Path('M:\\datasets\\dexter\\arch1_913')
    # data_dir = Path('M:\\datasets\\dexter\\\KingWashLaundromat')
    # imgnamesL = [f for f in (data_dir/'imL').glob('*.png') if not f.stem.startswith('gray')]
    # imgnamesL = sorted(imgnamesL, key=lambda v: int(v.stem))
    # def ns_feeder(index):
    #     imgnameL = imgnamesL[index % len(imgnamesL)]
    #     imgnameR = (data_dir/'imR'/imgnameL.stem).with_suffix('.png')
    #     img, depth, K = predict_depth(disp_net, str(imgnameL), str(imgnameR), cat='dexter')
    #     return img, np.clip(depth, 0, 30), K

    ''' point cloud test of RealSense data '''
    # data_dir = Path('M:\\datasets\\realsense\\rs_1584686924\\img')
    # idx_regex = re.compile('.*-([0-9]+)$')
    # images = sorted(Path(data_dir).glob('rs-output-Color-*.png'), key=lambda v: int(idx_regex.search(v.stem).group(1)))
    # def ns_feeder(index):
    #     imgname = images[index % len(images)]
    #     depthname = (data_dir/imgname.stem.replace('Color', 'Depth')).with_suffix('.png')
    #     img = utils.imread(imgname)
    #     depth = utils.imread(depthname) * 0.001
    #     K = np.array([[613, 0, 332], [0, 613, 242], [0, 0, 1]])
    #     K_depth = np.array([[385.345, 0, 320.409], [0, 385.345, 244.852], [0, 0, 1]])
    #     return img, np.clip(depth, 0, 30), K_depth

    scene = NavScene(ns_feeder)
    scene.run()
    scene.clear()

