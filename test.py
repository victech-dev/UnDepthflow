import tensorflow as tf
import os
import numpy as np
import imgtool
import cv2
import open3d as o3d
from pathlib import Path
import re
from tqdm import tqdm
import time

from disp_net import DispNet
from opt_helper import opt, autoflags
from cam_utils import query_K, rescale_K, resize_image_pairs
from estimate import NavScene, TmapDecoder, warp_topdown

def find_passage(tmap, max_angle=30, search_range=(2, 2), passage_width=1, ppm=20):
    Hs, Ws = tmap.shape[:2]
    Wd, Hd = (int(search_range[0] * ppm), int(search_range[1] * ppm))
    xc, y1 = 0.5*(Ws-1), Hs-1 # bottom center coord of tmap
    x0 = xc - 0.5 * search_range[0] * ppm # left
    x1 = xc + 0.5 * search_range[0] * ppm # right
    y0 = y1 - search_range[1] * ppm # top
    src_pts = np.float32([[x0, y0], [x0, y1], [x1, y1]])

    ksize = int(passage_width * ppm)
    kernel_1d = np.sin(np.linspace(0, np.pi, ksize))

    max_passage = -1
    max_passage_track = (0, 0)
    for angle in np.linspace(-max_angle, max_angle, 30):
        rot = np.deg2rad(angle)
        dst_pts = np.float32([[(Hd-1)*np.tan(rot),0], [0,Hd-1], [Wd-1, Hd-1]])
        tfm = cv2.getAffineTransform(src_pts, dst_pts)
        w = cv2.warpAffine(tmap, tfm, (Wd, Hd))
        w_1d = np.sum(w, axis=0) # [H,W] => [W]
        est_passage = np.correlate(w_1d, kernel_1d, mode='valid')
        max_idx = np.argmax(est_passage)
        if est_passage[max_idx] > max_passage:
            max_passage = est_passage[max_idx]
            max_passage_track = ((max_idx - 0.5 * (Wd-ksize)) / ppm, angle)

    # display guide
    offset, angle = max_passage_track
    row0, col0 = y1, xc + offset*ppm
    row1, col1 = y0, xc + offset*ppm + (Hd-1) * np.tan(-np.deg2rad(angle))
    guide = cv2.cvtColor(cv2.convertScaleAbs(tmap, alpha=255), cv2.COLOR_GRAY2RGB)
    cv2.line(guide, (int(col0), int(row0)), (int(col1), int(row1)), (0,255,0), thickness=3)            
    imgtool.imshow(guide, wait=False)
    print(max_passage, max_passage_track)


def predict_disp(model, imgnameL, imgnameR):
    imgL = imgtool.imread(imgnameL)
    imgR = imgtool.imread(imgnameR)

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

    dispL, _ = model.predict_single(imgL_fit, imgR_fit)
    dispL = np.squeeze(dispL) # [h, w, 1] => [h, w]

    height, width = imgL.shape[:2] # original height, width
    dispL = width * cv2.resize(dispL, (width, height))
    return imgL, dispL


def predict_tmap(model, imgnameL, imgnameR, cat):
    imgL = imgtool.imread(imgnameL)
    imgR = imgtool.imread(imgnameR)
    height, width = imgL.shape[:2] # original height, width
    K, baseline = query_K(cat)
    baseline = np.array(baseline, np.float32)

    # rescale to fit nn-input
    imgL_fit, imgR_fit, K_fit = resize_image_pairs(imgL, imgR, (opt.img_width, opt.img_height), np.float32, K)

    # session run
    t0 = time.time()
    dispL, tmap = model([imgL_fit[None], imgR_fit[None], K_fit[None], baseline[None]])
    dispL = np.squeeze(dispL) # normalized disparity of pyramid bottom
    tmap = np.squeeze(tmap)

    inv_depth = width * cv2.resize(dispL, (width, height))
    depth = K[0,0] * baseline / inv_depth

    K_tmap = rescale_K(K, (width, height), (dispL.shape[1], dispL.shape[0]))
    topdown = warp_topdown(tmap, K_tmap, elevation=0.5, fov=5, ppm=20)
    find_passage(topdown, max_angle=30)
    t1 = time.time()
    print("* elspaed:", t1 - t0)

    # mark traversability to pcd image
    img_to_show = np.copy(imgL)
    tmap_enlarged = cv2.resize(tmap, (width, height), interpolation=cv2.INTER_LINEAR)
    green = np.zeros((height, width, 3), np.uint8)
    green[:,:,1] = 255
    green = cv2.addWeighted(green, 0.5, img_to_show, 0.5, 0.0)
    img_to_show[tmap_enlarged > 0.5] = green[tmap_enlarged > 0.5]

    return img_to_show, depth, K

if __name__ == '__main__':
    opt.trace = '' # this should be empty because we have no output when testing
    opt.batch_size = 1
    opt.pretrained_model = '.results_stereosv/model-tf2'

    print('Constructing models and inputs.')
    disp_net = DispNet()
    imgL0 = np.ones((384, 512, 3), np.uint8)
    imgR0 = np.ones((384, 512, 3), np.uint8)
    disp_net.predict_single(imgL0, imgR0)

    # count weights
    var_list = disp_net.trainable_variables
    sizes = [np.prod(v.shape.as_list()) for v in var_list]
    print("*** Total weight count:", np.sum(sizes))

    # load weights
    disp_net.load_weights(opt.pretrained_model)
    tmap_dec = TmapDecoder(disp_net)

    ''' point cloud test of office image of inbo.yeo '''
    data_dir = Path('M:\\Users\\sehee\\camera_taker\\undist_fisheye')
    imgnamesL = sorted(Path(data_dir/'imL').glob('*.png'), key=lambda v: int(v.stem))
    def ns_feeder(index):
        imgnameL = imgnamesL[index % len(imgnamesL)]
        imgnameR = (data_dir/'imR'/imgnameL.stem).with_suffix('.png')
        img, depth, K = predict_tmap(tmap_dec, str(imgnameL), str(imgnameR), cat='victech')
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
    #     img = imgtool.imread(imgname)
    #     depth = imgtool.imread(depthname) * 0.001
    #     K = np.array([[613, 0, 332], [0, 613, 242], [0, 0, 1]])
    #     K_depth = np.array([[385.345, 0, 320.409], [0, 385.345, 244.852], [0, 0, 1]])
    #     return img, np.clip(depth, 0, 30), K_depth

    scene = NavScene(ns_feeder)
    scene.run()
    scene.clear()

