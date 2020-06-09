import cv2
import tensorflow as tf
import os
import numpy as np
from pathlib import Path
import re
from tqdm import tqdm
import time
import functools

from disp_net import DispNet
from opt_helper import opt, autoflags
from estimate import tmap_decoder, generate_gmap, get_visual_odometry, get_minimap
import utils

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


def predict_tmap(tf_pred, imgnameL, imgnameR, cat='victech'):
    imgL = utils.imread(imgnameL)
    imgR = utils.imread(imgnameR)
    height, width = imgL.shape[:2] # original height, width
    nK, baseline = utils.query_nK(cat)
    baseline = np.array(baseline, np.float32)

    # rescale to fit nn-input
    imgL_fit, imgR_fit = utils.resize_image_pairs(imgL, imgR, (opt.img_width, opt.img_height), np.float32)

    # calculate cte/ye
    t0 = time.time()
    outputs = tf_pred([imgL_fit[None], imgR_fit[None], nK[None], baseline[None]])
    if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
        p_abs, depth = outputs
        depth = np.squeeze(depth)
        depth = cv2.resize(np.squeeze(depth), (width, height))    
    else:
        p_abs = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        depth = None
    p_abs = np.squeeze(p_abs)
    gmap, eoa = generate_gmap(p_abs, nK)
    cte, ye = get_visual_odometry(gmap)
    t1 = time.time()
    print("*", Path(imgnameL).stem, "elspaed:", t1 - t0, "cte:", cte, "ye:", ye, "eoa:", eoa)

    # display minimap
    minimap = get_minimap(gmap, cte, ye)
    utils.imshow(minimap, wait=False)

    return imgL, depth, nK

def export_to_frozen_saved_model():
    # Workaround for 'TensorFlow Failed to get convolution algorithm'
    # https://medium.com/@JeansPantRushi/fix-for-tensorflow-v2-failed-to-get-convolution-algorithm-b367a088b56e
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    autoflags()
    opt.trace = '' # this should be empty because we have no output when testing
    opt.batch_size = 1
    opt.pretrained_model = '.results_stereosv/weights-log.h5'

    print('* Restoring model')
    disp_net = DispNet('test')
    disp_net.model.load_weights(opt.pretrained_model)
    pred_model = tmap_decoder(disp_net, with_depth=False)
    pred_fn = tf.function(functools.partial(pred_model.call, training=False))
    pred_fn_concrete = pred_fn.get_concrete_function(
        (tf.TensorSpec(shape=(1, 384, 512, 3), dtype=tf.float32, name="imL"),
        tf.TensorSpec(shape=(1, 384, 512, 3), dtype=tf.float32, name="imR"),
        tf.TensorSpec(shape=(1, 3, 3), dtype=tf.float32, name="nK"),
        tf.TensorSpec(shape=(1,), dtype=tf.float32, name="baseline")))

    pred_fn_frozen = convert_variables_to_constants_v2(pred_fn_concrete)
    tf.saved_model.save(pred_model, './frozen_models', signatures=pred_fn_frozen)

    pred_model_reloaded = tf.saved_model.load('./frozen_models')
    print(list(pred_model_reloaded.signatures.keys()))  # ["serving_default"]
    pred_model_reloaded = pred_model_reloaded.signatures["serving_default"]

    # To see saved signature
    #saved_model_cli show --dir .results_stereosv/fullmodel-tf2  --tag_set serve --signature_def serving_default
    def infer(i):
        output = pred_model_reloaded(imL=i[0], imR=i[1], nK=i[2], baseline=i[3])
        return output['output_0']
    pred_fn_reloaded = tf.function(infer)

    # To print all layers
    # layers = [op.name for op in tf_pred_frozen.graph.get_operations()]
    # print("-" * 50)
    # print("Frozen model layers: ")
    # for layer in layers:
    #     print(layer)

    # To print inputs/outputs
    # print("-" * 50)
    # print("Frozen model inputs: ")
    # print(tf_pred_frozen.inputs)
    # print("Frozen model outputs: ")
    # print(tf_pred_frozen.outputs)

    # Test prediction
    data_dir = Path('M:\\Users\\sehee\\camera_taker\\undist_fisheye')
    imgnamesL = sorted(Path(data_dir/'imL').glob('*.png'), key=lambda v: int(v.stem))
    for imgL_path in imgnamesL:
        imgR_path = (data_dir/'imR'/imgL_path.stem).with_suffix('.png')
        img, depth, nK = predict_tmap(pred_fn_reloaded, str(imgL_path), str(imgR_path))
        if cv2.waitKey(0) == 27:
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    from estimate.vis import NavScene

    autoflags()
    opt.trace = '' # this should be empty because we have no output when testing
    opt.batch_size = 1
    opt.pretrained_model = '.results_stereosv/weights-log.h5'

    print('* Restoring model')
    disp_net = DispNet('test')
    disp_net.model.load_weights(opt.pretrained_model)
    pred_model = tmap_decoder(disp_net, with_depth=True)
    tf_pred = tf.function(functools.partial(pred_model.call, training=False))

    ''' point cloud test of office image '''
    data_dir = Path('M:\\Users\\sehee\\camera_taker\\undist_fisheye')
    imgnamesL = sorted(Path(data_dir/'imL').glob('*.png'), key=lambda v: int(v.stem))
    def ns_feeder(index):
        imgnameL = imgnamesL[index % len(imgnamesL)]
        imgnameR = (data_dir/'imR'/imgnameL.stem).with_suffix('.png')
        img, depth, nK = predict_tmap(tf_pred, str(imgnameL), str(imgnameR))
        return img, np.clip(depth, 0, 30), nK

    ''' generate depth map prediction '''
    # nK, baseline = utils.query_nK('victech')
    # baseline = np.array(baseline, np.float32)
    # for imgnameL in tqdm(imgnamesL):
    #     imgnameR = (data_dir/'imR'/imgnameL.stem).with_suffix('.png')
    #     imgL = utils.imread(str(imgnameL))
    #     imgR = utils.imread(str(imgnameR))
    #     imgL, imgR = utils.resize_image_pairs(imgL, imgR, (opt.img_width, opt.img_height), np.float32)
    #     depth, _ = tf_pred([imgL[None], imgR[None], nK[None], baseline[None]])
    #     depth = np.squeeze(depth.numpy())
    #     outpath = (data_dir/'depthL'/imgnameL.stem).with_suffix('.pfm')
    #     outpath.parent.mkdir(parents=True, exist_ok=True)
    #     cv2.imwrite(str(outpath), depth)
    # exit()

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

