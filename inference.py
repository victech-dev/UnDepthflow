import cv2
import tensorflow as tf
import os
import numpy as np
import utils
from pathlib import Path
import time
import functools

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.compiler.tensorrt import trt_convert as trt

from disp_net import DispNet
from opt_helper import opt, autoflags

from estimate import tmap_decoder, generate_gmap, get_visual_odometry
from test import predict_tmap

def read_image(imgL_path, imgR_path):
    imgL_path = imgL_path.numpy().decode()
    imgR_path = imgR_path.numpy().decode()

    # note cv2.imread with IMREAD_COLOR would return 3-channels image (without alpha channel)
    imgL = utils.imread(imgL_path)
    imgR = utils.imread(imgR_path)
    H, W = imgL.shape[:2]

    imgL = (imgL / 255).astype(np.float32)
    imgL = cv2.resize(imgL, (opt.img_width, opt.img_height), interpolation=cv2.INTER_AREA)
    imgR = (imgR / 255).astype(np.float32)
    imgR = cv2.resize(imgR, (opt.img_width, opt.img_height), interpolation=cv2.INTER_AREA)

    return imgL, imgR

def batch_from_dataset(num_data, batch_size):
    ds = tf.data.TextLineDataset(opt.train_file)

    # shuffle and take N data
    ds = ds.shuffle(num_data).take(num_data)

    # convert line to (path0, path1, ... path4)
    data_dir = '/ext_ssd/datasets/dexter/'
    def _line2path(x):
        splits = tf.strings.split([x]).values
        imgL_path = tf.strings.join([data_dir, splits[0]])
        imgR_path = tf.strings.join([data_dir, splits[1]])
        return imgL_path, imgR_path
    ds = ds.map(_line2path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # cache for later use
    ds = ds.cache()

    # get k, baseline
    K, baseline = utils.query_K('victech')
    K_fit = utils.rescale_K(K, (640, 480), (opt.img_width, opt.img_height))

    # load image
    def _loaditems(imgL_path, imgR_path):
        imgL, imgR = tf.py_function(
            read_image, 
            [imgL_path, imgR_path],
            (tf.float32, tf.float32))
        return baseline, imgL, imgR, K_fit
    ds = ds.map(_loaditems, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # repeat, batch, prefetch
    ds = ds.repeat(1).batch(batch_size).prefetch(batch_size)
    return ds

def export_to_org_saved_model(out_dir):
    #autoflags()
    opt.trace = '' # this should be empty because we have no output when testing
    opt.batch_size = 1
    opt.pretrained_model = '.results_stereosv/weights-log.h5'

    print('* Restoring model')
    disp_net = DispNet('test')
    disp_net.model.load_weights(opt.pretrained_model, by_name=True)
    pred_model = tmap_decoder(disp_net, with_depth=False)
    pred_fn = tf.function(functools.partial(pred_model.call, training=False))
    pred_fn_concrete = pred_fn.get_concrete_function(
        (tf.TensorSpec(shape=(1, 384, 512, 3), dtype=tf.float32, name="iml"),
        tf.TensorSpec(shape=(1, 384, 512, 3), dtype=tf.float32, name="imr"),
        tf.TensorSpec(shape=(1, 3, 3), dtype=tf.float32, name="nk"),
        tf.TensorSpec(shape=(1,), dtype=tf.float32, name="baseline")))
    tf.saved_model.save(pred_model, out_dir, signatures=pred_fn_concrete)

def export_to_frozen_saved_model(out_dir):
    #autoflags()
    opt.trace = '' # this should be empty because we have no output when testing
    opt.batch_size = 1
    opt.pretrained_model = '.results_stereosv/weights-log.h5'

    print('* Restoring model')
    disp_net = DispNet('test')
    disp_net.model.load_weights(opt.pretrained_model, by_name=True)
    pred_model = tmap_decoder(disp_net, with_depth=False)
    pred_fn = tf.function(functools.partial(pred_model.call, training=False))
    pred_fn_concrete = pred_fn.get_concrete_function(
        (tf.TensorSpec(shape=(1, 384, 512, 3), dtype=tf.float32, name="iml"),
        tf.TensorSpec(shape=(1, 384, 512, 3), dtype=tf.float32, name="imr"),
        tf.TensorSpec(shape=(1, 3, 3), dtype=tf.float32, name="nk"),
        tf.TensorSpec(shape=(1,), dtype=tf.float32, name="baseline")))

    print('* save frozen model')
    pred_fn_frozen = convert_variables_to_constants_v2(pred_fn_concrete)
    tf.saved_model.save(pred_model, out_dir, signatures=pred_fn_frozen)

    # disp_net = create_model(training=False)
    # disp_net.load_weights(opt.pretrained_model)
    # tmap_dec = tmap_decoder(disp_net)
    #tf_pred = tf.function(functools.partial(tmap_dec.call, training=None, mask=None))
    # tf_pred_concrete = tf_pred.get_concrete_function(
    #     (tf.TensorSpec(shape=(1, 384, 512, 3), dtype=tf.float32, name="iml"),
    #     tf.TensorSpec(shape=(1, 384, 512, 3), dtype=tf.float32, name="imr"),
    #     tf.TensorSpec(shape=(1, 3, 3), dtype=tf.float32, name="k"),
    #     tf.TensorSpec(shape=(1,), dtype=tf.float32, name="baseline")))

    # tf_pred_frozen = convert_variables_to_constants_v2(tf_pred_concrete)
    # tf.saved_model.save(tmap_dec, out_dir, signatures=tf_pred_frozen)

def export_to_trt_fp32_model(input_dir, out_dir):
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=input_dir,
        conversion_params=conversion_params,
    )
    converter.convert()
    converter.save(out_dir)
    #predict_test_images(out_dir, 5)

def export_to_trt_fp16_model(input_dir, out_dir):
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    conversion_params = conversion_params._replace(precision_mode='FP16')
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=input_dir,
        conversion_params=conversion_params,
    )
    converter.convert()
    converter.save(out_dir)
    #predict_test_images(out_dir, 5)

def export_to_trt_int8_model(input_dir, out_dir):
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    conversion_params = conversion_params._replace(precision_mode='INT8')

    def input_fn(num_batch, batch_size):
        ds = batch_from_dataset(num_batch * batch_size, batch_size)
        for i, batch_data in enumerate(ds):
            #print("* batch_data type=", type(batch_data))
            #print("* batch_data shapes=", batch_data[0].shape, batch_data[1].shape, batch_data[2].shape, batch_data[3].shape)
            if i >= num_batch:
                break
            yield batch_data
            print("  step %d/%d" % (i + 1, num_batch))
            i += 1

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=input_dir,
        conversion_params=conversion_params,
    )
    calibration_input_fn = functools.partial(input_fn, 256, 1)
    converter.convert(calibration_input_fn=calibration_input_fn)
    converter.save(out_dir)

def predict_test_images(saved_model_path, predict_count):
    pred_model_loaded = tf.saved_model.load(saved_model_path)
    print(list(pred_model_loaded.signatures.keys()))  # ["serving_default"]
    pred_model_loaded = pred_model_loaded.signatures["serving_default"]

    # To see saved signature
    #saved_model_cli show --dir [DIR]  --tag_set serve --signature_def serving_default
    def infer(i):
        output = pred_model_loaded(iml=i[0], imr=i[1], nk=i[2], baseline=i[3])
        return output['output_0']
    pred_fn_loaded = tf.function(infer)

    # Test prediction
    data_dir = Path('/workspace/frozen_models_img')
    imgnamesL = sorted(Path(data_dir/'imL').glob('*.png'), key=lambda v: int(v.stem))
    imgnameL = imgnamesL[10 % len(imgnamesL)]
    imgnameR = (data_dir/'imR'/imgnameL.stem).with_suffix('.png')
    for _ in range(predict_count):
        img, depth, nK = predict_tmap(pred_fn_loaded, str(imgnameL), str(imgnameR), show_minimap=False)

def get_test_images():
    # Test prediction
    data_dir = Path('/workspace/frozen_models_img')
    imgnamesL = sorted(Path(data_dir/'imL').glob('*.png'), key=lambda v: int(v.stem))
    imgnameL = imgnamesL[10 % len(imgnamesL)]
    imgnameR = (data_dir/'imR'/imgnameL.stem).with_suffix('.png')

    imgL = utils.imread(str(imgnameL))
    imgR = utils.imread(str(imgnameR))
    height, width = imgL.shape[:2] # original height, width
    nK, baseline = utils.query_nK('victech')
    baseline = np.array(baseline, np.float32)

    # rescale to fit nn-input
    imgL_fit, imgR_fit = utils.resize_image_pairs(imgL, imgR, (opt.img_width, opt.img_height), np.float32)

    return imgL_fit, imgR_fit, nK, baseline

def predict_test_images_using_rest(predict_count):
    import json
    import requests

    def infer(iml, imr, nk, baseline):
        t0 = time.time()

        json = {
            "instances": [
                {
                    "iml": iml.tolist(),
                    "imr": imr.tolist(),
                    "nk": nk.tolist(),
                    "baseline": baseline.tolist()
                }
            ]
        }

        t1 = time.time()
        
        response = requests.post('http://localhost:8501/v1/models/stereosv/versions/1:predict', json=json)

        t2 = time.time()

        p_abs = np.array(response.json()["predictions"])
        p_abs = np.squeeze(p_abs)
        gmap = generate_gmap(p_abs, nk)
        cte, ye = get_visual_odometry(gmap)

        t3 = time.time()
        print("*"," elap:", t3 - t0, " elap0:", t1 - t0, " elap1:", t2 - t1, " elap2:", t3 - t2, "cte:", cte, "ye:", ye)
        return cte, ye

    iml, imr, nk, baseline = get_test_images()
    infer(iml, imr, nk, baseline)

def predict_test_images_using_grpc(predict_count):
    import grpc
    from tensorflow_serving.apis import predict_pb2
    from tensorflow_serving.apis import prediction_service_pb2_grpc

    channel = grpc.insecure_channel('localhost:8500')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    def infer(iml, imr, nk, baseline):

        t0 = time.time()

        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'stereosv'
        request.model_spec.signature_name = 'serving_default'
        request.inputs['iml'].CopyFrom(tf.make_tensor_proto(iml[None]))
        request.inputs['imr'].CopyFrom(tf.make_tensor_proto(imr[None]))
        request.inputs['nk'].CopyFrom(tf.make_tensor_proto(nk[None]))
        request.inputs['baseline'].CopyFrom(tf.make_tensor_proto(baseline[None]))

        t1 = time.time()

        response = stub.Predict(request, timeout=600)
        
        t2 = time.time()
        #print("[", i, "] elap : ", t2 - t1)

        output_0 = response.outputs['output_0']
        p_abs_shape = tf.TensorShape(output_0.tensor_shape)
        p_abs = np.array(output_0.float_val).reshape(p_abs_shape.as_list())
        p_abs = np.squeeze(p_abs)
        gmap = generate_gmap(p_abs, nk)
        cte, ye = get_visual_odometry(gmap)

        t3 = time.time()
        print("*"," elap:", t3 - t0, " elap0:", t1 - t0, " elap1:", t2 - t1, " elap2:", t3 - t2, "cte:", cte, "ye:", ye)
        return cte, ye
        
    iml, imr, nk, baseline = get_test_images()
    for i in range(predict_count):
        infer(iml, imr, nk, baseline)

if __name__ == '__main__':
    #predict_test_images_using_rest(20)
    predict_test_images_using_grpc(50)

# model_server start command:
# tensorflow_model_server --port=8500 --rest_api_port=8501 \
# --model_name=stereosv --model_base_path=/ext_ssd/stereosv_serve/xxxx_model &
if __name__ == '!__main__':
    # Workaround for 'TensorFlow Failed to get convolution algorithm'
    # https://medium.com/@JeansPantRushi/fix-for-tensorflow-v2-failed-to-get-convolution-algorithm-b367a088b56e
    is_low_end_gpu = False
    if is_low_end_gpu:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    base_dir = '/ext_ssd/stereosv/' # in Xavier external ssd
    org_model_dir = base_dir + 'org_model'
    frozen_model_dir = base_dir + 'frozen_model'
    trt_fp32_model_dir = base_dir + 'frozen_model_trt_fp32'
    trt_fp16_model_dir = base_dir + 'frozen_model_trt_fp16'
    trt_int8_model_dir = base_dir + 'frozen_model_trt_int8'

    export_to_org_saved_model(org_model_dir)
    #export_to_frozen_saved_model(frozen_model_dir)
    #export_to_trt_fp32_model(frozen_model_dir, trt_fp32_model_dir)
    #export_to_trt_fp16_model(frozen_model_dir, trt_fp16_model_dir)
    #export_to_trt_int8_model(frozen_model_dir, trt_int8_model_dir)

    #predict_test_images(trt_fp16_model_dir, 5)
    #predict_test_images(frozen_model_dir, 5)
    
    