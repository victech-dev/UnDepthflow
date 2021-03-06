import os
import tensorflow as tf
from tensorflow.python.platform import flags
from pathlib import Path
import re

flags.DEFINE_string('trace', 'AUTO', 'directory for model checkpoints.')
flags.DEFINE_integer('num_iterations', 300000, 'number of training iterations.')
flags.DEFINE_list('pretrained_model', [], 'filepath of a pretrained model to initialize from.')
flags.DEFINE_boolean("freeze_pretrained", False, "whether to freeze pretrained variables")
flags.DEFINE_string('mode', 'depth', 'selection from four modes of ["flow", "depth", "depthflow", "stereo"]')
flags.DEFINE_boolean("retrain", True, "whether to reset the iteration counter")

flags.DEFINE_string('data_dir', '', 'root filepath of data.')
flags.DEFINE_string('train_file', './filenames/kitti_train_files_png_4frames.txt', 'training file')
flags.DEFINE_string('gt_2012_dir', '', 'directory of ground truth of kitti 2012')
flags.DEFINE_string('gt_2015_dir', '', 'directory of ground truth of kitti 2015')

flags.DEFINE_integer('batch_size', 4, 'batch size for training')
flags.DEFINE_list('learning_rate', [1e-4, 1e-5], 'single value or range')
flags.DEFINE_integer('num_gpus', 1, 'the number of gpu to use')
flags.DEFINE_float('weight_decay', 1e-4, 'scale of l2 regularization')

flags.DEFINE_integer("img_height", 256, "Image height")
flags.DEFINE_integer("img_width", 832, "Image width")

# common for all mode
flags.DEFINE_float("ssim_weight", 0.85, "Weight for using ssim loss in pixel loss")
# for stereo
flags.DEFINE_float("disp_smooth_weight", 0.5, "Weight for disparity smoothness")
flags.DEFINE_float("lr_loss_weight", 2.0, "Weight for LR consistency")
# for flow
flags.DEFINE_float("flow_smooth_weight", 0.05, "Weight for flow smoothness")
# for depthflow
flags.DEFINE_float("flow_consist_weight", 0.01, "Weight for flow consistent")
flags.DEFINE_float("flow_diff_threshold", 4.0, "threshold when comparing optical flow and rigid flow ")

flags.DEFINE_string('eval_pose', '', 'pose seq to evaluate')

flags.DEFINE_integer("num_scales", 4, "Number of scales: 1/2^0, 1/2^1, ..., 1/2^(n-1)")
flags.DEFINE_boolean('eval_flow', False, '')
flags.DEFINE_boolean('eval_depth', False, '')
flags.DEFINE_boolean('eval_mask', False, '')
opt = flags.FLAGS


def path_fix_existing(path: Path):
    orgname = str(path.name)
    suffix = 1
    p = re.compile(orgname + r'_([1-9]\d*)$')
    for subdir in path.parent.iterdir():
        match = p.fullmatch(subdir.name)
        if match:
            suffix = max(suffix, int(match.group(1)) + 1)
    return path.parent/f'{orgname}_{suffix}'


def autoflags():
    # data path
    dirs = [os.path.expanduser('~/datasets/kitti_data'),
        '/media/data/datasets/kitti_data', '/media/vicnas/datasets/kitti_data', 
        'C:\\datasets\\kitti_data', 'D:\\datasets\\kitti_data', 
        'E:\\datasets\\kitti_data', 'M:\\datasets\\kitti_data']
    found = next((x for x in dirs if os.path.isdir(x)), None)
    if found is None:
        raise RuntimeError('KITTI data not found!!')

    opt.data_dir = os.path.join(found, 'kitti_raw_data') + os.path.sep
    opt.gt_2012_dir = os.path.join(found, 'kitti_stereo_2012', 'training') + os.path.sep
    opt.gt_2015_dir = os.path.join(found, 'kitti_stereo_2015', 'training') + os.path.sep
    assert os.path.isdir(opt.data_dir)
    assert os.path.isdir(opt.gt_2012_dir)
    assert os.path.isdir(opt.gt_2015_dir)
    print('*** KITTI data path:', found)

    # mode selection
    if opt.mode == "depthflow":  # stage 3: train depth and flow together
        from models import Model_depthflow as Model, Model_eval_depthflow as Model_eval
        opt.eval_flow, opt.eval_depth, opt.eval_mask = True, True, True
    elif opt.mode == "depth":  # stage 2: train depth
        from models import Model_depth as Model, Model_eval_depth as Model_eval
        opt.eval_flow, opt.eval_depth, opt.eval_mask = True, True, False
    elif opt.mode == "flow":  # stage 1: train flow
        from models import Model_flow as Model, Model_eval_flow as Model_eval
        opt.eval_flow, opt.eval_depth, opt.eval_mask = True, False, False
    elif opt.mode == "stereo":
        from models import Model_stereo as Model, Model_eval_stereo as Model_eval
        opt.eval_flow, opt.eval_depth, opt.eval_mask = False, True, False
    else:
        raise "mode must be one of flow, depth, depthflow or stereo"
    print('*** Model mode:', opt.mode)

    if opt.trace == 'AUTO':
        trace = Path(f'.results_{opt.mode}')
        if trace.exists():
            trace = path_fix_existing(trace)
        opt.trace = str(trace)
    print('*** Output path:', opt.trace)

    return Model, Model_eval


def collect_and_restore_variables(scope, sess):
    # collect variables
    with tf.variable_scope(scope):
        var_pose = list(set(tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=".*pose_net.*")))
        var_depth = list(set(tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=".*(depth_net|feature_net_disp).*")))
        var_flow = list(set(tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=".*(flow_net|feature_net_flow).*")))
    var_dict = dict(pose=var_pose, depth=var_depth, flow=var_flow)

    # restore variables
    restored = set()
    for model_path in opt.pretrained_model:
        for net, var_list in var_dict.items():
            if var_list:
                try:
                    saver_pretrained = tf.train.Saver(var_list, max_to_keep=1)
                    saver_pretrained.restore(sess, model_path)
                    print(f"*** [{net}] is restored from {model_path}")
                    restored.update(var_list)
                except:
                    pass
    restored = list(restored)

    # collect trainable variables
    if opt.mode == "depthflow":
        nets_to_train = set(['pose', 'depth', 'flow'])
    elif opt.mode == "depth":
        nets_to_train = set(['pose', 'depth', 'flow'])
    elif opt.mode == "flow":
        nets_to_train = set(['flow'])
    else:
        nets_to_train = set(['depth'])
    var_train_list = sum((var_dict[net] for net in nets_to_train), [])
    if opt.freeze_pretrained:
        var_train_list = list(set(var_train_list) - set(restored))
    print("*** Count of variables to train:", len(var_train_list), "restored:", len(restored))
    return var_train_list, restored


