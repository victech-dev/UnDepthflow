import os
import tensorflow as tf
from tensorflow.python.platform import flags
from pathlib import Path
import re

flags.DEFINE_string('trace', 'AUTO', 'directory for model checkpoints.')
flags.DEFINE_integer('num_iterations', 300000, 'number of training iterations.')
flags.DEFINE_list('pretrained_model', [], 'filepath of a pretrained model to initialize from.')
flags.DEFINE_boolean("freeze_pretrained", False, "whether to freeze pretrained variables")
flags.DEFINE_string('mode', 'stereosv', 'only stereosv')
flags.DEFINE_boolean("retrain", True, "whether to reset the iteration counter")

flags.DEFINE_string('data_dir', '', 'root filepath of data.')
#flags.DEFINE_string('train_file', './filenames/kitti_train_files_png_4frames.txt', 'training file')
flags.DEFINE_string('train_file', './filenames/dexter_filenames.txt', 'training file')
flags.DEFINE_string('gt_2012_dir', '', 'directory of ground truth of kitti 2012')
flags.DEFINE_string('gt_2015_dir', '', 'directory of ground truth of kitti 2015')

# training data augmentation (preprocessed by on-line manner)
flags.DEFINE_string('bayer_pattern', 'GB', 'bayer patterh noise injection during training')
flags.DEFINE_float('hue_delta', 0.1, 'hue noise injection during training')
flags.DEFINE_boolean('strong_contrast', True, 'over-saturated color effect (i.e, strong contrast) during training')
flags.DEFINE_list('rgb_shift', [0, 0, 0.2], 'rgb shift during training')
flags.DEFINE_list('gamma_transform', [0.7, 1.5], 'adjust gamma during training')
flags.DEFINE_integer('radial_blur', 5, 'max iteration count for radial blur during training')
flags.DEFINE_float('zoomin_scale', 1.2, 'random zoomin during training')
flags.DEFINE_float('elastic_distort', 8, 'elastic distort as pixel during training')

flags.DEFINE_integer('batch_size', 4, 'batch size for training')
flags.DEFINE_list('learning_rate', [1e-4, 1e-5], 'single value or range')
flags.DEFINE_integer('num_gpus', 1, 'the number of gpu to use')
flags.DEFINE_float('weight_decay', 1e-4, 'scale of l2 regularization')

#flags.DEFINE_integer("img_height", 256, "Image height")
#flags.DEFINE_integer("img_width", 832, "Image width")
flags.DEFINE_integer("img_height", 384, "Image height")
flags.DEFINE_integer("img_width", 512, "Image width")

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

# for stereosv
flags.DEFINE_string('loss_metric', 'charbonnier', 'charbonnier or rmsle')

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
    if opt.mode != 'stereosv':
        raise Exception("! Only support stereosv now")

    # data path
    dirs = [os.path.expanduser('~/datasets/dexter'),
        '/media/data/datasets/dexter', '/media/vicnas/datasets/dexter', 
        'C:\\datasets\\dexter', 'D:\\datasets\\dexter', 
        'E:\\datasets\\dexter', 'M:\\datasets\\dexter']
    found = next((x for x in dirs if os.path.isdir(x)), None)
    if found is None:
        raise RuntimeError('Dexter data not found!!')

    opt.data_dir = found + os.path.sep
    assert os.path.isdir(opt.data_dir)
    print('*** Dexter data path:', found)

    if opt.trace == 'AUTO':
        trace = Path(f'.results_{opt.mode}')
        if trace.exists():
            trace = path_fix_existing(trace)
        opt.trace = str(trace)
    print('*** Output path:', opt.trace)



