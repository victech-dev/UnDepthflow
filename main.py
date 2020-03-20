# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from monodepth_dataloader import MonodepthDataloader
from models import *

from eval.evaluate_flow import load_gt_flow_kitti
from eval.evaluate_mask import load_gt_mask
from loss_utils import average_gradients

from test import test

# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 100

# How often to run a batch through the validation model.
VAL_INTERVAL = 10000 # 2500

# How often to save a model checkpoint
SAVE_INTERVAL = 2500

FLAGS = flags.FLAGS

flags.DEFINE_string('trace', "./", 'directory for model checkpoints.')
flags.DEFINE_integer('num_iterations', 300000,
                     'number of training iterations.')
flags.DEFINE_string('pretrained_model', '',
                    'filepath of a pretrained model to initialize from.')
flags.DEFINE_string(
    'mode', '',
    'selection from four modes of ["flow", "depth", "depthflow", "stereo"]')
flags.DEFINE_string('train_test', 'train', 'whether to train or test')
flags.DEFINE_boolean("retrain", True, "whether to reset the iteration counter")

flags.DEFINE_string('data_dir', '', 'root filepath of data.')
flags.DEFINE_string('train_file',
                    './filenames/kitti_train_files_png_4frames.txt',
                    'training file')
flags.DEFINE_string('gt_2012_dir', '',
                    'directory of ground truth of kitti 2012')
flags.DEFINE_string('gt_2015_dir', '',
                    'directory of ground truth of kitti 2015')

flags.DEFINE_integer('batch_size', 4, 'batch size for training')
flags.DEFINE_float('learning_rate', 0.0001,
                   'the base learning rate of the generator')
flags.DEFINE_integer('num_gpus', 1, 'the number of gpu to use')
flags.DEFINE_float('weight_decay', 0.0001, 'scale of l2 regularization')

flags.DEFINE_integer("img_height", 256, "Image height")
flags.DEFINE_integer("img_width", 832, "Image width")

flags.DEFINE_float("depth_smooth_weight", 10.0, "Weight for depth smoothness")
flags.DEFINE_float("ssim_weight", 0.85,
                   "Weight for using ssim loss in pixel loss")
flags.DEFINE_float("flow_smooth_weight", 10.0, "Weight for flow smoothness")
flags.DEFINE_float("flow_consist_weight", 0.01, "Weight for flow consistent")
flags.DEFINE_float("flow_diff_threshold", 4.0,
                   "threshold when comparing optical flow and rigid flow ")

flags.DEFINE_string('eval_pose', '', 'pose seq to evaluate')

flags.DEFINE_integer("num_scales", 4, "Number of scales: 1/2^0, 1/2^1, ..., 1/2^(n-1)") #FLAGS.num_scales = 4
flags.DEFINE_boolean('eval_flow', False, '')
flags.DEFINE_boolean('eval_depth', False, '')
flags.DEFINE_boolean('eval_mask', False, '')
opt = FLAGS

def main(unused_argv):
    #VICTECH train
    from autoflags import autoflags
    Model, Model_eval = autoflags(opt, 'depthflow', True)
    opt.trace = './results_depthflow'
    opt.train_test = 'train'
    opt.retrain = True
    opt.weight_decay = 0.0001
    #VICTECH

    if FLAGS.trace == "":
        raise Exception("OUT_DIR must be specified")

    print('Constructing models and inputs.')

    if opt.num_gpus == 1:
        from train_single_gpu import train
    else:
        from train_multi_gpu import train
    train(Model, Model_eval)


if __name__ == '__main__':
    app.run()
