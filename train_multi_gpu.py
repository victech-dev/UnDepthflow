import tensorflow as tf
import numpy as np
from tqdm import trange
import sys
import functools

from monodepth_dataloader_v2 import batch_from_dataset
from eval.evaluate_flow import load_gt_flow_kitti
from eval.evaluate_mask import load_gt_mask
from loss_utils import average_gradients
from eval_kitti import evaluate_kitti

# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 100

# How often to run a batch through the validation model.
VAL_INTERVAL = 20000 # 2500

# How often to save a model checkpoint
SAVE_INTERVAL = 2500

# Cosine annealing LR Scheduler
def lr_scheduler(lr, prog):
    if isinstance(lr, (float, int)):
        return float(lr)
    elif isinstance(lr, (list, tuple)):
        lr_max, lr_min = lr
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(prog * np.pi))
    else:
        raise ValueError('Invalid learning rate')

def train(Model, Model_eval, opt):
    with tf.Graph().as_default():
        print('*** Constructing models and inputs.')
        global_step = tf.Variable(0, trainable=False)
        lr = tf.placeholder(tf.float32, name='learning_rate')
        train_op = tf.train.AdamOptimizer(lr)
        image1, image1r, image2, image2r, proj_cam2pix, proj_pix2cam = batch_from_dataset(opt)

        split_image1 = tf.split(
            axis=0, num_or_size_splits=opt.num_gpus, value=image1)
        split_image2 = tf.split(
            axis=0, num_or_size_splits=opt.num_gpus, value=image2)
        split_cam2pix = tf.split(
            axis=0, num_or_size_splits=opt.num_gpus, value=proj_cam2pix)
        split_pix2cam = tf.split(
            axis=0, num_or_size_splits=opt.num_gpus, value=proj_pix2cam)
        split_image_r = tf.split(
            axis=0, num_or_size_splits=opt.num_gpus, value=image1r)
        split_image_r_next = tf.split(
            axis=0, num_or_size_splits=opt.num_gpus, value=image2r)

        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()) as vs:
            for i in range(opt.num_gpus):
                with tf.device(f'/gpu:{i}'):
                    scopename = "train_model" if i == 0 else f'tower_{i}'
                    with tf.name_scope(scopename):
                        model = Model(split_image1[i], split_image2[i], 
                            split_image_r[i], split_image_r_next[i], 
                            split_cam2pix[i], split_pix2cam[i], reuse_scope=(i > 0), scope=vs)
                        if i == 0:
                            var_train_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

                        # Retain the summaries from the final tower.
                        if i == opt.num_gpus - 1:
                            reg_loss = tf.math.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                            tf.summary.scalar('reg_loss', reg_loss)
                            tf.summary.scalar('total_loss', model.loss)
                            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scopename)
                            eval_model = Model_eval(scope=vs)

                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = train_op.compute_gradients(model.loss, var_list=var_train_list)
                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

                    print(' - Model created:', scopename)

        grads = average_gradients(tower_grads)

        # Apply the gradients to adjust the shared variables.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            apply_gradient_op = train_op.apply_gradients(
                grads, global_step=global_step)

        # Create a saver.
        saver = tf.train.Saver(max_to_keep=10)

        # Make training session.
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.log_device_placement = False
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        summary_op = tf.summary.merge(summaries)
        summary_writer = tf.summary.FileWriter(
            opt.trace, graph=sess.graph, flush_secs=10)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        if opt.pretrained_model:
            if opt.mode == "depthflow":
                saver_rest = tf.train.Saver(
                    list(
                        set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)) -
                        set(
                            tf.get_collection(
                                tf.GraphKeys.GLOBAL_VARIABLES,
                                scope=".*(Adam_1|Adam).*"))),
                    max_to_keep=1)
                saver_rest.restore(sess, opt.pretrained_model)
            elif opt.mode == "depth":
                saver_flow = tf.train.Saver(
                    tf.get_collection(
                        tf.GraphKeys.MODEL_VARIABLES,
                        scope=".*(flow_net|feature_net_flow).*"),
                    max_to_keep=1)
                saver_flow.restore(sess, opt.pretrained_model)
            else:
                raise Exception(
                    "pretrained_model not used. Please set train_test=test or retrain=False"
                )
            if opt.retrain:
                sess.run(global_step.assign(0))

        start_itr = global_step.eval(session=sess)

        if opt.eval_flow:
            gt_flows_2012, noc_masks_2012 = load_gt_flow_kitti("kitti_2012")
            gt_flows_2015, noc_masks_2015 = load_gt_flow_kitti("kitti")
            gt_masks = load_gt_mask()
        else:
            gt_flows_2012, noc_masks_2012, gt_flows_2015, noc_masks_2015, gt_masks = \
              None, None, None, None, None

        # Run training.
        print('*** Start training')
        lr_func = functools.partial(lr_scheduler, opt.learning_rate)
        for itr in trange(start_itr, opt.num_iterations, file=sys.stdout):
            _, summary_str = sess.run([apply_gradient_op, summary_op],
                feed_dict={lr: lr_func(itr / opt.num_iterations)})

            if (itr) % (SUMMARY_INTERVAL) == 2:
                summary_writer.add_summary(summary_str, itr)

            if (itr) % (SAVE_INTERVAL) == 2 or itr == opt.num_iterations-1:
                saver.save(sess, opt.trace + '/model', global_step=global_step)

            if (itr) % (VAL_INTERVAL) == 100:
                result = evaluate_kitti(sess, eval_model, itr, gt_flows_2012, noc_masks_2012, gt_flows_2015, noc_masks_2015, gt_masks)
                flatten = [(f'{k1}/{k2}', v) for k1, k2v in result.items() for k2, v in k2v.items()]
                summary = tf.Summary()
                for k, v in flatten:
                    summary.value.add(tag=k, simple_value=v)
                summary_writer.add_summary(summary, itr)
                summary_writer.flush()


