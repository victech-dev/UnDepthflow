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
from opt_utils import collect_and_restore_variables

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
        lr_max, lr_min = map(float, lr)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(prog * np.pi))
    else:
        raise ValueError('Invalid learning rate')

def train(Model, Model_eval, opt):
    with tf.Graph().as_default():
        # Make training session.
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.log_device_placement = False
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        print('*** Constructing models and inputs.')
        global_step = tf.Variable(0, trainable=False, name='global_step')
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

                        # Note variables and summaries reside in first GPU
                        if i == 0:
                            var_train_list, var_restored = collect_and_restore_variables(vs, sess)
                            reg_loss = tf.math.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                            tf.summary.scalar('reg_loss', reg_loss)
                            tf.summary.scalar('total_loss', model.loss)

                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = train_op.compute_gradients(model.loss, var_list=var_train_list)
                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

                    # Create Evaluation model
                    if i == 0:
                        with tf.name_scope("eval_model"):
                            eval_model = Model_eval(scope=vs)

                    print(' - Model created:', scopename)

        grads = average_gradients(tower_grads)

        # Apply the gradients to adjust the shared variables.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            apply_gradient_op = train_op.apply_gradients(
                grads, global_step=global_step)

        # Create a saver.
        saver = tf.train.Saver(max_to_keep=10)

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(opt.trace, graph=sess.graph, flush_secs=10)

        # initialize variables (global + local - restored)
        vars_to_init = set(tf.global_variables() + tf.local_variables()) - set(var_restored)
        sess.run(tf.initialize_variables(list(vars_to_init)))

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
            fetches = dict(train=apply_gradient_op)
            if (itr) % (SUMMARY_INTERVAL) == 2:
                fetches['summary'] = summary_op
            outputs = sess.run(fetches, feed_dict={lr: lr_func(itr / opt.num_iterations)})

            if 'summary' in outputs:
                summary_writer.add_summary(outputs['summary'], itr)

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


