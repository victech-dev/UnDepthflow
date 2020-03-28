import tensorflow as tf
from tqdm import trange
import sys

from monodepth_dataloader_v2 import batch_from_dataset
from eval.evaluate_flow import load_gt_flow_kitti
from eval.evaluate_mask import load_gt_mask
from loss_utils import average_gradients
from eval.evaluate_kitti import evaluate_kitti

# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 100

# How often to run a batch through the validation model.
VAL_INTERVAL = 20000 # 2500

# How often to save a model checkpoint
SAVE_INTERVAL = 2500

def train(Model, Model_eval, opt):
    with tf.Graph().as_default():
        print('*** Constructing models and inputs.')
        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdamOptimizer(opt.learning_rate)
        image1, image1r, image2, image2r, proj_cam2pix, proj_pix2cam = batch_from_dataset(opt)

        with tf.variable_scope(tf.get_variable_scope()) as vs:
            with tf.name_scope("train_model") as ns:
                model = Model(image1, image2, image1r, image2r,
                    proj_cam2pix, proj_pix2cam, reuse_scope=False, scope=vs)

                var_pose = list(
                    set(
                        tf.get_collection(
                            tf.GraphKeys.TRAINABLE_VARIABLES,
                            scope=".*pose_net.*")))
                var_depth = list(
                    set(
                        tf.get_collection(
                            tf.GraphKeys.TRAINABLE_VARIABLES,
                            scope=".*(depth_net|feature_net_disp).*"
                        )))
                var_flow = list(
                    set(
                        tf.get_collection(
                            tf.GraphKeys.TRAINABLE_VARIABLES,
                            scope=".*(flow_net|feature_net_flow).*"
                        )))

                if opt.mode == "depthflow":
                    var_train_list = var_pose + var_depth + var_flow
                elif opt.mode == "depth":
                    var_train_list = var_pose + var_depth
                elif opt.mode == "flow":
                    var_train_list = var_flow
                else:
                    var_train_list = var_depth

                # VICTECH add regularization loss (why this is missed in original repo?)
                loss = model.loss                        
                reg_loss = tf.math.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                # total_loss = loss + reg_loss (for now, turn off regularization)
                total_loss = loss
                train_op = optimizer.minimize(total_loss, global_step=global_step)
                tf.summary.scalar('reg_loss', reg_loss)
                tf.summary.scalar('total_loss', total_loss)

            with tf.name_scope("eval_model") as ns:
                eval_model = Model_eval(scope=vs)

        # Create a saver.
        saver = tf.train.Saver(max_to_keep=10)

        # Make training session.
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.log_device_placement = False
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(opt.trace, graph=sess.graph, flush_secs=10)

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

        print('*** Loading gt data for evaluation if required')
        if opt.eval_flow:
            gt_flows_2012, noc_masks_2012 = load_gt_flow_kitti("kitti_2012")
            gt_flows_2015, noc_masks_2015 = load_gt_flow_kitti("kitti")
            gt_masks = load_gt_mask()
        else:
            gt_flows_2012, noc_masks_2012, gt_flows_2015, noc_masks_2015, gt_masks = \
              None, None, None, None, None

        # Run training.
        print('*** Start training')
        for itr in trange(start_itr, opt.num_iterations, file=sys.stdout):
            _, summary_str = sess.run([train_op, summary_op])

            if (itr) % (SUMMARY_INTERVAL) == 2:
                summary_writer.add_summary(summary_str, itr)

            if (itr) % (SAVE_INTERVAL) == 2 or itr == opt.num_iterations-1:
                saver.save(sess, opt.trace + '/model', global_step=global_step)

            # Launch tensorboard
            if itr == 16:
                from launch_tensorboard import launch_tensorboard
                launch_tensorboard(opt.trace)
                print('*** Tensorboard launched')

            # Evaluate and write to tensorboard
            if (itr) % (VAL_INTERVAL) == 100:
                result = evaluate_kitti(sess, eval_model, itr, gt_flows_2012, noc_masks_2012, gt_flows_2015, noc_masks_2015, gt_masks)
                flatten = [(f'{k1}/{k2}', v) for k1, k2v in result.items() for k2, v in k2v.items()]
                summary = tf.Summary()
                for k, v in flatten:
                    summary.value.add(tag=k, simple_value=v)
                summary_writer.add_summary(summary, itr)
                summary_writer.flush()

    print('*** Done')
