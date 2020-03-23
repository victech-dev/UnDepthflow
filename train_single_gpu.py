import tensorflow as tf

from tensorflow.python.platform import flags

from monodepth_dataloader_v2 import batch_from_dataset

from eval.evaluate_flow import load_gt_flow_kitti
from eval.evaluate_mask import load_gt_mask
from loss_utils import average_gradients

from test import test

# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 100

# How often to run a batch through the validation model.
VAL_INTERVAL = 20000 # 2500

# How often to save a model checkpoint
SAVE_INTERVAL = 2500

opt = flags.FLAGS

def train(Model, Model_eval):
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        train_op = tf.train.AdamOptimizer(opt.learning_rate)

        with tf.device('/cpu:0'):
            image1, image1r, image2, image2r, proj_cam2pix, proj_pix2cam = batch_from_dataset(opt)

        with tf.variable_scope(tf.get_variable_scope()) as vs, tf.name_scope("model") as ns:
            model = Model(image1, image2, image1r, image2r,
                proj_cam2pix, proj_pix2cam, reuse_scope=False, scope=vs)
            eval_model = Model_eval(scope=vs)

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

            summaries_additional = [tf.summary.scalar("reg_loss", reg_loss)]
            grads = train_op.compute_gradients(
                total_loss, var_list=var_train_list)

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = train_op.apply_gradients(grads, global_step=global_step)

        # Create a saver.
        saver = tf.train.Saver(max_to_keep=10)

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries_additional)

        # Make training session.
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.log_device_placement = False
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        summary_writer = tf.summary.FileWriter(
            opt.trace, graph=sess.graph, flush_secs=10)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        if opt.pretrained_model:
            if opt.train_test == "test" or (not opt.retrain):
                saver.restore(sess, opt.pretrained_model)
            elif opt.mode == "depthflow":
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
        for itr in range(start_itr, opt.num_iterations):
            if opt.train_test == "train":
                _, summary_str, summary_model_str = sess.run(
                    [apply_gradient_op, summary_op, model.summ_op])

                # VICTECH note we are not getting image summaries like original repo
                if (itr) % (SUMMARY_INTERVAL) == 2:
                    summary_writer.add_summary(summary_str, itr)
                    summary_writer.add_summary(summary_model_str, itr)

                if (itr) % (SAVE_INTERVAL) == 2:
                    saver.save(
                        sess, opt.trace + '/model', global_step=global_step)

            print('*** Iteration done:', itr)
            if (itr) % (VAL_INTERVAL) == 100 or opt.train_test == "test":
                test(sess, eval_model, itr, gt_flows_2012, noc_masks_2012,
                     gt_flows_2015, noc_masks_2015, gt_masks)


