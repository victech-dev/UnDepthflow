import tensorflow as tf

from tensorflow.python.platform import flags

from monodepth_dataloader import MonodepthDataloader

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

opt = flags.FLAGS

def train(Model, Model_eval):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.Variable(0, trainable=False)
        train_op = tf.train.AdamOptimizer(opt.learning_rate)

        tower_grads = []

        image1, image_r, image2, image2_r, proj_cam2pix, proj_pix2cam = MonodepthDataloader(
            opt).data_batch

        split_image1 = tf.split(
            axis=0, num_or_size_splits=opt.num_gpus, value=image1)
        split_image2 = tf.split(
            axis=0, num_or_size_splits=opt.num_gpus, value=image2)
        split_cam2pix = tf.split(
            axis=0, num_or_size_splits=opt.num_gpus, value=proj_cam2pix)
        split_pix2cam = tf.split(
            axis=0, num_or_size_splits=opt.num_gpus, value=proj_pix2cam)
        split_image_r = tf.split(
            axis=0, num_or_size_splits=opt.num_gpus, value=image_r)
        split_image_r_next = tf.split(
            axis=0, num_or_size_splits=opt.num_gpus, value=image2_r)

        summaries_cpu = tf.get_collection(tf.GraphKeys.SUMMARIES,
                                          tf.get_variable_scope().name)

        with tf.variable_scope(tf.get_variable_scope()) as vs:
            for i in range(opt.num_gpus):
                with tf.device('/gpu:%d' % i):
                    if i == opt.num_gpus - 1:
                        scopename = "model"
                    else:
                        scopename = '%s_%d' % ("tower", i)
                    with tf.name_scope(scopename) as ns:
                        if i == 0:
                            model = Model(
                                split_image1[i],
                                split_image2[i],
                                split_image_r[i],
                                split_image_r_next[i],
                                split_cam2pix[i],
                                split_pix2cam[i],
                                reuse_scope=False,
                                scope=vs)
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

                        else:
                            model = Model(
                                split_image1[i],
                                split_image2[i],
                                split_image_r[i],
                                split_image_r_next[i],
                                split_cam2pix[i],
                                split_pix2cam[i],
                                reuse_scope=True,
                                scope=vs)

                        # VICTECH add regularization loss (why this is missed in original repo?)
                        loss = model.loss                        
                        reg_loss = tf.math.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                        total_loss = loss + reg_loss
                        # Retain the summaries from the final tower.
                        if i == opt.num_gpus - 1:
                            summaries_additional = [tf.summary.scalar("reg_loss", reg_loss)]
                            eval_model = Model_eval(scope=vs)
                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = train_op.compute_gradients(
                            total_loss, var_list=var_train_list)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

        grads = average_gradients(tower_grads)

        # Apply the gradients to adjust the shared variables.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            apply_gradient_op = train_op.apply_gradients(
                grads, global_step=global_step)

        # Create a saver.
        saver = tf.train.Saver(max_to_keep=10)

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries_additional + summaries_cpu)

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
        tf.train.start_queue_runners(sess)

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
            # VICTECH turn off evaluation during training by default
            # if (itr) % (VAL_INTERVAL) == 2 or opt.train_test == "test":
            #     test(sess, eval_model, itr, gt_flows_2012, noc_masks_2012,
            #          gt_flows_2015, noc_masks_2015, gt_masks)


