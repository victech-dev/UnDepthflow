import numpy as np
import tensorflow as tf
from tensorflow.python.platform import app
import imgtool

from eval.evaluate_depth import load_depths, eval_depth
from eval.evaluate_flow import load_gt_flow_kitti, get_scaled_intrinsic_matrix, eval_flow_avg
from eval.evaluate_mask import eval_mask
from eval.evaluate_disp import eval_disp_avg
from eval.pose_evaluation_utils import pred_pose
from eval.eval_pose import eval_snippet, kittiEvalOdom
from eval.evaluate_mask import load_gt_mask

import re, os
import sys

from autoflags import opt, autoflags

def _parse_to_dict(result):
    # parse string 'name1,name2,...\nvalue1,value2,...' to dict of {name1:value1, name2:value2, ...}
    lines = list(filter(None, result.splitlines()))
    if len(lines) != 2:
        raise Exception('Cannot parse text to k,v pairs')
    keys = lines[0].replace(',', ' ').split()
    values = lines[1].replace(',', ' ').split()
    if len(keys) != len(values):
        raise Exception('Cannot parse text to k,v pairs')
    return dict([(k,float(v)) for k,v in zip(keys, values)])

def evaluate_kitti(sess, eval_model, itr, gt_flows_2012, noc_masks_2012, gt_flows_2015,
         noc_masks_2015, gt_masks):
    summaries = {}
    with tf.name_scope("evaluation"):
        sys.stderr.write("Evaluation at iter [" + str(itr) + "]: \n")
        if opt.eval_pose != "":
            seqs = opt.eval_pose.split(",")
            odom_eval = kittiEvalOdom("./pose_gt_data/")
            odom_eval.eval_seqs = seqs
            pred_pose(eval_model, opt, sess, seqs)

            for seq_no in seqs:
                sys.stderr.write("pose seq %s: \n" % seq_no)
                eval_snippet(
                    os.path.join(opt.trace, "pred_poses", seq_no),
                    os.path.join("./pose_gt_data/", seq_no))
            odom_eval.eval(opt.trace + "/pred_poses/")
            sys.stderr.write("pose_prediction_finished \n")
        for eval_data in ["kitti_2012", "kitti_2015"]:
            test_result_disp, test_result_flow_rigid, test_result_flow_optical, \
            test_result_mask, test_result_disp2, test_image1 = [], [], [], [], [], []

            if eval_data == "kitti_2012":
                total_img_num = 194
                gt_dir = opt.gt_2012_dir
                img1_dir, img2_dir, calib_dir = "colored_0", "colored_1", 'calib'
            else:
                total_img_num = 200
                gt_dir = opt.gt_2015_dir 
                img1_dir, img2_dir, calib_dir = "image_2", "image_3", 'calib_cam_to_cam'

            for i in range(total_img_num):
                img1 = imgtool.imread(os.path.join(gt_dir, img1_dir, f'{i:06}_10.png'))
                img1_orig = img1
                orig_H, orig_W = img1.shape[0:2]
                img1 = imgtool.imresize(img1, (opt.img_height, opt.img_width))

                img2 = imgtool.imread(os.path.join(gt_dir, img1_dir, f'{i:06}_11.png'))
                img2 = imgtool.imresize(img2, (opt.img_height, opt.img_width))

                imgr = imgtool.imread(os.path.join(gt_dir, img2_dir, f'{i:06}_10.png'))
                imgr = imgtool.imresize(imgr, (opt.img_height, opt.img_width))

                img2r = imgtool.imread(os.path.join(gt_dir, img2_dir, f'{i:06}_11.png'))
                img2r = imgtool.imresize(img2r, (opt.img_height, opt.img_width))

                img1 = np.expand_dims(img1, axis=0)
                img2 = np.expand_dims(img2, axis=0)
                imgr = np.expand_dims(imgr, axis=0)
                img2r = np.expand_dims(img2r, axis=0)

                calib_file = os.path.join(gt_dir, calib_dir,
                                          str(i).zfill(6) + ".txt")

                input_intrinsic = get_scaled_intrinsic_matrix(
                    calib_file,
                    zoom_x=1.0 * opt.img_width / orig_W,
                    zoom_y=1.0 * opt.img_height / orig_H)

                pred_flow_rigid, pred_flow_optical, \
                pred_disp, pred_disp2, pred_mask= sess.run([eval_model.pred_flow_rigid,
                                                         eval_model.pred_flow_optical,
                                                         eval_model.pred_disp,
                                                         eval_model.pred_disp2,
                                                         eval_model.pred_mask],
                                                          feed_dict = {eval_model.input_1: img1,
                                                                       eval_model.input_2: img2,
                                                                       eval_model.input_r: imgr,
                                                                       eval_model.input_2r:img2r,
                                                                       eval_model.input_intrinsic: input_intrinsic})

                test_result_flow_rigid.append(np.squeeze(pred_flow_rigid))
                test_result_flow_optical.append(np.squeeze(pred_flow_optical))
                test_result_disp.append(np.squeeze(pred_disp))
                test_result_disp2.append(np.squeeze(pred_disp2))
                test_result_mask.append(np.squeeze(pred_mask))
                test_image1.append(img1_orig)

            ## depth evaluation
            if opt.eval_depth and eval_data == "kitti_2015":
                sys.stderr.write(f"Evaluate depth at iter [{itr}] {eval_data}\n")
                gt_depths, pred_depths, gt_disparities, pred_disp_resized = load_depths(
                    test_result_disp, gt_dir, eval_occ=True)
                abs_rel, sq_rel, rms, log_rms, a1, a2, a3, d1_all = eval_depth(
                    gt_depths, pred_depths, gt_disparities, pred_disp_resized)
                depth_err = "{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10} \n". \
                    format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3') + \
                    "{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f} \n". \
                    format(abs_rel, sq_rel, rms, log_rms, d1_all, a1, a2, a3)
                sys.stderr.write(depth_err); summaries['kitti_2015_depth'] = _parse_to_dict(depth_err)

                disp_err = eval_disp_avg(test_result_disp, gt_dir, disp_num=0, moving_masks=gt_masks)
                sys.stderr.write("disp err 2015 is \n")
                sys.stderr.write(disp_err + "\n"); summaries['kitti_2015_disp'] = _parse_to_dict(disp_err)

                if opt.mode == "depthflow":
                    disp_err = eval_disp_avg(test_result_disp2, gt_dir, disp_num=1, moving_masks=gt_masks)
                    sys.stderr.write("disp2 err 2015 is \n")
                    sys.stderr.write(disp_err + "\n"); summaries['kitti_2015_disp2'] = _parse_to_dict(disp_err)

            if opt.eval_depth and eval_data == "kitti_2012":
                disp_err = eval_disp_avg(test_result_disp, gt_dir)
                sys.stderr.write("disp err 2012 is \n")
                sys.stderr.write(disp_err + "\n"); summaries['kitti_2012_disp'] = _parse_to_dict(disp_err)

            # flow evaluation
            if opt.eval_flow and eval_data == "kitti_2012":
                if opt.mode in ["depth", "depthflow"]:
                    epe = eval_flow_avg(gt_flows_2012, noc_masks_2012, test_result_flow_rigid, opt)
                    sys.stderr.write("epe 2012 rigid is \n")
                    sys.stderr.write(epe + "\n"); summaries['kitti_2012_flow_rigid'] = _parse_to_dict(epe)

                epe = eval_flow_avg(gt_flows_2012, noc_masks_2012, test_result_flow_optical, opt)
                sys.stderr.write("epe 2012 optical is \n")
                sys.stderr.write(epe + "\n"); summaries['kitti_2012_flow_optical'] = _parse_to_dict(epe)

            if opt.eval_flow and eval_data == "kitti_2015":
                if opt.mode in ["depth", "depthflow"]:
                    epe = eval_flow_avg(gt_flows_2015, noc_masks_2015, test_result_flow_rigid, opt, moving_masks=gt_masks)
                    sys.stderr.write("epe 2015 rigid is \n")
                    sys.stderr.write(epe + "\n"); summaries['kitti_2015_flow_rigid'] = _parse_to_dict(epe)

                epe = eval_flow_avg(gt_flows_2015, noc_masks_2015, test_result_flow_optical, opt, moving_masks=gt_masks)
                sys.stderr.write("epe 2015 optical is \n")
                sys.stderr.write(epe + "\n"); summaries['kitti_2015_flow_optical'] = _parse_to_dict(epe)

            # mask evaluation
            if opt.eval_mask and eval_data == "kitti_2015":
                mask_err = eval_mask(test_result_mask, gt_masks, opt)
                sys.stderr.write("mask_err is %s \n" % str(mask_err))
    sys.stderr.flush()
    return summaries

def main(unused_argv):
    opt.trace = '' # this should be empty because we have no output when testing
    opt.batch_size = 1
    opt.mode = 'stereo'
    opt.pretrained_model = '.results_original/model-stereo'
    Model, Model_eval = autoflags()

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        print('Constructing models and inputs.')
        image1 = tf.placeholder(tf.float32, [1, opt.img_height, opt.img_width, 3], name='dummy_input_1')
        image1r = tf.placeholder(tf.float32, [1, opt.img_height, opt.img_width, 3], name='dummy_input_1r')
        image2 = tf.placeholder(tf.float32, [1, opt.img_height, opt.img_width, 3], name='dummy_input_2')
        image2r = tf.placeholder(tf.float32, [1, opt.img_height, opt.img_width, 3], name='dummy_input_2r')
        cam2pix = tf.placeholder(tf.float32, [1, 4, 3, 3], name='dummy_cam2pix')
        pix2cam = tf.placeholder(tf.float32, [1, 4, 3, 3], name='dummy_pix2cam')

        with tf.variable_scope(tf.get_variable_scope()) as vs:
            with tf.name_scope("eval_model"):
                _ = Model(image1, image2, image1r, image2r, cam2pix, pix2cam, reuse_scope=False, scope=vs)
                eval_model = Model_eval(scope=vs)

        # Create a saver.
        saver = tf.train.Saver(max_to_keep=10)

        # Make training session.
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.log_device_placement = False
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, opt.pretrained_model)

        # evaluate KITTI gt 2012/2015
        if opt.eval_flow:
            gt_flows_2012, noc_masks_2012 = load_gt_flow_kitti("kitti_2012")
            gt_flows_2015, noc_masks_2015 = load_gt_flow_kitti("kitti")
            gt_masks = load_gt_mask()
        else:
            gt_flows_2012, noc_masks_2012, gt_flows_2015, noc_masks_2015, gt_masks = \
              None, None, None, None, None
        evaluate_kitti(sess, eval_model, 0, gt_flows_2012, noc_masks_2012,
                gt_flows_2015, noc_masks_2015, gt_masks)

if __name__ == '__main__':
    app.run()
