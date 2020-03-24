import os
from models import *

def autoflags(opt, mode, do_eval):
    opt.mode = mode
    opt.eval_flow, opt.eval_depth, opt.eval_mask = False, False, False

    # data path
    dirs = ['/media/data/datasets/kitti_data', '~/datasets/kitti_data', '/media/vicnas/datasets/kitti_data', 
        'C:\\datasets\\kitti_data', 'D:\\datasets\\kitti_data', 'M:\\datasets\\kitti_data']
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
    if mode == "depthflow":  # stage 3: train depth and flow together
        Model = Model_depthflow
        Model_eval = Model_eval_depthflow
        if do_eval:
            opt.eval_flow, opt.eval_depth, opt.eval_mask = True, True, True
    elif mode == "depth":  # stage 2: train depth
        Model = Model_depth
        Model_eval = Model_eval_depth
        if do_eval:
            opt.eval_flow, opt.eval_depth, opt.eval_mask = True, True, False
    elif mode == "flow":  # stage 1: train flow
        Model = Model_flow
        Model_eval = Model_eval_flow
        if do_eval:
            opt.eval_flow, opt.eval_depth, opt.eval_mask = True, False, False
    elif mode == "stereo":
        Model = Model_stereo
        Model_eval = Model_eval_stereo
        if do_eval:
            opt.eval_flow, opt.eval_depth, opt.eval_mask = False, True, False
    else:
        raise "mode must be one of flow, depth, depthflow or stereo"
    print('*** Model mode:', mode)
    return Model, Model_eval


if __name__ == '__main__':
    from recordclass import recordclass
    opt = dict(mode='', data_dir='', gt_2012_dir='', gt_2015_dir='', eval_flow=False, eval_depth=False, eval_mask=False)
    Option = recordclass('Option', opt.keys())
    opt = Option(**opt)
    autoflags(opt, 'stereo', False)