from tensorflow.python.platform import flags
import os
FLAGS = flags.FLAGS

def kitti_data_find():
    found = None
    if os.path.isdir('/media/data/datasets/kitti_data'):
        found = '/media/data/datasets/kitti_data'
    elif os.path.isdir('/media/vicnas/datasets/kitti_data'):
        found = '/media/vicnas/datasets/kitti_data'
    elif os.path.isdir('M:\\datasets\\kitti_data'):
        found = 'M:\\datasets\\kitti_data'
    else:
        raise RuntimeError('KITTI data not found!!')

    print('*** KITTI data path:', found)
    FLAGS.data_dir = os.path.join(found, 'kitti_raw_data') + os.path.sep
    FLAGS.gt_2012_dir = os.path.join(found, 'kitti_stereo_2012', 'training') + os.path.sep
    FLAGS.gt_2015_dir = os.path.join(found, 'kitti_stereo_2015', 'training') + os.path.sep
