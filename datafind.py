import os

def kitti_data_find(opt):
    dirs = ['/media/data/datasets/kitti_data', '/media/vicnas/datasets/kitti_data', 
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

if __name__ == '__main__':
    from recordclass import recordclass
    opt = dict(data_dir='', gt_2012_dir='', gt_2015_dir='')
    Option = recordclass('Option', opt.keys())
    opt = Option(**opt)
    kitti_data_find(opt)