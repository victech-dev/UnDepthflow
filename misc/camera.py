import numpy as np
import cv2


def query_K(cat):
    if cat == 'victech':
        f = 467.83661057
        cx, cy = 284.1095847, 256.36649503
        baseline = 0.120601 # abs(P2[1,4]) / Q[3,4]
    elif cat == 'dexter':
        f = 320
        cx, cy = (640-1)/2, (480-1)/2
        baseline = 0.12
    else:
        raise RuntimeError('Not supported dataset category')
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    return K, baseline


def rescale_K(K, size0, size1):
    w0, h0 = size0
    w1, h1 = size1
    rx, ry = w1 / w0, h1 / h0
    K_scale = np.array([[rx, 0, 0.5*(rx-1)], [0, ry, 0.5*(ry-1)], [0, 0, 1]], dtype=np.float32)
    return K_scale @ K


def resize_image_pairs(imgL, imgR, new_size, K=None):
    h0, w0 = imgL.shape[:2]
    w1, _ = new_size
    interp = cv2.INTER_AREA if w0 > w1 else cv2.INTER_LINEAR
    imgL = cv2.resize(imgL, new_size, interpolation=interp)
    imgR = cv2.resize(imgR, new_size, interpolation=interp)
    if K is None:
        return imgL, imgR
    else:
        return imgL, imgR, rescale_K(K, (w0, h0), new_size)



