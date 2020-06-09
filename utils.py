import numpy as np
import cv2

def imread(name):
    # using opencv api, but returned format is RGB not BGR
    im = cv2.imread(name, cv2.IMREAD_UNCHANGED)
    if len(im.shape) == 2:
        im = im[:,:,None]
    if im.shape[2] == 4: # assume this is BGRA format (like png with alpha channel)
        im = im[:,:,:3]
    if im.shape[2] == 3: # convert to RGB
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def imshow(img, name='imshow', wait=True, norm='color'):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    img = np.atleast_3d(img)
    if img.shape[2] == 3: # rgb 2 bgr
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif norm and img.shape[2] == 1 and img.dtype in [np.float, np.float32, np.float64]: # single channel
        n = img / max(np.percentile(img, 95), 1e-6)
        if norm == 'color':
            # [small value of img, large value of img] -> [blue, red]
            h = 120 * np.clip(1 - n, 0, 1) 
            hsv = np.full(n.shape[:2] + (3,), 255, np.uint8)
            hsv[:,:,0:1] = h.astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        elif norm == 'gray':
            img = cv2.convertScaleAbs(n, alpha=255)
        else:
            raise ValueError('Unsupported normalize type')
    cv2.imshow(name, img)
    if wait:
        return cv2.waitKey(0)
    return None


def query_nK(cat):
    if cat == 'victech':
        f = 467.83661057
        cx, cy = 284.1095847, 256.36649503
        baseline = 0.120601 # abs(P2[1,4]) / Q[3,4]
    elif cat == 'dexter':
        f = 320
        cx, cy = 0.5*640-0.5, 0.5*480-0.5
        baseline = 0.12
    else:
        raise RuntimeError('Not supported dataset category')

    # Note we are using normalized intrinsic
    W, H = 640, 480
    nfx, nfy = f/W, f/H
    ncx, ncy = (cx+0.5)/W, (cy+0.5)/H
    nK = np.array([[nfx, 0, ncx], [0, nfy, ncy], [0, 0, 1]], dtype=np.float32)

    return nK, np.float32(baseline)


def unnormalize_K(nK, size):
    w, h = size
    scale = np.array([[w, 0, -0.5], [0, h, -0.5], [0, 0, 1]], dtype=np.float32)
    return scale @ nK


def resize_image_pairs(imgL, imgR, new_size, cvt_type=None):
    w0, w1 = imgL.shape[1], new_size[0]

    interp = cv2.INTER_AREA if w0 > w1 else cv2.INTER_LINEAR
    imgL = cv2.resize(imgL, new_size, interpolation=interp)
    imgR = cv2.resize(imgR, new_size, interpolation=interp)

    if cvt_type is not None:
        if cvt_type in [np.float, np.float32, np.float64] and imgL.dtype in [np.uint8]:
            imgL = (imgL / 255).astype(cvt_type)
            imgR = (imgR / 255).astype(cvt_type)
        elif imgL.dtype in [np.float, np.float32, np.float64] and cvt_type in [np.uint8]:
            imgL = np.clip(imgL * 255, 0, 255).astype(cvt_type)
            imgR = np.clip(imgR * 255, 0, 255).astype(cvt_type)
        else:
            raise ValueError('Invalid cvt_type')
        
    return imgL, imgR
