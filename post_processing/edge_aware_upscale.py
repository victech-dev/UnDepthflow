import numpy as np
import cv2


def upsample_with_mask(src, mask, H, W):
    ''' Bilinear upsampling with mask
        src: 2D np.ndarray
        mask: 2D np.ndarray (can have different shape from src)
        H, W: target size
    '''
    Hs, Ws = src.shape[:2]
    # resize mask to src size (nearest)
    mask = cv2.resize(mask, (Ws, Hs), interpolation=cv2.INTER_NEAREST)
    
    def _take(x, y, inp):
        indices = x + y * Ws
        output = np.take(inp.reshape(-1), indices.reshape(-1))
        return output.reshape(H, W)
        
    # u(xt,yt) = xs, u(xt,yt) = ys
    xs, ys = np.meshgrid(np.linspace(0,Ws-1,W), np.linspace(0,Hs-1,H))
    # left-top corner (x0, y0), right-bottom corner (x1, y1)
    x0, y0 = np.floor(xs).astype(np.int32), np.floor(ys).astype(np.int32)
    x1, y1 = np.minimum(x0+1, Ws-1), np.minimum(y0+1, Hs-1)

    # calculate weights of 4 corners
    wa = (x1-xs) * (y1-ys) * _take(x0, y0, mask) # top-left
    wb = (x1-xs) * (ys-y0) * _take(x0, y1, mask) # bottom-left
    wc = (xs-x0) * (y1-ys) * _take(x1, y0, mask) # top-right\
    wd = (xs-x0) * (ys-y0) * _take(x1, y1, mask) # bottom-right
    w = np.stack([wa, wb, wc, wd], axis=-1) # [H,W,4]
    w /= np.maximum(np.sum(w, axis=-1, keepdims=True), 1e-7)
    
    # calculate src values of 4 corners
    sa = _take(x0, y0, src)
    sb = _take(x0, y1, src)
    sc = _take(x1, y0, src)
    sd = _take(x1, y1, src)
    s = np.stack([sa, sb, sc, sd], axis=-1) # [H,W,4]

    # output shape is [H, W]
    return np.sum(w*s, axis=-1)


def edge_aware_upscale(src, H, W, thresh=None):
    ''' Edge aware upscale for depth or disparity map (1/4 ~ 1/6 of original image)
        src: disparity or depth, 2D np.ndarray 
        H, W: original image size
    '''
    # do log scale transform of src
    # this makes LoG operator scale-invariant, 
    # and src, 1/src generates same edge mask (i.e, src can be either disparity or depth)
    src_transformed = np.log(src)

    # Laplacian of Gaussian Operation
    LoG = cv2.Laplacian(cv2.GaussianBlur(src_transformed, (7,7), 0), cv2.CV_32F)

    # find zero-crossing of LoG
    if thresh is None:
        thresh = (np.var(LoG)**0.5) * 0.75
    minLoG = cv2.morphologyEx(LoG, cv2.MORPH_ERODE, np.ones((3,3))) # min_pooling 2d
    maxLoG = cv2.morphologyEx(LoG, cv2.MORPH_DILATE, np.ones((3,3))) # max_pooling 2d
    mask = np.logical_and(minLoG * maxLoG < 0, maxLoG - minLoG > thresh)

    # invert mask, so that zero means edge (discontinuity)
    mask = 1 - mask.astype(np.float32)

    # upsample src with mask
    return upsample_with_mask(src, mask, H, W)
