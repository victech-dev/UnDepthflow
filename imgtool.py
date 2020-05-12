import numpy as np
import cv2
import sys
import re

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

def imshow(img, name='imshow', wait=True, norm=True):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    if len(img.shape)==3 and img.shape[2]==3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if norm and (img.dtype==np.float32 or img.dtype==np.float):
        img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
        img = cv2.convertScaleAbs(img, alpha=255)
    cv2.imshow(name, img)
    if wait:
        cv2.waitKey(0)

if __name__ == "__main__":
    from pathlib import Path
    root_dir = Path("M:\\Users\\sehee\\camera_taker\\undist_fisheye\\dispL")
    for disp_file in root_dir.glob('*.pfm'):
        disp = cv2.imread(str(disp_file), -1)
        imshow(disp)
    cv2.destroyAllWindows()    
