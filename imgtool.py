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

def read_pfm(file):
    if isinstance(file, bytes):
        file = file.decode()
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip().decode('utf-8')
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip().decode('utf-8'))
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data.astype(np.float32), scale

def write_pfm(file, image, scale=1):
    if isinstance(file, bytes):
        file = file.decode()
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')
    
    image = np.flipud(image)  

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write(('%d %d\n' % (image.shape[1], image.shape[0])).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))
    image.tofile(file)

if __name__ == "__main__":
    from pathlib import Path
    root_dir = Path("M:\\Users\\sehee\\camera_taker\\undist_fisheye\\dispL")
    for disp_file in root_dir.glob('*.pfm'):
        disp, _ = read_pfm(str(disp_file))
        imshow(disp)
    cv2.destroyAllWindows()    
