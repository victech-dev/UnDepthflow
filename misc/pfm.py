import sys
import numpy as np
import re

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
    import cv2
    root_dir = Path("M:\\Users\\sehee\\camera_taker\\undist_fisheye\\dispL")
    cv2.namedWindow('pfmshow', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('pfmshow', (1280, 720))
    for disp_file in root_dir.glob('*.pfm'):
        disp, _ = read_pfm(str(disp_file))
        disp = cv2.normalize(disp, None, 0, 1, cv2.NORM_MINMAX)
        disp = cv2.convertScaleAbs(disp, alpha=255) 
        cv2.imshow('pfmshow', disp)
        cv2.waitKey(0)
    cv2.destroyAllWindows()    