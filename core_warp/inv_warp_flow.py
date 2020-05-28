import tensorflow as tf
from core_warp import grid_sample

def inv_warp_flow(image, flow):
    _, H, W, _ = tf.unstack(tf.shape(image))
    Hf, Wf = tf.cast(H, tf.float32), tf.cast(W, tf.float32)
    # Turn the flow into a query points on the grid space
    gx, gy = tf.meshgrid(tf.linspace(0.0, Wf-1, W), tf.linspace(0.0, Hf-1, H))
    gy = 2 * (gy + flow[:,:,:,0]) / (Hf - 1) - 1
    gx = 2 * (gx + flow[:,:,:,1]) / (Wf - 1) - 1
    grid = tf.stack([gy, gx], axis=-1)
    return grid_sample(image, grid)

'''
from core_warp import inv_warp_flow
import numpy as np

image = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)
image = np.reshape(image, [1, 3, 3, 1])
print("*** Image:", np.squeeze(image))

flo = np.zeros((1, 3, 3, 2), dtype=np.float32)
flo[0, 1, 1, 1] = 1.0

# output should be [[1,2,3],[4,6,6],[7,8,9]]
image2 = inv_warp_flow(image, flo)
print("*** Image:", np.squeeze(image2))

# import tensorflow_addons as tfa
# image3 = tfa.image.dense_image_warp(image, -flo)
# print("*** Image from tfa:", tf.squeeze(image3))
'''