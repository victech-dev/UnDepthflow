import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import cv2

# Some were adopted from 
# https://github.com/tensorflow/models/tree/master/research/video_prediction
def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
      List of pairs of (gradient, variable) where the gradient has been averaged
      across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def mean_squared_error(true, pred):
    """L2 distance between tensors true and pred.
    Args:
      true: the ground truth image.
      pred: the predicted image.
    Returns:
      mean squared error between ground truth and predicted image.
    """
    return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))


def weighted_mean_squared_error(true, pred, weight):
    """L2 distance between tensors true and pred.
    Args:
      true: the ground truth image.
      pred: the predicted image.
    Returns:
      mean squared error between ground truth and predicted image.
    """

    tmp = tf.reduce_sum(
        weight * tf.square(true - pred), axis=[1, 2],
        keep_dims=True) / tf.reduce_sum(
            weight, axis=[1, 2], keep_dims=True)
    return tf.reduce_mean(tmp)


def mean_L1_error(true, pred):
    """L2 distance between tensors true and pred.

    Args:
      true: the ground truth image.
      pred: the predicted image.
    Returns:
      mean squared error between ground truth and predicted image.
    """
    return tf.reduce_sum(tf.abs(true - pred)) / tf.to_float(tf.size(pred))


def weighted_mean_L1_error(true, pred, weight):
    """L2 distance between tensors true and pred.

    Args:
      true: the ground truth image.
      pred: the predicted image.
    Returns:
      mean squared error between ground truth and predicted image.
    """
    return tf.reduce_sum(tf.abs(true - pred) *
                         weight) / tf.to_float(tf.size(pred))

def calc_grad2(img, level=0):
    ''' Calculate Iyy, Ixx by applying 1D Laplacian filter for each axis '''
    # x3 scale compared to vanila gradient of gradient caused by 3x3 Laplacian filter
    img_shape = tf.shape(img)
    fxx = np.array([[1,-2,1]]*3, dtype=np.float32)
    kernel = tf.convert_to_tensor(np.stack([fxx.T, fxx], axis=-1))
    kernel = tf.tile(kernel[:,:,None,:], [1, 1, img_shape[-1], 1]) # [3,3,C,2]
    padded = tf.pad(img, [[0,0],[1,1],[1,1],[0,0]], 'SYMMETRIC')
    grad2 = tf.nn.depthwise_conv2d(padded, kernel, [1, 1, 1, 1], padding='VALID') # [B,H,W,Cx2]
    shape = tf.concat([img_shape, [2]], 0)
    grad2 = tf.reshape(grad2, shape=shape)
    level_factor = 0.25 ** level
    return level_factor * grad2 # [B, H, W, C, (iyy,ixx)]


def edge_aware_weight(img, level=0):
    ''' Calculate edge aware weight from Iy, Ix by Sobel filter '''
    # x10 scale compared to vanila gradient caused by sobel filter and RGB factor
    rgb_factor = tf.constant([0.897, 1.761, 0.342], dtype=tf.float32)
    sobel = tf.image.sobel_edges(img) # [batch, height, width, 3, 2]
    sobel_weighted = sobel * rgb_factor[None,None,None,:,None]
    sobel_abs = tf.abs(sobel_weighted)
    g = tf.reduce_max(sobel_abs, axis=3, keepdims=True) 
    level_factor = np.sqrt(0.5) ** level
    return tf.exp(-level_factor * g) # [B, H, W, 1, (iy,ix)]


def disp_smoothness(disp, pyramid):
    grad2 = [calc_grad2(tf.math.log(d), level=s) for s, d in enumerate(disp)]
    weight = [edge_aware_weight(img, level=s) for s, img in enumerate(pyramid)]
    output = [g * w for g, w in zip(grad2, weight)]
    return output # array of [B, H, W, 1, 2]


def flow_smoothness(flow, image, level=0):
    g = calc_grad2(flow, level=level)
    w = edge_aware_weight(image, level=level)
    return tf.reduce_sum(tf.abs(g * w), axis=-1) # [B, H, W, 2]


def SSIM(x, y):
    # note using huge filter sigma to mimic averaging
    ssim = tf.image.ssim(x, y, 1.0, filter_size=3, filter_sigma=256)
    return tf.clip_by_value(0.5 * (1 - ssim), 0, 1)


def deprocess_image(image):
    # Assuming input image is float32
    return tf.image.convert_image_dtype(image, dtype=tf.uint8)


def preprocess_image(image):
    # Assuming input image is uint8
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image


def charbonnier_loss(x,
                     mask=None,
                     truncate=None,
                     alpha=0.45,
                     beta=1.0,
                     epsilon=0.001):
    """Compute the generalized charbonnier loss of the difference tensor x.
    All positions where mask == 0 are not taken into account.
    Args:
        x: a tensor of shape [num_batch, height, width, channels].
        mask: a mask of shape [num_batch, height, width, mask_channels],
            where mask channels must be either 1 or the same number as
            the number of channels of x. Entries should be 0 or 1.
    Returns:
        loss as tf.float32
    """
    batch, height, width, channels = tf.unstack(tf.shape(x))
    normalization = tf.cast(batch * height * width * channels, tf.float32)

    error = tf.pow(tf.square(x * beta) + tf.square(epsilon), alpha)

    if mask is not None:
        error = tf.multiply(mask, error)

    if truncate is not None:
        error = tf.minimum(error, truncate)

    return tf.reduce_sum(error) / normalization
