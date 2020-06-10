import tensorflow as tf

def dense_image_uwarp(image, uflow, name=None):
    with tf.name_scope(name or 'dense_image_uwarp'):
        B, H, W, C = tf.unstack(tf.shape(image))
        # flow -> target grid (Note we subtract flow from meshgrid for compatibility with tfa.image.dense_image_warp)
        gx = tf.cast(tf.range(W), tf.float32) 
        gx = gx[None,None,:,None] - uflow
        gx = tf.squeeze(gx, axis=-1)

        with tf.name_scope('interpolate_linear'):
            x0f = tf.floor(gx)
            xa = gx - x0f
            x0 = tf.cast(x0f, tf.int32)
            x1 = x0 + 1
            x0c, x1c = tf.clip_by_value(x0, 0, W-1), tf.clip_by_value(x1, 0, W-1)
            
            batch_offsets = tf.range(B) * H * W
            batch_offsets = batch_offsets[:,None,None] # would broadcast to [B,H,W]
            row_offsets = tf.range(H) * W
            row_offsets = row_offsets[None,:,None] # would broadcast to [B,H,W]
            flattened_img = tf.reshape(image, (-1, C))

            def gather_with_zero_padding(col, name):
                with tf.name_scope('gather_' + name):
                    linear_coordinates = batch_offsets + row_offsets + col
                    gathered_values = tf.gather(flattened_img, linear_coordinates)
                    return gathered_values

            left = gather_with_zero_padding(x0c, 'left')
            right = gather_with_zero_padding(x1c, 'right')

            # now, do the actual interpolation
            # this is not bilinear, just linear interpolation on single scan line
            interp = tf.expand_dims(xa, -1) * (right - left) + left

    return interp

# inverse warp by u-coord flow 
def inv_warp_uflow(image, uflow, padding='zeros'):
    if padding == 'border':
        return dense_image_uwarp(image, -uflow)
    elif padding == 'zeros':
        image_pad = tf.pad(image, [[0,0], [0,0], [1,1], [0,0]])
        uflow_pad = tf.pad(uflow, [[0,0], [0,0], [1,1], [0,0]])
        w = dense_image_uwarp(image_pad, -uflow_pad)
        return w[:,:,1:-1]
    else:
        raise ValueError('Not supported padding type')


if __name__ == '__main__':
    import numpy as np

    image = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)
    image = np.reshape(image, [1, 3, 3, 1])
    print("*** Image:", np.squeeze(image))

    flo = np.zeros((1, 3, 3, 1), dtype=np.float32)
    flo[0, 1, 1, 0] = 1.0

    # output should be [[1,2,3],[4,6,6],[7,8,9]]
    image2 = inv_warp_uflow(image, flo)
    print("*** Image:", np.squeeze(image2))
