import tensorflow as tf

# tensorflow version of torch.nn.functional.grid_sample (padding_mode='zeros')
# grid_x shape = [B, H, W, 1] of domain [-1, 1]
# image size and flowx size should be equal
def inv_warp_flow_x(image, flowx, name='inv_warp_flowx'):
    with tf.name_scope(name):
        B, H, W, C = tf.unstack(tf.shape(image))
        # flow -> target grid
        gx = tf.cast(tf.range(W), tf.float32) 
        gx = gx[None,None,:,None] + flowx
        gx = tf.squeeze(gx, axis=-1)

        with tf.name_scope('interpolate_linear'):
            x0f = tf.floor(gx)
            xa = gx - x0f
            x0 = tf.cast(x0f, tf.int32)
            x1 = x0 + 1
            x0c, x1c = tf.clip_by_value(x0, 0, W-1), tf.clip_by_value(x1, 0, W-1)
            x0m, x1m = tf.cast(x0c==x0, tf.float32), tf.cast(x1c==x1, tf.float32)
            
            batch_offsets = tf.range(B) * H * W
            batch_offsets = batch_offsets[:,None,None] # would broadcast to [B,H,W]
            row_offsets = tf.range(H) * W
            row_offsets = row_offsets[None,:,None] # would broadcast to [B,H,W]
            flattened_img = tf.reshape(image, (-1, C))

            def gather_with_zero_padding(col, mask, name):
                with tf.name_scope('gather_' + name):
                    linear_coordinates = batch_offsets + row_offsets + col
                    gathered_values = tf.gather(flattened_img, linear_coordinates)
                    gathered_values = gathered_values * tf.expand_dims(mask, -1)
                    return gathered_values

            left = gather_with_zero_padding(x0c, x0m, 'left')
            right = gather_with_zero_padding(x1c, x1m, 'right')

            # now, do the actual interpolation
            # this is not bilinear, just linear interpolation on single scan line
            interp = tf.expand_dims(xa, -1) * (right - left) + left

    return interp


if __name__ == '__main__':
    import numpy as np

    image = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)
    image = np.reshape(image, [1, 3, 3, 1])
    print("*** Image:", np.squeeze(image))

    flo = np.zeros((1, 3, 3, 1), dtype=np.float32)
    flo[0, 1, 1, 0] = 1.0

    # output should be [[1,2,3],[4,6,6],[7,8,9]]
    image2 = inv_warp_flowx(image, flo)
    print("*** Image:", np.squeeze(image2))
