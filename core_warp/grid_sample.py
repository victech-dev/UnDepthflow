import tensorflow as tf

# tensorflow version of torch.nn.functional.grid_sample (padding_mode='zeros')
def grid_sample(image, grid, name='grid_sample'):
    with tf.name_scope(name):
        B, H, W, C = tf.unstack(tf.shape(image))
        _, GH, GW, _ = tf.unstack(tf.shape(grid))
        Hf, Wf = tf.cast(H, tf.float32), tf.cast(W, tf.float32)

        # Turn the grid into a list of query points
        gy, gx = tf.unstack(grid, axis=-1) # domain [-1, 1] for each
        gy = (gy + 1) * 0.5 * (Hf - 1)
        gx = (gx + 1) * 0.5 * (Wf - 1)

        with tf.name_scope('interpolate_bilinear'):
            x, y = tf.reshape(gx, [B, -1]), tf.reshape(gy, [B, -1])
            x0f, y0f = tf.floor(x), tf.floor(y)
            x0, y0 = tf.cast(x0f, tf.int32), tf.cast(y0f, tf.int32)
            x1, y1 = x0 + 1, y0 + 1
            xa, ya = tf.expand_dims(x - x0f, -1), tf.expand_dims(y - y0f, -1)
            x0c, y0c = tf.clip_by_value(x0, 0, W-1), tf.clip_by_value(y0, 0, H-1)
            x1c, y1c = tf.clip_by_value(x1, 0, W-1), tf.clip_by_value(y1, 0, H-1)
            x0m, y0m = tf.cast(x0c==x0, tf.float32), tf.cast(y0c==y0, tf.float32)
            x1m, y1m = tf.cast(x1c==x1, tf.float32), tf.cast(y1c==y1, tf.float32)

            batch_offsets = tf.range(B) * H * W
            batch_offsets = batch_offsets[:,None]
            flattened_img = tf.reshape(image, (-1, C))

            def gather_with_zero_padding(row, col, mask, name):
                with tf.name_scope('gather_' + name):
                    linear_coordinates = batch_offsets + row * W + col
                    gathered_values = tf.gather(flattened_img, linear_coordinates)
                    gathered_values = gathered_values * tf.expand_dims(mask, -1)
                    return gathered_values

            top_left = gather_with_zero_padding(y0c, x0c, y0m*x0m, 'top_left')
            top_right = gather_with_zero_padding(y0c, x1c, y0m*x1m, 'top_right')
            bottom_left = gather_with_zero_padding(y1c, x0c, y1m*x0m, 'bottom_left')
            bottom_right = gather_with_zero_padding(y1c, x1c, y1m*x1m, 'bottom_right')    

            # now, do the actual interpolation
            with tf.name_scope('interpolate'):
                interp_top = xa * (top_right - top_left) + top_left
                interp_bottom = xa * (bottom_right - bottom_left) + bottom_left
                interp = ya * (interp_bottom - interp_top) + interp_top                            

    return tf.reshape(interp, [B, GH, GW, C])


if __name__ == '__main__':
    import numpy as np

    image = tf.constant(
        [1, 2, 3, 4, 5, 6, 7, 8, 9], shape=[1, 3, 3, 1], dtype="float32")
    print("*** Image:", tf.squeeze(image))

    gx, gy = np.meshgrid(np.linspace(-2,1,7), np.linspace(-1,1,7))
    grid = np.stack([gy, gx], axis=-1)
    grid = grid[None].astype(np.float32)

    image2 = grid_sample(image, grid)
    print("*** Image:", np.squeeze(image2.numpy()))

