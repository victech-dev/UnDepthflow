# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf

def inv_warp_flow(image, flo, out_size, name='inv_warp_flow'):
    """Backward warping layer

    Implements a backward warping layer described in 
    "Unsupervised Deep Learning for Optical Flow Estimation, Zhe Ren et al"

    Parameters
    ----------
    image : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    flo: float
         The optical flow used to do the backward warping.
         shape is [num_batch, height, width, 2]
    out_size: tuple of two ints
        The size of the output of the network (height, width)
    """

    def _interpolate_bilinear(im, xy, out_size, name='interpolate_bilinear'):
        with tf.name_scope(name):
            num_batch, height, width, channel = tf.unstack(tf.shape(im))
            height_f, width_f = tf.cast(height, tf.float32), tf.cast(width, tf.float32)
            out_height, out_width = out_size

            x, y = tf.unstack(xy, axis=-1)
            x0f = tf.clip_by_value(tf.floor(x), 0.0, width_f-2.0)
            y0f = tf.clip_by_value(tf.floor(y), 0.0, height_f-2.0)
            x0, y0 = tf.cast(x0f, tf.int32), tf.cast(y0f, tf.int32)
            x1, y1 = x0 + 1, y0 + 1
            xa, ya = tf.clip_by_value(x - x0f, 0, 1), tf.clip_by_value(y - y0f, 0, 1)
            xa, ya = tf.expand_dims(xa, -1), tf.expand_dims(ya, -1)

            batch_offsets = tf.range(num_batch) * width * height
            batch_offsets = batch_offsets[:,None]
            flattened_grid = tf.reshape(im, (num_batch * width * height, channel))

            def gather(y_coords, x_coords, name):
                with tf.name_scope('gather-' + name):
                    linear_coordinates = batch_offsets + y_coords * width + x_coords
                    gathered_values = tf.gather(flattened_grid, linear_coordinates)
                    return tf.reshape(gathered_values, [num_batch, out_height * out_width, channel])

            top_left = gather(y0, x0, 'top_left')
            top_right = gather(y0, x1, 'top_right')
            bottom_left = gather(y1, x0, 'bottom_left')
            bottom_right = gather(y1, x1, 'bottom_right')    

            # now, do the actual interpolation
            with tf.name_scope('interpolate'):
                interp_top = xa * (top_right - top_left) + top_left
                interp_bottom = xa * (bottom_right - bottom_left) + bottom_left
                interp = ya * (interp_bottom - interp_top) + interp_top                            

            return interp

    with tf.name_scope(name):
        num_batch, height, width, num_channels = tf.unstack(tf.shape(image))
        height_f, width_f = tf.cast(height, tf.float32), tf.cast(width, tf.float32)
        out_height, out_width = out_size

        # Turn the flow into a list of generate query points in the grid space
        gx, gy = tf.meshgrid(tf.linspace(0.0, width_f-1, out_width), tf.linspace(0.0, height_f-1, out_height))
        gxy = tf.stack([gx[None], gy[None]], axis=-1)
        scale = tf.convert_to_tensor([(width_f-1)/(out_width-1), (height_f-1)/(out_height-1)])
        gxy += flo * scale[None,None,None,:]
        gxy_flatten = tf.reshape(gxy, [num_batch, out_height*out_width, 2])

        input_transformed = _interpolate_bilinear(image, gxy_flatten, out_size)
        output = tf.reshape(
            input_transformed,
            tf.stack([num_batch, out_height, out_width, num_channels]))
        return output

if __name__ == '__main__':
    import numpy as np
    tf.enable_eager_execution()

    image = tf.constant(
        [1, 2, 3, 4, 5, 6, 7, 8, 9], shape=[1, 3, 3, 1], dtype="float32")
    print("*** Image:", tf.squeeze(image))

    flo = np.zeros((1, 3, 3, 2))
    flo[0, 1, 1, 0] = 1.0
    #flo[0, 1, 1, 1] = 1.0
    flo = tf.constant(flo, dtype="float32")

    # output should be [[1,2,3],[4,6,6],[7,8,9]]
    image2 = inv_warp_flow(image, flo, [3, 3])
    print("*** Image:", tf.squeeze(image2))
