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

def fwd_warp_flow(image, flo, name='fwd_warp_flow', backprop=False):
    """Forward warping layer

    Implements a backward warping layer described in 
    "Unsupervised Deep Learning for Optical Flow Estimation, Zhe Ren et al"

    Parameters
    ----------
    image : float [num_batch, height, width, num_channels].
    flo: float [num_batch, height, width, 2]
    """

    def _scatter_2x2(im, xy, name='scatter_2x2'):
        with tf.variable_scope('name'):
            B, H, W, C = tf.unstack(tf.shape(im))

            # top-left: (x0,y0) as int, (x0f,y0f) as float, (x0c, y0c) as clipped, (x0m, y0m) as in-rect mask
            # bottom-right: (x1,y1) as int, (x1f,y1f) as float, (x1c, y1c) as clipped, (x1m, y1m) as in-rect mask
            x, y = tf.unstack(xy, axis=-1) # shape of each = [B, H, W] 
            x0f, y0f = tf.floor(x), tf.floor(y)
            x0, y0 = tf.cast(x0f, tf.int32), tf.cast(y0f, tf.int32)
            x1f, y1f = x0f + 1, y0f + 1
            x1, y1 = x0 + 1, y0 + 1
            x0c, y0c = tf.clip_by_value(x0, 0, W-1), tf.clip_by_value(y0, 0, H-1)
            x1c, y1c = tf.clip_by_value(x1, 0, W-1), tf.clip_by_value(y1, 0, H-1)
            x0m, y0m = tf.cast(tf.equal(x0, x0c), tf.float32), tf.cast(tf.equal(y0, y0c), tf.float32)
            x1m, y1m = tf.cast(tf.equal(x1, x1c), tf.float32), tf.cast(tf.equal(y1, y1c), tf.float32)

            batch_offsets = tf.range(B) * (W * H)
            batch_offsets = batch_offsets[:,None,None] # [B, 1, 1]

            # shape of idx, weight = [B, H, W]
            idx_a = batch_offsets + y0c * W + x0c # top-left
            wa = (x1f - x) * (y1f - y) * (x0m * y0m)
            idx_b = batch_offsets + y1c * W + x0c # bottom-left
            wb = (x1f - x) * (y - y0f) * (x0m * y1m)
            idx_c = batch_offsets + y0c * W + x1c # top-right
            wc = (x - x0f) * (y1f - y) * (x1m * y0m)
            idx_d = batch_offsets + y1c * W + x1c # bottom-right
            wd = (x - x0f) * (y - y0f) * (x1m * y1m)

            #im_flat = tf.reshape(im, (-1, channel))
            updates = tf.stack([wa, wb, wc, wd], axis=0) # [4, B, H, W]
            updates = tf.expand_dims(updates, axis=-1) * im[None] # [4, B, H, W, C]
            updates = tf.reshape(updates, [-1, C]) # [4*B*H*W, C]
            indices = tf.stack([idx_a, idx_b, idx_c, idx_d], axis=0) # [4, B, H, W]
            indices = tf.reshape(indices, [-1, 1]) # [4*B*H*W, 1]

            shape = [B * H * W, C]
            output = tf.scatter_nd(indices, updates, shape)
            output = tf.reshape(output, [B, H, W, C])
            return output if backprop else tf.stop_gradient(output)

    with tf.name_scope(name):
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        if tf.rank(image) == 0:
            image = tf.ones_like(flo[:,:,:,0:1]) * image
        B, H, W, _ = tf.unstack(tf.shape(image))
        Hf, Wf = tf.cast(H, tf.float32), tf.cast(W, tf.float32)

        # Turn the flow into a list of query points on the grid space
        gx, gy = tf.meshgrid(tf.linspace(0.0, Wf-1, W), tf.linspace(0.0, Hf-1, H))
        gxy = tf.stack([gx[None], gy[None]], axis=-1) + flo

        input_transformed = _scatter_2x2(image, gxy)
        output = tf.reshape(input_transformed, [B, H, W, -1])
        return output

if __name__ == '__main__':
    import numpy as np
    tf.enable_eager_execution()

    image = tf.constant(
        [[0, 1, 2, 3, 4]]*2, shape=[1, 2, 5, 1], dtype="float32")
    print("*** Image:", tf.squeeze(image))

    flo_x = np.array([[2,3,1,-2,-3]]*2, dtype=np.float32).reshape(1,2,5,1)
    flo_y = np.zeros_like(flo_x)
    flo = np.concatenate([flo_x, flo_y], axis=-1)
    flo = tf.convert_to_tensor(flo)

    # output should be [[1,2,3],[4,6,6],[7,8,9]]
    image2 = fwd_warp_flow(image, flo)
    print("*** Image:", tf.squeeze(image2))

