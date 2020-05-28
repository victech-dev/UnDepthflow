# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Image warping using per-pixel flow vectors."""

import numpy as np
import tensorflow as tf

def interpolate_bilinear(grid, query_points, indexing="ij", name=None):
    """Similar to Matlab's interp2 function.

    Finds values for query points on a grid using bilinear interpolation.

    Args:
      grid: a 4-D float `Tensor` of shape `[batch, height, width, channels]`.
      query_points: a 3-D float `Tensor` of N points with shape
        `[batch, N, 2]`.
      indexing: whether the query points are specified as row and column (ij),
        or Cartesian coordinates (xy).
      name: a name for the operation (optional).

    Returns:
      values: a 3-D `Tensor` with shape `[batch, N, channels]`

    Raises:
      ValueError: if the indexing mode is invalid, or if the shape of the
        inputs invalid.
    """
    if indexing != "ij" and indexing != "xy":
        raise ValueError("Indexing mode must be 'ij' or 'xy'")

    with tf.name_scope(name or "interpolate_bilinear"):
        grid = tf.convert_to_tensor(grid)
        query_points = tf.convert_to_tensor(query_points)

        # grid shape checks
        grid_static_shape = grid.shape
        grid_shape = tf.shape(grid)
        if grid_static_shape.dims is not None:
            if len(grid_static_shape) != 4:
                raise ValueError("Grid must be 4D Tensor")
            if grid_static_shape[1] is not None and grid_static_shape[1] < 2:
                raise ValueError("Grid height must be at least 2.")
            if grid_static_shape[2] is not None and grid_static_shape[2] < 2:
                raise ValueError("Grid width must be at least 2.")
        else:
            with tf.control_dependencies(
                [
                    tf.debugging.assert_greater_equal(
                        grid_shape[1], 2, message="Grid height must be at least 2."
                    ),
                    tf.debugging.assert_greater_equal(
                        grid_shape[2], 2, message="Grid width must be at least 2."
                    ),
                    tf.debugging.assert_less_equal(
                        tf.cast(
                            grid_shape[0] * grid_shape[1] * grid_shape[2],
                            dtype=tf.dtypes.float32,
                        ),
                        np.iinfo(np.int32).max / 8.0,
                        message="The image size or batch size is sufficiently "
                        "large that the linearized addresses used by "
                        "tf.gather may exceed the int32 limit.",
                    ),
                ]
            ):
                pass

        # query_points shape checks
        query_static_shape = query_points.shape
        query_shape = tf.shape(query_points)
        if query_static_shape.dims is not None:
            if len(query_static_shape) != 3:
                raise ValueError("Query points must be 3 dimensional.")
            query_hw = query_static_shape[2]
            if query_hw is not None and query_hw != 2:
                raise ValueError("Query points last dimension must be 2.")
        else:
            with tf.control_dependencies(
                [
                    tf.debugging.assert_equal(
                        query_shape[2],
                        2,
                        message="Query points last dimension must be 2.",
                    )
                ]
            ):
                pass

        batch_size, height, width, channels = (
            grid_shape[0],
            grid_shape[1],
            grid_shape[2],
            grid_shape[3],
        )

        num_queries = query_shape[1]

        query_type = query_points.dtype
        grid_type = grid.dtype

        alphas = []
        floors = []
        ceils = []
        index_order = [0, 1] if indexing == "ij" else [1, 0]
        unstacked_query_points = tf.unstack(query_points, axis=2, num=2)

        for i, dim in enumerate(index_order):
            with tf.name_scope("dim-" + str(dim)):
                queries = unstacked_query_points[dim]

                size_in_indexing_dimension = grid_shape[i + 1]

                # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
                # is still a valid index into the grid.
                max_floor = tf.cast(size_in_indexing_dimension - 2, query_type)
                min_floor = tf.constant(0.0, dtype=query_type)
                floor = tf.math.minimum(
                    tf.math.maximum(min_floor, tf.math.floor(queries)), max_floor
                )
                int_floor = tf.cast(floor, tf.dtypes.int32)
                floors.append(int_floor)
                ceil = int_floor + 1
                ceils.append(ceil)

                # alpha has the same type as the grid, as we will directly use alpha
                # when taking linear combinations of pixel values from the image.
                alpha = tf.cast(queries - floor, grid_type)
                min_alpha = tf.constant(0.0, dtype=grid_type)
                max_alpha = tf.constant(1.0, dtype=grid_type)
                alpha = tf.math.minimum(tf.math.maximum(min_alpha, alpha), max_alpha)

                # Expand alpha to [b, n, 1] so we can use broadcasting
                # (since the alpha values don't depend on the channel).
                alpha = tf.expand_dims(alpha, 2)
                alphas.append(alpha)

            flattened_grid = tf.reshape(grid, [batch_size * height * width, channels])
            batch_offsets = tf.reshape(
                tf.range(batch_size) * height * width, [batch_size, 1]
            )

        # This wraps tf.gather. We reshape the image data such that the
        # batch, y, and x coordinates are pulled into the first dimension.
        # Then we gather. Finally, we reshape the output back. It's possible this
        # code would be made simpler by using tf.gather_nd.
        def gather(y_coords, x_coords, name):
            with tf.name_scope("gather-" + name):
                linear_coordinates = batch_offsets + y_coords * width + x_coords
                gathered_values = tf.gather(flattened_grid, linear_coordinates)
                return tf.reshape(gathered_values, [batch_size, num_queries, channels])

        # grab the pixel values in the 4 corners around each query point
        top_left = gather(floors[0], floors[1], "top_left")
        top_right = gather(floors[0], ceils[1], "top_right")
        bottom_left = gather(ceils[0], floors[1], "bottom_left")
        bottom_right = gather(ceils[0], ceils[1], "bottom_right")

        # now, do the actual interpolation
        with tf.name_scope("interpolate"):
            interp_top = alphas[1] * (top_right - top_left) + top_left
            interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
            interp = alphas[0] * (interp_bottom - interp_top) + interp_top

        return interp


def dense_image_warp(image, flow, name=None):
    """Image warping using per-pixel flow vectors.

    Apply a non-linear warp to the image, where the warp is specified by a
    dense flow field of offset vectors that define the correspondences of
    pixel values in the output image back to locations in the source image.
    Specifically, the pixel value at output[b, j, i, c] is
    images[b, j - flow[b, j, i, 0], i - flow[b, j, i, 1], c].

    The locations specified by this formula do not necessarily map to an int
    index. Therefore, the pixel value is obtained by bilinear
    interpolation of the 4 nearest pixels around
    (b, j - flow[b, j, i, 0], i - flow[b, j, i, 1]). For locations outside
    of the image, we use the nearest pixel values at the image boundary.

    Args:
      image: 4-D float `Tensor` with shape `[batch, height, width, channels]`.
      flow: A 4-D float `Tensor` with shape `[batch, height, width, 2]`.
      name: A name for the operation (optional).

      Note that image and flow can be of type tf.half, tf.float32, or
      tf.float64, and do not necessarily have to be the same type.

    Returns:
      A 4-D float `Tensor` with shape`[batch, height, width, channels]`
        and same type as input image.

    Raises:
      ValueError: if height < 2 or width < 2 or the inputs have the wrong
        number of dimensions.
    """
    with tf.name_scope(name or "dense_image_warp"):
        image = tf.convert_to_tensor(image)
        flow = tf.convert_to_tensor(flow)
        batch_size, height, width, channels = (
            tf.shape(image)[0],
            tf.shape(image)[1],
            tf.shape(image)[2],
            tf.shape(image)[3],
        )

        # The flow is defined on the image grid. Turn the flow into a list of query
        # points in the grid space.
        grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height))
        stacked_grid = tf.cast(tf.stack([grid_y, grid_x], axis=2), flow.dtype)
        batched_grid = tf.expand_dims(stacked_grid, axis=0)
        query_points_on_grid = batched_grid - flow
        query_points_flattened = tf.reshape(
            query_points_on_grid, [batch_size, height * width, 2]
        )
        # Compute values at the query points, then reshape the result back to the
        # image grid.
        interpolated = interpolate_bilinear(image, query_points_flattened)
        interpolated = tf.reshape(interpolated, [batch_size, height, width, channels])
        return interpolated

def inv_warp_flow(image, flow):
    return dense_image_warp(image, -flow)

# TODO 
# def inv_warp_flow(image, flow):
#     tfa.image.dense_image_warp(a, b)
#     _, H, W, _ = tf.unstack(tf.shape(image))
#     Hf, Wf = tf.cast(H, tf.float32), tf.cast(W, tf.float32)
#     # Turn the flow into a query points on the grid space
#     gx, gy = tf.meshgrid(tf.linspace(0.0, Wf-1, W), tf.linspace(0.0, Hf-1, H))
#     gy = 2 * (gy + flow[:,:,:,0]) / (Hf - 1) - 1
#     gx = 2 * (gx + flow[:,:,:,1]) / (Wf - 1) - 1
#     grid = tf.stack([gy, gx], axis=-1)
#     return grid_sample(image, grid)


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