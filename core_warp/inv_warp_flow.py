import tensorflow as tf
try:
    from core_warp import dense_image_warp
except:
    from dense_image_warp import dense_image_warp


def inv_warp_flow(image, flow, padding='zeros'):
    if padding == 'border':
        return dense_image_warp(image, -flow)
    elif padding == 'zeros':
        image_pad = tf.pad(image, [[0,0], [1,1], [1,1], [0,0]])
        flow_pad = tf.pad(flow, [[0,0], [1,1], [1,1], [0,0]])
        w = dense_image_warp(image_pad, -flow_pad)
        return w[:,1:-1,1:-1]
    else:
        raise ValueError('Not supported padding type')


if __name__ == '__main__':
    import numpy as np

    image = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)
    image = np.reshape(image, [1, 3, 3, 1])
    print("*** Image:", np.squeeze(image))

    flo = np.zeros((1, 3, 3, 2), dtype=np.float32)
    flo[0, 1, 1, 1] = 1.0

    # output should be [[1,2,3],[4,6,6],[7,8,9]]
    image2 = inv_warp_flow(image, flo)
    print("*** Image:", np.squeeze(image2))

