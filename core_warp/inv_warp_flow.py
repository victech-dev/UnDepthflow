try:
    from core_warp import dense_image_warp
except:
    from dense_image_warp import dense_image_warp


def inv_warp_flow(image, flow):
    # TODO padding mode like grid_sample of pytorch
    return dense_image_warp(image, -flow)


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

