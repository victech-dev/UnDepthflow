import argparse
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def jbu(source, reference, radius=2, sigma_spatial=2.5, sigma_range=None):
    # parser = argparse.ArgumentParser(description="Perform Joint Bilateral Upsampling with a source and reference image")
    # parser.add_argument("source", help="Path to the source image")
    # parser.add_argument("reference", help="Path to the reference image")
    # parser.add_argument("output", help="Path to the output image")
    # parser.add_argument('--radius', dest='radius', default=2, help='Radius of the filter kernels (default: 2)')
    # parser.add_argument('--sigma-spatial', dest='sigma_spatial', default=2.5, help='Sigma of the spatial weights (default: 2.5)')
    # parser.add_argument('--sigma-range', dest='sigma_range', help='Sigma of the range weights (default: standard deviation of the reference image)')
    # args = parser.parse_args()

    Hs, Ws = source.shape[0:2]
    Hr, Wr = reference.shape[0:2]
    source_upsampled = cv2.resize(source, (Wr, Hr), interpolation=cv2.INTER_LINEAR)
    source_upsampled = source_upsampled[:,:,None]

    scale = Ws / Wr
    diameter = 2 * radius + 1
    step = int(np.ceil(1 / scale))
    padding = radius * step
    sigma_range = sigma_range if sigma_range else np.std(reference)

    reference = np.pad(reference, ((padding, padding), (padding, padding), (0, 0)), 'symmetric').astype(np.float32)
    source_upsampled = np.pad(source_upsampled, ((padding, padding), (padding, padding), (0, 0)), 'symmetric').astype(np.float32)

    # Spatial Gaussian function.
    x, y = np.meshgrid(np.arange(diameter) - radius, np.arange(diameter) - radius)
    kernel_spatial = np.exp(-1.0 * (x**2 + y**2) /  (2 * sigma_spatial**2))
    kernel_spatial = np.repeat(kernel_spatial, 1).reshape(-1, 1)

    # Lookup table for range kernel.
    lut_range = np.exp(-1.0 * np.arange(256)**2 / (2 * sigma_range**2)) 

    def process_row(y):
        result = np.zeros((Wr, 1))
        y += padding
        for x in range(padding, Wr - padding):
            I_p = reference[y, x]
            patch_reference = reference[y - padding:y + padding + 1:step, x - padding:x + padding + 1:step].reshape(-1, 1)
            patch_source_upsampled = source_upsampled[y - padding:y + padding + 1:step, x - padding:x + padding + 1:step].reshape(-1, 1)

            kernel_range = lut_range[np.abs(patch_reference - I_p).astype(int)]
            weight = kernel_range * kernel_spatial
            k_p = weight.sum(axis=0)
            result[x - padding] = np.round(np.sum(weight * patch_source_upsampled, axis=0) / k_p)

        return result

    # executor = ProcessPoolExecutor()
    # result = executor.map(process_row, range(Hr))
    # executor.shutdown(True)
    # return np.array(list(result))
    results = [process_row(y) for y in range(Hr)]
    return np.array(results)
