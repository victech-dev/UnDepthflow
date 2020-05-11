import numpy as np
import tensorflow as tf


def tf_populate_pcd(depth, K):
    K_inv = tf.linalg.inv(K)
    _, H, W, _ = tf.unstack(tf.shape(depth))
    px, py = tf.meshgrid(tf.range(W), tf.range(H))
    px, py = tf.cast(px, tf.float32), tf.cast(py, tf.float32)
    xyz = tf.stack([px, py, tf.ones_like(px)], axis=-1)[None] * depth
    xyz = tf.squeeze(K_inv[:,None,None,:,:] @ xyz[:,:,:,:,None], axis=-1) # [b, H, W, 3]
    return xyz


def tf_detect_plane_xz(xyz):
    # Calculate covariance matrix of neighboring points
    # http://jacoposerafin.com/wp-content/uploads/bogoslavskyi13ecmr.pdf
    ksize = 5
    P = tf.nn.avg_pool2d(xyz, ksize, 1, 'SAME')
    xyz2 = xyz[:,:,:,:,None] @ xyz[:,:,:,None,:]
    xyz2_4d = tf.reshape(xyz2, tf.concat([tf.shape(xyz2)[:-2], [9]], 0))
    S_4d = tf.nn.avg_pool2d(xyz2_4d, ksize, 1, 'SAME')
    S = tf.reshape(S_4d, tf.concat([tf.shape(S_4d)[:-1], [3, 3]], 0))
    sigma = S - P[:,:,:,:,None] @ P[:,:,:,None,:]

    # eigenvalue solver of 3x3 symmetric matrix
    # much faster than tf.linalg.eigh when input is multiple matrices
    # https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
    A = sigma
    A_max = tf.reduce_max(tf.abs(A), axis=(-2,-1))
    eval_zero = tf.cast(A_max < 1e-6, tf.float32)

    A_scaled = A / tf.maximum(A_max[...,None,None], 1e-6)
    a00, a11, a22 = A_scaled[...,0,0], A_scaled[...,1,1], A_scaled[...,2,2]
    a01, a02, a12 = A_scaled[...,0,1], A_scaled[...,0,2], A_scaled[...,1,2]

    norm = a01**2 + a02**2 + a12**2
    q = (a00 + a11 + a22) / 3.0
    b00, b11, b22 = a00 - q, a11 - q, a22 - q
    p = tf.sqrt((b00**2 + b11**2 + b22**2 + 2.0 * norm) / 6.0)
    eig_triple = tf.cast(p < 1e-6, tf.float32) * (1.0 - eval_zero)

    c00 = b11 * b22 - a12 * a12
    c01 = a01 * b22 - a12 * a02
    c02 = a01 * a12 - b11 * a02
    det = (b00 * c00 - a01 * c01 + a02 * c02) / tf.maximum(p, 1e-6)**3
    half_det = tf.clip_by_value(0.5 * det, -1.0, 1.0)
    angle = tf.math.acos(half_det) / 3.0

    # beta0, beta1, beta2
    beta2 = tf.math.cos(angle) * 2.0
    beta0 = tf.math.cos(angle + np.pi * 2/3) * 2.0
    beta1 = -(beta0 + beta2)
    betas = tf.stack([beta0, beta1, beta2], axis=-1)

    # Merge 3 cases: all zeros, triple roots, ordinal case. (eval0 <= eval1 <= eval2)
    # Note this actually should be rescaled to initial values (eig *= A_max), 
    # but we are interesting in eigenvector, so let's just skip it. 
    evals = q[...,None] + p[...,None] * betas
    evals = (eig_triple * q)[...,None] + ((1.0 - eval_zero) * (1.0 - eig_triple))[...,None] * evals
    eval0, eval1, eval2 = tf.unstack(evals, axis=-1)

    # Calculate y-magnitude of eigenvector which has the smallest eigenvalue 
    # (= y-magnitude of normal vector from local pcd)
    # by using eigenvector-eigenvalue identity from https://arxiv.org/pdf/1908.03795.pdf
    denom2 = (eval0 - eval1) * (eval0 - eval2)
    denom = tf.sqrt(tf.maximum(denom2, 0))
    evec_zero = tf.cast(denom < 1e-6, tf.float32)
    ny2_num = eval0**2 - (a00 + a22) * eval0 + (a00 * a22 - a02**2)
    ny_num = tf.sqrt(tf.maximum(ny2_num, 0))
    ny = tf.clip_by_value(ny_num / tf.maximum(denom, 1e-6), 0, 1)
    return (1.0 - evec_zero) * ny
