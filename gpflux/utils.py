# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


from typing import Sequence

import numpy as np
import tensorflow as tf


def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))


def xavier_weights(input_dim: int, output_dim: int) -> np.ndarray:
    """
    Xavier initialization for the weights of NN layer
    :return: np.array
        weight matrix of shape `input_dim` x `output_dim`,
        where each element is drawn i.i.d. from N(0, sqrt(2. / (in + out)))

    See:
       Xavier Glorot and Yoshua Bengio (2010):
       Understanding the difficulty of training deep feedforward neural networks.
       International conference on artificial intelligence and statistics.
    """

    xavier_std = (2./(input_dim + output_dim)) ** 0.5
    return np.random.randn(input_dim, output_dim) * xavier_std



def dotconv(A: tf.Tensor, kernel_size: Sequence[int],
            back_prop: int = False,
            parallel: int = 10,
            swap_memory: bool = False) -> tf.Tensor:
    """
    Inner product of image sub-patches using convolution opertion.

    Args:
        A: Input tensor of shape NxHxWxC.
        kernel_size: Kernel size values [h, w], where h << H and w << W.
        parallel_iterations: (optional) The number of iterations allowed to run in parallel.
        back_prop: (optional) True enables support for back propagation.
        swap_memory: (optional) True enables GPU-CPU memory swapping.

    Returns:
        Tensor with NxPxPxC shape, where P = (H - h + 1) * (W - w + 1)
    """
    h, w = kernel_size
    rank_assert = tf.assert_rank(A, 4, message="Tensor with NxHxWxC shape expected")

    with tf.control_dependencies([rank_assert]):
        A_shape = tf.shape(A)
        N, H, W, C = [A_shape[i] for i in range(4)]

    ph, pw = H-h+1, W-h+1
    P = ph * pw

    At = tf.transpose(A, [1, 2, 0, 3])
    At = tf.reshape(At, [1, H, W, N*C])

    def inner_product(idx: tf.Tensor) -> tf.Tensor:
        sh, sw = idx // pw, idx % pw
        eh, ew = sh + h, sw + w

        At_kernel = At[:, sh:eh, sw:ew, :]
        At_kernel = tf.transpose(At_kernel, [1, 2, 3, 0])
        return tf.nn.depthwise_conv2d(At, At_kernel, strides=[1, 1, 1, 1], padding='VALID')

    indices = tf.range(P)
    products = tf.map_fn(inner_product, indices, swap_memory=swap_memory,
                         parallel_iterations=parallel,
                         back_prop=back_prop, dtype=A.dtype)  # (P, 1, ph, pw, N*C)

    products = tf.reshape(products, [P, P, N, C])
    return tf.transpose(products, [2, 0, 1, 3])
