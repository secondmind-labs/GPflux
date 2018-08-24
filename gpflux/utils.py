# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


from typing import Tuple, Optional, Union

import numpy as np
import tensorflow as tf


Int = Union[tf.Tensor, int]


def lrelu(x, alpha=0.3):
    """
    Linear-REctified Linear Unit tf operation.
    Returns max(x, alpha * x)
    E.g. for alpha=0.3 (default):
        lrelu(x) = x      if x > 0
        lrelu(x) = 0.3 x  if x < 0
    """
    return tf.maximum(x, tf.multiply(x, alpha))


def xavier_weights(input_dim: int, output_dim: int) -> np.ndarray:
    """
    Xavier initialization for the weights of a NN layer
    :return: np.ndarray
        weight matrix of shape `input_dim` x `output_dim`,
        where each element is drawn i.i.d. from a normal distribution with
        zero mean and variance 2 / (input_dim + output_dim).

    See:
       Xavier Glorot and Yoshua Bengio (2010):
       Understanding the difficulty of training deep feedforward neural networks.
       International Conference on Artificial Intelligence and Statistics.
    """

    xavier_std = (2./(input_dim + output_dim)) ** 0.5
    return np.random.randn(input_dim, output_dim) * xavier_std


def get_image_patches(img, image_shape, filter_shape):
    with tf.name_scope('image_patches'):
        N, W, H, C = image_shape
        h, w = filter_shape
        img = tf.transpose(tf.reshape(img, tf.stack([tf.shape(img)[0], -1, C])), [0, 2, 1])
        img = tf.reshape(img, [-1, H, W, 1])
        patches = tf.extract_image_patches(img, [1, h, w, 1], [1, 1, 1, 1], [1, 1, 1, 1], 'VALID')
        shp = tf.shape(patches)  # img x out_rows x out_cols
        return tf.reshape(patches, [tf.shape(img)[0], C * shp[1] * shp[2], shp[3]])

