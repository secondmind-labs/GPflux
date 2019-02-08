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


def get_image_patches(imgs, image_shape, patch_shape, channels_in_patch=True):
    """
    Extracts patches with valid padding. Output color channel can be either included
    in patch dimension [N, P, h*w*C] or grid dimension [N, P*C, h*w]
    :param imgs: Images tensor [N, H, W, C].
    :param image_shape: Image shape tuple.
    :param patch_shape: Patch shape tuple.
    :param channels_in_patch: Boolean keyword argument for choosing where channels
        dimension will be placed. True - patch dimension, False - grid dimension.
    :return: Tensor of [N, P, h*w*C] or [N, P*C, h*w] shape.
    """
    with tf.name_scope('image_patches'):
        _, _, C = image_shape
        h, w = patch_shape
        N = tf.shape(imgs)[0]
        ones = [1] * 4
        patches = tf.extract_image_patches(imgs, [1, h, w, 1], ones, ones, 'VALID')
        return tf.reshape(patches, [N, -1, h*w*C])
