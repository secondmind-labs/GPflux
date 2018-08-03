# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


import tensorflow as tf
import numpy as np

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

    xavier_std = (2./(input_dim + output_dim))**0.5
    return np.random.randn(input_dim, output_dim) * xavier_std
