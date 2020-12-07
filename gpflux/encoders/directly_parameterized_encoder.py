# Copyright (C) PROWLER.io 2020 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from typing import Any, Optional, Tuple

import numpy as np
import tensorflow as tf

from gpflow import Parameter
from gpflow.base import TensorType
from gpflow.utilities.bijectors import positive

from gpflux.exceptions import EncoderInitializationError
from gpflux.layers.trackable_layer import TrackableLayer


class DirectlyParameterizedNormalDiag(TrackableLayer):
    """
    Directly parameterize the posterior (and prior) of the latent variables
    associated with each data point with a diagonal multivariate Normal
    distribution.

    No amortization is used; each datapoint element has an associated mean and
    variance, IMPORTANT: Not compatible with minibatches
    """

    def __init__(self, num_data: int, latent_dim: int, means: Optional[np.ndarray] = None):
        """
        :param num_data: The number of data points
        :param latent_dim: The dimensionality of the latent variable
        :param means: The means of the Normal distribution to initialise to,
            shape [num_data, latent_dim]. By default, drawn from N(0, 1e-4).
        """
        super().__init__()
        if means is None:
            # break the symmetry in the means:
            means = 0.01 * np.random.randn(num_data, latent_dim)
        else:
            if np.any(tf.shape(means) != (num_data, latent_dim)):
                raise EncoderInitializationError

        # initialise distribution with a small standard deviation, as this has
        # been observed to help fitting:
        stds = 1e-5 * np.ones(shape=(num_data, latent_dim), dtype=np.float64)

        self.means = tf.Variable(means, name="w_means")
        self.stds = Parameter(stds, transform=positive(), name="w_stds")

    def call(
        self, inputs: Optional[TensorType] = None, *args: Any, **kwargs: Any
    ) -> Tuple[TensorType, TensorType]:
        if inputs is not None:
            tf.debugging.assert_shapes([(self.means, ["N", "W"]), (inputs, ["N", "D"])])
        return self.means, self.stds
