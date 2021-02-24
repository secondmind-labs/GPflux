# Copyright (C) PROWLER.io 2020 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from typing import Any, Optional, Tuple

import numpy as np
import tensorflow as tf

from gpflow import Parameter, default_float
from gpflow.base import TensorType
from gpflow.utilities.bijectors import positive

from gpflux.exceptions import EncoderInitializationError
from gpflux.layers import TrackableLayer


class DirectlyParameterizedNormalDiag(TrackableLayer):
    """
    Directly parameterize the posterior (and prior) of the latent variables
    associated with each data point with a diagonal multivariate Normal
    distribution.

    No amortization is used; each datapoint element has an associated mean and
    variance. IMPORTANT: Not compatible with minibatches.
    """

    def __init__(self, num_data: int, latent_dim: int, means: Optional[np.ndarray] = None):
        """
        :param num_data: The number of data points
        :param latent_dim: The dimensionality of the latent variable
        :param means: The means of the Normal distribution to initialise to,
            shape [num_data, latent_dim]. By default, drawn from N(0, 0.01^2).
        """
        super().__init__()
        if means is None:
            # break the symmetry in the means:
            means = 0.01 * np.random.randn(num_data, latent_dim)
        else:
            if np.any(means.shape != (num_data, latent_dim)):
                raise EncoderInitializationError(
                    f"means must have shape [num_data, latent_dim] = [{num_data}, {latent_dim}]; "
                    f"got {means.shape} instead."
                )

        # initialise distribution with a small standard deviation, as this has
        # been observed to help fitting:
        stds = 1e-5 * np.ones_like(means)

        self.means = Parameter(means, dtype=default_float(), name="w_means")
        self.stds = Parameter(stds, transform=positive(), dtype=default_float(), name="w_stds")

    def call(
        self, inputs: Optional[TensorType] = None, *args: Any, **kwargs: Any
    ) -> Tuple[TensorType, TensorType]:
        if inputs is not None:
            tf.debugging.assert_shapes(
                [(self.means, ["N", "W"]), (self.stds, ["N", "W"]), (inputs, ["N", "D"])]
            )
        return self.means, self.stds
