#
# Copyright (c) 2021 The GPflux Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
An "encoder" for parametrizing latent variables. Does not work with mini-batching.
"""

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
    This class implements direct parameterisation of the Normally-distributed
    posterior of the latent variables. A mean and standard deviation to parameterise a
    mean-field Normal distribution for each latent variable is created and learned
    during training. This type of encoder is not computationally very efficient for
    larger datasets, but can greatly simplify training, because no neural network is
    required to learn an amortized mapping.

    ..note::
        No amortization is used; each datapoint element has an associated mean and
        standard deviation. This is not compatible with mini-batching.

    See :cite:t:`dutordoir2018cde` for a more thorough explanation of latent variable models
    and encoders.
    """

    means: Parameter
    """
    Each row contains the value of the mean for a latent variable in the model.
    ``means`` is a tensor of rank two with the shape ``[N, W]`` because we have the same number
    of latent variables as datapoints, and each latent variable is ``W``-dimensional.
    Consequently, the mean for each latent variable is also ``W``-dimensional.
    """

    stds: Parameter
    """
    Each row contains the value of the diagonal covariances for a latent variable.
    ``stds`` is a tensor of rank two with the shape ``[N, W]`` because we have the same number
    of latent variables as datapoints, and each latent variable is ``W``-dimensional.
    Consequently, the diagonal elements of the square covariance matrix for each latent variable
    is also ``W``-dimensional.

    Initialised to ``1e-5 * np.ones((N, W))``.
    """

    def __init__(self, num_data: int, latent_dim: int, means: Optional[np.ndarray] = None):
        """
        Directly parameterise the posterior of the latent variables associated with
        each datapoint with a diagonal multivariate Normal distribution. Note that across
        latent variables we assume a mean-field approximation.

        See :cite:t:`dutordoir2018cde` for a more thorough explanation of
        latent variable models and encoders.

        :param num_data: The number of datapoints, ``N``.
        :param latent_dim: The dimensionality of the latent variable, ``W``.
        :param means: The initialisation of the mean of the latent variable posterior
            distribution. (see :attr:`means`). If `None` (the default setting), set to
            ``np.random.randn(N, W) * 0.01``; otherwise, ``means`` should be an array of
            rank two with the shape ``[N, W]``.
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

        # TODO: Rename to `scale` and `loc` to match tfp.distributions
        self.means = Parameter(means, dtype=default_float(), name="w_means")
        self.stds = Parameter(stds, transform=positive(), dtype=default_float(), name="w_stds")

    def call(
        self, inputs: Optional[TensorType] = None, *args: Any, **kwargs: Any
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Return the posterior's mean (see :attr:`means`) and standard deviation (see :attr:`stds`).
        """
        if inputs is not None:
            tf.debugging.assert_shapes(
                [(self.means, ["N", "W"]), (self.stds, ["N", "W"]), (inputs, ["N", "D"])]
            )
        return self.means, self.stds
