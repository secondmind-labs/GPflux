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
Kernel decompositon into features and coefficients based on Gauss-Christoffel
quadrature aka Gaussian quadrature.
"""

import warnings
from typing import Mapping

import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal, multivariate_t

import gpflow

# from gpflow.config import default_float
from gpflow.quadrature.gauss_hermite import ndgh_points_and_weights, repeat_as_list, reshape_Z_dZ

from gpflux.layers.basis_functions.fourier_features.quadrature.base import (
    QuadratureFourierFeaturesBase,
    TanTransform,
)
from gpflux.layers.basis_functions.fourier_features.utils import _matern_dof
from gpflux.types import ShapeType


class GaussianQuadratureFourierFeatures(QuadratureFourierFeaturesBase):

    def __init__(self, kernel: gpflow.kernels.Kernel, n_components: int, **kwargs: Mapping):
        super(GaussianQuadratureFourierFeatures, self).__init__(kernel, n_components, **kwargs)
        if tf.reduce_any(tf.less(kernel.lengthscales, 1e-1)):
            warnings.warn(
                "Fourier feature approximation of kernels with small " 
                "lengthscales using Gaussian quadrature can have "
                "unexpected behaviors!"
            )


class GaussHermiteQuadratureFourierFeatures(GaussianQuadratureFourierFeatures):

    SUPPORTED_KERNELS = (gpflow.kernels.SquaredExponential,)

    def build(self, input_shape: ShapeType) -> None:
        """
        Creates the variables of the layer.
        See `tf.keras.layers.Layer.build()
        <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#build>`_.
        """
        input_dim = input_shape[-1]
        # (L^D, D), (L^D, 1)
        abscissa_value, factors_value = ndgh_points_and_weights(
            dim=input_dim, n_gh=self.n_components
        )
        factors_value = tf.squeeze(factors_value, axis=-1)  # (L^D,)

        # Gauss-Christoffel nodes (L^D, D)
        self.abscissa = tf.Variable(initial_value=abscissa_value, trainable=False)
        # Gauss-Christoffel weights (L^D,)
        self.factors = tf.Variable(initial_value=factors_value, trainable=False)
        super(GaussHermiteQuadratureFourierFeatures, self).build(input_shape)


class GaussLegendreQuadratureFourierFeatures(GaussianQuadratureFourierFeatures):

    SUPPORTED_KERNELS = (
        gpflow.kernels.SquaredExponential,
        gpflow.kernels.Matern12,
        gpflow.kernels.Matern32,
        gpflow.kernels.Matern52,
    )

    def build(self, input_shape: ShapeType) -> None:
        """
        Creates the variables of the layer.
        See `tf.keras.layers.Layer.build()
        <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#build>`_.
        """
        input_dim = input_shape[-1]

        if isinstance(self.kernel, gpflow.kernels.SquaredExponential):
            dist = multivariate_normal(mean=np.zeros(input_dim))
        else:
            nu = _matern_dof(self.kernel)  # degrees of freedom
            dist = multivariate_t(loc=np.zeros(input_dim), df=nu)

        # raw 1-dimensional quadrature nodes and weights (L,) (L,)
        abscissa_value_flat, factors_value_flat = np.polynomial.legendre.leggauss(
            deg=self.n_components
        )

        # transformed 1-dimensional quadrature nodes and weights
        transform = TanTransform()
        factors_value_flat *= transform.multiplier(abscissa_value_flat)  # (L,)
        abscissa_value_flat = transform(abscissa_value_flat)  # (L,)

        # transformed D-dimensional quadrature nodes and weights
        abscissa_value_rep = repeat_as_list(abscissa_value_flat, n=input_dim)  # (L, ..., L)
        factors_value_rep = repeat_as_list(factors_value_flat, n=input_dim)  # (L, ..., L)
        # (L^D, D), (L^D, 1)
        abscissa_value, factors_value = reshape_Z_dZ(abscissa_value_rep, factors_value_rep)

        factors_value = tf.squeeze(factors_value, axis=-1)  # (L^D,)
        factors_value *= dist.pdf(abscissa_value)  # (L^D,)

        # Gauss-Christoffel nodes (L^D, D)
        self.abscissa = tf.Variable(initial_value=abscissa_value, trainable=False)
        # Gauss-Christoffel weights (L^D,)
        self.factors = tf.Variable(initial_value=factors_value, trainable=False)

        super(GaussLegendreQuadratureFourierFeatures, self).build(input_shape)


class QuadratureFourierFeatures(GaussHermiteQuadratureFourierFeatures):
    """
    Alias for `GaussHermiteQuadratureFourierFeatures`.
    """

    pass
