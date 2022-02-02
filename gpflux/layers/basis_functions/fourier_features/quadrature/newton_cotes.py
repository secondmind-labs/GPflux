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
Kernel decompositon into features and coefficients based on Newton-Cotes quadrature.
"""
import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal, multivariate_t

import gpflow
from gpflow.quadrature.gauss_hermite import repeat_as_list, reshape_Z_dZ

from gpflux.layers.basis_functions.fourier_features.quadrature.base import (
    QuadratureFourierFeaturesBase,
    TanTransform,
)
from gpflux.layers.basis_functions.fourier_features.utils import _matern_dof
from gpflux.types import ShapeType


class SimpsonQuadratureFourierFeatures(QuadratureFourierFeaturesBase):

    SUPPORTED_KERNELS = (
        gpflow.kernels.SquaredExponential,
        gpflow.kernels.Matern12,
        gpflow.kernels.Matern32,
        gpflow.kernels.Matern52,
    )

    def _compute_output_dim(self, input_shape: ShapeType) -> int:
        input_dim = input_shape[-1]
        n_abscissa = 2 * self.n_components + 1
        return 2 * n_abscissa ** input_dim

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

        stop = 1.
        start = -1.

        # `n_components` denotes half the desired number of intervals
        n_abscissa = 2 * self.n_components + 1
        width = np.true_divide(stop - start, n_abscissa - 1)

        # raw 1-dimensional quadrature nodes (L,)
        abscissa_value_flat = np.linspace(start, stop, n_abscissa)

        alpha = np.atleast_2d(4.)
        beta = np.atleast_2d(2.)
        a = np.hstack([beta, alpha])

        factors_value_flat = np.append(np.tile(a, reps=self.n_components), beta, axis=-1)
        factors_value_flat *= width
        factors_value_flat /= 3.
        factors_value_flat[..., [0, -1]] /= 2.  # halve first and last weight

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

        # Quadrature nodes (L^D, D)
        self.abscissa = tf.Variable(initial_value=abscissa_value, trainable=False)
        # Quadrature weights (L^D,)
        self.factors = tf.Variable(initial_value=factors_value, trainable=False)

        super(SimpsonQuadratureFourierFeatures, self).build(input_shape)
