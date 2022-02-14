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

from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import multivariate_normal, multivariate_t
from scipy.stats.qmc import MultivariateNormalQMC

import gpflow
from gpflow.base import DType, TensorType

from gpflux.layers.basis_functions.fourier_features.random.base import RandomFourierFeatures
from gpflux.layers.basis_functions.fourier_features.utils import _matern_dof
from gpflux.types import ShapeType

tfd = tfp.distributions


class QuasiRandomFourierFeatures(RandomFourierFeatures):

    """
    Quasi-random Fourier features (ORF) :cite:p:`yang2014quasi` for more
    efficient and accurate kernel approximations than random Fourier features.
    """

    def _weights_init(self, shape: TensorType, dtype: Optional[DType] = None) -> TensorType:
        n_components, input_dim = shape  # M, D
        sampler = MultivariateNormalQMC(mean=np.zeros(input_dim))
        return sampler.random(n=n_components)  # shape [M, D]


class ReweightedQuasiRandomFourierFeatures(QuasiRandomFourierFeatures):

    SUPPORTED_KERNELS = (
        gpflow.kernels.SquaredExponential,
        gpflow.kernels.Matern12,
        gpflow.kernels.Matern32,
        gpflow.kernels.Matern52,
    )

    def _compute_constant(self) -> tf.Tensor:
        """
        Compute normalizing constant for basis functions.

        :return: A tensor with the shape ``[]`` (i.e. a scalar).
        """
        return (
            tf.tile(tf.sqrt(self.importance_weight), multiples=[2])
            * super(ReweightedQuasiRandomFourierFeatures, self)._compute_constant()
        )

    def build(self, input_shape: ShapeType) -> None:
        """
        Creates the variables of the layer.
        See `tf.keras.layers.Layer.build()
        <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#build>`_.
        """
        super(ReweightedQuasiRandomFourierFeatures, self).build(input_shape)

        input_dim = input_shape[-1]
        importance_weight_value = tf.ones(self.n_components, dtype=self.dtype)

        if not isinstance(self.kernel, gpflow.kernels.SquaredExponential):
            nu = _matern_dof(self.kernel)  # degrees of freedom
            q = tfd.MultivariateNormalDiag(loc=tf.zeros(input_dim, dtype=self.dtype))
            p = tfd.MultivariateStudentTLinearOperator(
                df=nu,
                loc=tf.zeros(input_dim, dtype=self.dtype),
                scale=tf.linalg.LinearOperatorLowerTriangular(tf.eye(input_dim, dtype=self.dtype)),
            )
            importance_weight_value = tf.exp(p.log_prob(self.W) - q.log_prob(self.W))

        self.importance_weight = tf.Variable(initial_value=importance_weight_value,
                                             trainable=False)
