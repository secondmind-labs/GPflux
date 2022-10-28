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

import tensorflow as tf

import gpflow
from gpflow import Param, params_as_tensors


class NonstationaryKernel(gpflow.kernels.Kernel):
    def __init__(self, basekernel, lengthscales_dim, scaling_offset=0.0, positivity=tf.exp):
        """
        basekernel.input_dim == data_dim: dimension of original (first-layer) input
        lengthscales_dim: output dimension of previous layer

        For the nonstationary deep GP, the `positivity` is actually part of the *model*
        specification and does change model behaviour.
        `scaling_offset` behaves equivalent to a constant mean function in the previous
        layer; when giving the previous layer a constant mean function, it should be set
        to non-trainable.

        When the positivity is chosen to be exp(), then the scaling offset is also
        equivalent to the basekernel lengthscale, hence the default is 0 and non-trainable.
        For other positivity choices, it may be useful to enable scaling_offset to be trainable.

        Scaling offset can be scalar or be a vector of length `data_dim`.
        """
        if not isinstance(basekernel, gpflow.kernels.Stationary):
            raise TypeError("base kernel must be stationary")

        data_dim = basekernel.input_dim
        # must call super().__init__() before adding parameters:
        super().__init__(data_dim + lengthscales_dim)

        self.data_dim = data_dim
        self.lengthscales_dim = lengthscales_dim
        if lengthscales_dim not in (data_dim, 1):
            raise ValueError("lengthscales_dim must be equal to basekernel's input_dim or 1")

        self.basekernel = basekernel
        self.scaling_offset = Param(scaling_offset)
        if positivity is tf.exp:
            self.scaling_offset.trainable = False
        self._positivity = positivity  # XXX use softplus / log(1+exp)?

    @params_as_tensors
    def _split_scale(self, X):
        """
        X has data_dim + lengthscales_dim elements
        """
        original_input = X[..., : self.data_dim]
        log_lengthscales = X[..., self.data_dim :]
        lengthscales = self._positivity(log_lengthscales + self.scaling_offset)
        return original_input, lengthscales

    @params_as_tensors
    def K(self, X, X2=None):
        base_lengthscales = self.basekernel.lengthscales
        x1, lengthscales1 = self._split_scale(X)
        if X2 is not None:
            x2, lengthscales2 = self._split_scale(X2)
        else:
            x2, lengthscales2 = x1, lengthscales1

        # d = self.lengthscales_dim, either 1 or D
        # x1: [N, D], lengthscales1: [N, d]
        # x2: [M, D], lengthscales2: [M, d]

        # axis=0 and axis=-3 are equivalent, and axis=1 and axis=-2 are equivalent
        lengthscales1 = tf.expand_dims(lengthscales1, axis=-2)  # [N, 1, d]
        lengthscales2 = tf.expand_dims(lengthscales2, axis=-3)  # [1, M, d]
        squared_lengthscales_sum = lengthscales1 ** 2 + lengthscales2 ** 2  # [N, M, d]

        q = tf.reduce_prod(
            tf.sqrt(2 * lengthscales1 * lengthscales2 / squared_lengthscales_sum), axis=-1
        )  # [N, M]

        if self.lengthscales_dim == 1:  # also use this when data_dim==1
            # Last dimension should be 1, so the reduce_prod above *should* have no effect here.
            # The Stationary._scaled_square_dist() handles the base lengthscale rescaling,
            # but we need to correct the `q` prefactor:
            q = q ** tf.cast(self.data_dim, q.dtype)
            r2 = self.basekernel._scaled_square_dist(x1, None if X2 is None else x2)
            r2 /= 0.5 * squared_lengthscales_sum[..., 0]  # [N, M]

        elif self.lengthscales_dim == self.data_dim:
            assert self.data_dim > 1

            # Here, we have to handle the rescaling and expanding manually, as the full case cannot
            # be implemented with a matmul.
            x1 /= base_lengthscales
            x2 /= base_lengthscales
            x1 = tf.expand_dims(x1, axis=-2)  # [N, 1, D]
            x2 = tf.expand_dims(x2, axis=-3)  # [1, M, D]

            r2 = tf.reduce_sum(2 * (x1 - x2) ** 2 / squared_lengthscales_sum, axis=-1)  # [N, M]

        else:
            assert False, "lengthscales_dim must be 1 or same as data_dim"

        return q * self.basekernel.K_r2(r2)

    @params_as_tensors
    def Kdiag(self, X):
        return self.basekernel.Kdiag(X)
