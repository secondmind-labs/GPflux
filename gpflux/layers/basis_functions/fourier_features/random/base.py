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
from itertools import cycle
from typing import Callable, Mapping, Optional, Tuple, Type

import numpy as np
import tensorflow as tf

import gpflow
from gpflow.base import DType, TensorType
from gpflow.kernels import Kernel

from gpflux.layers.basis_functions.fourier_features.base import FourierFeaturesBase
from gpflux.layers.basis_functions.fourier_features.utils import (
    _bases_concat,
    _bases_cosine,
    _matern_number,
)
from gpflux.types import ShapeType

"""
Kernels supported by :class:`RandomFourierFeatures`.

You can build RFF for shift-invariant stationary kernels from which you can
sample frequencies from their power spectrum, following Bochner's theorem.
"""
RFF_SUPPORTED_KERNELS: Tuple[Type[gpflow.kernels.Stationary], ...] = (
    gpflow.kernels.SquaredExponential,
    gpflow.kernels.Matern12,
    gpflow.kernels.Matern32,
    gpflow.kernels.Matern52,
)

RFF_SUPPORTED_MULTIOUTPUTS: Tuple[Type[gpflow.kernels.MultioutputKernel], ...] = (
    gpflow.kernels.SeparateIndependent,
    gpflow.kernels.SharedIndependent,
)

RFF_SUPPORTED_COMBINATION: Tuple[Type[gpflow.kernels.Combination], ...] = (gpflow.kernels.Sum,)


def _sample_students_t(nu: float, shape: ShapeType, dtype: DType) -> TensorType:
    """
    Draw samples from a (central) Student's t-distribution using the following:
      BETA ~ Gamma(nu/2, nu/2) (shape-rate parameterization)
      X ~ Normal(0, 1/BETA)
    then:
      X ~ StudentsT(nu)

    Note this is equivalent to the more commonly used parameterization
      Z ~ Chi2(nu) = Gamma(nu/2, 1/2)
      EPSILON ~ Normal(0, 1)
      X = EPSILON * sqrt(nu/Z)

    To see this, note
      Z/nu ~ Gamma(nu/2, nu/2)
    and
      X ~ Normal(0, nu/Z)
    The equivalence becomes obvious when we set BETA = Z/nu
    """
    # Normal(0, 1)
    normal_rvs = tf.random.normal(shape=shape, dtype=dtype)
    shape = tf.concat([shape[:-1], [1]], axis=0)
    # Gamma(nu/2, nu/2)
    gamma_rvs = tf.random.gamma(shape, alpha=0.5 * nu, beta=0.5 * nu, dtype=dtype)
    # StudentsT(nu)
    students_t_rvs = tf.math.rsqrt(gamma_rvs) * normal_rvs
    return students_t_rvs


class RandomFourierFeaturesBase(FourierFeaturesBase):
    def __init__(self, kernel: gpflow.kernels.Kernel, n_components: int, **kwargs: Mapping):
        assert isinstance(
            kernel, (RFF_SUPPORTED_KERNELS, RFF_SUPPORTED_MULTIOUTPUTS, RFF_SUPPORTED_COMBINATION)
        ), "Unsupported Kernel: only the following kernel types are supported: {}".format(
            [
                k.__name__
                for k in (
                    RFF_SUPPORTED_MULTIOUTPUTS + RFF_SUPPORTED_KERNELS + RFF_SUPPORTED_COMBINATION
                )
            ]
        )
        if isinstance(kernel, RFF_SUPPORTED_MULTIOUTPUTS):
            for k in kernel.latent_kernels:
                assert isinstance(k, RFF_SUPPORTED_KERNELS), (
                    f"Unsupported Kernel within the multioutput kernel; only the following"
                    f"kernel types are supported: "
                    f"{[k.__name__ for k in RFF_SUPPORTED_KERNELS]}"
                )
        elif isinstance(kernel, RFF_SUPPORTED_COMBINATION):
            assert all(isinstance(k, RFF_SUPPORTED_KERNELS) for k in kernel.kernels), (
                f"Unsupported Kernel within the combination kernel; only the following"
                f"kernel types are supported: "
                f"{[k.__name__ for k in RFF_SUPPORTED_KERNELS]}"
            )
        super(RandomFourierFeaturesBase, self).__init__(kernel, n_components, **kwargs)

    def build(self, input_shape: ShapeType) -> None:
        """
        Creates the variables of the layer.
        See `tf.keras.layers.Layer.build()
        <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#build>`_.
        """
        input_dim = input_shape[-1]
        self._weights_build(input_dim, n_components=self.n_components)
        super(RandomFourierFeaturesBase, self).build(input_shape)

    def _active_input_dim(self, input_dim: int, kernel: Kernel) -> int:
        dummy_X = tf.zeros((0, input_dim), dtype=tf.float64)
        return kernel.slice(dummy_X, None)[0].shape[-1]

    def _weights_build(self, input_dim: int, n_components: int) -> None:
        if self.is_batched:
            # TODO: handle nested active_dims
            self.W = [
                self.add_weight(
                    name="weights",
                    trainable=False,
                    shape=(n_components, self._active_input_dim(input_dim, k)),
                    dtype=self.dtype,
                    initializer=self._weights_init(k),
                )
                # SharedIndependent repeatedly use the same sub_kernel
                for _, k in zip(range(self.batch_size), cycle(self.sub_kernels))
            ]
        else:
            self.W = self.add_weight(
                name="weights",
                trainable=False,
                shape=(n_components, self._active_input_dim(input_dim, self.kernel)),
                dtype=self.dtype,
                initializer=self._weights_init(self.kernel),
            )

    def _weights_init_individual(
        self,
        kernel: gpflow.kernels.Kernel,
        shape: TensorType,
        dtype: Optional[DType] = None,
    ) -> TensorType:
        if isinstance(kernel, gpflow.kernels.SquaredExponential):
            return tf.random.normal(shape, dtype=dtype)
        else:
            p = _matern_number(kernel)
            nu = 2.0 * p + 1.0  # degrees of freedom
            return _sample_students_t(nu, shape, dtype)

    def _weights_init(self, kernel: Kernel) -> Callable[[TensorType, Optional[DType]], TensorType]:
        def _initializer(shape: TensorType, dtype: Optional[DType] = None) -> TensorType:
            return self._weights_init_individual(kernel, shape, dtype)  # [M, D]

        return _initializer

    @staticmethod
    def rff_constant(variance: TensorType, output_dim: int) -> tf.Tensor:
        """
        Normalizing constant for random Fourier features.
        """
        return tf.sqrt(tf.math.truediv(2.0 * variance, output_dim))


class RandomFourierFeatures(RandomFourierFeaturesBase):
    r"""
    Random Fourier features (RFF) is a method for approximating kernels. The essential
    element of the RFF approach :cite:p:`rahimi2007random` is the realization that Bochner's theorem
    for stationary kernels can be approximated by a Monte Carlo sum.

    We will approximate the kernel :math:`k(\mathbf{x}, \mathbf{x}')`
    by :math:`\Phi(\mathbf{x})^\top \Phi(\mathbf{x}')`
    where :math:`\Phi: \mathbb{R}^{D} \to \mathbb{R}^{M}` is a finite-dimensional feature map.

    The feature map is defined as:

    .. math::

      \Phi(\mathbf{x}) = \sqrt{\frac{2 \sigma^2}{\ell}}
        \begin{bmatrix}
          \cos(\boldsymbol{\theta}_1^\top \mathbf{x}) \\
          \sin(\boldsymbol{\theta}_1^\top \mathbf{x}) \\
          \vdots \\
          \cos(\boldsymbol{\theta}_{\frac{M}{2}}^\top \mathbf{x}) \\
          \sin(\boldsymbol{\theta}_{\frac{M}{2}}^\top \mathbf{x})
        \end{bmatrix}

    where :math:`\sigma^2` is the kernel variance.
    The features are parameterised by random weights:

    - :math:`\boldsymbol{\theta} \sim p(\boldsymbol{\theta})`
      where :math:`p(\boldsymbol{\theta})` is the spectral density of the kernel.

    At least for the squared exponential kernel, this variant of the feature
    mapping has more desirable theoretical properties than its counterpart form
    from phase-shifted cosines :class:`RandomFourierFeaturesCosine` :cite:p:`sutherland2015error`.
    """

    def compute_output_dim(self, input_shape: ShapeType) -> int:
        dim = 2 * self.n_components
        if self.is_batched and not self.is_multioutput:
            # For combination kernels, the number of features is multiplied by the number of
            # sub-kernels.
            dim *= self.batch_size
        return dim

    def _compute_bases(self, inputs: TensorType, batch: Optional[int]) -> tf.Tensor:
        """
        Compute basis functions.

        :return: A tensor with the shape ``[N, 2M]`` or ``[P, N, 2M]``.
        """
        return _bases_concat(inputs, self.W if batch is None else self.W[batch])

    def _compute_constant(self) -> tf.Tensor:
        """
        Compute normalizing constant for basis functions.

        :return: A tensor with the shape ``[]`` (i.e. a scalar).
        """
        if self.is_batched:
            constants = [
                self.rff_constant(k.variance, output_dim=2 * self.n_components)
                for k in self.sub_kernels
            ]
            return tf.stack(constants, 0)[:, None, None]  # [P, 1, 1]
        else:
            return self.rff_constant(self.kernel.variance, output_dim=2 * self.n_components)


class RandomFourierFeaturesCosine(RandomFourierFeaturesBase):
    r"""
    Random Fourier Features (RFF) is a method for approximating kernels. The essential
    element of the RFF approach :cite:p:`rahimi2007random` is the realization that Bochner's theorem
    for stationary kernels can be approximated by a Monte Carlo sum.

    We will approximate the kernel :math:`k(\mathbf{x}, \mathbf{x}')`
    by :math:`\Phi(\mathbf{x})^\top \Phi(\mathbf{x}')` where
    :math:`\Phi: \mathbb{R}^{D} \to \mathbb{R}^{M}` is a finite-dimensional feature map.

    The feature map is defined as:

    .. math::
      \Phi(\mathbf{x}) = \sqrt{\frac{2 \sigma^2}{\ell}}
        \begin{bmatrix}
          \cos(\boldsymbol{\theta}_1^\top \mathbf{x} + \tau) \\
          \vdots \\
          \cos(\boldsymbol{\theta}_M^\top \mathbf{x} + \tau)
        \end{bmatrix}

    where :math:`\sigma^2` is the kernel variance.
    The features are parameterised by random weights:

    - :math:`\boldsymbol{\theta} \sim p(\boldsymbol{\theta})`
      where :math:`p(\boldsymbol{\theta})` is the spectral density of the kernel
    - :math:`\tau \sim \mathcal{U}(0, 2\pi)`

    Equivalent to :class:`RandomFourierFeatures` by elementary trigonometric identities.
    """

    def build(self, input_shape: ShapeType) -> None:
        """
        Creates the variables of the layer.
        See `tf.keras.layers.Layer.build()
        <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#build>`_.
        """
        self._bias_build(n_components=self.n_components)
        super(RandomFourierFeaturesCosine, self).build(input_shape)

    def _bias_build(self, n_components: int) -> None:
        if self.is_batched:
            self.b = [
                self.add_weight(
                    name="bias",
                    trainable=False,
                    shape=(1, n_components),
                    dtype=self.dtype,
                    initializer=self._bias_init,
                )
                for _ in range(self.batch_size)
            ]
        else:
            self.b = self.add_weight(
                name="bias",
                trainable=False,
                shape=(1, n_components),
                dtype=self.dtype,
                initializer=self._bias_init,
            )

    def _bias_init(self, shape: TensorType, dtype: Optional[DType] = None) -> TensorType:
        return tf.random.uniform(shape=shape, maxval=2.0 * np.pi, dtype=dtype)

    def compute_output_dim(self, input_shape: ShapeType) -> int:
        dim = self.n_components
        if self.is_batched and not self.is_multioutput:
            # For combination kernels, the number of features is multiplied by the number of
            # sub-kernels.
            dim *= self.batch_size
        return dim

    def _compute_bases(self, inputs: TensorType, batch: Optional[int]) -> tf.Tensor:
        """
        Compute basis functions.

        :return: A tensor with the shape ``[N, M]`` or ``[P, N, M]``.
        """
        return _bases_cosine(
            inputs,
            self.W if batch is None else self.W[batch],
            self.b if batch is None else self.b[batch],
        )

    def _compute_constant(self) -> tf.Tensor:
        """
        Compute normalizing constant for basis functions.

        :return: A tensor with the shape ``[]`` (i.e. a scalar).
        """
        if self.is_batched:
            constants = [
                self.rff_constant(k.variance, output_dim=self.n_components)
                for k in self.sub_kernels
            ]
            return tf.stack(constants, 0)[:, None, None]  # [1, 1, 1] or [P, 1, 1]
        else:
            return self.rff_constant(self.kernel.variance, output_dim=self.n_components)
