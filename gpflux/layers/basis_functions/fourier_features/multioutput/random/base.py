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
from typing import Mapping, Optional, Tuple, Type

import numpy as np
import tensorflow as tf

import gpflow
from gpflow.base import DType, TensorType

from gpflux.layers.basis_functions.fourier_features.multioutput.base import (
    MultiOutputFourierFeaturesBase,
)
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


# TODO -- import this from the univariate folder
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


class MultiOutputRandomFourierFeaturesBase(MultiOutputFourierFeaturesBase):
    def __init__(
        self, kernel: gpflow.kernels.MultioutputKernel, n_components: int, **kwargs: Mapping
    ):

        if isinstance(kernel, gpflow.kernels.SeparateIndependent):
            for ker in kernel.kernels:
                assert isinstance(ker, RFF_SUPPORTED_KERNELS), "Unsupported Kernel"
        elif isinstance(kernel, gpflow.kernels.SharedIndependent):
            assert isinstance(kernel.kernel, RFF_SUPPORTED_KERNELS), "Unsupported Kernel"
        else:
            raise ValueError("kernel specified is not supported.")
        super(MultiOutputRandomFourierFeaturesBase, self).__init__(kernel, n_components, **kwargs)

    # NOTE -- same as univariate case
    def build(self, input_shape: ShapeType) -> None:
        """
        Creates the variables of the layer.
        See `tf.keras.layers.Layer.build()
        <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#build>`_.
        """
        input_dim = input_shape[-1]
        self._weights_build(input_dim, n_components=self.n_components)
        super(MultiOutputRandomFourierFeaturesBase, self).build(input_shape)

    def _weights_build(self, input_dim: int, n_components: int) -> None:

        shape = (self.kernel.num_latent_gps, n_components, input_dim)

        self.W = self.add_weight(
            name="weights",
            trainable=False,
            shape=shape,
            dtype=self.dtype,
            initializer=self._weights_init,
        )
        tf.ensure_shape(self.W, shape)

    def _weights_init(self, shape: TensorType, dtype: Optional[DType] = None) -> TensorType:

        if isinstance(self.kernel, gpflow.kernels.SeparateIndependent):

            list_inits = []
            for ker in self.kernel.kernels:
                if isinstance(ker, gpflow.kernels.SquaredExponential):
                    list_inits.append(tf.random.normal(shape[1:], dtype=dtype))
                else:
                    p = _matern_number(ker)
                    nu = 2.0 * p + 1.0  # degrees of freedom
                    list_inits.append(_sample_students_t(nu, shape[1:], dtype))
            return tf.stack(list_inits, axis=0)

        elif isinstance(self.kernel, gpflow.kernels.SharedIndependent):

            if isinstance(self.kernel.kernel, gpflow.kernels.SquaredExponential):
                return tf.random.normal(shape, dtype=dtype)
            else:
                p = _matern_number(self.kernel.kernel)
                nu = 2.0 * p + 1.0  # degrees of freedom
                return _sample_students_t(nu, shape, dtype)
        else:
            raise ValueError("kernel is not supported.")

    # NOTE -- same as univariate case
    @staticmethod
    def rff_constant(variance: TensorType, output_dim: int) -> tf.Tensor:
        """
        Normalizing constant for random Fourier features.
        """
        return tf.sqrt(tf.math.truediv(2.0 * variance, output_dim))


class MultiOutputRandomFourierFeatures(MultiOutputRandomFourierFeaturesBase):
    r"""
    #TODO -- update documentation to suit multioutput case
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

    # NOTE -- same as univariate case
    def _compute_output_dim(self, input_shape: ShapeType) -> int:
        return 2 * self.n_components

    def _compute_bases(self, inputs: TensorType) -> tf.Tensor:
        """
        Compute basis functions.

        :return: A tensor with the shape ``[P, N, 2M]``  .
        """
        return _bases_concat(inputs, self.W)

    def _compute_constant(self) -> tf.Tensor:
        """
        Compute normalizing constant for basis functions.

        :return: A tensor with the shape ``[]`` (i.e. a scalar).
        """

        if hasattr(self.kernel, "kernels"):
            _kernel_variance = tf.stack([ker.variance for ker in self.kernel.kernels], axis=0)
            tf.ensure_shape(
                _kernel_variance,
                [
                    self.kernel.num_latent_gps,
                ],
            )

        else:
            _kernel_variance = self.kernel.kernel.variance

        return self.rff_constant(_kernel_variance, output_dim=2 * self.n_components)


class MultiOutputRandomFourierFeaturesCosine(MultiOutputRandomFourierFeaturesBase):
    r"""
    #TODO -- update documentation to suit multioutput case
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
        super(MultiOutputRandomFourierFeaturesCosine, self).build(input_shape)

    def _bias_build(self, n_components: int) -> None:

        shape = (self.kernel.num_latent_gps, 1, n_components)

        self.b = self.add_weight(
            name="bias",
            trainable=False,
            shape=shape,
            dtype=self.dtype,
            initializer=self._bias_init,
        )

    # NOTE -- same as univariate case
    def _bias_init(self, shape: TensorType, dtype: Optional[DType] = None) -> TensorType:
        return tf.random.uniform(shape=shape, maxval=2.0 * np.pi, dtype=dtype)

    # NOTE -- same as univariate case
    def _compute_output_dim(self, input_shape: ShapeType) -> int:
        return self.n_components

    # NOTE -- same as univariate case
    def _compute_bases(self, inputs: TensorType) -> tf.Tensor:
        """
        Compute basis functions.

        :return: A tensor with the shape ``[N, M]``.
        """
        return _bases_cosine(inputs, self.W, self.b)

    def _compute_constant(self) -> tf.Tensor:
        """
        Compute normalizing constant for basis functions.

        :return: A tensor with the shape ``[]`` (i.e. a scalar).
        """

        if hasattr(self.kernel, "kernels"):
            _kernel_variance = tf.stack([ker.variance for ker in self.kernel.kernels], axis=0)
            tf.ensure_shape(
                _kernel_variance,
                [
                    self.kernel.num_latent_gps,
                ],
            )
        else:
            _kernel_variance = self.kernel.kernel.variance

        return self.rff_constant(_kernel_variance, output_dim=self.n_components)
