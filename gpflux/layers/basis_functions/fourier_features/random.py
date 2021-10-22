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
""" A kernel's features and coefficients using Random Fourier Features (RFF). """

from typing import Mapping, Optional

import numpy as np
import tensorflow as tf

import gpflow
from gpflow.base import DType, TensorType

from gpflux.layers.basis_functions.fourier_features.base import FourierFeaturesBase
from gpflux.layers.basis_functions.fourier_features.utils import (
    ORF_SUPPORTED_KERNELS,
    RFF_SUPPORTED_KERNELS,
    _bases_concat,
    _bases_cosine,
    _ceil_divide,
    _matern_number,
    _sample_chi,
    _sample_students_t,
)
from gpflux.types import ShapeType


class RandomFourierFeaturesBase(FourierFeaturesBase):
    def __init__(self, kernel: gpflow.kernels.Kernel, n_components: int, **kwargs: Mapping):
        assert isinstance(kernel, RFF_SUPPORTED_KERNELS), "Unsupported Kernel"
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

    def _weights_build(self, input_dim: int, n_components: int) -> None:
        shape = (n_components, input_dim)
        self.W = self.add_weight(
            name="weights",
            trainable=False,
            shape=shape,
            dtype=self.dtype,
            initializer=self._weights_init,
        )

    def _weights_init(self, shape: TensorType, dtype: Optional[DType] = None) -> TensorType:
        if isinstance(self.kernel, gpflow.kernels.SquaredExponential):
            return tf.random.normal(shape, dtype=dtype)
        else:
            p = _matern_number(self.kernel)
            nu = 2.0 * p + 1.0  # degrees of freedom
            return _sample_students_t(nu, shape, dtype)

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

    def _compute_output_dim(self, input_shape: ShapeType) -> int:
        return 2 * self.n_components

    def _compute_bases(self, inputs: TensorType) -> tf.Tensor:
        """
        Compute basis functions.

        :return: A tensor with the shape ``[N, 2M]``.
        """
        return _bases_concat(inputs, self.W)

    def _compute_constant(self) -> tf.Tensor:
        """
        Compute normalizing constant for basis functions.

        :return: A tensor with the shape ``[]`` (i.e. a scalar).
        """
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
        shape = (1, n_components)
        self.b = self.add_weight(
            name="bias",
            trainable=False,
            shape=shape,
            dtype=self.dtype,
            initializer=self._bias_init,
        )

    def _bias_init(self, shape: TensorType, dtype: Optional[DType] = None) -> TensorType:
        return tf.random.uniform(shape=shape, maxval=2.0 * np.pi, dtype=dtype)

    def _compute_output_dim(self, input_shape: ShapeType) -> int:
        return self.n_components

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
        return self.rff_constant(self.kernel.variance, output_dim=self.n_components)


class OrthogonalRandomFeatures(RandomFourierFeatures):
    r"""
    Orthogonal random Fourier features (ORF) :cite:p:`yu2016orthogonal` for more
    efficient and accurate kernel approximations than :class:`RandomFourierFeatures`.
    """

    def __init__(self, kernel: gpflow.kernels.Kernel, n_components: int, **kwargs: Mapping):
        assert isinstance(kernel, ORF_SUPPORTED_KERNELS), "Unsupported Kernel"
        super(OrthogonalRandomFeatures, self).__init__(kernel, n_components, **kwargs)

    def _weights_init(self, shape: TensorType, dtype: Optional[DType] = None) -> TensorType:
        n_components, input_dim = shape  # M, D
        n_reps = _ceil_divide(n_components, input_dim)  # K, smallest integer s.t. K*D >= M

        W = tf.random.normal(shape=(n_reps, input_dim, input_dim), dtype=dtype)
        Q, _ = tf.linalg.qr(W)  # throw away R; shape [K, D, D]

        s = _sample_chi(nu=input_dim, shape=(n_reps, input_dim), dtype=dtype)  # shape [K, D]
        U = tf.expand_dims(s, axis=-1) * Q  # equiv: S @ Q where S = diag(s); shape [K, D, D]
        V = tf.reshape(U, shape=(-1, input_dim))  # shape [K*D, D]

        return V[: self.n_components]  # shape [M, D] (throw away K*D - M rows)
