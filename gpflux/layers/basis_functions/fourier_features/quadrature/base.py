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

from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from gpflow.base import TensorType

from gpflux.layers.basis_functions.fourier_features.base import FourierFeaturesBase
from gpflux.layers.basis_functions.fourier_features.utils import _bases_concat
from gpflux.types import ShapeType


class QuadratureFourierFeaturesBase(FourierFeaturesBase):
    def _compute_output_dim(self, input_shape: ShapeType) -> int:
        input_dim = input_shape[-1]
        return 2 * self.n_components ** input_dim

    def _compute_bases(self, inputs: TensorType) -> tf.Tensor:
        """
        Compute basis functions.

        :return: A tensor with the shape ``(N, 2L^D)``.
        """
        return _bases_concat(inputs, self.abscissa)

    def _compute_constant(self) -> tf.Tensor:
        """
        Compute normalizing constant for basis functions.

        :return: A tensor with the shape ``(2L^D,)``
        """
        return tf.tile(tf.sqrt(self.kernel.variance * self.factors), multiples=[2])


class Transform(ABC):
    r"""
    This class encapsulates functions :math:`g(x) = u` and :math:`h(x)` such that
    .. math::
        \int_{g(a)}^{g(b)} z(u) f(u) du
        = \int_a^b w(x) h(x) f(g(x)) dx
    for some integrand :math:`f(u)` and weight function :math:`z(u)`.
    """

    @abstractmethod
    def __call__(self, x: TensorType) -> tf.Tensor:
        pass

    @abstractmethod
    def multiplier(self, x: TensorType) -> tf.Tensor:
        pass


class TanTransform(Transform):
    r"""
    This class encapsulates functions :math:`g(x) = u` and :math:`h(x) = du/dx`
    such that
    .. math::
        \int_{-\infty}^{\infty} f(u) du
        = \int_{-1}^{1} f(g(x)) h(x) dx
    """
    CONST = 0.5 * np.pi

    def __call__(self, x: TensorType) -> tf.Tensor:
        return tf.tan(TanTransform.CONST * x)

    def multiplier(self, x: TensorType) -> tf.Tensor:
        return TanTransform.CONST / tf.square(tf.cos(TanTransform.CONST * x))


class NormalWeightTransform(Transform):
    r"""
    This class encapsulates functions :math:`g(x) = u` and :math:`h(x)` such that
    .. math::
        \int_{-\infty}^{\infty} \mathcal{N}(u|0,1) f(u) du
        = \int_{-infty}^{infty} e^{-x^2} f(g(x)) h(x) dx
    """

    def __call__(self, x: TensorType) -> tf.Tensor:
        return tf.sqrt(2.0) * x

    def multiplier(self, x: TensorType) -> tf.Tensor:
        return tf.rsqrt(np.pi)
