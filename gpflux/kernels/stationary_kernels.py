# Copyright 2017-2020 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from gpflow.base import Parameter, TensorType
from gpflow.utilities import positive

from gpflux.kernels.base_kernel import DistributionalKernel
from gpflux.utils.ops import square_distance, wasserstein_2_distance


class Stationary(DistributionalKernel):
    """
    Base class for kernels that are stationary, that is, they only depend on
        d = x - x'
    This class handles 'ard' behaviour, which stands for 'Automatic Relevance
    Determination'. This means that the kernel has one lengthscale per
    dimension, otherwise the kernel is isotropic (has a single lengthscale).
    """

    def __init__(
        self, variance: TensorType = 1.0, lengthscales: TensorType = 1.0, **kwargs: Any
    ) -> None:
        """
        :param variance: the (initial) value for the variance parameter.
        :param lengthscales: the (initial) value for the lengthscale
            parameter(s), to induce ARD behaviour this must be initialised as
            an array the same length as the the number of active dimensions
            e.g. [1., 1., 1.]. If only a single value is passed, this value
            is used as the lengthscale of each dimension.
        :param kwargs: accepts `name` and `active_dims`, which is a list or
            slice of indices which controls which columns of X are used (by
            default, all columns are used).
        """
        for kwarg in kwargs:
            if kwarg not in {"name", "active_dims"}:
                raise TypeError(f"Unknown keyword argument: {kwarg}")

        super().__init__(**kwargs)
        self.variance = Parameter(
            variance,
            transform=positive(),
            name=f"{self.name}_kernel_variance" if self.name else "kernel_variance",
        )
        self.lengthscales = Parameter(
            lengthscales,
            transform=positive(),
            name=f"{self.name}_kernel_lengthscales" if self.name else "kernel_lengthscales",
        )

    def scale(self, X: TensorType) -> TensorType:
        X_scaled = X / self.lengthscales if X is not None else X
        return X_scaled

    def K_diag(self, X: TensorType) -> tf.Tensor:
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))


class IsotropicStationary(Stationary):
    """
    Base class for isotropic stationary kernels, i.e. kernels that only
    depend on
        r = ‖x - x'‖
    Derived classes should implement one of:
        K_r2(self, r2): Returns the kernel evaluated on r² (r2), which is the
        squared scaled Euclidean distance Should operate element-wise on r2.
        K_r(self, r): Returns the kernel evaluated on r, which is the scaled
        Euclidean distance. Should operate element-wise on r.
    """

    # Only used for SquaredExponential
    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:

        r2 = self.scaled_squared_euclid_dist(X, X2)
        return self.K_r2(r2)

    def K_r2(self, r2: TensorType) -> tf.Tensor:
        if hasattr(self, "K_r"):
            # Clipping around the (single) float precision which is ~1e-45.
            r = tf.sqrt(tf.maximum(r2, 1e-36))
            return self.K_r(r)  # pylint: disable=no-member
        raise NotImplementedError

    def scaled_squared_euclid_dist(
        self, X: TensorType, X2: Optional[TensorType] = None
    ) -> tf.Tensor:
        """
        Returns ‖(X - X2ᵀ) / ℓ‖², i.e. the squared L₂-norm.
        """
        return square_distance(self.scale(X), self.scale(X2))


class SquaredExponential(IsotropicStationary):
    """
    The radial basis function (RBF) or squared exponential kernel. The kernel equation is
        k(r) = σ² exp{-½ r²}
    where:
    r   is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ.
    σ²  is the variance parameter
    Functions drawn from a GP with this kernel are infinitely differentiable!
    """

    def K_r2(self, r2: TensorType) -> tf.Tensor:
        return self.variance * tf.exp(-0.5 * r2)


class Matern12(IsotropicStationary):
    """
    The Matern 1/2 kernel. Functions drawn from a GP with this kernel are not
    differentiable anywhere. The kernel equation is
    k(r) = σ² exp{-r}
    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ.
    σ² is the variance parameter
    """

    def K_r(self, r: TensorType) -> tf.Tensor:
        return self.variance * tf.exp(-r)


class Matern32(IsotropicStationary):
    """
    The Matern 3/2 kernel. Functions drawn from a GP with this kernel are once
    differentiable. The kernel equation is
    k(r) = σ² (1 + √3r) exp{-√3 r}
    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ,
    σ² is the variance parameter.
    """

    def K_r(self, r: TensorType) -> tf.Tensor:
        sqrt3 = np.sqrt(3.0)
        return self.variance * (1.0 + sqrt3 * r) * tf.exp(-sqrt3 * r)


class Matern52(IsotropicStationary):
    """
    The Matern 5/2 kernel. Functions drawn from a GP with this kernel are twice
    differentiable. The kernel equation is
    k(r) = σ² (1 + √5r + 5/3r²) exp{-√5 r}
    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ,
    σ² is the variance parameter.
    """

    def K_r(self, r: TensorType) -> tf.Tensor:
        sqrt5 = np.sqrt(5.0)
        return self.variance * (1.0 + sqrt5 * r + 5.0 / 3.0 * tf.square(r)) * tf.exp(-sqrt5 * r)


class Hybrid(IsotropicStationary):

    """
    The radial basis function (RBF) or squared exponential kernel multiplied by the Wasserstein-2 distance based kernel. The kernel equation is
        k(r) = σ² exp{-½ r²} exp{ -0.5 * W_{2}^{2}\left(\mu_{1}, \mu_{2} \right) }
    should also work with Matern kernels
    where:
    r   is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ.
    σ²  is the variance parameter
    Functions drawn from a GP with this kernel are infinitely differentiable!
    TODO -- this remains to be seen as this also implies a discussion around
    Wasserstein Gradient Flows
    """

    def __init__(self, baseline_kernel, **kwargs: Any) -> None:
        """
        :param baseline_kernel: string specifying the underlying kernel to be used withing the Hybrid kernel framework
        :param kwargs: accepts `name` and `active_dims`, which is a list or
            slice of indices which controls which columns of X are used (by
            default, all columns are used).
        """
        super().__init__(**kwargs)

        self.baseline_kernel = baseline_kernel

    # Overides default K from IsotropicStationary base class
    def K(
        self,
        X: tfp.distributions.MultivariateNormalDiag,
        X2: Optional[tfp.distributions.MultivariateNormalDiag] = None,
        *,
        seed: Optional[Any] = None,
    ) -> tf.Tensor:

        w2 = self.scaled_squared_Wasserstein_2_dist(X, X2)

        tf.random.set_seed(seed)
        X_sampled = X.sample(seed=seed)

        if X2 is not None:
            assert isinstance(X2, tfp.distributions.MultivariateNormalDiag)
            X2_sampled = X2.sample()

        else:
            X2_sampled = None

        r2 = self.scaled_squared_euclid_dist(X_sampled, X2_sampled)

        if self.baseline_kernel == "squared_exponential":

            return self.K_r2(r2, w2)

        elif self.baseline_kernel == "matern12":

            r = tf.sqrt(tf.maximum(r2, 1e-36))
            w = tf.sqrt(tf.maximum(w2, 1e-36))

            return self.variance * tf.exp(-r) * tf.exp(-w)

        elif self.baseline_kernel == "matern32":

            r = tf.sqrt(tf.maximum(r2, 1e-36))
            w = tf.sqrt(tf.maximum(w2, 1e-36))

            sqrt3 = np.sqrt(3.0)
            return (
                self.variance
                * (1.0 + sqrt3 * r)
                * tf.exp(-sqrt3 * r)
                * (1.0 + sqrt3 * w)
                * tf.exp(-sqrt3 * w)
            )

        elif self.baseline_kernel == "matern52":

            r = tf.sqrt(tf.maximum(r2, 1e-36))
            w = tf.sqrt(tf.maximum(w2, 1e-36))

            sqrt5 = np.sqrt(5.0)
            return (
                self.variance
                * (1.0 + sqrt5 * r + 5.0 / 3.0 * tf.square(r))
                * tf.exp(-sqrt5 * r)
                * (1.0 + sqrt5 * w + 5.0 / 3.0 * tf.square(w))
                * tf.exp(-sqrt5 * w)
            )

    # Overides default K_diag from Stationary base class
    def K_diag(self, X: tfp.distributions.MultivariateNormalDiag) -> tf.Tensor:

        X_sampled = X.sample()
        return tf.fill(tf.shape(X_sampled)[:-1], tf.squeeze(self.variance))

    def K_r2(self, r2: TensorType, w2: TensorType) -> tf.Tensor:

        # r2 -- is the squared euclidean distance
        # w2 - is the squared Wasserstein-2 distance

        return self.variance * tf.exp(-0.5 * r2) * tf.exp(-0.5 * w2)

    def scaled_squared_Wasserstein_2_dist(
        self,
        mu1: tfp.distributions.MultivariateNormalDiag,
        mu2: Optional[tfp.distributions.MultivariateNormalDiag] = None,
    ) -> tf.Tensor:
        """
        Scales the raw Wasserstein-2 distance which is computed per input dimension
        """
        w2 = wasserstein_2_distance(mu1, mu2)

        return tf.reduce_sum(self.scale(w2), axis=-1, keepdims=False)
