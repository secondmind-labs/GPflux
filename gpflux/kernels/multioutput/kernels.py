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
import abc
from typing import Optional, Sequence, Tuple, Union

import tensorflow as tf
import tensorflow_probability as tfp

from gpflow.base import TensorType

from gpflux.kernels.base_kernel import DistributionalCombination, DistributionalKernel


class DistributionalMultioutputKernel(DistributionalKernel):
    """
    Multi Output Kernel class.
    This kernel can represent correlation between outputs of different datapoints.
    Therefore, subclasses of Mok should implement `K` which returns:
    - [N, P, N, P] if full_output_cov = True
    - [P, N, N] if full_output_cov = False
    and `K_diag` returns:
    - [N, P, P] if full_output_cov = True
    - [N, P] if full_output_cov = False
    The `full_output_cov` argument holds whether the kernel should calculate
    the covariance between the outputs. In case there is no correlation but
    `full_output_cov` is set to True the covariance matrix will be filled with zeros
    until the appropriate size is reached.
    """

    @property
    @abc.abstractmethod
    def num_latent_gps(self) -> int:
        """The number of latent GPs in the multioutput kernel"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def latent_kernels(self) -> Tuple[DistributionalKernel, ...]:
        """The underlying kernels in the multioutput kernel"""
        raise NotImplementedError

    @abc.abstractmethod
    def K(
        self,
        X: Union[TensorType, tfp.distributions.MultivariateNormalDiag],
        X2: Optional[Union[TensorType, tfp.distributions.MultivariateNormalDiag]] = None,
        full_output_cov: bool = True,
    ) -> tf.Tensor:
        """
        Returns the correlation of f(X) and f(X2), where f(.) can be multi-dimensional.
        :param X: data matrix, [N1, D]
        :param X2:f"dist_gp_{i_layer}" data matrix, [N2, D]
        :param full_output_cov: calculate correlation between outputs.
        :return: cov[f(X), f(X2)] with shape
        - [N1, P, N2, P] if `full_output_cov` = True
        - [P, N1, N2] if `full_output_cov` = False
        """
        raise NotImplementedError

    @abc.abstractmethod
    def K_diag(self, X: TensorType, full_output_cov: bool = True) -> tf.Tensor:
        """
        Returns the correlation of f(X) and f(X), where f(.) can be multi-dimensional.
        :param X: data matrix, [N, D]
        :param full_output_cov: calculate correlation between outputs.
        :return: var[f(X)] with shape
        - [N, P, N, P] if `full_output_cov` = True
        - [N, P] if `full_output_cov` = False
        """
        raise NotImplementedError

    def __call__(
        self,
        X: Union[TensorType, tfp.distributions.MultivariateNormalDiag],
        X2: Optional[Union[TensorType, tfp.distributions.MultivariateNormalDiag]] = None,
        *,
        full_cov: bool = False,
        full_output_cov: bool = True,
        presliced: bool = False,
    ) -> tf.Tensor:

        """
        if not presliced:
            # Warning - only works for SquaredExponential kernel in Euclidean case
            X, X2 = self.slice(X, X2)
        """
        if not full_cov and X2 is not None:
            raise ValueError(
                "Ambiguous inputs: passing in `X2` is not compatible with `full_cov=False`."
            )
        if not full_cov:
            return self.K_diag(X, full_output_cov=full_output_cov)
        return self.K(X, X2, full_output_cov=full_output_cov)


class DistributionalSharedIndependent(DistributionalMultioutputKernel):
    """
    - Shared: we use the same kernel for each latent GP
    - Independent: Latents are uncorrelated a priori.
    Note: this class is created only for testing and comparison purposes.
    Use `gpflow.kernels` instead for more efficient code.
    """

    def __init__(
        self, kernel: DistributionalKernel, output_dim: int, *, name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)
        self.kernel = kernel
        self.output_dim = output_dim

    @property
    def num_latent_gps(self) -> int:
        # In this case number of latent GPs (L) == output_dim (P)
        return self.output_dim

    @property
    def latent_kernels(self) -> Tuple[DistributionalKernel, ...]:
        """The underlying kernels in the multioutput kernel"""
        return (self.kernel,)

    def K(
        self,
        X: Union[TensorType, tfp.distributions.MultivariateNormalDiag],
        X2: Optional[Union[TensorType, tfp.distributions.MultivariateNormalDiag]] = None,
        full_output_cov: bool = True,
    ) -> tf.Tensor:

        K = self.kernel.K(X, X2)  # [N, N2]
        if full_output_cov:
            Ks = tf.tile(K[..., None], [1, 1, self.output_dim])  # [N, N2, P]
            return tf.transpose(tf.linalg.diag(Ks), [0, 2, 1, 3])  # [N, P, N2, P]
        else:
            return tf.tile(K[None, ...], [self.output_dim, 1, 1])  # [P, N, N2]

    def K_diag(self, X: TensorType, full_output_cov: bool = True) -> tf.Tensor:
        K = self.kernel.K_diag(X)  # N
        Ks = tf.tile(K[:, None], [1, self.output_dim])  # [N, P]
        return tf.linalg.diag(Ks) if full_output_cov else Ks  # [N, P, P] or [N, P]


class DistributionalSeparateIndependent(DistributionalMultioutputKernel, DistributionalCombination):
    """
    - Separate: we use different kernel for each output latent
    - Independent: Latents are uncorrelated a priori.
    """

    def __init__(self, kernels: Sequence[DistributionalKernel], name: Optional[str] = None) -> None:
        super().__init__(kernels=kernels, name=name)

    @property
    def num_latent_gps(self) -> int:
        return len(self.kernels)

    @property
    def latent_kernels(self) -> Tuple[DistributionalKernel, ...]:
        """The underlying kernels in the multioutput kernel"""
        return tuple(self.kernels)

    def K(
        self,
        X: Union[TensorType, tfp.distributions.MultivariateNormalDiag],
        X2: Optional[Union[TensorType, tfp.distributions.MultivariateNormalDiag]] = None,
        full_output_cov: bool = True,
    ) -> tf.Tensor:
        if full_output_cov:
            Kxxs = tf.stack([k.K(X, X2) for k in self.kernels], axis=2)  # [N, N2, P]
            return tf.transpose(tf.linalg.diag(Kxxs), [0, 2, 1, 3])  # [N, P, N2, P]
        else:
            return tf.stack([k.K(X, X2) for k in self.kernels], axis=0)  # [P, N, N2]

    def K_diag(self, X: TensorType, full_output_cov: bool = False) -> tf.Tensor:
        stacked = tf.stack([k.K_diag(X) for k in self.kernels], axis=1)  # [N, P]
        return tf.linalg.diag(stacked) if full_output_cov else stacked  # [N, P, P]  or  [N, P]
