#
# Copyright (c) 2022 The GPflux Contributors.
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
"""This module declares configurations for building various types of architectures"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type

import tensorflow_probability as tfp

import gpflow.likelihoods
from gpflow.kernels import Stationary


@dataclass
class LikelihoodConfig(ABC):
    """Config for the model likelihood"""

    @abstractmethod
    def create(self) -> gpflow.likelihoods.Likelihood:
        """Create a likelihood instance with the parameters of the config"""


@dataclass
class GaussianLikelihoodConfig(LikelihoodConfig):
    """Config for a :class:`~gpflow.likelihoods.Gaussian` likelihood"""

    noise_variance: float
    """The variance of the likelihood"""

    def create(self) -> gpflow.likelihoods.Gaussian:
        return gpflow.likelihoods.Gaussian(variance=self.noise_variance)


@dataclass
class StudenttLikelihoodConfig(LikelihoodConfig):
    """Config for a :class:`~gpflow.likelihoods.Studentt` likelihood"""

    df: float
    """The number of degrees of freedom"""

    scale: float
    """The scale parameter"""

    def create(self) -> gpflow.likelihoods.StudentT:
        return gpflow.likelihoods.StudentT(df=self.df, scale=self.scale)


@dataclass
class HeteroSkedasticLikelihoodConfig(LikelihoodConfig):
    """Configuration for a :class:`~gpflow.likelihoods.HeteroskedasticTFPConditional`"""

    distribution_class: Type[tfp.distributions.Distribution] = tfp.distributions.Normal
    """The distribution class"""

    def create(self) -> gpflow.likelihoods.HeteroskedasticTFPConditional:
        return gpflow.likelihoods.HeteroskedasticTFPConditional(
            distribution_class=self.distribution_class
        )


@dataclass
class HyperParametersConfig(ABC):
    """Configuration of the hyperparameters of a model"""

    num_layers: int
    """The number of GP layers in the model, excluded the likelihood one"""

    kernel: Type[Stationary]
    """The (stationary) kernel to use in the layers"""

    likelihood: LikelihoodConfig
    """Configuration for the model likelihood"""

    inner_layer_qsqrt_factor: float
    """
    A multiplicative factor used to rescale the hidden layers'
    :attr:`~gpflux.layers.GPLayer.q_sqrt`. Typically this value is chosen to be small
    (e.g., 1e-5) to reduce noise at the start of training.
    """

    whiten: bool
    """
    Determines the parameterisation of the inducing variables.
    If `True`, :math:``p(u) = N(0, I)``, otherwise :math:``p(u) = N(0, K_{uu})``.
    .. seealso:: :attr:`gpflux.layers.GPLayer.whiten`
    """

    def __post_init__(self):
        assert self.num_layers > 0, "Cannot have non-positive number of layers"
        assert self.whiten, "Non-whitened case not yet supported"


@dataclass
class ModelHyperParametersConfig(HyperParametersConfig):
    """The configuration used to build a DeepGP model"""

    num_inducing: int
    """
    The number of inducing variables, *M*. The Deep GP uses the same number
    of inducing variables in each layer.
    """


@dataclass
class OrthogonalModelHyperparametersConfig(HyperParametersConfig):
    """The configuration used to build an OrthDeepGP model"""

    num_inducing_u: int
    """
    The number of inducing variables, *M*. The Deep GP uses the same number
    of inducing variables in each layer.
    """

    num_inducing_v: int
    """
    The number of inducing variables, *M*. The Deep GP uses the same number
    of inducing variables in each layer.
    """
