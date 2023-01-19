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
from typing import Optional, Tuple, Type, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from gpflow import kernels
from gpflow.base import MeanAndVariance, Module, TensorType
from gpflow.conditionals.util import *
from gpflow.config import default_jitter
from gpflow.inducing_variables import InducingVariables, SharedIndependentInducingVariables
from gpflow.kernels import Kernel, SeparateIndependent, SharedIndependent
from gpflow.mean_functions import MeanFunction
from gpflow.utilities import Dispatcher

from gpflux.covariances.multioutput.kufs import Kuf
from gpflux.covariances.multioutput.kuus import Kuu
from gpflux.inducing_variables import (
    DistributionalInducingVariables,
    SharedIndependentDistributionalInducingVariables,
)
from gpflux.kernels import *

get_posterior_class = Dispatcher("get_posterior_class")


class AbstractPosterior(Module, ABC):
    def __init__(
        self,
        kernel: Kernel,
        X_data: Union[tf.Tensor, InducingVariables, DistributionalInducingVariables],
        cache: Optional[Tuple[tf.Tensor, ...]] = None,
        mean_function: Optional[MeanFunction] = None,
    ) -> None:
        """
        Users should use `create_posterior` to create instances of concrete
        subclasses of this AbstractPosterior class instead of calling this
        constructor directly. For `create_posterior` to be able to correctly
        instantiate subclasses, developers need to ensure their subclasses
        don't change the constructor signature.
        """
        super().__init__()

        self.kernel = kernel
        self.X_data = X_data
        self.cache = cache
        self.mean_function = mean_function

    def _add_mean_function(self, Xnew: TensorType, mean: TensorType) -> tf.Tensor:
        if self.mean_function is None:
            return mean
        else:
            return mean + self.mean_function(Xnew)

    def fused_predict_f(
        self,
        Xnew: Union[TensorType, tfp.distributions.MultivariateNormalDiag],
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> MeanAndVariance:
        """
        Computes predictive mean and (co)variance at Xnew, including mean_function
        Does not make use of caching
        """
        mean, cov = self._conditional_fused(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )
        return self._add_mean_function(Xnew, mean), cov

    @abstractmethod
    def _conditional_fused(
        self,
        Xnew: Union[TensorType, tfp.distributions.MultivariateNormalDiag],
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> MeanAndVariance:
        """
        Computes predictive mean and (co)variance at Xnew, *excluding* mean_function
        Does not make use of caching
        """


class BasePosterior(AbstractPosterior):
    def __init__(
        self,
        kernel: Kernel,
        inducing_variable: Union[InducingVariables, DistributionalInducingVariables],
        q_mu: tf.Tensor,
        q_sqrt: tf.Tensor,
        whiten: bool = True,
        mean_function: Optional[MeanFunction] = None,
    ):

        super().__init__(kernel, inducing_variable, mean_function=mean_function)
        self.whiten = whiten
        self.q_mu = q_mu
        self.q_sqrt = q_sqrt
        # self._set_qdist(q_mu, q_sqrt)

    """
    @property
    def q_mu(self) -> tf.Tensor:
        return self._q_dist.q_mu
    @property
    def q_sqrt(self) -> tf.Tensor:
        return self._q_dist.q_sqrt
    def _set_qdist(self, q_mu: TensorType, q_sqrt: TensorType) -> tf.Tensor:
        if q_sqrt is None:
            self._q_dist = _DeltaDist(q_mu)
        elif len(q_sqrt.shape) == 2:  # q_diag
            self._q_dist = _DiagNormal(q_mu, q_sqrt)
        else:
            self._q_dist = _MvNormal(q_mu, q_sqrt)
    """


class IndependentPosterior(BasePosterior):
    def _post_process_mean_and_cov(
        self, mean: TensorType, cov: TensorType, full_cov: bool, full_output_cov: bool
    ) -> MeanAndVariance:
        return mean, expand_independent_outputs(cov, full_cov, full_output_cov)

    # NOTE -- we actually don't use this function in the end
    def _get_Kff(
        self, Xnew: Union[TensorType, tfp.distributions.MultivariateNormalDiag], full_cov: bool
    ) -> tf.Tensor:

        # TODO: this assumes that Xnew has shape [N, D] and no leading dims

        if isinstance(self.kernel, kernels.SeparateIndependent):
            # NOTE calling kernel(Xnew, full_cov=full_cov, full_output_cov=False) directly would
            # return
            # if full_cov: [P, N, N] -- this is what we want
            # else: [N, P] instead of [P, N] as we get from the explicit stack below
            Kff = tf.stack([k(Xnew, full_cov=full_cov) for k in self.kernel.kernels], axis=0)
        elif isinstance(self.kernel, kernels.MultioutputKernel):
            # effectively, SharedIndependent path
            Kff = self.kernel.kernel(Xnew, full_cov=full_cov)
            # NOTE calling kernel(Xnew, full_cov=full_cov, full_output_cov=False) directly would
            # return
            # if full_cov: [P, N, N] instead of [N, N]
            # else: [N, P] instead of [N]
        else:
            # standard ("single-output") kernels
            Kff = self.kernel(Xnew, full_cov=full_cov)  # [N, N] if full_cov else [N]

        return Kff


# NOTE -- we don't actually make use of this
class IndependentPosteriorSingleOutput(IndependentPosterior):

    # could almost be the same as IndependentPosteriorMultiOutput ...
    def _conditional_fused(
        self,
        Xnew: Union[TensorType, tfp.distributions.MultivariateNormalDiag],
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> MeanAndVariance:
        # same as IndependentPosteriorMultiOutput, Shared~/Shared~ branch, except for following
        # line:
        Knn = self.kernel(Xnew, full_cov=full_cov)

        Kmm = Kuu(self.X_data, self.kernel, jitter=default_jitter())  # [M, M]
        Kmn = Kuf(self.X_data, self.kernel, Xnew)  # [M, N]

        fmean, fvar = base_conditional(
            Kmn, Kmm, Knn, self.q_mu, full_cov=full_cov, q_sqrt=self.q_sqrt, white=self.whiten
        )  # [N, P],  [P, N, N] or [N, P]
        return self._post_process_mean_and_cov(fmean, fvar, full_cov, full_output_cov)


class IndependentPosteriorMultiOutput(IndependentPosterior):
    def _conditional_fused(
        self,
        Xnew: Union[TensorType, tfp.distributions.MultivariateNormalDiag],
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> MeanAndVariance:

        # TODO -- could probably be done more neatly with

        if isinstance(self.X_data, SharedIndependentInducingVariables) and isinstance(
            self.kernel, kernels.SharedIndependent
        ):
            # same as IndependentPosteriorSingleOutput except for following line

            Knn = self.kernel.kernel(Xnew, full_cov=full_cov)
            # we don't call self.kernel() directly as that would do unnecessary tiling

            Kmm = Kuu(self.X_data, self.kernel, jitter=default_jitter())  # [M, M]
            Kmn = Kuf(self.X_data, self.kernel, Xnew)  # [M, N]

            fmean, fvar = base_conditional(
                Kmn, Kmm, Knn, self.q_mu, full_cov=full_cov, q_sqrt=self.q_sqrt, white=self.whiten
            )  # [N, P],  [P, N, N] or [N, P]
        elif isinstance(
            self.X_data, SharedIndependentDistributionalInducingVariables
        ) and isinstance(self.kernel, DistributionalSharedIndependent):
            # same as IndependentPosteriorSingleOutput except for following line

            # TODO -- this is expecting a positional argument X_moments
            Knn = self.kernel.kernel(Xnew, full_cov=full_cov)
            # we don't call self.kernel() directly as that would do unnecessary tiling

            lcl_seed = np.random.randint(1e5)
            tf.random.set_seed(lcl_seed)

            Kmm = Kuu(self.X_data, self.kernel, jitter=default_jitter(), seed=lcl_seed)  # [M, M]
            Kmn = Kuf(self.X_data, self.kernel, Xnew, seed=lcl_seed)  # [M, N]

            fmean, fvar = base_conditional(
                Kmn, Kmm, Knn, self.q_mu, full_cov=full_cov, q_sqrt=self.q_sqrt, white=self.whiten
            )  # [N, P],  [P, N, N] or [N, P]

        else:
            raise NotImplementedError

        return self._post_process_mean_and_cov(fmean, fvar, full_cov, full_output_cov)


"""
#NOTE -- I don't think we need this here
@get_posterior_class.register(kernels.Kernel, InducingVariables)
def _get_posterior_base_case(
    kernel: Kernel, inducing_variable: InducingVariables
) -> Type[BasePosterior]:
    # independent single output
    return IndependentPosteriorSingleOutput
"""


@get_posterior_class.register(
    (SharedIndependent, SeparateIndependent),
    SharedIndependentInducingVariables,
)
def _get_posterior_independent_mo(
    kernel: Kernel, inducing_variable: InducingVariables
) -> Type[BasePosterior]:
    # independent multi-output
    return IndependentPosteriorMultiOutput


@get_posterior_class.register(
    DistributionalSharedIndependent,
    SharedIndependentDistributionalInducingVariables,
)
def _get_posterior_independent_mo_distributional(
    kernel: Kernel, inducing_variable: DistributionalInducingVariables
) -> Type[BasePosterior]:
    # independent multi-output
    return IndependentPosteriorMultiOutput


"""
#NOTE -- I don't think we need this
def create_posterior(
    kernel: Kernel,
    inducing_variable: InducingVariables,
    q_mu: TensorType,
    q_sqrt: TensorType,
    whiten: bool,
    mean_function: Optional[MeanFunction] = None,
    precompute_cache: Union[PrecomputeCacheType, str, None] = PrecomputeCacheType.TENSOR,
) -> BasePosterior:
    posterior_class = get_posterior_class(kernel, inducing_variable)
    precompute_cache = _validate_precompute_cache_type(precompute_cache)
    return posterior_class(  # type: ignore
        kernel,
        inducing_variable,
        q_mu,
        q_sqrt,
        whiten,
        mean_function,
        precompute_cache=precompute_cache,
    )
"""
