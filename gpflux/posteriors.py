import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Type, Union

import tensorflow as tf
import tensorflow_probability as tfp

from gpflow.base import MeanAndVariance, Module, Parameter, RegressionData, TensorType
from gpflow.conditionals.util import *
from gpflow.config import default_float, default_jitter
from gpflux.covariances import Kuf, Kuu, Cvf, Cvv
from gpflow.inducing_variables import (
    InducingPoints,
    InducingVariables,
    SharedIndependentInducingVariables,
)
from gpflux.conditionals.util import *

from gpflow import kernels

from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
import numpy as np

class AbstractPosterior(Module, ABC):
    def __init__(
        self,
        kernel: Kernel,
        X_data: Union[tf.Tensor, InducingVariables],
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
        self, Xnew: Union[TensorType,tfp.distributions.MultivariateNormalDiag], full_cov: bool = False, full_output_cov: bool = False
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
        self, Xnew: Union[TensorType,tfp.distributions.MultivariateNormalDiag], full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Computes predictive mean and (co)variance at Xnew, *excluding* mean_function
        Does not make use of caching
        """


class BasePosterior(AbstractPosterior):
    def __init__(
        self,
        kernel: Kernel,
        inducing_variable: InducingVariables,
        q_mu: tf.Tensor,
        q_sqrt: tf.Tensor,
        whiten: bool = True,
        mean_function: Optional[MeanFunction] = None,
    ):

        super().__init__(kernel, inducing_variable, mean_function=mean_function)
        self.whiten = whiten
        self.q_mu = q_mu
        self.q_sqrt = q_sqrt
        #self._set_qdist(q_mu, q_sqrt)

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


    #NOTE -- we actually don't use this function in the end
    def _get_Kff(self, Xnew: Union[TensorType,tfp.distributions.MultivariateNormalDiag], full_cov: bool) -> tf.Tensor:

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

#NOTE -- we don't actually make use of this
class IndependentPosteriorSingleOutput(IndependentPosterior):
    
    # could almost be the same as IndependentPosteriorMultiOutput ...
    def _conditional_fused(
        self, Xnew: TensorType, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        # same as IndependentPosteriorMultiOutput, Shared~/Shared~ branch, except for following
        # line:
        Knn = self.kernel(Xnew, full_cov=full_cov)

        Kmm = Kuu(self.inducing_variable_u, self.kernel, jitter=default_jitter())  # [M, M]
        Kmn = Kuf(self.inducing_variable_u, self.kernel, Xnew)  # [M, N]

        fmean, fvar = base_conditional(
            Kmn, Kmm, Knn, self.q_mu, full_cov=full_cov, q_sqrt=self.q_sqrt, white=self.whiten
        )  # [N, P],  [P, N, N] or [N, P]
        return self._post_process_mean_and_cov(fmean, fvar, full_cov, full_output_cov)

class IndependentPosteriorMultiOutput(IndependentPosterior):
    def _conditional_fused(
        self, Xnew: TensorType, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:


        #TODO -- could probably be done more neatly with

        if isinstance(self.X_data, SharedIndependentInducingVariables) and isinstance(
            self.kernel, kernels.SharedIndependent):
            # same as IndependentPosteriorSingleOutput except for following line

            Knn = self.kernel.kernel(Xnew, full_cov=full_cov)
            # we don't call self.kernel() directly as that would do unnecessary tiling

            Kmm = Kuu(self.X_data, self.kernel, jitter=default_jitter())  # [M, M]
            Kmn = Kuf(self.X_data, self.kernel, Xnew)  # [M, N]

            fmean, fvar = base_conditional(
                Kmn, Kmm, Knn, self.q_mu, full_cov=full_cov, q_sqrt=self.q_sqrt, white=self.whiten
            )  # [N, P],  [P, N, N] or [N, P]

        else:
            raise NotImplementedError

        return self._post_process_mean_and_cov(fmean, fvar, full_cov, full_output_cov)



from gpflow.utilities import Dispatcher


get_posterior_class = Dispatcher("get_posterior_class")
from gpflow.kernels import SharedIndependent, SeparateIndependent


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
    SharedIndependent,
    SharedIndependentInducingVariables,
    SharedIndependentInducingVariables,
)
def _get_posterior_independent_mo_orthogonal(
    kernel: Kernel, inducing_variable_u: InducingVariables, inducing_variable_v: InducingVariables 
) -> Type[BasePosterior]:
    # independent multi-output
    return IndependentOrthogonalPosteriorMultiOutput


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





class AbstractOrthogonalPosterior(Module, ABC):
    def __init__(
        self,
        kernel: Kernel,
        inducing_variable_u: Union[tf.Tensor, InducingVariables],
        inducing_variable_v: Union[tf.Tensor, InducingVariables],
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
        self.inducing_variable_u = inducing_variable_u
        self.inducing_variable_v = inducing_variable_v
        self.cache = cache
        self.mean_function = mean_function

    def _add_mean_function(self, Xnew: TensorType, mean: TensorType) -> tf.Tensor:
        if self.mean_function is None:
            return mean
        else:
            return mean + self.mean_function(Xnew)

    def fused_predict_f(
        self, Xnew: TensorType, full_cov: bool = False, full_output_cov: bool = False
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
        self, Xnew: TensorType, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Computes predictive mean and (co)variance at Xnew, *excluding* mean_function
        Does not make use of caching
        """


class BaseOrthogonalPosterior(AbstractOrthogonalPosterior):
    def __init__(
        self,
        kernel: Kernel,
        inducing_variable_u: InducingVariables,
        inducing_variable_v: InducingVariables,
        q_mu_u: tf.Tensor,
        q_mu_v: tf.Tensor,
        q_sqrt_u: tf.Tensor,
        q_sqrt_v: tf.Tensor,
        whiten: bool = True,
        mean_function: Optional[MeanFunction] = None,
    ):

        super().__init__(kernel, inducing_variable_u, inducing_variable_v, mean_function=mean_function)
        self.whiten = whiten
        self.q_mu_u = q_mu_u
        self.q_sqrt_u = q_sqrt_u
        self.q_mu_v = q_mu_v
        self.q_sqrt_v = q_sqrt_v
        #self._set_qdist(q_mu, q_sqrt)

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

class IndependentOrthogonalPosterior(BaseOrthogonalPosterior):
    
    def _post_process_mean_and_cov(
        self, mean: TensorType, cov: TensorType, full_cov: bool, full_output_cov: bool
    ) -> MeanAndVariance:
        return mean, expand_independent_outputs(cov, full_cov, full_output_cov)


    #NOTE -- we actually don't use this function in the end
    def _get_Kff(self, Xnew: TensorType, full_cov: bool) -> tf.Tensor:

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


    def _get_Cff(self, Xnew: TensorType, full_cov: bool) -> tf.Tensor:

        # TODO: this assumes that Xnew has shape [N, D] and no leading dims

        if isinstance(self.kernel, SeparateIndependent):
            
            # TODO -- need to finish this at one point
            # NOTE calling kernel(Xnew, full_cov=full_cov, full_output_cov=False) directly would
            # return
            # if full_cov: [P, N, N] -- this is what we want
            # else: [N, P] instead of [P, N] as we get from the explicit stack below
            #Kff = tf.stack([k(Xnew, full_cov=full_cov) for k in self.kernel.kernels], axis=0)

            raise NotImplementedError

        #elif isinstance(self.kernel, kernels.MultioutputKernel):
        else:
            # effectively, SharedIndependent path
            Kff = self.kernel.kernel(Xnew, full_cov=full_cov)
            # NOTE calling kernel(Xnew, full_cov=full_cov, full_output_cov=False) directly would
            # return
            # if full_cov: [P, N, N] instead of [N, N]
            # else: [N, P] instead of [N]

            Kmm = Kuu(self.inducing_variable_u, self.kernel, jitter=default_jitter())
            L_Kmm = tf.linalg.cholesky(Kmm)

            Kmf = Kuf(self.inducing_variable_u,self.kernel, Xnew)
            L_Kmm_inv_Kmf = tf.linalg.triangular_solve(L_Kmm, Kmf)

            # compute the covariance due to the conditioning
            if full_cov:
                Cff = Kff - tf.linalg.matmul(L_Kmm_inv_Kmf, L_Kmm_inv_Kmf, transpose_a=True)  # [..., N, N]
                #num_func = tf.shape(self.q_mu_u)[-1]
                #N = tf.shape(Kuf)[-1]
                #cov_shape = [num_func, N, N]
                #Cff = tf.broadcast_to(tf.expand_dims(Cff, -3), cov_shape)  # [..., R, N, N]
            else:
                Cff = Kff - tf.reduce_sum(tf.square(L_Kmm_inv_Kmf), -2)  # [..., N]
                #num_func = tf.shape(self.q_mu_u)[-1]
                #N = tf.shape(Kuf)[-1]
                #cov_shape = [num_func, N]  # [..., R, N]
                #Cff = tf.broadcast_to(tf.expand_dims(Cff, -2), cov_shape)  # [..., R, N]
        """
        else:
            # standard ("single-output") kernels
            Kuu = self.kernel(self.inducing_variable_u.Z, full_cov=True)
            L_Kuu = tf.linalg.cholesky(Kuu)

            #Kvv = self.kernel.kernel(self.inducing_variable_v.Z, full_cov=full_cov)
            Kuf = self.kernel(self.inducing_variable_u.Z, Xnew)

            L_Kuu_inv_Kuf = tf.linalg.triangular_solve(L_Kuu, Kuf)
            
            if full_cov:
                Cff = Kff - tf.linalg.matmul(
                    L_Kuu_inv_Kuf, L_Kuu_inv_Kuf, transpose_a=True)
            else:
                Cff = Kff - tf.reduce_sum(tf.square(L_Kuu_inv_Kuf), -2)  # [..., N]
                #NOTE -- I don't think this is necessary
                #Cff = Cff[:,tf.newaxis] # [..., N, 1]
        """

        return Cff, L_Kmm

#NOTE -- we don't actually make use of this
class IndependentOrthogonalPosteriorSingleOutput(IndependentOrthogonalPosterior):
    
    # could almost be the same as IndependentPosteriorMultiOutput ...
    def _conditional_fused(
        self, Xnew: TensorType, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        # same as IndependentPosteriorMultiOutput, Shared~/Shared~ branch, except for following
        # line:

        Knn = self._get_Kff(Xnew, full_cov=full_output_cov)
        Cnn = self._get_Cff(Xnew, full_cov=full_output_cov)

        Kmm = Kuu(self.inducing_variable_u, self.kernel, jitter=default_jitter())  # [M_u, M_u]
        Kmn = Kuf(self.inducing_variable_u, self.kernel, Xnew)  # [M_U, N]

        Cmm = Cvv(self.inducing_variable_u, self.inducing_variable_v, self.kernel, jitter=default_jitter())  # [M_v, M_v]
        Cmn = Cvf(self.inducing_variable_u, self.inducing_variable_v, self.kernel, Xnew)  # [M_v, N]

        fmean, fvar = base_orthogonal_conditional(
            Kmn, Kmm, Knn, Cmn, Cmm, Cnn,
            self.q_mu_u, self.q_mu_v, full_cov=full_cov, q_sqrt_u=self.q_sqrt_u, q_sqrt_v=self.q_sqrt_v , white=self.whiten
        )  # [N, P],  [P, N, N] or [N, P]
        return self._post_process_mean_and_cov(fmean, fvar, full_cov, full_output_cov)

class IndependentOrthogonalPosteriorMultiOutput(IndependentOrthogonalPosterior):
    
    def _conditional_fused(
        self, Xnew: TensorType, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:

        if isinstance(self.inducing_variable_u, SharedIndependentInducingVariables) and isinstance(
            self.kernel, SharedIndependent):
            # same as IndependentPosteriorSingleOutput except for following line

            Knn = self._get_Kff(Xnew, full_cov=full_output_cov)
            Cnn, L_Kuu = self._get_Cff(Xnew, full_cov=full_output_cov)

            Kmm = Kuu(self.inducing_variable_u, self.kernel, jitter=default_jitter())  # [M_u, M_u]
            Kmn = Kuf(self.inducing_variable_u, self.kernel, Xnew)  # [M_U, N]

            Cmm = Cvv(self.inducing_variable_u, self.inducing_variable_v, self.kernel, jitter=default_jitter(), L_Kuu = L_Kuu)  # [M_v, M_v]
            Cmn = Cvf(self.inducing_variable_u, self.inducing_variable_v, self.kernel, Xnew, L_Kuu = L_Kuu)  # [M_v, N]

            fmean, fvar = base_orthogonal_conditional(
                Kmn, Kmm, Knn, Cmn, Cmm, Cnn,
                self.q_mu_u, self.q_mu_v, full_cov=full_cov, 
                q_sqrt_u=self.q_sqrt_u, q_sqrt_v=self.q_sqrt_v , 
                white=self.whiten, Lm=L_Kuu
            )  # [N, P],  [P, N, N] or [N, P]

        else:
            raise NotImplementedError

        return self._post_process_mean_and_cov(fmean, fvar, full_cov, full_output_cov)





