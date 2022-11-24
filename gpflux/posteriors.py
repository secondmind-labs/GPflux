import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Type, Union

import numpy as np
import tensorflow as tf
from check_shapes import (
    ErrorContext,
    Shape,
    check_shapes,
    get_shape,
    inherit_check_shapes,
    register_get_shape,
)

from gpflow import kernels
from gpflow.base import MeanAndVariance, Module, Parameter, RegressionData, TensorType
from gpflow.conditionals.util import *
from gpflow.config import default_float, default_jitter
from gpflow.covariances import Kuf, Kuu
from gpflow.inducing_variables import (
    InducingPoints,
    InducingVariables,
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
)
from gpflow.kernels import Kernel, SeparateIndependent, SharedIndependent
from gpflow.mean_functions import MeanFunction
from gpflow.posteriors import (
    AbstractPosterior,
    BasePosterior,
    _DeltaDist,
    _DiagNormal,
    _MvNormal,
    get_posterior_class,
)

from gpflux.conditionals.util import *
from gpflux.covariances import Cvf, Cvv

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


class AbstractOrthogonalPosterior(AbstractPosterior):
    def __init__(
        self,
        kernel: Kernel,
        inducing_variable_u: Union[tf.Tensor, InducingVariables],
        inducing_variable_v: Union[tf.Tensor, InducingVariables],
        cache: Optional[Tuple[tf.Tensor, ...]] = None,
        mean_function: Optional[MeanFunction] = None,
    ) -> None:
        """
        TODO -- add documentation here
        """
        super().__init__(kernel, inducing_variable_u, mean_function=mean_function)

        self.kernel = kernel
        self.inducing_variable_u = inducing_variable_u
        self.inducing_variable_v = inducing_variable_v
        self.cache = cache
        self.mean_function = mean_function


class BaseOrthogonalPosterior(AbstractOrthogonalPosterior):
    # TODO -- introduce a suitable check_shapes here
    # @check_shapes(
    #    "inducing_variable_u: [M, D, broadcast P]",
    #    "q_mu_u: [N, P]",
    #    "q_sqrt_U: [N_P_or_P_N_N...]",
    # )
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

        super().__init__(
            kernel, inducing_variable_u, inducing_variable_v, mean_function=mean_function
        )
        self.whiten = whiten
        self._set_qdist(q_mu_u, q_sqrt_u, q_mu_v, q_sqrt_v)

    @property
    def q_mu_u(self) -> tf.Tensor:
        return self._q_dist_u.q_mu

    @property
    def q_mu_v(self) -> tf.Tensor:
        return self._q_dist_v.q_mu

    @property
    def q_sqrt_u(self) -> tf.Tensor:
        return self._q_dist_u.q_sqrt

    @property
    def q_sqrt_v(self) -> tf.Tensor:
        return self._q_dist_v.q_sqrt

    def _set_qdist(
        self, q_mu_u: TensorType, q_sqrt_u: TensorType, q_mu_v: TensorType, q_sqrt_v: TensorType
    ) -> tf.Tensor:

        if q_sqrt_u is None:
            self._q_dist_u = _DeltaDist(q_mu_u)
        elif len(q_sqrt_u.shape) == 2:  # q_diag
            self._q_dist_u = _DiagNormal(q_mu_u, q_sqrt_u)
        else:
            self._q_dist_u = _MvNormal(q_mu_u, q_sqrt_u)

        if q_sqrt_v is None:
            self._q_dist_v = _DeltaDist(q_mu_v)
        elif len(q_sqrt_v.shape) == 2:  # q_diag
            self._q_dist_v = _DiagNormal(q_mu_v, q_sqrt_v)
        else:
            self._q_dist_v = _MvNormal(q_mu_v, q_sqrt_v)

    def _precompute(self):

        """
        #TODO -- needs to be implemented
        """

        pass


class IndependentOrthogonalPosterior(BaseOrthogonalPosterior):
    @check_shapes(
        "mean: [batch..., N, P]",
        "cov: [batch..., P, N, N] if full_cov",
        "cov: [batch..., N, P] if not full_cov",
        "return[0]: [batch..., N, P]",
        "return[1]: [batch..., N, P, N, P] if full_cov and full_output_cov",
        "return[1]: [batch..., N, P, P] if (not full_cov) and full_output_cov",
        "return[1]: [batch..., P, N, N] if full_cov and (not full_output_cov)",
        "return[1]: [batch..., N, P] if (not full_cov) and (not full_output_cov)",
    )
    def _post_process_mean_and_cov(
        self, mean: TensorType, cov: TensorType, full_cov: bool, full_output_cov: bool
    ) -> MeanAndVariance:
        return mean, expand_independent_outputs(cov, full_cov, full_output_cov)

    @check_shapes(
        "Xnew: [N, D]",
        "return: [broadcast P, N, N] if full_cov",
        "return: [broadcast P, N] if (not full_cov)",
    )
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

    @check_shapes(
        "Xnew: [N, D]",
        "return: [broadcast P, N, N] if full_cov",
        "return: [broadcast P, N] if (not full_cov)",
    )
    def _get_Cff(self, Xnew: TensorType, full_cov: bool) -> tf.Tensor:

        # TODO: this assumes that Xnew has shape [N, D] and no leading dims

        if isinstance(self.kernel, SeparateIndependent):

            # TODO -- need to finish this at one point
            # NOTE calling kernel(Xnew, full_cov=full_cov, full_output_cov=False) directly would
            # return
            # if full_cov: [P, N, N] -- this is what we want
            # else: [N, P] instead of [P, N] as we get from the explicit stack below
            # Kff = tf.stack([k(Xnew, full_cov=full_cov) for k in self.kernel.kernels], axis=0)

            raise NotImplementedError

        # elif isinstance(self.kernel, kernels.MultioutputKernel):
        else:
            # effectively, SharedIndependent path
            Kff = self.kernel.kernel(Xnew, full_cov=full_cov)
            # NOTE calling kernel(Xnew, full_cov=full_cov, full_output_cov=False) directly would
            # return
            # if full_cov: [P, N, N] instead of [N, N]
            # else: [N, P] instead of [N]

            Kmm = Kuu(self.inducing_variable_u, self.kernel, jitter=default_jitter())
            L_Kmm = tf.linalg.cholesky(Kmm)

            Kmf = Kuf(self.inducing_variable_u, self.kernel, Xnew)
            L_Kmm_inv_Kmf = tf.linalg.triangular_solve(L_Kmm, Kmf)

            # compute the covariance due to the conditioning
            if full_cov:
                # TODO -- need to add broadcasting capability
                Cff = Kff - tf.linalg.matmul(
                    L_Kmm_inv_Kmf, L_Kmm_inv_Kmf, transpose_a=True
                )  # [..., N, N]
                # num_func = tf.shape(self.q_mu_u)[-1]
                # N = tf.shape(Kuf)[-1]
                # cov_shape = [num_func, N, N]
                # Cff = tf.broadcast_to(tf.expand_dims(Cff, -3), cov_shape)  # [..., R, N, N]
            else:
                # TODO -- need to add broadcasting capability
                Cff = Kff - tf.reduce_sum(tf.square(L_Kmm_inv_Kmf), -2)  # [..., N]
                # num_func = tf.shape(self.q_mu_u)[-1]
                # N = tf.shape(Kuf)[-1]
                # cov_shape = [num_func, N]  # [..., R, N]
                # Cff = tf.broadcast_to(tf.expand_dims(Cff, -2), cov_shape)  # [..., R, N]
        """
        #TODO -- this needs to be implememented
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

    def _conditional_with_precompute(
        self,
        cache: Tuple[tf.Tensor, ...],
        Xnew: TensorType,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ):

        """
        #TODO -- need to implement this
        """

        pass


class IndependentOrthogonalPosteriorSingleOutput(IndependentOrthogonalPosterior):

    # could almost be the same as IndependentPosteriorMultiOutput ...
    # TODO -- @inherit_check_shapes results in an error atm
    # @inherit_check_shapes
    def _conditional_fused(
        self, Xnew: TensorType, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        # same as IndependentPosteriorMultiOutput, Shared~/Shared~ branch, except for following
        # line:

        Knn = self._get_Kff(Xnew, full_cov=full_output_cov)
        Cnn = self._get_Cff(Xnew, full_cov=full_output_cov)

        Kmm = Kuu(self.inducing_variable_u, self.kernel, jitter=default_jitter())  # [M_u, M_u]
        Kmn = Kuf(self.inducing_variable_u, self.kernel, Xnew)  # [M_U, N]

        Cmm = Cvv(
            self.inducing_variable_u, self.inducing_variable_v, self.kernel, jitter=default_jitter()
        )  # [M_v, M_v]
        Cmn = Cvf(self.inducing_variable_u, self.inducing_variable_v, self.kernel, Xnew)  # [M_v, N]

        fmean, fvar = base_orthogonal_conditional(
            Kmn,
            Kmm,
            Knn,
            Cmn,
            Cmm,
            Cnn,
            self.q_mu_u,
            self.q_mu_v,
            full_cov=full_cov,
            q_sqrt_u=self.q_sqrt_u,
            q_sqrt_v=self.q_sqrt_v,
            white=self.whiten,
        )  # [N, P],  [P, N, N] or [N, P]
        return self._post_process_mean_and_cov(fmean, fvar, full_cov, full_output_cov)


class IndependentOrthogonalPosteriorMultiOutput(IndependentOrthogonalPosterior):
    # TODO -- @inherit_check_shapes results in an error atm
    # @inherit_check_shapes
    def _conditional_fused(
        self, Xnew: TensorType, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:

        if isinstance(self.inducing_variable_u, SharedIndependentInducingVariables) and isinstance(
            self.kernel, SharedIndependent
        ):
            # same as IndependentPosteriorSingleOutput except for following line

            Knn = self._get_Kff(Xnew, full_cov=full_output_cov)
            Cnn, L_Kuu = self._get_Cff(Xnew, full_cov=full_output_cov)

            Kmm = Kuu(self.inducing_variable_u, self.kernel, jitter=default_jitter())  # [M_u, M_u]
            Kmn = Kuf(self.inducing_variable_u, self.kernel, Xnew)  # [M_U, N]

            Cmm = Cvv(
                self.inducing_variable_u,
                self.inducing_variable_v,
                self.kernel,
                jitter=default_jitter(),
                L_Kuu=L_Kuu,
            )  # [M_v, M_v]
            Cmn = Cvf(
                self.inducing_variable_u, self.inducing_variable_v, self.kernel, Xnew, L_Kuu=L_Kuu
            )  # [M_v, N]

            fmean, fvar = base_orthogonal_conditional(
                Kmn,
                Kmm,
                Knn,
                Cmn,
                Cmm,
                Cnn,
                self.q_mu_u,
                self.q_mu_v,
                full_cov=full_cov,
                q_sqrt_u=self.q_sqrt_u,
                q_sqrt_v=self.q_sqrt_v,
                white=self.whiten,
                Lm=L_Kuu,
            )  # [N, P],  [P, N, N] or [N, P]

        else:
            # TODO -- this needs to be implemented
            raise NotImplementedError

        return self._post_process_mean_and_cov(fmean, fvar, full_cov, full_output_cov)


@get_posterior_class.register(kernels.Kernel, InducingVariables, InducingVariables)
def _get_posterior_base_case(
    kernel: Kernel, inducing_variable_u: InducingVariables, inducing_variable_v: InducingVariables
) -> Type[BasePosterior]:
    # independent single output
    return IndependentOrthogonalPosteriorSingleOutput


@get_posterior_class.register(
    (kernels.SharedIndependent, kernels.SeparateIndependent),
    (SeparateIndependentInducingVariables, SharedIndependentInducingVariables),
    (SeparateIndependentInducingVariables, SharedIndependentInducingVariables),
)
def _get_posterior_independent_mo(
    kernel: Kernel, inducing_variable_u: InducingVariables, inducing_variable_v: InducingVariables
) -> Type[BasePosterior]:
    # independent multi-output
    return IndependentOrthogonalPosteriorMultiOutput
