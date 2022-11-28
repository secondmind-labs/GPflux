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
from typing import Optional, Tuple, Type, Union

import tensorflow as tf

from gpflow import kernels
from gpflow.base import MeanAndVariance, TensorType
from gpflow.conditionals.util import expand_independent_outputs
from gpflow.config import default_jitter
from gpflow.covariances import Kuf, Kuu
from gpflow.experimental.check_shapes import check_shapes
from gpflow.inducing_variables import (
    InducingVariables,
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
)
from gpflow.kernels import Kernel, SeparateIndependent, SharedIndependent
from gpflow.mean_functions import MeanFunction
from gpflow.posteriors import (
    AbstractPosterior,
    PrecomputedValue,
    _DeltaDist,
    _DiagNormal,
    _MvNormal,
    get_posterior_class,
)

from gpflux.conditionals.util import (  # expand_independent_outputs,  duplicate of gpflow?
    base_orthogonal_conditional,
    separate_independent_orthogonal_conditional_implementation,
)
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

    def _precompute(self) -> Tuple[PrecomputedValue, ...]:
        """
        #TODO -- needs to be implemented
        """
        raise NotImplementedError


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

    #TODO -- check_shapes has to be updated
    #@check_shapes(
    #    "Xnew: [N, D]",
    #    "return: [N, N] if full_cov",
    #    "return: [N] if (not full_cov)",
    #)
    def _get_single_Cff(
        self,
        Xnew: TensorType,
        kernel: Kernel,
        inducing_variable_u: InducingVariables,
        full_cov: bool,
    ) -> tf.Tensor:

        # TODO: this assumes that Xnew has shape [N, D] and no leading dims

        Kff = kernel(Xnew, full_cov=full_cov)
        # NOTE calling kernel(Xnew, full_cov=full_cov, full_output_cov=False) directly would
        # return
        # if full_cov: [P, N, N] instead of [N, N]
        # else: [N, P] instead of [N]

        Kmm = Kuu(inducing_variable_u, kernel, jitter=default_jitter())
        L_Kmm = tf.linalg.cholesky(Kmm)

        Kmf = Kuf(inducing_variable_u, kernel, Xnew)
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

        return Cff, L_Kmm

    #TODO -- need to update check_shapes
    #@check_shapes(
    #    "Xnew: [N, D]",
    #    "return: [broadcast P, N, N] if full_cov",
    #    "return: [broadcast P, N] if (not full_cov)",
    #)
    def _get_Cff(self, Xnew: TensorType, full_cov: bool) -> tf.Tensor:

        # TODO: this assumes that Xnew has shape [N, D] and no leading dims

        if isinstance(self.kernel, SeparateIndependent):
            # NOTE calling kernel(Xnew, full_cov=full_cov, full_output_cov=False) directly would
            # return
            # if full_cov: [P, N, N] -- this is what we want
            # else: [N, P] instead of [P, N] as we get from the explicit stack below

     
            # TODO -- this could probably be done in a smarter way
            #NOTE -- at the moment it's incurring a double computation
            Cff = tf.stack(
                [
                    self._get_single_Cff(Xnew, k, ind_var, full_cov)[0]
                    for k, ind_var in zip(self.kernel.kernels, self.inducing_variable_u.inducing_variable_list)
                ],
                axis=0
            )

            L_Kmm = tf.stack(
                [
                    self._get_single_Cff(Xnew, k, ind_var, full_cov)[1]
                    for k, ind_var in zip(self.kernel.kernels, self.inducing_variable_u.inducing_variable_list)
                ],
                axis=0
            )


        elif isinstance(self.kernel, kernels.MultioutputKernel):
            # effectively, SharedIndependent path
            Cff, L_Kmm = self._get_single_Cff(Xnew, self.kernel, self.inducing_variable_u, full_cov)

        else:
            # standard ("single-output") kernels
            Cff, L_Kmm = self._get_single_Cff(
                Xnew, self.kernel, self.inducing_variable_u, full_cov
            )  # [N, N] if full_cov else [N]

        return Cff, L_Kmm

    def _conditional_with_precompute(
        self,
        cache: Tuple[tf.Tensor, ...],
        Xnew: TensorType,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> MeanAndVariance:
        """
        #TODO -- need to implement this
        """
        raise NotImplementedError


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

            # Following are: [P, M, M]  -  [P, M, N]  -  [P, N](x N)
            Kmms = Kuu(self.X_data, self.kernel, jitter=default_jitter())  # [P, M, M]
            Kmns = Kuf(self.X_data, self.kernel, Xnew)  # [P, M, N]
            Knns = self._get_Kff(Xnew, full_cov=full_output_cov)  # [P, N](x N)

            Cnns, L_Kuus = self._get_Cff(Xnew, full_cov=full_output_cov)  # [P, N](x N)
            Cmms = Cvv(
                self.inducing_variable_u,
                self.inducing_variable_v,
                self.kernel,
                jitter=default_jitter(),
                L_Kuu=L_Kuus,
            )  # [P, M_v, M_v]
            Cmns = Cvf(
                self.inducing_variable_u, self.inducing_variable_v, self.kernel, Xnew, L_Kuu=L_Kuus
            )  # [P, M_v, N]

            # TODO -- this needs to be implemented
            fmean, fvar = separate_independent_orthogonal_conditional_implementation(
                Kmns,
                Kmms,
                Knns,
                Cmns,
                Cmms,
                Cnns,
                self.q_mu_u,
                self.q_mu_v,
                full_cov=full_cov,
                q_sqrt_u=self.q_sqrt_u,
                q_sqrt_v=self.q_sqrt_v,
                white=self.whiten,
            )

        return self._post_process_mean_and_cov(fmean, fvar, full_cov, full_output_cov)


@get_posterior_class.register(kernels.Kernel, InducingVariables, InducingVariables)
def _get_posterior_base_case(
    kernel: Kernel, inducing_variable_u: InducingVariables, inducing_variable_v: InducingVariables
) -> Type[BaseOrthogonalPosterior]:
    # independent single output
    return IndependentOrthogonalPosteriorSingleOutput


@get_posterior_class.register(
    (kernels.SharedIndependent, kernels.SeparateIndependent),
    (SeparateIndependentInducingVariables, SharedIndependentInducingVariables),
    (SeparateIndependentInducingVariables, SharedIndependentInducingVariables),
)
def _get_posterior_independent_mo(
    kernel: Kernel, inducing_variable_u: InducingVariables, inducing_variable_v: InducingVariables
) -> Type[BaseOrthogonalPosterior]:
    # independent multi-output
    return IndependentOrthogonalPosteriorMultiOutput
