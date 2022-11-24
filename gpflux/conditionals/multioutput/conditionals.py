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

from typing import Optional
from pkg_resources import Distribution

import tensorflow as tf
import tensorflow_probability as tfp

from gpflow.base import MeanAndVariance

from gpflow.inducing_variables import (
    FallbackSeparateIndependentInducingVariables,
    FallbackSharedIndependentInducingVariables,
    InducingPoints,
    MultioutputInducingVariables,
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
)
from gpflow.kernels import (
    MultioutputKernel,
    SeparateIndependent,
    SharedIndependent,
)


from  gpflux.conditionals.dispatch import conditional

from gpflux.posteriors import BasePosterior, get_posterior_class


@conditional._gpflow_internal_register(
    object, SharedIndependentInducingVariables, SharedIndependentInducingVariables, SharedIndependent, object, object
)
def shared_independent_orthogonal_conditional(
    Xnew: tf.Tensor,
    inducing_variable_u: SharedIndependentInducingVariables,
    inducing_variable_v: SharedIndependentInducingVariables,
    kernel: SharedIndependent,
    f_u: tf.Tensor,
    f_v: tf.Tensor,
    *,
    full_cov: bool = False,
    full_output_cov: bool = False,
    q_sqrt_u: Optional[tf.Tensor] = None,
    q_sqrt_v: Optional[tf.Tensor] = None,
    white: bool = False
) -> MeanAndVariance:
    """Multioutput conditional for an independent kernel and shared inducing inducing.
    Same behaviour as conditional with non-multioutput kernels.
    The covariance matrices used to calculate the conditional have the following shape:
    - Kuu: [M, M]
    - Kuf: [M, N]
    - Kff: N or [N, N]

    Further reference
    -----------------
    - See `gpflow.conditionals._conditional` for a detailed explanation of
      conditional in the single-output case.
    - See the multioutput notebook for more information about the multioutput framework.
    Parameters
    ----------
    :param Xnew: data matrix, size [N, D].
    :param f: data matrix, [M, P]
    :param full_cov: return the covariance between the datapoints
    :param full_output_cov: return the covariance between the outputs.
        Note: as we are using a independent kernel these covariances will be zero.
    :param q_sqrt: matrix of standard-deviations or Cholesky matrices,
        size [M, P] or [P, M, M].
    :param white: boolean of whether to use the whitened representation
    :return:
        - mean:     [N, P]
        - variance: [N, P], [P, N, N], [N, P, P] or [N, P, N, P]
        Please see `gpflow.conditional._expand_independent_outputs` for more information
        about the shape of the variance, depending on `full_cov` and `full_output_cov`.
    """

    posterior_class = get_posterior_class(kernel, inducing_variable_u, inducing_variable_v)

    posterior = posterior_class(
        kernel,
        inducing_variable_u,
        inducing_variable_v,
        f_u,
        f_v,
        q_sqrt_u,
        q_sqrt_v,
        whiten=white,
        mean_function=None,
    )
    return posterior.fused_predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)


