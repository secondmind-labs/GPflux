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

import tensorflow as tf

from gpflow.base import MeanAndVariance
from gpflow.inducing_variables import InducingVariables
from gpflow.kernels import Kernel

from gpflux.conditionals.dispatch import conditional
from gpflux.posteriors import get_posterior_class


@conditional._gpflow_internal_register(
    object, InducingVariables, InducingVariables, Kernel, object, object
)
def _sparse_orthogonal_conditional(
    Xnew: tf.Tensor,
    inducing_variable_u: InducingVariables,
    inducing_variable_v: InducingVariables,
    kernel: Kernel,
    f_u: tf.Tensor,
    f_v: tf.Tensor,
    *,
    full_cov: bool = False,
    full_output_cov: bool = False,
    q_sqrt_u: Optional[tf.Tensor] = None,
    q_sqrt_v: Optional[tf.Tensor] = None,
    white: bool = False
) -> MeanAndVariance:
    """
    Single-output distributional orthogonal GP conditional.

    The covariance matrices used to calculate the conditional have the following shape:
    - Kuu: [M, M]
    - Kuf: [M, N]
    - Kff: [N, N]

    Further reference
    -----------------
    - See `gpflow.conditionals._dense_conditional` (below) for a detailed explanation of
      conditional in the single-output case.
    - See the multiouput notebook for more information about the multiouput framework.

    Parameters
    ----------
    :param Xnew: data matrix, size [N, D].
    :param f: data matrix, [M, R]
    :param full_cov: return the covariance between the datapoints
    :param full_output_cov: return the covariance between the outputs.
           NOTE: as we are using a single-output kernel with repetitions
                 these covariances will be zero.
    :param q_sqrt: matrix of standard-deviations or Cholesky matrices,
        size [M, R] or [R, M, M].
    :param white: boolean of whether to use the whitened representation
    :return:
        - mean:     [N, R]
        - variance: [N, R], [R, N, N], [N, R, R] or [N, R, N, R]
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
