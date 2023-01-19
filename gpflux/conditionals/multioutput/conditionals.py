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
from typing import Optional

import tensorflow as tf
import tensorflow_probability as tfp

from gpflow.base import MeanAndVariance

from gpflux.inducing_variables import SharedIndependentDistributionalInducingVariables

from gpflow.inducing_variables import (
    SharedIndependentInducingVariables,
)

from gpflux.kernels import DistributionalSharedIndependent

from gpflux.posteriors import (
    IndependentPosteriorMultiOutput,
)
from gpflux.conditionals.dispatch import conditional


@conditional._gpflow_internal_register(
    tfp.distributions.MultivariateNormalDiag, SharedIndependentDistributionalInducingVariables, DistributionalSharedIndependent, object
)
def shared_independent_distributional_conditional(
    Xnew: tf.Tensor,
    inducing_variable: SharedIndependentInducingVariables,
    kernel: DistributionalSharedIndependent,
    f: tf.Tensor,
    *,
    full_cov: bool = False,
    full_output_cov: bool = False,
    q_sqrt: Optional[tf.Tensor] = None,
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
    posterior = IndependentPosteriorMultiOutput(
        kernel,
        inducing_variable,
        f,
        q_sqrt,
        whiten=white,
        mean_function=None,
    )
    #TODO -- I think I need also a Xnew_moments here
    return posterior.fused_predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
