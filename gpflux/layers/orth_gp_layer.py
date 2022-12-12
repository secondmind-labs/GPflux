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
"""This module provides the implementation of the sparse orthogonal variational GP layer"""
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from gpflow import Parameter, default_float
from gpflow.base import TensorType
from gpflow.conditionals import conditional
from gpflow.functions import MeanFunction
from gpflow.inducing_variables import MultioutputInducingVariables
from gpflow.kernels import MultioutputKernel
from gpflow.kullback_leiblers import prior_kl
from gpflow.utilities import triangular

from gpflux.layers import GPLayer


class OrthGPLayer(GPLayer):
    """
    A sparse orthogonal variational multioutput GP layer. This layer holds the kernel,
    inducing variables and variational distribution, and mean function.
    """

    q_mu_u: Parameter
    r"""
    The mean of ``q(v)`` or ``q(u)`` (depending on whether :attr:`whiten`\ ed
    parametrisation is used).
    """

    q_mu_v: Parameter
    r"""
    The mean of ``q(v)`` or ``q(u)`` (depending on whether :attr:`whiten`\ ed
    parametrisation is used).
    """

    q_sqrt_u: Parameter
    r"""
    The lower-triangular Cholesky factor of the covariance of ``q(v)`` or ``q(u)``
    (depending on whether :attr:`whiten`\ ed parametrisation is used).
    """

    q_sqrt_v: Parameter
    r"""
    The lower-triangular Cholesky factor of the covariance of ``q(v)`` or ``q(u)``
    (depending on whether :attr:`whiten`\ ed parametrisation is used).
    """

    def __init__(
        self,
        kernel: MultioutputKernel,
        inducing_variable_u: MultioutputInducingVariables,
        inducing_variable_v: MultioutputInducingVariables,
        num_data: int,
        mean_function: Optional[MeanFunction] = None,
        *,
        num_samples: Optional[int] = None,
        full_cov: bool = False,
        full_output_cov: bool = False,
        num_latent_gps: int = None,
        whiten: bool = True,
        name: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        :param kernel: The multioutput kernel for this layer.
        :param inducing_variable: The inducing features for this layer.
        :param num_data: The number of points in the training dataset (see :attr:`num_data`).
        :param mean_function: The mean function that will be applied to the
            inputs. Default: :class:`~gpflow.mean_functions.Identity`.

            .. note:: The Identity mean function requires the input and output
                dimensionality of this layer to be the same. If you want to
                change the dimensionality in a layer, you may want to provide a
                :class:`~gpflow.mean_functions.Linear` mean function instead.

        :param num_samples: The number of samples to draw when converting the
            :class:`~tfp.layers.DistributionLambda` into a `tf.Tensor`, see
            :meth:`_convert_to_tensor_fn`. Will be stored in the
            :attr:`num_samples` attribute.  If `None` (the default), draw a
            single sample without prefixing the sample shape (see
            :class:`tfp.distributions.Distribution`'s `sample()
            <https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Distribution#sample>`_
            method).
        :param full_cov: Sets default behaviour of calling this layer
            (:attr:`full_cov` attribute):
            If `False` (the default), only predict marginals (diagonal
            of covariance) with respect to inputs.
            If `True`, predict full covariance over inputs.
        :param full_output_cov: Sets default behaviour of calling this layer
            (:attr:`full_output_cov` attribute):
            If `False` (the default), only predict marginals (diagonal
            of covariance) with respect to outputs.
            If `True`, predict full covariance over outputs.
        :param num_latent_gps: The number of (latent) GPs in the layer
            (which can be different from the number of outputs, e.g. with a
            :class:`~gpflow.kernels.LinearCoregionalization` kernel).
            This is used to determine the size of the
            variational parameters :attr:`q_mu` and :attr:`q_sqrt`.
            If possible, it is inferred from the *kernel* and *inducing_variable*.
        :param whiten: If `True` (the default), uses the whitened parameterisation
            of the inducing variables; see :attr:`whiten`.
        :param name: The name of this layer.
        :param verbose: The verbosity mode. Set this parameter to `True`
            to show debug information.
        """

        super().__init__(
            kernel=kernel,
            inducing_variable=inducing_variable_u,
            num_data=num_data,
            mean_function=mean_function,
            num_samples=num_samples,
            full_cov=full_cov,
            full_output_cov=full_output_cov,
            num_latent_gps=num_latent_gps,
            whiten=whiten,
            name=name,
            verbose=verbose,
        )

        self.inducing_variable_u = inducing_variable_u
        self.inducing_variable_v = inducing_variable_v

        """
        try:
            num_inducing, self.num_latent_gps = verify_compatibility(
                kernel, mean_function, inducing_variable
            )
            # TODO: if num_latent_gps is not None, verify it is equal to self.num_latent_gps
        except GPLayerIncompatibilityException as e:
            if num_latent_gps is None:
                raise e

            if verbose:
                warnings.warn(
                    "Could not verify the compatibility of the `kernel`, `inducing_variable` "
                    "and `mean_function`. We advise using `gpflux.helpers.construct_*` to create "
                    "compatible kernels and inducing variables. As "
                    f"`num_latent_gps={num_latent_gps}` has been specified explicitly, this will "
                    "be used to create the `q_mu` and `q_sqrt` parameters."
                )

            num_inducing, self.num_latent_gps = (
                inducing_variable.num_inducing,
                num_latent_gps,
            )
        """
        num_inducing_u = self.inducing_variable_u.num_inducing
        num_inducing_v = self.inducing_variable_v.num_inducing

        ########################################################
        #      Introduce variational parameters for q(U)       #
        ########################################################

        self.q_mu_u = Parameter(
            np.random.uniform(
                -0.5, 0.5, (num_inducing_u, self.num_latent_gps)
            ),  # np.zeros((num_inducing, self.num_latent_gps)),
            dtype=default_float(),
            name=f"{self.name}_q_mu_u" if self.name else "q_mu_u",
        )  # [num_inducing, num_latent_gps]

        self.q_sqrt_u = Parameter(
            np.stack([np.eye(num_inducing_u) for _ in range(self.num_latent_gps)]),
            transform=triangular(),
            dtype=default_float(),
            name=f"{self.name}_q_sqrt_u" if self.name else "q_sqrt_u",
        )  # [num_latent_gps, num_inducing, num_inducing]

        ########################################################
        #      Introduce variational parameters for q(V)       #
        ########################################################

        self.q_mu_v = Parameter(
            np.random.uniform(
                -0.5, 0.5, (num_inducing_v, self.num_latent_gps)
            ),  # np.zeros((num_inducing, self.num_latent_gps)),
            dtype=default_float(),
            name=f"{self.name}_q_mu_v" if self.name else "q_mu_v",
        )  # [num_inducing, num_latent_gps]

        self.q_sqrt_v = Parameter(
            np.stack([np.eye(num_inducing_v) for _ in range(self.num_latent_gps)]),
            transform=triangular(),
            dtype=default_float(),
            name=f"{self.name}_q_sqrt_v" if self.name else "q_sqrt_v",
        )  # [num_latent_gps, num_inducing, num_inducing]

        self.num_samples = num_samples

    def predict(
        self,
        inputs: TensorType,
        *,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Make a prediction at N test inputs for the Q outputs of this layer,
        including the mean function contribution.

        The covariance and its shape is determined by *full_cov* and *full_output_cov* as follows:

        +--------------------+---------------------------+--------------------------+
        | (co)variance shape | ``full_output_cov=False`` | ``full_output_cov=True`` |
        +--------------------+---------------------------+--------------------------+
        | ``full_cov=False`` | [N, Q]                    | [N, Q, Q]                |
        +--------------------+---------------------------+--------------------------+
        | ``full_cov=True``  | [Q, N, N]                 | [N, Q, N, Q]             |
        +--------------------+---------------------------+--------------------------+

        :param inputs: The inputs to predict at, with a shape of [N, D], where D is
            the input dimensionality of this layer.
        :param full_cov: Whether to return full covariance (if `True`) or
            marginal variance (if `False`, the default) w.r.t. inputs.
        :param full_output_cov: Whether to return full covariance (if `True`)
            or marginal variance (if `False`, the default) w.r.t. outputs.

        :returns: posterior mean (shape [N, Q]) and (co)variance (shape as above) at test points
        """
        mean_function = self.mean_function(inputs)
        mean_cond, cov = conditional(
            inputs,
            self.inducing_variable_u,
            self.inducing_variable_v,
            self.kernel,
            self.q_mu_u,
            self.q_mu_v,
            q_sqrt_u=self.q_sqrt_u,
            q_sqrt_v=self.q_sqrt_v,
            full_cov=full_cov,
            full_output_cov=full_output_cov,
            white=self.whiten,
        )

        return mean_cond + mean_function, cov

    def prior_kl(self) -> tf.Tensor:
        r"""
        Returns the KL divergence ``KL[q(u)∥p(u)]`` from the prior ``p(u)`` to
        the variational distribution ``q(u)``.  If this layer uses the
        :attr:`whiten`\ ed representation, returns ``KL[q(v)∥p(v)]``.
        """
        return prior_kl(
            self.inducing_variable_u, self.kernel, self.q_mu_u, self.q_sqrt_u, whiten=self.whiten
        ) + prior_kl(
            self.inducing_variable_v, self.kernel, self.q_mu_v, self.q_sqrt_v, whiten=self.whiten
        )
