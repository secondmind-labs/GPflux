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
"""
This module provides :class:`GIGPLayer`, which implements a 'global inducing' point posterior for
a GP layer. Currently restricted to squared exponential kernel, inducing points, etc... See Ober and
Aitchison (2021): https://arxiv.org/abs/2005.08140 for details.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import math

from gpflow import Parameter, default_float, default_jitter
from gpflow.base import TensorType
from gpflow.kernels import SquaredExponential
from gpflow.mean_functions import Identity, MeanFunction
from gpflow.utilities.bijectors import triangular, positive


class BatchingSquaredExponential(SquaredExponential):
    """Implementation of squared exponential kernel that batches in the following way: given X with
    shape [..., N, D], and X2 with shape [..., M, D], we return [..., N, M] instead of the current
    behavior, which returns [..., N, ..., M]"""

    def scaled_squared_euclid_dist(self, X, X2=None):
        X = self.scale(X)
        X2 = self.scale(X2)

        if X2 is None:
            X2 = X

        Xs = tf.reduce_sum((X**2), -1)[..., :, None]
        X2s = tf.reduce_sum((X2**2), -1)[..., None, :]
        return Xs + X2s - 2 * tf.linalg.matmul(X, X2, transpose_b=True)


class GIGPLayer(tf.keras.layers.Layer):
    """
    A sparse variational multioutput GP layer. This layer holds the kernel,
    inducing variables and variational distribution, and mean function.
    """

    num_data: int
    """
    The number of points in the training dataset. This information is used to
    obtain the correct scaling between the data-fit and the KL term in the
    evidence lower bound (ELBO).
    """

    v: Parameter
    r"""
    The pseudo-targets. Note that this does not have the same meaning as in much of the GP 
    literature, where it represents a whitened version of the inducing variables. While we do use
    whitened representations to compute the KL, we maintain the use of `u` throughout for the 
    inducing variables, leaving `v` for the pseudo-targets, which follows the notation of Ober &
    Aitchison (2021).
    """

    L_loc: Parameter
    r"""
    The lower-triangular Cholesky factor of the precision of ``q(v|u)``.
    """

    L_scale: Parameter
    r"""
    Scale parameter for L.
    """

    def __init__(
        self,
        input_dim: int,
        num_latent_gps: int,
        num_data: int,
        num_inducing: int,
        *,
        inducing_targets: Optional[tf.Tensor] = None,
        prec_init: Optional[float] = 10.,
        mean_function: Optional[MeanFunction] = None,
        kernel_variance_init: Optional[float] = 1.,
        name: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        :param input_dim: The dimension of the input for this layer.
        :param num_latent_gps: The number of latent GPs in this layer (i.e. the output dimension).
            Unlike for the :class:`GPLayer`, this must be provided for global inducing layers.
        :param num_data: The number of points in the training dataset (see :attr:`num_data`).
        :param num_inducing: The number of inducing points; for global inducing this should be the
            same for the whole model.
        :param inducing_targets: An optional initialization for `v`. The most useful case for this
            is the last layer, where it may be initialized to (a subset of) the output data.
        :param prec_init: Initialization for the precision parameter. See Ober and Aitchison (2021)
            for more details.
        :param mean_function: The mean function that will be applied to the
            inputs. Default: :class:`~gpflow.mean_functions.Identity`.

            .. note:: The Identity mean function requires the input and output
                dimensionality of this layer to be the same. If you want to
                change the dimensionality in a layer, you may want to provide a
                :class:`~gpflow.mean_functions.Linear` mean function instead.

        :param kernel_variance_init: Initialization for the kernel variance
        :param name: The name of this layer.
        :param verbose: The verbosity mode. Set this parameter to `True`
            to show debug information.
        """

        super().__init__(
            dtype=default_float(),
            name=name,
        )

        if kernel_variance_init <= 0:
            raise ValueError("Kernel variance must be positive.")
        if prec_init <= 0:
            raise ValueError("Precision init must be positive")

        self.kernel = BatchingSquaredExponential(
            lengthscales=[1.]*input_dim, variance=kernel_variance_init)

        self.input_dim = input_dim

        self.num_data = num_data

        if mean_function is None:
            mean_function = Identity()
            if verbose:
                warnings.warn(
                    "Beware, no mean function was specified in the construction of the `GPLayer` "
                    "so the default `gpflow.mean_functions.Identity` is being used. "
                    "This mean function will only work if the input dimensionality "
                    "matches the number of latent Gaussian processes in the layer."
                )
        self.mean_function = mean_function

        self.verbose = verbose

        self.num_latent_gps = num_latent_gps

        self.num_inducing = num_inducing

        if inducing_targets is None:
            inducing_targets = np.zeros((self.num_latent_gps, num_inducing, 1))
        elif tf.rank(inducing_targets) == 2:
            inducing_targets = tf.expand_dims(tf.linalg.adjoint(inducing_targets), -1)
        if inducing_targets.shape != (self.num_latent_gps, num_inducing, 1):
            raise ValueError("Incorrect shape was provided for the inducing targets.")

        self.v = Parameter(
            inducing_targets,
            dtype=default_float(),
            name=f"{self.name}_v" if self.name else "v",
        )

        self.L_loc = Parameter(
            np.stack([np.eye(num_inducing) for _ in range(self.num_latent_gps)]),
            transform=triangular(),
            dtype=default_float(),
            name=f"{self.name}_L_loc" if self.name else "L_loc",
        )  # [num_latent_gps, num_inducing, num_inducing]

        self.L_scale = Parameter(
            np.sqrt(self.kernel.variance.numpy()*prec_init)*np.ones((self.num_latent_gps, 1, 1)),
            transform=positive(),
            dtype=default_float(),
            name=f"{self.name}_L_scale" if self.name else "L_scale"
        )

    @property
    def L(self) -> tf.Tensor:
        """
        :return: the Cholesky of the precision hyperparameter. We parameterize L using L_loc and
            L_scale to achieve greater stability during optimization.
        """
        norm = tf.reshape(tf.reduce_mean(tf.linalg.diag_part(self.L_loc), axis=-1), [-1, 1, 1])
        return self.L_loc * self.L_scale/norm

    def mvnormal_log_prob(self, sigma_L: TensorType, X: TensorType) -> tf.Tensor:
        """
        Calculates the log probability of a zero-mean multivariate Gaussian with covariance sigma
        and evaluation points X, with batching of both the covariance and X.

        TODO: look into whether this can be replaced with a tfp.distributions.Distribution
        :param sigma_L: Cholesky of covariance sigma, shape [..., 1, D, D]
        :param X: evaluation point for log_prob, shape [..., M, D, 1]
        :return: the log probability, shape [..., M]
        """
        in_features = tf.cast(tf.shape(X)[-2], dtype=default_float())
        out_features = tf.cast(tf.shape(X)[-1], dtype=default_float())
        trace_quad = tf.reduce_sum(tf.linalg.triangular_solve(sigma_L, X)**2, [-1, -2])
        logdet_term = 2.0*tf.reduce_sum(tf.math.log(tf.linalg.diag_part(sigma_L)), -1)
        return -0.5*trace_quad - 0.5*out_features*(logdet_term + in_features*math.log(2*math.pi))

    def call(
        self,
        inputs: TensorType,
        *args: List[Any],
        **kwargs: Dict[str, Any]
    ) -> tf.Tensor:
        """
        Sample-based propagation of both inducing points and function values. See Ober & Aitchison
        (2021) for details.
        """
        mean_function = self.mean_function(inputs)

        Kuu = self.kernel(inputs[..., :self.num_inducing, :])
        Kuf = self.kernel(inputs[..., :self.num_inducing, :], inputs[..., self.num_inducing:, :])
        Kfu = tf.linalg.adjoint(Kuf)
        Kff = self.kernel.K_diag(inputs[..., self.num_inducing:, :])

        Kuu, Kuf, Kfu = tf.expand_dims(Kuu, 1), tf.expand_dims(Kuf, 1), tf.expand_dims(Kfu, 1)

        u, chol_lKlpI, chol_Kuu = self.sample_u(Kuu)

        if kwargs.get("training"):
            loss_per_datapoint = self.prior_kl(tf.linalg.adjoint(self.L), chol_lKlpI, u) / self.num_data
        else:
            # TF quirk: add_loss must always add a tensor to compile
            loss_per_datapoint = tf.constant(0.0, dtype=default_float())
        self.add_loss(loss_per_datapoint)

        # Metric names should be unique; otherwise they get overwritten if you
        # have multiple with the same name
        name = f"{self.name}_prior_kl" if self.name else "prior_kl"
        self.add_metric(loss_per_datapoint, name=name, aggregation="mean")

        if kwargs.get("full_cov"):
            f_samples = self.sample_conditional(u, Kff, Kuf, chol_Kuu, inputs=inputs, full_cov=True)
        else:
            f_samples = self.sample_conditional(u, Kff, Kuf, chol_Kuu)

        all_samples = tf.concat(
            [
                tf.linalg.adjoint(tf.squeeze(u, -1)),
                f_samples,
            ],
            axis=-2
        )

        return all_samples + mean_function

    def sample_u(
        self,
        Kuu: TensorType
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # Samples inducing locations u

        S = tf.shape(Kuu)[0]

        Iuu = tf.eye(self.num_inducing, dtype=default_float())

        L = self.L
        LT = tf.linalg.adjoint(L)

        KuuL = Kuu @ L

        lKlpI = LT @ KuuL + Iuu
        chol_lKlpI = tf.linalg.cholesky(lKlpI)
        Sigma = Kuu - KuuL @ tf.linalg.cholesky_solve(chol_lKlpI, tf.linalg.adjoint(KuuL))

        eps_1 = tf.random.normal(
            [S, self.num_latent_gps, self.num_inducing, 1],
            dtype=default_float()
        )
        eps_2 = tf.random.normal(
            [S, self.num_latent_gps, self.num_inducing, 1],
            dtype=default_float()
        )

        Kuu = Kuu + default_jitter()*tf.eye(self.num_inducing, dtype=default_float())
        chol_Kuu = tf.linalg.cholesky(Kuu)
        chol_Kuu_T = tf.linalg.adjoint(chol_Kuu)

        inv_Kuu_noise = tf.linalg.triangular_solve(chol_Kuu_T, eps_1, lower=False)
        L_noise = L @ eps_2
        prec_noise = inv_Kuu_noise + L_noise
        u = Sigma @ ((L @ LT) @ self.v + prec_noise)

        return u, chol_lKlpI, chol_Kuu

    def predict(
        self,
        u: TensorType,
        Kff: TensorType,
        Kuf: TensorType,
        chol_Kuu: TensorType,
        inputs: Optional[TensorType] = None,
        full_cov: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        Kfu_invKuu = tf.linalg.adjoint(tf.linalg.cholesky_solve(chol_Kuu, Kuf))
        Ef = tf.linalg.adjoint(tf.squeeze((Kfu_invKuu @ u), -1))

        if full_cov:
            assert inputs is not None
            Kff = self.kernel(inputs[..., self.num_inducing:, :])
            Vf = Kff - tf.squeeze(Kfu_invKuu @ Kuf, 1)
            Vf = Vf + default_jitter()*tf.eye(tf.shape(Vf)[-1], dtype=default_float())
        else:
            Vf = Kff - tf.squeeze(tf.reduce_sum((Kfu_invKuu*tf.linalg.adjoint(Kuf)), -1), 1)

            Vf = Vf[..., None]

        return Ef, Vf

    def predict(
        self,
        u: TensorType,
        Kff: TensorType,
        Kuf: TensorType,
        chol_Kuu: TensorType,
        inputs: Optional[TensorType] = None,
        full_cov: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        Kfu_invKuu = tf.linalg.adjoint(tf.linalg.cholesky_solve(chol_Kuu, Kuf))
        Ef = tf.linalg.adjoint(tf.squeeze((Kfu_invKuu @ u), -1))

        if full_cov:
            assert inputs is not None
            Kff = self.kernel(inputs[..., self.num_inducing:, :])
            Vf = Kff - tf.squeeze(Kfu_invKuu @ Kuf, 1)
            Vf = Vf + default_jitter()*tf.eye(tf.shape(Vf)[-1], dtype=default_float())
        else:
            Vf = Kff - tf.squeeze(tf.reduce_sum((Kfu_invKuu*tf.linalg.adjoint(Kuf)), -1), 1)

            Vf = Vf[..., None]

        return Ef, Vf

    def sample_conditional(
        self,
        u: TensorType,
        Kff: TensorType,
        Kuf: TensorType,
        chol_Kuu: TensorType,
        inputs: Optional[TensorType] = None,
        full_cov: bool = False,
    ) -> tf.Tensor:
        """
        Samples function values f based off samples of u.

        :param u: Samples of the inducing points, shape [S, Lout, M, 1]
        :param Kff: The diag of the kernel evaluated at input function values, shape [S, N]
        :param Kuf: The kernel evaluated between inducing locations and input function values, shape
            [S, 1, M, N]
        :param chol_Kuu: Cholesky factor of kernel evaluated for inducing points, shape [S, 1, M, M]
        :param inputs: Input data points, required for full_cov = True, shape [S, N, Lin]
        :param full_cov: Whether to use the full covariance predictive, which gives consistent
            samples if true
        :return: samples of f, shape [S, M, Lout]
        """
        Ef, Vf = self.predict(u, Kff, Kuf, chol_Kuu, inputs=inputs, full_cov=full_cov)

        eps_f = tf.random.normal(
            tf.shape(Ef),
            dtype=default_float()
        )

        if full_cov:
            chol_Vf = tf.linalg.cholesky(Vf)

            var_part = chol_Vf @ eps_f
        else:
            var_part = tf.math.sqrt(Vf)*eps_f

        return Ef + var_part

    def prior_kl(
        self,
        LT: tf.Tensor,
        chol_lKlpI: tf.Tensor,
        u: tf.Tensor,
    ) -> tf.Tensor:
        """
        Returns sample-based estimates of the KL divergence between the approximate posterior and
        the prior, KL(q(u)||p(u)). Note that we use a whitened representation to compute the KL:

        P = L LT
        u = N(0, K)
        v | u = N(u, P^{-1})
        u | v = N(S P u, S)  - this is the approximate posterior form for u, where
        S = (K^{-1} + P)^{-1} = K - K L (LT K L + I)^{-1} LT K

        To compute the KL:
        lu = LT u
        lv = LT v
        lv | u = N(lu, I)
        lv = N(0, LT K L + I)

        P(u)/P(u|lv) = P(lv)/P(lv|u)

        :param LT: transpose of L, shape [Lout, M, M]
        :param chol_lKlpI: Cholesky of LT @ Kuu @ L + I, shape [S, Lout, M, M]
        :param u: Samples of the inducing points, shape [S, Lout, M, 1]
        :return: Samples-based estimate of the KL, shape []
        """
        lv = LT @ self.v

        logP = tf.reduce_sum(self.mvnormal_log_prob(chol_lKlpI, lv), -1)
        logQ = tf.reduce_sum(tfp.distributions.Normal(LT@u, 1.).log_prob(lv), [-1, -2, -3])

        logpq = logP - logQ

        return -tf.reduce_mean(logpq)
