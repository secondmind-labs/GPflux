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

import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import math

from gpflow import Parameter, default_float, default_jitter
from gpflow.base import TensorType
from gpflow.kernels import SquaredExponential
from gpflow.mean_functions import Identity, MeanFunction
from gpflow.utilities.bijectors import triangular, positive

from .general import KG, SqExpKernelGram


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


def bartlett(Kshape, nu, sample_shape):
    """Returns A from the standard Bartlett (i.e. scale I) - ignores K"""
    random_samples = tf.random.normal(tf.concat([sample_shape, Kshape], 0), dtype=default_float())
    n = tf.experimental.numpy.tril(
        random_samples, -1
    )
    # n = tf.linalg.band_part(random_samples, -1, 0) - tf.linalg.band_part(random_samples, 0, 0)

    I = tf.eye(Kshape[-1], dtype=default_float())

    dof = nu - tf.range(Kshape[-1], dtype=default_float())
    gamma = tfp.distributions.Gamma(dof/2., 0.5)

    c = tf.sqrt(gamma.sample(tf.concat([sample_shape, Kshape[:-2], [1]], 0)))
    A = n + I*c
    return A


def mvlgamma(input, p):
    result = math.log(math.pi)*p*(p-1)/4.

    for i in range(tf.cast(p, tf.int32)):
        result = result + tf.math.lgamma(input - 0.5*tf.cast(i, default_float()))

    return result


class InverseWishart:
    def __init__(self, K, nu):
        tf.debugging.assert_less(tf.cast(tf.shape(K)[-1], default_float()), nu)
        self.nu = nu
        self.K = K
        self.Kshape = tf.shape(K)

    def rsample_log_prob(self, sample_shape):
        x, A = self._rsample(sample_shape)
        log_prob = self.log_prob(x, A)
        return x, log_prob

    def _rsample(self, sample_shape):
        L = tf.linalg.cholesky(self.K)
        A = bartlett(self.Kshape, self.nu, sample_shape)
        return L @ tf.linalg.cholesky_solve(A, tf.linalg.adjoint(L)), A

    def rsample(self, sample_shape):
        return self._rsample(sample_shape)[0]

    def log_prob(self, x, A=None):
        if A is None:
            Lx = tf.linalg.cholesky(x)
            AAT = tf.linalg.cholesky_solve(Lx, self.K)
        else:
            AAT = A @ tf.linalg.adjoint(A)

        nu = self.nu
        p = tf.cast(tf.shape(self.K)[-1], default_float())

        res = -((nu + p + 1)/2)*tf.linalg.logdet(x)
        res = res + (nu/2)*tf.linalg.logdet(self.K)
        res = res - 0.5*tf.reduce_sum(tf.linalg.diag_part(AAT), -1)
        res = res - 0.5*nu*p * math.log(2)
        res = res - mvlgamma(0.5*nu*tf.ones([], dtype=gpflow.default_float()), p)

        return res


class IWLayer(tf.keras.layers.Layer):

    num_data: int

    delta: Parameter

    V: Parameter

    diag: Parameter

    gamma: Parameter

    def __init__(
        self,
        num_data: int,
        num_inducing: int,
        *,
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

        self.kernel_gram = SqExpKernelGram(height=kernel_variance_init, trainable_noise=False)

        self.num_data = num_data

        self.verbose = verbose

        self.P = num_inducing

        self.delta = Parameter(
            tf.ones([]),
            transform=positive(),
            dtype=default_float(),
            name=f"{self.name}_delta" if self.name else "delta",
        )

        self.V = Parameter(
            tf.random.normal([num_inducing, num_inducing]),
            dtype=default_float(),
            transform=triangular(),
            name=f"{self.name}_V" if self.name else "V",
        )

        self.diag = Parameter(
            tf.ones([]),
            transform=positive(),
            dtype=default_float(),
            name=f"{self.name}_diag" if self.name else "diag",
        )

        self.gamma = Parameter(
            tf.ones([]),
            transform=positive(),
            dtype=default_float(),
            name=f"{self.name}_gamma" if self.name else "gamma"
        )

    @property
    def prior_nu(self) -> tf.Tensor:
        return self.delta + self.P + 1

    @property
    def post_nu(self) -> tf.Tensor:
        return self.prior_nu + self.gamma

    def prior_psi(self, dKii: TensorType) -> TensorType:
        return dKii

    def post_psi(self, dKii: TensorType):
        return dKii + tf.linalg.matmul(self.V*self.diag, self.V, transpose_b=True)/self.P

    def PGii(self, dKii: TensorType) -> InverseWishart:
        return InverseWishart(self.prior_psi(dKii), self.prior_nu)

    def QGii(self, dKii: TensorType) -> InverseWishart:
        return InverseWishart(self.post_psi(dKii), self.post_nu)

    def Gii(self, dKii: TensorType) -> Tuple[tf.Tensor, tf.Tensor]:
        PGii = self.PGii(dKii)
        QGii = self.QGii(dKii)

        Gii, logQ = QGii.rsample_log_prob([])
        logP = PGii.log_prob(Gii)

        logpq = logP - logQ

        return Gii, logpq

    def call(
        self,
        K: KG,
        *args: List[Any],
        **kwargs: Dict[str, Any]
    ) -> KG:
        K = self.kernel_gram(K, full_cov=kwargs.get("full_cov"))

        dKii = self.delta * K.ii
        dKit = self.delta * K.it
        dKtt = self.delta * K.tt

        Gii, logpq = self.Gii(dKii)

        if kwargs.get("training"):
            loss_per_datapoint = -tf.reduce_mean(logpq) / self.num_data
        else:
            # TF quirk: add_loss must always add a tensor to compile
            loss_per_datapoint = tf.constant(0.0, dtype=default_float())
        self.add_loss(loss_per_datapoint)

        chol_Kii = tf.linalg.cholesky(dKii)
        inv_Kii_kit = tf.linalg.cholesky_solve(chol_Kii, dKit)

        if kwargs.get("full_cov"):
            Git, Gtt = self.sample_conditional_full(Gii, dKii, dKit, dKtt, inv_Kii_kit)
        else:
            Git, Gtt = self.sample_conditional_marginal(Gii, dKii, dKit, dKtt, inv_Kii_kit)

        return KG(Gii, Git, Gtt)

    def sample_conditional_marginal(
        self,
        Gii: TensorType,
        dKii: TensorType,
        dKit: TensorType,
        dktt: TensorType,
        inv_Kii_kit: TensorType,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        dktti = dktt - tf.reduce_sum(dKit * inv_Kii_kit, -2)
        alpha = (self.delta + self.P + tf.cast(tf.shape(dktt)[-1], default_float()) + 1)/2
        Ptt = tfp.distributions.Gamma(alpha, dktti/2)
        gtti = tf.math.reciprocal(Ptt.sample([]))

        chol_dKii = tf.linalg.cholesky(dKii)
        inv_Gii_git = inv_Kii_kit + tf.linalg.triangular_solve(
            tf.linalg.adjoint(chol_dKii),
            tf.random.normal(tf.shape(dKit), dtype=default_float()),
            lower=False
        ) * tf.sqrt(gtti)[:, None, :]
        git = Gii @ inv_Gii_git

        gtt = gtti + tf.reduce_sum(git * inv_Gii_git, -2)

        return git, gtt

    def sample_conditional_full(
        self,
        Gii: TensorType,
        dKii: TensorType,
        dKit: TensorType,
        dKtt: TensorType,
        inv_Kii_kit: TensorType,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        dKtti = dKtt - tf.linalg.matmul(dKit, inv_Kii_kit, transpose_a=True)
        nu = self.delta + self.P + tf.cast(tf.shape(dKtt)[-1], default_float()) + 1
        Ptti = InverseWishart(dKtti, nu)
        Gtti_sample = Ptti.rsample([])
        Gtti = Gtti_sample + default_jitter()*tf.reduce_max(Gtti_sample)*tf.eye(tf.shape(Gtti_sample)[-1], dtype=default_float())

        chol_dKii = tf.linalg.cholesky(dKii)
        inv_Gii_git = inv_Kii_kit + tf.linalg.matmul(tf.linalg.triangular_solve(
            tf.linalg.adjoint(chol_dKii),
            tf.random.normal(tf.shape(dKit), dtype=default_float()),
            lower=False
        ), tf.linalg.cholesky(Gtti), transpose_b=True)
        Git = Gii @ inv_Gii_git

        Gtt = Gtti + tf.linalg.matmul(Git, inv_Gii_git, transpose_a=True)

        return Git, Gtt


class KGIGPLayer(tf.keras.layers.Layer):
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
        num_latent_gps: int,
        num_data: int,
        num_inducing: int,
        *,
        inducing_targets: Optional[tf.Tensor] = None,
        prec_init: Optional[float] = 10.,
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

        if prec_init <= 0:
            raise ValueError("Precision init must be positive")

        self.kernel_gram = SqExpKernelGram(height=kernel_variance_init, trainable_noise=False)

        self.num_data = num_data

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
            np.sqrt(kernel_variance_init*prec_init)*np.ones((self.num_latent_gps, 1, 1)),
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
        inputs: KG,
        *args: List[Any],
        **kwargs: Dict[str, Any]
    ) -> tf.Tensor:
        """
        Sample-based propagation of both inducing points and function values. See Ober & Aitchison
        (2021) for details.
        """
        inputs = self.kernel_gram(inputs, full_cov=kwargs.get("full_cov"))

        Kuu = inputs.ii
        Kuf = inputs.it
        Kfu = tf.linalg.adjoint(Kuf)
        Kff = inputs.tt

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
            f_samples = self.sample_conditional(u, Kff, Kuf, chol_Kuu, full_cov=True)
        else:
            f_samples = self.sample_conditional(u, Kff, Kuf, chol_Kuu)

        all_samples = tf.concat(
            [
                tf.linalg.adjoint(tf.squeeze(u, -1)),
                f_samples,
            ],
            axis=-2
        )

        return all_samples

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
        full_cov: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        Kfu_invKuu = tf.linalg.adjoint(tf.linalg.cholesky_solve(chol_Kuu, Kuf))
        Ef = tf.linalg.adjoint(tf.squeeze((Kfu_invKuu @ u), -1))

        if full_cov:
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
        Ef, Vf = self.predict(u, Kff, Kuf, chol_Kuu, full_cov=full_cov)

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
