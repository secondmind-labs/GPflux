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
a GP layer. Currently restricted to single-output kernels, inducing points, etc... See Ober and
Aitchison (2021) for details.
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

from gpflux.sampling.sample import Sample


class BatchingSquaredExponential(SquaredExponential):
    """Implementation of squared exponential kernel that batches in the following way: given X with
    shape [..., N, D], and X2 with shape [..., M, D], we return [..., N, M] instead of the current
    behavior, which returns [..., N, ..., M]"""

    def K_r2(self, r2):
        return self.variance * tf.exp(-0.5 * r2)

    def scaled_squared_euclid_dist(self, X, X2=None):
        X = self.scale(X)
        X2 = self.scale(X2)

        if X2 is None:
            X2 = X

        Xs = tf.reduce_sum((X**2), -1)[..., :, None]
        X2s = tf.reduce_sum((X2**2), -1)[..., None, :]
        return Xs + X2s - 2*X@tf.linalg.adjoint(X2)


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
    The pseudo-targets. 
    """

    L_loc: Parameter
    r"""
    The lower-triangular Cholesky factor of the precision of ``q(u|v)``.
    """

    L_scale: Parameter
    r"""
    Scale parameter for L
    """

    def __init__(
        self,
        num_latent_gps: int,
        num_data: int,
        num_inducing: int,
        input_dim: int,
        inducing_targets: Optional[tf.Tensor] = None,
        prec_init: Optional[float] = 1.,
        mean_function: Optional[MeanFunction] = None,
        *,
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
            dtype=default_float(),
            name=name,
        )

        self.kernel = BatchingSquaredExponential(lengthscales=[1.]*input_dim)

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
            np.sqrt(prec_init)*np.ones((self.num_latent_gps, 1, 1)),
            transform=positive(),
            dtype=default_float(),
            name=f"{self.name}_L_scale" if self.name else "L_scale"
        )

    @property
    def L(self):
        norm = tf.reshape(tf.reduce_mean(tf.linalg.diag_part(self.L_loc), axis=-1), [-1, 1, 1])
        return self.L_loc * self.L_scale/norm

    def mvnormal_log_prob(self, sigma_L, X):
        in_features = self.input_dim
        out_features = self.num_latent_gps
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
        Sample-based propagation of both inducing points and function values.
        """
        mean_function = self.mean_function(inputs)

        Kuu = self.kernel(inputs[..., :self.num_inducing, :])
        Kuf = self.kernel(inputs[..., :self.num_inducing, :], inputs[..., self.num_inducing:, :])
        Kfu = tf.linalg.adjoint(Kuf)
        Kff = self.kernel.K_diag(inputs[..., self.num_inducing:, :])

        Kuu, Kuf, Kfu = tf.expand_dims(Kuu, 1), tf.expand_dims(Kuf, 1), tf.expand_dims(Kfu, 1)

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

        if kwargs.get("training"):
            loss_per_datapoint = self.prior_kl(LT, chol_lKlpI, u) / self.num_data
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

    def sample_conditional(
        self,
        u: TensorType,
        Kff: TensorType,
        Kuf: TensorType,
        chol_Kuu: TensorType,
        inputs: Optional[TensorType] = None,
        full_cov: bool = False,
    ) -> tf.Tensor:
        Kfu_invKuu = tf.linalg.adjoint(tf.linalg.cholesky_solve(chol_Kuu, Kuf))
        Ef = tf.linalg.adjoint(tf.squeeze((Kfu_invKuu @ u), -1))

        eps_f = tf.random.normal(
            tf.shape(Ef),
            dtype=default_float()
        )

        if full_cov:
            assert inputs is not None
            Kff = self.kernel(inputs[..., self.num_inducing:, :])
            Vf = Kff - tf.squeeze(Kfu_invKuu @ Kuf, 1)
            Vf = Vf + default_jitter()*tf.eye(tf.shape(Vf)[-1], dtype=default_float())
            chol_Vf = tf.linalg.cholesky(Vf)

            var_part = chol_Vf @ eps_f
        else:
            Vf = Kff - tf.squeeze(tf.reduce_sum((Kfu_invKuu*tf.linalg.adjoint(Kuf)), -1), 1)

            var_part = tf.math.sqrt(Vf)[..., None]*eps_f

        return Ef + var_part

    def prior_kl(
        self,
        LT: tf.Tensor,
        chol_lKlpI: tf.Tensor,
        u: tf.Tensor,
    ) -> tf.Tensor:
        r"""
        Returns the KL divergence ``KL[q(u)∥p(u)]`` from the prior ``p(u)`` to
        the variational distribution ``q(u)``.  If this layer uses the
        :attr:`whiten`\ ed representation, returns ``KL[q(v)∥p(v)]``.
        """
        lv = LT @ self.v

        logP = tf.reduce_sum(self.mvnormal_log_prob(chol_lKlpI, lv), -1)
        logQ = tf.reduce_sum(tfp.distributions.Normal(LT@u, 1.).log_prob(lv), [-1, -2, -3])

        logpq = logP - logQ

        return -tf.reduce_mean(logpq)

    def sample(self, inputs: TensorType) -> tf.Tensor:
        """
        .. todo:: TODO: Document this.
        """

        return self.call(inputs, kwargs={"training": None, "full_cov": True})
