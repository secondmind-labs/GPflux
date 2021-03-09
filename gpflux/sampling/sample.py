#  Copyright (C) PROWLER.io 2020 - All Rights Reserved
#  Unauthorised copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
"""
Sampling functions
"""
import abc
from typing import Callable, Optional, Union

import tensorflow as tf

from gpflow.base import TensorType
from gpflow.conditionals import conditional
from gpflow.config import default_float, default_jitter
from gpflow.covariances import Kuf, Kuu
from gpflow.inducing_variables import InducingVariables
from gpflow.kernels import Kernel
from gpflow.utilities import Dispatcher

from gpflux.sampling.kernel_with_feature_decomposition import KernelWithFeatureDecomposition
from gpflux.sampling.utils import compute_A_inv_b, draw_conditional_sample

efficient_sample = Dispatcher("efficient_sample")


class Sample(abc.ABC):
    """
    Represents a sampled function from a GP that can be evaluated at
    new locations within the support of the GP.

    Importantly, the same function draw (sample) is evaluated when calling it sequentially
    - we call this property consistency. Achieving consistency for vanilla GPs is costly
    and scales cubicly with the number of evaluation points.
    """

    @abc.abstractmethod
    def __call__(self, X: TensorType) -> tf.Tensor:
        """
        Returns f(X) for f ~ GP(0, k)

        :param X: [N, D]
        :return: Tensor of shape [N, P]
        """
        raise NotImplementedError

    def __add__(self, other: Union["Sample", Callable[[TensorType], TensorType]]) -> "Sample":
        """ Allows to sum two Sample instances or simply with another callable."""
        this = self.__call__

        class AddSample(Sample):
            def __call__(self, X: TensorType) -> tf.Tensor:
                return this(X) + other(X)

        return AddSample()


@efficient_sample.register(InducingVariables, Kernel, object)
def _efficient_sample_conditional_gaussian(
    inducing_variable: InducingVariables,
    kernel: Kernel,
    q_mu: tf.Tensor,
    *,
    q_sqrt: Optional[TensorType] = None,
    white: bool = False,
) -> Sample:
    """
    Most costly implementation for obtaining a consistent GP sample.
    However, this method can be used for any kernel.
    """
    assert not white, "Currently only white=False is supported"

    class SampleConditional(Sample):
        # N_old is 0 at first, we then start keeping track of past evaluation points.
        X = None  # [N_old, D]
        P = tf.shape(q_mu)[-1]  # num latent GPs
        f = tf.zeros((0, P), dtype=default_float())  # [N_old, P]

        def __call__(self, X_new: TensorType) -> tf.Tensor:
            N_old = tf.shape(self.f)[0]
            N_new = tf.shape(X_new)[0]

            if self.X is None:
                self.X = X_new
            else:
                self.X = tf.concat([self.X, X_new], axis=0)

            mean, cov = conditional(
                self.X,
                inducing_variable,
                kernel,
                q_mu,
                q_sqrt=q_sqrt,
                white=white,
                full_cov=True,
            )  # mean: [N_old+N_new, P], cov: [P, N_old+N_new, N_old+N_new]
            mean = tf.linalg.matrix_transpose(mean)  # [P, N_old+N_new]
            f_old = tf.linalg.matrix_transpose(self.f)  # [P, N_old]
            f_new = draw_conditional_sample(mean, cov, f_old)  # [P, N_new]
            f_new = tf.linalg.matrix_transpose(f_new)  # [N_new, P]
            self.f = tf.concat([self.f, f_new], axis=0)  # [N_old + N_new, P]

            tf.debugging.assert_equal(tf.shape(self.f), [N_old + N_new, self.P])
            tf.debugging.assert_equal(tf.shape(f_new), [N_new, self.P])

            return f_new

    return SampleConditional()


@efficient_sample.register(InducingVariables, KernelWithFeatureDecomposition, object)
def _efficient_sample_matheron_rule(
    inducing_variable: InducingVariables,
    kernel: KernelWithFeatureDecomposition,
    q_mu: tf.Tensor,
    *,
    q_sqrt: Optional[TensorType] = None,
    white: bool = False,
) -> Sample:
    """
    :param q_mu: [M, P]
    :param q_sqrt: [P, M, M]

    Implements the sampling rule from:
    "Efficiently Sampling Functions from Gaussian Process Posteriors"
    (Wilson et al, 2020).
    """
    # TODO(VD): allow for both white=True and False, currently only support False.
    # Remember u = Luu v, with Kuu = Luu Luu^T and p(v) = N(0, I)
    # so that p(u) = N(0, Luu Luu^T) = N(0, Kuu).
    assert not white, "Currently only white=False is supported"
    L = tf.shape(kernel.feature_coefficients)[0]  # num eigenfunctions  # noqa: F841

    prior_weights = tf.sqrt(kernel.feature_coefficients) * tf.random.normal(
        tf.shape(kernel.feature_coefficients), dtype=default_float()
    )  # [L, 1]

    M, P = tf.shape(q_mu)[0], tf.shape(q_mu)[1]  # num inducing, num output heads
    u_sample_noise = tf.matmul(
        q_sqrt,
        tf.random.normal((P, M, 1), dtype=default_float()),  # [P, M, M]  # [P, M, 1]
    )  # [P, M, 1]
    u_sample = q_mu + tf.linalg.matrix_transpose(u_sample_noise[..., 0])  # [M, P]
    Kmm = Kuu(inducing_variable, kernel, jitter=default_jitter())  # [M, M]
    tf.debugging.assert_equal(tf.shape(Kmm), [M, M])
    phi_Z = kernel.feature_functions(inducing_variable.Z)  # [M, L]
    weight_space_prior_Z = phi_Z @ prior_weights  # [M, 1]
    diff = u_sample - weight_space_prior_Z  # [M, P] -- using implicit broadcasting
    v = compute_A_inv_b(Kmm, diff)  # [M, P]
    tf.debugging.assert_equal(tf.shape(v), [M, P])

    class WilsonSample(Sample):
        def __call__(self, X: TensorType) -> tf.Tensor:
            """
            :param X: evaluation points [N, D]
            :return: function value of sample [N, P]
            """
            N = tf.shape(X)[0]
            phi_X = kernel.feature_functions(X)  # [N, L]
            weight_space_prior_X = phi_X @ prior_weights  # [N, 1]
            Knm = tf.linalg.matrix_transpose(Kuf(inducing_variable, kernel, X))  # [N, M]
            function_space_update_X = Knm @ v  # [N, P]

            tf.debugging.assert_equal(tf.shape(weight_space_prior_X), [N, 1])
            tf.debugging.assert_equal(tf.shape(function_space_update_X), [N, P])

            return weight_space_prior_X + function_space_update_X  # [N, P]

    return WilsonSample()
