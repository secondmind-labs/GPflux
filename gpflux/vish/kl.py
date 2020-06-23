import numpy as np
import tensorflow as tf

import gpflow
from gpflow.base import TensorLike
from gpflow.kullback_leiblers import prior_kl

from gpflux.vish.inducing_variables import SphericalHarmonicInducingVariable
from gpflux.vish.kernels import ZonalKernel
from gpflux.vish.conditional import Lambda_diag_elements


@prior_kl.register(
    SphericalHarmonicInducingVariable, ZonalKernel, object, object
)
def _kl(harmonics, kernel, q_mu, q_sqrt, whiten=False):
    """
    Compute the KL divergence KL[q(u) || p(u)] with
        q(u) = N(q_mu, q_sqrt q_sqrt^T)
        p(u) = N(0, Kuu), where Kuu = Lambda
    """
    assert not whiten
    Kuu = tf.linalg.LinearOperatorDiag(Lambda_diag_elements(harmonics, kernel))
    return gauss_kl_for_K_diagonal_operator(q_mu, q_sqrt, Kuu)


def gauss_kl_for_K_diagonal_operator(
    q_mu: TensorLike, q_sqrt: TensorLike, K: tf.linalg.LinearOperator
):
    """
    Author: ST--
    Source: GPflow notebook on variational fourier features.

    :param q_mu: [M, K]
    :param q_sqrt: [K, M, M]
    :param K: is a LinearOperator that provides efficient methods
        for solve(), log_abs_determinant(), and trace()
    """
    # KL(N₀ || N₁) = ½ [tr(Σ₁⁻¹ Σ₀) + (μ₁ - μ₀)ᵀ Σ₁⁻¹ (μ₁ - μ₀) - k + ln(det(Σ₁)/det(Σ₀))]
    # N₀ = q; μ₀ = q_mu, Σ₀ = q_sqrt q_sqrtᵀ
    # N₁ = p; μ₁ = 0, Σ₁ = K
    # KL(q || p) =
    #     ½ [tr(K⁻¹ q_sqrt q_sqrtᵀA + q_muᵀ K⁻¹ q_mu - k + logdet(K) - logdet(q_sqrt q_sqrtᵀ)]
    # k = number of dimensions, if q_sqrt is m x m this is m²
    Kinv_q_mu = K.solve(q_mu)

    mahalanobis_term = tf.squeeze(tf.matmul(q_mu, Kinv_q_mu, transpose_a=True))

    # GPflow: q_sqrt is num_latent x N x N
    num_latent = tf.cast(tf.shape(q_mu)[1], gpflow.default_float())
    logdet_prior = num_latent * K.log_abs_determinant()

    product_of_dimensions__int = tf.reduce_prod(
        tf.shape(q_sqrt)[:-1]
    )  # dimensions are integers
    constant_term = tf.cast(product_of_dimensions__int, gpflow.default_float())

    Lq = tf.linalg.band_part(q_sqrt, -1, 0)  # force lower triangle
    logdet_q = tf.reduce_sum(tf.math.log(tf.square(tf.linalg.diag_part(Lq))))

    # S = tf.matmul(q_sqrt, q_sqrt, transpose_b=True)
    # trace_term = tf.trace(K.solve(S))
    trace_term = tf.squeeze(
        tf.reduce_sum(Lq * K.solve(Lq), axis=[-1, -2])
    )  # [O(N²) instead of O(N³)

    twoKL = (
        trace_term + mahalanobis_term - constant_term + logdet_prior - logdet_q
    )
    return 0.5 * twoKL
