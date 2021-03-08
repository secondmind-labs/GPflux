#  Copyright (C) PROWLER.io 2020 - All Rights Reserved
#  Unauthorised copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
import tensorflow as tf

from gpflow.base import TensorType
from gpflow.conditionals.util import sample_mvn
from gpflow.config import default_float, default_jitter


def draw_conditional_sample(mean: TensorType, cov: TensorType, f_old: TensorType) -> tf.Tensor:
    r"""
    Draws a sample \tilde{f_new} from the conditional multivariate Gaussian p(f_new | f_old),
    where mean and cov are the mean and covariance matrix of the joint multivariate Gaussian
    over (f_old, f_new).

    :param mean: is a mean vector [..., D, N+M]
        2x1 block matrix of the following form, for each [..., D]
       |---------------|
       | mean(old) [N] |
       |---------------|
       | mean(new) [M] |
       |---------------|
    :param cov: is a covariance matrix [..., D, N+M, N+M]
        2x2 block matrix of the following form, for each [..., D]
        |----------------------------------------------|
        |  Cov(old, old) [N, N] | Cov(old, new) [N, M] |
        |----------------------------------------------|
        |  Cov(new, old) [M, N] | Cov(new, new) [M, M] |
        |----------------------------------------------|
    :param f_old: are the observations [..., D, N]

    :return: sample from p(f_new | f_old) of shape [..., D, M]
    """
    N, D = tf.shape(f_old)[-1], tf.shape(f_old)[-2]  # noqa: F841
    M = tf.shape(mean)[-1] - N
    cov_old = cov[..., :N, :N]  # [..., D, N, N]
    cov_new = cov[..., -M:, -M:]  # [..., D, M, M]
    cov_cross = cov[..., :N, -M:]  # [..., D, N, M]
    jitter_mat = default_jitter() * tf.eye(N, dtype=default_float())
    L_old = tf.linalg.cholesky(cov_old + jitter_mat)  # [..., D, N, N]
    A = tf.linalg.triangular_solve(L_old, cov_cross, lower=True)  # [..., D, N, M]
    var_new = cov_new - tf.matmul(A, A, transpose_a=True)  # [..., D, M, M]
    mean_new = mean[..., -M:]  # [..., D, M]
    mean_old = mean[..., :N]  # [..., D, N]
    mean_old_diff = (f_old - mean_old)[..., None]  # [..., D, N, 1]
    AM = tf.linalg.triangular_solve(L_old, mean_old_diff)  # [..., D, N, 1]
    mean_new = mean_new + (tf.matmul(A, AM, transpose_a=True)[..., 0])  # [..., D, M]
    return sample_mvn(mean_new, var_new, full_cov=True)


def compute_A_inv_b(A: TensorType, b: TensorType) -> tf.Tensor:
    r"""
    Computes `A^{-1} b` using the Cholesky of `A` instead of the explicit inverse,
    as this is often numerically more stable.

    :param A: [..., M, M], p.s.d matrix
    :param b: [..., M, D]

    :return: [..., M, D]
    """
    # A = L L^T
    L = tf.linalg.cholesky(A)
    # A^{-1} = L^{-T} L^{-1}
    L_inv_b = tf.linalg.triangular_solve(L, b)
    A_inv_b = tf.linalg.triangular_solve(L, L_inv_b, adjoint=True)  # adjoint = transpose
    return A_inv_b
