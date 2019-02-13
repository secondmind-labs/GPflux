# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import gpflow
import tensorflow as tf
from gpflow import features, settings, decors, name_scope
from gpflow.dispatch import conditional

from .features import StochasticInvariantInducingPoints
from .kernels import StochasticInvariant

logger = settings.logger()


@name_scope()
def sub_conditional(Kmn, Kmm, fKnn, f, *, full_cov=False, q_sqrt=None, white=False):
    """
    Adapted version of base_conditional. Need this because things like Lm do not get memoised yet.
    :param Kmn: [M, N, C]
    :param Kmm: [M, M]
    :param Knn: [N, C, C]
    :param f: [M, R]
    :param full_cov: bool
    :param q_sqrt: None or [R, M, M] (lower triangular)
    :param white: bool
    :return: [N, R]  or [R, N, N]
    """
    logger.debug("Conditional: invgp/conditionals.py:sub_conditional")
    # Things to start
    N, M, C = tf.shape(Kmn)[1], tf.shape(Kmn)[0], tf.shape(Kmn)[2]
    if full_cov:
        raise NotImplementedError

    # compute kernel stuff
    num_func = tf.shape(f)[1]  # [R]
    Lm = tf.cholesky(Kmm)

    # Compute the projection matrix A
    dA = tf.matrix_triangular_solve(Lm, tf.reshape(Kmn, (M, N * C)), lower=True)  # [M, NC]
    sA = tf.reduce_mean(tf.reshape(dA, (M, N, C)), 2)  # M x N

    # compute the covariance due to the conditioning
    mKnn = tf.reduce_mean(fKnn, (1, 2))
    dKnn = tf.reshape(tf.matrix_diag_part(fKnn), (N * C,))
    sfvar, dfvar = [Knn - tf.reduce_sum(tf.square(A), 0) for A, Knn in zip([sA, dA], [mKnn, dKnn])]  # [R, N]
    sfvar, dfvar = [tf.tile(fvar[None, :], [num_func, 1]) for fvar in [sfvar, dfvar]]  # [R, N]

    # another backsubstitution in the unwhitened case
    if not white:
        sA, dA = [tf.matrix_triangular_solve(tf.transpose(Lm), A, lower=False) for A in [sA, dA]]

    # construct the conditional mean
    sfmean, dfmean = [tf.matmul(A, f, transpose_a=True) for A in [sA, dA]]

    if q_sqrt is not None:
        if q_sqrt.get_shape().ndims == 2:
            sLTA, dLTA = [A * tf.expand_dims(tf.transpose(q_sqrt), 2) for A in [sA, dA]]  # [R, M, N]
        elif q_sqrt.get_shape().ndims == 3:
            L = tf.matrix_band_part(q_sqrt, -1, 0)  # R x M x M
            sA_tiled, dA_tiled = [tf.tile(tf.expand_dims(A, 0), tf.stack([num_func, 1, 1])) for A in [sA, dA]]
            sLTA, dLTA = [tf.matmul(L, A_tiled, transpose_a=True) for A_tiled in [sA_tiled, dA_tiled]]  # [R, M, N]
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: {}".format(q_sqrt.get_shape().ndims))
        sfvar, dfvar = [fvar + tf.reduce_sum(tf.square(LTA), 1)
                        for fvar, LTA in zip([sfvar, dfvar], [sLTA, dLTA])]  # [R, N]

    if not full_cov:
        sfvar, dfvar = [tf.transpose(fvar) for fvar in [sfvar, dfvar]]  # [N, R]

    return sfmean, sfvar, dfmean, dfvar  # [N, R], [N, R], NC x R, NC x R


@conditional.register(object, StochasticInvariantInducingPoints, StochasticInvariant, object)
@decors.name_scope("conditional")
def _conditional(Xnew, feat, kern, f, *, full_cov=False, full_output_cov=False, q_sqrt=None, white=False):
    logger.debug("Conditional: invgp/conditionals.py StochasticInvariantInducingPoints StochasticInvariant")
    if full_output_cov:
        # full_output_cov is misused here
        raise gpflow.GPflowError("Can not handle `full_output_cov`.")
    if full_cov:
        raise gpflow.GPflowError("Can not handle `full_cov`.")
    Kuu = features.Kuu(feat, kern, jitter=settings.jitter)
    Kuf = features.Kuf(feat, kern, Xnew)  # [M, N, C]  where C is the "orbit minibatch" size
    # Kuf = tf.Print(Kuf, [tf.shape(Kuf), Kuf, tf.reduce_mean(Kuf, 2)], summarize=100)
    Xp = kern.orbit.get_orbit(Xnew)  # [N, C, D]
    Knn = kern.basekern.K(Xp)  # [N, C, C]

    est_fmu, full_fvar_mean, fmu, fvar = sub_conditional(Kuf, Kuu, Knn, f, q_sqrt=q_sqrt, white=white)
    # est_fmu, full_fvar_mean = base_conditional(tf.reduce_mean(Kuf, 2), Kuu, tf.reduce_mean(Knn, (1, 2)), f,
    #                                            full_cov=False, q_sqrt=q_sqrt, white=white)
    # fmu, fvar = base_conditional(tf.reshape(Kuf, (M, N * C)), Kuu, tf.reshape(tf.matrix_diag_part(Knn), (N * C,)), f,
    #                              full_cov=False, q_sqrt=q_sqrt, white=white)  # [NC, R]
    # [N, R], 𝔼[est[μ]] = μ -- predictive mean

    M, N, C = tf.shape(Kuf)[0], tf.shape(Kuf)[1], tf.shape(Kuf)[2]
    diag_fvar_mean = tf.reduce_mean(tf.reshape(fvar, (N, C, -1)), 1)  # [N, R]
    est_fvar = full_fvar_mean * kern.mw_full + diag_fvar_mean * kern.mw_diag
    # [N, R], 𝔼[est[σ²]] = σ² -- predictive variance

    diag_fmu2_mean = tf.reduce_mean(tf.reshape(fmu ** 2.0, (N, C, -1)), 1)
    est_fmu2_minus = est_fmu ** 2.0 * (kern.mw_full - 1.0) + diag_fmu2_mean * kern.mw_diag

    # [N, R], est[μ²] - est[μ]²
    est2 = est_fvar + est_fmu2_minus
    kern._parent._hack_est_fmu2_minus = est_fmu2_minus  # store est_fmu2_minus in model object

    # We return:
    # - est[μ],                      such that 𝔼[est[μ]] = μ
    # - est[σ²] + est[μ²] - est[μ]², such that 𝔼[est[σ²] + est[μ²] - est[μ]²] = σ² + μ² - 𝔼[est[μ]²]
    # This ensures that the Gaussian likelihood gives an unbiased estimate for the variational expectations.
    return est_fmu, est2
