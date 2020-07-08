import tensorflow as tf

from gpflow.conditionals import conditional

from gpflux.vish.inducing_variables import SphericalHarmonicInducingVariable
from gpflux.vish.kernels import ZonalKernel
from gpflux.vish.spherical_harmonics import num_harmonics
from gpflux.vish.misc import chain, l2norm


def Lambda_diag_elements(
    harmonics: SphericalHarmonicInducingVariable, kernel: ZonalKernel,
) -> tf.Tensor:
    r"""
    Returns the diagonal elements of Kuu as a flat array. This array corresponds to
    the concatentation of eigenvalues \lambda_i. The eigenvalue for each harmonic in
    a level is the same, so the eigenvalue is repeated the amount of harmonics in the level.

    Lambda = (\lambda_0, \lambda_1, ..., \lambda_1,
                \lambda_2, ..., \lambda_2, ..., lambda_L, ..., \lambda_L)

    :return: [M], where M = \sum_{level=0}^{num_level} N(dimension, level) and
        N(dim, level) give the number of harmonics per level, see
        `spherical_harmonics.num_harmonics`.
    """
    num = harmonics.num_levels()
    eigenvalues, degrees = kernel.get_first_num_eigenvalues_and_degrees(num)
    # eigenvalues = tf.squeeze(eigenvalues, axis=-1)  # [L]
    # degrees = tf.squeeze(degrees, axis=-1)  # [L]
    number_of_harmonics_per_level = [
        num_harmonics(kernel.dimension, level) for level in degrees
    ]  # [L]
    return tf.squeeze(chain(eigenvalues, number_of_harmonics_per_level))  # [M]


@conditional.register(object, SphericalHarmonicInducingVariable, ZonalKernel, object)
def _conditional(
    Xnew: tf.Tensor,
    harmonics: SphericalHarmonicInducingVariable,
    kernel: ZonalKernel,
    q_mu: tf.Tensor,
    *,
    full_cov=False,
    full_output_cov=False,
    q_sqrt=None,
    white=False,
):
    """
    :param q_mu: [M, K]
    :param q_sqrt: [K, M, M]
    """
    assert not white
    assert not full_output_cov
    assert not full_cov

    Xnew = tf.reshape(kernel.weight_variances ** 0.5, (1, -1)) * Xnew  # [N, D]
    Xr = l2norm(Xnew)  # [N, 1]
    Xnew = Xnew / Xr

    # Kuu = Lambda * Phi(X)
    Phi_Xnew = harmonics(Xnew)  # [M, N]

    # mean = Kuf Kuu^\inv q_mu
    # Kuf = Lambda * Phi(X)
    # Kuu = Lambda
    # => mean = Phi(X) q_mu
    mean = tf.matmul(Phi_Xnew, q_mu, transpose_a=True)  # [N, K]

    Lambda_diag = Lambda_diag_elements(harmonics, kernel)[..., None]  # [M, 1]

    if full_cov:
        raise NotImplementedError
        # Kff = kernel.K(Xnew)  # [N, N]
    else:
        Kff = kernel.K_diag(Xnew)[:, None]  # [N, 1]
        # Kuf Kuu^\inv Kuf
        # = Phi(X) \Lambda \Lambda^\inv \Lambda Phi(X)^T
        # = Phi(X) \Lambda Phi(X)^T
        Kuf_KuuInv_Kuf_diag = tf.reduce_sum(
            (Phi_Xnew ** 2) * Lambda_diag, axis=0, keepdims=True
        )  # [1, N]
        Kuf_KuuInv_Kuf_diag = tf.transpose(Kuf_KuuInv_Kuf_diag)  # [N, 1]

        # Kuf Kuu^\inv S Kuu^\inv Kuf
        # = \Phi \Lambda Lambda^\inv S Lambda^\inv \Lambda \Phi
        # = \Phi S \Phi
        # = \Phi L_S L_S^\top  \Phi
        Phi_q_sqrt = tf.matmul(Phi_Xnew[None], q_sqrt, transpose_a=True)  # [K, N, M]
        Phi_S_Phi_diag = tf.reduce_sum(Phi_q_sqrt ** 2, axis=2)  # [K, N]
        var = Kff - Kuf_KuuInv_Kuf_diag + tf.transpose(Phi_S_Phi_diag)  # [N, K]

    # Project to data-plane
    mean = Xr * mean
    var = (Xr ** 2) * var

    return mean, var
