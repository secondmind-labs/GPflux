# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


import gpflow
import numpy as np
import tensorflow as tf

from gpflow import settings
from gpflow.dispatch import conditional, dispatch
from gpflow.multioutput.conditionals import independent_interdomain_conditional
from gpflow.conditionals import base_conditional
from gpflow.multioutput.features import debug_kuf, debug_kuu

from .convolution_kernel import ConvKernel, IndexedConvKernel
from .inducing_patch import InducingPatch, IndexedInducingPatch


@dispatch(InducingPatch, ConvKernel, object)
def Kuf(feat, kern, Xnew):
    """
    :param Xnew: N x WH
    :return:  M x L x N x P
    """
    debug_kuf(feat, kern)
    Xp = kern._get_patches(Xnew)  # N x P x wh
    N, P = tf.shape(Xp)[0], tf.shape(Xp)[1]
    Kmn = kern.basekern.K(feat.Z, tf.reshape(Xp, (N * P, -1)))  # M x NP
    return tf.reshape(Kmn, (len(feat), 1, N, P))  # M x L/1 x N x P  TODO: add L


@dispatch(InducingPatch, ConvKernel)
def Kuu(feat, kern, *, jitter=0.0):
    debug_kuu(feat, kern, jitter)
    Kmm = kern.basekern.K(feat.Z)  # M x M
    jittermat = jitter * tf.eye(len(feat), dtype=settings.dtypes.float_type)  # M x M
    return (Kmm + jittermat)[None, :, :]  # L/1 x M x M  TODO: add L


# ------------
# Condtitional
# ------------

@conditional.register(object, InducingPatch, ConvKernel, object)
def _conditional(Xnew, feat, kern, f, *, full_cov=False, full_output_cov=False, q_sqrt=None, white=False):
    """
    :param Xnew: N x D
    :param f: M x L
    :param full_cov:
    :param full_cov_output:
    :param q_sqrt: L x M  or L x M x M
    :param white:
    :return:
    """
    settings.logger().debug("conditional: InducingPatch -- ConvKernel")
    Kmm = Kuu(feat, kern, jitter=settings.numerics.jitter_level)  # L x M x M
    Kmn = Kuf(feat, kern, Xnew)  # M x L x N x P
    if full_cov:
        Knn = kern.K(Xnew, full_output_cov=full_output_cov)  # N x P x N x P  or  P x N x N
    else:
        Knn = kern.Kdiag(Xnew, full_output_cov=full_output_cov)  # N x P (x P)

    return independent_interdomain_conditional(Kmn, Kmm, Knn, f,
                                               full_cov=full_cov,
                                               full_output_cov=full_output_cov,
                                               q_sqrt=q_sqrt, white=white)



@conditional.register(object, IndexedInducingPatch, IndexedConvKernel, object)
def _conditional(Xnew, feat, kern, f, *, full_cov=False, full_output_cov=False, q_sqrt=None, white=False):
    """
    :param Xnew: N x D
    :param f: M x L
    :param full_cov:
    :param full_cov_output:
    :param q_sqrt: L x M  or L x M x M
    :param white:
    :return:
    """
    settings.logger().debug("conditional: IndexedInducingPatch -- IndexedConvKernel ")

    Pmm = kern.index_kernel.K(feat.indices.Z)  # M x M
    Kmm2 = Kuu(feat.patches, kern.conv_kernel, jitter=settings.numerics.jitter_level) # L x M x M
    Kmm = Kmm2 * Pmm[None, ...]  # L x M x M

    # IJ: N x 2, cartesian product of output indices
    H_out = kern.conv_kernel.Hout
    W_out = kern.conv_kernel.Wout
    IJ = np.vstack([x.flatten() \
                    for x \
                    in np.meshgrid(np.arange(H_out), np.arange(W_out))]).T  # P x 2
    IJ = IJ.astype(settings.float_type)  # (H_out * W_out) x 2 = P x 2

    Pmn = kern.index_kernel.K(feat.indices.Z, IJ)  # M x P
    Kmn = Kuf(feat.patches, kern.conv_kernel, Xnew)  # M x L x N x P
    Kmn = Kmn * Pmn[:, None, None, :]  # M x L x N x P

    # Pnn = kern.index_kernel.K(IJ)  # P x P
    Pnn = kern.index_kernel.Kdiag(IJ)  # P
    if full_cov:
        # Knn = kern.conv_kernel.K(Xnew, full_output_cov=True)  # N x P x N x P
        # Knn = Knn * Pnn[None, :, None, :]  # N x P x N x P
        Knn = kern.conv_kernel.K(Xnew, full_output_cov=full_output_cov)  # P x N x N
    else:
        # Knn = kern.conv_kernel.Kdiag(Xnew, full_output_cov=True)  # N x P x P
        # Knn = Knn * Pnn[None, :, :]  # N x P x P
        Knn = kern.conv_kernel.Kdiag(Xnew, full_output_cov=full_output_cov)  # N x P
        Knn = Knn * Pnn[None, :]  # N x P

    m, v = independent_interdomain_conditional(Kmn, Kmm, Knn, f,
                                               full_cov=full_cov,
                                               full_output_cov=full_output_cov,
                                               q_sqrt=q_sqrt, white=white)
    return m, v  # N x P, N x P
