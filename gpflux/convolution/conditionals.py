# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


import gpflow
import numpy as np
import tensorflow as tf

from gpflow import settings
from gpflow.dispatch import conditional, dispatch, sample_conditional
from gpflow.multioutput.conditionals import independent_interdomain_conditional
from gpflow.conditionals import base_conditional, _sample_mvn
from gpflow.multioutput.features import debug_kuf, debug_kuu

from .convolution_kernel import ConvKernel, WeightedSum_ConvKernel
from .inducing_patch import InducingPatch, IndexedInducingPatch


# -------------------------------------
# (Indexed)InducingPatch and ConvKernel
# -------------------------------------

@dispatch(InducingPatch, ConvKernel, object)
@gpflow.name_scope("Kuf_InducingPatch_ConvKernel")
def Kuf(feat, kern, Xnew):
    """
    :param Xnew: N x WH
    :return:  M x L x N x P
    """
    debug_kuf(feat, kern)
    Xp = kern._get_patches(Xnew)  # N x P x wh
    N, P, M = tf.shape(Xp)[0], tf.shape(Xp)[1], len(feat)

    Kmn = kern.basekern.K(feat.Z, tf.reshape(Xp, (N * P, -1)))  # M x NP
    Kmn = tf.reshape(Kmn, [M, N, P])

    if kern.with_indexing:
        if not isinstance(feat, IndexedInducingPatch):
            raise ValueError("When kern is configured with "
                             "`with_indexing` a IndexedInducingPatch "
                             "should be used")

        Pmn = kern.index_kernel.K(feat.indices, kern.IJ)  # M x P
        Kmn = Kmn * Pmn[:, None, :]  # M x N x P

    if kern.pooling > 1:
        Kmn = tf.reshape(Kmn, [M, N, kern.Hout, kern.pooling, kern.Wout, kern.pooling])
        Kmn = tf.reduce_sum(Kmn, axis=[3, 5])  # M x N x P'

    return tf.reshape(Kmn, [M, 1, N, kern.num_patches])  # M x L/1 x N x P  TODO: add L


@dispatch(InducingPatch, ConvKernel)
@gpflow.name_scope("Kuu_InducingPatch_ConvKernel")
def Kuu(feat, kern, *, jitter=0.0):
    debug_kuu(feat, kern, jitter)
    Kmm = kern.basekern.K(feat.Z)  # M x M
    jittermat = jitter * tf.eye(len(feat), dtype=settings.dtypes.float_type)  # M x M

    if kern.with_indexing:
        Pmm = kern.index_kernel.K(feat.indices)  # M x M
        Kmm = Kmm * Pmm

    return (Kmm + jittermat)[None, :, :]  # L/1 x M x M  TODO: add L


@conditional.register(object, InducingPatch, ConvKernel, object)
def _conditional(Xnew, feat, kern, f, *, full_cov=False, full_output_cov=False, q_sqrt=None, white=False):
    """
    :param Xnew: N x D
    :param f: M x L
    :param full_cov:
    :param full_output_cov:
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


@sample_conditional.register(object, InducingPatch, ConvKernel, object)
@gpflow.name_scope("sample_conditional")
def _sample_conditional(Xnew, feat, kern, f, *, q_sqrt=None, white=False, **kwargs):
    settings.logger().debug("sample conditional: InducingPatch, ConvKernel")
    mean, var = conditional(Xnew, feat, kern, f, full_cov=False, full_output_cov=True, q_sqrt=q_sqrt, white=white)  # N x P, N x P x P
    sample = _sample_mvn(mean, var, cov_structure="full")
    return sample

# -------------------------------------------------
# (Indexed)InducingPatch and WeightedSum_ConvKernel
# -------------------------------------------------

@dispatch(InducingPatch, WeightedSum_ConvKernel, object)
@gpflow.name_scope("Kuu_InducingPatch_WeightedSum_ConvKernel")
def Kuf(feat, kern, Xnew):
    debug_kuf(feat, kern)
    Kuf_func = Kuf.dispatch(InducingPatch, ConvKernel, object)
    Kmn = Kuf_func(feat, kern, Xnew)[:, 0, :, :]  # M x N x P
    Kmn = tf.einsum("mnp,p->mn", Kmn, kern.weights)
    return Kmn / kern.num_patches  # M x N


@dispatch(InducingPatch, WeightedSum_ConvKernel)
@gpflow.name_scope("Kuu_InducingPatch_WeightedSum_ConvKernel")
def Kuu(feat, kern, *, jitter=0.0):
    debug_kuu(feat, kern, jitter)
    Kuu_func = Kuu.dispatch(InducingPatch, ConvKernel)
    Kmm = Kuu_func(feat, kern, jitter=jitter)
    return Kmm[0, ...]  # M x M


@conditional.register(object, InducingPatch, WeightedSum_ConvKernel, object)
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
    settings.logger().debug("Conditional: InducingPatch, WeightedSum_ConvKernel")
    Kmm = Kuu(feat, kern, jitter=settings.numerics.jitter_level)  # M x M
    Kmn = Kuf(feat, kern, Xnew)  # M x N
    Knn = kern.K(Xnew, full_output_cov=True) if full_cov else kern.Kdiag(Xnew, full_output_cov=True)

    fmean, fvar = base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov,
                                   q_sqrt=q_sqrt, white=white)  # N x R,  R x N x N or N x R
    return fmean, fvar

