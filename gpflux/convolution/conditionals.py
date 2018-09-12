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

from .convolution_kernel import ConvKernel, WeightedSumConvolutional
from .inducing_patch import InducingPatch, IndexedInducingPatch
from ..conv_square_dists import image_patch_conv_square_dist


# -------------------------------------
# (Indexed)InducingPatch and ConvKernel
# -------------------------------------

@dispatch(InducingPatch, ConvKernel, object)
@gpflow.name_scope("Kuf_InducingPatch_Convolutional")
def Kuf(feat, kern, Xnew):
    """
    :param Xnew: NxWH
    :return:  MxLxNxP
    """
    debug_kuf(feat, kern)

    M = len(feat)
    N = tf.shape(Xnew)[0]

    Knm = kern.basekern.K_image_inducing_patches(Xnew, feat.Z)  # [N, P, M]
    Kmn = tf.transpose(Knm, [2, 0, 1])  # MxNxP

    if kern.with_indexing:
        if not isinstance(feat, IndexedInducingPatch):
            raise ValueError("When kern is configured with "
                             "`with_indexing` a IndexedInducingPatch "
                             "should be used")

        Pmn = kern.index_kernel.K(feat.indices, kern.IJ)  # MxP
        Kmn = Kmn * Pmn[:, None, :]  # MxNxP

    if kern.pooling > 1:
        Kmn = tf.reshape(Kmn, [M, N, kern.Hout, kern.pooling, kern.Wout, kern.pooling])
        Kmn = tf.reduce_sum(Kmn, axis=[3, 5])  # MxNxP'

    return tf.reshape(Kmn, [M, 1, N, kern.num_patches])  # MxL/1xNxP  TODO: add L


@dispatch(InducingPatch, ConvKernel)
@gpflow.name_scope("Kuu_InducingPatch_Convolutional")
def Kuu(feat, kern, *, jitter=0.0):
    debug_kuu(feat, kern, jitter)
    Kmm = kern.basekern.K(feat.Z)  # MxM
    jittermat = jitter * tf.eye(len(feat), dtype=Kmm.dtype)  # MxM

    if kern.with_indexing:
        Pmm = kern.index_kernel.K(feat.indices)  # MxM
        Kmm = Kmm * Pmm

    return (Kmm + jittermat)[None, :, :]  # L/1xMxM  TODO: add L


@conditional.register(object, InducingPatch, ConvKernel, object)
def _conditional(Xnew, feat, kern, f, *, full_cov=False, full_output_cov=False, q_sqrt=None, white=False):
    """
    :param Xnew: NxD
    :param f: MxL
    :param full_cov:
    :param full_cov_output:
    :param q_sqrt: LxM  or LxMxM
    :param white:
    :return:
    """
    settings.logger().debug("conditional: InducingPatch -- ConvKernel")
    Kmm = Kuu(feat, kern, jitter=settings.numerics.jitter_level)  # LxMxM
    Kmn = Kuf(feat, kern, Xnew)  # MxLxNxP
    if full_cov:
        Knn = kern.K(Xnew, full_output_cov=full_output_cov)  # NxPxNxP  or  PxNxN
    else:
        Knn = kern.Kdiag(Xnew, full_output_cov=full_output_cov)  # NxP (x P)

    return independent_interdomain_conditional(Kmn, Kmm, Knn, f,
                                               full_cov=full_cov,
                                               full_output_cov=full_output_cov,
                                               q_sqrt=q_sqrt, white=white)


@sample_conditional.register(object, InducingPatch, ConvKernel, object)
@gpflow.name_scope("sample_conditional")
def _sample_conditional(Xnew, feat, kern, f, *, q_sqrt=None, white=False, **kwargs):
    settings.logger().debug("sample conditional: InducingPatch, ConvKernel")
    mean, var = conditional(Xnew, feat, kern, f, full_cov=False, full_output_cov=True, q_sqrt=q_sqrt, white=white)  # NxP, NxPxP
    sample = _sample_mvn(mean, var, cov_structure="full")
    return sample

# -------------------------------------------------
# (Indexed)InducingPatch and WeightedSumConvolutional
# -------------------------------------------------

@dispatch(InducingPatch, WeightedSumConvolutional, object)
@gpflow.name_scope("Kuu_InducingPatch_WeightedSumConvolutional")
def Kuf(feat, kern, Xnew):
    debug_kuf(feat, kern)
    Kuf_func = Kuf.dispatch(InducingPatch, ConvKernel, object)
    Kmn = Kuf_func(feat, kern, Xnew)[:, 0, :, :]  # MxNxP
    Kmn = tf.einsum("mnp,p->mn", Kmn, kern.weights)
    return Kmn / kern.num_patches  # MxN


@dispatch(InducingPatch, WeightedSumConvolutional)
@gpflow.name_scope("Kuu_InducingPatch_WeightedSumConvolutional")
def Kuu(feat, kern, *, jitter=0.0):
    debug_kuu(feat, kern, jitter)
    Kuu_func = Kuu.dispatch(InducingPatch, ConvKernel)
    Kmm = Kuu_func(feat, kern, jitter=jitter)
    return Kmm[0, ...]  # MxM


@conditional.register(object, InducingPatch, WeightedSumConvolutional, object)
def _conditional(Xnew, feat, kern, f, *, full_cov=False, full_output_cov=False, q_sqrt=None, white=False):
    """
    :param Xnew: NxD
    :param f: MxL
    :param full_cov:
    :param full_cov_output:
    :param q_sqrt: LxM  or LxMxM
    :param white:
    :return:
    """
    settings.logger().debug("Conditional: InducingPatch, WeightedSumConvolutional")
    Kmm = Kuu(feat, kern, jitter=settings.numerics.jitter_level)  # MxM
    Kmn = Kuf(feat, kern, Xnew)  # MxN
    Knn = kern.K(Xnew, full_output_cov=True) if full_cov else kern.Kdiag(Xnew, full_output_cov=True)

    fmean, fvar = base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov,
                                   q_sqrt=q_sqrt, white=white)  # NxR,  RxNxN or NxR
    return fmean, fvar

