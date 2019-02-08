# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


import gpflow
import numpy as np
import tensorflow as tf

from gpflow import settings
from gpflow.dispatch import conditional, dispatch, sample_conditional
from gpflow.multioutput.conditionals import independent_interdomain_conditional, fully_correlated_conditional_repeat
from gpflow.conditionals import base_conditional, _sample_mvn
from gpflow.multioutput.features import debug_kuf, debug_kuu

from .convolution_kernel import K_image_inducing_patches, ConvKernel, WeightedSumConvKernel
from .inducing_patch import InducingPatch, IndexedInducingPatch
from ..conv_square_dists import image_patch_conv_square_dist


# -------------------------------------
# (Indexed)InducingPatch and ConvKernel
# -------------------------------------

@dispatch(InducingPatch, ConvKernel, object)
@gpflow.name_scope("Kuf_InducingPatch_ConvKernel")
def Kuf(feat, kern, Xnew):
    """
    :param Xnew: NxWH
    :return:  MxLxNxP
    """
    debug_kuf(feat, kern)

    M = len(feat)
    N = tf.shape(Xnew)[0]

    Knm = K_image_inducing_patches(kern.basekern, kern.patch_handler, Xnew, feat.Z)  # [N, P, M]
    Kmn = tf.transpose(Knm, [2, 0, 1])  # [M, N, P]

    if kern.with_indexing:
        if not isinstance(feat, IndexedInducingPatch):
            raise ValueError("When kern is configured with `with_indexing` "
                             "a IndexedInducingPatch should be used")

        Pmn = kern.spatio_indices_kernel.K(feat.indices, kern.spatio_indices)  # [M, P]
        Kmn = Kmn * Pmn[:, None, :]  # [M, N, P]

    if kern.pooling > 1:
        Kmn = tf.reshape(Kmn, [M, N, kern.Hout, kern.pooling, kern.Wout, kern.pooling])
        Kmn = tf.reduce_sum(Kmn, axis=[3, 5])  # [M, N, P']

    return tf.reshape(Kmn, [M, 1, N, kern.patch_handler.config.num_patches])  # [M, 1, N, P]


@dispatch(InducingPatch, ConvKernel)
@gpflow.name_scope("Kuu_InducingPatch_ConvKernel")
def Kuu(feat, kern, *, jitter=0.0):
    debug_kuu(feat, kern, jitter)
    Kmm = kern.basekern.K(feat.Z)  # [M, M]
    jittermat = jitter * tf.eye(len(feat), dtype=Kmm.dtype)  # [M, M]

    if kern.with_indexing:
        Pmm = kern.spatio_indices_kernel.K(feat.indices)  # [M, M]
        Kmm = Kmm * Pmm

    return (Kmm + jittermat)[None, :, :]  # [L|1, M, M]  TODO: add L


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
    assert (not full_cov) and (not full_output_cov)

    settings.logger().debug("conditional: InducingPatch -- ConvKernel")
    Kmm = Kuu(feat, kern, jitter=settings.numerics.jitter_level)  # LxMxM
    Kmn = Kuf(feat, kern, Xnew)  # MxLxNxP
    if full_cov:
        Knn = kern.K(Xnew, full_output_cov=full_output_cov)  # [N, P, N, P]  or  PxNxN
    else:
        Knn = kern.Kdiag(Xnew, full_output_cov=full_output_cov)  # NxP (x P)

    Kmm = Kmm[0]  # [M, M]
    Kmn = Kmn[:, 0, ...]  # [M, N, P]
    m, v = fully_correlated_conditional_repeat(Kmn, Kmm, Knn, f,
                                               full_cov=full_cov,
                                               full_output_cov=full_output_cov,
                                               q_sqrt=q_sqrt,
                                               white=white)  # [C, N, P], [C, N, P]
    N = tf.shape(m)[1]
    m = tf.reshape(tf.transpose(m, [1, 2, 0]), [N, -1])  # [N, P*C]
    v = tf.reshape(tf.transpose(v, [1, 2, 0]), [N, -1])  # [N, P*C]

    return m, v


@sample_conditional.register(object, InducingPatch, ConvKernel, object)
@gpflow.name_scope("sample_conditional")
def _sample_conditional(Xnew, feat, kern, f, *, q_sqrt=None, white=False, **kwargs):
    settings.logger().debug("sample conditional: InducingPatch, ConvKernel")
    mean, var = conditional(
        Xnew,
        feat,
        kern,
        f,
        full_cov=False,
        full_output_cov=False,
        q_sqrt=q_sqrt,
        white=white
    )  # NxP, NxP
    sample = _sample_mvn(mean, var, cov_structure="diag")
    return sample

# -------------------------------------------------
# (Indexed)InducingPatch and WeightedSumConvKernel
# -------------------------------------------------

@dispatch(InducingPatch, WeightedSumConvKernel, object)
@gpflow.name_scope("Kuu_InducingPatch_WeightedSumConvKernel")
def Kuf(feat, kern, Xnew):
    debug_kuf(feat, kern)
    Kuf_func = Kuf.dispatch(InducingPatch, ConvKernel, object)
    Kmn = Kuf_func(feat, kern, Xnew)[:, 0, :, :]  # [M, N, P]
    weights = kern.weights
    weights = tf.convert_to_tensor(weights) if isinstance(weights, np.ndarray) else weights
    Kmn = tf.einsum("mnp,p->mn", Kmn, weights)
    return Kmn / kern.patch_handler.config.num_patches # [M, N]


@dispatch(InducingPatch, WeightedSumConvKernel)
@gpflow.name_scope("Kuu_InducingPatch_WeightedSumConvKernel")
def Kuu(feat, kern, *, jitter=0.0):
    debug_kuu(feat, kern, jitter)
    Kuu_func = Kuu.dispatch(InducingPatch, ConvKernel)
    Kmm = Kuu_func(feat, kern, jitter=jitter)
    return Kmm[0, ...]  # MxM


@conditional.register(object, InducingPatch, WeightedSumConvKernel, object)
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
    settings.logger().debug("Conditional: InducingPatch, WeightedSumConvKernel")
    Kmm = Kuu(feat, kern, jitter=settings.numerics.jitter_level)  # MxM
    Kmn = Kuf(feat, kern, Xnew)  # MxN
    Knn = kern.K(Xnew, full_output_cov=True) if full_cov else kern.Kdiag(Xnew, full_output_cov=True)

    fmean, fvar = base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov,
                                   q_sqrt=q_sqrt, white=white)  # NxR,  RxNxN or NxR
    return fmean, fvar
