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

    assert kern.colour_channels == 1
    assert kern.basekern.ARD == False
    C = kern.colour_channels
    H, W = kern.img_size
    lengthscales = kern.basekern.lengthscales
    Z = feat.Z
    M = len(feat)

    X = tf.reshape(Xnew, [-1, H, W, C])
    N = tf.shape(X)[0]
    dist = image_patch_conv_square_dist(X, Z, kern.patch_size)  # NxMxP
    dist = tf.check_numerics(dist, "NAN in dist", name="check_op_Kuf")
    dist /= lengthscales ** 2
    Kmn = kern.basekern.K_r2(dist)
    Kmn = tf.transpose(Kmn, [1, 0, 2])  # MxNxP

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
@gpflow.name_scope("Kuu_InducingPatch_ConvKernel")
def Kuu(feat, kern, *, jitter=0.0):
    debug_kuu(feat, kern, jitter)
    Kmm = kern.basekern.K(feat.Z)  # MxM
    jittermat = jitter * tf.eye(len(feat), dtype=Kmm.dtype)  # MxM

    # Kmm = tf.check_numerics(Kmm, "nan in Kuu 1")
    # Kmm = tf.Print(Kmm, ["Kmm1", Kmm])

    if kern.with_indexing:
        Pmm = kern.index_kernel.K(feat.indices)  # MxM
        # Pmm = tf.check_numerics(Pmm, "nan in Puu")
        Kmm = Kmm * Pmm

        # def _save_vals(p, k):
        #     np.savez("Kuu_values", Kmm=k, Pmm=p)
        #     return True

        # debug_op = tf.py_func(_save_vals, [Pmm, Kmm], [tf.bool])
        # with tf.control_dependencies(debug_op):
        #     Kmm = tf.identity(Kmm, name='out')
    # Kmm = tf.check_numerics(Kmm, "nan in Kuu 2")
    # Kmm = tf.Print(Kmm, ["Kmm2", Kmm])

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
# (Indexed)InducingPatch and WeightedSum_ConvKernel
# -------------------------------------------------

@dispatch(InducingPatch, WeightedSum_ConvKernel, object)
@gpflow.name_scope("Kuu_InducingPatch_WeightedSum_ConvKernel")
def Kuf(feat, kern, Xnew):
    debug_kuf(feat, kern)
    Kuf_func = Kuf.dispatch(InducingPatch, ConvKernel, object)
    Kmn = Kuf_func(feat, kern, Xnew)[:, 0, :, :]  # MxNxP
    Kmn = tf.einsum("mnp,p->mn", Kmn, kern.weights)
    return Kmn / kern.num_patches  # MxN


@dispatch(InducingPatch, WeightedSum_ConvKernel)
@gpflow.name_scope("Kuu_InducingPatch_WeightedSum_ConvKernel")
def Kuu(feat, kern, *, jitter=0.0):
    debug_kuu(feat, kern, jitter)
    Kuu_func = Kuu.dispatch(InducingPatch, ConvKernel)
    Kmm = Kuu_func(feat, kern, jitter=jitter)
    return Kmm[0, ...]  # MxM


@conditional.register(object, InducingPatch, WeightedSum_ConvKernel, object)
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
    settings.logger().debug("Conditional: InducingPatch, WeightedSum_ConvKernel")
    Kmm = Kuu(feat, kern, jitter=settings.numerics.jitter_level)  # MxM
    Kmn = Kuf(feat, kern, Xnew)  # MxN
    Knn = kern.K(Xnew, full_output_cov=True) if full_cov else kern.Kdiag(Xnew, full_output_cov=True)

    fmean, fvar = base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov,
                                   q_sqrt=q_sqrt, white=white)  # NxR,  RxNxN or NxR
    return fmean, fvar

