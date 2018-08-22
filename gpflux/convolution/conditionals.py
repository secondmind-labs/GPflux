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

from .convolution_kernel import ConvKernel, IndexedConvKernel, PoolingIndexedConvKernel
from .inducing_patch import InducingPatch, IndexedInducingPatch


@gpflow.name_scope("Kuf")
@dispatch(InducingPatch, ConvKernel, object)
def Kuf(feat, kern, Xnew):
    """
    :param Xnew: NxWH
    :return:  MxLxNxP
    """

    assert kern.colour_channels == 1
    C = self.colour_channels

    N = tf.shape(Xnew)[0]
    M = len(feat)
    P = kern.num_patches
    H, W = kern.Hin, kern.Win
    h, w = kern.patch_size
    strides = [1, 1, 1, 1]  # TODO(VD) add striding from kernel

    Xr = tf.reshape(Xnew, [N, H, W, C])
    ones_kernel = tf.ones((h, w, C, 1), dtype=Xr.dtype)
    XtX = tf.nn.conv2d(Xr ** 2, ones_kernel, strides, padding="VALID")
    # XtX = tf.reshape(XtX, [N, P])  # NxP

    # Z_filter = tf.transpose(tf.reshape(feat.Z, [M, h, w, 1]), [1, 2, 3, 0])  # hxwx1xM
    # # Z_filter = tf.cast(Z_filter, tf.float32)
    # XtZ = tf.nn.conv2d(Xr, Z_filter, pad, padding="VALID")  # NxPhxPwxM
    # XtZ = tf.reshape(tf.transpose(XtZ, [0, 3, 1, 2]), [N, M, P])  # NxMxP

    # ZtZ = tf.reduce_sum(feat.Z**2, axis=1)  # M
    # # ZtZ = tf.cast(ZtZ, tf.float32)

    # r = XtX[:, None, :] + ZtZ[None, :, None] - 2 * XtZ
    # # r = tf.cast(r, tf.float64)
    # K = kern.basekern.Kr(r)
    # return tf.transpose(K, [1, 0, 2])[:, None, ...]  # MxL/1xNxP

    # ZtZ = tf.nn.conv2d(Xr**2, tf.ones((5, 5, 1, 1)), [1, 1, 1, 1], padding="VALID") # NxHxWxO

    debug_kuf(feat, kern)
    Xp = kern._get_patches(Xnew)  # NxPxwh
    N, P = tf.shape(Xp)[0], tf.shape(Xp)[1]
    Kmn = kern.basekern.K(feat.Z, tf.reshape(Xp, (N * P, -1)))  # MxNP
    return tf.reshape(Kmn, (len(feat), 1, N, P))  # MxL/1xNxP  TODO: add L


@dispatch(InducingPatch, ConvKernel)
def Kuu(feat, kern, *, jitter=0.0):
    debug_kuu(feat, kern, jitter)
    Kmm = kern.basekern.K(feat.Z)  # MxM
    jittermat = jitter * tf.eye(len(feat), dtype=settings.dtypes.float_type)  # MxM
    return (Kmm + jittermat)[None, :, :]  # L/1xMxM  TODO: add L


# ------------
# Condtitional
# ------------

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


@conditional.register(object, IndexedInducingPatch, IndexedConvKernel, object)
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
    settings.logger().debug("conditional: IndexedInducingPatch -- IndexedConvKernel ")

    Pmm = kern.index_kernel.K(feat.indices.Z)  # MxM
    Kmm2 = Kuu(feat.patches, kern.conv_kernel, jitter=settings.numerics.jitter_level) # LxMxM
    Kmm = Kmm2 * Pmm[None, ...]  # LxMxM

    # IJ: Nx2, cartesian product of output indices
    H_out = kern.conv_kernel.Hout
    W_out = kern.conv_kernel.Wout
    IJ = np.vstack([x.flatten() \
                    for x\
                    in np.meshgrid(np.arange(H_out), np.arange(W_out))]).T  # Px2
    IJ = IJ.astype(settings.float_type)  # (H_out * W_out)x2 = Px2

    Pmn = kern.index_kernel.K(feat.indices.Z, IJ)  # MxP
    Kmn = Kuf(feat.patches, kern.conv_kernel, Xnew)  # MxLxNxP
    Kmn = Kmn * Pmn[:, None, None, :]  # MxLxNxP

    # Pnn = kern.index_kernel.K(IJ)  # PxP
    Pnn = kern.index_kernel.Kdiag(IJ)  # P
    if full_cov:
        # Knn = kern.conv_kernel.K(Xnew, full_output_cov=True)  # NxPxNxP
        # Knn = Knn * Pnn[None, :, None, :]  # NxPxNxP
        Knn = kern.conv_kernel.K(Xnew, full_output_cov=full_output_cov)  # PxNxN
    else:
        # Knn = kern.conv_kernel.Kdiag(Xnew, full_output_cov=True)  # NxPxP
        # Knn = Knn * Pnn[None, :, :]  # NxPxP
        Knn = kern.conv_kernel.Kdiag(Xnew, full_output_cov=full_output_cov)  # NxP
        Knn = Knn * Pnn[None, :]  # NxP

    m, v = independent_interdomain_conditional(Kmn, Kmm, Knn, f,
                                               full_cov=full_cov,
                                               full_output_cov=full_output_cov,
                                               q_sqrt=q_sqrt, white=white)
    return m, v  # NxP, NxP


@conditional.register(object, IndexedInducingPatch, PoolingIndexedConvKernel, object)
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
    # assert full_cov_output == False and full_cov == False

    settings.logger().debug("conditional: IndexedInducingPatch -- PullingIndexedConvKernel ")

    Pmm = kern.index_kernel.K(feat.indices.Z)  # MxM
    Kmm2 = Kuu(feat.patches, kern.conv_kernel, jitter=settings.numerics.jitter_level) # LxMxM
    Kmm = Kmm2 * Pmm[None, ...]  # LxMxM

    # IJ: Nx2, cartesian product of output indices
    H_out = kern.conv_kernel.Hout
    W_out = kern.conv_kernel.Wout
    IJ = np.vstack([x.flatten() \
                    for x\
                    in np.meshgrid(np.arange(H_out), np.arange(W_out))]).T  # Px2
    IJ = IJ.astype(settings.float_type)  # (H_out * W_out)x2 = Px2

    W = kern.weights  # Px1
    Pmn = kern.index_kernel.K(feat.indices.Z, IJ)  # MxP
    Pmn = Pmn * tf.transpose(W)  # MxP
    Kmn = Kuf(feat.patches, kern.conv_kernel, Xnew)  # MxLxNxP
    Kmn = Kmn * Pmn[:, None, None, :]  # MxLxNxP

    # Pnn = kern.index_kernel.Kdiag(IJ)  # P
    # if full_cov:
    #     # Knn = kern.conv_kernel.K(Xnew, full_output_cov=True)  # NxPxNxP
    #     # Knn = Knn * Pnn[None, :, None, :]  # NxPxNxP
    #     Knn = kern.conv_kernel.K(Xnew, full_output_cov=full_output_cov)  # PxNxN
    # else:
    WW = tf.matmul(W, W, transpose_b=True)  # PxP
    Pnn = kern.index_kernel.K(IJ) * WW  # PxP
    Knn = kern.conv_kernel.Kdiag(Xnew, full_output_cov=True)  # NxPxP
    Knn = Knn * Pnn[None, :, :]  # NxPxP

    Kmn = tf.reduce_sum(Kmn, axis=3, keepdims=True)  # MxLxNx1
    Kmn = Kmn[:, 0, :, 0] / kern.conv_kernel.num_patches  # MxN
    Knn = tf.reduce_sum(Knn, axis=[1, 2], keepdims=False)  # N
    Knn = Knn / kern.conv_kernel.num_patches  ** 2 # N
    Kmm = Kmm[0, ...]  # MxM

    return base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white)
