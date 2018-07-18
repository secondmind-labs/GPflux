import gpflow
import numpy as np
import tensorflow as tf

from gpflow import settings
from gpflow.dispatch import conditional, dispatch
from gpflow.multioutput.conditionals import independent_interdomain_conditional
from gpflow.multioutput.features import debug_kuf, debug_kuu

from .convolution_kernel import ConvKernel, IndexedConvKernel
from .inducing_patch import InducingPatch, IndexedInducingPatch

logger = settings.logger()

# ----------------
# Kernel matrices
# ----------------

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
    jittermat = jitter * tf.eye(len(feat), dtype=gpflow.settings.dtypes.float_type)  # M x M
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
    logger.debug("conditional: InducingPatch -- ConvKernel")
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
    logger.debug("conditional: InducingPatch -- ConvKernel")
    inducing_patches = feat.inducing_patches
    inducing_indices = feat.inducing_indices
    conv_kernel = kern.conv_kernel
    index_kernel = kern.index_kernel

    Pmm = index_kernel.K(inducing_indices)  # M x M
    Kmm = Kuu(inducing_patches, conv_kernel, jitter=settings.numerics.jitter_level)  # L x M x M
    Kmm = Kmm * Pmm[None, ...]  # L x M x M

    # IJ: N x 2, cartesian product of output indices
    W = 28
    IJ = np.vstack([x.flatten() for x in np.meshgrid(np.arange(W), np.arange(W))]).T  # P x 2

    Pmn = index_kernel.K(inducing_indices, IJ)  # M x P
    Kmn = Kuf(feat, kern, Xnew)  # M x L x N x P
    Kmn = Kmn * Pmn[:, None, None, :]  # M x L x N x P

    if full_cov:
        Knn = kern.K(Xnew, full_output_cov=full_output_cov)  # N x P x N x P  or  P x N x N
    else:
        Knn = kern.Kdiag(Xnew, full_output_cov=full_output_cov)  # N x P (x P)

    if full_output_cov:
        Pnn = index_kernel.K(IJ)  # P x P
        if full_cov:
            Pnn = Pnn[None, :, None, :]
        else:
            Pnn = Pnn[None, :, :]
    else:
        Pnn = index_kernel.Kdiag(IJ)  # P
        if full_cov:
            Pnn = Pnn[:, None, None]
        else:
            Pnn = Pnn[None, :]
    
    Knn = Knn * Pnn

    return independent_interdomain_conditional(Kmn, Kmm, Knn, f,
                                               full_cov=full_cov,
                                               full_output_cov=full_output_cov,
                                               q_sqrt=q_sqrt, white=white)