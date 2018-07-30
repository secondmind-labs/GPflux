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
    from_settings = False
    j = settings.numerics.jitter_level if from_settings else 1e-3
    Kmm = Kuu(feat, kern, jitter=j)  # L x M x M
    Kmn = Kuf(feat, kern, Xnew)  # M x L x N x P
    if full_cov:
        Knn = kern.K(Xnew, full_output_cov=True)  # N x P x N x P  or  P x N x N
    else:
        Knn = kern.Kdiag(Xnew, full_output_cov=True)  # N x P (x P)

    ## new code
    assert full_cov == False and full_output_cov == False
    Kmn = tf.reduce_sum(Kmn, axis=3, keepdims=True)  # M x L x N x 1
    Kmn = Kmn[:, 0, :, 0] / kern.num_patches  # M x N
    Knn = tf.reduce_sum(Knn, axis=[1, 2], keepdims=False)  # N
    Knn = Knn / kern.num_patches  ** 2 # N
    Kmm = Kmm[0, ...]  # M x M

    return base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white)

    # return independent_interdomain_conditional(Kmn, Kmm, Knn, f,
    #                                            full_cov=full_cov,
    #                                            full_output_cov=full_output_cov,
    #                                            q_sqrt=q_sqrt, white=white)



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
    logger.debug("conditional: IndexedInducingPatch -- IndexedConvKernel ")
    inducing_patches = feat.inducing_patches
    inducing_indices = feat.inducing_indices
    conv_kernel = kern.conv_kernel
    index_kernel = kern.index_kernel

    # jitter_mat = 1e-3 * tf.eye(len(feat), dtype=settings.float_type)
    Pmm = index_kernel.K(inducing_indices.Z)  # M x M
    Kmm2 = Kuu(inducing_patches, conv_kernel, jitter=0.0) # settings.numerics.jitter_level)  # L x M x M
    Kmm = Kmm2 * Pmm[None, ...]  # L x M x M
    # jitter_mat = settings.numerics.jitter_level * tf.eye(len(feat), dtype=settings.float_type)
    jitter_mat = 1e-3 * tf.eye(len(feat), dtype=settings.float_type)
    Kmm = Kmm + jitter_mat[None, ...]
    # Kmm = tf.Print(Kmm, [Kmm, Pmm, "conv kernel", Kmm2], summarize=3)

    # IJ: N x 2, cartesian product of output indices
    H_out = conv_kernel.Hout
    W_out = conv_kernel.Wout
    IJ = np.vstack([x.flatten() \
                    for x \
                    in np.meshgrid(np.arange(H_out), np.arange(W_out))]).T  # P x 2
    IJ = IJ.astype(float)

    Pmn = index_kernel.K(inducing_indices.Z, IJ)  # M x P
    Kmn = Kuf(inducing_patches, conv_kernel, Xnew)  # M x L x N x P
    Kmn = Kmn * Pmn[:, None, None, :]  # M x L x N x P

    full_output_cov = True
    if full_cov:
        Knn = conv_kernel.K(Xnew, full_output_cov=full_output_cov)  # N x P x N x P  or  P x N x N
    else:
        Knn = conv_kernel.Kdiag(Xnew, full_output_cov=full_output_cov)  # N x P (x P)

    if full_output_cov:
        Pnn = index_kernel.K(IJ)  # P x P
        if full_cov:
            Knn = Knn * Pnn[None, :, None, :]  # N x P x N x P
        else:
            Knn = Knn * Pnn[None, :, :]  # N x P x P
    else:
        Pnn = index_kernel.Kdiag(IJ)  # P
        if full_cov:
            Knn = Knn * Pnn[:, None, None]  # P x N x N
        else:
            Knn = Knn * Pnn[None, :]  # N x P



    ## new code
    assert full_cov == False and full_output_cov == True
    Kmn = tf.reduce_sum(Kmn, axis=3, keepdims=True)  # M x L x N x 1
    Kmn = Kmn[:, 0, :, 0] / conv_kernel.num_patches  # M x N
    Knn = tf.reduce_sum(Knn, axis=[1, 2], keepdims=False)  # N
    Knn = Knn / conv_kernel.num_patches  ** 2 # N
    Kmm = Kmm[0, ...]  # M x M

    return base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white)

    # m, v =  independent_interdomain_conditional(Kmn, Kmm, Knn, f,
    #                                            full_cov=full_cov,
    #                                            full_output_cov=full_output_cov,
    #                                            q_sqrt=q_sqrt, white=white)

    # m = tf.Print(m, ["before recude_sum shape posterior mean", tf.shape(m),\
    #                  "before recude_sum shape posterior variance", tf.shape(v)])

    # m = tf.reduce_sum(m, axis=1)
    # v = tf.reduce_sum(v, axis=1)

    # m = tf.Print(m, ["shape posterior mean", tf.shape(m),\
                    #  "shape posterior variance", tf.shape(v)])

    return m, v
