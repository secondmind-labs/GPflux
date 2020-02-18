# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from typing import Optional

import gpflow
import tensorflow as tf

from gpflow import settings, params_as_tensors_for
from gpflow.dispatch import conditional, dispatch, sample_conditional
from gpflow.multioutput.conditionals import fully_correlated_conditional_repeat
from gpflow.conditionals import base_conditional, _sample_mvn
from gpflow.multioutput.features import debug_kuf, debug_kuu

from gpflux.convolution.convolution_kernel import ConvKernel, WeightedSumConvKernel
from gpflux.convolution.convolution_utils import K_image_inducing_patches
from gpflux.convolution.inducing_patch import InducingPatch, IndexedInducingPatch
from gpflux.types import TensorLike


# (Indexed)InducingPatch and ConvKernel


@dispatch(InducingPatch, ConvKernel, TensorLike)
@gpflow.name_scope("Kuf_InducingPatch_ConvKernel")
def Kuf(inducing_patch: InducingPatch,
        kern: ConvKernel,
        x_new: TensorLike) -> TensorLike:
    """
    Given inducing patches, kernel and new images calculates the kernel values between patches
    extracted from `x_new` and inducing patches.

    :param inducing_patch: inducing patch represented by an instance of InducingPatch
    :param kern: kernel, an instance of ConvKernel
    :param x_new: tensor of shape num_datapoints x (image width*image height*image channels)
    -> [N x WHC]
    :return: kernel matrix K_uf of shape num_inducing_patches x num_datapoints
    x num_patches_in_image_x -> [M x N x P]
    """
    debug_kuf(inducing_patch, kern)

    M = len(inducing_patch)
    N = tf.shape(x_new)[0]
    K_nm = K_image_inducing_patches(kern.basekern, kern.patch_handler, x_new,
                                    inducing_patch.patches)  # [N, P, M]
    K_mn = tf.transpose(K_nm, [2, 0, 1])  # [M, N, P]

    if kern.with_indexing:
        if not isinstance(inducing_patch, IndexedInducingPatch):
            raise ValueError("When kern is configured with `with_indexing` "
                             "a IndexedInducingPatch should be used")

        P_mn = kern.spatio_indices_kernel.K(inducing_patch.indices, kern.spatio_indices)  # [M, P]
        K_mn = K_mn * P_mn[:, None, :]  # [M, N, P]
    if kern.pooling > 1:
        raise RuntimeError('This needs to be implemented')
        # TODO: fix this - current implementation messes up shapes
        # K_mn = tf.reshape(K_mn,
        #                   [M, N, kern.config.Hout, kern.pooling, kern.config.Wout, kern.pooling])
        # K_mn = tf.reduce_sum(K_mn, axis=[3, 5])  # [M, N, patch_h, patch_w]
    return tf.reshape(K_mn, [M, N, kern.patch_handler.config.num_patches])  # [M, N, P]


@dispatch(InducingPatch, ConvKernel)
@gpflow.name_scope("Kuu_InducingPatch_ConvKernel")
def Kuu(inducing_patch: InducingPatch,
        kern: ConvKernel,
        *,
        jitter: float = 0.0):
    """
    Calculates kernel matrix K_mm between inducing patches.

    :param inducing_patch: inducing patch represented by an instance of InducingPatch.
    :param kern: kernel, an instance of ConvKernel
    :param jitter: jitter times diagonal matrix is added to K_mm
    :return: kernel matrix of shape [M, M]
    """
    debug_kuu(inducing_patch, kern, jitter)
    K_mm = kern.basekern.K(inducing_patch.patches)  # [M, M]
    jittermat = jitter * tf.eye(len(inducing_patch), dtype=K_mm.dtype)  # [M, M]

    if kern.with_indexing:
        P_mm = kern.spatio_indices_kernel.K(inducing_patch.indices)  # [M, M]
        K_mm = K_mm * P_mm

    return K_mm + jittermat  # [M, M]


@conditional.register(TensorLike, InducingPatch, ConvKernel, TensorLike)
def _conditional(x_new: TensorLike,
                 inducing_patch: InducingPatch,
                 kern: ConvKernel,
                 f: TensorLike,
                 *,
                 full_cov: bool = False,
                 full_output_cov: bool = False,
                 q_sqrt: TensorLike = None,
                 white: bool = False):
    """
    For explanation look at gpflow conditional docs.

    :param x_new: tensor of shape num_datapoints x (image width*image height*image channels)
    :param inducing_patch: inducing patch represented by an instance of InducingPatch
    :param kern: kernel, an instance of ConvKernel
    """

    assert (not full_cov) and (not full_output_cov)

    settings.logger().debug("conditional: InducingPatch -- ConvKernel")
    K_mm = Kuu(inducing_patch, kern, jitter=settings.numerics.jitter_level)  # [M, M]
    K_mn = Kuf(inducing_patch, kern, x_new)  # [M, N, P]
    K_nn = kern.Kdiag(x_new, full_output_cov=full_output_cov)  # [N, P]
    K_mm = K_mm  # [M, M]
    K_mn = K_mn  # [M, N, P]
    m, v = fully_correlated_conditional_repeat(K_mn, K_mm, K_nn, f,
                                               full_cov=full_cov,
                                               full_output_cov=full_output_cov,
                                               q_sqrt=q_sqrt,
                                               white=white)  # [L, N, P], [L, N, P]
    N = tf.shape(m)[1]
    m = tf.reshape(tf.transpose(m, [1, 2, 0]), [N, -1])  # [N, P*L]
    v = tf.reshape(tf.transpose(v, [1, 2, 0]), [N, -1])  # [N, P*L]
    return m, v


@sample_conditional.register(TensorLike, InducingPatch, ConvKernel, TensorLike)
@gpflow.name_scope("sample_conditional")
def _sample_conditional(x_new: TensorLike,
                        inducing_patch: InducingPatch,
                        kern: ConvKernel,
                        f: TensorLike,
                        *,
                        q_sqrt: TensorLike = None,
                        num_samples: int = 1,
                        white: bool = False,
                        full_cov: bool = False,
                        full_output_cov: bool = True):
    """
    Sampling from conditional distribution. For explanation look at gpflow conditional docs.
    """
    settings.logger().debug("sample conditional: InducingPatch, ConvKernel")
    mean, var = conditional(
        x_new,
        inducing_patch,
        kern,
        f,
        full_cov=full_cov,
        full_output_cov=False,
        q_sqrt=q_sqrt,
        white=white
    )  # [N, P], [N, P]
    sample = _sample_mvn(mean, var, cov_structure="diag", num_samples=num_samples)
    return sample, mean, var  # [N, P], [N, P], [N, P]


# (Indexed)InducingPatch and WeightedSumConvKernel

@dispatch(InducingPatch, WeightedSumConvKernel, TensorLike)
@gpflow.name_scope("Kuu_InducingPatch_WeightedSumConvKernel")
def Kuf(inducing_patch,
        kern,
        x_new):
    debug_kuf(inducing_patch, kern)
    K_uf_func = Kuf.dispatch(InducingPatch, ConvKernel, TensorLike)
    K_mn = K_uf_func(inducing_patch, kern, x_new)  # [M, N, P]
    with params_as_tensors_for(kern):
        weights = kern.weights  # [P]
        K_mn = tf.reduce_sum(K_mn * weights[None, None, :], axis=-1)
    return K_mn / kern.patch_handler.config.num_patches  # [M, N]


@conditional.register(TensorLike, InducingPatch, WeightedSumConvKernel, TensorLike)
def _conditional(x_new: TensorLike,
                 inducing_patch: InducingPatch,
                 kern: WeightedSumConvKernel,
                 f: TensorLike,
                 *,
                 full_cov: Optional[bool] = False,
                 full_output_cov: Optional[bool] = False,
                 q_sqrt: Optional[TensorLike] = None,
                 white: Optional[bool] = False):
    """
    For explanation look at gpflow conditional docs.

    :param x_new: tensor of shape num_datapoints x (image width*image height*image channels)
    :param inducing_patch: inducing patch represented by an instance of InducingPatch
    :param kern: kernel, an instance of ConvKernel
    """
    settings.logger().debug("Conditional: InducingPatch, WeightedSumConvKernel")
    Kmm = Kuu(inducing_patch, kern, jitter=settings.numerics.jitter_level)  # [M, M]
    Kmn = Kuf(inducing_patch, kern, x_new)  # [M, N]
    Knn = kern.K(x_new, full_output_cov=True) if full_cov else kern.Kdiag(x_new,
                                                                          full_output_cov=True)
    fmean, fvar = base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov,
                                   q_sqrt=q_sqrt, white=white)  # [N,R],  [R,N,N] or [N,R]
    return fmean, fvar
