# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


import numpy as np
import tensorflow as tf

import gpflow
from gpflow import kernels, settings
from gpflow.multioutput.kernels import Mok

from gpflux.convolution.convolution_utils import (
    ExtractPatchHandler,
    ImagePatchConfig,
    ImageShape,
    K_image_symm,
    PatchHandler,
    PatchShape,
)


class ConvKernel(Mok):
    """
    Multi-output kernel for a GP from images to images. Constructed by
    convolving a patch function g(.) over the input image.
    """

    def __init__(self,
                 basekern: kernels.Kernel,
                 image_shape,
                 patch_shape,
                 pooling: int = 1,
                 with_indexing: bool = False,
                 patch_handler: PatchHandler = None):
        """
        :param basekern: gpflow.Kernel that operates on the vectors of length w*h
        """

        super().__init__(basekern.input_dim)

        self.config = ImagePatchConfig(image_shape, patch_shape, pooling=pooling)

        is_handler = isinstance(patch_handler, PatchHandler)
        self.patch_handler = ExtractPatchHandler(self.config) if not is_handler else patch_handler

        self.basekern = basekern
        self.with_indexing = with_indexing
        if self.with_indexing:
            self._setup_spatio_indices()
            self.spatio_indices_kernel = kernels.Matern52(2, lengthscales=.1)

    @gpflow.name_scope("convolutional_K")
    @gpflow.params_as_tensors
    def K(self, X, X2=None, full_output_cov=False):
        """
        :param X: 2 dimensional, Nx(W*H)
        :param X2: 2 dimensional, Nx(W*H)
        """
        raise NotImplementedError

    @gpflow.name_scope("convolutional_K_diag")
    @gpflow.params_as_tensors
    def Kdiag(self, X, full_output_cov=False):

        cfg = self.patch_handler.config
        K = K_image_symm(self.basekern, self.patch_handler, X, full_output_cov=full_output_cov)

        if self.with_indexing:
            if full_output_cov:
                Pij = self.spatio_indices_kernel.K(self.spatio_indices)  # [P, P]
                K = K * Pij[None, :, :]
            else:
                Pij = self.spatio_indices_kernel.Kdiag(self.spatio_indices)  # [P]
                K = K * Pij[None, :]

        if cfg.pooling > 1:
            if not full_output_cov:
                msg = "Pooling is not implemented in ConvKernel.Kdiag() for " \
                      "`full_output_cov` False."
                raise NotImplementedError(msg)
            HpWp = cfg.Hout, cfg.pooling, cfg.Wout, cfg.pooling
            K = tf.reshape(K, [-1, *HpWp, *HpWp])
            K = tf.reduce_sum(K, axis=[2, 4, 6, 8])
            HW = cfg.Hout * cfg.Wout
            K = tf.reshape(K, [-1, HW, HW])  # K is [N, P, P]

        return K

    def _setup_spatio_indices(self):
        cfg = self.patch_handler.config
        Hout = np.arange(cfg.Hout, dtype=settings.float_type)
        Wout = np.arange(cfg.Wout, dtype=settings.float_type)
        grid = np.meshgrid(Hout, Wout)
        spatio_indices = np.vstack([x.flatten() for x in grid]).T  # [P, 2]
        self.spatio_indices = spatio_indices

    @property
    def pooling(self):
        return self.patch_handler.config.pooling

    @property
    def patch_len(self):
        return np.prod(self.patch_handler.config.patch_shape)

    @property
    def num_outputs(self):
        return self.patch_handler.config.num_patches


class WeightedSumConvKernel(ConvKernel):
    def __init__(self,
                 basekern: kernels.Kernel,
                 image_shape: ImageShape,
                 patch_shape: PatchShape,
                 pooling: int = 1,
                 with_indexing: bool = False,
                 with_weights: bool = False,
                 patch_handler: PatchHandler = None):
        super().__init__(basekern, image_shape, patch_shape,
                         pooling=pooling, with_indexing=with_indexing,
                         patch_handler=patch_handler)

        self.with_weights = with_weights
        weights = np.ones([self.num_outputs], dtype=settings.float_type)  # [P]
        self.weights = gpflow.Param(weights, trainable=with_weights)

    @gpflow.params_as_tensors
    def K(self, X, X2=None, full_output_cov=False):
        """
        :param X: 2 dimensional, Nx(W*H)
        :param X2: 2 dimensional, Nx(W*H)
        """
        raise NotImplementedError

    @gpflow.params_as_tensors
    def Kdiag(self, X, full_output_cov=False):
        if not full_output_cov:
            raise NotImplementedError

        K = super().Kdiag(X, full_output_cov)

        #  K is NxPxP
        WtW = self.weights[:, None] * self.weights[None, :]  # PxP
        K = K * WtW[None, :, :]  # NxPxP
        K = tf.reduce_sum(K, axis=[1, 2], keepdims=False)  # N
        K = K / self.num_outputs ** 2  # N
        return K
