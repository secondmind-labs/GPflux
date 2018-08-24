# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


from typing import List

import gpflow
import numpy as np
import tensorflow as tf
from gpflow import Param, params_as_tensors, params_as_tensors_for, settings
from gpflow.kernels import Kern
from gpflow.multioutput.kernels import Mok

from ..conv_square_dists import (diag_conv_square_dist, full_conv_square_dist,
                                 patchwise_conv_square_dist)


class ConvKernel(Mok):
    """
    Multi-output kernel for a GP from images to images. Constructed by
    convolving a patch function g(.) over the input image.
    """

    def __init__(self,
                 basekern: gpflow.kernels.Kern,
                 img_size: List,
                 patch_size: List,
                 pooling: int = 1,
                 with_indexing: bool = False,
                 colour_channels: int = 1):
        """
        :param basekern: gpflow.Kernel that operates on the vectors of length w*h
        :param img_size: tuple, (WxH)
        :param patch_size: tuple, (WxH)
        """
        gpflow.kernels.Kern.__init__(self, np.prod(img_size))
        self.img_size = img_size
        self.patch_size = patch_size
        self.basekern = basekern
        self.colour_channels = colour_channels
        assert self.colour_channels == 1

        self.pooling = pooling
        self.with_indexing = with_indexing
        if self.with_indexing:
            self._setup_indices()
            self.index_kernel = gpflow.kernels.RBF(len(img_size), lengthscales=3.0)

    def _setup_indices(self):
        # IJ: Nx2, cartesian product of output indices
        H_out = self.img_size[0] - self.patch_size[0] + 1
        W_out = self.img_size[1] - self.patch_size[1] + 1
        IJ = np.vstack([x.flatten() for x in np.meshgrid(np.arange(H_out), np.arange(W_out))]).T  # Px2
        self.IJ = IJ.astype(settings.float_type)  # (H_out * W_out)x2 = Px2

    @gpflow.name_scope("conv_kernel_K")
    @gpflow.params_as_tensors
    def K(self, X, X2=None, full_output_cov=False):
        """
        :param X: 2 dimensional, Nx(W*H)
        :param X2: 2 dimensional, Nx(W*H)
        """
        if self.pooling > 1 or self.with_indexing:
            raise NotImplementedError

        H, W = self.img_size
        C = self.colour_channels
        assert self.basekern.ARD == False
        assert C == 1

        X = tf.reshape(X, (-1, H, W, C))
        X2 = X if X2 is None else tf.reshape(X2, (-1, H, W, C))

        if full_output_cov:
            dist = full_conv_square_dist(X, X2, self.patch_size)  # NxPxN2xPxC
            dist = tf.squeeze(dist, axis=[4])  # TODO: get rid of colour channel dimension; it assumes that C is 1.
        else:
            dist = patchwise_conv_square_dist(X, X2, self.patch_size)  # PxNxNxC
            dist = tf.squeeze(dist, axis=[3])  # TODO: get rid of colour channel dimension; it assumes that C is 1.

        lengthscales = self.basekern.lengthscales
        dist /= lengthscales ** 2
        return self.basekern.K_r2(dist)  # NxPxN2xP


    @gpflow.name_scope("conv_kernel_K_diag")
    @gpflow.params_as_tensors
    def Kdiag(self, X, full_output_cov=False):
        H, W = self.img_size
        C = self.colour_channels

        X = tf.reshape(X, (-1, H, W, C))
        dist = diag_conv_square_dist(X, self.patch_size, back_prop=False)  # NxPxPx1
        dist = tf.squeeze(dist, axis=[3]) # TODO: get rid of colour channel dimension; it assumes that C is 1.
        dist /= self.basekern.lengthscales ** 2  # Dividing after computing distances
                                                 # helps to avoid unnecessary backpropagation.

        if full_output_cov:
            K = self.basekern.K_r2(dist)  # NxPxP
            if self.with_indexing:
                Pij = self.index_kernel.K(self.IJ)  # PxP
                K = K * Pij[None, :, :]  # NxPxP

            if self.pooling > 1:
                HpWp = self.Hout, self.pooling, self.Wout, self.pooling
                K = tf.reshape(K, [-1, *HpWp, *HpWp])
                K = tf.reduce_sum(K, axis=[2, 4, 6, 8])
                HW = self.Hout * self.Wout
                K = tf.reshape(K, [-1, HW, HW])
            return K  # NxP'xP'

        if self.pooling > 1 or self.with_indexing:
            raise NotImplementedError

        dist_diag = tf.matrix_diag_part(dist)  # NxP
        return self.basekern.K_r2(dist_diag)


    @property
    def patch_len(self):
        return np.prod(self.patch_size)

    @property
    def num_patches(self):
        # TODO(vincent): allow for C>1
        return self.Hout * self.Wout * self.colour_channels

    @property
    def num_outputs(self):
        return self.num_patches

    @property
    def Hin(self):
        return self.img_size[0]

    @property
    def Win(self):
        return self.img_size[1]

    @property
    def Hout(self):
        Hout = self.Hin - self.patch_size[0] + 1
        assert Hout % self.pooling == 0
        return Hout // self.pooling

    @property
    def Wout(self):
        Wout = self.Win - self.patch_size[1] + 1
        assert Wout % self.pooling == 0
        return Wout // self.pooling

    @gpflow.autoflow((gpflow.settings.tf_float,))
    def compute_patches(self, X):
        return self._get_patches(X)


class WeightedSum_ConvKernel(ConvKernel):

    def __init__(self,
                 basekern: gpflow.kernels.Kern,
                 img_size: List,
                 patch_size: List,
                 pooling: int = 1,
                 with_indexing: bool = False,
                 with_weights: bool = False,
                 colour_channels: int = 1):

        super().__init__(basekern,
                         img_size,
                         patch_size,
                         pooling,
                         with_indexing,
                         colour_channels)

        self.with_weights = with_weights

        weights = np.ones([self.num_outputs], dtype=settings.float_type)  # P
        if with_weights:
            self.weights = Param(weights)  # P
        else:
            self.weights = weights  # P

    @gpflow.params_as_tensors
    def K(self, X, X2=None, full_output_cov=False):
        """
        :param X: 2 dimensional, Nx(W*H)
        :param X2: 2 dimensional, Nx(W*H)
        """
        raise NotImplementedError

    @gpflow.params_as_tensors
    def Kdiag(self, X, full_output_cov=False):

        K = super().Kdiag(X, full_output_cov)

        if full_output_cov:
            #  K is NxPxP
            WtW = self.weights[:, None] * self.weights[None, :]  # PxP
            K = K * WtW[None, :, :]  # NxPxP
            K = tf.reduce_sum(K, axis=[1, 2], keepdims=False)  # N
            K = K / self.num_outputs  ** 2 # N
            return K
        else:
            raise NotImplementedError
