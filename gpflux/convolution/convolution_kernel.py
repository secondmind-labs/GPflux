# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


from typing import List

import gpflow
import numpy as np
import tensorflow as tf
from gpflow import Param, params_as_tensors, settings
from gpflow.kernels import Kern
from gpflow.multioutput.kernels import Mok

from ..conv_square_dists import (diag_conv_dist_squared,
                                 full_conv_dist_squared,
                                 patchwise_conv_dist_squared)


class ConvKernel(Mok):
    """
    Multi-output kernel for a GP from images to images. Constructed by
    convolving a patch function g(.) over the input image.
    """

    def __init__(self,
                 basekern: gpflow.kernels.Kern,
                 img_size: List,
                 patch_size: List,
                 colour_channels: int = 1):
        """
        :param basekern: gpflow.Kernel that operates on the vectors of length w*h
        :param img_size: tuple, WxH
        :param patch_size: tuple, wxh
        """
        gpflow.kernels.Kern.__init__(self, np.prod(img_size))
        self.img_size = img_size
        self.patch_size = patch_size
        self.basekern = basekern
        self.colour_channels = colour_channels
        assert self.colour_channels == 1


    @gpflow.name_scope("conv_kernel_K")
    @gpflow.params_as_tensors
    def K(self, X, X2=None, full_output_cov=False):
        """
        :param X: 2 dimensional, Nx(W*H)
        :param X2: 2 dimensional, Nx(W*H)
        """
        H, W = self.img_size
        C = self.colour_channels
        assert C == 1

        lengthscales = self.basekern.lengthscales
        X = tf.reshape(X / lengthscales, (-1, H, W, C))
        X2 = X if X2 is None else tf.reshape(X2 / lengthscales, (-1, H, W, C))

        if full_output_cov:
            dist = full_conv_dist_squared(X, X2, self.patch_size)  # NxPxN2xP
        else:
            dist = patchwise_conv_dist_squared(X, X2, self.patch_size)  # PxNxN

        dist = tf.squeeze(dist)  # TODO: get rid of colour channel dimension. Assumes that only C is 1.
        return self.basekern.K_r2(dist)  # NxPxN2xP


    @gpflow.name_scope("conv_kernel_K_diag")
    @gpflow.params_as_tensors
    def Kdiag(self, X, full_output_cov=False):
        assert self.colour_channels == 1
        assert len(self.patch_size) == 2

        N = tf.shape(X)[0]
        C = self.colour_channels
        P = self.num_patches
        H, W = self.img_size

        X = tf.reshape(X, (-1, H, W, C))
        # TODO(@awav): works only for stationary non-ARD case
        # Oterwise, you need divide Ximg by lengthscales before passing it to dotconv (and use back_prop)
        dist = diag_conv_dist_squared(X, self.patch_size, back_prop=False)  # NxPxPx1
        dist = tf.squeeze(dist, axis=[3])

        dist /= self.basekern.lengthscales ** 2

        if not full_output_cov:
            Xr = tf.matrix_diag_part(dist)  # NxP

        return self.basekern.K_r2(dist)

    @property
    def patch_len(self):
        return np.prod(self.patch_size)

    @property
    def num_patches(self):
        return ((self.img_size[0] - self.patch_size[0] + 1) *
                (self.img_size[1] - self.patch_size[1] + 1) *
                self.colour_channels)

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
        return self.Hin - self.patch_size[0] + 1

    @property
    def Wout(self):
        return self.Win - self.patch_size[1] + 1

    @gpflow.autoflow((gpflow.settings.tf_float,))
    def compute_patches(self, X):
        return self._get_patches(X)


class IndexedConvKernel(Mok):

    def __init__(self, conv_kernel, index_kernel):
        Kern.__init__(self, conv_kernel.input_dim + index_kernel.input_dim)
        self.conv_kernel = conv_kernel
        self.index_kernel = index_kernel


class PoolingIndexedConvKernel(IndexedConvKernel):

    def __init__(self, conv_kernel, index_kernel):
        super().__init__(conv_kernel, index_kernel)
        self.weights = Param(np.ones([conv_kernel.num_outputs, 1]),
                             dtype=settings.float_type)
