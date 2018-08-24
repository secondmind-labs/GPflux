# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


from typing import List

import gpflow
import numpy as np
import tensorflow as tf

from gpflow import params_as_tensors, params_as_tensors_for, settings, Param
from gpflow.multioutput.kernels import Mok
from gpflow.kernels import Kern


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

        # IJ: N x 2, cartesian product of output indices
        H_out = self.img_size[0] - self.patch_size[0] + 1
        W_out = self.img_size[1] - self.patch_size[1] + 1
        IJ = np.vstack([x.flatten() \
                        for x \
                        in np.meshgrid(np.arange(H_out), np.arange(W_out))]).T  # P x 2
        self.IJ = IJ.astype(settings.float_type)  # (H_out * W_out) x 2 = P x 2

    @gpflow.decors.params_as_tensors
    def _get_patches(self, X):
        """
        Extracts patches from the images X.
        Patches are extracted separately for each of the colour channels.
        :param X: N x (W*H*C)
        :return: Patches (N, num_patches, patch_size)
        """
        # Roll the colour channel to the front, so it appears to `tf.extract_image_patches()` as separate images. Then
        # extract patches and reshape to have the first axis the same as the number of images. The separate patches will
        # then be in the second axis.
        castX = tf.transpose(tf.reshape(X, tf.stack([tf.shape(X)[0], -1, self.colour_channels])),
                             [0, 2, 1])
        castX = tf.cast(castX, tf.float32)
        castX_r = tf.reshape(castX, [-1, self.img_size[0], self.img_size[1], 1], name="rX")
        patches = tf.extract_image_patches(castX_r,
                                           [1, self.patch_size[0], self.patch_size[1], 1],
                                           [1, 1, 1, 1],
                                           [1, 1, 1, 1], "VALID")
        shp = tf.shape(patches)  # img x out_rows x out_cols
        patches_r = tf.reshape(patches, [tf.shape(X)[0], self.colour_channels * shp[1] * shp[2], shp[3]])
        patches_r = tf.cast(patches_r, settings.float_type)
        return patches_r

    @gpflow.name_scope("conv_kernel_K")
    @gpflow.params_as_tensors
    def K(self, X, X2=None, full_output_cov=False):
        """
        :param X: 2 dimensional, N x (W*H)
        :param X2: 2 dimensional, N x (W*H)
        """
        if self.pooling > 1 or self.with_indexing:
            raise NotImplementedError

        Xp = self._get_patches(X)  # N x P x wh
        N = tf.shape(Xp)[0]
        P = tf.shape(Xp)[1]

        if full_output_cov:
            Xp2 = tf.reshape(self._get_patches(X2), (-1, self.patch_len)) if X2 is not None else None
            N2 = tf.shape(Xp2)[0] if Xp2 is not None else N
            Xp = tf.reshape(Xp, (N * P, self.patch_len))  # NP x wh
            return tf.reshape(self.basekern.K(Xp, Xp2), (N, P, N2, P))  # N x P x N2 x P

        else:
            Xp_t = tf.transpose(Xp, [1, 0, 2])  # P x N x wh
            if X2 is not None:
                Xp2_t = tf.transpose(self._get_patches(X2), [1, 0, 2])  # P x N x wh
                return tf.map_fn(lambda Xs: self.basekern.K(*Xs),
                                 (Xp_t, Xp2_t),
                                 dtype=settings.float_type)  # P x N x N2
            else:
                return tf.map_fn(lambda X: self.basekern.K(X),
                                 Xp_t,
                                 dtype=settings.float_type)  # P x N x N


    @gpflow.name_scope("conv_kernel_K_diag")
    @gpflow.params_as_tensors
    def Kdiag(self, X, full_output_cov=False):
        Xp = self._get_patches(X)  # N x P x wh

        if full_output_cov:
            K = tf.map_fn(lambda x: self.basekern.K(x), Xp)  # N x P x P

            if self.with_indexing:
                Pij = self.index_kernel.K(self.IJ)  # P x P
                K = K * Pij[None, :, :]  # N x P x P

            if self.pooling > 1:
                K = tf.reshape(K, [-1, self.Hout, self.pooling, self.Wout, self.pooling,
                                       self.Hout, self.pooling, self.Wout, self.pooling])
                K = tf.reduce_sum(K, axis=[2, 4, 6, 8])
                K = tf.reshape(K, [-1, self.Hout * self.Wout, self.Hout * self.Wout])

            return K  # N x P' x P'

        else:
            if self.pooling > 1:
                raise NotImplementedError("Pooling is not implemented in ConvKernel.Kdiag() for "
                                          "`full_output_cov` False.")

            K = tf.map_fn(lambda x: self.basekern.Kdiag(x), Xp)  # N x P

            if self.with_indexing:
                Pij = self.index_kernel.Kdiag(self.IJ)  # P
                K = K * Pij[None, :]  # N x P

            return K

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
        :param X: 2 dimensional, N x (W*H)
        :param X2: 2 dimensional, N x (W*H)
        """
        raise NotImplementedError

    @gpflow.params_as_tensors
    def Kdiag(self, X, full_output_cov=False):

        K = super().Kdiag(X, full_output_cov)

        if full_output_cov:
            #  K is N x P x P
            WtW = self.weights[:, None] * self.weights[None, :]  # P x P
            K = K * WtW[None, :, :]  # N x P x P
            K = tf.reduce_sum(K, axis=[1, 2], keepdims=False)  # N
            K = K / self.num_outputs  ** 2 # N
            return K
        else:
            raise NotImplementedError
