# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from abc import ABC, abstractmethod

from typing import List

import gpflow
import numpy as np
import tensorflow as tf
from gpflow import Param, params_as_tensors, params_as_tensors_for, settings
from gpflow.kernels import Kern
from gpflow.multioutput.kernels import Mok

from ..conv_square_dists import (diag_conv_square_dist, full_conv_square_dist,
                                 patchwise_conv_square_dist, diag_conv_inner_prod)


class ImageKernel(ABC):
    padding = 'VALID'
    strides = (1, 1, 1, 1)

    def __init__(self, image_size, patch_size):
        """
        :param image_size: [Height, Width, Num Color Channels] = [N, H, W, C]
        :param patch_size: [height, width] = [h, w]
        """
        super().__init__()
        if len(image_size) == 2:
            # TODO(VD) deal with color channel
            image_size = image_size + [1]

        self.image_size = image_size
        self.patch_size = patch_size
        self.colour_channels = self.image_size[-1]
        assert self.colour_channels == 1

    def _rehape_images(self, X):
        """ reshapes X [N, H*W*C] from being rank 2 to [N, H, W, C] """
        return tf.reshape(X, [-1, *self.image_size])

    def _image_patches_inner_product(self, X, **kwargs):
        """ 
        Returns the inner product between all patches in every image in `X`.
        `ret[n, p, p'] = Xn[p] Xn[p']^T` Xn is the n-th image and [q] the q-th patch
        :param X: Tensor containing image data [N, H*W*C]
        :return: Tensor [N, P, P]
        """
        X = self._rehape_images(X)
        inner_prod = diag_conv_inner_prod(X, self.patch_size, **kwargs)  # [N, P, P, 1]
        return tf.squeeze(inner_prod, axis=[3])  # [N, P, P]

    def _image_patches_squared_norm(self, X):
        """
        Returns the squared norm for every patch for every image in `X`.
        Corresponds to the diagonal elements of `_image_patches_inner_product`.
        `ret[n, p] = Xn[p] Xn[p]^T` Xn is the n-th image and [p] the p-th patch
        :param X: Tensor containing image data [N, H*W*C]
        :return: Tensor [N, P]
        """
        X = self._rehape_images(X)
        ones = tf.ones((*self.patch_size, self.colour_channels, 1), dtype=X.dtype)
        XpXpt = tf.nn.conv2d(X ** 2, ones, ImageKernel.strides, ImageKernel.padding)
        return tf.reshape(XpXpt, (tf.shape(X)[0], -1))  # [N, P]
    
    def _inducing_patches_squared_norm(self, Z):
        """
        Returns the squared norm of every row in `Z`.
        `ret[i] = Z[i] Z[i]^T`
        :param Z: Tensor, inducing patches [M, h*w]
        :return: Tensor [M]
        """
        return tf.reduce_sum(Z ** 2, axis=1)  # M

    def _image_patches_inducing_patches_inner_product(self, X, Z):
        """ 
        Returns the inner product between every patch and every inducing
        point in `Z` (for every image in `X`).
        `ret[n, p, m] = Xn[p] Zm^T` Xn is the n-th image and [q] the q-th patch,
        and Zm is the m-th inducing patch.
        :param X: Tensor containing image data [N, H*W*C]
        :param Z: Tensor containing inducing patches [N, h*w]
        :return: Tensor [N, P, M]
        """
        X = self._rehape_images(X)
        M, N = tf.shape(Z)[0], tf.shape(X)[0]
        Zr = tf.reshape(Z, (M, *self.patch_size, self.colour_channels))  # [M, h, w, C]
        Z_filter = tf.transpose(Zr, [1, 2, 3, 0])  # [h, w, C, M]
        XpZ = tf.nn.conv2d(X, Z_filter, ImageKernel.strides, ImageKernel.padding)  # [N, Ph, Pw, M]
        return tf.reshape(XpZ, (N, -1, M))  # [N, P, M]

    @abstractmethod
    def K_image_inducing_patches(self, X, Z):
        """ 
        Kernel between every patch in `X` and every inducing patch in `Z`.
        `ret[m,n,p] = k(Zm, Xn[p])`, where [p] operator selects the p-th patch.
        :param X: Tensor of image data, [N, H*W*C]
        :param Z: Tensor of inducing patches, [M, h*w]
        :return: covariance matrix [N, P, M]
        """
        raise NotImplementedError
    
    @abstractmethod
    def K_image(self, X, full_output_cov=False):
        """ 
        Kernel between every 2 patches in every image in `X`
        `ret[m,p,p'] = k(Xn[p], Xn[p'])`, where [p] operator selects the p-th patch.
        :param X: Tensor of image data, [N, H*W*C]
        :param full_output_cov: boolean
            If `False` only the diagonal elements are returned
        :return: covariance matrix [N, P, P] if `full_output_cov`=True, else [N, P].
        """
        raise NotImplementedError

    @gpflow.decors.autoflow((gpflow.settings.float_type, [None, None]))
    def compute_K_image(self, X):
        return self.K_image(X, full_output_cov=False)

    @gpflow.decors.autoflow((gpflow.settings.float_type, [None, None]))
    def compute_K_image_full_output_cov(self, X):
        return self.K_image(X, full_output_cov=True)

    @gpflow.decors.autoflow((gpflow.settings.float_type, [None, None]),
                            (gpflow.settings.float_type, [None, None]))
    def compute_K_image_inducing_patches(self, X, Z):
        return self.K_image_inducing_patches(X, Z)


class StationaryImageKernel(gpflow.kernels.Stationary, ImageKernel):

    def __init__(self, image_size=None, patch_size=None, **kwargs):
        input_dim = np.prod(patch_size)
        # TODO(VD): do this properly with super!! - requires changes in gpflow...
        gpflow.kernels.Stationary.__init__(self, input_dim, **kwargs)
        ImageKernel.__init__(self, image_size, patch_size)
        assert not self.ARD

    def _image_patches_square_dist(self, X, **map_kwargs):
        """
        Calculates the squared distance between every patch in each image of `X`
        ```
            ret[n,p,p'] = || Xn[p] - Xn[p'] ||^2 
                        = Xn[p]^T Xn[p] + Xn[p']^T Xn[p'] - 2 Xn[p]^T Xn[p'],
            where Xn is the n-th image and .[p] operator selects the p-th patch.
        ```
        :param X: Tensor of shape [N, H, W, C]
        :return: Tensor of shape [N, P, P].
        """

        Xp1tXp2 = self._image_patches_inner_product(X, **map_kwargs)  # [N, P, P]
        Xp_squared = self._image_patches_squared_norm(X)  # [N, P]
        return -2 * Xp1tXp2 + Xp_squared[:, :, None] + Xp_squared[:, None, :]  # [N, P, P]

    def _image_patches_inducing_patches_square_dist(self, X, Z):
        """
        Calculates the squared distance between every patch in each image of `X`
        ```
            ret[n,p,m] = || Xn[p] - Zm ||^2 
                        = Xn[p]^T Xn[p] + Zm^T Zm - 2 Xn[p]^T Zm
        ```
        and every inducing patch in `Z`.
        :param X: Tensor of shape [N, H, W, C]
        :param Z: Tensor of shape [N, h*w]
        :return: Tensor of shape [N, P, M].
        """
        Xp_squared = self._image_patches_squared_norm(X)  # [N, P]
        Zm_squared = self._inducing_patches_squared_norm(Z)  # M
        XptZm = self._image_patches_inducing_patches_inner_product(X, Z)  # [N, P, M]
        return Xp_squared[:, :, None] + Zm_squared[None, None, :] - 2 * XptZm

    @gpflow.decors.params_as_tensors
    def K_image_inducing_patches(self, X, Z):
        """ returns [N, P, M] """
        dist = self._image_patches_inducing_patches_square_dist(X, Z)  # [N, P, M]
        dist /= self.lengthscales ** 2  # Dividing after computing distances
                                        # helps to avoid unnecessary backpropagation.
        return self.K_r2(dist)  # [N, P, M]
    
    @gpflow.decors.params_as_tensors
    def K_image(self, X, full_output_cov=False):
        """ returns [N, P, P] if full_output_cov=True, else [N, P] """
        if full_output_cov:
            dist = self._image_patches_square_dist(X, back_prop=False)  # [N, P, P]
            dist /= self.lengthscales ** 2  # Dividing after computing distances
                                            # helps to avoid unnecessary backpropagation.
            return self.K_r2(dist)  # [N, P, P]
        else:
            P = ((self.image_size[0] - self.patch_size[0] + 1) * 
                 (self.image_size[1] - self.patch_size[1] + 1))
            return self.variance * tf.ones([tf.shape(X)[0], P], dtype=X.dtype)  # [N, P]


class RBFImageKernel(StationaryImageKernel, gpflow.kernels.RBF):
    pass


class Matern12ImageKernel(StationaryImageKernel, gpflow.kernels.Matern12):
    pass


class ArcCosineImageKernel(gpflow.kernels.ArcCosine, ImageKernel):

    def __init__(self, image_size=None, patch_size=None, **kwargs):
        input_dim = np.prod(patch_size)
        # TODO(VD): do this properly with super!!
        gpflow.kernels.ArcCosine.__init__(self, input_dim, **kwargs)
        ImageKernel.__init__(self, image_size, patch_size)
        assert not self.ARD

    @gpflow.decors.params_as_tensors
    def K_image_inducing_patches(self, X, Z):
        """ returns N x P x M """
        XpZ = self._image_patches_inducing_patches_inner_product(X, Z)  # [N, P, M]
        XpZ = self.weight_variances * XpZ + self.bias_variance  # [N, P, M]
        ZZt = self._inducing_patches_squared_norm(Z)  # M
        ZZt = tf.sqrt(self.weight_variances * ZZt + self.bias_variance)  # M
        XpXpt = self._image_patches_squared_norm(X)
        XpXpt = tf.sqrt(self.weight_variances * XpXpt + self.bias_variance)  # [N, P]
        cos_theta = XpZ / ZZt[None, None, :] / XpXpt[:, :, None]
        jitter = 1e-15
        theta = tf.acos(jitter + (1 - 2 * jitter) * cos_theta)

        return (self.variance * (1. / np.pi) * self._J(theta) *
                ZZt[None, None, :] ** self.order *
                XpXpt[:, :, None] ** self.order)
    
    @gpflow.decors.params_as_tensors
    def K_image(self, X, full_output_cov=False):
        """ returns N x P x P if full_output_cov=True, else N x P """

        if full_output_cov:
            Xp1tXp2 = self._image_patches_inner_product(X, back_prop=False)  # [N, P, P]
            Xp1tXp2 = self.weight_variances * Xp1tXp2 + self.bias_variance  # [N, P, P]
            Xp_squared = tf.sqrt(tf.matrix_diag_part(Xp1tXp2))  # [N, P]
            cos_theta = Xp1tXp2 / Xp_squared[:, None, :] / Xp_squared[:, :, None]
            jitter = 1e-15
            theta = tf.acos(jitter + (1 - 2 * jitter) * cos_theta)

            return (self.variance * (1. / np.pi) * self._J(theta) *
                    Xp_squared[:, None, :] ** self.order *
                    Xp_squared[:, :, None] ** self.order)
        else:
            X_patches_squared_norm = self._image_patches_squared_norm(X)
            X_patches_squared_norm = self.weight_variances * X_patches_squared_norm + self.bias_variance
            theta = tf.cast(0, gpflow.settings.float_type)
            return self.variance * (1. / np.pi) * self._J(theta) * X_patches_squared_norm ** self.order


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
            self.index_kernel = gpflow.kernels.Matern52(len(img_size), lengthscales=3.0)

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
        raise NotImplementedError

        # if self.pooling > 1 or self.with_indexing:
        #     raise NotImplementedError

        # H, W = self.img_size
        # C = self.colour_channels
        # assert self.basekern.ARD == False
        # assert C == 1

        # X = tf.reshape(X, (-1, H, W, C))
        # X2 = X if X2 is None else tf.reshape(X2, (-1, H, W, C))

        # if full_output_cov:
        #     dist = full_conv_square_dist(X, X2, self.patch_size)  # NxPxN2xPxC
        #     dist = tf.squeeze(dist, axis=[4])  # TODO: get rid of colour channel dimension; it assumes that C is 1.
        # else:
        #     dist = patchwise_conv_square_dist(X, X2, self.patch_size)  # PxNxNxC
        #     dist = tf.squeeze(dist, axis=[3])  # TODO: get rid of colour channel dimension; it assumes that C is 1.

        # lengthscales = self.basekern.lengthscales
        # dist /= lengthscales ** 2
        # return self.basekern.K_r2(dist)  # NxPxN2xP

    @gpflow.name_scope("conv_kernel_K_diag")
    @gpflow.params_as_tensors
    def Kdiag(self, X, full_output_cov=False):

        K = self.basekern.K_image(X, full_output_cov=full_output_cov)

        if full_output_cov:
            # K is [N, P, P]

            if self.with_indexing:
                Pij = self.index_kernel.K(self.IJ)  # [P, P]
                K = K * Pij[None, :, :]  # [N, P, P]

            if self.pooling > 1:
                HpWp = self.Hout, self.pooling, self.Wout, self.pooling
                K = tf.reshape(K, [-1, *HpWp, *HpWp])
                K = tf.reduce_sum(K, axis=[2, 4, 6, 8])
                HW = self.Hout * self.Wout
                K = tf.reshape(K, [-1, HW, HW])
            return K  # [N, P', P']
        else:
            # K is [N, P]

            if self.pooling > 1:
                raise NotImplementedError("Pooling is not implemented in ConvKernel.Kdiag() for "
                                          "`full_output_cov` False.")

            if self.with_indexing:
                Pij = self.index_kernel.Kdiag(self.IJ)  # P
                K = K * Pij[None, :]  # [N, P]

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
