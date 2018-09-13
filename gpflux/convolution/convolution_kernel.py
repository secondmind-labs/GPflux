# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from abc import ABC, abstractmethod
from typing import List

import gpflow
import numpy as np
import tensorflow as tf
from gpflow import Param
from gpflow import kernels
from gpflow import params_as_tensors, params_as_tensors_for, settings
from gpflow.multioutput.kernels import Mok

from ..conv_square_dists import (diag_conv_inner_prod,
                                 diag_conv_square_dist,
                                 full_conv_square_dist,
                                 patchwise_conv_square_dist)


class ImageBasedKernel(ABC):
    padding = 'VALID'  # Static variable, used by convolutional operations.
    strides = (1, 1, 1, 1)  # Static variable, used by convolutional operations.

    def __init__(self, *args, image_shape=None, patch_shape=None, **kwargs):
        """
        :param image_shape: [height, width, color channels] = [H, W, C]
        :param patch_shape: [height, width] = [h, w]
        """
        if not (isinstance(image_shape, (list, tuple)) and
                isinstance(patch_shape, (list, tuple))):
            raise ValueError('Shapes must be a tuple or list.')

        super().__init__(*args, **kwargs)

        image_shape = list(image_shape)
        patch_shape = list(patch_shape)

        if len(image_shape) == 2:
            # TODO(VD) deal with color channel
            image_shape = image_shape + [1]

        height, width = image_shape[:2]
        h, w = patch_shape
        assert height >= h
        assert width >= w
        self.image_shape = image_shape
        self.patch_shape = patch_shape
        self.patch_grid_size = (height - h + 1, width - w + 1)
        self.colour_channels = self.image_shape[-1]
        assert self.colour_channels == 1

    def image_patches_inner_product(self, X, **kwargs):
        """
        Returns the inner product between all patches in every image in `X`.
        `ret[n, p, p'] = Xn[p] Xn[p']^T` Xn is the n-th image and [q] the q-th patch

        :param X: Tensor containing image data [N, H*W*C]
        :return: Tensor [N, P, P]
        """
        X = self._reshape_to_image(X)
        inner_prod = diag_conv_inner_prod(X, self.patch_shape, **kwargs)  # [N, P, P, 1]
        return tf.squeeze(inner_prod, axis=[3])  # [N, P, P]

    def image_patches_squared_norm(self, X):
        """
        Returns the squared norm for every patch for every image in `X`.
        Corresponds to the diagonal elements of `image_patches_inner_product`.
        `ret[n, p] = Xn[p] Xn[p]^T` Xn is the n-th image and [p] the p-th patch

        :param X: Tensor containing image data [N, H*W*C]
        :return: Tensor [N, P]
        """
        X = self._reshape_to_image(X)
        ones = tf.ones((*self.patch_shape, self.colour_channels, 1), dtype=X.dtype)
        XpXpt = tf.nn.conv2d(X ** 2, ones, self.strides, self.padding)
        return tf.reshape(XpXpt, (tf.shape(X)[0], -1))  # [N, P]

    def inducing_patches_squared_norm(self, Z):
        """
        Returns the squared norm of every row in `Z`. `ret[i] = Z[i] Z[i]^T`.

        :param Z: Tensor, inducing patches [M, h*w]
        :return: Tensor [M]
        """
        return tf.reduce_sum(Z ** 2, axis=1)  # M

    def image_patches_inducing_patches_inner_product(self, X, Z):
        """
        Returns the inner product between every patch and every inducing
        point in `Z` (for every image in `X`).

        `ret[n, p, m] = Xn[p] Zm^T` Xn is the n-th image and [q] the q-th patch,
        and Zm is the m-th inducing patch.

        :param X: Tensor containing image data [N, H*W*C]
        :param Z: Tensor containing inducing patches [N, h*w]
        :return: Tensor [N, P, M]
        """
        X = self._reshape_to_image(X)
        M, N = tf.shape(Z)[0], tf.shape(X)[0]
        Zr = tf.reshape(Z, (M, *self.patch_shape, self.colour_channels))  # [M, h, w, C]
        Z_filter = tf.transpose(Zr, [1, 2, 3, 0])  # [h, w, C, M]
        XpZ = tf.nn.conv2d(X, Z_filter, self.strides, self.padding)  # [N, Ph, Pw, M]
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
        pass

    @abstractmethod
    def K_image_symm(self, X, full_output_cov=False):
        """
        Kernel between every 2 patches in every image in `X`
        `ret[m,p,p'] = k(Xn[p], Xn[p'])`, where [p] operator selects the p-th patch.

        :param X: Tensor of image data, [N, H*W*C]
        :param full_output_cov: boolean. If `False` only the diagonal elements are returned
        :return: covariance matrix [N, P, P] if `full_output_cov`=True, else [N, P].
        """
        pass

    @gpflow.decors.autoflow((gpflow.settings.float_type, [None, None]))
    def compute_K_image_symm(self, X):
        return self.K_image_symm(X, full_output_cov=False)

    @gpflow.decors.autoflow((gpflow.settings.float_type, [None, None]))
    def compute_K_image_full_output_cov(self, X):
        return self.K_image_symm(X, full_output_cov=True)

    @gpflow.decors.autoflow((gpflow.settings.float_type, [None, None]),
                            (gpflow.settings.float_type, [None, None]))
    def compute_K_image_inducing_patches(self, X, Z):
        return self.K_image_inducing_patches(X, Z)

    def _reshape_to_image(self, X):
        """Reshapes X [N, H*W*C] from being rank 2 to proper image shape [N, H, W, C]"""
        return tf.reshape(X, [-1, *self.image_shape])


class ImageStationary(ImageBasedKernel, kernels.Stationary):
    def __init__(self, image_shape=None, patch_shape=None, **kwargs):
        input_dim = np.prod(patch_shape)
        super().__init__(input_dim, image_shape=image_shape, patch_shape=patch_shape, **kwargs)
        assert not self.ARD

    def image_patches_square_dist(self, X, **map_kwargs):
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

        Xp1tXp2 = self.image_patches_inner_product(X, **map_kwargs)  # [N, P, P]
        Xp_squared = self.image_patches_squared_norm(X)  # [N, P]
        return -2 * Xp1tXp2 + Xp_squared[:, :, None] + Xp_squared[:, None, :]  # [N, P, P]

    def image_patches_inducing_patches_square_dist(self, X, Z):
        """
        Calculates the squared distance between patches in X image and Z patch
        ```
            ret[n,p,m] = || Xn[p] - Zm ||^2
                       = Xn[p]^T Xn[p] + Zm^T Zm - 2 Xn[p]^T Zm
        ```
        and every inducing patch in `Z`.

        :param X: Tensor of shape [N, H, W, C]
        :param Z: Tensor of shape [N, h*w]
        :return: Tensor of shape [N, P, M].
        """
        Xp_squared = self.image_patches_squared_norm(X)  # [N, P]
        Zm_squared = self.inducing_patches_squared_norm(Z)  # M
        XptZm = self.image_patches_inducing_patches_inner_product(X, Z)  # [N, P, M]
        return Xp_squared[:, :, None] + Zm_squared[None, None, :] - 2 * XptZm

    @gpflow.decors.params_as_tensors
    def K_image_inducing_patches(self, X, Z):
        """ returns [N, P, M] """
        dist = self.image_patches_inducing_patches_square_dist(X, Z)  # [N, P, M]
        dist /= self.lengthscales ** 2  # Dividing after computing distances
                                        # helps to avoid unnecessary backpropagation.
                                        # But it will not work with ARD case.
        return self.K_r2(dist)  # [N, P, M]

    @gpflow.decors.params_as_tensors
    def K_image_symm(self, X, full_output_cov=False):
        """ returns [N, P, P] if full_output_cov=True, else [N, P] """
        if full_output_cov:
            dist = self.image_patches_square_dist(X, back_prop=False)  # [N, P, P]
            dist /= self.lengthscales ** 2  # Dividing after computing distances
                                            # helps to avoid unnecessary backpropagation.
            return self.K_r2(dist)  # [N, P, P]
        else:
            P = np.prod(self.patch_grid_size)
            return self.variance * tf.ones([tf.shape(X)[0], P], dtype=X.dtype)  # [N, P]


class ImageArcCosine(ImageBasedKernel, kernels.ArcCosine):
    def __init__(self, image_shape=None, patch_shape=None, **kwargs):
        input_dim = np.prod(patch_shape)
        super().__init__(input_dim, image_shape=image_shape, patch_shape=patch_shape, **kwargs)
        assert not self.ARD, "ARD is not supported."

    @gpflow.decors.params_as_tensors
    def K_image_inducing_patches(self, X, Z):
        """ Returns N x P x M """
        XpZ = self.image_patches_inducing_patches_inner_product(X, Z)  # [N, P, M]
        XpZ = self.weight_variances * XpZ + self.bias_variance  # [N, P, M]
        ZZt = self.inducing_patches_squared_norm(Z)  # M
        ZZt = tf.sqrt(self.weight_variances * ZZt + self.bias_variance)  # M
        XpXpt = self.image_patches_squared_norm(X)
        XpXpt = tf.sqrt(self.weight_variances * XpXpt + self.bias_variance)  # [N, P]
        cos_theta = XpZ / ZZt[None, None, :] / XpXpt[:, :, None]
        jitter = 1e-15
        theta = tf.acos(jitter + (1 - 2 * jitter) * cos_theta)

        ZZto = ZZt[None, None, :] ** self.order
        XpXpto = XpXpt[:, :, None] ** self.order

        return self.variance * (1. / np.pi) * self._J(theta) * ZZto * XpXpto

    @gpflow.decors.params_as_tensors
    def K_image_symm(self, X, full_output_cov=False):
        """ Returns N x P x P if full_output_cov=True, else N x P """

        if full_output_cov:
            Xp1tXp2 = self.image_patches_inner_product(X, back_prop=False)  # [N, P, P]
            Xp1tXp2 = self.weight_variances * Xp1tXp2 + self.bias_variance  # [N, P, P]
            Xp_squared = tf.sqrt(tf.matrix_diag_part(Xp1tXp2))  # [N, P]
            cos_theta = Xp1tXp2 / Xp_squared[:, None, :] / Xp_squared[:, :, None]
            jitter = 1e-15
            theta = tf.acos(jitter + (1 - 2 * jitter) * cos_theta)

            Xp_squared_o1 = Xp_squared[:, None, :] ** self.order
            Xp_squared_o2 = Xp_squared[:, :, None] ** self.order
            return self.variance * (1. / np.pi) * self._J(theta) * Xp_squared_o1 * Xp_squared_o2
        else:
            X_patches_squared_norm = self.image_patches_squared_norm(X)
            X_patches_squared_norm = self.weight_variances * X_patches_squared_norm + self.bias_variance
            theta = tf.cast(0, gpflow.settings.float_type)
            X_patches_squared_norm_o = X_patches_squared_norm ** self.order
            return self.variance * (1. / np.pi) * self._J(theta) * X_patches_squared_norm_o


class ConvKernel(Mok):
    """
    Multi-output kernel for a GP from images to images. Constructed by
    convolving a patch function g(.) over the input image.
    """

    def __init__(self,
                 basekern: kernels.Kernel,
                 pooling: int = 1,
                 with_indexing: bool = False):
        """
        :param basekern: gpflow.Kernel that operates on the vectors of length w*h
        """

        if not isinstance(basekern, ImageBasedKernel):
            raise ValueError('ConvKernel kernel works only with image based kernels.')

        assert basekern.colour_channels == 1

        super().__init__(basekern.input_dim)
        self.Hin, self.Win = basekern.image_shape[:2]

        Hout, Wout = basekern.patch_grid_size

        assert Hout % pooling == 0
        assert Wout % pooling == 0

        self.pooling = pooling
        self.Hout = Hout // pooling
        self.Wout = Wout // pooling
        self.colour_channels = basekern.colour_channels
        self.num_patches = self.Hout * self.Wout * self.colour_channels

        self.basekern = basekern
        self.with_indexing = with_indexing
        if self.with_indexing:
            self._setup_indices()
            # TODO(@awav): pass index kernel via arguments
            self.index_kernel = kernels.Matern52(len(basekern.image_shape), lengthscales=3.0)

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

        K = self.basekern.K_image_symm(X, full_output_cov=full_output_cov)

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
                msg = "Pooling is not implemented in ConvKernel.Kdiag() for `full_output_cov` False."
                raise NotImplementedError(msg)

            if self.with_indexing:
                Pij = self.index_kernel.Kdiag(self.IJ)  # P
                K = K * Pij[None, :]  # [N, P]

            return K

    def _setup_indices(self):
        # IJ: Nx2, cartesian product of output indices
        grid = np.meshgrid(np.arange(self.Hout), np.arange(self.Wout))
        IJ = np.vstack([x.flatten() for x in grid]).T  # Px2
        self.IJ = IJ.astype(settings.float_type)  # (H_out * W_out)x2 = Px2

    @property
    def patch_len(self):
        return np.prod(self.basekern.patch_shape)

    @property
    def num_outputs(self):
        return self.num_patches


class WeightedSumConvKernel(ConvKernel):
    def __init__(self,
                 basekern: kernels.Kernel,
                 pooling: int = 1,
                 with_indexing: bool = False,
                 with_weights: bool = False):

        super().__init__(basekern, pooling, with_indexing)

        self.with_weights = with_weights
        weights = np.ones([self.num_outputs], dtype=settings.float_type)  # P
        self.weights = Param(weights) if with_weights else weights

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
        K = K / self.num_outputs  ** 2 # N
        return K


# Simple stationary kernels gotten by mixin ImageStationary with originals.

class ImageRBF(ImageStationary, kernels.RBF): pass
class ImageMatern12(ImageStationary, kernels.Matern12): pass
