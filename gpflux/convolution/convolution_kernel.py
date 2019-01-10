# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from abc import ABC, abstractmethod
from functools import partial, lru_cache
from typing import List, Tuple, Type, Union

import numpy as np
import tensorflow as tf
from multipledispatch import dispatch

import gpflow
from gpflow import kernels, params_as_tensors, params_as_tensors_for, settings
from gpflow.multioutput.kernels import Mok
from .. import utils

from ..conv_square_dists import (diag_conv_inner_prod, diag_conv_square_dist,
                                 full_conv_square_dist,
                                 patchwise_conv_square_dist)

Any = object
PatchShape = Union[List[int], Tuple[int, int]]
ImageShape = Union[PatchShape, Tuple[int, int, int]]


gpflux_image_kernel = dict()
dispatch = partial(dispatch, namespace=gpflux_image_kernel)


class ImagePatchConfig:
    """
    Data class containing necessary information related to image-patch processing by
    patch handler.
    """

    def __init__(self, image_shape: ImageShape, patch_shape: PatchShape, pooling: int = 1):
        """
        :param image_shape: Image shape. It can be either 2 element tuple [H, W] or 3 element tuple [H, W, C],
            where H - height, W - width, C - channel. [H, W] is equivalent to [H, W, 1]
        :param patch_shape: Patch or filter shape. It must be 2 element tuple [h, w], where h <= H and w <= W.
        :param pooling: Reducing resolution factor.
        """

        if not (isinstance(image_shape, (list, tuple)) and
                isinstance(patch_shape, (list, tuple))):
            raise ValueError('Shapes must be a tuple or list.')

        image_shape = list(image_shape)
        patch_shape = list(patch_shape)

        if len(image_shape) == 2:
            image_shape = image_shape + [1]

        height, width = image_shape[:2]
        h, w = patch_shape

        assert height >= h
        assert width >= w

        Cin = image_shape[-1]
        self.Hin, self.Win = height, width

        Hout, Wout = height - h + 1, width - w + 1

        assert Hout % pooling == 0
        assert Wout % pooling == 0

        Hout //= pooling
        Wout //= pooling

        self.pooling = pooling
        self.Hout = Hout
        self.Wout = Wout
        self.image_shape = image_shape
        self.patch_shape = patch_shape
        self.Cin = Cin
        self.num_patches = Hout * Wout


class PatchHandler(ABC):
    """
    Interface for major image-patch operations:
        * Inner product between image patches
        * Squared norm of image patches
        * Squared norm of inducing patches
        * Inner product between image patches and inducing patches
    """

    def __init__(self, config: ImagePatchConfig):
        self.config = config

    @abstractmethod
    def image_patches_inner_product(self, X, **kwargs):
        pass

    @abstractmethod
    def image_patches_squared_norm(self, X, **kwargs):
        pass

    @abstractmethod
    def inducing_patches_squared_norm(self, Z, **kwargs):
        pass

    @abstractmethod
    def image_patches_inducing_patches_inner_product(self, X, Z, **kwargs):
        pass

    def reshape_to_image(self, X):
        cfg = self.config
        return tf.reshape(X, [-1, cfg.Hin, cfg.Win, cfg.Cin])


class ExtractPatchHandler(PatchHandler):
    padding = 'VALID'  # Static variable, used by convolutional operations.
    strides = (1, 1, 1, 1)  # Static variable, used by convolutional operations.

    @lru_cache(maxsize=10)
    def get_image_patches(self, X: tf.Tensor):
        """
        Extract image patches method with LRU cache.

        :param X: input tensor.
        :return: Extracted image patches. When the same object tensor passed to the function, cached value is returned back. Output tensor shape - [N, P*C, wh].
        """
        X = self.reshape_to_image(X)
        image_shape = self.config.image_shape
        patch_shape = self.config.patch_shape
        return utils.get_image_patches(X, image_shape, patch_shape)

    def image_patches_inner_product(self, X, **map_fn_kwargs):
        """
        Returns the inner product between all patches in every image in `X`.
        `ret[i, p, p'] = Xᵢ⁽ᵖ⁾ᵀ Xᵢ⁽ᵖ'⁾`, Xᵢ is the i-th image and [q] the q-th patch

        :param X: Tensor containing image data [N, R], where R = H * W * C
        :return: Tensor [N, P*C, P*C]
        """

        Xp = self.get_image_patches(X)
        return tf.matmul(Xp, Xp, transpose_b=True)  # [N, P*C, P*C]

    def image_patches_squared_norm(self, X):
        """
        Returns the squared norm for every patch for every image in `X`.
        Corresponds to the diagonal elements of `image_patches_inner_product`.
        `ret[i, p] = Xᵢ⁽ᵖ⁾ᵀ Xᵢ⁽ᵖ⁾`, where Xᵢ is the i-th image and ⁽ᵖ⁾ the p-th patch

        :param X: Tensor containing image data [N, H*W*C]
        :return: Tensor [N, P*C]
        """
        Xp = self.get_image_patches(X)
        XpXpt = tf.reduce_sum(Xp ** 2, axis=-1)
        return tf.reshape(XpXpt, (tf.shape(X)[0], -1))  # [N, P*C]

    def inducing_patches_squared_norm(self, Z):
        """
        Returns the squared norm of every row in `Z`.
        `ret[i] = Z⁽ⁱ⁾ᵀ Z⁽ⁱ⁾`

        :param Z: Tensor, inducing patches [M, h*w]
        :return: Tensor [M]
        """
        return tf.reduce_sum(Z ** 2, axis=1)  # M

    def image_patches_inducing_patches_inner_product(self, X, Z):
        """
        Returns the inner product between every patch and every inducing
        point in `Z` (for every image in `X`).

        `ret[i, p, r] = Xᵢ⁽ᵖ⁾ᵀ Zᵣ`, where Xᵢ is the i-th image and (p) the p-th patch,
        and Zᵣ is the r-th inducing patch.

        :param X: Tensor containing image data [N, H*W*C]
        :param Z: Tensor containing inducing patches [M, h*w]
        :return: Tensor [N, P*C, M]
        """

        Xp = self.get_image_patches(X)
        shape = tf.concat([tf.shape(Xp)[:-2], tf.shape(Z)], 0)  # [..., M, w*h]
        return tf.matmul(Xp, tf.broadcast_to(Z, shape), transpose_b=True)  # [N, P*C, M]


class ConvPatchHandler(PatchHandler):
    padding = 'VALID'  # Static variable, used by convolutional operations.
    strides = (1, 1, 1, 1)  # Static variable, used by convolutional operations.

    def image_patches_inner_product(self, X, **map_fn_kwargs):
        """
        Returns the inner product between all patches in every image in `X`.
        `ret[i, p, p'] = Xᵢ⁽ᵖ⁾ Xᵢ⁽ᵖ'⁾ᵀ`, Xᵢ is the i-th image and [q] the q-th patch

        :param X: Tensor containing image data [N, R], where R = H * W * C
        :return: Tensor [N, P*C, P]
        """
        X = self.reshape_to_image(X)
        inner_prod = diag_conv_inner_prod(X, self.config.patch_shape, **map_fn_kwargs)  # [N, P, C, P, C]
        inner_prod_shape = tf.shape(inner_prod)
        P, C = inner_prod_shape[-2], inner_prod_shape[-1]
        return tf.reshape(inner_prod, [-1, P*C, P*C])  # [N, P*C, P*C]

    def image_patches_squared_norm(self, X):
        """
        Returns the squared norm for every patch for every image in `X`.
        Corresponds to the diagonal elements of `image_patches_inner_product`.
        `ret[i, p] = Xᵢ⁽ᵖ⁾ Xᵢ⁽ᵖ⁾ᵀ`, where Xᵢ is the i-th image and ⁽ᵖ⁾ the p-th patch
        :param X: Tensor containing image data [N, H*W*C]
        :return: Tensor [N, P*C]
        """
        cfg = self.config
        X = self.reshape_to_image(X)
        ones = tf.ones((*cfg.patch_shape, cfg.Cin, 1), dtype=X.dtype)
        XpXpt = tf.nn.conv2d(X ** 2, ones, self.strides, self.padding)
        return tf.reshape(XpXpt, (tf.shape(X)[0], -1))  # [N, P*C]

    def inducing_patches_squared_norm(self, Z):
        """
        Returns the squared norm of every row in `Z`.
        `ret[i] = Z⁽ⁱ⁾ᵀ Z⁽ⁱ⁾`
        :param Z: Tensor, inducing patches [M, h*w]
        :return: Tensor [M]
        """
        return tf.reduce_sum(Z ** 2, axis=1)  # M

    def image_patches_inducing_patches_inner_product(self, X, Z):
        """
        Returns the inner product between every patch and every inducing
        point in `Z` (for every image in `X`).
        `ret[i, p, r] = Xᵢ⁽ᵖ⁾ᵀ Zᵣ`, where Xᵢ is the i-th image and (p) the p-th patch,
        and Zᵣ is the r-th inducing patch.
        :param X: Tensor containing image data [N, H*W*C]
        :param Z: Tensor containing inducing patches [M, h*w]
        :return: Tensor [N, P*C, M]
        """
        X = self.reshape_to_image(X) # [N, H, W, C]
        cfg = self.config

        Xshape = tf.shape(X)
        M = tf.shape(Z)[0]
        N, H, W, C = Xshape[0], Xshape[1], Xshape[2], Xshape[3]
        Z = tf.reshape(Z, (M, *cfg.patch_shape, 1))  # [M, h, w, 1]
        Z_filter = tf.transpose(Z, [1, 2, 3, 0])  # [h, w, 1, M]

        X = tf.transpose(X, [0, 3, 1, 2])  # [N, C, H, W]
        X = tf.reshape(X, [N*C, H, W, 1])
        XpZ = tf.nn.depthwise_conv2d(X, Z_filter, self.strides, self.padding)  # [N*C, Ph, Pw, M]
        XpZ = tf.reshape(XpZ, [N, C, -1, M])
        XpZ = tf.transpose(XpZ, [0, 2, 1, 3])
        return tf.reshape(XpZ, [N, -1, M])  # [N, P*C, M]


@dispatch(kernels.Stationary, PatchHandler, Any, Any)
@gpflow.name_scope()
def K_image_inducing_patches(kernel, patch_handler, X, Z) -> tf.Tensor:
    """ returns [N, P, M] """

    def image_patches_inducing_patches_square_dist():
        """
        Calculates the squared distance between patches in X image and Z patch
        ```
            ret[i,p,r] = ||Xᵢ⁽ᵖ⁾ - Zᵣ||² = Xᵢ⁽ᵖ⁾ᵀ Xᵢ⁽ᵖ⁾ + Zᵣᵀ Zᵣ - 2 Xᵢ⁽ᵖ⁾ᵀ Zᵣ
        ```
        and every inducing patch in `Z`.

        :param X: Tensor of shape [N, H, W, C]
        :param Z: Tensor of shape [N, h*w]
        :return: Tensor of shape [N, P, M].
        """
        Xp_squared = patch_handler.image_patches_squared_norm(X)  # [N, P]
        Zm_squared = patch_handler.inducing_patches_squared_norm(Z)  # M
        XptZm = patch_handler.image_patches_inducing_patches_inner_product(X, Z)  # [N, P, M]
        return Xp_squared[:, :, None] + Zm_squared[None, None, :] - 2 * XptZm

    with gpflow.params_as_tensors_for(kernel):
        dist = image_patches_inducing_patches_square_dist()  # [N, P, M]
        dist /= kernel.lengthscales ** 2  # Dividing after computing distances
                                          # helps to avoid unnecessary backpropagation.
                                          # But it will not work with ARD case.
        return kernel.K_r2(dist)  # [N, P, M]


@dispatch(kernels.ArcCosine, PatchHandler, Any, Any)
@gpflow.name_scope()
def K_image_inducing_patches(kernel, patch_handler, X, Z):
    """:return: Tensor [N, P, M]"""
    with gpflow.params_as_tensors_for(kernel):
        XpZ = patch_handler.image_patches_inducing_patches_inner_product(X, Z)  # [N, P, M]
        XpZ = kernel.weight_variances * XpZ + kernel.bias_variance  # [N, P, M]
        ZZt = patch_handler.inducing_patches_squared_norm(Z)  # M
        ZZt = tf.sqrt(kernel.weight_variances * ZZt + kernel.bias_variance)  # M
        XpXpt = patch_handler.image_patches_squared_norm(X)
        XpXpt = tf.sqrt(kernel.weight_variances * XpXpt + kernel.bias_variance)  # [N, P]
        cos_theta = XpZ / ZZt[None, None, :] / XpXpt[:, :, None]
        jitter = 1e-15
        theta = tf.acos(jitter + (1 - 2 * jitter) * cos_theta)

        ZZto = ZZt[None, None, :] ** kernel.order
        XpXpto = XpXpt[:, :, None] ** kernel.order

        return kernel.variance * (1. / np.pi) * kernel._J(theta) * ZZto * XpXpto


@dispatch(kernels.Stationary, PatchHandler, Any)
@gpflow.name_scope()
def K_image_symm(kernel, patch_handler, X, full_output_cov=False):
    """:return: Tensor [N, P]. If full_output_cov is `True` then tensor with [N, P, P]
    shape returned."""

    def image_patches_square_dist(X):
        """
        Calculates the squared distance between every patch in each image of `X`
        ```
            ret[i,p,p'] = ||Xᵢ⁽ᵖ⁾ - Xᵢ⁽ᵖ'⁾||²
                        = Xᵢ⁽ᵖ⁾ᵀ Xᵢ⁽ᵖ⁾ + Xᵢ⁽ᵖ'⁾ᵀ Xᵢ⁽ᵖ'⁾ - 2 Xᵢ⁽ᵖ⁾ᵀ Xᵢ⁽ᵖ'⁾,
            where Xᵢ is the i-th image and `⁽ᵖ⁾` operator selects the p-th patch.
        ```
        :param X: Tensor of shape [N, H, W, C]
        :return: Tensor of shape [N, P, P].
        """
        Xp1tXp2 = patch_handler.image_patches_inner_product(X, back_prop=False)  # [N, P, P]
        Xp_squared = patch_handler.image_patches_squared_norm(X)  # [N, P]
        return Xp_squared[:, :, None] + Xp_squared[:, None, :] - 2 * Xp1tXp2  # [N, P, P]

    with gpflow.params_as_tensors_for(kernel):
        if full_output_cov:
            dist = image_patches_square_dist(X)  # [N, P, P]
            dist /= kernel.lengthscales ** 2  # Dividing after computing distances
                                              # helps to avoid unnecessary backpropagation.
            return kernel.K_r2(dist)  # [N, P, P]
        else:
            P = patch_handler.config.num_patches
            return kernel.variance * tf.ones([tf.shape(X)[0], P], dtype=X.dtype)  # [N, P]


@dispatch(kernels.ArcCosine, PatchHandler, Any)
@gpflow.name_scope()
def K_image_symm(kernel, patch_handler, X, full_output_cov=False):
    """:return: Tentosr [N, P]. If full_output_cov is `True` then tensor [N, P, P] is returned"""
    with gpflow.params_as_tensors_for(kernel):
        if full_output_cov:
            Xp1tXp2 = patch_handler.image_patches_inner_product(X, back_prop=False)  # [N, P, P]
            Xp1tXp2 = kernel.weight_variances * Xp1tXp2 + kernel.bias_variance  # [N, P, P]
            Xp_squared = tf.sqrt(tf.matrix_diag_part(Xp1tXp2))  # [N, P]
            cos_theta = Xp1tXp2 / Xp_squared[:, None, :] / Xp_squared[:, :, None]
            jitter = 1e-15
            theta = tf.acos(jitter + (1 - 2 * jitter) * cos_theta)

            Xp_squared_o1 = Xp_squared[:, None, :] ** kernel.order
            Xp_squared_o2 = Xp_squared[:, :, None] ** kernel.order
            return kernel.variance * (1. / np.pi) * kernel._J(theta) * Xp_squared_o1 * Xp_squared_o2
        else:
            X_patches_squared_norm = patch_handler.image_patches_squared_norm(X)
            X_patches_squared_norm = kernel.weight_variances * X_patches_squared_norm + kernel.bias_variance
            theta = tf.cast(0, gpflow.settings.float_type)
            X_patches_squared_norm_o = X_patches_squared_norm ** kernel.order
            return kernel.variance * (1. / np.pi) * kernel._J(theta) * X_patches_squared_norm_o


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

        config = ImagePatchConfig(image_shape, patch_shape, pooling=pooling)
        is_handler = isinstance(patch_handler, PatchHandler)
        self.patch_handler = ExtractPatchHandler(config) if not is_handler else patch_handler

        self.basekern = basekern
        self.with_indexing = with_indexing
        if self.with_indexing:
            self._setup_spatio_indices()
            self.spatio_indices_kernel = kernels.Matern52(2, lengthscales=3.0)

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
                Pij self.spatio_indices_kernel.K(self.spatio_indices)  # [P, P]
                K = K * Pij[None, :, :]
            else:
                Pij = self.spatio_indices_kernel.Kdiag(self.spatio_indices)  # [P]
                K = K * Pij[None, :]

        if cfg.pooling > 1:
            if not full_output_cov:
                msg = "Pooling is not implemented in ConvKernel.Kdiag() for `full_output_cov` False."
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
        self.spatio_indices = tf.convert_to_tensor(spatio_indices)

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
                 with_weights: bool = False):

        super().__init__(basekern, image_shape, patch_shape,
                         pooling=pooling, with_indexing=with_indexing)

        self.with_weights = with_weights
        weights = np.ones([self.num_outputs], dtype=settings.float_type)  # P
        self.weights = gpflow.Param(weights) if with_weights else weights

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
