from dataclasses import dataclass
from typing import Sequence
import numpy as np
import gpflow
from gpflow.kernels import Convolutional
import tensorflow as tf


Tensor = tf.Tensor
Shape = Sequence[int]
Image = Tensor


class IdentityConvolutionalMean(gpflow.mean_functions.MeanFunction):
    def __init__(
        self,
        image_shape: Shape,
        filter_size: int,
        out_size: int = 1,
        stride: int = 1,
    ):
        super().__init__()

        image_shape = [*image_shape, 1] if len(image_shape) == 2 else image_shape
        self._image_shape = image_shape
        self._filter_size = filter_size
        self._in_size = image_shape[-1]
        self._out_size = out_size
        self._stride = stride
        self._conv_filter = _initial_filter(filter_size, self._in_size, self._out_size)

    def __call__(self, input_tensor: tf.Tensor) -> tf.Tensor:
        N = tf.shape(input_tensor)[0]  # Reshape to [N, W, H, C]
        x = tf.reshape(input_tensor, (N, *self._image_shape))
        strides = (1, self._stride, self._stride, 1)
        conv = tf.nn.conv2d(
            x, self._conv_filter, strides=strides, padding="VALID", data_format="NHWC"
        )
        return tf.reshape(conv, [N, -1])


def _initial_filter(filter_size: int, in_size: int, out_size: int):
    filter_shape = (
        filter_size,
        filter_size,
        in_size,
        out_size,
    )
    identity = np.zeros(filter_shape, dtype=gpflow.default_float())
    identity[filter_size // 2, filter_size // 2, :, :] = 1.0
    return identity


@dataclass
class ImagePatchShapes:
    """
    Contains image-patch processing information.
    """

    image_shape: Shape  # Expects square shape
    patch_shape: Shape  # Expects square shape
    pooling: int = 1

    def __post_init__(self):
        image_shape = self.image_shape
        channel = [1] if len(image_shape) == 2 else []
        self.image_shape = [*image_shape, *channel]

        height, width = self.image_shape[:2]
        h, w = self.patch_shape

        assert height >= h
        assert width >= w

    @property
    def in_size(self) -> int:
        return self.image_shape[0]

    @property
    def out_size(self) -> int:
        image_width = self.image_shape[0]
        patch_width = self.patch_shape[0]
        return (image_width - patch_width + 1) // self.pooling

    @property
    def channel_size(self) -> int:
        return self.image_shape[-1]

    @property
    def num_patches(self) -> int:
        return self.out_size ** 2


class MultioutputConvolutionalKernel(gpflow.kernels.MultioutputKernel):
    """
    Multi-output kernel for a GP from images to images. Uses
    a convolving patch function :math:`g(\cdot)` on an input image.
    """

    def __init__(
        self,
        base_kernel: gpflow.kernels.Kernel,
        image_shape: Shape,
        patch_shape: Shape,
        num_output: int,
        pooling: int = 1,
        patch_handler: PatchHandler = None,
    ):
        """
        :param base_kernel: gpflow.Kernel that operates on the vectors of length :math:`w * h`,
            where :math:`w` and :math:`h` are patch width and height respectively.
        :param image_shape:
        :param patch_shape:
        :param num_output:
        """

        super().__init__()

        shapes = ImagePatchShapes(image_shape, patch_shape, pooling=pooling)
        self.shapes = shapes

        is_handler = isinstance(patch_handler, PatchHandler)
        self.patch_handler = ExtractPatchHandler(config) if not is_handler else patch_handler
        self.base_kernel = base_kernel

    def K(self, X, X2=None, full_output_cov=False):
        """
        :param X: 2 dimensional, Nx(W*H)
        :param X2: 2 dimensional, Nx(W*H)
        """
        raise NotImplementedError

    def Kdiag(self, X, full_output_cov=False):

        shapes = self.shapes
        K = _K_image_symm(self.base_kernel, image, self.shapes, full_output_cov=full_output_cov)

        if full_output_cov and cfg.pooling > 1:
            # K is [N, P, P]

            HpWp = (shapes.out_size, shapes.pooling, shapes.out_size, shape.pooling)
            K = tf.reshape(K, [-1, *HpWp, *HpWp])
            K = tf.reduce_sum(K, axis=[2, 4, 6, 8])  # TODO(awav): how to make it less obscure?
            HW = shapes.out_size ** 2
            K = tf.reshape(K, [-1, HW, HW])  # [N, P', P']
        elif not full_output_cov and cfg.pooling > 1:  # K \in [N, P]
            msg = "Pooling is not implemented in ConvKernel.Kdiag() for `full_output_cov` False."
            raise NotImplementedError(msg)

        return K

    def _setup_indices(self):
        # IJ: Nx2, cartesian product of output indices
        cfg = self.patch_handler.config
        grid = np.meshgrid(np.arange(cfg.Hout), np.arange(cfg.Wout))
        IJ = np.vstack([x.flatten() for x in grid]).T  # Px2
        self.IJ = IJ.astype(settings.float_type)  # (H_out * W_out)x2 = Px2


def _image_patches_inner_product(image_patches: Image) -> Tensor:
    """
    Returns the inner product between all patches in every image in `X`.
    `ret[i, p, p'] = Xᵢ⁽ᵖ⁾ᵀ Xᵢ⁽ᵖ'⁾`, Xᵢ is the i-th image and [q] the q-th patch
    :param X: Tensor containing image data [N, R], where R = H * W * C
    :return: Tensor [N, P*C, P*C]
    """

    return tf.matmul(image_patches, image_patches, transpose_b=True)  # [N, P*C, P*C]


def _image_patches_squared_norm(image_patches: Image) -> Tensor:
    """
    Returns the squared norm for every patch for every image in `X`.
    Corresponds to the diagonal elements of `image_patches_inner_product`.
    `ret[i, p] = Xᵢ⁽ᵖ⁾ᵀ Xᵢ⁽ᵖ⁾`, where Xᵢ is the i-th image and ⁽ᵖ⁾ the p-th patch
    :param X: Tensor containing image data [N, H*W*C]
    :return: Tensor [N, P*C]
    """
    batch = tf.shape(image_patches)[0]
    image_patches_sqsum = tf.reduce_sum(tf.square(image_patches), axis=-1)
    return tf.reshape(image_patches_sqsum, (batch, -1))  # [N, P*C]


def _inducing_patches_squared_norm(inducing_patches: Image) -> Tensor:
    """
    Returns the squared norm of every row in `Z`.
    `ret[i] = Z⁽ⁱ⁾ᵀ Z⁽ⁱ⁾`
    :param Z: Tensor, inducing patches [M, h*w]
    :return: Tensor [M]
    """
    return tf.reduce_sum(inducing_patches ** 2, axis=1)  # M


def _image_patches_inducing_patches_inner_product(
    image_patches: Image, inducing_patches: Image
) -> Tensor:
    """
    Returns the inner product between every patch and every inducing
    point in `Z` (for every image in `X`).
    `ret[i, p, r] = Xᵢ⁽ᵖ⁾ᵀ Zᵣ`, where Xᵢ is the i-th image and (p) the p-th patch,
    and Zᵣ is the r-th inducing patch.
    :param X: Tensor containing image data [N, H*W*C]
    :param Z: Tensor containing inducing patches [M, h*w]
    :return: Tensor [N, P*C, M]
    """

    image_patches_shape = tf.shape(image_patches)[:-2]
    inducing_patches_shape = tf.shape(inducing_patches)
    out_shape = tf.concat([image_patches_shape, inducing_patches_shape], 0)  # [..., M, w*h]
    return tf.matmul(
        image_patches, tf.broadcast_to(inducing_patches, out_shape), transpose_b=True
    )  # [N, P*C, M]


def _image_patches_square_dist(image: Image) -> Tensor:
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
    image_patches
    Xp1tXp2 = patch_handler.image_patches_inner_product(image, back_prop=False)  # [N, P, P]
    Xp_squared = patch_handler.image_patches_squared_norm(image)  # [N, P]
    return Xp_squared[:, :, None] + Xp_squared[:, None, :] - 2 * Xp1tXp2  # [N, P, P]


def _get_patches(image: Image, shapes: ImagePatchShapes):
    """
    Extracts patches from the images X. Patches are extracted separately for each of the colour channels.
    :param X: (N x input_dim)
    :return: Patches (N, num_patches, patch_shape)
    """
    # Roll the colour channel to the front, so it appears to
    # `tf.extract_image_patches()` as separate images. Then extract patches
    # and reshape to have the first axis the same as the number of images.
    # The separate patches will then be in the second axis.
    pad = [1, 1, 1, 1]
    num_data = tf.shape(image)[0]
    image_cast = tf.transpose(tf.reshape(image, [num_data, -1, shapes.channel_size]), [0, 2, 1])
    image_shape = [-1, shapes.in_size, shapes.in_size, 1]
    patch_shape = [1, shapes.patch_size, shapes.patch_size, 1]
    patches = tf.image.extract_patches(
        tf.reshape(image_cast, image_shape),
        patch_shape,
        pad,
        pad,
        "VALID",
    )
    patches_shape = tf.shape(patches)  # img x out_rows x out_cols
    out_shape = (
        num_data,
        shapes.channel_size * patches_shape[1] * patches_shape[2],
        patches_shape[3],
    )
    reshaped_patches = tf.reshape(patches, out_shape)
    return gpflow.utilities.to_default_float(reshaped_patches)


def _K_image_symm(kernel: gpflow.kernels.Stationary, image: Image, full_output_cov=False):
    """:return: Tensor [N, P]. If full_output_cov is `True` then tensor with [N, P, P]
    shape returned."""

    patches = _get_patches(image)

    if full_output_cov:
        dist = _image_patches_square_dist(patches)  # [N, P, P]
        dist /= kernel.lengthscales ** 2  # Dividing after computing distances
        # helps to avoid unnecessary backpropagation.
        return kernel.K_r2(dist)  # [N, P, P]
    else:
        P = shapes.num_patches
        return kernel.variance * tf.ones([tf.shape(image)[0], P], dtype=image.dtype)  # [N, P]