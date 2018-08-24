from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
import gpflow


__all__ = [
    'diag_conv_square_dist',
    'full_conv_square_dist',
    'pathwise_conv_square_dist',
    'image_patch_conv_square_dist'
]

Int = Union[tf.Tensor, int]
FilterShape = Tuple[Int, Int]  # [h, w]
ImageShape = Tuple[Int, Int, Int, Int]  # [N, H, W, C]


padding = 'VALID'
strides = (1, 1, 1, 1)


@gpflow.name_scope()
def image_patch_conv_square_dist(A: tf.Tensor, B: tf.Tensor, filter_shape: FilterShape) -> tf.Tensor:
    """
    Square distance between image and patch using convolution operations.

    Args:
        A: Image tensor of shape [N, H, W, C]
        B: Patch tensor of shape [M, h, w, C]
        filter_shape: Two element tuple with height and width sizes of filter respectively.
    Returns:
        Tensor of shape [N, M, P].
    """
    h, w = filter_shape
    M = tf.shape(B)[0]
    N, H, W, C = _image_shape(A)
    P = _grid_patch_size(H, W, filter_shape)  # Ph * Pw = (H - h + 1) * (W - w + 1)
    dtype = A.dtype

    ones = tf.ones((h, w, C, 1), dtype=dtype)

    AtA = tf.nn.conv2d(A ** 2, ones, strides, padding)
    AtA = tf.reshape(AtA, (N, P))

    B_filter = tf.transpose(tf.reshape(B, (M, h, w, C)), [1, 2, 3, 0])  # [h, w, C, M]
    ABt = tf.nn.conv2d(A, B_filter, strides, padding)  # [N, Ph, Pw, M]
    ABt = tf.reshape(tf.transpose(ABt, [0, 3, 1, 2]), (N, M, P))

    BtB = tf.reduce_sum(B ** 2, axis=1)  # M

    return -2 * ABt + AtA[:, None, :] + BtB[None, :, None]  # [N, M, P]


@gpflow.name_scope()
def diag_conv_square_dist(A: tf.Tensor, filter_shape: FilterShape,
                          **map_kwargs) -> tf.Tensor:
    """
    Convolutional squared distances for diagonal case. Extracts filters from the input image
    applies them to the same image using depthwise convolution operation along batch
    dimention N.

    Args:
        A: Tensor of shape [N, H, W, C]
        B: Tensor of shape [N, H, W, C]
        filter_shape: Two element tuple with height and width sizes of filter respectively.
    Returns:
        Tensor of shape [N, P, P, C].
    """
    asserts = _input_tensor_asserts(A)
    with tf.control_dependencies(asserts):
        image_shape = _image_shape(A)

    AtA, _ = self_inner_prod(A, None, filter_shape)
    AAt = diag_conv_inner_prod(A, filter_shape, **map_kwargs)
    return -2 * AAt + AtA[:, :, None, :] + AtA[:, None, :, :]


@gpflow.name_scope()
def full_conv_square_dist(A: tf.Tensor, B: tf.Tensor, filter_shape: FilterShape,
                          **map_kwargs) -> tf.Tensor:
    """
    Convolutional squared distances for full output case. Extracts filters from the input
    image A and applies them to B image using depthwise convolution operation for each
    sub-image in the batch.

    Args:
        A: Tensor of shape [N, H, W, C]
        B: Tensor of shape [N, H, W, C]
        filter_shape: Two element tuple with height and width sizes of filter respectively.
    Returns:
        Tensor of shape [N, P, N, P, C].
    """
    asserts = _input_tensor_asserts(A, B)
    with tf.control_dependencies(asserts):
        image_shape = _image_shape(A)

    AtA, BtB = self_inner_prod(A, B, filter_shape)  # [N, P, C], [N, P, C]
    ABt = full_conv_inner_prod(A, B, filter_shape, **map_kwargs)  # [N, P, N, P, C]
    return -2 * ABt + AtA[:, :, None, None, :] + BtB[None, None, :, :, :]


@gpflow.name_scope()
def patchwise_conv_square_dist(A: tf.Tensor, B: tf.Tensor, filter_shape: FilterShape,
                               **map_kwargs) -> tf.Tensor:
    """
    Convolutional squared distances for patchwise case. Extracts filters from the input
    image A and applies them to B image using depthwise convolution operation along batch
    dimension.

    Args:
        A: Tensor of shape [N, H, W, C]
        B: Tensor of shape [N, H, W, C]
        filter_shape: Two element tuple with height and width sizes of filter respectively
        map_kwargs: (optional) `tf.map_fn` arguments.

    Returns:
        Tensor of shape [P, N, N, C]
    """
    asserts = _input_tensor_asserts(A, B)
    with tf.control_dependencies(asserts):
        image_shape = _image_shape(A)

    AtA, BtB = self_inner_prod(A, B, filter_shape)
    AtA = tf.transpose(AtA, [1, 0, 2])  # [P, N, C]
    BtB = AtA if AtA is BtB else tf.transpose(BtB, [1, 0, 2])  # [P, N, C]
    ABt = patchwise_conv_inner_prod(A, B, filter_shape, **map_kwargs)
    return -2 * ABt + AtA[:, :, None, :] + BtB[:, None, :, :]


### Inner products


def diag_conv_inner_prod(A: tf.Tensor, filter_shape: FilterShape, **map_kwargs) -> tf.Tensor:
    """
    Inner product of image sub-patches using convolution operation.

    Args:
        A: Input tensor of shape [N, H, W, C].
        filter_shape: Kernel size values [h, w], where h << H and w << W.
        map_kwargs: (optional) `tf.map_fn` arguments.

    Returns:
        Tensor with [N,P,P,C] shape, where P = (H - h + 1) * (W - w + 1)
    """
    N, H, W, C = _image_shape(A)
    Ph, Pw = _grid_patch_shape(H, W, filter_shape)
    P = _grid_patch_size(H, W, filter_shape)

    At = tf.transpose(A, [1, 2, 0, 3])  # [1, H, W, C, N]
    At = tf.reshape(At, [1, H, W, C*N])  # [1, H, W, C*N]

    def fn(idx: tf.Tensor) -> tf.Tensor:
        kernel = _extract_kernel(At, idx, filter_shape, Ph, Pw)  # [1, h, w, C*N]
        kernel = tf.transpose(kernel, [1, 2, 3, 0])  # [h, w, C*N, 1]
        return tf.nn.depthwise_conv2d(At, kernel, strides=[1, 1, 1, 1], padding='VALID')

    result = _map_indices(fn, P, dtype=A.dtype, **map_kwargs)  # [P, 1, Ph, Pw, (C*N)]
    result = tf.reshape(result, [P, P, C, N])
    return tf.transpose(result, [3, 0, 1, 2])  # [N, P, P, C]


def full_conv_inner_prod(A: tf.Tensor,
                         B: tf.Tensor,
                         filter_shape: FilterShape,
                         **map_kwargs) -> tf.Tensor:
    """
    Inner product beetween input images.

    Args:
        A: Input tensor of shape [N, H, W, C].
        B: Input tensor of shape [N, H, W, C].
        filter_shape: Kernel size values [h, w], where h << H and w << W.
        map_kwargs: (optional) `tf.map_fn` arguments.

    Returns:
        Tensor with [N, P, N, P, C] shape, where P = (H - h + 1) * (W - w + 1)
    """
    N, H, W, C = _image_shape(A)
    Ph, Pw = _grid_patch_shape(H, W, filter_shape)
    P = _grid_patch_size(H, W, filter_shape)

    def fn(idx: tf.Tensor) -> tf.Tensor:
        kernel = _extract_kernel(A, idx, filter_shape, Ph, Pw)  # [N, H, W, C]
        kernel = tf.transpose(kernel, [1, 2, 3, 0])  # [h, w, C, N]
        return tf.nn.depthwise_conv2d(B, kernel, strides, padding)

    result = _map_indices(fn, P, dtype=A.dtype, **map_kwargs)  # [P, N, Ph, Pw, (C*N)]
    result = tf.reshape(result, [P, N, P, C, N])
    return tf.transpose(result, [4, 0, 1, 2, 3])  # [N, P, N, P, C]


def patchwise_conv_inner_prod(A: tf.Tensor,
                              B: tf.Tensor,
                              filter_shape: FilterShape,
                              **map_kwargs) -> tf.Tensor:
    """
    Inner product beetween input images.

    Args:
        A: Input tensor of shape [N, H, W, C].
        B: Input tensor of shape [N, H, W, C].
        filter_shape: Kernel size values [h, w], where h << H and w << W.
        map_kwargs: (optional) `tf.map_fn` arguments.

    Returns:
        Tensor with [N,P,P,C] shape, where P = (H - h + 1) * (W - w + 1)
    """
    N, H, W, C = _image_shape(A)
    Ph, Pw = _grid_patch_shape(H, W, filter_shape)
    P = _grid_patch_size(H, W, filter_shape)

    def fn(idx: tf.Tensor) -> tf.Tensor:
        A_patch = _extract_kernel(A, idx, filter_shape, Ph, Pw)  # [N, H, W, C]
        B_patch = _extract_kernel(B, idx, filter_shape, Ph, Pw)  # [N, H, W, C]
        kernel = tf.transpose(B_patch, [1, 2, 3, 0])  # [h, w, C, N]
        return tf.nn.depthwise_conv2d(A_patch, kernel, strides, padding)

    result = _map_indices(fn, P, dtype=A.dtype, **map_kwargs)  # [P, N, 1, 1, (C*N)]
    result = tf.reshape(result, [P, N, C, N])
    return tf.transpose(result, [0, 1, 3, 2])  # [P, N, N, C]


def self_inner_prod(A: tf.Tensor, B: tf.Tensor, filter_shape: ImageShape) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Inner product of input matrices with itself respectively.
    Computes `AᵀA` and `BᵀB` square parts of `|A - B|² = AᵀA - 2*ABᵀ + BᵀB`.
    When second input is None or the same object as the first one, then it avoids double computation.

    Args:
        A: Input tensor of shape [N, H, W, C].
        B: Input tensor of shape [N, H, W, C].
        filter_shape: Kernel size values [h, w], where h << H and w << W.

    Returns:
        Tuple tensors with [[N, P, C], [N, P, C]] shape, where P = (H - h + 1) * (W - w + 1)
    """

    dtype = A.dtype
    h, w = filter_shape
    N, H, W, C = _image_shape(A)
    P = _grid_patch_size(H, W, filter_shape)

    ones_kernel = tf.ones((h, w, C, 1), dtype=dtype)

    AtA = tf.nn.depthwise_conv2d(A ** 2, ones_kernel, strides, padding)
    AtA = tf.reshape(AtA, [N, P, C])
    if B is None or A is B:
        return AtA, AtA

    BtB = tf.nn.depthwise_conv2d(B ** 2, ones_kernel, strides, padding)
    BtB = tf.reshape(BtB, [N, P, C])
    return AtA, BtB


# /////////////////////////////////////////////////////////////////////////////


def _map_indices(fn: Callable, stop: Int, **map_fn_kwargs) -> tf.Tensor:
    assert callable(fn)
    indices = tf.range(stop)
    return tf.map_fn(fn, indices, **map_fn_kwargs)


def _image_shape(A: tf.Tensor) -> Tuple[Int, Int, Int, Int]:
    shape = tf.shape(A)
    return [shape[i] for i in range(4)]


def _input_tensor_asserts(A: tf.Tensor, B: Optional[tf.Tensor] = None) -> List[tf.Tensor]:
    asserts = []
    asserts.append(tf.assert_rank(A, 4, message="Tensor with [N, H, W, C] shape expected"))
    if B is not None:
        A_shape = tf.shape(A)
        B_shape = tf.shape(B)
        asserts.append(tf.assert_rank(B, 4, message="Tensor with [N, H, W, C] shape expected"))
        asserts.append(tf.assert_equal(A_shape, B_shape, message="Input matrices must have same shapes"))
    return asserts


def _extract_kernel(A: tf.Tensor,
                    idx: Int,
                    filter_shape: FilterShape,
                    height: Int,
                    width: Int) -> tf.Tensor:
    h, w = filter_shape
    sh, sw = idx // width, idx % width
    eh, ew = sh + h, sw + w
    return A[:, sh:eh, sw:ew, :]


def _grid_patch_shape(H: Int, W: Int, filter_shape: FilterShape) -> Tuple[Int, Int]:
    h, w = filter_shape
    return H-h+1, W-w+1


def _grid_patch_size(H: Int, W: Int, filter_shape: FilterShape) -> Int:
    ph, pw = _grid_patch_shape(H, W, filter_shape)
    return ph * pw
