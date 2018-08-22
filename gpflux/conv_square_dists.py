from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf


__all__ = [
    'diag_conv_dist_squared',
    'full_conv_dist_squared',
    'pathwise_conv_dist_squared',
]

Int = Union[tf.Tensor, int]
FilterShape = Tuple[Int, Int]
ImageShape = Tuple[Int, Int, Int, Int]


padding = 'VALID'
strides = (1, 1, 1, 1)


def diag_conv_dist_squared(A: tf.Tensor, filter_shape: FilterShape,
                           **map_kwargs) -> tf.Tensor:
    asserts = _input_tensor_asserts(A)
    with tf.control_dependencies(asserts):
        image_shape = _image_shape(A)

    AtA, _ = self_inner_prod(A, None, image_shape, filter_shape)
    AAt = diag_conv_inner_prod(A, filter_shape, **map_kwargs)
    return -2 * AAt + AtA[:, :, None, :] + AtA[:, None, :, :]


def full_conv_dist_squared(A: tf.Tensor, B: tf.Tensor, filter_shape: FilterShape,
                           **map_kwargs) -> tf.Tensor:
    """
    Args:
        A: Tensor of shape NxHxWxC
        B: Tensor of shape NxHxWxC
        filter_shape: Two element tuple with height and width sizes of filter respectively.
    Returns:
        Tensor of shape NxPxNxPxC.
    """
    asserts = _input_tensor_asserts(A, B)
    with tf.control_dependencies(asserts):
        image_shape = _image_shape(A)

    AtA, BtB = self_inner_prod(A, B, image_shape, filter_shape)
    ABt = full_conv_inner_prod(A, B, filter_shape, **map_kwargs)
    return -2 * ABt + AtA[None, :, None, :, :] + BtB[:, None, :, None, :]


def patchwise_conv_dist_squared(A: tf.Tensor, B: tf.Tensor, filter_shape: FilterShape,
                                **map_kwargs) -> tf.Tensor:
    """
    Args:
        A: Tensor of shape NxHxWxC
        B: Tensor of shape NxHxWxC
        filter_shape: Two element tuple with height and width sizes of filter respectively
    Returns:
        Tensor of shape PxNxNxC
    """
    asserts = _input_tensor_asserts(A, B)
    with tf.control_dependencies(asserts):
        image_shape = _image_shape(A)

    AtA, BtB = self_inner_prod(A, B, image_shape, filter_shape)
    ABt = patchwise_conv_inner_prod(A, B, filter_shape, **map_kwargs)
    return -2 * ABt + AtA[:, None, :, :] + BtB[:, :, None, :]


### Inner products


def diag_conv_inner_prod(A: tf.Tensor, filter_shape: FilterShape,
                         image_shape: Optional[ImageShape] = None,
                         back_prop: int = True,
                         parallel: int = 10,
                         swap_memory: bool = False) -> tf.Tensor:
    """
    Inner product of image sub-patches using convolution operation.

    Args:
        A: Input tensor of shape NxHxWxC.
        filter_shape: Kernel size values [h, w], where h << H and w << W.
        image_shape: (optional) Input image shape.
        parallel_iterations: (optional) The number of iterations allowed to run in parallel.
        back_prop: (optional) True enables support for back propagation.
        swap_memory: (optional) True enables GPU-CPU memory swapping.

    Returns:
        Tensor with NxPxPxC shape, where P = (H - h + 1) * (W - w + 1)
    """
    N, H, W, C = image_shape or _image_shape(A)
    Ph, Pw = _grid_patch_shape(H, W, filter_shape)
    P = _grid_patch_size(H, W, filter_shape)

    At = tf.transpose(A, [1, 2, 0, 3])  # 1xHxWxCxN
    At = tf.reshape(At, [1, H, W, C*N])  # 1xHxWx(C*N)

    def fn(idx: tf.Tensor) -> tf.Tensor:
        kernel = _extract_kernel(At, idx, filter_shape, Ph, Pw)  # 1xhxhx(C*N)
        kernel = tf.transpose(kernel, [1, 2, 3, 0])  # hxwx(C*N)x1
        return tf.nn.depthwise_conv2d(At, kernel, strides=[1, 1, 1, 1], padding='VALID')

    result = _map_indices(fn, P,
                          swap_memory=swap_memory,
                          parallel_iterations=parallel,
                          back_prop=back_prop, dtype=A.dtype)  # PxNxPhxPwx(C*N)

    result = tf.reshape(result, [P, P, C, N])
    return tf.transpose(result, [3, 0, 1, 2])  # NxPxPxC


def full_conv_inner_prod(A: tf.Tensor,
                         B: tf.Tensor,
                         filter_shape: FilterShape,
                         image_shape: Optional[ImageShape] = None,
                         back_prop: int = True,
                         parallel: int = 10,
                         swap_memory: bool = False) -> tf.Tensor:
    """
    Inner product of image sub-patches using convolution operation.

    Args:
        A: Input tensor of shape NxHxWxC.
        B: Input tensor of shape NxHxWxC.
        filter_shape: Kernel size values [h, w], where h << H and w << W.
        image_shape: (optional) Input image shape.
        parallel_iterations: (optional) The number of iterations allowed to run in parallel.
        back_prop: (optional) True enables support for back propagation.
        swap_memory: (optional) True enables GPU-CPU memory swapping.

    Returns:
        Tensor with NxPxPxC shape, where P = (H - h + 1) * (W - w + 1)
    """
    N, H, W, C = image_shape or _image_shape(A)
    Ph, Pw = _grid_patch_shape(H, W, filter_shape)
    P = _grid_patch_size(H, W, filter_shape)

    def fn(idx: tf.Tensor) -> tf.Tensor:
        kernel = _extract_kernel(B, idx, filter_shape, Ph, Pw)  # NxhxwxC
        kernel = tf.transpose(kernel, [1, 2, 3, 0])  # hxwxCxN
        return tf.nn.depthwise_conv2d(A, kernel, strides, padding)

    result = _map_indices(fn, P,
                          swap_memory=swap_memory,
                          parallel_iterations=parallel,
                          back_prop=back_prop, dtype=A.dtype)  # PxNxPhxPwx(C*N)

    result = tf.reshape(result, [P, N, P, C, N])
    return tf.transpose(result, [0, 1, 2, 4, 3])  # PxNxPxNxC


def patchwise_conv_inner_prod(A: tf.Tensor,
                              B: tf.Tensor,
                              filter_shape: FilterShape,
                              image_shape: Optional[ImageShape] = None,
                              back_prop: int = True,
                              parallel: int = 10,
                              swap_memory: bool = False) -> tf.Tensor:
    """
    Inner product of image sub-patches using convolution operation.

    Args:
        A: Input tensor of shape NxHxWxC.
        B: Input tensor of shape NxHxWxC.
        filter_shape: Kernel size values [h, w], where h << H and w << W.
        image_shape: (optional) Input image shape.
        parallel_iterations: (optional) The number of iterations allowed to run in parallel.
        back_prop: (optional) True enables support for back propagation.
        swap_memory: (optional) True enables GPU-CPU memory swapping.

    Returns:
        Tensor with NxPxPxC shape, where P = (H - h + 1) * (W - w + 1)
    """
    N, H, W, C = image_shape or _image_shape(A)
    Ph, Pw = _grid_patch_shape(H, W, filter_shape)
    P = _grid_patch_size(H, W, filter_shape)

    def fn(idx: tf.Tensor) -> tf.Tensor:
        A_patch = _extract_kernel(A, idx, filter_shape, Ph, Pw)  # NxhxwxC
        B_patch = _extract_kernel(B, idx, filter_shape, Ph, Pw)  # NxhxwxC
        kernel = tf.transpose(B_patch, [1, 2, 3, 0])  # hxwxCxN
        return tf.nn.depthwise_conv2d(A_patch, kernel, strides, padding)

    result = _map_indices(fn, P,
                          swap_memory=swap_memory,
                          parallel_iterations=parallel,
                          back_prop=back_prop, dtype=A.dtype)  # PxNx1x1x(C*N)

    result = tf.reshape(result, [P, N, C, N])
    return tf.transpose(result, [0, 1, 3, 2])  # PxNxNxC


def self_inner_prod(A: tf.Tensor,
                    B: tf.Tensor,
                    image_shape: FilterShape,
                    filter_shape: ImageShape) -> Tuple[tf.Tensor, tf.Tensor]:
    dtype = A.dtype
    h, w = filter_shape
    N, H, W, C = image_shape or _image_shape(A)
    P = _grid_patch_size(H, W, filter_shape)

    ones_kernel = tf.ones((h, w, C, 1), dtype=dtype)

    AtA = tf.nn.depthwise_conv2d(A ** 2, ones_kernel, strides, padding)
    AtA = tf.reshape(AtA, [N, P, C])  # NxPxC
    if B is None or A is B:
        return AtA, AtA

    BtB = tf.nn.depthwise_conv2d(B ** 2, ones_kernel, strides, padding)
    BtB = tf.reshape(BtB, [N, P, C])  # NxPxC
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
    asserts.append(tf.assert_rank(A, 4, message="Tensor with NxHxWxC shape expected"))
    if B is not None:
        A_shape = tf.shape(A)
        B_shape = tf.shape(B)
        asserts.append(tf.assert_rank(B, 4, message="Tensor with NxHxWxC shape expected"))
        asserts.append(tf.assert_equal(A_shape, B_shape, message="Input matrices must have same shapes"))
    return asserts


def _extract_kernel(A: tf.Tensor,
                    idx: Int,
                    filter_shape: FilterShape,
                    height: Int,
                    width: Int) -> tf.Tensor:
    sh, sw = idx // width, idx % width
    h, w = filter_shape
    return A[:, sh:(sh + h), sw:(sw + w), :]


def _grid_patch_shape(H: Int, W: Int, filter_shape: FilterShape) -> Tuple[Int, Int]:
    h, w = filter_shape
    return H-h+1, W-w+1


def _grid_patch_size(H: Int, W: Int, filter_shape: FilterShape) -> Int:
    ph, pw = _grid_patch_shape(H, W, filter_shape)
    return ph * pw
