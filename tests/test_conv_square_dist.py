import gpflow
import numpy as np
import pytest
import tensorflow as tf
from gpflow.kernels import RBF
from gpflow.likelihoods import Gaussian
from gpflow.models import SVGP
from gpflow.test_util import session_tf
from numpy.testing import assert_allclose

import gpflux
from gpflux.conv_square_dists import (diag_conv_square_dist,
                                      full_conv_square_dist,
                                      patchwise_conv_square_dist)
from gpflux.convolution.convolution_kernel import ConvKernel
from gpflux.utils import get_image_patches


class DT:
    N, H, W, C = image_shape = 2, 3, 3, 1
    h, w = filter_shape = 2, 2
    filter_size = h * w
    Ph, Pw = H - h + 1, W - w + 1
    P = Ph * Pw
    rng = np.random.RandomState(911911)
    img1 = rng.randn(*image_shape)
    img2 = rng.randn(*image_shape)


def create_rbf(filter_size=None):
    return RBF(filter_size or DT.filter_size)


def create_conv_kernel(image_size=None, filter_size=None, colour_channels=1):
    rbf = create_rbf(filter_size)
    return ConvKernel(rbf, image_size, filter_size, colour_channels=colour_channels)


def test_diag_conv_square_dist(session_tf):
    img = tf.convert_to_tensor(DT.img1)
    image_size = DT.H * DT.W
    patch_size = np.prod(DT.filter_size)

    dtype = img.dtype
    rbf = create_rbf()
    X = get_image_patches(img, DT.image_shape, DT.filter_shape)

    dist = diag_conv_square_dist(img, DT.filter_shape)
    dist = tf.squeeze(dist)

    gotten = rbf.K_r2(dist)
    expect = tf.map_fn(lambda x: rbf.K(x), X, dtype=dtype)
    gotten_np, expect_np = session_tf.run([gotten, expect])
    assert_allclose(expect_np, gotten_np)

    gotten_diag = tf.matrix_diag_part(rbf.K_r2(dist))
    expect_diag = tf.map_fn(lambda x: rbf.Kdiag(x), X, dtype=dtype)
    gotten_diag_np, expect_diag_np = session_tf.run([gotten_diag, expect_diag])
    assert_allclose(expect_diag_np, gotten_diag_np)


def test_full_conv_square_dist(session_tf):
    rbf = create_rbf()
    N, H, W, C = DT.image_shape
    P = DT.P
    img1 = tf.convert_to_tensor(DT.img1)
    img2 = tf.convert_to_tensor(DT.img2)

    X1 = get_image_patches(img1, DT.image_shape, DT.filter_shape)
    X2 = get_image_patches(img2, DT.image_shape, DT.filter_shape)
    X1 = tf.reshape(X1, (-1, DT.filter_size))
    X2 = tf.reshape(X2, (-1, DT.filter_size))

    expect = tf.reshape(rbf.K(X1, X2), (N, P, N, P))
    expect = tf.squeeze(expect)

    dist = full_conv_square_dist(img1, img2, DT.filter_shape)
    dist = tf.squeeze(dist)
    gotten = rbf.K_r2(dist)

    gotten_np, expect_np = session_tf.run([gotten, expect])
    assert_allclose(expect_np, gotten_np)


def test_pairwise_conv_square_dist(session_tf):
    rbf = create_rbf()
    N, H, W, C = DT.image_shape
    P = DT.P
    img1 = tf.convert_to_tensor(DT.img1)
    img2 = tf.convert_to_tensor(DT.img2)
    dtype = img1.dtype

    X1 = get_image_patches(img1, DT.image_shape, DT.filter_shape)
    X2 = get_image_patches(img2, DT.image_shape, DT.filter_shape)
    X1t = tf.transpose(X1, [1, 0, 2])
    X2t = tf.transpose(X2, [1, 0, 2])

    expect = tf.map_fn(lambda Xs: rbf.K(*Xs), (X1t, X2t), dtype=dtype)
    expect = tf.squeeze(expect)
    dist = patchwise_conv_square_dist(img1, img2, DT.filter_shape)
    dist = tf.squeeze(dist)
    gotten = rbf.K_r2(dist)

    gotten_np, expect_np = session_tf.run([gotten, expect])
    assert_allclose(expect_np, gotten_np)

    expect = tf.map_fn(lambda x: rbf.K(x), X1t, dtype=dtype)
    dist = patchwise_conv_square_dist(img1, img1, DT.filter_shape)
    dist = tf.squeeze(dist)
    gotten = rbf.K_r2(dist)

    gotten_np, expect_np = session_tf.run([gotten, expect])
    assert_allclose(expect_np, gotten_np)

