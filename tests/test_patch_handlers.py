# Copyright (C) PROWLER.io 2018, 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import gpflow
import numpy as np
import pytest
import tensorflow as tf
from gpflow.test_util import session_tf
from numpy.testing import assert_allclose

import gpflux
from gpflux.convolution import ExtractPatchHandler, ConvPatchHandler
from gpflux.convolution import ImagePatchConfig


class DT:
    N, H, W, C = 64, 28, 28, 1
    image_shape = [H, W, C]
    M = 33
    h, w = filter_shape = 5, 5
    feat_size = M * h * w
    filter_size = h * w
    Ph, Pw = H - h + 1, W - w + 1
    P = Ph * Pw
    rng = np.random.RandomState(1010101)
    img1 = rng.randn(N, *image_shape)
    img2 = rng.randn(N, *image_shape)
    feat = rng.randn(M, h * w)


def config():
    return ImagePatchConfig(DT.image_shape, DT.filter_shape)


@pytest.fixture
def extract_handler():
    return ExtractPatchHandler(config())


@pytest.fixture
def conv_handler():
    return ConvPatchHandler(config())


def test_image_patches_inner_product(session_tf, extract_handler, conv_handler):
    x = tf.convert_to_tensor(DT.img1)
    extract, conv = session_tf.run([
        extract_handler.image_patches_inner_product(x),
        conv_handler.image_patches_inner_product(x),
    ])
    assert_allclose(extract, conv)


def test_image_patches_squared_norm(session_tf, extract_handler, conv_handler):
    x = tf.convert_to_tensor(DT.img1)
    extract, conv = session_tf.run([
        extract_handler.image_patches_squared_norm(x),
        conv_handler.image_patches_squared_norm(x),
    ])
    assert_allclose(extract, conv)


def test_image_patches_inducing_patches_inner_product(session_tf, extract_handler, conv_handler):
    x = tf.convert_to_tensor(DT.img1)
    z = tf.convert_to_tensor(DT.feat)
    extract, conv = session_tf.run([
        extract_handler.image_patches_inducing_patches_inner_product(x, z),
        conv_handler.image_patches_inducing_patches_inner_product(x, z),
    ])
    assert_allclose(extract, conv)
