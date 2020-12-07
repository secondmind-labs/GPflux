# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


import numpy as np
import pytest
import tensorflow as tf

from gpflux.convolution.convolution_utils import (
    ConvPatchHandler,
    ExtractPatchHandler,
    ImagePatchConfig,
)


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


handlers = [ExtractPatchHandler, ConvPatchHandler]


def config():
    return ImagePatchConfig(DT.image_shape, DT.filter_shape)


def run_benchmark(bench, session, op):
    def run_op():
        session.run(op)

    bench.pedantic(run_op, warmup_rounds=2, rounds=100, iterations=100)


@pytest.mark.parametrize('handler', handlers)
def test_image_patches_inner_product(session_tf, benchmark, handler):
    h = handler(config())
    x = tf.convert_to_tensor(DT.img1)
    op = h.image_patches_inner_product(x)
    run_benchmark(benchmark, session_tf, op)


@pytest.mark.parametrize('handler', handlers)
def test_image_patches_squared_norm(session_tf, benchmark, handler):
    h = handler(config())
    x = tf.convert_to_tensor(DT.img1)
    op = h.image_patches_squared_norm(x)
    run_benchmark(benchmark, session_tf, op)


@pytest.mark.parametrize('handler', handlers)
def test_image_patches_inducing_patches_inner_product(session_tf, benchmark, handler):
    h = handler(config())
    x = tf.convert_to_tensor(DT.img1)
    z = tf.convert_to_tensor(DT.feat)
    op = h.image_patches_inducing_patches_inner_product(x, z)
    run_benchmark(benchmark, session_tf, op)
