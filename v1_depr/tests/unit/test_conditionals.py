# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from collections import namedtuple

import numpy as np
import pytest
import tensorflow as tf

import gpflow
from gpflow import params_as_tensors_for
from gpflux.convolution.conditionals import conditional, sample_conditional, Kuf, Kuu
from gpflux.convolution.convolution_kernel import ConvKernel, WeightedSumConvKernel
from gpflux.convolution.convolution_utils import ImagePatchConfig, ExtractPatchHandler
from gpflux.convolution.inducing_patch import InducingPatch, IndexedInducingPatch

DTYPE = np.float64


def _set_up(kernel_class, _test_config, with_indexing, with_weights):
    gpflow.reset_default_graph_and_session()
    patch_handler_config = ImagePatchConfig((_test_config.image_size, _test_config.image_size),
                                            (_test_config.patch_dim, _test_config.patch_dim),
                                            pooling=_test_config.pooling)
    inducing_patches_data = np.ones(
        (_test_config.num_inducing_patches, _test_config.patch_dim ** 2), dtype=DTYPE)
    kernel = gpflow.kernels.RBF(input_dim=_test_config.patch_dim ** 2)
    patch_handler = ExtractPatchHandler(config=patch_handler_config)

    if not with_indexing:
        inducing_patch = InducingPatch(patches=inducing_patches_data)
    else:
        indices = np.repeat(np.arange(0, _test_config.num_inducing_patches)[..., None], repeats=2,
                            axis=1)
        inducing_patch = IndexedInducingPatch(patches=inducing_patches_data,
                                              indices=indices)

    if with_weights:
        conv_kernel = kernel_class(basekern=kernel,
                                   image_shape=(_test_config.image_size, _test_config.image_size),
                                   patch_shape=(_test_config.patch_dim, _test_config.patch_dim),
                                   patch_handler=patch_handler,
                                   with_indexing=with_indexing,
                                   with_weights=with_weights)
    else:
        conv_kernel = kernel_class(basekern=kernel,
                                   image_shape=(_test_config.image_size, _test_config.image_size),
                                   patch_shape=(_test_config.patch_dim, _test_config.patch_dim),
                                   patch_handler=patch_handler,
                                   with_indexing=with_indexing)

    return inducing_patch, conv_kernel


def _get_random_square_binary_image(size):
    rs = np.random.get_state()
    np.random.seed(0)
    im = (np.random.random((size, size)) < 0.5).astype(DTYPE)
    np.random.set_state(rs)
    return im[None]  # [1, size, size]


def _set_up_with_image(_test_config, kernel_class, with_indexing, with_weights):
    gpflow.reset_default_graph_and_session()
    inducing_patch, conv_kernel = _set_up(kernel_class,
                                          _test_config,
                                          with_indexing=with_indexing,
                                          with_weights=with_weights)

    x_new = _get_random_square_binary_image(_test_config.image_size)
    x_new = np.repeat(x_new, _test_config.num_test_images, axis=0)
    non_zero = (x_new > 0).sum()
    zero = np.prod(x_new.shape) - non_zero
    x_new = tf.convert_to_tensor(x_new)
    return zero, non_zero, x_new, inducing_patch, conv_kernel


_test_config = namedtuple('_test_config',
                          'patch_dim, image_size, num_test_images, num_inducing_patches, pooling')

TEST_CONFIGS = \
    [
        _test_config(*args) for args in [(5, 14, 3, 2, 1),
                                         (3, 15, 3, 2, 1),
                                         (2, 3, 3, 2, 1),
                                         (1, 2, 3, 2, 1)]
    ]

TEST_CONFIGS_SAME_PATCH = \
    [
        _test_config(*args) for args in [(1, 14, 3, 2, 1),
                                         (1, 15, 3, 2, 1),
                                         (1, 3, 3, 2, 1),
                                         (1, 2, 3, 2, 1)]
    ]


@pytest.mark.parametrize('with_indexing',
                         [True, False])
@pytest.mark.parametrize('_test_config',
                         TEST_CONFIGS)
def test_Kuf_ConvKernel_shape(with_indexing, _test_config):
    inducing_patch, conv_kernel = _set_up(ConvKernel, _test_config,
                                          with_indexing=with_indexing,
                                          with_weights=False)
    x_new = tf.ones(
        (_test_config.num_test_images, _test_config.image_size * _test_config.image_size),
        dtype=DTYPE)

    with params_as_tensors_for(inducing_patch):
        k_uf = Kuf(inducing_patch, conv_kernel, x_new)
        assert k_uf.shape.as_list() == [_test_config.num_inducing_patches,
                                        _test_config.num_test_images,
                                        (_test_config.image_size - _test_config.patch_dim + 1) ** 2]


@pytest.mark.parametrize('with_indexing',
                         [True, False])
@pytest.mark.parametrize('_test_config',
                         TEST_CONFIGS)
def test_Kuu_ConvKernel_shape(with_indexing, _test_config):
    inducing_patch, conv_kernel = _set_up(ConvKernel, _test_config,
                                          with_indexing=with_indexing,
                                          with_weights=False)
    with params_as_tensors_for(inducing_patch):
        k_uu = Kuu(inducing_patch, conv_kernel)
        assert k_uu.shape.as_list() == [_test_config.num_inducing_patches,
                                        _test_config.num_inducing_patches]


@pytest.mark.parametrize('with_weights',
                         [True, False])
@pytest.mark.parametrize('with_indexing',
                         [True, False])
@pytest.mark.parametrize('_test_config',
                         TEST_CONFIGS)
def test_Kuf_WeightedSumConvKernel_shape(with_weights,
                                         with_indexing,
                                         _test_config
                                         ):
    inducing_patch, conv_kernel = _set_up(WeightedSumConvKernel,
                                          _test_config,
                                          with_indexing=with_indexing,
                                          with_weights=with_weights)

    x_new = tf.ones(
        (_test_config.num_test_images, _test_config.image_size * _test_config.image_size),
        dtype=DTYPE)
    with params_as_tensors_for(inducing_patch):
        k_uf = Kuf(inducing_patch, conv_kernel, x_new)
        assert k_uf.shape.as_list() == [_test_config.num_inducing_patches,
                                        _test_config.num_test_images]


@pytest.mark.parametrize('with_weights',
                         [True, False])
@pytest.mark.parametrize('with_indexing',
                         [True, False])
@pytest.mark.parametrize('_test_config',
                         TEST_CONFIGS)
def test_Kuu_WeightedSumConvKernel_shape(with_weights,
                                         with_indexing,
                                         _test_config):
    inducing_patch, conv_kernel = _set_up(WeightedSumConvKernel,
                                          _test_config,
                                          with_indexing=with_indexing,
                                          with_weights=with_weights)
    with params_as_tensors_for(inducing_patch):
        k_uu = Kuu(inducing_patch, conv_kernel)
        assert k_uu.shape.as_list() == [_test_config.num_inducing_patches,
                                        _test_config.num_inducing_patches]


@pytest.mark.parametrize('with_indexing',
                         [True, False])
@pytest.mark.parametrize('_test_config', TEST_CONFIGS_SAME_PATCH)
def test_Kuf_ConvKernel_value(with_indexing, _test_config):
    zero, non_zero, x_new, inducing_patch, conv_kernel = \
        _set_up_with_image(_test_config,
                           ConvKernel,
                           with_indexing=False,
                           with_weights=False)

    with params_as_tensors_for(inducing_patch):
        k_uf = Kuf(inducing_patch, conv_kernel, x_new)
        k_uf_value = gpflow.get_default_session().run(k_uf)
        expected = _test_config.num_inducing_patches * (zero * np.exp(-0.5) + non_zero * np.exp(0))
        np.testing.assert_almost_equal(k_uf_value.sum(), expected)


@pytest.mark.parametrize('with_weights',
                         [True, False])
@pytest.mark.parametrize('_test_config',
                         TEST_CONFIGS_SAME_PATCH)
def test_Kuf_WeightedConvKernel_value(with_weights, _test_config):
    zero, non_zero, x_new, inducing_patch, conv_kernel = \
        _set_up_with_image(_test_config,
                           WeightedSumConvKernel,
                           with_indexing=False,
                           with_weights=with_weights)
    with params_as_tensors_for(inducing_patch):
        k_uf = Kuf(inducing_patch, conv_kernel, x_new)
        k_uf_value = gpflow.get_default_session().run(k_uf)
        expected = \
            _test_config.num_inducing_patches * (zero * np.exp(-0.5) + non_zero * np.exp(0)) \
            / (_test_config.image_size - _test_config.patch_dim + 1) ** 2
        np.testing.assert_almost_equal(k_uf_value.sum(), expected)


@pytest.mark.parametrize('_test_config',
                         TEST_CONFIGS_SAME_PATCH)
def test_Kuu_ConvKernel_value(_test_config):
    *_, inducing_patch, conv_kernel = _set_up_with_image(_test_config,
                                                         ConvKernel,
                                                         with_indexing=False,
                                                         with_weights=False)

    with params_as_tensors_for(inducing_patch):
        k_uu = Kuu(inducing_patch, conv_kernel)
        k_uu_value = gpflow.get_default_session().run(k_uu)
        expected = np.prod(k_uu_value.shape)
        np.testing.assert_almost_equal(k_uu_value.sum(), expected)


@pytest.mark.parametrize('with_weights',
                         [True, False])
@pytest.mark.parametrize('_test_config',
                         TEST_CONFIGS_SAME_PATCH)
def test_Kuu_WeightedConvKernel_value(with_weights, _test_config):
    *_, inducing_patch, conv_kernel = _set_up_with_image(_test_config,
                                                         WeightedSumConvKernel,
                                                         with_indexing=False,
                                                         with_weights=with_weights)

    with params_as_tensors_for(inducing_patch):
        k_uu = Kuu(inducing_patch, conv_kernel)
        k_uu_value = gpflow.get_default_session().run(k_uu)
        expected = np.prod(k_uu_value.shape)
        np.testing.assert_almost_equal(k_uu_value.sum(), expected)


@pytest.mark.skip('unskip once a PR with fix is merged into GPflow')
@pytest.mark.parametrize('with_indexing',
                         [True, False])
@pytest.mark.parametrize('_test_config',
                         TEST_CONFIGS)
def test_conditional(with_indexing, _test_config):
    inducing_patch, conv_kernel = _set_up(ConvKernel,
                                          _test_config,
                                          with_indexing=with_indexing,
                                          with_weights=False)
    x_new = tf.ones(
        (_test_config.num_test_images, _test_config.image_size * _test_config.image_size),
        dtype=DTYPE)
    f = tf.ones((len(inducing_patch), 1), dtype=x_new.dtype)
    with params_as_tensors_for(inducing_patch):
        fmean, fvar = conditional(x_new, inducing_patch, conv_kernel, f, white=True)
        assert fmean.shape == fvar.shape


@pytest.mark.skip('unskip once a PR with fix is merged into GPflow')
@pytest.mark.parametrize('with_indexing',
                         [True, False])
@pytest.mark.parametrize('_test_config',
                         TEST_CONFIGS)
def test_sample_conditional(with_indexing,
                            _test_config):
    num_samples = 5
    inducing_patch, conv_kernel = _set_up(ConvKernel,
                                          _test_config,
                                          with_indexing=with_indexing,
                                          with_weights=False)

    x_new = tf.ones(
        (_test_config.num_test_images, _test_config.image_size * _test_config.image_size),
        dtype=DTYPE)
    f = tf.ones((len(inducing_patch), 1), dtype=x_new.dtype)

    with params_as_tensors_for(inducing_patch):
        s, fmean, fvar = sample_conditional(x_new, inducing_patch, conv_kernel, f, white=True,
                                            num_samples=num_samples)
        assert fmean.shape == fvar.shape
