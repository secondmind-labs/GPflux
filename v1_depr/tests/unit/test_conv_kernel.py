# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import numpy as np
import pytest

import gpflow
from gpflux.convolution.convolution_kernel import ConvKernel, K_image_symm


class Data:
    N = 10
    w, h = 3, 3
    W, H = 28, 28


def input_dim():
    return Data.w * Data.h


@pytest.fixture
def conv_kernel(session_tf):
    base_kern = gpflow.kernels.RBF(input_dim(), variance=1.2, lengthscales=2.1)
    return ConvKernel(base_kern, image_shape=[Data.W, Data.H], patch_shape=[Data.w, Data.h])


def test_num_outputs(session_tf, conv_kernel):
    desired = 1 * (Data.W - Data.w + 1) * (Data.H - Data.h + 1)
    config = conv_kernel.patch_handler.config
    actual_1 = config.num_patches
    actual_2 = config.Hout * config.Wout
    np.testing.assert_equal(desired, actual_1)
    np.testing.assert_equal(desired, actual_2)


def test_patch_len(conv_kernel):
    np.testing.assert_equal(conv_kernel.patch_len, Data.w * Data.h)


@pytest.mark.skip(reason="Convolutional K_symm is not implemented")
def test_K(session_tf, conv_kernel):
    ones_patch = np.ones((1, Data.h * Data.w))
    desired_value = conv_kernel.basekern.compute_K_symm(ones_patch)
    desired_value = desired_value.flatten()[0]
    images = np.ones((Data.N, Data.H * Data.W))
    output = conv_kernel.compute_K_symm(images)
    np.testing.assert_equal(output[0, 0, 0], desired_value)
    np.testing.assert_array_equal(output.shape,
                                  [conv_kernel.patch_handler.config.num_patches, Data.N, Data.N])


def test_K_diag(session_tf, conv_kernel):
    ones_patch = np.ones((1, Data.h * Data.w))
    desired_value = session_tf.run(
        K_image_symm(conv_kernel.basekern, conv_kernel.patch_handler, ones_patch))
    desired_value = desired_value.flatten()[0]

    images = np.ones((Data.N, Data.H * Data.W))
    output = conv_kernel.compute_Kdiag(images)

    np.testing.assert_equal(output[0, 0], desired_value)
    np.testing.assert_array_equal(output.shape,
                                  [Data.N, conv_kernel.patch_handler.config.num_patches])
