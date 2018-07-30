# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidentialimport numpy as np

import pytest
import numpy as np

from gpflow.kernels import RBF
from gpflux.convolution import ConvKernel


class Data:
    N = 10
    w, h = 3, 3
    W, H = 28, 28



@pytest.fixture
def conv_kernel():
    base_kern = RBF(Data.w * Data.h, variance=1.2, lengthscales=2.1)
    return ConvKernel(base_kern, [Data.W, Data.H], [Data.w, Data.h])


def test_num_outputs(conv_kernel):
    desired = 1 * (Data.W - Data.w + 1) * (Data.H - Data.h + 1)
    actual_1 = conv_kernel.num_patches
    actual_2 = conv_kernel.Hout * conv_kernel.Wout
    assert desired == actual_1
    assert desired == actual_2


def test_patch_les(conv_kernel):
    assert conv_kernel.patch_len == Data.w * Data.h


def test_patch_extraction(conv_kernel):
    desired = np.zeros((Data.N, conv_kernel.num_patches, conv_kernel.patch_len))
    images = np.zeros((Data.N, Data.H, Data.W))
    patches = conv_kernel.compute_patches(images)
    np.testing.assert_array_equal(patches, desired)


def test_K(conv_kernel):
    ones_patch = np.ones((1, Data.h * Data.w))
    desired_value = conv_kernel.basekern.compute_K_symm(ones_patch)
    desired_value = desired_value.flatten()[0]

    images = np.ones((Data.N, Data.H * Data.W))
    output = conv_kernel.compute_K_symm(images)

    assert output[0, 0, 0] == desired_value
    assert output.shape == (conv_kernel.num_patches, Data.N, Data.N)


def test_K_diag(conv_kernel):
    ones_patch = np.ones((1, Data.h * Data.w))
    desired_value = conv_kernel.basekern.compute_K_symm(ones_patch)
    desired_value = desired_value.flatten()[0]

    images = np.ones((Data.N, Data.H * Data.W))
    output = conv_kernel.compute_Kdiag(images)

    assert output[0, 0] == desired_value
    assert output.shape == (Data.N, conv_kernel.num_patches)
